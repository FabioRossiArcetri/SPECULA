"""
Diagnostics for Conv2dNetTrainer.

They are meant to tell apart the possible bottlenecks that stop the CNN from
reaching a lower reconstruction error:

- architecture/capacity (does a plain linear map from the same inputs do as
  well or better? do the predictions span as many independent directions as
  the labels need?)
- optimization (gradient clipping, generalization to each new batch)
- loop transients (error vs. frames since the loop came back to full gain)
- information content of the input (error vs. the residual in modes the
  network doesn't predict, i.e. aliasing and pyramid nonlinearity; shrinkage
  of the predictions towards the mean, typical of noise-limited modes)
- a moving target (drift of the normalization statistics)

Everything here only observes: nothing feeds back into training.
"""

import json

import numpy as np
import torch

from specula.lib.ridge_regression import ridge_select_and_fit


# A ridge solve costs min(samples, features)^3 -- ridge_fit() uses the
# primal or the dual form, whichever is smaller -- so this bounds that,
# not the number of input pixels.
RIDGE_MAX_SOLVE_DIM = 20000
MIN_RIDGE_SAMPLES = 100
# Bins of frames since the loop came back to full gain ((lo, hi), inclusive;
# hi None = no upper bound). 0 = loop not at full gain.
TRANSIENT_BINS = ((0, 0), (1, 10), (11, 50), (51, 200), (201, 1000), (1001, None))


def _ridge_predictor(X, Y):
    W, xm, ym, s, _ = ridge_select_and_fit(X, Y)
    return (lambda Z: (Z - xm) @ W + ym), s


def mode_groups(nmodes):
    edges = [e for e in (0, 5, 20, 50, 100, 200, 500) if e < nmodes] + [nmodes]
    return list(zip(edges[:-1], edges[1:]))


def effective_dims(x, frac=0.99):
    """Number of principal components carrying `frac` of the variance of x (N, K)."""
    xc = x - x.mean(dim=0, keepdim=True)
    energy = torch.linalg.svdvals(xc) ** 2
    total = energy.sum()
    if total <= 0:
        return 0
    cum = torch.cumsum(energy, 0) / total
    return int((cum < frac).sum().item()) + 1


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = torch.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float('nan')


class TrainingDiagnostics:

    def __init__(self, name, nmodes, max_val, ridge_samples, jsonl_filename,
                 clip_value, device, settled_after=200):
        self.name = name
        # Frames since the loop came back to full gain after which the loop
        # is considered settled (for the settled-only ridge fit).
        self.settled_after = settled_after
        self.res_fsr = None
        self.nmodes = nmodes
        self.max_val = max_val
        self.ridge_samples = ridge_samples
        self.jsonl_filename = jsonl_filename
        self.clip_value = clip_value
        self.device = device

        self.valid = None
        self.res_x = None
        self.res_y = None
        self.val_x = None
        self.val_oob = None
        self.last_n = None
        self.last_fresh = None
        self.last_run_step = None
        self._printed_note = False
        self._reset_window()

    def _reset_window(self):
        self.win_fresh = []
        self.win_train = []
        self.win_grad = []
        self.win_std_drift = []
        self.win_mean_drift = []
        self.win_inband_rms = []
        self.win_oob_rms = []
        self.win_fwd_fvu = []
        # CNN predictions on each new batch before training on it, with each
        # sample's frames since the loop came back to full gain.
        self.win_fresh_P = []
        self.win_fresh_T = []
        self.win_fresh_fsr = []

    def _print(self, msg):
        print(f"[{self.name}][diag] {msg}", flush=True)

    # ------------------------------------------------------------------
    #   Per-trigger recording
    # ------------------------------------------------------------------

    def add_batch(self, raw_train_x, train_y, raw_val_x, val_oob_rms,
                  inband_rms, oob_rms, train_fsr=None):
        """All arguments are CPU torch tensors. raw_*_x are the network inputs
        *before* normalization, one row per sample. The val rows must be
        appended in the same order (and trimmed to the same max_val) as the
        trainer's own validation set, so that they stay row-aligned."""
        tx = raw_train_x.flatten(start_dim=1)
        vx = raw_val_x.flatten(start_dim=1)
        if self.valid is None:
            # Pixels outside the pupil are always exactly zero: drop them,
            # both to save memory and to keep the ridge problem small.
            self.valid = torch.cat([tx, vx]).abs().amax(dim=0) > 0
        tx = tx[:, self.valid]
        vx = vx[:, self.valid]

        def _fifo(buf, new, cap):
            return new[-cap:].clone() if buf is None else torch.cat([buf, new])[-cap:]

        self.res_x = _fifo(self.res_x, tx, self.ridge_samples)
        self.res_y = _fifo(self.res_y, train_y, self.ridge_samples)
        if train_fsr is None:
            train_fsr = torch.full((tx.shape[0],), -1, dtype=torch.int64)
        self.res_fsr = _fifo(self.res_fsr, train_fsr.to(torch.int64), self.ridge_samples)
        self.last_n = min(tx.shape[0], self.res_x.shape[0])
        self.last_fresh = None
        self.val_x = _fifo(self.val_x, vx, self.max_val)
        self.val_oob = _fifo(self.val_oob, val_oob_rms, self.max_val)

        self.win_inband_rms.append(float(torch.sqrt((inband_rms ** 2).mean())))
        if oob_rms.numel() > 0:
            self.win_oob_rms.append(float(torch.sqrt((oob_rms ** 2).mean())))

    def record_normalization_drift(self, old_mean, old_std, new_mean, new_std):
        old_mean = np.asarray(old_mean, dtype=np.float64)
        old_std = np.asarray(old_std, dtype=np.float64)
        new_mean = np.asarray(new_mean, dtype=np.float64)
        new_std = np.asarray(new_std, dtype=np.float64)
        self.win_std_drift.append(float(np.max(np.abs(new_std - old_std) / old_std)))
        self.win_mean_drift.append(float(np.max(np.abs(new_mean - old_mean) / old_std)))

    def record_fresh_predictions(self, preds, targets, fsr=None):
        """Physical-unit CNN predictions on the newest batch's training
        samples, made *before* training on them (CPU tensors). Same rows, in
        the same order, as the last batch passed to add_batch(). fsr: each
        sample's frames since the loop came back to full gain, if known."""
        P = preds.to(torch.float64)
        T = targets.to(torch.float64)
        self.last_fresh = (P, T)
        if fsr is not None:
            self.win_fresh_P.append(P)
            self.win_fresh_T.append(T)
            self.win_fresh_fsr.append(fsr.to(torch.int64))
        var = T.var(0, unbiased=False).sum()
        if var > 0:
            self.win_fwd_fvu.append(float(((P - T) ** 2).mean(0).sum() / var))

    def record_step(self, fresh_loss, train_loss, grad_norms):
        if fresh_loss is not None:
            self.win_fresh.append(fresh_loss)
        self.win_train.append(train_loss)
        self.win_grad.extend(grad_norms)

    # ------------------------------------------------------------------
    #   Periodic analysis
    # ------------------------------------------------------------------

    def _ridge_unavailable(self, n_needed=MIN_RIDGE_SAMPLES):
        if self.res_x is None or self.res_x.shape[0] < n_needed:
            return 'too few training samples yet'
        n, d = self.res_x.shape
        if min(n, d) > RIDGE_MAX_SOLVE_DIM:
            return f'{n} samples x {d} features, too large for a direct solve'
        return None

    def _ridge_val(self, val_targets):
        """Linear ridge fitted on the recent training samples, evaluated on
        the same validation set as the CNN. Returns (per-mode MSE, note)."""
        why = self._ridge_unavailable()
        if why:
            return None, why
        if self.val_x.shape[0] != val_targets.shape[0]:
            return None, 'validation set not row-aligned (skipped)'
        dev = self.device
        predict, s = _ridge_predictor(self.res_x.to(dev, torch.float64),
                                      self.res_y.to(dev, torch.float64))
        T = val_targets.to(dev, torch.float64)
        mse = ((predict(self.val_x.to(dev, torch.float64)) - T) ** 2).mean(0)
        n, d = self.res_x.shape
        return mse.cpu(), f'fit on {n} samples x {d} features, lambda scale {s:g}'

    def _ridge_forward(self):
        """Linear ridge fitted on every reservoir sample *before* the newest
        batch, tested on the newest batch: an honest forward-in-time test,
        free of the near-duplicate neighbouring frames that a random split
        within a batch leaves between training and validation samples.
        Returns (per-mode MSE, test labels) or (None, None)."""
        n_last = self.last_n
        if n_last is None or self._ridge_unavailable(n_last + MIN_RIDGE_SAMPLES):
            return None, None
        dev = self.device
        X = self.res_x.to(dev, torch.float64)
        Y = self.res_y.to(dev, torch.float64)
        predict, _ = _ridge_predictor(X[:-n_last], Y[:-n_last])
        mse = ((predict(X[-n_last:]) - Y[-n_last:]) ** 2).mean(0)
        return mse.cpu(), self.res_y[-n_last:].to(torch.float64)

    def _ridge_forward_settled(self):
        """Like _ridge_forward(), restricted to settled frames (more than
        settled_after frames since the loop came back to full gain), both in
        the fit and in the test. Returns (per-mode MSE, test labels) or
        (None, None)."""
        n_last = self.last_n
        if n_last is None or self.res_fsr is None or self._ridge_unavailable():
            return None, None
        settled = self.res_fsr > self.settled_after
        fit, test = settled[:-n_last], settled[-n_last:]
        if int(fit.sum()) < MIN_RIDGE_SAMPLES or int(test.sum()) < 20:
            return None, None
        dev = self.device
        X = self.res_x.to(dev, torch.float64)
        Y = self.res_y.to(dev, torch.float64)
        fit_d, test_d = fit.to(dev), test.to(dev)
        predict, _ = _ridge_predictor(X[:-n_last][fit_d], Y[:-n_last][fit_d])
        T = Y[-n_last:][test_d]
        mse = ((predict(X[-n_last:][test_d]) - T) ** 2).mean(0)
        return mse.cpu(), T.cpu()

    def _transient_report(self, K, fwd, rec):
        """Residual and CNN error (on each new batch, before training on it)
        by frames since the loop came back to full gain, plus the linear map
        fitted and tested on settled frames only."""
        if not self.win_fresh_fsr:
            return
        P = torch.cat(self.win_fresh_P)
        T = torch.cat(self.win_fresh_T)
        fsr = torch.cat(self.win_fresh_fsr)
        if int((fsr >= 0).sum()) == 0:
            return
        groups = mode_groups(K)
        names = "/".join(f"{a}-{b - 1}" for a, b in groups)

        def group_rms(x, rows):
            return [float(torch.sqrt((x[rows][:, a:b] ** 2).sum(1).mean())) for a, b in groups]

        self._print(f"loop transient -- by frames since the loop came back to full gain; CNN on each "
                    f"new batch before training on it; RMS per mode group {names}:")
        bins = []
        for lo, hi in TRANSIENT_BINS:
            rows = (fsr >= lo) & (fsr <= hi) if hi is not None else (fsr >= lo)
            n = int(rows.sum())
            if n == 0:
                continue
            label = f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi is not None else f">{lo - 1}")
            resid = group_rms(T, rows)
            err = group_rms(P - T, rows)
            bins.append({'frames': [lo, hi], 'n': n, 'residual_rms': resid, 'cnn_err_rms': err})
            self._print(f"  frames {label:>9} n={n:>6} | residual " + "/".join(f"{v:.0f}" for v in resid)
                        + " | CNN error " + "/".join(f"{v:.0f}" for v in err))
        rec['transient_bins'] = bins

        mse, T_settled = self._ridge_forward_settled()
        if mse is not None:
            var = T_settled.var(0, unbiased=False).clamp_min(1e-30)
            settled_fvu = [float(mse[a:b].sum() / var[a:b].sum()) for a, b in groups]
            rec['fwd_ridge_fvu_settled_by_group'] = settled_fvu
            line = (f"linear ridge forward in time, settled frames only (> {self.settled_after} frames "
                    f"after full gain), FVU per group {names}: "
                    + "/".join(f"{v:.2f}" for v in settled_fvu))
            if fwd is not None:
                all_fvu = [float(fwd['ridge_mse'][a:b].sum() / fwd['var'][a:b].sum()) for a, b in groups]
                line += " (all frames: " + "/".join(f"{v:.2f}" for v in all_fvu) + ")"
            self._print(line)

    def run(self, model, val_inputs, val_targets, meanmodes_t, stdmodes_t, step):
        if val_inputs is None or val_targets is None or val_targets.shape[0] < 3:
            return
        self.last_run_step = step

        model.eval()
        with torch.no_grad():
            preds = model(val_inputs) * stdmodes_t + meanmodes_t
        P = preds.detach().to('cpu', torch.float64)
        T = val_targets.detach().to('cpu', torch.float64)
        N, K = T.shape

        err = P - T
        mse_k = (err ** 2).mean(0)
        var_k = T.var(0, unbiased=False)
        cov_k = ((T - T.mean(0)) * (P - P.mean(0))).mean(0)
        safe_var = var_k.clamp_min(1e-30)
        fvu_k = mse_k / safe_var
        gain_k = cov_k / safe_var

        ridge_mse_k, ridge_note = self._ridge_val(val_targets)

        # Forward in time: CNN predictions made before training on the
        # newest batch vs. a ridge fitted only on older samples, both on the
        # same newest-batch rows (FVU relative to that batch's own variance).
        fwd = None
        ridge_fwd_mse, fwd_T = self._ridge_forward()
        if (self.last_fresh is not None and ridge_fwd_mse is not None
                and self.last_fresh[1].shape == fwd_T.shape
                and torch.allclose(self.last_fresh[1], fwd_T)):
            cnn_P, _ = self.last_fresh
            fwd = {
                'var': fwd_T.var(0, unbiased=False).clamp_min(1e-30),
                'cnn_mse': ((cnn_P - fwd_T) ** 2).mean(0),
                'ridge_mse': ridge_fwd_mse,
            }

        rec = {
            'step': step,
            'n_val': N,
            'cnn_err_rms': float(torch.sqrt(mse_k.sum())),
            'label_rms': float(torch.sqrt(var_k.sum())),
            'cnn_fvu_total': float(mse_k.sum() / safe_var.sum()),
            'cnn_fvu_mean_per_mode': float(fvu_k.mean()),
            'per_mode': {
                'label_std': torch.sqrt(var_k).tolist(),
                'cnn_err_rms': torch.sqrt(mse_k).tolist(),
                'cnn_fvu': fvu_k.tolist(),
                'cnn_gain': gain_k.tolist(),
            },
        }

        if not self._printed_note:
            self._print("RMS values are quadrature sums over modes (= wavefront RMS if the modal "
                        "basis is orthonormal), in the labels' units. FVU = fraction of variance "
                        "unexplained (0 perfect, 1 = no better than predicting the mean). "
                        "gain = cov(pred,true)/var(true) (1 ideal, <1 predictions shrunk).")
            self._printed_note = True

        self._print(f"step {step} | val N={N} | CNN error {rec['cnn_err_rms']:.2f} RMS on labels of "
                    f"{rec['label_rms']:.2f} RMS -> FVU total {rec['cnn_fvu_total']:.3f}, "
                    f"mean per-mode {rec['cnn_fvu_mean_per_mode']:.3f}")

        # --- Per mode-group table, CNN vs. linear ridge baseline -----------
        header = f"  {'modes':>9} {'label_rms':>10} {'cnn_err':>9} {'cnn_FVU':>8} {'cnn_gain':>9}"
        if ridge_mse_k is not None:
            header += f" | {'ridge_err':>9} {'ridge_FVU':>9}"
        if fwd is not None:
            header += f" || fwd: {'cnn_FVU':>8} {'ridge_FVU':>9}"
        self._print(header)
        groups = []
        for a, b in mode_groups(K):
            g = {
                'modes': [a, b],
                'label_rms': float(torch.sqrt(var_k[a:b].sum())),
                'cnn_err_rms': float(torch.sqrt(mse_k[a:b].sum())),
                'cnn_fvu': float(mse_k[a:b].sum() / safe_var[a:b].sum()),
                'cnn_gain_mean': float(gain_k[a:b].mean()),
            }
            line = (f"  {f'{a}-{b - 1}':>9} {g['label_rms']:>10.2f} {g['cnn_err_rms']:>9.2f} "
                    f"{g['cnn_fvu']:>8.3f} {g['cnn_gain_mean']:>9.3f}")
            if ridge_mse_k is not None:
                g['ridge_err_rms'] = float(torch.sqrt(ridge_mse_k[a:b].sum()))
                g['ridge_fvu'] = float(ridge_mse_k[a:b].sum() / safe_var[a:b].sum())
                line += f" | {g['ridge_err_rms']:>9.2f} {g['ridge_fvu']:>9.3f}"
            if fwd is not None:
                v = fwd['var'][a:b].sum()
                g['fwd_cnn_fvu'] = float(fwd['cnn_mse'][a:b].sum() / v)
                g['fwd_ridge_fvu'] = float(fwd['ridge_mse'][a:b].sum() / v)
                line += f" ||      {g['fwd_cnn_fvu']:>8.3f} {g['fwd_ridge_fvu']:>9.3f}"
            groups.append(g)
            self._print(line)
        rec['groups'] = groups

        # --- Linear baseline -------------------------------------------------
        if ridge_mse_k is not None:
            rec['ridge_err_rms'] = float(torch.sqrt(ridge_mse_k.sum()))
            rec['ridge_fvu_total'] = float(ridge_mse_k.sum() / safe_var.sum())
            rec['per_mode']['ridge_err_rms'] = torch.sqrt(ridge_mse_k).tolist()
            self._print(f"linear ridge baseline (same inputs, {ridge_note}): "
                        f"error {rec['ridge_err_rms']:.2f} RMS, FVU {rec['ridge_fvu_total']:.3f}")
        else:
            self._print(f"linear ridge baseline: not available ({ridge_note})")

        if fwd is not None:
            v = fwd['var'].sum()
            rec['fwd_cnn_fvu_total'] = float(fwd['cnn_mse'].sum() / v)
            rec['fwd_ridge_fvu_total'] = float(fwd['ridge_mse'].sum() / v)
            rec['fwd_cnn_err_rms'] = float(torch.sqrt(fwd['cnn_mse'].sum()))
            rec['fwd_ridge_err_rms'] = float(torch.sqrt(fwd['ridge_mse'].sum()))
            if self.win_fwd_fvu:
                rec['window_fwd_cnn_fvu_mean'] = float(np.mean(self.win_fwd_fvu))
            self._print(f"forward in time (both fitted only on data *before* the newest batch, tested on it; "
                        f"FVU vs. that batch's own variance): CNN error {rec['fwd_cnn_err_rms']:.2f} RMS, "
                        f"FVU {rec['fwd_cnn_fvu_total']:.3f}"
                        + (f" (window mean {rec['window_fwd_cnn_fvu_mean']:.3f})" if self.win_fwd_fvu else "")
                        + f" | ridge error {rec['fwd_ridge_err_rms']:.2f} RMS, FVU {rec['fwd_ridge_fvu_total']:.3f}")

        self._transient_report(K, fwd, rec)

        # --- Rank of predictions vs. labels ----------------------------------
        rec['label_dims_99'] = effective_dims(T)
        rec['cnn_pred_dims_99'] = effective_dims(P)
        self._print(f"independent directions carrying 99% of the variance: labels {rec['label_dims_99']}, "
                    f"CNN predictions {rec['cnn_pred_dims_99']}")

        # --- Error vs. residual in modes outside the predicted range ---------
        e_i = torch.sqrt((err ** 2).sum(1))
        amp_i = torch.sqrt((T ** 2).sum(1))
        rec['corr_err_vs_label_amplitude'] = _pearson(e_i, amp_i)
        oob_line = f"per-sample error vs. label amplitude: corr {rec['corr_err_vs_label_amplitude']:.2f}"
        if self.val_oob is not None and self.val_oob.shape[0] == N and float(self.val_oob.abs().sum()) > 0:
            oob = self.val_oob.to(torch.float64)
            rec['corr_err_vs_out_of_band'] = _pearson(e_i, oob)
            order = torch.argsort(oob)
            terciles = [order[i * N // 3:(i + 1) * N // 3] for i in range(3)]
            rec['err_rms_by_out_of_band_tercile'] = [float(torch.sqrt((e_i[t] ** 2).mean())) for t in terciles]
            rec['out_of_band_rms_by_tercile'] = [float(torch.sqrt((oob[t] ** 2).mean())) for t in terciles]
            oob_line += (f" | vs. residual in modes >= {self.nmodes}: corr {rec['corr_err_vs_out_of_band']:.2f}, "
                         f"CNN error by out-of-band tercile (low/mid/high "
                         + "/".join(f"{v:.1f}" for v in rec['out_of_band_rms_by_tercile']) + "): "
                         + "/".join(f"{v:.2f}" for v in rec['err_rms_by_out_of_band_tercile']))
        self._print(oob_line)

        # --- Data regime in this window ----------------------------------------
        if self.win_inband_rms:
            rec['window_inband_label_rms'] = float(np.mean(self.win_inband_rms))
            line = f"labels this window: modes 0-{self.nmodes - 1} {rec['window_inband_label_rms']:.2f} RMS"
            if self.win_oob_rms:
                rec['window_out_of_band_rms'] = float(np.mean(self.win_oob_rms))
                line += f", modes >= {self.nmodes} (not predicted) {rec['window_out_of_band_rms']:.2f} RMS"
            self._print(line)

        # --- Optimization ------------------------------------------------------
        if self.win_train:
            rec['window_train_loss'] = float(np.mean(self.win_train))
            line = f"optimization this window: train loss {rec['window_train_loss']:.4g}"
            if self.win_fresh:
                rec['window_fresh_loss'] = float(np.mean(self.win_fresh))
                line = (f"optimization this window: loss on each new batch before training on it "
                        f"{rec['window_fresh_loss']:.4g} vs. training loss {rec['window_train_loss']:.4g} "
                        f"(on that batch, or on the replay buffer if enabled)")
            if self.win_grad:
                g = np.asarray(self.win_grad)
                rec['grad_norm_median'] = float(np.median(g))
                rec['grad_norm_max'] = float(np.max(g))
                rec['grad_clipped_fraction'] = float(np.mean(g > self.clip_value))
                line += f" | grad norm median {rec['grad_norm_median']:.3g}, max {rec['grad_norm_max']:.3g}, "
                if np.isinf(self.clip_value):
                    line += "no clipping"
                else:
                    line += (f"clipped (>{self.clip_value:g}) on {100 * rec['grad_clipped_fraction']:.0f}% "
                             f"of steps")
            self._print(line)

        if self.win_std_drift:
            rec['norm_stdmodes_max_rel_change'] = float(np.max(self.win_std_drift))
            rec['norm_meanmodes_max_shift_in_std'] = float(np.max(self.win_mean_drift))
            self._print(f"normalization drift per step (max over window): stdmodes "
                        f"{100 * rec['norm_stdmodes_max_rel_change']:.1f}%, meanmodes "
                        f"{rec['norm_meanmodes_max_shift_in_std']:.3f} std")

        rec['hints'] = self._hints(rec)
        for hint in rec['hints']:
            self._print(f"hint: {hint}")

        self._write(rec)
        self._reset_window()

    def _hints(self, rec):
        hints = []
        if 'fwd_ridge_fvu_total' in rec:
            linear_wins = rec['fwd_ridge_fvu_total'] <= rec['fwd_cnn_fvu_total']
            linear_good = rec['fwd_ridge_fvu_total'] < 0.5
        else:
            linear_wins = 'ridge_fvu_total' in rec and rec['ridge_fvu_total'] <= rec['cnn_fvu_total']
            linear_good = True
        if linear_wins and linear_good:
            hints.append("a LINEAR map from the same inputs does as well or better than the CNN: the "
                         "limit is the network/training, not the information in the slopes")
        if not linear_good:
            hints.append("even a linear map fitted on the recent past explains little of the next batch: "
                         "the input -> mode relation itself changes over time (non-stationary data), "
                         "so this is not only a network problem")
        if rec.get('grad_clipped_fraction', 0) > 0.9:
            hints.append(f"gradients are clipped on almost every step (median norm "
                         f"{rec['grad_norm_median']:.3g} vs clip {self.clip_value:g}): with Adam this mostly "
                         "makes every batch weigh the same regardless of its error, rather than shrinking "
                         "the step -- the clip is always on, not just a guard against rare spikes")
        fresh, train = rec.get('window_fresh_loss'), rec.get('window_train_loss')
        if fresh and train and fresh > 1.5 * train:
            hints.append("loss on each new batch (before training on it) is much higher than the training "
                         "loss: the network fits the data it trains on but generalizes poorly to new data")
        low_gain = [g for g in rec['groups'] if g['cnn_gain_mean'] < 0.7]
        if low_gain:
            names = ", ".join(f"{g['modes'][0]}-{g['modes'][1] - 1}" for g in low_gain)
            hints.append(f"predictions strongly shrunk towards the mean for modes {names}: typical of "
                         "modes whose signal is buried in noise/aliasing (or not learned yet)")
        terc = rec.get('err_rms_by_out_of_band_tercile')
        if terc and terc[0] > 0 and terc[2] > 1.5 * terc[0] and rec.get('corr_err_vs_out_of_band', 0) > 0.3:
            hints.append(f"error grows markedly with the residual in modes >= {self.nmodes}: likely limited "
                         "by aliasing / pyramid nonlinearity from the modes the loop doesn't correct")
        if rec.get('norm_stdmodes_max_rel_change', 0) > 0.05:
            hints.append("normalization statistics still moving by >5% per step: the network is chasing "
                         "a moving target (consider a smaller norm_alpha, or freezing them)")
        return hints

    def _write(self, rec):
        try:
            with open(self.jsonl_filename, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except OSError as e:
            self._print(f"could not write {self.jsonl_filename}: {e}")
