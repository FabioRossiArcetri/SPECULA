import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.lib.cnn_checkpoint import (check_trained_settings, load_stats, load_weights,
                                        save_checkpoint, stats_filename)
from specula.lib.efficient_u_net import UNetRegressor
from specula.lib.frame_stacker import FrameStacker
from specula.lib.nn_training_diagnostics import TrainingDiagnostics


def to_torch(arr, device='cpu'):
    """numpy/cupy array -> float32 torch tensor on `device`."""
    return torch.from_numpy(np.ascontiguousarray(cpuArray(arr))).to(device=device, dtype=torch.float32)


def network_input(x, input_channels):
    """The network input from a buffered 'input_2d_batch' value: the value
    itself, one map per frame, except for the historical single-channel
    convention -- input_channels 1 and a value that still has a channel axis,
    (B, 2, H, W), e.g. amplitude and phase -- where it is the per-pixel
    product of its two channels. A value with one map per frame, (B, H, W)
    (e.g. Slopes2D on slopes-from-intensity slopes), is used as is, with
    input_channels 1."""
    if input_channels == 1 and x.ndim == 4:
        return x[:, 1] * x[:, 0]
    return x


class EarlyStopping:
    """Returns True once the loss hasn't improved by more than min_delta for
    `patience` consecutive calls."""
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Conv2dNetTrainer(BaseProcessingObj):
    """
    Trains a UNetRegressor (specula/lib/efficient_u_net.py) to predict modes
    from 2D maps, on batches buffered by a DataBuffer during the simulation.

    On every trigger, the newest batch is:

    1. read (``input_2d_batch`` -> network input, ``labels`` -> modes to
       predict), with the frames stacked over time (n_frames) and the
       samples taken while the loop wasn't at full gain dropped (optional
       ``gain_mod`` input);
    2. used to set the normalization (from the first batch, or from the
       first init_samples samples), or to update it (EMA, unless
       freeze_norm);
    3. split into training and validation samples; the validation ones join
       a validation set of the most recent 1000;
    4. trained on: epoch_len steps on the whole batch or, with a replay
       buffer (replay_size > 0), replay_steps minibatches drawn from all the
       training samples collected so far;
    5. followed by a validation step; every 10 steps the checkpoint is
       saved (see specula/lib/cnn_checkpoint.py), every diag_interval steps
       the diagnostics are printed (specula/lib/nn_training_diagnostics.py).

    The network works on normalized inputs and outputs, but the loss is
    computed in the labels' physical units (e.g. nm): the network output is
    denormalized with meanmodes/stdmodes first.

    Parameters
    ----------
    network_filename : str
        Checkpoint file (``.pth`` is appended if there's no extension); the
        stats are saved next to it as ``<name>_stats.json``.
    nmodes : int
        Number of modes the network predicts.
    load_from_file : bool
        Resume from network_filename if it exists (it must have been
        trained with the same network parameters).

    Network: channels, depth, conv_block_type, head_type, head_grid, dropout
    -- see UNetRegressor (channels = base_channels, dropout = dropout_level).
    Conv2dNetTester/Conv2dNetRec must use the same values.

    input_channels : int
        Channels of the network input, per frame: 2 for the x/y slope maps
        Slopes2D gives for ordinary slopes, 1 for a single map per frame
        (e.g. Slopes2D on slopes-from-intensity slopes, or the per-pixel
        product of a 2-channel buffered value -- see network_input()).
    n_frames : int
        Consecutive frames stacked along the channel axis as the network
        input: sample t gets [x(t), x(t-1), ..., x(t-n_frames+1)], so the
        network sees input_channels * n_frames channels. Over a few frames
        the turbulence is strongly correlated while noise isn't.

    Labels:

    label_offset : int
        Index of the first mode within the buffered 'labels' value (default
        1, for labels whose first entry is a placeholder).
    baseline_offset : int
        Index of the first mode within the buffered optional 'baseline'
        input. When it is connected, the network is trained on the residual
        labels - baseline (residual learning: it only has to learn what the
        baseline reconstructor misses).
    max_label_rms : float
        If set, samples whose labels (modes 0..nmodes-1, in quadrature) are
        larger than this are dropped -- see _label_rms_mask(). In the labels'
        units; pick it from the diagnostics' "labels this window" line.

    Loop transients (optional 'gain_mod' input, the loop gain modulation):

    gain_mod_threshold : float
        Samples with gain_mod below this are dropped: while the loop is
        open or at reduced gain the residual is much larger than in closed
        loop, and not representative of it.
    settle_frames : int
        The first settle_frames frames after the loop is back at full gain
        (gain_mod >= 1) count as its re-convergence transient; the count
        carries over across triggers.
    transient_weight : float
        Loss weight, in [0, 1], of those transient frames (1: same as the
        others; 0: dropped).

    Normalization (inputs: one mean/std over all pixels; outputs: per mode):

    freeze_norm : bool
        If True, estimated once (or restored from the stats file when
        resuming) and never updated; otherwise updated on every batch by an
        exponential moving average with rate norm_alpha.
    norm_alpha : float
        EMA rate, in (0, 1]; a time constant of ~1/norm_alpha triggers.
    init_samples : int
        If > 0, collect this many samples (over as many triggers as needed,
        without training) to estimate the normalization; with a replay
        buffer they also seed it. 0: estimate it from the first batch.

    Training:

    val_split : float
        Fraction of each batch used for validation.
    epoch_len : int
        Full-batch steps per trigger, without a replay buffer.
    replay_size : int
        If > 0, keep up to this many training samples (a ring buffer in CPU
        RAM: replay_size x (input pixels + nmodes) x 4 bytes) and train on
        replay_steps minibatches of replay_batch samples drawn from all of
        them. Each batch covers a fraction of a second of one atmospheric
        condition: trained on alone, the network forgets earlier conditions.
    replay_subsample : int
        Only every Nth training frame enters the replay buffer: consecutive
        frames at kHz rates are nearly identical.
    replay_steps, replay_batch : int
        Minibatch steps per trigger and minibatch size, with replay.
    loss_delta : float
        Huber loss transition point, in the labels' units: errors above it
        count linearly instead of quadratically, so occasional
        large-residual samples don't dominate the loss.
    norm_loss_weight : float
        Weight of an extra loss term: the squared error per mode normalized
        by that mode's std (stdmodes). In the physical-unit loss a mode counts
        in proportion to its variance, so small modes get little push to be
        learned; this term weighs every mode by its own spread. The reported
        losses don't include it.
    grad_clip_value : float
        Maximum total gradient norm; None or <= 0 disables clipping.
    lr_decay : float
        If set, in (0, 1): the learning rate (initially 1e-3) is multiplied
        by it after every trigger (never below 5e-6). Default None: reduced
        by 0.8 when the validation loss stops improving for patience // 10
        triggers.
    patience : int
        Training stops after this many triggers without improvement of the
        validation loss.

    Diagnostics:

    diag_interval : int
        Every this many steps, print diagnostics meant to locate what limits
        the error, and append them as one JSON line to
        ``<name>_diag.jsonl``. 0 disables them.
    diag_ridge_samples : int
        Most recent training samples used to fit the diagnostics' linear
        (ridge regression) baseline.
    """

    def __init__(self,
                 network_filename,
                 nmodes=20,
                 load_from_file=False,
                 channels=32,
                 depth=5,
                 conv_block_type=0,
                 head_type='pooled',
                 head_grid=32,
                 dropout=0.01,
                 input_channels=1,
                 n_frames=1,
                 label_offset=1,
                 baseline_offset=0,
                 max_label_rms=None,
                 gain_mod_threshold=1.0,
                 settle_frames=0,
                 transient_weight=1.0,
                 freeze_norm=False,
                 norm_alpha=0.001,
                 init_samples=0,
                 val_split=0.2,
                 epoch_len=20,
                 replay_size=0,
                 replay_subsample=1,
                 replay_steps=100,
                 replay_batch=64,
                 loss_delta=20.0,
                 norm_loss_weight=0.0,
                 grad_clip_value=1.0,
                 lr_decay=None,
                 patience=600,
                 diag_interval=10,
                 diag_ridge_samples=4000,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if not 0 < norm_alpha <= 1:
            raise ValueError(f'norm_alpha must be in (0, 1], got {norm_alpha}')
        if not loss_delta > 0:
            raise ValueError(f'loss_delta must be > 0, got {loss_delta}')
        if init_samples < 0:
            raise ValueError(f'init_samples must be >= 0, got {init_samples}')
        if replay_size < 0 or replay_subsample < 1 or replay_steps < 1 or replay_batch < 1:
            raise ValueError('replay_size must be >= 0, and replay_subsample, replay_steps '
                             'and replay_batch >= 1')
        if lr_decay is not None and not 0 < lr_decay < 1:
            raise ValueError(f'lr_decay must be in (0, 1) or None, got {lr_decay}')
        if settle_frames < 0 or not 0 <= transient_weight <= 1:
            raise ValueError('settle_frames must be >= 0 and transient_weight in [0, 1]')
        if n_frames < 1 or norm_loss_weight < 0:
            raise ValueError('n_frames must be >= 1 and norm_loss_weight >= 0')

        if not os.path.splitext(network_filename)[1]:
            network_filename += '.pth'
        self.network_filename = network_filename
        self.stats_filename = stats_filename(network_filename)

        self.nmodes = nmodes
        self.head_type = head_type
        self.head_grid = head_grid
        self.input_channels = input_channels
        self.n_frames = n_frames
        self.label_offset = label_offset
        self.baseline_offset = baseline_offset
        self.max_label_rms = max_label_rms
        self.gain_mod_threshold = gain_mod_threshold
        self.settle_frames = settle_frames
        self.transient_weight = transient_weight
        self.freeze_norm = freeze_norm
        self.norm_alpha = norm_alpha
        self.init_samples = init_samples
        self.val_split = val_split
        self.epoch_len = epoch_len
        self.replay_size = replay_size
        self.replay_subsample = replay_subsample
        self.replay_steps = replay_steps
        self.replay_batch = replay_batch
        self.loss_delta = loss_delta
        self.norm_loss_weight = norm_loss_weight
        # float('inf'): clip_grad_norm_ then only measures the norm.
        self.grad_clip_value = grad_clip_value if grad_clip_value and grad_clip_value > 0 else float('inf')
        self.lr_decay = lr_decay
        self.min_lr = 5e-6

        self.stacker = FrameStacker(n_frames, self.xp)
        self._frames_at_full_gain = 0
        self._warned_no_gain_mod = False
        self._empty_batches = 0
        self._init_batches = []     # (inputs, modes, weights) collected before training starts

        # Normalization: restored from the stats file when resuming,
        # otherwise estimated from the first data (see trigger()).
        self.meanp = self.stdp = self.meanmodes = self.stdmodes = None

        model = UNetRegressor(
            input_channels=input_channels * n_frames,
            output_size=nmodes,
            base_channels=channels,
            dropout_level=dropout,
            depth=depth,
            conv_block_type=conv_block_type,
            head_type=head_type,
            head_grid=head_grid,
        )
        if load_from_file:
            self._load(model)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            model = nn.DataParallel(model.to(self.device), device_ids=list(range(n_gpus)))
            print(f'[{self.name}] Training on {n_gpus} GPUs (DataParallel)', flush=True)
        else:
            print(f'[{self.name}] Training on {self.device}', flush=True)
        self.model = model.to(self.device)

        # Plain Huber loss in physical units, used for the reported losses;
        # training uses _loss(), which can add weights and the normalized term.
        self.loss_fn = nn.HuberLoss(delta=loss_delta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=patience // 10, min_lr=self.min_lr)
        self.early_stopping = EarlyStopping(patience=patience)
        self.should_stop = False

        self.inputs['input_2d_batch'] = InputValue(type=BaseValue)
        self.inputs['labels'] = InputValue(type=BaseValue)
        self.inputs['baseline'] = InputValue(type=BaseValue, optional=True)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.outputs['loss'] = BaseValue(target_device_idx=target_device_idx)
        self.outputs['val_loss'] = BaseValue(target_device_idx=target_device_idx)

        self.max_val = 1000
        self.val_inputs = self.val_targets = None
        self.replay_x = self.replay_y = self.replay_w = None
        self.replay_count = 0
        self._replay_pos = 0
        self.step_count = 0
        self.mode_gain = None
        self.loss = self.val_loss = None
        self.min_loss = float('inf')

        self.diag_interval = diag_interval
        self.diag = None
        if diag_interval > 0:
            self.diag = TrainingDiagnostics(
                name=self.name,
                nmodes=nmodes,
                max_val=self.max_val,
                ridge_samples=diag_ridge_samples,
                jsonl_filename=os.path.splitext(network_filename)[0] + '_diag.jsonl',
                clip_value=self.grad_clip_value,
                device=self.device,
                settled_after=settle_frames if settle_frames > 0 else 200,
            )

    # ------------------------------------------------------------------
    #   Checkpoint
    # ------------------------------------------------------------------

    def _load(self, model):
        if not os.path.isfile(self.network_filename):
            print(f'[{self.name}] No checkpoint at {self.network_filename}: training a new network.',
                  flush=True)
            return
        # Checked before loading: a mismatch must stop here, not end with a
        # new network overwriting the checkpoint.
        check_trained_settings(self.network_filename, n_frames=self.n_frames,
                               head_type=self.head_type, head_grid=self.head_grid)
        load_weights(model, self.network_filename)
        print(f'[{self.name}] Model loaded from {self.network_filename}', flush=True)
        try:
            stats = load_stats(self.network_filename)
        except FileNotFoundError:
            print(f'[{self.name}] WARNING: stats file not found: the normalization will be '
                  f'estimated again from the data.', flush=True)
            return
        self.meanp = stats['meanp']
        self.stdp = stats['stdp']
        self.meanmodes = self.xp.array(stats['meanmodes'], dtype=self.dtype)
        self.stdmodes = self.xp.array(stats['stdmodes'], dtype=self.dtype)

    def _inner_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _save(self, val_loss):
        save_checkpoint(self._inner_model(), self.network_filename, {
            'meanp': float(self.meanp),
            'stdp': float(self.stdp),
            'meanmodes': cpuArray(self.meanmodes).tolist(),
            'stdmodes': cpuArray(self.stdmodes).tolist(),
            'nmodes': self.nmodes,
            'n_frames': self.n_frames,
            'head_type': self.head_type,
            'head_grid': self.head_grid,
            'mode_gain': self.mode_gain.tolist() if self.mode_gain is not None else None,
            'min_loss': val_loss,   # this save's validation loss (not the best ever)
        })
        print(f'[{self.name}] Model saved to {self.network_filename}', flush=True)

    # ------------------------------------------------------------------
    #   Data
    # ------------------------------------------------------------------

    def _read_batch(self):
        """The newest buffered batch, without the samples dropped because of
        gain_mod or max_label_rms: (network inputs, not normalized yet; modes
        to predict; label modes beyond nmodes, for the diagnostics; frames
        since the loop came back to full gain, or None; loss weights, or
        None). None if there is nothing to train on."""
        x_in = self.local_inputs['input_2d_batch']
        labels_in = self.local_inputs['labels']
        if x_in is None or labels_in is None:
            return None
        # Stacked before any frame is dropped: each sample keeps its true
        # preceding frames.
        x = self.stacker(network_input(x_in.get_value(), self.input_channels))
        labels = labels_in.get_value()
        first = self.label_offset
        modes = labels[:, first:first + self.nmodes]
        out_of_band = labels[:, first + self.nmodes:]
        baseline_in = self.local_inputs['baseline']
        if baseline_in is not None:
            b = self.baseline_offset
            modes = modes - baseline_in.get_value()[:, b:b + self.nmodes]

        fsr = None
        keep = self._label_rms_mask(modes)
        gain_mod_in = self.local_inputs['gain_mod']
        if gain_mod_in is None:
            if self.settle_frames > 0 and not self._warned_no_gain_mod:
                print(f"[{self.name}] WARNING: settle_frames needs the 'gain_mod' input, "
                      f"which isn't connected: ignored.", flush=True)
                self._warned_no_gain_mod = True
        else:
            gain_mod = self.xp.asarray(gain_mod_in.get_value()).reshape(-1)
            fsr = self._frames_since_reclose(gain_mod)
            gain_keep = gain_mod >= self.gain_mod_threshold
            if self.settle_frames > 0 and self.transient_weight == 0:
                gain_keep = gain_keep & self.xp.asarray(fsr > self.settle_frames)
            keep = gain_keep if keep is None else keep & gain_keep

        if keep is None:
            self._empty_batches = 0
            return x, modes, out_of_band, None, None
        if not self.xp.any(keep):
            self._empty_batches += 1
            if self._empty_batches == 3:
                print(f'[{self.name}] WARNING: 3 batches in a row with no usable sample. If this '
                      f'continues, the loop has most likely diverged (check the Strehl and the '
                      f'batch medians above): training is idle from here on.', flush=True)
            return None
        self._empty_batches = 0
        if fsr is not None:
            fsr = fsr[cpuArray(keep)]
        weights = None
        if fsr is not None and self.settle_frames > 0 and 0 < self.transient_weight < 1:
            weights = np.where(fsr <= self.settle_frames, self.transient_weight, 1.0).astype(np.float32)
        return x[keep], modes[keep], out_of_band[keep], fsr, weights

    def _label_rms_mask(self, modes):
        """Which samples to keep by the size of their labels, or None when
        max_label_rms is off. Samples whose residual is far larger than usual
        are not representative of the loop this network is for: at that size a
        non-modulated pyramid is outside its linear range, so there is no
        input -> mode relation left to learn (a linear map fitted on such
        batches fails on the next one just as badly), and a diverging loop
        would otherwise fill the replay buffer with them."""
        if self.max_label_rms is None:
            return None
        rms = self.xp.sqrt(self.xp.sum(modes ** 2, axis=1))
        keep = rms <= self.max_label_rms
        dropped = int(modes.shape[0] - self.xp.sum(keep))
        if dropped > 0.1 * modes.shape[0]:
            print(f'[{self.name}] {dropped}/{modes.shape[0]} samples dropped: label RMS above '
                  f'max_label_rms={self.max_label_rms:g} (batch median {float(self.xp.median(rms)):.0f})',
                  flush=True)
        return keep

    def _frames_since_reclose(self, gain_mod):
        """For each frame (in time order), how many consecutive frames the
        loop has been at full gain (gain_mod >= 1) up to and including it;
        0 while it isn't. Carries over from the previous trigger."""
        g = cpuArray(gain_mod).reshape(-1)
        out = np.empty(g.shape[0], dtype=np.int64)
        count = self._frames_at_full_gain
        for i, value in enumerate(g):
            count = count + 1 if value >= 1.0 - 1e-6 else 0
            out[i] = count
        self._frames_at_full_gain = count
        return out

    # ------------------------------------------------------------------
    #   Normalization
    # ------------------------------------------------------------------

    def _collect_init_samples(self, x, modes, weights):
        """Collect samples until init_samples are available, then set the
        normalization from all of them and seed the replay buffer with the
        ones from earlier batches. True once done (the current batch then
        goes on to training)."""
        self._init_batches.append((to_torch(x), to_torch(modes),
                                   torch.from_numpy(weights) if weights is not None
                                   else torch.ones(x.shape[0])))
        n = sum(b[0].shape[0] for b in self._init_batches)
        if n < self.init_samples:
            print(f'[{self.name}] Collecting initialization samples: {n}/{self.init_samples}', flush=True)
            return False
        earlier, self._init_batches = self._init_batches[:-1], []
        all_x = torch.cat([b[0] for b in earlier] + [to_torch(x)])
        all_y = torch.cat([b[1] for b in earlier] + [to_torch(modes)])
        self._set_normalization(all_x, all_y)
        if self.replay_size > 0 and earlier:
            self._replay_add(*(torch.cat(parts) for parts in zip(*earlier)))
        return True

    def _set_normalization(self, x, y):
        """From CPU torch tensors: inputs and modes (physical units)."""
        x = x.to(torch.float64)
        y = y.to(torch.float64)
        self.meanp = float(x.mean())
        self.stdp = float(x.std(unbiased=False)) + 1e-8
        self.meanmodes = self.xp.asarray(y.mean(0).numpy(), dtype=self.dtype)
        self.stdmodes = self.xp.asarray(y.std(0, unbiased=False).numpy() + 1e-8, dtype=self.dtype)
        frozen = ', frozen from now on' if self.freeze_norm else ''
        print(f'[{self.name}] Normalization initialized from {x.shape[0]} samples{frozen}.', flush=True)

    def _update_normalization(self, x, modes):
        """Exponential moving average with rate norm_alpha."""
        old_mean, old_std = cpuArray(self.meanmodes).copy(), cpuArray(self.stdmodes).copy()
        a = self.norm_alpha
        self.meanp = (1 - a) * self.meanp + a * float(self.xp.mean(x))
        self.stdp = (1 - a) * self.stdp + a * (float(self.xp.std(x)) + 1e-8)
        self.meanmodes = (1 - a) * self.meanmodes + a * self.xp.mean(modes, axis=0)
        self.stdmodes = (1 - a) * self.stdmodes + a * (self.xp.std(modes, axis=0) + 1e-8)
        if self.diag is not None:
            self._diag_safe(self.diag.record_normalization_drift, old_mean, old_std,
                            cpuArray(self.meanmodes), cpuArray(self.stdmodes))

    # ------------------------------------------------------------------
    #   Replay buffer
    # ------------------------------------------------------------------

    def _replay_add(self, x, y, w=None):
        """Add samples (CPU tensors, rows in time order: network inputs before
        normalization, labels in physical units and, optionally, loss
        weights) to the ring buffer, keeping every replay_subsample-th one."""
        if w is None:
            w = torch.ones(x.shape[0])
        step = self.replay_subsample
        x, y, w = x[::step][-self.replay_size:], y[::step][-self.replay_size:], w[::step][-self.replay_size:]
        if x.shape[0] == 0:
            return
        if self.replay_x is None:
            self.replay_x = torch.empty((self.replay_size,) + tuple(x.shape[1:]), dtype=torch.float32)
            self.replay_y = torch.empty((self.replay_size, y.shape[1]), dtype=torch.float32)
            self.replay_w = torch.empty(self.replay_size, dtype=torch.float32)
        n = x.shape[0]
        idx = (self._replay_pos + torch.arange(n)) % self.replay_size
        self.replay_x[idx] = x.to(torch.float32)
        self.replay_y[idx] = y.to(torch.float32)
        self.replay_w[idx] = w.to(torch.float32)
        self._replay_pos = (self._replay_pos + n) % self.replay_size
        self.replay_count = min(self.replay_count + n, self.replay_size)

    def _replay_minibatch(self):
        """A uniformly drawn minibatch, with the inputs normalized with the
        *current* statistics: (inputs, labels, loss weights)."""
        idx = torch.randint(0, self.replay_count, (min(self.replay_batch, self.replay_count),))
        x = (self.replay_x[idx].to(self.device) - float(self.meanp)) / float(self.stdp)
        if x.dim() == 3:    # single-channel frames stored without a channel axis
            x = x.unsqueeze(1)
        return x, self.replay_y[idx].to(self.device), self.replay_w[idx].to(self.device)

    # ------------------------------------------------------------------
    #   Training
    # ------------------------------------------------------------------

    def trigger(self):
        if self.should_stop:
            return
        batch = self._read_batch()
        if batch is None:
            return
        x, modes, out_of_band, fsr, weights = batch

        if self.meanmodes is None:
            if self.init_samples > 0:
                if not self._collect_init_samples(x, modes, weights):
                    return
            else:
                self._set_normalization(to_torch(x), to_torch(modes))
        elif not self.freeze_norm:
            self._update_normalization(x, modes)

        x_norm = (x - self.meanp) / self.stdp
        if x_norm.ndim == 3:    # single-channel frames: add the channel axis
            x_norm = x_norm[:, self.xp.newaxis]

        n = x.shape[0]
        n_val = max(1, int(n * self.val_split))
        perm = torch.randperm(n).numpy()
        train_idx, val_idx = perm[:n - n_val], perm[n - n_val:]

        dev = self.device
        inputs = to_torch(x_norm[train_idx], dev)
        targets = to_torch(modes[train_idx], dev)
        train_w = torch.from_numpy(weights[train_idx]).to(dev) if weights is not None else None
        mean_t = to_torch(self.meanmodes, dev)
        std_t = to_torch(self.stdmodes, dev)
        self._add_validation(to_torch(x_norm[val_idx], dev), to_torch(modes[val_idx], dev))

        fresh_loss = None
        if self.diag is not None:
            self._diag_add_batch(x, modes, out_of_band, fsr, train_idx, val_idx, targets)
            fresh_loss = self._fresh_loss(inputs, targets, mean_t, std_t,
                                          fsr[train_idx] if fsr is not None else None)

        if self.replay_size > 0:
            in_time_order = np.sort(train_idx)
            self._replay_add(to_torch(x[in_time_order]), to_torch(modes[in_time_order]),
                             torch.from_numpy(weights[in_time_order]) if weights is not None else None)

        loss, grad_norms = self._train(inputs, targets, train_w, mean_t, std_t)
        if loss is None:
            return
        self.loss = loss
        self.step_count += 1
        if self.lr_decay is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = max(group['lr'] * self.lr_decay, self.min_lr)
        if self.diag is not None:
            self._diag_safe(self.diag.record_step, fresh_loss, self.loss, grad_norms)

        self._validate(mean_t, std_t)

    def _add_validation(self, inputs, targets):
        """Append to the validation set, keeping the most recent max_val samples."""
        if self.val_inputs is None:
            self.val_inputs, self.val_targets = inputs, targets
        else:
            self.val_inputs = torch.cat([self.val_inputs, inputs])[-self.max_val:]
            self.val_targets = torch.cat([self.val_targets, targets])[-self.max_val:]

    def _loss(self, preds, targets, weights, std_t):
        """Training loss, in physical units: the Huber loss per sample (mean
        over modes), plus norm_loss_weight times the squared error
        normalized per mode by std_t; then the mean over samples, weighted
        by `weights` if given."""
        per_sample = F.huber_loss(preds, targets, reduction='none', delta=self.loss_delta).mean(dim=1)
        if self.norm_loss_weight > 0:
            per_sample = per_sample + self.norm_loss_weight * (((preds - targets) / std_t) ** 2).mean(dim=1)
        if weights is None:
            return per_sample.mean()
        return (weights * per_sample).sum() / weights.sum()

    def _train(self, inputs, targets, weights, mean_t, std_t):
        """This trigger's gradient steps: replay_steps minibatches from the
        replay buffer or, without one, epoch_len steps on the new batch.
        Returns the training loss (plain Huber loss in physical units: the
        mean over the replay steps, or the last full-batch step) and the
        gradient norms, or (None, ...) if the loss is not finite."""
        self.model.train()
        use_replay = self.replay_size > 0
        losses, grad_norms = [], []
        for _ in range(self.replay_steps if use_replay else self.epoch_len):
            x, y, w = self._replay_minibatch() if use_replay else (inputs, targets, weights)
            self.optimizer.zero_grad()
            preds = self.model(x) * std_t + mean_t      # denormalized: physical units
            loss = self._loss(preds, y, w, std_t)
            if not torch.isfinite(loss):
                print(f'[{self.name}] NaN/Inf loss, skipping this batch.', flush=True)
                return None, grad_norms
            loss.backward()
            grad_norms.append(float(nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)))
            self.optimizer.step()
            losses.append(self.loss_fn(preds.detach(), y).item())
        return (float(np.mean(losses)) if use_replay else losses[-1]), grad_norms

    def _validate(self, mean_t, std_t):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.val_inputs) * std_t + mean_t
            self.val_loss = self.loss_fn(preds, self.val_targets).item()
        self._update_mode_gain(preds, self.val_targets)
        if self.lr_decay is None:
            self.plateau_scheduler.step(self.val_loss)
        self.min_loss = min(self.min_loss, self.val_loss)

        # Periodic, not only on improvement: otherwise a run whose validation
        # loss never beats an early best step would save nothing to resume from.
        if self.step_count % 10 == 0:
            self._save(self.val_loss)

        lr = self.optimizer.param_groups[0]['lr']
        print(f'[{self.name}] Step {self.step_count} | LR {lr:.2e} | Train {self.loss:.6f} | '
              f'Val {self.val_loss:.6f} | Best {self.min_loss:.6f}', flush=True)

        if self.diag is not None and self.step_count % self.diag_interval == 0:
            self._run_diagnostics(mean_t, std_t)

        if self.step_count % 100 == 0:
            self.logger.info(
                f'[{self.name}] Step {self.step_count}: loss between CNN output and '
                f'modal_analysis ground truth (unweighted Huber loss, physical units) -- '
                f'train={self.loss:.6f}, val={self.val_loss:.6f}')

        if self.early_stopping(self.val_loss):
            self.should_stop = True
            print(f'[{self.name}] Early stopping triggered at step {self.step_count}', flush=True)

    def _update_mode_gain(self, preds, targets):
        """Per-mode prediction gain cov(pred, true) / var(true) on the
        validation set: how much the network shrinks each mode towards the
        mean, which in a loop acts exactly like a lower gain on that mode. It
        is saved in the stats file so Conv2dNetRec / Conv2dNetTester can undo
        it (see cnn_checkpoint.calibration_factor), which keeps the loop gains
        in a config meaningful across retrains -- a network that shrinks less
        after a retrain otherwise silently raises the effective loop gain.
        Smoothed over saves, since one validation window is noisy."""
        p = preds.detach().to(torch.float64)
        t = targets.to(torch.float64)
        centred = t - t.mean(0)
        gain = ((centred * (p - p.mean(0))).mean(0)
                / centred.pow(2).mean(0).clamp_min(1e-30)).cpu().numpy()
        alpha = 0.3
        self.mode_gain = gain if self.mode_gain is None else (1 - alpha) * self.mode_gain + alpha * gain

    # ------------------------------------------------------------------
    #   Diagnostics (observe only; must never break training)
    # ------------------------------------------------------------------

    def _diag_safe(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f'[{self.name}][diag] ERROR in {fn.__name__}: {e}', flush=True)

    def _diag_add_batch(self, x, modes, out_of_band, fsr, train_idx, val_idx, train_targets):
        xp = self.xp
        inband_rms = to_torch(xp.sqrt(xp.sum(modes ** 2, axis=1)))
        if out_of_band.shape[1] > 0:
            oob_rms = to_torch(xp.sqrt(xp.sum(out_of_band ** 2, axis=1)))
            val_oob_rms = oob_rms[torch.as_tensor(val_idx)]
        else:
            oob_rms = torch.zeros(0)
            val_oob_rms = torch.zeros(len(val_idx))
        self._diag_safe(
            self.diag.add_batch,
            raw_train_x=to_torch(x[train_idx]),
            train_y=train_targets.cpu(),
            raw_val_x=to_torch(x[val_idx]),
            val_oob_rms=val_oob_rms,
            inband_rms=inband_rms,
            oob_rms=oob_rms,
            train_fsr=torch.from_numpy(fsr[train_idx]) if fsr is not None else None,
        )

    def _fresh_loss(self, inputs, targets, mean_t, std_t, fsr):
        """Loss on the new batch *before* training on it: an out-of-sample
        number, to compare with the training loss."""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(inputs) * std_t + mean_t
        self._diag_safe(self.diag.record_fresh_predictions, preds.cpu(), targets.cpu(),
                        torch.from_numpy(fsr) if fsr is not None else None)
        return self.loss_fn(preds, targets).item()

    def _run_diagnostics(self, mean_t, std_t):
        self.diag.name = self.name
        self._diag_safe(self.diag.run, self.model, self.val_inputs, self.val_targets,
                        mean_t, std_t, self.step_count)

    # ------------------------------------------------------------------

    def post_trigger(self):
        super().post_trigger()
        # generation_time must follow, or downstream consumers (e.g. a
        # PlotDisplay) never see these outputs as new.
        for name, value in (('loss', self.loss), ('val_loss', self.val_loss)):
            if value is not None:
                self.outputs[name].set_value(value)
                self.outputs[name].generation_time = self.current_time

    def finalize(self):
        if self.diag is not None and self.step_count > 0 and self.diag.last_run_step != self.step_count:
            dev = self.val_targets.device
            self._run_diagnostics(to_torch(self.meanmodes, dev), to_torch(self.stdmodes, dev))
        print(f'[{self.name}] Training complete!', flush=True)
        print(f'  Steps: {self.step_count}', flush=True)
        print(f'  Best loss: {self.min_loss:.6f}', flush=True)
        print(f'  Stats saved: {self.stats_filename}', flush=True)
        if self.diag is not None:
            print(f'  Diagnostics: {self.diag.jsonl_filename}', flush=True)
