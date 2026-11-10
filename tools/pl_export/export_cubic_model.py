#!/usr/bin/env python
"""Export a SPECULA ``LanternCubicModel`` file from PLREC's fitted per-wavelength models.

PLREC fits, for every wavelength of a dispersed photonic-lantern dataset, a sparse
degree-3 polynomial that regresses the phasor ``[sin, cos](2*pi*p/period)`` of each
petal piston from the lantern's channel powers (``ml_reconstruct.unwrap.
fit_wrapped_models``), and caches each fit as a pickle of scikit-learn objects.
Those pickles depend on the scikit-learn version and carry the whole feature
pipeline, so they are not something a simulation should read: this tool flattens
them into plain numbers.

For each wavelength the model is

    phasors = monomials((signals - input_mean) / input_std)[terms] @ weights + offset

where ``terms`` are the non-zero monomials of the sparse fit, given by their
exponents, and the standardization of the monomials has been folded into
``weights``/``offset``. Nothing of scikit-learn survives in the output, and the
evaluation is a gather, a product and a small matrix product.

    python export_cubic_model.py \\
        --dataset /raid1/frossi/dev/PLREC/data/ml/pl19_1200/spectral_41wl.npz \\
        --cache /raid1/frossi/dev/PLREC/data/ml/pl19_1200/cache_perlambda \\
        --tag wrap1-d3-K1282-f19-n8000 --wavelengths 1150:1250:10 \\
        --span 2000 --out pl_rec/pl19_1200_k1282.npz

``--check N`` compares the exported evaluation with the cached scikit-learn
pipeline on N random dataset rows (it uses the pickles only for that).
"""
import argparse
import json
import os
import pickle

import numpy as np


def parse_wavelengths(text):
    """'1150:1250:10' or '1150,1160,1170' -> list of floats"""
    if ':' in text:
        parts = [p for p in text.split(':') if p.strip()]
        if len(parts) not in (2, 3):
            raise ValueError(f'cannot parse wavelengths {text!r}: use START:STOP[:STEP]')
        start, stop = float(parts[0]), float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 10.0
        if step <= 0:
            raise ValueError('the wavelength step must be positive')
        n = int(np.round((stop - start) / step)) + 1
        values = start + step * np.arange(n)
        if abs(values[-1] - stop) > 1e-6:
            values = np.append(values, stop)
        return [float(v) for v in values]
    return [float(v) for v in text.replace(';', ',').split(',') if v.strip()]


def write_npz(path, header, arrays):
    """Write the npz layout of iteras.lib.storage (JSON header + named arrays)"""
    cards = [[key, value, ''] for key, value in header.items()]
    arrays = {name: value for name, value in arrays.items() if value is not None}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, __header__=np.array(json.dumps(cards)), **arrays)
    os.replace(tmp, path)


def subsample_rows(n, share, seed=0):
    """The subset PLREC fitted on (ml_reconstruct.data.subsample_rows)"""
    n_want = max(1, int(round(share * n)))
    if n_want >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=n_want, replace=False))


def flatten_model(pf, sc, fits):
    """
    Flatten one ``(PolynomialFeatures, StandardScaler, [(omp, sel), ...])`` fit.

    Returns ``(terms, weights, offset)``: ``terms`` is ``(n_terms, n_inputs)`` int8
    (the monomial exponents), ``weights`` ``(n_terms, n_targets)`` float, and
    ``offset`` ``(n_targets,)`` float. The targets are the 2*M phasor components,
    ``[sin_0 ... sin_{M-1}, cos_0 ... cos_{M-1}]``: ``fits`` holds one OMP per
    target.
    """
    n_inputs = pf.n_features_in_
    columns = np.unique(np.concatenate([sel for _, sel in fits]))
    position = {int(c): i for i, c in enumerate(columns)}
    terms = np.asarray(pf.powers_, dtype=np.int8)[columns]
    n_targets = len(fits)
    weights = np.zeros((len(columns), n_targets), dtype=np.float64)
    offset = np.zeros(n_targets, dtype=np.float64)
    for j, (omp, sel) in enumerate(fits):
        coef = np.zeros(len(columns))
        for c, value in zip(sel, np.ravel(omp.coef_)):
            coef[position[int(c)]] = value
        # fold the standardization of the monomials into the weights
        w = coef / sc.scale_
        weights[:, j] = w
        offset[j] = float(np.ravel([omp.intercept_])[0]) - float(np.dot(w, sc.mean_))
    return terms, weights, offset


def predict_standardised(model, x):
    """
    The exported evaluation for one wavelength, from a vector of *standardised*
    signals ``x`` (i.e. ``(signals - input_mean) / input_std``): the same
    computation ``LanternCubicModel.predict_phasors`` does.
    """
    terms, weights, offset = model
    monomials = np.prod(x[None, :] ** terms, axis=1)
    return monomials @ weights + offset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='spectral_41wl.npz the models were fitted on')
    ap.add_argument('--cache', required=True, help='PLREC cache directory (cache_perlambda)')
    ap.add_argument('--tag', default='wrap1-d3-K1282-f19-n8000',
                    help='fit tag, as used by fit_wrapped_models ("<tag>-w<index>.pkl")')
    ap.add_argument('--wavelengths', required=True, help='START:STOP[:STEP] in nm, or a comma list')
    ap.add_argument('--span', type=float, default=2000.0,
                    help='capture range of the unwrap, +-nm (default 2000 for ANDES)')
    ap.add_argument('--period-factor', type=float, default=1.0,
                    help='wrap period as a fraction of the wavelength (1.0 in the reference fits)')
    ap.add_argument('--fit-share', type=float, default=0.2,
                    help='fraction of the training rows the models were fitted on (for sigma)')
    ap.add_argument('--fit-seed', type=int, default=0, help='seed of that subsample')
    ap.add_argument('--conf-min', type=float, default=0.15)
    ap.add_argument('--huber-c', type=float, default=2.0)
    ap.add_argument('--n-irls', type=int, default=12)
    ap.add_argument('--pl-name', default='', help='lantern name, for the file header')
    ap.add_argument('--out', required=True, help='output .npz file')
    ap.add_argument('--check', type=int, default=200,
                    help='rows of the self-check against the pickled pipeline (0 to skip)')
    ap.add_argument('--seed', type=int, default=0, help='seed of the self-check rows')
    args = ap.parse_args()

    ds = np.load(args.dataset)
    signals = ds['signals']
    if signals.ndim != 3:
        raise SystemExit(f'{args.dataset}: signals must be (n, n_wl, n_cores), not {signals.shape}')
    n_total, n_wl, n_cores = signals.shape
    n_train = int(ds['n_train'])
    ds_wl = np.asarray(ds['wavelengths_nm'], dtype=float)

    # the input standardisation: per (wavelength, core), training rows only
    mean_flat = signals[:n_train].reshape(n_train, -1).mean(axis=0)
    std_flat = signals[:n_train].reshape(n_train, -1).std(axis=0) + 1e-6
    input_mean = mean_flat.reshape(n_wl, n_cores)
    input_std = std_flat.reshape(n_wl, n_cores)
    X3 = ((signals - input_mean) / input_std).reshape(n_total, n_wl, n_cores)

    rows = subsample_rows(n_train, args.fit_share, args.fit_seed)
    wavelengths = parse_wavelengths(args.wavelengths)
    rng = np.random.default_rng(args.seed)
    check_rows = rng.choice(n_total, size=min(args.check, n_total), replace=False) \
        if args.check > 0 else np.zeros(0, int)

    out_wl, out_w, out_period, out_n_terms, out_powers, out_weights, out_offsets, out_sigma = \
        [], [], [], [], [], [], [], []
    worst = 0.0
    for wl in wavelengths:
        candidates = np.where(np.abs(ds_wl - wl) < 0.5)[0]
        if len(candidates) != 1:
            raise SystemExit(f'{wl} nm is not a wavelength of {args.dataset} '
                             f'(it has {ds_wl[0]:.0f}-{ds_wl[-1]:.0f} nm)')
        w = int(candidates[0])
        path = os.path.join(args.cache, f'{args.tag}-w{w}.pkl')
        if not os.path.exists(path):
            raise SystemExit(f'no cached fit {path}')
        with open(path, 'rb') as f:
            pf, sc, fits = pickle.load(f)

        terms, weights, offset = flatten_model(pf, sc, fits)
        if pf.n_features_in_ != n_cores:
            raise SystemExit(f'{path}: fitted on {pf.n_features_in_} inputs, the dataset has {n_cores}')

        # wrapped-piston RMSE on the fitted rows, as PLREC's unwrap weights expect
        period = args.period_factor * wl
        n_out = len(fits) // 2
        X = X3[rows][:, w]
        pred = np.stack([predict_standardised((terms, weights, offset), x) for x in X], axis=0)
        sig = np.zeros(n_out)
        pistons_true = np.asarray(ds['coeff'], float)[rows]
        for j in range(n_out):
            g = np.arctan2(pred[:, j], pred[:, n_out + j]) * period / (2 * np.pi)
            d = g - pistons_true[:, j]
            d = d - period * np.round(d / period)
            sig[j] = max(float(np.sqrt(np.mean(d ** 2))), 1e-3)

        if len(check_rows):
            # compare with the pickled scikit-learn pipeline on random dataset rows
            Z = sc.transform(pf.transform(X3[check_rows][:, w]))
            ref = np.stack([omp.predict(Z[:, sel]) for omp, sel in fits], axis=1)
            mine = np.stack([predict_standardised((terms, weights, offset), x)
                             for x in X3[check_rows][:, w]], axis=0)
            worst = max(worst, float(np.max(np.abs(ref - mine))))

        out_wl.append(wl)
        out_w.append(w)
        out_period.append(period)
        out_n_terms.append(terms.shape[0])
        out_powers.append(terms)
        out_weights.append(weights)
        out_offsets.append(offset)
        out_sigma.append(sig)
        print(f'  w{w:02d} {wl:7.1f} nm: {terms.shape[0]:4d} terms, '
              f'sigma {np.mean(sig):.4f} nm', flush=True)

    powers = np.concatenate(out_powers, axis=0)
    weights = np.concatenate(out_weights, axis=0)
    header = {
        'VERSION': 1,
        'OBJ_TYPE': 'LanternCubicModel',
        'PLNAME': str(args.pl_name),
        'NWL': len(out_wl),
        'NINPUTS': int(n_cores),
        'NOUTPUTS': len(out_sigma[0]),
        'TOTTERMS': int(powers.shape[0]),
        'SPAN': float(args.span),
        'CONFMIN': float(args.conf_min),
        'HUBERC': float(args.huber_c),
        'NIRLS': int(args.n_irls),
    }
    arrays = {
        'wavelengths_nm': np.asarray(out_wl, dtype=np.float64),
        'periods_nm': np.asarray(out_period, dtype=np.float64),
        'n_terms': np.asarray(out_n_terms, dtype=np.int32),
        'powers': powers.astype(np.int8),
        'weights': weights.astype(np.float32),
        'offsets': np.asarray(out_offsets, dtype=np.float32),
        'input_mean': input_mean[out_w].astype(np.float32),
        'input_std': input_std[out_w].astype(np.float32),
        'sigma_nm': np.asarray(out_sigma, dtype=np.float32),
    }
    write_npz(args.out, header, arrays)
    print(f'\nwrote {args.out}  ({os.path.getsize(args.out) / 1e6:.1f} MB)'
          f'\n  {len(out_wl)} wavelengths x {n_cores} inputs -> {arrays["weights"].shape[1] // 2} pistons'
          f'\n  self-check vs the pickled pipeline: max |dphasor| {worst:.2e} over {len(check_rows)} rows')
    if len(check_rows) and worst > 1e-5:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
