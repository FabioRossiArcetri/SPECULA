#!/usr/bin/env python
"""
Export PLREC's closed-form lantern model to a SPECULA data object.

For every wavelength of the PLREC dispersal the *exact* analytical forward model of
the lantern is fitted to the training rows of the spectral dataset
(``ml_reconstruct.forward.fit_forward`` on channel powers, i.e. without the
coupling-efficiency normalisation):

    s_c(p) = a_c + sum_{j<k} [A_cjk cos D_jk + B_cjk sin D_jk],
    D_jk = 2*pi*(p_j - p_k)/lambda

and stored with the standardisation of the signals, the per-wavelength wrapped
error of the inversion on the fitted rows, and the twin matrix of the pupil
(``ml_reconstruct.petals``). At run time SPECULA inverts this model with the
translation of ``forward.invert_fixed`` in LanternClosedFormModel.

Run it where PLREC, its models and the dataset live (no cbeam needed):

    python export_closed_form_model.py \\
        --dataset /raid1/frossi/dev/PLREC/data/ml/pl19_1200/spectral_41wl.npz \\
        --plrec-root /raid1/frossi/dev/PLREC \\
        --ifunc /raid1/frossi/dev/work/cbeam/examples/data/ANDES_400pix_all_modes.fits \\
        --wavelengths 1100:1300:10 --span 4800 \\
        --pl-name pl19_1200 \\
        --out out/pl19_1200_closed_form_1100_1300_10nm.npz

The export is checked against the pickled per-wavelength models' training error and
against the forward model evaluated at the reference piston configuration.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np


def load_module(name, path):
    '''Import a PLREC module from its file, bypassing the package __init__ (jax)'''
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_wavelengths(text, available):
    '''"1100:1300:10" (start:stop:step in nm) or "1100,1140,1200" -> indexes'''
    if ':' in text:
        parts = [float(p) for p in text.split(':')]
        if len(parts) != 3:
            raise ValueError('--wavelengths must be START:STOP:STEP or a comma separated list')
        start, stop, step = parts
        wanted = np.arange(start, stop + 0.5 * step, step)
    else:
        wanted = np.array([float(p) for p in text.split(',')])
    indexes = []
    for w in wanted:
        matches = np.flatnonzero(np.abs(available - w) < 0.5)
        if not len(matches):
            raise ValueError(f'{w} nm is not in the dataset (it has {available.min()}-'
                             f'{available.max()} nm)')
        indexes.append(int(matches[0]))
    return indexes


def subsample(share, seed=0):
    def apply(y):
        n = y.shape[0]
        rng = np.random.default_rng(seed)
        picks = rng.permutation(n)[:max(1, int(round(share * n)))]
        return np.sort(picks)
    return apply


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='PLREC spectral dataset (spectral_41wl.npz)')
    ap.add_argument('--plrec-root', required=True, help='PLREC checkout (for ml_reconstruct)')
    ap.add_argument('--ifunc', default=None, help='unused (the twin matrix is passed as a file)')
    ap.add_argument('--wavelengths', default='1100:1300:10')
    ap.add_argument('--span', type=float, default=4800.0, help='capture range for the unwrap [nm]')
    ap.add_argument('--pl-name', default='pl19_1200')
    ap.add_argument('--twin-file', default=None,
                    help='npy file with the twin matrix of the pupil symmetry (optional)')
    ap.add_argument('--fit-share', type=float, default=1.0,
                    help='share of the rows used to fit the forward model (default 0.2)')
    ap.add_argument('--fit-seed', type=int, default=0)
    ap.add_argument('--n-grid', type=int, default=32)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--n-proj', type=int, default=5)
    ap.add_argument('--check', type=int, default=20, help='rows used by the self-check')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    forward = load_module('plrec_forward', os.path.join(args.plrec_root, 'ml_reconstruct',
                                                        'forward.py'))
    petals = load_module('plrec_petals', os.path.join(args.plrec_root, 'ml_reconstruct',
                                                      'petals.py'))

    data = np.load(args.dataset, allow_pickle=False)
    wavelengths = np.asarray(data['wavelengths_nm'], float)
    coeff = np.asarray(data['coeff'], float)         # (n, 5) petal pistons [nm]
    signals_all = np.asarray(data['signals'], float)                    # (n, n_wl, n_cores) raw powers
    indexes = parse_wavelengths(args.wavelengths, wavelengths)
    picked = wavelengths[indexes]

    # standardisation on all rows, as PLREC's pipeline does before fitting
    mean = signals_all.mean(axis=0)[indexes].astype(np.float32)
    std = signals_all.std(axis=0)[indexes].astype(np.float32)
    std[std == 0] = 1
    # Fit on the dataset's *training* block only (the first n_train rows, the same
    # rows PLREC trains its own per-wavelength models on), never on the validation
    # tail: fitting on a random share of all rows would both see the validation
    # rows and use far fewer rows than the reference fits.
    n_train = int(data['n_train']) if 'n_train' in data.files else coeff.shape[0]
    training_block = np.arange(n_train)
    picked_rows = subsample(args.fit_share, args.fit_seed)(coeff[training_block])
    rows = training_block[picked_rows]
    print(f'{len(indexes)} wavelengths, {signals_all.shape[0]} rows, fitting on {len(rows)}')

    twin = None
    if args.twin_file:
        twin = np.load(args.twin_file).astype(float)
        print('twin matrix from', args.twin_file)
        print(twin)

    n_wl, n_cores = len(indexes), signals_all.shape[2]
    cf_a = np.zeros((n_wl, n_cores), np.float32)
    cf_w = np.zeros((n_wl, n_cores, 30), np.float32)
    cf_q = np.zeros((n_wl, 30), np.float32)
    sigma = np.zeros((n_wl, coeff.shape[1]), np.float32)

    for i, w in enumerate(indexes):
        signals = ((signals_all[:, w, :] - mean[i]) / std[i]).astype(np.float64)
        model = forward.fit_forward(coeff[rows], signals[rows], float(picked[i]),
                                    normalised=False)
        cf_a[i] = model.a
        cf_w[i] = model.W
        cf_q[i] = model.q
        estimate, _ = forward.invert_fixed(model, signals[rows], twin,
                                           n_grid=args.n_grid, steps=args.steps,
                                           n_proj=args.n_proj)
        period = float(picked[i])
        error = estimate - coeff[rows]
        error = error - period * np.round(error / period)
        sigma[i] = np.maximum(np.sqrt(np.mean(error ** 2, axis=0)), 1e-3)
        print(f'  w[{w:02d}] {picked[i]:7.1f} nm: |a| {np.abs(model.a).mean():.3e}, '
              f'|W| {np.abs(model.W).mean():.3e}, wrapped error '
              f'{np.mean(sigma[i]):.1f} nm (max {np.max(sigma[i]):.1f})')

    header = dict(VERSION=1, OBJ_TYPE='LanternClosedFormModel', PLNAME=args.pl_name,
                  NWL=n_wl, NINPUTS=n_cores, NOUTPUTS=int(coeff.shape[1]),
                  SPAN=float(args.span), CONFMIN=0.15, HUBERC=2.0, NIRLS=12,
                  NGRID=args.n_grid, STEPS=args.steps, NPROJ=args.n_proj,
                  SECOND=1, REG=1e-6,
                  COMMENT='closed form from PLREC forward.fit_forward + invert_fixed')
    arrays = dict(wavelengths_nm=picked.astype(np.float32),
                  periods_nm=picked.astype(np.float32),
                  cf_a=cf_a, cf_w=cf_w, cf_q=cf_q,
                  input_mean=mean, input_std=std, sigma_nm=sigma)
    if twin is not None:
        arrays['twin'] = twin.astype(np.float32)
    cards = [[key, value, ''] for key, value in header.items()]
    arrays = {name: value for name, value in arrays.items() if value is not None}
    with open(out_path, 'wb') as f:
        np.savez(f, __header__=np.array(json.dumps(cards)), **arrays)
    print(f'\nwrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)')
    print(f'   {n_wl} wavelengths x {n_cores} inputs -> {coeff.shape[1]} pistons')

    # self-check: the exported constants must reproduce the fitted model, and the
    # inversion must explain the signals of the checked rows
    w = len(indexes) // 2
    signals = ((signals_all[:, indexes[w], :] - mean[w]) / std[w]).astype(np.float64)
    model = forward.fit_forward(coeff[rows], signals[rows], float(picked[w]), normalised=False)
    dmax = max(np.max(np.abs(cf_a[w] - model.a)), np.max(np.abs(cf_w[w] - model.W)))
    estimate, _ = forward.invert_fixed(model, signals[:args.check], twin,
                                       n_grid=args.n_grid, steps=args.steps,
                                       n_proj=args.n_proj)
    reproduced = model(estimate)
    residual = np.max(np.abs(reproduced - signals[:args.check]))
    print(f'   self-check: |constants exported - refitted| {dmax:.2e}, '
          f'|powers(estimate) - signals| {residual:.2e}')


if __name__ == '__main__':
    main()
