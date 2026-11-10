#!/usr/bin/env python
"""
Build the closed-form lantern model from the *characterization* alone.

No dataset and no fit are involved. With piston-only aberrations the field at every
output core is linear in the six petal phasors, so the constants of

    N_c(p) = a_c + sum_{j<k} [A_cjk cos D_jk + B_cjk sin D_jk]

are exactly

    a_c = sum_j |h_cj|**2,   A_cjk = 2 Re(h_cj conj(h_ck)),   B_cjk = -2 Im(h_cj conj(h_ck))

where h_cj is the complex response of core c to the field of petal j alone. That
response is what the characterization already contains: the focal field of a
unit-amplitude petal propagates to the sampling mesh, the projector ``P`` turns it
into mode amplitudes, and the channel matrix ``A`` into core amplitudes. The same
computation is what PLREC's derivation.petal_channel_amplitudes performs from the
cbeam characterization.

So the model needs, per wavelength, only the P and A matrices that
export_pl_characterization.py writes; the petal geometry comes from the DM
influence functions, and the sampling mesh from the same DispersedLanternSensor
the simulation uses, which keeps the two consistent by construction.

    python build_closed_form_from_char.py \\
        --char calib/pl_char/pl19_1200_1100_1300_10nm.npz \\
        --ifunc calib/ifunc/ANDES_400pix_all_modes.fits \\
        --twin-file calib/ifunc/petals_twin.npy \\
        --out calib/pl_rec/pl19_1200_closed_form_char_1100_1300_10nm.npz

Self-check: with --fitted <the model exported from the fitted constants> it compares
the two in natural units, after undoing the fitted model's signal standardisation.
"""
import argparse
import json
import os
import sys

import numpy as np

# Use the checkout this script lives in
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import specula
specula.init(0)

from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.ifunc import IFunc
from specula.data_objects.lantern_characterization import LanternCharacterization
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.dispersed_lantern_sensor import DispersedLanternSensor


def petal_indicators(ifunc, n_pistons=5, tol=1e-6):
    """
    (n_pistons + 1, npix) indicators of the petal of each piston mode and of the
    reference petal (the pupil pixels none of the modes moves), in the mask order.
    """
    mask = np.asarray(cpuArray(ifunc.mask_inf_func)) > 0
    modes = np.asarray(cpuArray(ifunc.influence_function))[:n_pistons]
    label = np.full(modes.shape[1], n_pistons)
    for k in range(n_pistons):
        label[np.abs(modes[k]) > tol * np.abs(modes[k]).max()] = k
    return np.stack([label == k for k in range(n_pistons + 1)]).astype(np.float32), mask


def core_amplitudes(char, mask, indicators, simul_params, pixel_scale_um, pad_factor=4):
    """
    h[w, c, j]: complex response of core c to petal j at wavelength w.

    The petal fields go through the same pupil -> focal plane -> mesh -> projector
    path as the simulation: DispersedLanternSensor with normalize_amplitude off.
    ``mask`` is the (ny, nx) pupil and ``indicators`` the (n_petals, npix)
    indicators in mask order.
    """
    sensor = DispersedLanternSensor(simul_params, char, pad_factor=pad_factor,
                                    normalize_amplitude=False, target_device_idx=-1)
    rows, cols = np.nonzero(mask)
    coverage = np.any(indicators > 0, axis=0)
    pupil = np.zeros(mask.shape, np.float32)
    pupil[rows[coverage], cols[coverage]] = 1
    # the first field fixes the sensor's amplitude mask: give it the whole pupil
    ef = ElectricField(mask.shape[0], mask.shape[1], pixel_scale_um * 1e-6, target_device_idx=-1)
    ef.A[:] = pupil
    ef.generation_time = 1
    sensor.inputs['in_ef'].set(ef)
    sensor.setup()
    sensor.check_ready(1)
    sensor.trigger()
    sensor.post_trigger()

    n_petals = indicators.shape[0]
    channels = np.asarray(cpuArray(char.channel_matrix), dtype=np.complex128)   # (n_wl, n_ports, n_modes)
    out = np.zeros((char.n_wl, char.channel_matrix.shape[1], n_petals), np.complex128)
    for j in range(n_petals):
        ef = ElectricField(mask.shape[0], mask.shape[1], pixel_scale_um * 1e-6, target_device_idx=-1)
        ef.A[:] = 0
        sel = indicators[j] > 0
        ef.A[rows[sel], cols[sel]] = 1.0
        ef.generation_time = 2 + j
        sensor.inputs['in_ef'].set(ef)
        sensor.check_ready(2 + j)
        sensor.trigger()
        sensor.post_trigger()
        modes = np.asarray(cpuArray(sensor.outputs['out_mode_amplitudes'].value))
        out[:, :, j] = np.einsum('wcm,wm->wc', channels, modes)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--char', required=True, help='LanternCharacterization npz')
    ap.add_argument('--ifunc', required=True, help='DM influence functions (the petal modes)')
    ap.add_argument('--twin-file', default=None, help='npy with the twin matrix')
    ap.add_argument('--span', type=float, default=4800.0)
    ap.add_argument('--pad-factor', type=int, default=4)
    ap.add_argument('--pl-name', default='pl19_1200')
    ap.add_argument('--fitted', default=None,
                    help='model exported from fitted constants, for the self-check')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    char = LanternCharacterization.restore(args.char, target_device_idx=-1)
    ifunc = IFunc.restore(args.ifunc, target_device_idx=-1)
    indicators, mask = petal_indicators(ifunc)
    print(f'{char.n_wl} wavelengths, {char.channel_matrix.shape[1]} cores, '
          f'{indicators.shape[0]} petals, pupil {int(mask.sum())} px')

    # pixel_scale_um is per wavelength in the characterization; the simulation
    # has one pupil sampling, so take the reference (central) value
    pixel_scale_um = float(np.asarray(cpuArray(char.pixel_scale_um)).ravel()[
        int(np.argmin(np.abs(np.asarray(cpuArray(char.wavelengths_nm)) - 1200.0)))])
    simul_params = SimulParams(pixel_pupil=mask.shape[0], pixel_pitch=pixel_scale_um / 1000.0,
                               root_dir='.', total_time=1.0, time_step=0.1)
    h = core_amplitudes(char, mask, indicators, simul_params, pixel_scale_um,
                        pad_factor=args.pad_factor)
    n_petals = indicators.shape[0]
    pairs = [(j, k) for j in range(n_petals) for k in range(j + 1, n_petals)]
    n_pairs = len(pairs)

    n_wl, n_cores = h.shape[0], h.shape[1]
    a = np.sum(np.abs(h) ** 2, axis=2).astype(np.float32)
    w = np.zeros((n_wl, n_cores, 2 * n_pairs), np.float32)
    for p, (j, k) in enumerate(pairs):
        prod = h[:, :, j] * np.conj(h[:, :, k])
        w[:, :, p] = 2 * prod.real
        w[:, :, n_pairs + p] = -2 * prod.imag
    q = np.zeros((n_wl, 2 * n_pairs), np.float32)
    twin = np.load(args.twin_file).astype(np.float32) if args.twin_file else None

    # The inversion is a non-convex search, so the *scale* of the signals matters
    # numerically even though the model is linear in them. Standardise per core
    # with the model's own scale, the mean channel power at zero piston (the
    # diagonal of a), which needs no data: this is what the dataset standardisation
    # of the fitted model approximates.
    scale = np.mean(a, axis=0).astype(np.float32)                  # (n_cores,)
    a = (a / scale[None, :]).astype(np.float32)
    w = (w / scale[None, :, None]).astype(np.float32)

    header = dict(VERSION=1, OBJ_TYPE='LanternClosedFormModel', PLNAME=args.pl_name,
                  NWL=n_wl, NINPUTS=n_cores, NOUTPUTS=n_petals - 1, SPAN=float(args.span),
                  CONFMIN=0.15, HUBERC=2.0, NIRLS=12, NGRID=32, STEPS=5, NPROJ=5,
                  SECOND=1, REG=1e-6,
                  COMMENT='closed form derived from the characterization (no fit)')
    arrays = dict(wavelengths_nm=np.asarray(cpuArray(char.wavelengths_nm), np.float32),
                  periods_nm=np.asarray(cpuArray(char.wavelengths_nm), np.float32),
                  cf_a=a, cf_w=w, cf_q=q,
                  input_mean=np.zeros((n_wl, n_cores), np.float32),
                  input_std=np.tile(scale[None, :], (n_wl, 1)).astype(np.float32),
                  sigma_nm=np.full((n_wl, n_petals - 1), 1e-3, np.float32))
    if twin is not None:
        arrays['twin'] = twin
    cards = [[key, value, ''] for key, value in header.items()]
    with open(out_path, 'wb') as f:
        np.savez(f, __header__=np.array(json.dumps(cards)), **arrays)
    print(f'wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.2f} MB)')
    print(f'   {n_wl} wavelengths x {n_cores} inputs -> {n_petals - 1} pistons')
    print(f'   mean |a| {np.abs(a).mean():.3e}, mean |W| {np.abs(w).mean():.3e}')

    if args.fitted:
        from specula.data_objects.lantern_closed_form_model import LanternClosedFormModel
        model = LanternClosedFormModel.restore(args.fitted, target_device_idx=-1)
        mean = np.asarray(cpuArray(model.input_mean))
        std = np.asarray(cpuArray(model.input_std))
        # compare in natural units: each model's own standardisation undone
        a_char_nat = a * scale[None, :]
        w_char_nat = w * scale[None, :, None]
        a_fit = np.asarray(cpuArray(model.cf_a)) * std + mean
        w_fit = np.asarray(cpuArray(model.cf_w)) * std[:, :, None]
        print(f'   self-check against the fitted model: '
              f'|a_char - a_fit| / |a_fit| = '
              f'{np.abs(a_char_nat - a_fit).max() / np.abs(a_fit).max():.2e}, '
              f'|W_char - W_fit| / |W_fit| = '
              f'{np.abs(w_char_nat - w_fit).max() / np.abs(w_fit).max():.2e}')


if __name__ == '__main__':
    main()
