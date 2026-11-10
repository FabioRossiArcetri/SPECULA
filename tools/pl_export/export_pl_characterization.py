#!/usr/bin/env python
"""Export a SPECULA ``LanternCharacterization`` file from cbeam characterizations.

The characterization of a photonic lantern at one wavelength is a
``ChainPropagator`` (cbeam): the FEM eigenmodes of the lantern input face, the
coupling matrices along the taper, and the output channel basis. The whole chain
from the field on the input face to the complex amplitude of every output
channel is linear, so it reduces to two matrices, which is what SPECULA needs:

* ``mode_projector`` ``(n_modes, n_mesh)``: ``g = P @ E_mesh`` projects the field
  sampled at the mesh nodes onto the lantern's LP modes (rows ``conj(v_k)^T B``,
  cbeam's ``ModalProjector``);
* ``channel_matrix`` ``(n_ports, n_modes)``: ``c = A @ g``, the propagation
  through the lantern in the channel basis, divided by the incident-power
  normalization, so that ``|c|**2`` is the fraction of the incident power in each
  channel.

This script writes both for every requested wavelength, in the npz layout of
``iteras.lib.storage`` (so ``LanternCharacterization.restore`` reads it) and with the
real/imaginary encoding FITS would need as well:

    python export_pl_characterization.py --lantern pl19_1200 \\
        --wavelengths 1150:1250:5 \\
        --cbeam-examples /raid1/frossi/dev/work/cbeam/examples \\
        --out pl_char/pl19_1200.npz

A wavelength whose characterization is not already cached is *reported*, never
solved: characterizing one wavelength takes ~13 minutes and gigabytes of RAM, and
this tool must be side-effect free on the cbeam cache.

``--check N`` (default 5) is a self-check against cbeam itself, and it is the
reason the exported numbers can be trusted: for random complex modal vectors it
compares ``A @ g`` with ``propagate()`` + ``to_channel_basis()``, and for random
mesh fields it compares ``P @ E_mesh`` with ``ModalProjector.project_batch()``.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np


def parse_wavelengths(text):
    """'1150:1250:5' or '1150,1160,1170' -> list of floats"""
    if ':' in text:
        parts = [p for p in text.split(':') if p.strip()]
        if len(parts) not in (2, 3):
            raise ValueError(f'cannot parse wavelengths {text!r}: use START:STOP[:STEP]')
        start, stop = float(parts[0]), float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 5.0
        if step <= 0:
            raise ValueError('the wavelength step must be positive')
        n = int(np.round((stop - start) / step)) + 1
        values = start + step * np.arange(n)
        if abs(values[-1] - stop) > 1e-6:
            values = np.append(values, stop)
        return [float(v) for v in values]
    return [float(v) for v in text.replace(';', ',').split(',') if v.strip()]


CACHE_SUBDIRS = ('eigenvalues', 'eigenmodes', 'zvals')


def check_cache_present(tag, data_dir='data'):
    """Raise with the missing files if the characterization of *tag* is not cached"""
    missing = []
    for segment in ('front', 'back'):
        for sub in CACHE_SUBDIRS:
            path = os.path.join(data_dir, sub, f'{sub}_{tag}_{segment}.npy')
            if not os.path.exists(path):
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            'no cached characterization for tag %r; missing for example %s. '
            'This tool never solves a characterization: run cbeam/characterize.py '
            'for this wavelength first.' % (tag, missing[0]))


def to_real_imag(array):
    """Complex array -> real array with a trailing dimension of 2 (re, im)"""
    return np.stack((np.real(array), np.imag(array)), axis=-1).astype(np.float32)


def write_npz(path, header, arrays):
    """Write the npz layout of iteras.lib.storage (JSON header + named arrays)"""
    cards = [[key, value, ''] for key, value in header.items()]
    arrays = {name: value for name, value in arrays.items() if value is not None}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.savez(f, __header__=np.array(json.dumps(cards)), **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lantern', required=True,
                    help='lantern preset (pl19_1200, pl19_800, pl7_1200)')
    ap.add_argument('--wavelengths', required=True,
                    help='START:STOP[:STEP] in nm, or a comma-separated list')
    ap.add_argument('--cbeam-examples', required=True,
                    help='path to the cbeam checkout "examples" directory, the one '
                         'whose data/ holds the characterizations')
    ap.add_argument('--out', required=True, help='output .npz file')
    ap.add_argument('--check', type=int, default=5,
                    help='number of random vectors of the self-check against cbeam (0 to skip)')
    ap.add_argument('--seed', type=int, default=0, help='seed of the self-check vectors')
    args = ap.parse_args()

    examples = os.path.abspath(args.cbeam_examples)
    cbeam_root = os.path.dirname(examples)
    if not os.path.isdir(os.path.join(examples, 'data')):
        raise SystemExit(f'{examples}/data does not exist: is --cbeam-examples right?')
    sys.path.insert(0, os.path.join(cbeam_root, 'src'))
    sys.path.insert(0, examples)

    # The output path is resolved before changing directory: the cbeam caches are
    # found relative to the working directory, which this script must set.
    out_path = os.path.abspath(args.out)

    # The cbeam caches are found relative to the working directory
    os.chdir(examples)

    import batch_pipeline as bp
    from batch_pipeline.engine import characterization_tag
    from multi_wvl.construct import build_and_characterize_lantern_at_wavelength

    wavelengths = parse_wavelengths(args.wavelengths)
    base = bp.get_simulation_parameters(lantern=args.lantern)
    ifunc = bp.load_ifunc(base['ifunc_file'])
    bp.center_ifunc_mask(ifunc)
    mask = np.asarray(ifunc.mask_inf_func) > 0
    n_ports = int(base['n_output_positions'])
    n_modes = n_ports + 1
    rng = np.random.default_rng(args.seed)

    print(f'lantern {args.lantern}: {n_ports} ports, {n_modes} modes, '
          f'{len(wavelengths)} wavelengths {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm')

    # Fail early, with the full list, before doing any work
    missing = []
    for wl in wavelengths:
        try:
            check_cache_present(characterization_tag(base, wl))
        except FileNotFoundError as e:
            missing.append((wl, str(e)))
    if missing:
        print(f'\n{len(missing)} of {len(wavelengths)} characterizations are not cached:',
              file=sys.stderr)
        for wl, msg in missing:
            print(f'  {wl:.0f} nm: {msg}', file=sys.stderr)
        raise SystemExit(1)

    projectors, channels, pixel_scales, norms = [], [], [], []
    mesh_xy = None
    worst_a = worst_p = 0.0

    for i, wl in enumerate(wavelengths):
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                p_lambda, prop12 = build_and_characterize_lantern_at_wavelength(
                    base, wl, verbose=False)
                wvg = bp.get_waveguide_properties(prop12, mesh_z=0)
                projector = bp.ModalProjector(wvg)
                fg = bp.IncidentFieldGenerator(p_lambda, ifunc)
                M = np.asarray(prop12.compute_transfer_matrix(channel_basis=True))
        except (FileNotFoundError, OSError) as e:
            raise SystemExit(f'{wl:.0f} nm: the characterization is not cached ({e})')

        points = np.asarray(wvg['points'], dtype=float)
        if mesh_xy is None:
            mesh_xy = points
        elif points.shape != mesh_xy.shape or not np.allclose(points, mesh_xy, atol=1e-9):
            raise SystemExit(f'{wl:.0f} nm: the mesh changed between wavelengths')

        focal_norm = float(fg.focal_plane_norm)
        P = np.asarray(projector.projection_matrix, dtype=np.complex128)
        A = M[:n_ports, :n_modes] / focal_norm

        # ---- self-check against cbeam itself -----------------------------
        err_a = err_p = 0.0
        coupling = 0.0
        if args.check > 0:
            for _ in range(args.check):
                g = rng.normal(size=n_modes) + 1j * rng.normal(size=n_modes)
                g /= np.linalg.norm(g)
                with contextlib.redirect_stdout(io.StringIO()):
                    uf = prop12.propagate(g)[2]
                c_ref = np.asarray(prop12.to_channel_basis(uf))[:n_ports] / focal_norm
                c = A @ g
                err_a = max(err_a, float(np.linalg.norm(c - c_ref) / max(np.linalg.norm(c_ref), 1e-30)))

                E = rng.normal(size=points.shape[0]) + 1j * rng.normal(size=points.shape[0])
                u0, eff = projector.project_batch(E[None, :])
                err_p = max(err_p, float(np.linalg.norm(P @ E - u0[0] * eff[0]) /
                                         max(np.linalg.norm(P @ E), 1e-30)))
                # power fraction of a random (unit-norm) field at the lantern face:
                # an order-of-magnitude check of the normalization, must be << 1
                coupling = max(coupling, float(np.sum(np.abs(A @ (P @ E)) ** 2)))
        worst_a, worst_p = max(worst_a, err_a), max(worst_p, err_p)

        projectors.append(P)
        channels.append(A)
        pixel_scales.append(float(p_lambda['pixel_scale_um']))
        norms.append(focal_norm)
        print(f'  [{i + 1:2d}/{len(wavelengths)}] {wl:7.1f} nm  '
              f'{time.time() - t0:5.1f} s  norm {focal_norm:.3e}  '
              f'coupling {coupling:.3f}  max |dA| {err_a:.2e}  max |dP| {err_p:.2e}',
              flush=True)

    header = {
        'VERSION': 1,
        'OBJ_TYPE': 'LanternCharacterization',
        'LANTERN': str(args.lantern),
        'NWL': len(wavelengths),
        'NPORTS': n_ports,
        'NMODES': n_modes,
        'NMESH': int(mesh_xy.shape[0]),
        'PADFACT': int(base['pad_factor']),
        'DESWL': float(base['wavelength_nm']),
        'PSFFILL': float(base['psf_fill_factor']),
    }
    arrays = {
        'wavelengths_nm': np.asarray(wavelengths, dtype=np.float64),
        'mesh_xy_um': np.asarray(mesh_xy, dtype=np.float64),
        'mode_projector_ri': to_real_imag(np.asarray(projectors)),
        'channel_matrix_ri': to_real_imag(np.asarray(channels)),
        'pixel_scale_um': np.asarray(pixel_scales, dtype=np.float64),
        'focal_plane_norm': np.asarray(norms, dtype=np.float64),
        'pupil_mask': mask.astype(np.uint8),
    }
    write_npz(out_path, header, arrays)

    size = os.path.getsize(out_path) / 1e6
    print(f'\nwrote {out_path}  ({size:.1f} MB)'
          f'\n  mode_projector {np.asarray(projectors).shape}, '
          f'channel_matrix {np.asarray(channels).shape}'
          f'\n  self-check vs cbeam: max |dA| {worst_a:.2e}, max |dP| {worst_p:.2e}')
    if args.check > 0 and max(worst_a, worst_p) > 1e-6:
        print('WARNING: the exported matrices differ from cbeam by more than 1e-6',
              file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
