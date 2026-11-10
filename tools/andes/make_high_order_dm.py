#!/usr/bin/env python
"""Generate the high-order DM and its reconstructor for the ANDES PL example.

The other ANDES configurations close the high orders with the M4 mirror (4000
modes, ``ifunc_m4``) and the pyramid/MMSE reconstructor. Their calibration files
are not part of this repository, so this script builds an equivalent stage on the
same pupil, with two deliberate simplifications, both documented in the example
configuration:

* the high-order mirror is a zonal DM: Gaussian influence functions on a regular
  grid of ``--n-act`` actuators across the pupil of the ANDES influence functions,
  masked to that pupil;
* its reconstructor is a *perfect* (noise-free, aliasing-free) modal analysis of
  the residual wavefront, instead of a pyramid with its calibrated MMSE matrix.

The influence functions are additionally made orthogonal to the five petal-piston
modes of the ANDES basis. The high-order loop is therefore blind to the petal
pistons by construction, which is what the ANDES design does with them in practice
(the pyramid loop is not meant to control them, and the piston sensors are): the
lantern keeps something to measure, and the two loops do not fight over the same
degrees of freedom.

Outputs a zonal ``IFunc`` and its ``IFuncInv``, ready for the ``ifunc_ref`` and
``ifunc_inv_ref`` entries of the example:

    python make_high_order_dm.py --n-act 40 \\
        --out-ifunc ../../calib/ifunc/andes_ho40.npz \\
        --out-inv   ../../calib/ifunc/andes_ho40_inv.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Use the checkout this script lives in, not whatever specula is installed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import specula
specula.init(0)

from specula import cpuArray
from specula.data_objects.ifunc import IFunc
from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--piston-ifunc', required=True,
                    help='IFunc with the ANDES modes: its pupil mask and its first 5 modes '
                         '(the petal pistons) are used')
    ap.add_argument('--n-pistons', type=int, default=5,
                    help='number of petal-piston modes to keep out of the high-order DM')
    ap.add_argument('--n-act', type=int, default=40, help='actuators across the pupil')
    ap.add_argument('--out-ifunc', required=True, help='output IFunc file')
    ap.add_argument('--out-inv', required=True, help='output IFuncInv file')
    args = ap.parse_args()

    piston_ifunc = IFunc.restore(args.piston_ifunc, target_device_idx=-1)
    mask = np.asarray(cpuArray(piston_ifunc.mask_inf_func)) > 0
    pistons = np.asarray(cpuArray(piston_ifunc.influence_function))[:args.n_pistons]
    npix = int(mask.sum())
    if pistons.shape[1] != npix:
        raise SystemExit(f'{args.piston_ifunc}: {pistons.shape[1]} points for a mask of {npix}')
    print(f'pupil: {npix} points, {args.n_act} actuators across')

    zonal, zonal_mask, _, _ = compute_zonal_ifunc(mask.shape[0], args.n_act,
                                                  geom='square', mask=mask.astype(float),
                                                  xp=np, dtype=np.float32)
    zonal = np.asarray(zonal, dtype=np.float32)
    # Drop the actuators that fall outside the pupil: their influence function is
    # (nearly) zero on the masked pixels, which would make the reconstructor
    # ill-conditioned and let the loop drive those null modes.
    rms = np.sqrt(np.mean(zonal ** 2, axis=1))
    keep = rms > 0.01 * rms.max()
    n_dropped = int(np.count_nonzero(~keep))
    zonal = zonal[keep]
    n_modes = zonal.shape[0]
    print(f'zonal DM: {n_modes} actuators inside the pupil'
          f' ({n_dropped} of the {n_modes + n_dropped} grid points dropped)')

    # Keep the petal pistons out of the high-order DM: remove, from every
    # influence function, its projection on the (orthonormalised) piston modes.
    q, _ = np.linalg.qr(pistons.T.astype(np.float64))     # (npix, n_pistons)
    projection = q @ (q.T @ zonal.T.astype(np.float64))   # (npix, n_modes)
    zonal = (zonal.T.astype(np.float64) - projection).T.astype(np.float32)
    print(f'  removed the {args.n_pistons} petal piston modes from its span')

    ifunc = IFunc(zonal, mask=mask.astype(float), nmodes=n_modes,
                  target_device_idx=-1, precision=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_ifunc)), exist_ok=True)
    ifunc.save(args.out_ifunc, overwrite=True)
    print(f'wrote {args.out_ifunc} ({os.path.getsize(args.out_ifunc) / 1e6:.0f} MB)')

    inv = ifunc.inverse(remove_piston=True)
    inv.save(args.out_inv, overwrite=True)
    print(f'wrote {args.out_inv} ({os.path.getsize(args.out_inv) / 1e6:.0f} MB), '
          f'{inv.nmodes()} modes')
    # what the example configuration needs for the integrator
    print(f'\nuse in the yaml:  n_modes: [{inv.nmodes()}]')


if __name__ == '__main__':
    main()
