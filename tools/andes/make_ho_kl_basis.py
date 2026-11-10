#!/usr/bin/env python
"""KL modal basis of the ANDES-PL high-order DM (the zonal andes_ho79 of make_high_order_dm.py),
as a modes-to-commands matrix for the DM's ``m2c_object``.

The pyramid loop of andes_pl_piston_pyr.yml must control global modes: with the zonal IFunc
alone, ``nmodes: N`` selects the first N actuators -- a strip of the pupil -- so the loop cannot
correct the rest of it. This basis is built as the ANDES one is
(``ANDES_400pix_79nacts_cir__5zern_10000.0cn``): KL modes of the DM for a von Karman spectrum, the
first five replaced by Zernikes (tip, tilt, focus, astigmatism), the influence-function covariance
truncated at a condition number of 1e4 (make_modal_base_from_ifs_fft). The DM's influence
functions are orthogonal to the five petal pistons, so the basis is too.

    python tools/andes/make_ho_kl_basis.py --ifunc calib/ifunc/andes_ho79.npz \\
        --out calib/m2c/andes_ho79_kl.fits [--diameter 39.0 --r0 0.126 --L0 50]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import specula
specula.init(-1)

from specula import cpuArray
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ifunc', required=True)
    ap.add_argument('--out', required=True, help='output M2C (.fits)')
    ap.add_argument('--diameter', type=float, default=39.0, help='[m] 400 px x 0.0975 m')
    ap.add_argument('--r0', type=float, default=0.126, help='[m] at 500 nm: 0.8" seeing')
    ap.add_argument('--L0', type=float, default=50.0, help='[m] outer scale, as the atmosphere')
    ap.add_argument('--zern', type=int, default=5)
    ap.add_argument('--cond', type=float, default=1e4)
    ap.add_argument('--oversampling', type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    ifunc = IFunc.restore(args.ifunc, target_device_idx=-1)
    mask = np.asarray(cpuArray(ifunc.mask_inf_func)) > 0
    IF = np.asarray(cpuArray(ifunc.influence_function), dtype=np.float32)
    if IF.shape[0] == mask.sum():
        IF = IF.T
    print(f'IFunc: {IF.shape[0]} actuators x {IF.shape[1]} pupil points ({time.time() - t0:.0f} s)', flush=True)

    kl, m2c, sv = make_modal_base_from_ifs_fft(mask, args.diameter, IF, args.r0, args.L0,
                                               zern_modes=args.zern, oversampling=args.oversampling,
                                               if_max_condition_number=args.cond, xp=np, dtype=np.float32)
    kl, m2c = np.asarray(kl), np.asarray(m2c)
    print(f'basis: {kl.shape[0]} modes, m2c {m2c.shape} ({time.time() - t0:.0f} s)', flush=True)

    # checks: the commands reproduce the modes, the modes are orthogonal, their rms
    rec = m2c.T @ IF                                   # (nmodes, npix)
    err = np.linalg.norm(rec - kl, axis=1) / np.linalg.norm(kl, axis=1)
    rms = np.sqrt(np.mean(kl ** 2, axis=1))
    G = (kl[:200] @ kl[:200].T) / kl.shape[1]
    off = np.abs(G - np.diag(np.diag(G))).max() / np.abs(np.diag(G)).max()
    print(f'm2c.T @ IF vs modes: max rel err {err.max():.2e}; mode rms {rms.min():.3f}-{rms.max():.3f}; '
          f'orthogonality (first 200) {off:.1e}')
    print(f'mean over the pupil, max |.| / rms: {np.abs(kl.mean(1)).max() / rms.mean():.1e}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    M2C(m2c=m2c.astype(np.float32), target_device_idx=-1).save(args.out, overwrite=True)
    np.save(os.path.splitext(args.out)[0] + '_rms.npy', rms)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB) in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
