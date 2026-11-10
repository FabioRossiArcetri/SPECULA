#!/usr/bin/env python
"""Measure the three configurations of the ANDES photonic-lantern example.

For each configuration prints, step by step and converged:
  total wavefront rms, the part carried by the five petal pistons, the AO residual
  (what is left after projecting them out), Strehl, the 19-channel throughput, the
  per-wavelength piston error of the cubic model, and the error of the UNWRAPPED
  piston against the modal reference.

Configurations: no piston injected, piston injected with the piston loop open,
piston injected with the piston loop closed. The reference for the piston is the
least-squares fit of the five petal piston modes to the propagated wavefront
(ModalAnalysis in the configuration), not the injected value: with the atmosphere
the petals carry their own pistons.

    python tools/andes/measure_scenarios.py --seeing 0.8 --steps 20
"""
import argparse
import os
import copy
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Use the checkout this script lives in, not whatever specula is installed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import specula
specula.init(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piston_gate                                         # PL_PISTON_GATE_STEP: close the piston loop late

CPU = os.environ.get('SPECULA_CPU', '1') == '1'           # SPECULA_CPU=0: run on the GPU
SUFFIX = os.environ.get('SPECULA_OUT_SUFFIX', '')           # appended to the output stores
from iteras.loop_control import LoopControl
from iteras import simul as simul_mod
from specula import cpuArray
from specula.data_objects.ifunc import IFunc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='config/ANDES/andes_pl_piston.yml')
    ap.add_argument('--piston-ifunc', default='calib/ifunc/ANDES_400pix_all_modes.fits',
                    help='IFunc whose first five modes are the petal pistons')
    ap.add_argument('--seeing', type=float, default=None, help='override the seeing [arcsec]')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--piston-gain', type=float, default=0.2, help='integrator gain when closed')
    ap.add_argument('--sigma-floor', type=float, default=None,
                    help='per-wavelength error assumed by the unwrap when the loop is closed [nm]')
    ap.add_argument('--model-class', default=None,
                    help='override the class of the pl_model entry (reconstructor strategy)')
    ap.add_argument('--model-tag', default=None,
                    help='override the tag of the pl_model entry')
    ap.add_argument('--scenario', choices=['all', 'open', 'closed'], default='all',
                    help="which scenarios to run (default all three)")
    ap.add_argument('--converged', type=int, default=10,
                    help='number of last steps averaged for the converged figures')
    ap.add_argument('--verbose', action='store_true', help='also print the step-by-step table')
    args = ap.parse_args()

    ifunc = IFunc.restore(args.piston_ifunc, target_device_idx=-1)
    mask = np.asarray(cpuArray(ifunc.mask_inf_func)) > 0
    pistons_basis = np.asarray(cpuArray(ifunc.influence_function), float)[:5]
    pistons_basis -= pistons_basis.mean(axis=1, keepdims=True)
    basis_inv = np.linalg.pinv(pistons_basis).T

    state = {'objs': {}, 'rows': []}
    orig_iter = LoopControl.iter

    def patched_iter(self):
        r = orig_iter(self)
        o = state['objs']
        if o:
            ef = o['prop'].outputs['out_wfs_source_ef']
            phase = np.asarray(cpuArray(ef.phaseInNm), float)[mask]
            res = np.asarray(cpuArray(o['modal_analysis'].outputs['out_modes'].value), float)
            wrapped = np.asarray(cpuArray(o['pl_rec'].outputs['out_pistons_wrapped'].value), float)
            lam = np.asarray(cpuArray(o['pl_model'].periods_nm), float)
            meas = np.asarray(cpuArray(o['pl_unwrap'].outputs['out_pistons'].value), float)
            err = wrapped - res[:, None]
            err = err - lam[None, :] * np.round(err / lam[None, :])
            pist = pistons_basis.T @ (basis_inv @ phase)
            state['rows'].append(dict(
                t=self.t_to_seconds(self.t), total=phase.std(), pist=pist.std(),
                ao=(phase - pist).std(),
                sr=float(cpuArray(o['psf'].outputs['out_sr'].value)),
                thr=float(np.asarray(cpuArray(o['pl_sensor'].outputs['out_throughput'].value)).mean()),
                per_lambda=float(np.abs(err).mean()),
                unwrap=float(np.abs(meas - res).mean()),
                res=res.copy(), meas=meas.copy()))
        return r

    LoopControl.iter = patched_iter
    orig_build = simul_mod.Simul.build_objects

    def patched_build(self, params):
        orig_build(self, params)
        state['objs'] = dict(self.objs)

    simul_mod.Simul.build_objects = patched_build

    base = yaml.safe_load(open(args.config))
    if args.seeing is not None:
        base['seeing']['constant'] = args.seeing
    if args.model_class is not None:
        base['pl_model']['class'] = args.model_class
    if args.model_tag is not None:
        base['pl_model']['tag'] = args.model_tag
    seeing = base['seeing']['constant']

    scenarios = [
        ('no ejection, piston loop open', [0.0] * 10, 0.0),
        ('ejection, piston loop open', [1600.0] + [0.0] * 9, 0.0),
        ('ejection, piston loop closed', [1600.0] + [0.0] * 9, args.piston_gain),
    ]
    if args.scenario == 'closed':
        scenarios = scenarios[2:]
    elif args.scenario == 'open':
        scenarios = scenarios[:2]
    for name, injection, gain in scenarios:
        p = copy.deepcopy(base)
        p['main']['total_time'] = args.steps * base['main']['time_step']
        p['disturbance_petals']['constant'] = list(injection)
        p['integrator']['int_gain'] = [gain]
        if args.sigma_floor is not None:
            p['pl_unwrap']['sigma_floor_nm'] = args.sigma_floor
        p['data_store']['store_dir'] = './output/measure_' + name.replace(' ', '_').replace(',', '') + SUFFIX
        p = piston_gate.maybe_apply(p, args.steps)
        yaml.safe_dump(p, open('ptmp_measure.yml', 'w'))
        state['rows'] = []
        state['objs'] = {}
        t0 = time.time()
        specula.main_simul(['ptmp_measure.yml'], cpu=CPU, no_speed_report=True, log_level='ERROR')
        rows = state['rows']
        print(f'\n=== seeing {seeing}" | {name} | {len(rows)} steps, {time.time() - t0:.0f} s ===')
        if args.verbose:
            print('  t[ms]  total  pistons  AO-res  Strehl   thru   per-lam  unwrapped  |  true petal0  meas petal0')
            for r in rows:
                print(f"  {r['t'] * 1e3:5.1f} {r['total']:6.0f} {r['pist']:7.0f} {r['ao']:7.1f} "
                      f"{r['sr']:7.4f} {r['thr']:6.3f} {r['per_lambda']:8.0f} {r['unwrap']:9.0f}  |"
                      f"  {r['res'][0]:9.0f} {r['meas'][0]:9.0f}")
        # Every figure below comes from the same window, so that it is one coherent
        # statistic: a single step of the loop is noisy (the Strehl swings by a
        # factor of a few from step to step as the piston commands settle).
        last = rows[-args.converged:]
        print(f"  converged (last {len(last)} of {len(rows)} steps), all figures from the same window:")
        print(f"    AO residual        {np.mean([r['ao'] for r in last]):7.1f} nm")
        print(f"    residual pistons   {np.mean([r['pist'] for r in last]):7.0f} nm "
              f"(rms of the 5 petal pistons left on the wavefront)")
        print(f"    Strehl             {np.mean([r['sr'] for r in last]):7.4f} "
              f"(per-step spread {np.min([r['sr'] for r in last]):.4f}-{np.max([r['sr'] for r in last]):.4f})")
        print(f"    throughput         {np.mean([r['thr'] for r in last]):7.3f}")
        print(f"    per-lambda error   {np.mean([r['per_lambda'] for r in last]):7.0f} nm")
        print(f"    unwrapped error    {np.mean([r['unwrap'] for r in last]):7.0f} nm")


if __name__ == '__main__':
    main()
