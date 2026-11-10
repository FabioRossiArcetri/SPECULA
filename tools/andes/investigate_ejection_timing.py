#!/usr/bin/env python
"""Was the cubic OMP's bad behaviour caused by ejecting before the AO converged?

The measurements in this session always injected the 1600 nm piston at t = 0, i.e.
while the high-order loop was still converging from a 3.3 um wavefront. A model
fitted on *converged* AO residuals then sees, for the first ~15 steps, a wavefront
20 times rougher than anything in its training set, and the piston integrator --
which accumulates -- carries that error forward.

This script delays the ejection until the AO has converged and repeats the
measurement, for both strategies and both seeings, so the two effects are separated.
The step is made with a square wave whose half period is the delay (no timeout
parameter exists): constant +-amp gives 0 before the delay and 2*amp after it.

    python tools/andes/investigate_ejection_timing.py --delay-ms 60 --steps 60
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import yaml

# Use the checkout this script lives in, not whatever specula is installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import specula
specula.init(0)
from iteras.loop_control import LoopControl
from iteras import simul as simul_mod
from specula import cpuArray
from specula.data_objects.ifunc import IFunc

INJECTED = 1600.0          # [nm] on petal 0


def schedule_delayed_step(params, delay_s, amp=INJECTED):
    '''
    Replace the constant ejection of *params* with one that starts at *delay_s*.

    A SQUARE wave of amplitude -amp/2 around +amp/2 is 0 before the first half
    period and +amp after it, which is exactly a step at delay_s.
    '''
    block = params['disturbance_petals']
    n_modes = len(block['constant'])
    block['class'] = 'WaveGenerator'
    block['constant'] = [amp / 2.0] + [0.0] * (n_modes - 1)
    block['amp'] = [-amp / 2.0] + [0.0] * (n_modes - 1)
    block['freq'] = [1.0 / (2.0 * delay_s)] + [0.0] * (n_modes - 1)
    block['duty_cycle'] = [0.5] * n_modes
    block['wave_type'] = 'SQUARE'
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/ANDES/andes_pl_piston.yml')
    ap.add_argument('--delay-ms', type=float, default=60.0)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--gain', type=float, default=0.2)
    ap.add_argument('--converged', type=int, default=10)
    args = ap.parse_args()

    ifunc = IFunc.restore('calib/ifunc/ANDES_400pix_all_modes.fits', target_device_idx=-1)
    mask = np.asarray(cpuArray(ifunc.mask_inf_func)) > 0
    basis = np.asarray(cpuArray(ifunc.influence_function), float)[:5]
    basis -= basis.mean(axis=1, keepdims=True)
    binv = np.linalg.pinv(basis).T

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
            e = wrapped - res[:, None]
            e = e - lam[None, :] * np.round(e / lam[None, :])
            pist = basis.T @ (binv @ phase)
            state['rows'].append(dict(
                pist=pist.std(), ao=(phase - pist).std(),
                sr=float(cpuArray(o['psf'].outputs['out_sr'].value)),
                thr=float(np.asarray(cpuArray(o['pl_sensor'].outputs['out_throughput'].value)).mean()),
                per_lambda=float(np.abs(e).mean()),
                unwrap=float(np.abs(meas - res).mean())))
        return r

    LoopControl.iter = patched_iter
    orig_build = simul_mod.Simul.build_objects

    def patched_build(self, params):
        orig_build(self, params)
        state['objs'] = dict(self.objs)

    simul_mod.Simul.build_objects = patched_build

    base = yaml.safe_load(open(args.config))
    cases = [
        ('closed form', 'LanternClosedFormModel', 'pl19_1200_closed_form_ao120_1100_1300_10nm'),
        ('cubic OMP', 'LanternCubicModel', 'pl19_1200_ao120_k1282_1100_1300_10nm'),
    ]
    print(f'ejection at {args.delay_ms:.0f} ms (AO has converged), {args.steps} steps\n')
    print(f'{"seeing":>7} {"model":<12} {"AO res":>8} {"per-lam":>8} {"unwrapped":>10} '
          f'{"pistons":>8} {"Strehl":>7} {"thru":>6}')
    for seeing in (0.8, 0.6):
        for name, klass, tag in cases:
            p = copy.deepcopy(base)
            p['main']['total_time'] = args.steps * base['main']['time_step']
            p['seeing']['constant'] = seeing
            p['integrator']['int_gain'] = [args.gain]
            p['pl_model']['class'] = klass
            p['pl_model']['tag'] = tag
            p['data_store']['store_dir'] = f'./output/delay_{int(seeing * 10)}_{name.split()[0]}'
            schedule_delayed_step(p, args.delay_ms * 1e-3)
            yaml.safe_dump(p, open('ptmp_delay.yml', 'w'))
            state['rows'] = []
            state['objs'] = {}
            t0 = time.time()
            specula.main_simul(['ptmp_delay.yml'], cpu=True, no_speed_report=True, log_level='ERROR')
            last = state['rows'][-args.converged:]
            print(f'{seeing:7.1f} {name:<12} {np.mean([r["ao"] for r in last]):8.1f} '
                  f'{np.mean([r["per_lambda"] for r in last]):8.0f} '
                  f'{np.mean([r["unwrap"] for r in last]):10.0f} '
                  f'{np.mean([r["pist"] for r in last]):8.0f} '
                  f'{np.mean([r["sr"] for r in last]):7.3f} '
                  f'{np.mean([r["thr"] for r in last]):6.3f}   ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
