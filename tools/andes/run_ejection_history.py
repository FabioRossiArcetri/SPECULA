#!/usr/bin/env python
"""Long history of piston ejections, for the deck's "history" slide.

Builds a schedule of random piston ejections in +-8 lambda on all five petals,
saves it as a TimeHistory, runs the ANDES configuration playing it back through
TimeHistoryGenerator, and stores the truth and the measurement with DataStore.

The first ejection is placed after the high-order loop has converged (~10 steps),
so that the history shows the sensor and the piston loop working on a realistic
wavefront. One run per number of ejections.

    python tools/andes/run_ejection_history.py --events 1 --steps 60 --seed 1
    python tools/andes/run_ejection_history.py --events 2 --steps 60 --seed 2
    python tools/andes/run_ejection_history.py --events 5 --steps 60 --seed 3
"""
import argparse
import copy
import os
import sys

import numpy as np
import yaml

# Use the checkout this script lives in, not whatever specula is installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import specula
specula.init(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piston_gate                                         # PL_PISTON_GATE_STEP: close the piston loop late

CPU = os.environ.get('SPECULA_CPU', '1') == '1'           # SPECULA_CPU=0: run on the GPU
SUFFIX = os.environ.get('SPECULA_OUT_SUFFIX', '')           # appended to the output stores
from iteras.data_objects.time_history import TimeHistory

N_MODES = 10            # rows of the disturbance: 5 petal modes + 5 unused
FIRST_STEP = int(os.environ.get('PL_FIRST_EJECTION', 15))   # first ejection, after the high-order loop has converged
SPAN_NM = 9600.0        # +-8 lambda at 1200 nm


def build_schedule(n_pistons, steps, seed, span=SPAN_NM, first=FIRST_STEP, spacing=15):
    '''
    (steps, N_MODES) history: a new ejection every *spacing* steps, driving
    *n_pistons* of the five petals (the first ones) to fresh random positions in
    +-span; the other petals stay at zero. The first ejection is after the
    high-order loop has converged.
    '''
    rng = np.random.default_rng(seed)
    history = np.zeros((steps, N_MODES), float)
    event_steps = list(range(first, steps - 1, spacing))
    current = np.zeros(5)
    for step in range(steps):
        if step in event_steps:
            current = np.zeros(5)
            current[:n_pistons] = rng.uniform(-span, span, n_pistons)
        history[step, :5] = current
    return history, event_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/ANDES/andes_pl_piston.yml')
    ap.add_argument('--pistons', type=int, default=1, choices=(1, 2, 5),
                    help='how many of the five petals are driven')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--spacing', type=int, default=15,
                    help='steps between ejections (15 = 30 ms)')
    ap.add_argument('--seeing', type=float, default=0.8)
    ap.add_argument('--gain', type=float, default=0.5)
    ap.add_argument('--delay', type=int, default=0)
    ap.add_argument('--span-nm', type=float, default=SPAN_NM,
                    help='half range of the random ejections [nm]')
    ap.add_argument('--pspan', type=float, default=None,
                    help='capture range of the unwrap [nm], default 1.1x span')
    args = ap.parse_args()

    p = copy.deepcopy(yaml.safe_load(open(args.config)))
    step_s = p['main']['time_step']
    history, event_steps = build_schedule(args.pistons, args.steps, args.seed,
                                          args.span_nm, spacing=args.spacing)

    tag = f'pl19_1200_ejections{args.pistons}_seed{args.seed}'
    path = os.path.join('calib', 'time_hist', tag + '.fits')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    TimeHistory(history, target_device_idx=-1).save(path)
    print(f'{args.pistons} piston(s) driven, an ejection every {args.spacing} steps '
          f'at {[round(s * step_s * 1e3) for s in event_steps]} ms:')
    for step in event_steps[:3]:
        print(f'  step {step:3d}: ' + ' '.join(f'{v:+7.0f}' for v in history[step, :5])
              + ' nm')

    p['main']['total_time'] = args.steps * step_s
    p['seeing']['constant'] = args.seeing
    p['integrator']['int_gain'] = [args.gain]
    p['integrator']['delay'] = args.delay
    p['pl_unwrap']['pspan'] = float(args.pspan if args.pspan is not None else 1.1 * args.span_nm)
    p['disturbance_hist'] = {'class': 'TimeHistory', 'tag': tag}
    p['disturbance_petals'] = {'class': 'TimeHistoryGenerator', 'time_hist_ref': 'disturbance_hist'}
    p['data_store']['store_dir'] = f'./output/history_{args.pistons}_seed{args.seed}' + SUFFIX
    p = piston_gate.maybe_apply(p, args.steps)
    yaml.safe_dump(p, open('ptmp_history.yml', 'w'))
    specula.main_simul(['ptmp_history.yml'], cpu=CPU, no_speed_report=True)
    print('stored in', p['data_store']['store_dir'])


if __name__ == '__main__':
    main()
