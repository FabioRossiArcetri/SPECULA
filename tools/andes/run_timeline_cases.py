"""
Re-run the two timeline cases with the ejection *after* the AO has converged.

Same runs as investigate_ejection_timing.py, but writing the DataStore outputs that
make_figures.py reads: an atmosphere case and a post-AO case, 60 steps, the 1600 nm
step on petal 0 at 60 ms (the AO has converged by ~20 ms).
"""
import argparse
import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import specula
specula.init(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import piston_gate                                         # PL_PISTON_GATE_STEP: close the piston loop late

CPU = os.environ.get('SPECULA_CPU', '1') == '1'           # SPECULA_CPU=0: run on the GPU
SUFFIX = os.environ.get('SPECULA_OUT_SUFFIX', '')           # appended to the output stores

DELAY_MS = float(os.environ.get('PL_EJECTION_MS', 60.0))   # after the AO has converged
AMP = 1600.0

ap = argparse.ArgumentParser()
ap.add_argument('--gain', type=float, default=0.2)
ap.add_argument('--steps', type=int, default=60)
ap.add_argument('--delay', type=int, default=1)
ap.add_argument('--config', default='config/ANDES/andes_pl_piston.yml')
args = ap.parse_args()
STEPS = args.steps


def delayed(params, delay_s=DELAY_MS * 1e-3, amp=AMP):
    block = params['disturbance_petals']
    n = len(block['constant'])
    block.update(class_=None)
    block['class'] = 'WaveGenerator'
    block['constant'] = [amp / 2.0] + [0.0] * (n - 1)
    block['amp'] = [-amp / 2.0] + [0.0] * (n - 1)
    block['freq'] = [1.0 / (2.0 * delay_s)] + [0.0] * (n - 1)
    block['duty_cycle'] = [0.5] * n
    block['wave_type'] = 'SQUARE'
    block.pop('class_', None)
    return params


base = yaml.safe_load(open(args.config))
for seeing, store in ((0.8, './output/fig_atmo'), (0.0, './output/fig_postao')):
    p = copy.deepcopy(base)
    p['main']['total_time'] = STEPS * base['main']['time_step']
    p['seeing']['constant'] = seeing
    delayed(p)
    p['integrator']['int_gain'] = [args.gain]
    p['integrator']['delay'] = args.delay
    p['data_store']['store_dir'] = store + SUFFIX
    p = piston_gate.maybe_apply(p, STEPS)
    yaml.safe_dump(p, open('ptmp_tl.yml', 'w'))
    specula.main_simul(['ptmp_tl.yml'], cpu=CPU, no_speed_report=True, log_level='ERROR')
    print(f'seeing {seeing}: stored in {store}{SUFFIX}')
os.remove('ptmp_tl.yml')
