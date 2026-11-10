#!/usr/bin/env python
"""Record the residual wavefront of the closed-loop simulation, to train a piston model on
the residual the lantern actually sees.

Runs measure_scenarios.py unchanged (same arguments) and, at every step from --from-step on,
stores the propagated wavefront on the pupil with the five petal pistons projected out.

    PL_DUMP_OUT=out.npz PL_DUMP_FROM=90 python tools/andes/dump_residual_phase.py \
        --config config/ANDES/andes_pl_piston_pyr.yml --steps 120 --scenario closed ...

Output: phase (n_steps, n_pix) [nm], mask (n, n), t [s].
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_scenarios as ms                               # noqa: E402  (initialises specula)
from iteras.loop_control import LoopControl                  # noqa: E402
from iteras import simul as simul_mod                        # noqa: E402
from specula import cpuArray                                 # noqa: E402
from specula.data_objects.ifunc import IFunc                 # noqa: E402

OUT = os.environ['PL_DUMP_OUT']
FROM = int(os.environ.get('PL_DUMP_FROM', '0'))
ifunc = IFunc.restore('calib/ifunc/ANDES_400pix_all_modes.fits', target_device_idx=-1)
mask = np.asarray(cpuArray(ifunc.mask_inf_func)) > 0
B = np.asarray(cpuArray(ifunc.influence_function), float)[:5]
B = np.vstack([np.ones((1, B.shape[1])), B])
Q, _ = np.linalg.qr(B.T)
store = {'objs': {}, 'phase': [], 't': [], 'k': 0}

_iter = LoopControl.iter
_build = simul_mod.Simul.build_objects


def iter_(self):
    r = _iter(self)
    o = store['objs']
    if o:
        store['k'] += 1
        if store['k'] > FROM:
            ph = np.asarray(cpuArray(o['prop'].outputs['out_wfs_source_ef'].phaseInNm), float)[mask]
            store['phase'].append((ph - Q @ (Q.T @ ph)).astype(np.float32))
            store['t'].append(self.t_to_seconds(self.t))
    return r


def build_(self, params):
    _build(self, params)
    store['objs'] = dict(self.objs)
    store['k'] = 0


LoopControl.iter = iter_
simul_mod.Simul.build_objects = build_
ms.main()
ph = np.stack(store['phase'])
np.savez(OUT, phase=ph, mask=mask, t=np.asarray(store['t']))
print(f'saved {OUT}: {ph.shape}, rms {np.sqrt(np.mean(ph ** 2, axis=1)).mean():.1f} nm')
