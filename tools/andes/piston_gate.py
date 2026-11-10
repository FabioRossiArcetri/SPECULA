"""Close the piston loop only once the high-order loop has converged.

With the pyramid high-order loop (andes_pl_piston_pyr.yml) the wavefront takes ~45 steps to
converge from the open-loop 2.7 um, and while it does the lantern measurements are meaningless:
a piston loop closed from the start integrates them and pushes some petals beyond the capture
range of the unwrap, where it cannot bring them back. A real system closes the high orders first.

``apply(p, open_step, steps)`` gates the piston integrator with its ``gain_mod`` input: 0 before
``open_step``, 1 from it on (a TimeHistory played by TimeHistoryGenerator). The tools call it when
the environment variable PL_PISTON_GATE_STEP is set (unset: the loop is closed from the start,
as before).
"""
import os

import numpy as np

from iteras.data_objects.time_history import TimeHistory

N_PISTONS = 5


def gate_step():
    v = os.environ.get('PL_PISTON_GATE_STEP')
    return int(v) if v else None


def apply(p, open_step, steps):
    tag = f'pl19_1200_piston_gate_{open_step}_of_{steps}'
    path = os.path.join('calib', 'time_hist', tag + '.fits')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hist = np.zeros((steps, N_PISTONS), float)
    hist[open_step:] = 1.0
    if os.path.exists(path):
        os.remove(path)
    TimeHistory(hist, target_device_idx=-1).save(path)
    p['piston_gate_hist'] = {'class': 'TimeHistory', 'tag': tag}
    p['piston_gate'] = {'class': 'TimeHistoryGenerator', 'time_hist_ref': 'piston_gate_hist'}
    p['integrator'].setdefault('inputs', {})['gain_mod'] = 'piston_gate.output'
    return p


def maybe_apply(p, steps):
    s = gate_step()
    return apply(p, s, steps) if s is not None else p
