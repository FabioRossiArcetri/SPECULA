import numpy as np


def modal_amplitude_sweep_signal(
    n_modes,
    amplitudes,
    first_mode=0,
    nsamples=1,
    xp=np,
):
    """
    Generate a modal time history that sweeps each mode, one at a time,
    across a shared grid of amplitudes.

    For calibration of a wavefront sensor's nonlinear response: unlike
    :func:`~specula.lib.modal_pushpull_signal.modal_pushpull_signal` (which
    perturbs each mode with a fixed push-pull amplitude), this holds each
    mode active at every amplitude in `amplitudes` in turn, so the full
    per-mode response curve (including saturation) can be sampled.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    amplitudes : sequence of float
        Amplitude grid applied to every mode, in the order given. Include
        both positive and negative values to sample an odd-symmetric
        response.
    first_mode : int, optional
        First mode to actuate. Default 0.
    nsamples : int, optional
        Number of timesteps to hold each (mode, amplitude) pair. Default 1.
    xp : module, optional
        Array module to use (numpy or cupy). Default numpy.

    Returns
    -------
    time_hist : ndarray [n_steps, n_modes]
        Modal time history: at most one non-zero entry per row.
    """
    amplitudes = xp.asarray(amplitudes)
    real_n_modes = n_modes - first_mode
    n_amplitudes = len(amplitudes)

    time_hist = xp.zeros((real_n_modes * n_amplitudes, n_modes))
    for mode in range(first_mode, n_modes):
        hist_idx = mode - first_mode
        start = hist_idx * n_amplitudes
        time_hist[start:start + n_amplitudes, mode] = amplitudes

    return xp.repeat(time_hist, nsamples, axis=0)
