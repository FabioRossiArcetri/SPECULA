import os

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration


class PyramidNonlinearCalibrator(BaseProcessingObj):
    """
    Nonlinear response calibrator for a wavefront sensor (typically a
    non-modulated pyramid).

    Meant to be driven by a :class:`~specula.processing_objects.modal_amplitude_sweep_generator.ModalAmplitudeSweepGenerator`
    (via a DM), which activates one mode at a time across a grid of
    amplitudes. For each active (mode, amplitude) sample, the measured
    slopes are projected onto that mode's small-signal interaction vector
    (from a pre-calibrated linear `intmat`) to obtain a scalar "response".
    At `finalize()`, the per-mode (amplitude, response) samples are
    averaged over repeated amplitudes, sorted, and saved as a
    :class:`~specula.data_objects.nonlinear_calibration.NonlinearCalibration`.

    Note
    ----
    Like :class:`~specula.processing_objects.im_calibrator.ImCalibrator`,
    the active mode/amplitude is inferred from the non-zero entry of
    `in_commands`, so the amplitude grid used by the driving generator
    must not include exactly zero.
    """

    def __init__(self,
                 nmodes: int,
                 intmat: Intmat,
                 data_dir: str,
                 calib_tag: str,
                 first_mode: int = 0,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.nmodes = nmodes
        self.first_mode = first_mode
        self.intmat = intmat
        self.data_dir = data_dir
        self.overwrite = overwrite

        self.calib_path = os.path.join(data_dir, calib_tag)
        if not self.calib_path.endswith('.fits'):
            self.calib_path += '.fits'
        if os.path.exists(self.calib_path) and not overwrite:
            raise FileExistsError(f'Calibration file {self.calib_path} already exists, please remove it')

        # reference small-signal interaction vectors, and their squared norm
        self._ref_vectors = [self.to_xp(intmat.modes[m].copy()) for m in range(nmodes)]
        self._ref_norms = [float(self.xp.dot(v, v)) for v in self._ref_vectors]

        self._amplitude_samples = [[] for _ in range(nmodes)]
        self._response_samples = [[] for _ in range(nmodes)]

        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.inputs['in_commands'] = InputValue(type=BaseValue)

    @classmethod
    def input_names(cls):
        return {'in_slopes': InputDesc(Slopes, 'Measured slopes during the amplitude sweep'),
                'in_commands': InputDesc(BaseValue, 'Commanded modal amplitude vector')}

    @classmethod
    def output_names(cls):
        return {}

    def trigger_code(self):
        commands = self.local_inputs['in_commands'].value
        idx = self.xp.nonzero(commands)[0]
        if len(idx) == 0:
            return

        mode = int(idx[0]) - self.first_mode
        if mode < 0 or mode >= self.nmodes:
            return

        amplitude = float(commands[idx[0]])
        slopes = self.to_xp(self.local_inputs['in_slopes'].slopes)

        norm = self._ref_norms[mode]
        if norm <= 0:
            return
        response = float(self.xp.dot(slopes, self._ref_vectors[mode]) / norm)

        self._amplitude_samples[mode].append(amplitude)
        self._response_samples[mode].append(response)

    def finalize(self):
        import numpy as np

        # Collect the union of amplitudes actually sampled (assumed shared
        # across modes, as produced by ModalAmplitudeSweepGenerator).
        shared_amplitudes = sorted(set(
            round(a, 12) for samples in self._amplitude_samples for a in samples
        ))
        if not shared_amplitudes:
            raise RuntimeError('No calibration samples were collected: '
                               'check that the driving generator actually '
                               'excited every mode.')
        shared_amplitudes = np.array(shared_amplitudes)

        responses = np.zeros((self.nmodes, len(shared_amplitudes)))
        for mode in range(self.nmodes):
            amps = np.round(np.array(self._amplitude_samples[mode]), 12)
            resp = np.array(self._response_samples[mode])
            for j, a in enumerate(shared_amplitudes):
                sel = amps == a
                if not np.any(sel):
                    raise RuntimeError(
                        f'Mode {mode} was not sampled at amplitude {a}; '
                        'every mode must be swept over the same amplitude grid.'
                    )
                responses[mode, j] = resp[sel].mean()

        calib = NonlinearCalibration(shared_amplitudes, responses,
                                     target_device_idx=self.target_device_idx)
        os.makedirs(self.data_dir, exist_ok=True)
        calib.save(self.calib_path, overwrite=self.overwrite)
