from specula.processing_objects.base_generator import BaseGenerator
from specula.lib.modal_amplitude_sweep_signal import modal_amplitude_sweep_signal


class ModalAmplitudeSweepGenerator(BaseGenerator):
    """
    Modal Amplitude Sweep Generator processing object.
    Generates, for each mode in turn, a sweep across a shared grid of
    amplitudes, for nonlinear wavefront sensor response calibration.
    """
    def __init__(self,
                 nmodes: int,
                 amplitudes: list,
                 first_mode: int = 0,
                 nsamples: int = 1,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(
            output_size=nmodes,
            target_device_idx=target_device_idx,
            precision=precision
        )

        time_hist = modal_amplitude_sweep_signal(
            nmodes,
            amplitudes,
            first_mode=first_mode,
            nsamples=nsamples,
            xp=self.xp,
        )
        self.time_hist = self.to_xp(time_hist)

    def niters(self):
        """Number of timesteps needed to complete the full sweep."""
        return self.time_hist.shape[0]

    def trigger_code(self):
        self.output.value[:] = self.time_hist[self.iter_counter]
