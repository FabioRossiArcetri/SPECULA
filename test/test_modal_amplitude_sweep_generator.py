import unittest

import specula
specula.init(0)

from specula import np, cpuArray
from specula.lib.modal_amplitude_sweep_signal import modal_amplitude_sweep_signal
from specula.processing_objects.modal_amplitude_sweep_generator import ModalAmplitudeSweepGenerator

from test.specula_testlib import cpu_and_gpu


class TestModalAmplitudeSweepSignal(unittest.TestCase):

    def test_shape_and_step_count(self):
        n_modes = 3
        amplitudes = [-1.0, 1.0]
        hist = modal_amplitude_sweep_signal(n_modes, amplitudes)
        self.assertEqual(hist.shape, (n_modes * len(amplitudes), n_modes))

    def test_one_mode_active_per_row(self):
        hist = modal_amplitude_sweep_signal(4, [-2.0, -1.0, 1.0, 2.0])
        nonzero_per_row = np.count_nonzero(hist, axis=1)
        np.testing.assert_array_equal(nonzero_per_row, np.ones(hist.shape[0]))

    def test_sweeps_each_mode_through_full_amplitude_grid_in_order(self):
        amplitudes = [-3.0, 0.5, 4.0]
        hist = modal_amplitude_sweep_signal(2, amplitudes)
        # mode 0 rows
        np.testing.assert_allclose(hist[0:3, 0], amplitudes)
        np.testing.assert_allclose(hist[0:3, 1], [0, 0, 0])
        # mode 1 rows
        np.testing.assert_allclose(hist[3:6, 1], amplitudes)
        np.testing.assert_allclose(hist[3:6, 0], [0, 0, 0])

    def test_first_mode_is_skipped(self):
        hist = modal_amplitude_sweep_signal(3, [1.0, 2.0], first_mode=1)
        self.assertEqual(hist.shape, (2 * 2, 3))
        np.testing.assert_array_equal(hist[:, 0], np.zeros(4))

    def test_nsamples_repeats_each_row(self):
        hist = modal_amplitude_sweep_signal(2, [1.0, 2.0], nsamples=3)
        self.assertEqual(hist.shape, (2 * 2 * 3, 2))
        np.testing.assert_allclose(hist[0], hist[1])
        np.testing.assert_allclose(hist[0], hist[2])
        self.assertFalse(np.allclose(hist[0], hist[3]))


class TestModalAmplitudeSweepGenerator(unittest.TestCase):

    @cpu_and_gpu
    def test_niters_matches_time_history_length(self, target_device_idx, xp):
        gen = ModalAmplitudeSweepGenerator(
            nmodes=3, amplitudes=[-1.0, 1.0], nsamples=2,
            target_device_idx=target_device_idx)
        self.assertEqual(gen.niters(), 3 * 2 * 2)

    @cpu_and_gpu
    def test_trigger_outputs_current_row(self, target_device_idx, xp):
        gen = ModalAmplitudeSweepGenerator(
            nmodes=2, amplitudes=[5.0, -5.0], target_device_idx=target_device_idx)

        gen.setup()
        for step in range(gen.niters()):
            gen.check_ready(gen.seconds_to_t(step))
            gen.trigger()
            gen.post_trigger()
            expected_row = cpuArray(gen.time_hist[step])
            np.testing.assert_allclose(cpuArray(gen.output.value), expected_row)


if __name__ == '__main__':
    unittest.main()
