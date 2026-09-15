import os
import tempfile
import unittest

import specula
specula.init(0)

from specula import np, cpuArray
from specula.data_objects.nonlinear_calibration import NonlinearCalibration

from test.specula_testlib import cpu_and_gpu


class TestNonlinearCalibration(unittest.TestCase):

    def _build(self, target_device_idx):
        amplitudes = np.linspace(-10.0, 10.0, 21)
        # mode 0: linear response; mode 1: saturating (tanh-like) response
        responses = np.vstack([
            amplitudes,
            5.0 * np.tanh(amplitudes / 5.0),
        ])
        return NonlinearCalibration(amplitudes, responses, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_rejects_non_increasing_amplitudes(self, target_device_idx, xp):
        with self.assertRaises(ValueError):
            NonlinearCalibration([1.0, 1.0, 2.0], np.zeros((1, 3)),
                                 target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_rejects_shape_mismatch(self, target_device_idx, xp):
        with self.assertRaises(ValueError):
            NonlinearCalibration([1.0, 2.0, 3.0], np.zeros((2, 4)),
                                 target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_invert_is_identity_for_linear_mode(self, target_device_idx, xp):
        calib = self._build(target_device_idx)
        response = xp.array([0.0, 3.0, -7.5])
        modes = xp.array([0, 0, 0])
        amplitude = cpuArray(calib.invert(modes, response))
        np.testing.assert_allclose(amplitude, cpuArray(response), atol=1e-6)

    @cpu_and_gpu
    def test_invert_recovers_saturating_mode_amplitude(self, target_device_idx, xp):
        calib = self._build(target_device_idx)
        true_amplitude = 4.0
        response = 5.0 * np.tanh(true_amplitude / 5.0)
        estimated = cpuArray(calib.invert(xp.array([1]), xp.array([response])))
        self.assertAlmostEqual(float(estimated[0]), true_amplitude, places=1)

    @cpu_and_gpu
    def test_invert_clamps_outside_calibrated_range(self, target_device_idx, xp):
        calib = self._build(target_device_idx)
        # response far beyond the saturating mode's calibrated range
        estimated = cpuArray(calib.invert(xp.array([1]), xp.array([1000.0])))
        self.assertAlmostEqual(float(estimated[0]), 10.0, places=6)

    @cpu_and_gpu
    def test_save_and_restore_roundtrip(self, target_device_idx, xp):
        calib = self._build(target_device_idx)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'calib.fits')
            calib.save(path)
            restored = NonlinearCalibration.restore(path, target_device_idx=target_device_idx)

        np.testing.assert_allclose(cpuArray(restored.amplitudes), cpuArray(calib.amplitudes))
        np.testing.assert_allclose(cpuArray(restored.responses), cpuArray(calib.responses))
        self.assertEqual(restored.nmodes, calib.nmodes)
        self.assertEqual(restored.nsamples, calib.nsamples)


if __name__ == '__main__':
    unittest.main()
