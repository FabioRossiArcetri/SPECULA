import os
import tempfile
import unittest

import specula
specula.init(0)

from specula import np, cpuArray
from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration
from specula.processing_objects.pyramid_nonlinear_calibrator import PyramidNonlinearCalibrator

from test.specula_testlib import cpu_and_gpu


AMPLITUDES = np.array([-8.0, -4.0, -1.0, 1.0, 4.0, 8.0])


def linear_response(a):
    return a


def saturating_response(a):
    return 5.0 * np.tanh(a / 5.0)


def build_calibrator(tmp_dir, target_device_idx, nmodes=2, nslopes=2):
    intmat_array = np.eye(nslopes, nmodes)
    intmat = Intmat(intmat_array, target_device_idx=target_device_idx)
    return PyramidNonlinearCalibrator(
        nmodes=nmodes,
        intmat=intmat,
        data_dir=tmp_dir,
        calib_tag='calib_test',
        target_device_idx=target_device_idx,
    )


def drive_sweep(calibrator, response_fns, nmodes, xp, target_device_idx):
    """Feed one (mode, amplitude) sample per timestep, mirroring what
    ModalAmplitudeSweepGenerator + a real WFS forward model would produce."""
    step = 0
    for mode in range(nmodes):
        v = cpuArray(calibrator.intmat.modes[mode])
        for a in AMPLITUDES:
            commands = xp.zeros(nmodes)
            commands[mode] = a
            slopes_vec = response_fns[mode](a) * xp.array(v)

            cmd_val = BaseValue(value=commands, target_device_idx=target_device_idx)
            slopes_val = Slopes(slopes=slopes_vec, target_device_idx=target_device_idx)
            t = calibrator.seconds_to_t(step)
            cmd_val.generation_time = t
            slopes_val.generation_time = t

            calibrator.inputs['in_commands'].set(cmd_val)
            calibrator.inputs['in_slopes'].set(slopes_val)
            calibrator.check_ready(t)
            calibrator.trigger()
            calibrator.post_trigger()
            step += 1


class TestPyramidNonlinearCalibrator(unittest.TestCase):

    @cpu_and_gpu
    def test_recovers_linear_and_saturating_curves(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as d:
            calibrator = build_calibrator(d, target_device_idx)
            drive_sweep(calibrator, [linear_response, saturating_response], 2, xp, target_device_idx)
            calibrator.finalize()

            calib = NonlinearCalibration.restore(calibrator.calib_path,
                                                 target_device_idx=target_device_idx)

        np.testing.assert_allclose(cpuArray(calib.amplitudes), np.sort(AMPLITUDES))
        order = np.argsort(AMPLITUDES)
        np.testing.assert_allclose(cpuArray(calib.responses[0]), AMPLITUDES[order], atol=1e-6)
        np.testing.assert_allclose(cpuArray(calib.responses[1]), saturating_response(AMPLITUDES[order]), atol=1e-6)

    @cpu_and_gpu
    def test_raises_if_file_exists_without_overwrite(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, 'calib_test.fits'), 'w').close()
            with self.assertRaises(FileExistsError):
                build_calibrator(d, target_device_idx)

    @cpu_and_gpu
    def test_finalize_raises_when_no_samples_collected(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as d:
            calibrator = build_calibrator(d, target_device_idx)
            with self.assertRaises(RuntimeError):
                calibrator.finalize()

    @cpu_and_gpu
    def test_averages_repeated_amplitude_samples(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as d:
            calibrator = build_calibrator(d, target_device_idx, nmodes=1, nslopes=1)
            v = cpuArray(calibrator.intmat.modes[0])

            # feed the same amplitude twice with slightly different (noisy) responses
            for response in (0.9, 1.1):
                cmd_val = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)
                slopes_val = Slopes(slopes=response * xp.array(v), target_device_idx=target_device_idx)
                t = calibrator.seconds_to_t(0)
                cmd_val.generation_time = t
                slopes_val.generation_time = t
                calibrator.inputs['in_commands'].set(cmd_val)
                calibrator.inputs['in_slopes'].set(slopes_val)
                calibrator.check_ready(t)
                calibrator.trigger()
                calibrator.post_trigger()

            calibrator.finalize()
            calib = NonlinearCalibration.restore(calibrator.calib_path,
                                                 target_device_idx=target_device_idx)
        self.assertEqual(calib.nsamples, 1)
        self.assertAlmostEqual(float(cpuArray(calib.responses[0, 0])), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
