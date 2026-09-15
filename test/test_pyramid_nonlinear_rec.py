import unittest

import specula
specula.init(0)

from specula import np, cpuArray
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat
from specula.data_objects.intmat import Intmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration
from specula.processing_objects.modalrec import Modalrec
from specula.processing_objects.pyramid_nonlinear_rec import PyramidNonlinearRec

from test.specula_testlib import cpu_and_gpu


def linear_calibration(nmodes, target_device_idx, amp_range=10.0):
    amplitudes = np.linspace(-amp_range, amp_range, 21)
    responses = np.tile(amplitudes, (nmodes, 1))
    return NonlinearCalibration(amplitudes, responses, target_device_idx=target_device_idx)


def saturating_calibration(target_device_idx, amp_range=10.0):
    amplitudes = np.linspace(-amp_range, amp_range, 41)
    responses = 5.0 * np.tanh(amplitudes[None, :] / 5.0)
    return NonlinearCalibration(amplitudes, responses, target_device_idx=target_device_idx)


def build_rec(recmat_array, intmat_array, calib, target_device_idx, **kwargs):
    recmat = Recmat(recmat_array, target_device_idx=target_device_idx)
    intmat = Intmat(intmat_array, target_device_idx=target_device_idx)
    return PyramidNonlinearRec(recmat=recmat, intmat=intmat, nonlinear_calib=calib,
                               target_device_idx=target_device_idx, **kwargs)


def run_once(rec, slopes_vec, target_device_idx):
    slopes = Slopes(slopes=slopes_vec, target_device_idx=target_device_idx)
    slopes.generation_time = rec.seconds_to_t(1)
    rec.inputs['in_slopes'].set(slopes)
    rec.setup()
    t = rec.seconds_to_t(1)
    rec.check_ready(t)
    rec.trigger()
    rec.post_trigger()
    return cpuArray(rec.outputs['out_modes'].value).copy()


class TestPyramidNonlinearRec(unittest.TestCase):

    @cpu_and_gpu
    def test_reduces_to_modalrec_with_linear_calibration_and_no_orthogonal_step(self, target_device_idx, xp):
        """With a linear calibration curve and the orthogonal correction
        disabled, PyramidNonlinearRec must match plain Modalrec exactly."""
        recmat_array = np.array([[0.6, -0.2, 0.1]])
        intmat_array = recmat_array.T  # irrelevant here since correction is off
        calib = linear_calibration(nmodes=1, target_device_idx=target_device_idx)

        nlrec = build_rec(recmat_array, intmat_array, calib, target_device_idx,
                          apply_orthogonal_correction=False)
        modalrec = Modalrec(Recmat(recmat_array, target_device_idx=target_device_idx),
                            target_device_idx=target_device_idx)

        slopes_vec = xp.array([1.0, 2.0, -3.0])
        out_nlrec = run_once(nlrec, slopes_vec, target_device_idx)
        out_modalrec = run_once(modalrec, slopes_vec, target_device_idx)

        np.testing.assert_allclose(out_nlrec, out_modalrec, atol=1e-6)

    @cpu_and_gpu
    def test_desaturates_single_mode_amplitude(self, target_device_idx, xp):
        """A single mode whose measured signal has saturated must be
        recovered exactly by the parallel (de-saturation) channel alone,
        given a calibration built from the same forward model."""
        v = xp.array([1.0, 0.0])
        recmat_array = cpuArray(v).reshape(1, -1) / float(xp.dot(v, v))
        intmat_array = cpuArray(v).reshape(-1, 1)
        calib = saturating_calibration(target_device_idx)

        rec = build_rec(recmat_array, intmat_array, calib, target_device_idx,
                        apply_orthogonal_correction=False)

        true_amplitude = 8.0
        measured_response = 5.0 * np.tanh(true_amplitude / 5.0)  # saturated, far from true_amplitude
        slopes_vec = measured_response * v

        out = run_once(rec, slopes_vec, target_device_idx)
        self.assertAlmostEqual(out[0], true_amplitude, places=1)
        # sanity: the naive linear estimate would have badly underestimated it
        self.assertLess(measured_response, true_amplitude - 1.0)

    @cpu_and_gpu
    def test_orthogonal_correction_is_noop_when_self_consistent(self, target_device_idx, xp):
        """Regression test: in a self-consistent single-mode case (recmat and
        intmat are exact pseudo-inverses of each other, and the mode's own
        calibration fully explains the signal), enabling the orthogonal
        correction must not perturb the already-correct de-saturated
        estimate. This is what the calibration.forward() re-prediction step
        (rather than a plain linear intmat @ a_parallel) guarantees."""
        v = xp.array([0.0, 1.0])
        recmat_array = cpuArray(v).reshape(1, -1) / float(xp.dot(v, v))
        intmat_array = cpuArray(v).reshape(-1, 1)
        calib = saturating_calibration(target_device_idx)

        rec_with = build_rec(recmat_array, intmat_array, calib, target_device_idx,
                             apply_orthogonal_correction=True)
        rec_without = build_rec(recmat_array, intmat_array, calib, target_device_idx,
                                apply_orthogonal_correction=False)

        true_amplitude = 6.0
        measured_response = 5.0 * np.tanh(true_amplitude / 5.0)
        slopes_vec = measured_response * v

        out_with = run_once(rec_with, slopes_vec, target_device_idx)
        out_without = run_once(rec_without, slopes_vec, target_device_idx)

        np.testing.assert_allclose(out_with, out_without, atol=1e-6)

    @cpu_and_gpu
    def test_orthogonal_correction_reduces_bias_of_imperfect_reconstructor(self, target_device_idx, xp):
        """When recmat is a deliberately suboptimal (but not physically
        wrong) linear estimator that leaves recoverable information in the
        slopes, one orthogonal-correction pass (using the true forward
        intmat) must move the estimate closer to the true amplitude than
        the uncorrected linear/parallel estimate -- a single Gauss-Newton
        relinearization step, not a full solve."""
        true_forward = xp.array([1.0, 1.0])       # true physical response direction
        recmat_array = np.array([[0.5, 0.0]])      # only half-weights slope 0, ignores slope 1
        intmat_array = cpuArray(true_forward).reshape(-1, 1)
        calib = linear_calibration(nmodes=1, target_device_idx=target_device_idx, amp_range=20.0)

        rec = build_rec(recmat_array, intmat_array, calib, target_device_idx,
                        apply_orthogonal_correction=True)

        true_amplitude = 4.0
        slopes_vec = true_amplitude * true_forward   # exact linear forward model, no noise

        naive_estimate = float((recmat_array @ cpuArray(slopes_vec))[0])  # what Modalrec alone would give
        out = run_once(rec, slopes_vec, target_device_idx)

        self.assertLess(abs(out[0] - true_amplitude), abs(naive_estimate - true_amplitude))


if __name__ == '__main__':
    unittest.main()
