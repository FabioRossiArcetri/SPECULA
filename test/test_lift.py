import specula
specula.init(0)  # Default target device

import numpy as np
import unittest

from specula import cpuArray
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pixels import Pixels
from specula.lib.make_mask import make_mask
from specula.processing_objects.lift import Lift

from test.specula_testlib import cpu_and_gpu


def make_lift(target_device_idx, nmodes=5, nPistons=1, nZern=4,
              defocus_amp=0.5, pixel_pupil=32, pixel_pitch=0.05,
              wavelengthInNm=750.0, pix_scale=0.02, npix_side=16,
              cropped_size=8, n_iter=5, fft_res=2, quiet=True, fix=True):
    """Helper to build a Lift instance with a synthetic Zernike IFunc."""
    simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
    ifunc = IFunc(
        type_str='zernike',
        npixels=pixel_pupil,
        nmodes=nmodes,
        target_device_idx=target_device_idx,
    )
    lift = Lift(
        simul_params=simul_params,
        defocus_amp=defocus_amp,
        nPistons=nPistons,
        nZern=nZern,
        wavelengthInNm=wavelengthInNm,
        pix_scale=pix_scale,
        npix_side=npix_side,
        cropped_size=cropped_size,
        ifunc=ifunc,
        n_iter=n_iter,
        fft_res=fft_res,
        quiet=quiet,
        fix=fix,
        target_device_idx=target_device_idx,
    )
    return lift, ifunc


class TestLift(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Initialisation
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_init_outputs(self, target_device_idx, xp):
        """Lift must expose the expected output keys after construction."""
        lift, _ = make_lift(target_device_idx)
        for key in ('out_modes', 'out_modes_0', 'out_modes_1',
                    'out_modes_2', 'out_modes_3', 'out_modes_4'):
            self.assertIn(key, lift.outputs,
                          msg=f"Missing output key '{key}'")

    @cpu_and_gpu
    def test_init_input(self, target_device_idx, xp):
        """Lift must expose an 'in_pixels' input."""
        lift, _ = make_lift(target_device_idx)
        self.assertIn('in_pixels', lift.inputs)

    @cpu_and_gpu
    def test_init_nmodes(self, target_device_idx, xp):
        """nmodes must equal nPistons + nZern."""
        nPistons, nZern = 1, 4
        lift, _ = make_lift(target_device_idx, nPistons=nPistons, nZern=nZern,
                             nmodes=nPistons + nZern)
        self.assertEqual(lift.nmodes, nPistons + nZern)

    @cpu_and_gpu
    def test_init_out_modes_shape(self, target_device_idx, xp):
        """out_modes value must have length nmodes."""
        nmodes = 5
        lift, _ = make_lift(target_device_idx, nmodes=nmodes)
        val = cpuArray(lift.outputs['out_modes'].value)
        self.assertEqual(val.shape, (nmodes,))

    @cpu_and_gpu
    def test_init_phase_ref_shape(self, target_device_idx, xp):
        """phase_ref must be a 2-D array of shape (gridSize, gridSize)."""
        lift, _ = make_lift(target_device_idx)
        pr = cpuArray(lift.phase_ref)
        self.assertEqual(pr.ndim, 2)
        self.assertEqual(pr.shape[0], pr.shape[1])
        self.assertEqual(pr.shape[0], lift.gridSize)

    @cpu_and_gpu
    def test_init_modes_list_length(self, target_device_idx, xp):
        """modes list must contain nmodes entries."""
        nmodes = 5
        lift, _ = make_lift(target_device_idx, nmodes=nmodes)
        self.assertEqual(len(lift.modes), nmodes)

    @cpu_and_gpu
    def test_init_modes_cube_shape(self, target_device_idx, xp):
        """modesCube must have shape (nmodes, gridSize, gridSize)."""
        nmodes = 5
        lift, _ = make_lift(target_device_idx, nmodes=nmodes)
        cube = cpuArray(lift.modesCube)
        self.assertEqual(cube.shape, (nmodes, lift.gridSize, lift.gridSize))

    # ------------------------------------------------------------------
    # 2. calc_geometry (static method)
    # ------------------------------------------------------------------

    def test_calc_geometry_returns_named_tuple(self):
        """calc_geometry must return a named tuple with the expected fields."""
        result = Lift.calc_geometry(
            phase_sampling=32,
            pixel_pitch=0.05,
            wavelengthInNm=750.0,
            pix_scale=0.02,
            npix_side=16,
        )
        for field in ('sampling_ratio', 'fft_sampling', 'fft_padding',
                      'fft_size', 'actual_fov'):
            self.assertTrue(hasattr(result, field),
                            msg=f"Missing field '{field}' in WFS_Settings")

    def test_calc_geometry_fft_size_even(self):
        """fft_size must be an even integer."""
        result = Lift.calc_geometry(32, 0.05, 750.0, 0.02, 16)
        self.assertEqual(result.fft_size % 2, 0)

    def test_calc_geometry_consistency(self):
        """fft_size must equal fft_sampling + fft_padding."""
        result = Lift.calc_geometry(32, 0.05, 750.0, 0.02, 16)
        self.assertEqual(result.fft_size,
                         result.fft_sampling + result.fft_padding)

    def test_calc_geometry_positive_values(self):
        """All WFS_Settings fields must be positive."""
        result = Lift.calc_geometry(32, 0.05, 750.0, 0.02, 16)
        for field in result._fields:
            self.assertGreater(getattr(result, field), 0,
                               msg=f"Field '{field}' is not positive")

    # ------------------------------------------------------------------
    # 3. set_modalbase / gridSize / fftSize
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_grid_and_fft_sizes_positive(self, target_device_idx, xp):
        """gridSize and fftSize must both be positive after construction."""
        lift, _ = make_lift(target_device_idx)
        self.assertGreater(lift.gridSize, 0)
        self.assertGreater(lift.fftSize, 0)

    @cpu_and_gpu
    def test_fft_size_ge_grid_size(self, target_device_idx, xp):
        """fftSize must be >= gridSize."""
        lift, _ = make_lift(target_device_idx)
        self.assertGreaterEqual(lift.fftSize, lift.gridSize)

    # ------------------------------------------------------------------
    # 4. phaseFromCoeffs
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_phase_from_zero_coeffs(self, target_device_idx, xp):
        """phaseFromCoeffs with all-zero coefficients must return a zero array."""
        lift, _ = make_lift(target_device_idx)
        coeffs = xp.zeros(lift.nmodes, dtype=lift.dtype)
        phase = lift.phaseFromCoeffs(coeffs)
        phase_cpu = cpuArray(phase)
        np.testing.assert_array_almost_equal(
            phase_cpu, np.zeros_like(phase_cpu),
            err_msg="phaseFromCoeffs with zero coefficients should be all-zero"
        )

    @cpu_and_gpu
    def test_phase_from_coeffs_shape(self, target_device_idx, xp):
        """phaseFromCoeffs must return a (gridSize, gridSize) array."""
        lift, _ = make_lift(target_device_idx)
        coeffs = xp.zeros(lift.nmodes, dtype=lift.dtype)
        phase = lift.phaseFromCoeffs(coeffs)
        self.assertEqual(cpuArray(phase).shape,
                         (lift.gridSize, lift.gridSize))

    @cpu_and_gpu
    def test_phase_from_coeffs_linearity(self, target_device_idx, xp):
        """phaseFromCoeffs must be linear: 2*a coeffs → 2*a phase."""
        lift, _ = make_lift(target_device_idx)
        coeffs = xp.ones(lift.nmodes, dtype=lift.dtype)
        p1 = cpuArray(lift.phaseFromCoeffs(coeffs))
        p2 = cpuArray(lift.phaseFromCoeffs(2.0 * coeffs))
        np.testing.assert_array_almost_equal(2.0 * p1, p2)

    # ------------------------------------------------------------------
    # 5. ft_ft2
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_ft_ft2_shape(self, target_device_idx, xp):
        """ft_ft2 output must have shape (gridSize, gridSize)."""
        lift, _ = make_lift(target_device_idx)
        x = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.complex_dtype)
        result = lift.ft_ft2(x)
        self.assertEqual(cpuArray(result).shape,
                         (lift.gridSize, lift.gridSize))

    @cpu_and_gpu
    def test_ft_ft2_zero_input(self, target_device_idx, xp):
        """ft_ft2 of an all-zero input must be all-zero."""
        lift, _ = make_lift(target_device_idx)
        x = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.complex_dtype)
        result = cpuArray(lift.ft_ft2(x))
        np.testing.assert_array_almost_equal(
            np.abs(result), np.zeros_like(np.abs(result))
        )

    # ------------------------------------------------------------------
    # 6. abs2
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_abs2_real_array(self, target_device_idx, xp):
        """abs2 of a real array must equal the element-wise square."""
        lift, _ = make_lift(target_device_idx)
        arr = xp.array([1.0, 2.0, 3.0], dtype=lift.dtype)
        result = cpuArray(lift.abs2(arr))
        np.testing.assert_array_almost_equal(result, np.array([1.0, 4.0, 9.0]))

    @cpu_and_gpu
    def test_abs2_complex_array(self, target_device_idx, xp):
        """abs2 of a complex array must equal |z|^2 for each element."""
        lift, _ = make_lift(target_device_idx)
        arr = xp.array([1.0 + 1.0j, 0.0 + 2.0j], dtype=lift.complex_dtype)
        result = cpuArray(lift.abs2(arr))
        expected = np.array([2.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    # ------------------------------------------------------------------
    # 7. computeCoG
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_computeCoG_centred_peak(self, target_device_idx, xp):
        """CoG of a single central peak must be close to the array centre."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        frame = xp.zeros((n, n), dtype=lift.dtype)
        cy, cx = n // 2, n // 2
        frame[cy, cx] = 1.0
        yc, xc = lift.computeCoG(frame)
        self.assertAlmostEqual(float(cpuArray(xp.array(xc))), cx, delta=1.0)
        self.assertAlmostEqual(float(cpuArray(xp.array(yc))), cy, delta=1.0)

    # ------------------------------------------------------------------
    # 8. crop / calcCroppedFlux
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_crop_shape(self, target_device_idx, xp):
        """crop must return an array of shape (2*side, 2*side)."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        frame = xp.ones((n, n), dtype=lift.dtype)
        center = (n // 2, n // 2)
        side = lift.cropped_size
        cropped = lift.crop(frame, center, side=side)
        expected_side = 2 * side
        self.assertEqual(cpuArray(cropped).shape,
                         (expected_side, expected_side))

    @cpu_and_gpu
    def test_calcCroppedFlux_positive(self, target_device_idx, xp):
        """calcCroppedFlux must return a positive value for a uniform frame."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        frame = xp.ones((n, n), dtype=lift.dtype)
        center = (n // 2, n // 2)
        flux = float(cpuArray(lift.calcCroppedFlux(frame, center)))
        self.assertGreater(flux, 0.0)

    # ------------------------------------------------------------------
    # 9. complexField / focalPlaneImageFromFFT
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_complexField_shapes(self, target_device_idx, xp):
        """complexField must return two (gridSize, gridSize) arrays."""
        lift, _ = make_lift(target_device_idx)
        phase = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        cf, cfft = lift.complexField(phase)
        self.assertEqual(cpuArray(cf).shape,
                         (lift.gridSize, lift.gridSize))
        self.assertEqual(cpuArray(cfft).shape,
                         (lift.gridSize, lift.gridSize))

    @cpu_and_gpu
    def test_focalPlaneImageFromFFT_non_negative(self, target_device_idx, xp):
        """PSF (intensity) must be non-negative everywhere."""
        lift, _ = make_lift(target_device_idx)
        phase = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        _, cfft = lift.complexField(phase)
        psf = cpuArray(lift.focalPlaneImageFromFFT(cfft))
        self.assertTrue(np.all(psf >= 0.0),
                        msg="PSF contains negative values")

    @cpu_and_gpu
    def test_focalPlaneImageFromFFT_set_flux(self, target_device_idx, xp):
        """set_flux must normalise the PSF so its sum equals the requested flux."""
        lift, _ = make_lift(target_device_idx)
        phase = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        _, cfft = lift.complexField(phase)
        target_flux = 1234.0
        psf = cpuArray(lift.focalPlaneImageFromFFT(cfft, set_flux=target_flux))
        self.assertAlmostEqual(float(psf.sum()), target_flux, delta=1.0)

    # ------------------------------------------------------------------
    # 10. computeNoiseCovarianceDiag
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_noise_cov_diag_all_positive(self, target_device_idx, xp):
        """Noise covariance diagonal must be strictly positive."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        image = xp.ones((n, n), dtype=lift.dtype)
        diag = cpuArray(lift.computeNoiseCovarianceDiag(image))
        self.assertTrue(np.all(diag > 0.0),
                        msg="Noise covariance diagonal has non-positive values")

    @cpu_and_gpu
    def test_noise_cov_diag_length(self, target_device_idx, xp):
        """Noise covariance diagonal length must match gridSize^2."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        image = xp.ones((n, n), dtype=lift.dtype)
        diag = cpuArray(lift.computeNoiseCovarianceDiag(image))
        self.assertEqual(diag.size, n * n)

    # ------------------------------------------------------------------
    # 11. computeReconstructor
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_computeReconstructor_shapes(self, target_device_idx, xp):
        """computeReconstructor must return (Rinv, P_ML) with correct shapes."""
        lift, _ = make_lift(target_device_idx)
        npix = lift.gridSize * lift.gridSize
        nmodes = lift.nmodes
        H = xp.eye(npix, nmodes, dtype=lift.dtype)
        Rdiag = xp.ones(npix, dtype=lift.dtype)
        Rinv, P_ML = lift.computeReconstructor(H, Rdiag)
        Rinv_cpu = cpuArray(Rinv)
        P_ML_cpu = cpuArray(P_ML)
        self.assertEqual(Rinv_cpu.shape, (npix,))
        self.assertEqual(P_ML_cpu.shape, (nmodes, npix))

    # ------------------------------------------------------------------
    # 12. setRefTT
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_setRefTT(self, target_device_idx, xp):
        """setRefTT must update ref_tip and ref_tilt."""
        lift, _ = make_lift(target_device_idx)
        lift.setRefTT(0.1, -0.2)
        self.assertAlmostEqual(lift.ref_tip, 0.1)
        self.assertAlmostEqual(lift.ref_tilt, -0.2)

    # ------------------------------------------------------------------
    # 13. calcCenter / fix flag
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_calcCenter_fix_true(self, target_device_idx, xp):
        """With fix=True, calcCenter must always return the geometric centre."""
        lift, _ = make_lift(target_device_idx, fix=True)
        n = lift.gridSize
        frame = xp.zeros((n, n), dtype=lift.dtype)
        frame[0, 0] = 1.0   # off-centre peak — should be ignored
        cx, cy = lift.calcCenter(frame)
        self.assertAlmostEqual(cx, 0.5 * n)
        self.assertAlmostEqual(cy, 0.5 * n)

    # ------------------------------------------------------------------
    # 14. crop_or_enlarge_around_peak
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_crop_or_enlarge_exact_size(self, target_device_idx, xp):
        """crop_or_enlarge_around_peak must return an array of the desired width."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        frame = xp.zeros((n, n), dtype=lift.dtype)
        frame[n // 2, n // 2] = 1.0
        desired = n
        result = cpuArray(lift.crop_or_enlarge_around_peak(frame, desired))
        self.assertEqual(result.shape, (desired, desired))

    @cpu_and_gpu
    def test_crop_or_enlarge_small_input(self, target_device_idx, xp):
        """crop_or_enlarge_around_peak must pad a smaller input to desired_width."""
        lift, _ = make_lift(target_device_idx)
        small = xp.ones((4, 4), dtype=lift.dtype)
        desired = 16
        result = cpuArray(lift.crop_or_enlarge_around_peak(small, desired))
        self.assertEqual(result.shape, (desired, desired))

    # ------------------------------------------------------------------
    # 15. focalPlaneImageLIFT
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_focalPlaneImageLIFT_shape(self, target_device_idx, xp):
        """focalPlaneImageLIFT must return a (gridSize, gridSize) array."""
        lift, _ = make_lift(target_device_idx)
        phase = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        psf = cpuArray(lift.focalPlaneImageLIFT(phase))
        self.assertEqual(psf.shape, (lift.gridSize, lift.gridSize))

    @cpu_and_gpu
    def test_focalPlaneImageLIFT_non_negative(self, target_device_idx, xp):
        """focalPlaneImageLIFT output must be non-negative."""
        lift, _ = make_lift(target_device_idx)
        phase = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        psf = cpuArray(lift.focalPlaneImageLIFT(phase))
        self.assertTrue(np.all(psf >= 0.0))

    # ------------------------------------------------------------------
    # 16. calc_psf
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_calc_psf_shape(self, target_device_idx, xp):
        """calc_psf must return a 2-D array."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        phase = xp.zeros((n, n), dtype=lift.dtype)
        amp = xp.ones((n, n), dtype=lift.dtype)
        psf = cpuArray(lift.calc_psf(phase, amp))
        self.assertEqual(psf.ndim, 2)

    @cpu_and_gpu
    def test_calc_psf_normalize(self, target_device_idx, xp):
        """calc_psf with normalize=True must return a PSF that sums to 1."""
        lift, _ = make_lift(target_device_idx)
        n = lift.gridSize
        phase = xp.zeros((n, n), dtype=lift.dtype)
        amp = xp.ones((n, n), dtype=lift.dtype)
        psf = cpuArray(lift.calc_psf(phase, amp, normalize=True))
        self.assertAlmostEqual(float(psf.sum()), 1.0, places=5)

    # ------------------------------------------------------------------
    # 17. phaseEstimation (smoke test — low n_iter)
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_phaseEstimation_returns_correct_types(self, target_device_idx, xp):
        """phaseEstimation must return (phase, coeffs, niters) with the right types."""
        lift, _ = make_lift(target_device_idx, n_iter=3, fix=True)
        # Build a synthetic PSF from a flat wavefront
        phase0 = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        psf = lift.focalPlaneImageLIFT(phase0)
        phase_est, coeffs, niters = lift.phaseEstimation(psf)
        phase_est_cpu = cpuArray(phase_est)
        coeffs_cpu = np.asarray(coeffs)
        self.assertEqual(phase_est_cpu.shape,
                         (lift.gridSize, lift.gridSize))
        self.assertEqual(coeffs_cpu.shape, (lift.nmodes,))
        self.assertIsInstance(int(niters), int)
        self.assertGreater(int(niters), 0)

    @cpu_and_gpu
    def test_phaseEstimation_raises_without_modal_base(self, target_device_idx, xp):
        """phaseEstimation must raise an exception if modal base is not set."""
        lift, _ = make_lift(target_device_idx, n_iter=2)
        lift.modes = []  # clear the modal base
        phase0 = xp.zeros((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        psf = xp.ones((lift.gridSize, lift.gridSize), dtype=lift.dtype)
        with self.assertRaises(Exception):
            lift.phaseEstimation(psf)

    # ------------------------------------------------------------------
    # 18. trigger
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_trigger_updates_out_modes(self, target_device_idx, xp):
        """After trigger, out_modes value must have the correct length."""
        lift, _ = make_lift(target_device_idx, n_iter=2, fix=True)
        # Wire a synthetic Pixels input
        n = lift.gridSize
        pix = Pixels(n, n, target_device_idx=target_device_idx)
        phase0 = xp.zeros((n, n), dtype=lift.dtype)
        psf = lift.focalPlaneImageLIFT(phase0)
        pix.pixels[:] = lift.xp.array(psf, dtype=pix.pixels.dtype)
        lift.inputs['in_pixels'].set(pix)
        lift.prepare_trigger(0)
        lift.trigger()
        out = cpuArray(lift.outputs['out_modes'].value)
        self.assertEqual(out.shape, (lift.nmodes,))

    @cpu_and_gpu
    def test_trigger_updates_scalar_outputs(self, target_device_idx, xp):
        """After trigger, out_modes_0 … out_modes_4 must be finite scalars."""
        lift, _ = make_lift(target_device_idx, n_iter=2, fix=True)
        n = lift.gridSize
        pix = Pixels(n, n, target_device_idx=target_device_idx)
        phase0 = xp.zeros((n, n), dtype=lift.dtype)
        psf = lift.focalPlaneImageLIFT(phase0)
        pix.pixels[:] = lift.xp.array(psf, dtype=pix.pixels.dtype)
        lift.inputs['in_pixels'].set(pix)
        lift.prepare_trigger(0)
        lift.trigger()
        for k in ('out_modes_0', 'out_modes_1', 'out_modes_2',
                  'out_modes_3', 'out_modes_4'):
            val = float(lift.outputs[k].value)
            self.assertTrue(np.isfinite(val),
                            msg=f"Output '{k}' is not finite: {val}")

    @cpu_and_gpu
    def test_trigger_no_pixels_is_no_op(self, target_device_idx, xp):
        """trigger must be a no-op (not raise) when no pixels have been set."""
        lift, _ = make_lift(target_device_idx)
        lift.in_pixels = None
        lift.trigger()   # should not raise

    # ------------------------------------------------------------------
    # 19. finalize
    # ------------------------------------------------------------------

    @cpu_and_gpu
    def test_finalize_does_not_raise(self, target_device_idx, xp):
        """finalize must run without raising any exception."""
        lift, _ = make_lift(target_device_idx)
        lift.finalize()
