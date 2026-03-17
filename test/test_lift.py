import specula
specula.init(-1, precision=1)

import unittest
from types import SimpleNamespace

import numpy as np

from specula import cpuArray
from specula.lib.calc_psf import calc_psf
from specula.data_objects.ifunc import IFunc
from specula.processing_objects.lift import Lift
from specula.data_objects.simul_params import SimulParams


def build_lift(n_pistons=2, n_zern=5, fft_res=2):
    nmodes = n_pistons + n_zern
    mask = np.ones((16, 16), dtype=np.float32)
    n_valid = int(mask.sum())
    influence = np.empty((nmodes, n_valid), dtype=np.float32)
    for mode in range(nmodes):
        influence[mode, :] = (mode + 1) * 1e-3

    ifunc = SimpleNamespace(mask_inf_func=mask, influence_function=influence)
    simul_params = SimulParams(pixel_pupil=16, pixel_pitch=1.0)

    return Lift(
        simul_params=simul_params,
        defocus_amp=0.1,
        nPistons=n_pistons,
        nZern=n_zern,
        wavelengthInNm=750.0,
        pix_scale=0.01,
        npix_side=16,
        cropped_size=4,
        ifunc=ifunc,
        fft_res=fft_res,
        target_device_idx=-1,
        precision=1,
    )


class TestLift(unittest.TestCase):

    def test_has_only_out_modes_output(self):
        lift = build_lift()
        self.assertEqual(set(lift.outputs.keys()), {'out_modes'})

    def test_radians_per_pixel_matches_geometry_fft_res(self):
        lift = build_lift(fft_res=3)
        settings = Lift.calc_geometry(
            phase_sampling=16,
            pixel_pitch=1.0,
            wavelengthInNm=750.0,
            pix_scale=0.01,
            npix_side=16,
            fft_res=3,
        )
        expected = np.pi / (4.0 * settings.fft_res)
        self.assertAlmostEqual(lift.radians_per_pixel, expected)

    def test_dtype_applied_to_internal_arrays(self):
        lift = build_lift()
        self.assertEqual(lift.airef.dtype, lift.dtype)
        self.assertEqual(lift.out_modes.value.dtype, lift.dtype)
        self.assertEqual(lift.mask.dtype, lift.dtype)
        self.assertEqual(lift.modesCube.dtype, lift.dtype)

    def test_set_ref_tt_uses_image_coordinates(self):
        lift = build_lift()
        lift.radians_per_pixel = 0.2
        lift.setRefTT(center_x=7.0, center_y=5.0, image_size=10.0)
        self.assertAlmostEqual(lift.ref_tip, 0.0)
        self.assertAlmostEqual(lift.ref_tilt, 0.4)

    def test_trigger_updates_only_out_modes(self):
        lift = build_lift()
        fake_psf = np.ones((lift.gridSize, lift.gridSize), dtype=np.float32)
        coeffs = np.arange(lift.nmodes, dtype=np.float32)
        lift.in_pixels = SimpleNamespace(get_value=lambda: fake_psf)
        lift.phaseEstimation = lambda psf: (lift.xp.zeros_like(lift.mask), coeffs, 1)
        lift.current_time = 123

        lift.trigger()

        np.testing.assert_array_equal(specula.cpuArray(lift.outputs['out_modes'].value), coeffs)
        self.assertEqual(lift.outputs['out_modes'].generation_time, 123)

    def test_compute_cog_available_and_consistent(self):
        lift = build_lift()
        frame = lift.xp.zeros((16, 16), dtype=lift.dtype)
        frame[3, 5] = 10.0
        yc, xc = lift.computeCoG(frame)
        self.assertAlmostEqual(float(yc), 3.0)
        self.assertAlmostEqual(float(xc), 5.0)

    def test_phase_estimation_recovers_defocus(self):
        """
        Build a noiseless PSF with calc_psf from a known phase (reference
        defocus + a small unknown defocus), feed it to LIFT, and check that
        the estimated defocus coefficient is close to the known value.

        Geometry: with fft_res=1 LIFT's ft_ft2 is just fftshift(fft2(x))/N,
        identical to calc_psf, so pixel scales match with no imwidth padding.
        """
        npixels = 32
        pixel_pitch = 0.5       # m  →  D = 16 m
        wavelengthInNm = 750.0
        n_pistons = 1           # one piston mode
        n_zern = 3              # tip, tilt, defocus
        nmodes = n_pistons + n_zern
        defocus_amp = 0.5 * np.pi      # rad  — reference defocus (lambda/4)
        unknown_rad = 0.35             # rad  — unknown defocus to recover
        unknown_nm = unknown_rad * wavelengthInNm / (2.0 * np.pi)
        defocus_idx = n_pistons + 2   # = 3: defocus position in modal vector

        # Real Zernike IFunc: rows = [piston, tip, tilt, defocus]
        ifunc_obj = IFunc(
            type_str='zernike', nmodes=nmodes, npixels=npixels,
            precision=1, target_device_idx=-1,
        )
        influence = cpuArray(ifunc_obj.influence_function)  # (4, n_valid)
        mask_2d   = cpuArray(ifunc_obj.mask_inf_func)       # (32, 32)
        idx = np.where(mask_2d > 0)

        # Total phase = LIFT reference defocus + unknown defocus (in radians)
        coeffs_in = np.zeros(nmodes, dtype=np.float32)
        coeffs_in[defocus_idx] = defocus_amp + unknown_rad
        phase_rad = np.zeros((npixels, npixels), dtype=np.float32)
        phase_rad[idx] = coeffs_in @ influence
        amp = mask_2d.astype(np.float32)

        # PSF via calc_psf  (no imwidth → 32×32, same pixel scale as LIFT
        # with fft_res=1)
        psf = calc_psf(phase_rad, amp, xp=np, complex_dtype=np.complex64)
        psf = psf.astype(np.float32)

        # pix_scale chosen so sampling_ratio = 1.0  →  gridSize = npixels
        rad2arcsec = 206264.806247
        fov_internal = (wavelengthInNm * 1e-9 / pixel_pitch) * rad2arcsec
        pix_scale = fov_internal / npixels

        simul_params = SimulParams(pixel_pupil=npixels, pixel_pitch=pixel_pitch)
        lift = Lift(
            simul_params=simul_params,
            defocus_amp=defocus_amp,
            nPistons=n_pistons,
            nZern=n_zern,
            wavelengthInNm=wavelengthInNm,
            pix_scale=pix_scale,
            npix_side=npixels,
            cropped_size=8,
            ifunc=ifunc_obj,
            n_iter=30,
            fft_res=1,
            target_device_idx=-1,
            precision=1,
        )

        _, coeffs_out, _ = lift.phaseEstimation(psf)
        coeffs_out = cpuArray(coeffs_out)

        np.testing.assert_allclose(
            coeffs_out[defocus_idx], unknown_nm,
            atol=20.0,
            err_msg=f"LIFT defocus estimate {coeffs_out[defocus_idx]:.1f} nm, "
                    f"expected {unknown_nm:.1f} nm",
        )
