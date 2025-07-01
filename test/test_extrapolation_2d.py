import specula
specula.init(0)

import unittest

from specula import np
from specula import cpuArray
from specula.lib.extrapolation_2d import calculate_extrapolation_indices_coeffs, apply_extrapolation
from specula.lib.zernike_generator import ZernikeGenerator
from specula.lib.mask import CircularMask

from test.specula_testlib import cpu_and_gpu

class TestExtrapolation2D(unittest.TestCase):

    def _create_test_data(self, shape, outer_radius, inner_radius, zernike_mode, xp):
        """
        Create synthetic test data using Zernike polynomials

        Parameters:
        - shape: array dimensions
        - outer_radius: radius of the full circular mask (true data)
        - inner_radius: radius of the smaller mask (data to extrapolate from)
        - zernike_mode: which Zernike mode to use (1=piston, 2=tip, 3=tilt, etc.)

        Returns: true_data, input_data, full_mask, reduced_mask, annular_region
        """
        center = xp.array(shape) / 2.0
        full_mask = CircularMask(shape, outer_radius, center, xp=xp)
        reduced_mask = CircularMask(shape, inner_radius, center, xp=xp)

        zg = ZernikeGenerator(full_mask, xp=xp, dtype=xp.float64)
        true_data = zg.getZernike(zernike_mode)
        input_data = true_data.copy()

        full_mask = full_mask.mask()
        reduced_mask = reduced_mask.mask()

        input_data[reduced_mask] = 0  # Zero outside the reduced mask

        # Annular region: inside full_mask and outside reduced_mask
        annular_region = (~full_mask) & reduced_mask
        return true_data, input_data, full_mask, reduced_mask, annular_region

    @cpu_and_gpu
    def test_extrapolation_piston(self, target_device_idx, xp):
        """
        Test extrapolation with piston (constant) - should be perfect
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=1, xp=xp)

        # Calculate extrapolation coefficients using the reduced mask
        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
            ~reduced_mask)  # Invert mask: True=valid, False=invalid

        # Apply extrapolation
        result = apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp=xp
        )

        # Check that extrapolated values match the true data in the annular region
        np.testing.assert_array_almost_equal(
            cpuArray(result[annular_region]),
            cpuArray(true_data[annular_region]),
            decimal=10,
            err_msg="Piston extrapolation should be perfect"
        )

    @cpu_and_gpu 
    def test_extrapolation_tip(self, target_device_idx, xp):
        """
        Test extrapolation with tip (linear in X) - should be very good
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=2, xp=xp)

        # Calculate extrapolation coefficients
        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
            ~reduced_mask)

        # Apply extrapolation
        result = apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp=xp
        )

        # Check extrapolated values (tip is linear, so extrapolation should be very accurate)
        np.testing.assert_array_almost_equal(
            cpuArray(result[annular_region]),
            cpuArray(true_data[annular_region]),
            decimal=5,
            err_msg="Tip extrapolation should be very accurate for linear function"
        )

    @cpu_and_gpu
    def test_extrapolation_tilt(self, target_device_idx, xp):
        """
        Test extrapolation with tilt (linear in Y) - should be very good
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=3, xp=xp)

        # Calculate extrapolation coefficients
        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
            ~reduced_mask)

        # Apply extrapolation
        result = apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp=xp
        )

        # Check extrapolated values
        np.testing.assert_array_almost_equal(
            cpuArray(result[annular_region]),
            cpuArray(true_data[annular_region]),
            decimal=5,
            err_msg="Tilt extrapolation should be very accurate for linear function"
        )

    @cpu_and_gpu
    def test_extrapolation_focus(self, target_device_idx, xp):
        """
        Test extrapolation with focus (quadratic) - should be reasonable
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=4, xp=xp)

        # Calculate extrapolation coefficients
        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
            ~reduced_mask)

        # Apply extrapolation
        result = apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp=xp
        )

        # Check extrapolated values (focus is quadratic, so less accurate but should be reasonable)
        # Use RMS error instead of point-by-point comparison for quadratic
        rms_error = xp.sqrt(xp.mean((result[annular_region] - true_data[annular_region])**2))
        true_rms = xp.sqrt(xp.mean(true_data[annular_region]**2))
        relative_error = rms_error / true_rms

        self.assertLess(relative_error, 0.1,
                       f"Focus extrapolation relative RMS error {relative_error:.3f} should be < 10%")

    @cpu_and_gpu
    def test_extrapolation_preserves_input(self, target_device_idx, xp):
        """
        Test that apply_extrapolation doesn't modify values inside the original mask
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=8, zernike_mode=2, xp=xp)

        # Store original values inside the reduced mask
        original_inside = input_data[~reduced_mask].copy()

        # Calculate and apply extrapolation
        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(
            ~reduced_mask)
        result = apply_extrapolation(input_data, edge_pixels, reference_indices, coefficients)

        # Check that values inside the original mask are unchanged
        np.testing.assert_array_equal(
            cpuArray(result[~reduced_mask]),
            cpuArray(original_inside),
            err_msg="Values inside the input mask should not be modified"
        )

    @cpu_and_gpu
    def test_calculate_extrapolation_indices_coeffs_output_shape(self, target_device_idx, xp):
        """
        Test that the calculation function returns arrays with expected shapes
        """
        # Create a simple test mask
        mask = xp.ones((20, 20), dtype=bool)
        mask[5:15, 5:15] = False  # 10x10 inner region is valid

        edge_pixels, reference_indices, coefficients = calculate_extrapolation_indices_coeffs(mask)

        # Check shapes are consistent
        n_max = len(edge_pixels)
        self.assertEqual(reference_indices.shape, (n_max, 8))
        self.assertEqual(coefficients.shape, (n_max, 8))

        # Check that we have some valid edge pixels
        valid_pixels = xp.sum(edge_pixels >= 0)
        self.assertGreater(valid_pixels, 0)

if __name__ == '__main__':
    unittest.main()
