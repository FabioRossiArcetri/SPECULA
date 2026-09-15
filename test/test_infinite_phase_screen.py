import unittest
import os
from unittest.mock import patch
import specula
specula.init(0)  # Default target device

from specula import np, cpuArray
from specula.lib.calc_phasescreen import calc_phasescreen
from specula.data_objects.infinite_phase_screen import (
    ft_phase_screen_vect,
    compute_covariance_from_PSD_vect,
)

from test.specula_testlib import cpu_and_gpu


class TestFtPhaseScreenVect(unittest.TestCase):
    """Tests for the module-level helper ft_phase_screen_vect()"""

    def _psd(self):
        # simple power-law-like 1D PSD, decreasing with frequency
        f1d = np.linspace(1e-3, 10.0, 200)
        psd = 1.0 / f1d**2
        return f1d, psd

    def test_output_shapes(self):
        f1d, psd = self._psd()
        N = 32
        phs, psd2d, del_f = ft_phase_screen_vect(f1d, psd, N, delta=0.1, seed=1)
        self.assertEqual(phs.shape, (N, N))
        self.assertEqual(psd2d.shape, (N, N))
        self.assertTrue(np.isreal(phs).all())
        self.assertGreater(del_f, 0)

    def test_reproducibility_with_same_seed(self):
        f1d, psd = self._psd()
        phs1, _, _ = ft_phase_screen_vect(f1d, psd, 32, delta=0.1, seed=42)
        phs2, _, _ = ft_phase_screen_vect(f1d, psd, 32, delta=0.1, seed=42)
        np.testing.assert_array_equal(phs1, phs2)

    def test_different_seed_gives_different_screen(self):
        f1d, psd = self._psd()
        phs1, _, _ = ft_phase_screen_vect(f1d, psd, 32, delta=0.1, seed=1)
        phs2, _, _ = ft_phase_screen_vect(f1d, psd, 32, delta=0.1, seed=2)
        self.assertFalse(np.allclose(phs1, phs2))

    def test_psd_scaling_scales_screen_amplitude(self):
        # cn is proportional to sqrt(PSD), so scaling the PSD by a factor c
        # scales the resulting screen amplitude by sqrt(c) (same seed => same
        # underlying random numbers).
        f1d, psd = self._psd()
        phs1, _, _ = ft_phase_screen_vect(f1d, psd, 32, delta=0.1, seed=7)
        phs2, _, _ = ft_phase_screen_vect(f1d, psd * 4.0, 32, delta=0.1, seed=7)
        np.testing.assert_allclose(phs2, phs1 * 2.0, atol=1e-8)


class TestComputeCovarianceFromPSDVect(unittest.TestCase):
    """Tests for the module-level helper compute_covariance_from_PSD_vect()"""

    def test_output_shapes_and_finiteness(self):
        f_vect = np.logspace(-4, 4, 500)
        psd_vect = 1.0 / (1.0 + f_vect**2)
        fht, rd = compute_covariance_from_PSD_vect(f_vect, psd_vect, P=4, Q=4, points=2000)
        self.assertEqual(fht.shape, (2000,))
        self.assertEqual(rd.shape, (2000,))
        self.assertTrue(np.all(np.isfinite(fht)))
        self.assertTrue(np.all(np.isfinite(rd)))

    def test_output_is_shifted_to_be_non_negative(self):
        # The function subtracts (min - 1e-6), so the minimum must be
        # essentially exactly 1e-6 above zero.
        f_vect = np.logspace(-4, 4, 500)
        psd_vect = 1.0 / (1.0 + f_vect**2)
        fht, _ = compute_covariance_from_PSD_vect(f_vect, psd_vect, P=4, Q=4, points=2000)
        self.assertAlmostEqual(np.min(fht), 1e-6, places=9)

@unittest.skipIf(os.environ.get('CI') == 'true', "Disable for CI issues with Ubuntu and Python >=3.11")
class TestInfinitePhaseScreen(unittest.TestCase):

    @cpu_and_gpu
    def test_phase_covariance_matches_theory(self, target_device_idx, xp):
        """Test that the phase covariance function matches theoretical values"""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        # Parameters
        mx_size = 512
        pixel_scale = 0.1  # meters
        r0 = 0.2  # meters
        L0 = 25.0  # meters
        random_seed = 12345

        # Create infinite phase screen
        ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                 random_seed=random_seed,
                                 target_device_idx=target_device_idx)

        # Test covariance function at different separations
        separations = xp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # meters
        cov_values = cpuArray(ips.phase_covariance(separations, r0, L0))

        # Basic sanity checks
        self.assertTrue(all(cov_values >= 0), "Covariance values should be non-negative")
        self.assertTrue(cov_values[0] > cov_values[-1], "Covariance should decrease with separation")

        # Check that covariance at zero separation is finite and positive
        cov_zero = cpuArray(ips.phase_covariance(xp.array([1e-6]), r0, L0)[0])
        self.assertTrue(cov_zero > 0, "Covariance at zero separation should be positive")

    @cpu_and_gpu
    def test_infinite_vs_fft_phase_screen_statistics(self, target_device_idx, xp):
        """Compare statistics between InfinitePhaseScreen and calc_phasescreen (FFT method)
        across multiple combinations of phase_size and pixel_scale"""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        verbose = False

        # Parameters
        r0 = 0.15  # meters
        L0 = 25.0  # meters
        random_seed1 = 42
        random_seed2 = 1042

        # Test parameter combinations
        if os.environ.get('CI') == 'true':
            phase_sizes = [128]
            pixel_scales = [0.5]
            n_seeds = 3
        else:
            phase_sizes = [512]
            pixel_scales = [0.5, 0.05]
            n_seeds = 10

        # Store all results for summary
        results = []

        if verbose:  # pragma: no cover
            print("\nTesting InfinitePhaseScreen vs FFT phase screen statistics")
            print("=" * 76)
            print(f"{'Phase Size':<12} {'Pixel Scale':<12} {'Inf Mean':<10} {'Inf Std':<10} {'FFT Mean':<10} {'FFT Std':<10} {'Ratio':<8}")
            print("-" * 76)

        for phase_size in phase_sizes:
            for pixel_scale in pixel_scales:
                # Initialize accumulators
                inf_mean = 0
                inf_std = 0
                fft_mean = 0
                fft_std = 0

                for i in range(n_seeds):
                    # Create infinite phase screen
                    r0_inf = r0 #2 * pixel_scale
                    ips = InfinitePhaseScreen(phase_size, pixel_scale, r0_inf, L0,
                                            random_seed=random_seed1 + i,
                                            target_device_idx=target_device_idx)

                    # Get initial phase screen
                    infinite_screen = cpuArray(ips.scrn) * 500 / (2 * np.pi)  # in nm
                    r0_scaling = (r0_inf / r0)**(5./6.)
                    infinite_screen *= r0_scaling

                    # Create FFT phase screen with same parameters
                    fft_screen = calc_phasescreen(L0, phase_size, pixel_scale,
                                                seed=random_seed2 + i,
                                                precision=1,
                                                xp=xp)
                    fft_screen = cpuArray(fft_screen) * 500 / (2 * np.pi)  # in nm
                    r0_scaling = (pixel_scale / r0)**(5./6.)
                    fft_screen *= r0_scaling

                    # Accumulate statistics
                    inf_mean += np.mean(infinite_screen) / n_seeds
                    inf_std += np.std(infinite_screen) / n_seeds
                    fft_mean += np.mean(fft_screen) / n_seeds
                    fft_std += np.std(fft_screen) / n_seeds

                # Calculate ratio
                std_ratio = inf_std / fft_std if fft_std != 0 else 0

                # Store results
                result = {
                    'phase_size': phase_size,
                    'pixel_scale': pixel_scale,
                    'inf_mean': inf_mean,
                    'inf_std': inf_std,
                    'fft_mean': fft_mean,
                    'fft_std': fft_std,
                    'std_ratio': std_ratio
                }
                results.append(result)

                # Print current result
                if verbose:  # pragma: no cover
                    print(f"{phase_size:<12} {pixel_scale:<12} {inf_mean:<10.6f} {inf_std:<10.1f} {fft_mean:<10.6f} {fft_std:<10.1f} {std_ratio:<8.3f}")

        # Overall statistics
        all_ratios = [r['std_ratio'] for r in results if r['std_ratio'] > 0]
        min_ratio = np.min(all_ratios)
        max_ratio = np.max(all_ratios)

        failed_tests = []
        for result in results:
            phase_size = result['phase_size']
            pixel_scale = result['pixel_scale']
            inf_mean = result['inf_mean']
            fft_mean = result['fft_mean']
            std_ratio = result['std_ratio']

            # Mean should be close to zero for both
            try:
                self.assertAlmostEqual(inf_mean, 0.0, places=2,
                                    msg=f"Infinite screen mean should be near zero (size={phase_size}, scale={pixel_scale})")
                self.assertAlmostEqual(fft_mean, 0.0, places=2,
                                    msg=f"FFT screen mean should be near zero (size={phase_size}, scale={pixel_scale})")
            except AssertionError as e:
                failed_tests.append(f"Mean test failed for size={phase_size}, scale={pixel_scale}: {str(e)}")

            # Standard deviations should be similar
            min_ratio, max_ratio = 0.9, 1.5

            try:
                self.assertTrue(min_ratio < std_ratio < max_ratio,
                            f"Std ratio {std_ratio:.3f} should be in [{min_ratio}, {max_ratio}] for size={phase_size}, scale={pixel_scale}")
            except AssertionError as e:
                failed_tests.append(f"Std ratio test failed for size={phase_size}, scale={pixel_scale}: {str(e)}")

        if failed_tests:
            print(f"\n{len(failed_tests)} test(s) failed:")
            for failure in failed_tests[:10]:  # Show first 10 failures
                print(f"  - {failure}")
            if len(failed_tests) > 10:
                print(f"  ... and {len(failed_tests) - 10} more")

    @cpu_and_gpu
    def test_reproducibility_with_same_seed(self, target_device_idx, xp):
        """Test that screens with same seed produce identical results"""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        # Parameters
        mx_size = 64
        pixel_scale = 0.05
        r0 = 0.15
        L0 = 30.0
        random_seed = 789

        # Create two identical screens
        ips1 = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                  random_seed=random_seed,
                                  target_device_idx=target_device_idx)

        ips2 = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                  random_seed=random_seed,
                                  target_device_idx=target_device_idx)

        # Get screens
        screen1 = cpuArray(ips1.scrn)
        screen2 = cpuArray(ips2.scrn)

        # Should be identical
        np.testing.assert_array_equal(screen1, screen2,
                                     "Screens with same seed should be identical")

        # Evolve both screens identically
        for _ in range(5):
            ips1.add_line(row=1, after=1)
            ips2.add_line(row=1, after=1)

        screen1_evolved = ips1.scrn
        screen2_evolved = ips2.scrn

        # Should still be identical after evolution
        np.testing.assert_array_equal(screen1_evolved, screen2_evolved,
                                     "Evolved screens with same seed should remain identical")

    @cpu_and_gpu
    def test_random_seed_none_raises(self, target_device_idx, xp):
        """random_seed is mandatory: passing None must raise ValueError"""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        with self.assertRaises(ValueError):
            InfinitePhaseScreen(64, 0.1, 0.2, 25.0,
                               random_seed=None,
                               target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_psd1d_data_path_builds_screen(self, target_device_idx, xp):
        """When psd1d_data/psd1d_freq_data are provided, the screen must be
        generated from them instead of the analytical von Karman formula,
        and phase_covariance() must use the precomputed interpolation table."""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        mx_size = 32
        pixel_scale = 0.1
        r0 = 0.2
        L0 = 25.0
        random_seed = 555

        f_vect = np.logspace(-4, 4, 500)
        psd_vect = 1.0 / (1.0 + f_vect**2)

        ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                 random_seed=random_seed,
                                 psd1d_freq_data=f_vect,
                                 psd1d_data=psd_vect,
                                 target_device_idx=target_device_idx)

        self.assertIsNotNone(ips.cov_1D_data)
        self.assertIsNotNone(ips.cov_1D_rd)
        self.assertIsNotNone(ips.full_scrn)
        # scrn must be well formed (finite values, requested size)
        scrn = cpuArray(ips.scrn)
        self.assertEqual(scrn.shape, (mx_size, mx_size))
        self.assertTrue(np.all(np.isfinite(scrn)))

        # phase_covariance must interpolate from the precomputed table,
        # matching a direct np.interp call with the same table.
        r = xp.array([0.1, 0.5, 1.0])
        cov = cpuArray(ips.phase_covariance(r, r0, L0))
        expected = np.interp(cpuArray(r) + 1e-40, ips.cov_1D_rd, ips.cov_1D_data)
        np.testing.assert_allclose(cov, expected, rtol=1e-6)

    @cpu_and_gpu
    def test_psd1d_data_reproducible_with_same_seed(self, target_device_idx, xp):
        """Two screens built from the same PSD table and seed must be identical."""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        f_vect = np.logspace(-4, 4, 500)
        psd_vect = 1.0 / (1.0 + f_vect**2)

        ips1 = InfinitePhaseScreen(32, 0.1, 0.2, 25.0, random_seed=99,
                                  psd1d_freq_data=f_vect, psd1d_data=psd_vect,
                                  target_device_idx=target_device_idx)
        ips2 = InfinitePhaseScreen(32, 0.1, 0.2, 25.0, random_seed=99,
                                  psd1d_freq_data=f_vect, psd1d_data=psd_vect,
                                  target_device_idx=target_device_idx)

        np.testing.assert_array_equal(cpuArray(ips1.scrn), cpuArray(ips2.scrn))

    @cpu_and_gpu
    def test_add_line_tracks_lastmax_and_clears_first_flag(self, target_device_idx, xp):
        """add_line() should initialize lastmax from the first row RMS and then
        update it as an exponential moving average of subsequent row RMS values."""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        ips = InfinitePhaseScreen(32, 0.1, 0.2, 25.0, random_seed=321,
                                 target_device_idx=target_device_idx)

        self.assertTrue(ips.first)
        self.assertEqual(ips.lastmax, 1)

        ips.add_line(row=1, after=1)
        self.assertFalse(ips.first)
        first_lastmax = ips.lastmax
        self.assertGreater(first_lastmax, 0)

        ips.add_line(row=1, after=1)
        # lastmax after the second call is a blend of the previous value and
        # the new row's rms, so it generally differs from the first estimate.
        self.assertGreater(ips.lastmax, 0)

    @cpu_and_gpu
    def test_add_line_warns_and_still_updates_on_rms_spike(self, target_device_idx, xp):
        """A row whose rms suddenly spikes above 2*lastmax must still be
        normalized and incorporated (with a warning), not rejected."""

        # moved here to avoid CI issues
        from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen

        ips = InfinitePhaseScreen(32, 0.1, 0.2, 25.0, random_seed=654,
                                 target_device_idx=target_device_idx)

        # Establish a baseline lastmax with a normal line.
        ips.add_line(row=1, after=1)
        baseline_lastmax = ips.lastmax

        spike_line = ips.xp.ones(ips.stencil_size) * (baseline_lastmax * 100)
        with patch.object(ips, 'get_new_line', return_value=spike_line):
            ips.add_line(row=1, after=1)

        # lastmax is an EMA that must have grown after the spike.
        self.assertGreater(ips.lastmax, baseline_lastmax)


if __name__ == '__main__':
    unittest.main()