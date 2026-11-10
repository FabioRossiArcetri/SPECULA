import os
import unittest

import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.data_objects.lantern_cubic_model import LanternCubicModel

from test.specula_testlib import cpu_and_gpu


N_IN, N_OUT = 3, 1


def make_model(target_device_idx=None, n_wl=2, **kwargs):
    '''
    Two models with a different number of terms and a known, hand-checkable
    result: for wavelength 0 the monomials are [1, x0, x1**2] with weights
    [[1, 2], [3, 4], [5, 6]] and offset [0.5, -0.5]; for wavelength 1 a single
    constant term.
    '''
    n_terms = np.array([3, 1])
    powers = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [0, 2, 0],
                       [0, 0, 0]], dtype=np.int8)
    weights = np.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0]], dtype=np.float64)
    kw = dict(wavelengths_nm=[1150.0, 1160.0],
              periods_nm=[1150.0, 1160.0],
              n_terms=n_terms,
              powers=powers,
              weights=weights,
              offsets=np.array([[0.5, -0.5], [0.0, 0.0]]),
              input_mean=np.zeros((2, N_IN)),
              input_std=np.ones((2, N_IN)),
              sigma_nm=np.full((2, N_OUT), 3.5),
              target_device_idx=target_device_idx)
    kw.update(kwargs)
    return LanternCubicModel(**kw)


class TestLanternCubicModel(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_pl_cubic.fits')

    def tearDown(self):
        for f in (self.filename, self.filename.replace('.fits', '.npz')):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    @cpu_and_gpu
    def test_properties(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        self.assertEqual((model.n_wl, model.n_inputs, model.n_outputs), (2, N_IN, N_OUT))
        self.assertEqual(model.degree, 2)
        self.assertEqual(model.term_slice(0), (0, 3))
        self.assertEqual(model.term_slice(1), (3, 4))
        self.assertEqual(model.n_irls, 12)
        self.assertAlmostEqual(model.conf_min, 0.15)
        self.assertAlmostEqual(model.huber_c, 2.0)

    @cpu_and_gpu
    def test_predict_phasors_is_the_hand_computation(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        pred = cpuArray(model.predict_phasors(xp.asarray([0.3, 0.7, 0.0]), 0))
        # monomials [1, 0.3, 0.49]
        expected = np.array([1 + 3 * 0.3 + 5 * 0.49 + 0.5,
                             2 + 4 * 0.3 + 6 * 0.49 - 0.5])
        np.testing.assert_allclose(pred, expected, rtol=1e-6)

    @cpu_and_gpu
    def test_standardisation_is_applied(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx,
                           input_mean=np.array([[0.3, 0.7, 0.0], [0.0, 0.0, 0.0]]),
                           input_std=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
        pred = cpuArray(model.predict_phasors(xp.asarray([0.6, 1.4, 0.0]), 0))
        expected = np.array([1 + 3 * 0.3 + 5 * 0.49 + 0.5,
                             2 + 4 * 0.3 + 6 * 0.49 - 0.5])
        np.testing.assert_allclose(pred, expected, rtol=1e-6)

    @cpu_and_gpu
    def test_decode_phasors(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        # phasor of a piston of a quarter period
        period = 1150.0
        theta = np.pi / 2
        p, conf = model.decode_phasors(xp.asarray([np.sin(theta), np.cos(theta)]), 0)
        np.testing.assert_allclose(cpuArray(p), [period / 4], rtol=1e-6)
        np.testing.assert_allclose(cpuArray(conf), [1.0], rtol=1e-6)
        # a piston beyond the wrap gives the wrapped value back
        pistons, _ = model.decode_phasors(
            xp.asarray([np.sin(2 * np.pi * (period / 4 + period) / period),
                        np.cos(2 * np.pi * (period / 4 + period) / period)]), 0)
        np.testing.assert_allclose(cpuArray(pistons), [period / 4], rtol=1e-6)

    @cpu_and_gpu
    def test_predict_then_decode_recovers_a_piston(self, target_device_idx, xp):
        # a model whose sin/cos targets are exact for a piston of 40 nm
        period, piston = 1150.0, 40.0
        theta = 2 * np.pi * piston / period
        model = make_model(target_device_idx=target_device_idx,
                           n_terms=np.array([1, 1]),
                           powers=np.zeros((2, N_IN), dtype=np.int8),
                           weights=np.array([[np.sin(theta), np.cos(theta)],
                                             [0.0, 0.0]]),
                           offsets=np.zeros((2, 2)))
        phasors = model.predict_phasors(xp.zeros(N_IN), 0)
        p, conf = model.decode_phasors(phasors, 0)
        np.testing.assert_allclose(cpuArray(p), [piston], rtol=1e-5)
        np.testing.assert_allclose(cpuArray(conf), [1.0], rtol=1e-6)

    @cpu_and_gpu
    def test_input_shape_checks(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            model.predict_phasors(xp.zeros(N_IN + 1), 0)
        with self.assertRaises(ValueError):
            model.decode_phasors(xp.zeros(3), 0)

    @cpu_and_gpu
    def test_validation(self, target_device_idx, xp):
        base = dict(wavelengths_nm=[1150.0, 1160.0],
                    periods_nm=[1150.0, 1160.0],
                    n_terms=[1, 1],
                    powers=np.zeros((2, N_IN), dtype=np.int8),
                    weights=np.zeros((2, 2 * N_OUT)),
                    offsets=np.zeros((2, 2 * N_OUT)),
                    input_mean=np.zeros((2, N_IN)),
                    input_std=np.ones((2, N_IN)),
                    sigma_nm=np.zeros((2, N_OUT)),
                    target_device_idx=target_device_idx)
        LanternCubicModel(**base)  # sanity: the base case is valid

        with self.assertRaises(ValueError):   # term count mismatch
            LanternCubicModel(**{**base, 'n_terms': [1, 2]})
        with self.assertRaises(ValueError):   # no model
            LanternCubicModel(**{**base, 'wavelengths_nm': []})
        with self.assertRaises(ValueError):   # not increasing
            LanternCubicModel(**{**base, 'wavelengths_nm': [1160.0, 1150.0]})
        with self.assertRaises(ValueError):   # non-positive period
            LanternCubicModel(**{**base, 'periods_nm': [0.0, 1160.0]})
        with self.assertRaises(ValueError):   # odd number of weight columns
            LanternCubicModel(**{**base, 'weights': np.zeros((2, 3))})
        with self.assertRaises(ValueError):   # negative exponent
            LanternCubicModel(**{**base, 'powers': -np.ones((2, N_IN), dtype=np.int8)})
        with self.assertRaises(ValueError):   # zero standard deviation
            LanternCubicModel(**{**base, 'input_std': np.zeros((2, N_IN))})
        with self.assertRaises(ValueError):   # wrong offset shape
            LanternCubicModel(**{**base, 'offsets': np.zeros((2, 3))})
        with self.assertRaises(ValueError):   # one sigma per wavelength and piston
            LanternCubicModel(**{**base, 'sigma_nm': np.zeros((2, 2 * N_OUT))})

    @cpu_and_gpu
    def test_roundtrip_fits(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx, span_nm=2000.0,
                           conf_min=0.2, huber_c=2.5, n_irls=8, pl_name='pl_test')
        model.save(self.filename, overwrite=True)
        restored = LanternCubicModel.restore(self.filename, target_device_idx=target_device_idx)
        np.testing.assert_allclose(cpuArray(restored.wavelengths_nm), [1150.0, 1160.0])
        np.testing.assert_allclose(cpuArray(restored.periods_nm), [1150.0, 1160.0])
        np.testing.assert_array_equal(cpuArray(restored.n_terms), [3, 1])
        np.testing.assert_array_equal(cpuArray(restored.powers), cpuArray(model.powers))
        np.testing.assert_allclose(cpuArray(restored.weights), cpuArray(model.weights), rtol=1e-6)
        np.testing.assert_allclose(cpuArray(restored.offsets), cpuArray(model.offsets), rtol=1e-6)
        np.testing.assert_allclose(cpuArray(restored.sigma_nm), cpuArray(model.sigma_nm), rtol=1e-6)
        self.assertEqual(restored.powers.dtype, np.int8)
        self.assertAlmostEqual(restored.span_nm, 2000.0)
        self.assertAlmostEqual(restored.conf_min, 0.2)
        self.assertAlmostEqual(restored.huber_c, 2.5)
        self.assertEqual(restored.n_irls, 8)
        self.assertEqual(restored.pl_name, 'pl_test')

    @cpu_and_gpu
    def test_roundtrip_npz(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        filename = self.filename.replace('.fits', '.npz')
        model.save(filename, overwrite=True)
        restored = LanternCubicModel.restore(filename, target_device_idx=target_device_idx)
        np.testing.assert_allclose(cpuArray(restored.weights), cpuArray(model.weights), rtol=1e-6)

    @cpu_and_gpu
    def test_restore_checks_object_type(self, target_device_idx, xp):
        from astropy.io import fits
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Something else'
        fits.writeto(self.filename, np.zeros((2, 2)), hdr, overwrite=True)
        with self.assertRaises(ValueError):
            LanternCubicModel.restore(self.filename)

    @cpu_and_gpu
    def test_wavelength_index(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        self.assertEqual(model.wavelength_index(1150.0), 0)
        with self.assertRaises(ValueError):
            model.wavelength_index(1200.0)


if __name__ == '__main__':
    unittest.main()
