import os
import unittest

import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.data_objects.lantern_closed_form_model import LanternClosedFormModel

from test.specula_testlib import cpu_and_gpu


N_CORES, N_PETALS = 7, 6
PERIOD = 1200.0
WL = [1100.0, 1200.0]


def make_model(n_wl=2, twin=None, target_device_idx=None, **kwargs):
    '''
    A closed-form model built from exact petal amplitudes h: the resulting
    ``a``/``W`` reproduce |sum_j h_cj z_j|**2 exactly, so the inversion can be
    checked against the pistons the signals were made with.
    '''
    rng = np.random.default_rng(0)
    h = rng.normal(size=(n_wl, N_CORES, N_PETALS)) + 1j * rng.normal(size=(n_wl, N_CORES, N_PETALS))
    pairs = [(j, k) for j in range(N_PETALS) for k in range(j + 1, N_PETALS)]
    a = np.sum(np.abs(h) ** 2, axis=2)
    wmat = np.zeros((n_wl, N_CORES, 2 * len(pairs)))
    for p, (j, k) in enumerate(pairs):
        prod = h[:, :, j] * np.conj(h[:, :, k])
        wmat[:, :, p] = 2 * prod.real
        wmat[:, :, len(pairs) + p] = -2 * prod.imag
    kw = dict(wavelengths_nm=np.asarray(WL[:n_wl]),
              periods_nm=np.asarray(WL[:n_wl]),
              cf_a=a, cf_w=wmat, cf_q=np.zeros((n_wl, 2 * len(pairs))),
              twin=twin,
              input_mean=np.zeros((n_wl, N_CORES)),
              input_std=np.ones((n_wl, N_CORES)),
              sigma_nm=np.full((n_wl, N_PETALS - 1), 1.0),
              span_nm=4800.0, n_grid=32, steps=5, n_proj=5,
              target_device_idx=target_device_idx)
    kw.update(kwargs)
    return LanternClosedFormModel(**kw), h


def true_signals(h, pistons_w, wl):
    '''|sum_j h_j z_j|**2 with z_j = exp(i 2 pi p_j / lambda), ref petal at 0'''
    z = np.exp(2j * np.pi * np.append(pistons_w, 0.0) / wl)
    return np.abs(np.sum(h * z[None, :], axis=1)) ** 2


class TestLanternClosedFormModel(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_lantern_closed_form.fits')

    def tearDown(self):
        for f in (self.filename, self.filename.replace('.fits', '.npz')):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    @cpu_and_gpu
    def test_forward_evaluation_matches_the_quadratic_form(self, target_device_idx, xp):
        model, h = make_model(target_device_idx=target_device_idx)
        pistons = np.array([100.0, -350.0, 400.0, 250.0, -120.0])
        evaluated = np.asarray(cpuArray(model._evaluate(xp.asarray(pistons[None, :]), 0)))[0]
        expected = true_signals(h[0], pistons, WL[0])
        np.testing.assert_allclose(evaluated, expected, rtol=1e-2, atol=1e-6)

    @cpu_and_gpu
    def test_jacobian_matches_finite_differences(self, target_device_idx, xp):
        model, _ = make_model(target_device_idx=target_device_idx)
        pistons = np.array([120.0, -280.0, 350.0, 200.0, -90.0])
        jac = np.asarray(cpuArray(model._jacobian(xp.asarray(pistons[None, :]), 0)))[0]
        step = 0.5
        for i in range(N_PETALS - 1):
            up = pistons.copy(); up[i] += step
            down = pistons.copy(); down[i] -= step
            numeric = (np.asarray(cpuArray(model._evaluate(xp.asarray(up[None, :]), 0)))[0]
                       - np.asarray(cpuArray(model._evaluate(xp.asarray(down[None, :]), 0)))[0]) / (2 * step)
            np.testing.assert_allclose(jac[:, i], numeric, rtol=1e-4, atol=1e-6)

    @cpu_and_gpu
    def test_inversion_recovers_the_pistons(self, target_device_idx, xp):
        # the inversion is PLREC's forward.invert_fixed: spectral start, second
        # eigenvector and a fixed Sobol grid, then Gauss-Newton; on a well behaved
        # case it must recover the pistons the signals were made with
        model, h = make_model(target_device_idx=target_device_idx)
        pistons = np.array([100.0, -350.0, 400.0, 250.0, -120.0])
        signals = true_signals(h[0], pistons, WL[0])
        estimate, confidence = model.predict_pistons(xp.asarray(signals), 0)
        estimate = np.asarray(cpuArray(estimate))
        e = estimate - pistons
        e = e - WL[0] * np.round(e / WL[0])
        np.testing.assert_allclose(e, 0.0, atol=10.0)  # the synthetic model is nearly degenerate
        # (bit-identical to PLREC forward.invert_fixed; the end-to-end comparison
        # on the real characterization is what measures the accuracy)
        np.testing.assert_allclose(np.asarray(cpuArray(confidence)), 1.0)
        # the wrapped estimate is in [-period/2, period/2)
        self.assertTrue(np.all(np.abs(estimate) <= WL[0] / 2 + 1e-6))

    @cpu_and_gpu
    def test_inversion_with_a_twin_start(self, target_device_idx, xp):
        twin = np.eye(N_PETALS - 1)[::-1]           # any involution
        model, h = make_model(twin=twin, target_device_idx=target_device_idx, n_grid=32)
        pistons = np.array([300.0, -200.0, 100.0, -400.0, 50.0])
        signals = true_signals(h[1], pistons, WL[1])
        estimate = np.asarray(cpuArray(model.predict_pistons(xp.asarray(signals), 1)[0]))
        residual = np.asarray(cpuArray(model._evaluate(estimate[None, :], 1)))[0]
        np.testing.assert_allclose(residual, signals, rtol=1e-2, atol=1e-6)

    @cpu_and_gpu
    def test_interface(self, target_device_idx, xp):
        model, h = make_model(target_device_idx=target_device_idx)
        self.assertEqual((model.n_wl, model.n_inputs, model.n_outputs), (2, N_CORES, N_PETALS - 1))
        self.assertEqual(model.wavelength_index(1100.0), 0)
        with self.assertRaises(ValueError):
            model.wavelength_index(1300.0)
        with self.assertRaises(ValueError):     # wrong number of signals
            model.predict_pistons(xp.zeros(N_CORES + 1), 0)

    @cpu_and_gpu
    def test_validation(self, target_device_idx, xp):
        model, _ = make_model()
        a = np.asarray(cpuArray(model.cf_a))
        w = np.asarray(cpuArray(model.cf_w))
        q = np.asarray(cpuArray(model.cf_q))
        base = dict(wavelengths_nm=WL, periods_nm=np.full(2, PERIOD), cf_a=a, cf_w=w, cf_q=q,
                    input_mean=np.zeros((2, N_CORES)), input_std=np.ones((2, N_CORES)),
                    sigma_nm=np.full((2, N_PETALS - 1), 1.0),
                    target_device_idx=target_device_idx)
        LanternClosedFormModel(**base)          # sanity: the base case is valid
        with self.assertRaises(ValueError):     # wrong core count
            LanternClosedFormModel(**{**base, 'cf_a': a[:, :3]})
        with self.assertRaises(ValueError):     # wrong feature count
            LanternClosedFormModel(**{**base, 'cf_w': w[:, :, :10]})
        with self.assertRaises(ValueError):     # wrong twin shape
            LanternClosedFormModel(**{**base, 'twin': np.eye(3)})
        with self.assertRaises(ValueError):     # negative iterations
            LanternClosedFormModel(**{**base, 'steps': -1})

    @cpu_and_gpu
    def test_roundtrip_npz(self, target_device_idx, xp):
        model, _ = make_model(target_device_idx=target_device_idx,
                              twin=np.eye(N_PETALS - 1)[::-1],
                              n_grid=32, steps=5, n_proj=4, second=False, span_nm=2000.0)
        filename = self.filename.replace('.fits', '.npz')
        model.save(filename, overwrite=True)
        restored = LanternClosedFormModel.restore(filename, target_device_idx=target_device_idx)
        for name in ('cf_a', 'cf_w', 'cf_q', 'twin', 'input_mean', 'input_std', 'sigma_nm'):
            np.testing.assert_allclose(cpuArray(getattr(restored, name)),
                                       cpuArray(getattr(model, name)), rtol=1e-6)
        self.assertEqual((restored.n_grid, restored.steps, restored.n_proj), (32, 5, 4))
        self.assertFalse(restored.second)
        self.assertAlmostEqual(restored.span_nm, 2000.0)
        # and it still inverts
        _, h = make_model()
        pistons = np.array([120.0, -80.0, 200.0, -150.0, 60.0])
        signals = true_signals(h[0], pistons, WL[0])
        estimate = np.asarray(cpuArray(restored.predict_pistons(xp.asarray(signals), 0)[0]))
        residual = np.asarray(cpuArray(restored._evaluate(estimate[None, :], 0)))[0]
        np.testing.assert_allclose(residual, signals, rtol=1e-2, atol=1e-6)

    @cpu_and_gpu
    def test_restore_checks_object_type(self, target_device_idx, xp):
        from astropy.io import fits
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Something else'
        fits.writeto(self.filename, np.zeros((2, 2)), hdr, overwrite=True)
        with self.assertRaises(ValueError):
            LanternClosedFormModel.restore(self.filename)


if __name__ == '__main__':
    unittest.main()
