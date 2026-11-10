import unittest

import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.lantern_cubic_model import LanternCubicModel
from specula.processing_objects.spectral_piston_unwrapper import SpectralPistonUnwrapper

from test.specula_testlib import cpu_and_gpu


WL = np.arange(1150.0, 1251.0, 10.0)      # 11 wavelengths, as in the reference fits
N_OUT = 3
SPAN = 2000.0


def make_model(target_device_idx=None, sigma=0.001, span=SPAN, conf_min=0.15):
    n_wl = len(WL)
    return LanternCubicModel(
        wavelengths_nm=WL,
        periods_nm=WL.copy(),
        n_terms=np.ones(n_wl, int),
        powers=np.zeros((n_wl, 19), dtype=np.int8),
        weights=np.zeros((n_wl, 2 * N_OUT)),
        offsets=np.zeros((n_wl, 2 * N_OUT)),
        input_mean=np.zeros((n_wl, 19)),
        input_std=np.ones((n_wl, 19)),
        sigma_nm=np.full((n_wl, N_OUT), sigma),
        span_nm=span,
        conf_min=conf_min,
        target_device_idx=target_device_idx)


def wrap(x, period):
    return x - period * np.round(x / period)


def make_inputs(xp, pistons, conf=None, extra_shift=None):
    '''Exact wrapped estimates of *pistons* (n_out,) at every wavelength'''
    pistons = np.asarray(pistons, float)
    ghat = wrap(pistons[:, None], WL[None, :])
    if extra_shift is not None:
        ghat = ghat + extra_shift
    if conf is None:
        conf = np.ones((len(pistons), len(WL)))
    wrapped = BaseValue(target_device_idx=-1 if xp is np else 0)
    wrapped.value = xp.asarray(ghat)
    wrapped.generation_time = 1
    confidence = BaseValue(target_device_idx=-1 if xp is np else 0)
    confidence.value = xp.asarray(np.asarray(conf, float))
    confidence.generation_time = 1
    return {'in_pistons_wrapped': wrapped, 'in_confidence': confidence}


def run(obj, inputs, t=1):
    for name, value in inputs.items():
        obj.inputs[name].set(value)
    obj.setup()
    obj.check_ready(t)
    obj.trigger()
    obj.post_trigger()


class TestSpectralPistonUnwrapper(unittest.TestCase):

    @cpu_and_gpu
    def test_exact_estimates_are_unwrapped(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        pistons = [0.0, 350.0, -1200.0]
        run(obj, make_inputs(xp, pistons))
        out = cpuArray(obj.outputs['out_pistons'].value)
        agreement = cpuArray(obj.outputs['out_agreement'].value)
        np.testing.assert_allclose(out, pistons, atol=1.0)
        np.testing.assert_allclose(agreement, 1.0, rtol=1e-6)

    @cpu_and_gpu
    def test_pistons_much_larger_than_a_wavelength(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        pistons = [1600.0, -1600.0, 900.0]
        run(obj, make_inputs(xp, pistons))
        np.testing.assert_allclose(cpuArray(obj.outputs['out_pistons'].value), pistons, atol=1.0)

    @cpu_and_gpu
    def test_one_bad_wavelength_is_rejected(self, target_device_idx, xp):
        # one wavelength carries a non-integer error (a wrong estimate, not a
        # wrong fringe): the robust refinement must not be dragged away
        model = make_model(target_device_idx=target_device_idx)
        obj = SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        pistons = [500.0, -800.0, 0.0]
        shift = np.zeros((N_OUT, len(WL)))
        shift[:, 3] = 400.0
        run(obj, make_inputs(xp, pistons, extra_shift=shift))
        out = cpuArray(obj.outputs['out_pistons'].value)
        np.testing.assert_allclose(out, pistons, atol=20.0)
        agreement = cpuArray(obj.outputs['out_agreement'].value)
        self.assertTrue(np.all(agreement < 1.0))

    @cpu_and_gpu
    def test_low_confidence_wavelengths_are_gated(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        pistons = [300.0, -300.0, 0.0]
        conf = np.ones((N_OUT, len(WL)))
        shift = np.zeros((N_OUT, len(WL)))
        # a badly wrong, low-confidence wavelength must not matter at all
        conf[:, 4] = 0.01
        shift[:, 4] = 400.0
        run(obj, make_inputs(xp, pistons, conf=conf, extra_shift=shift))
        np.testing.assert_allclose(cpuArray(obj.outputs['out_pistons'].value), pistons, atol=5.0)

    @cpu_and_gpu
    def test_all_confidence_zero_still_returns_an_estimate(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        pistons = [200.0, 0.0, -200.0]
        conf = np.zeros((N_OUT, len(WL)))
        run(obj, make_inputs(xp, pistons, conf=conf))
        np.testing.assert_allclose(cpuArray(obj.outputs['out_pistons'].value), pistons, atol=1.0)

    @cpu_and_gpu
    def test_capture_range_is_respected(self, target_device_idx, xp):
        # a piston outside the capture range is clipped to 1.5 * pspan
        model = make_model(target_device_idx=target_device_idx, span=200.0)
        obj = SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        pistons = [1500.0, 0.0, 0.0]
        run(obj, make_inputs(xp, pistons))
        out = cpuArray(obj.outputs['out_pistons'].value)
        self.assertLessEqual(abs(out[0]), 1.5 * 200.0 + 1e-3)
        # and the smaller override of the capture range wins
        obj = SpectralPistonUnwrapper(model, pspan=1800.0, target_device_idx=target_device_idx)
        run(obj, make_inputs(xp, pistons))
        np.testing.assert_allclose(cpuArray(obj.outputs['out_pistons'].value), pistons, atol=1.0)

    @cpu_and_gpu
    def test_shape_validation(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        inputs = make_inputs(xp, [0.0, 0.0, 0.0])
        inputs['in_confidence'].value = xp.asarray(np.ones((N_OUT, len(WL) - 1)))
        with self.assertRaises(ValueError):
            run(obj, inputs)

    @cpu_and_gpu
    def test_needs_a_capture_range(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx, span=None)
        with self.assertRaises(ValueError):
            SpectralPistonUnwrapper(model, target_device_idx=target_device_idx)
        SpectralPistonUnwrapper(model, pspan=1800.0, target_device_idx=target_device_idx)

    def test_names(self):
        for name in ('in_pistons_wrapped', 'in_confidence'):
            self.assertIn(name, SpectralPistonUnwrapper.input_names())
        for name in ('out_pistons', 'out_agreement'):
            self.assertIn(name, SpectralPistonUnwrapper.output_names())


if __name__ == '__main__':
    unittest.main()
