import unittest

import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.lantern_cubic_model import LanternCubicModel
from specula.processing_objects.lantern_piston_reconstructor import LanternPistonReconstructor

from test.specula_testlib import cpu_and_gpu


N_IN, N_OUT = 3, 1
WL = [1150.0, 1160.0]
PISTON = 40.0
PERIOD = 1150.0


def make_model(wavelengths=WL, target_device_idx=None):
    '''A model whose sin/cos targets are the exact phasors of a piston of 40 nm'''
    n_wl = len(wavelengths)
    theta = 2 * np.pi * PISTON / PERIOD
    return LanternCubicModel(
        wavelengths_nm=np.asarray(wavelengths, float),
        periods_nm=np.full(n_wl, PERIOD),
        n_terms=np.ones(n_wl, int),
        powers=np.zeros((n_wl, N_IN), dtype=np.int8),
        weights=np.tile(np.array([np.sin(theta), np.cos(theta)]), (n_wl, 1)),
        offsets=np.zeros((n_wl, 2 * N_OUT)),
        input_mean=np.zeros((n_wl, N_IN)),
        input_std=np.ones((n_wl, N_IN)),
        sigma_nm=np.full((n_wl, N_OUT), 1.0),
        span_nm=2000.0,
        target_device_idx=target_device_idx)


def make_signals(xp, values, wavelengths=None):
    signals = BaseValue(target_device_idx=-1 if xp is np else 0)
    signals.value = xp.asarray(np.asarray(values, float))
    if wavelengths is not None:
        signals.wavelengths_nm = [float(w) for w in wavelengths]
    signals.generation_time = 1
    return signals


def run(obj, inputs, t=1):
    for name, value in inputs.items():
        obj.inputs[name].set(value)
    obj.setup()
    obj.check_ready(t)
    obj.trigger()
    obj.post_trigger()


class TestLanternPistonReconstructor(unittest.TestCase):

    @cpu_and_gpu
    def test_pistons_and_confidence(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = LanternPistonReconstructor(model, target_device_idx=target_device_idx)
        signals = make_signals(xp, np.full((len(WL), N_IN), 0.01), wavelengths=WL)
        run(obj, {'in_signals': signals})
        pistons = cpuArray(obj.outputs['out_pistons_wrapped'].value)
        confidence = cpuArray(obj.outputs['out_confidence'].value)
        self.assertEqual(pistons.shape, (N_OUT, len(WL)))
        self.assertEqual(confidence.shape, (N_OUT, len(WL)))
        np.testing.assert_allclose(pistons, PISTON, rtol=1e-4)
        np.testing.assert_allclose(confidence, 1.0, rtol=1e-4)
        self.assertEqual(obj.outputs['out_pistons_wrapped'].generation_time, 1)

    @cpu_and_gpu
    def test_rows_are_matched_by_wavelength(self, target_device_idx, xp):
        # the signals carry three wavelengths, the model only two: rows 0 and 2 are used
        model = make_model(target_device_idx=target_device_idx)
        obj = LanternPistonReconstructor(model, target_device_idx=target_device_idx)
        obj.inputs['in_signals'].set(
            make_signals(xp, np.zeros((3, N_IN)), wavelengths=[1150.0, 1155.0, 1160.0]))
        obj.setup()
        self.assertEqual(obj._rows, [0, 2])

    @cpu_and_gpu
    def test_row_mismatch_raises(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = LanternPistonReconstructor(model, target_device_idx=target_device_idx)
        # no declared wavelengths and the wrong number of rows
        obj.inputs['in_signals'].set(make_signals(xp, np.zeros((3, N_IN))))
        with self.assertRaises(ValueError):
            obj.setup()
        # a model wavelength missing from the declared ones
        obj = LanternPistonReconstructor(model, target_device_idx=target_device_idx)
        obj.inputs['in_signals'].set(
            make_signals(xp, np.zeros((2, N_IN)), wavelengths=[1150.0, 1170.0]))
        with self.assertRaises(ValueError):
            obj.setup()

    @cpu_and_gpu
    def test_channel_mismatch_raises(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = LanternPistonReconstructor(model, target_device_idx=target_device_idx)
        obj.inputs['in_signals'].set(
            make_signals(xp, np.zeros((2, N_IN + 1)), wavelengths=WL))
        with self.assertRaises(ValueError):
            obj.setup()

    @cpu_and_gpu
    def test_1d_signals_raise(self, target_device_idx, xp):
        model = make_model(target_device_idx=target_device_idx)
        obj = LanternPistonReconstructor(model, target_device_idx=target_device_idx)
        obj.inputs['in_signals'].set(make_signals(xp, np.zeros(N_IN)))
        with self.assertRaises(ValueError):
            obj.setup()

    def test_names(self):
        self.assertIn('in_signals', LanternPistonReconstructor.input_names())
        for name in ('out_pistons_wrapped', 'out_confidence'):
            self.assertIn(name, LanternPistonReconstructor.output_names())


if __name__ == '__main__':
    unittest.main()
