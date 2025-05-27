

import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.time_history import TimeHistory
from specula.processing_objects.func_generator import FuncGenerator

from test.specula_testlib import cpu_and_gpu

class TestFuncGenerator(unittest.TestCase):

    @cpu_and_gpu
    def test_func_generator_constant(self, target_device_idx, xp):
        constant = [4,3]
        f = FuncGenerator('SIN', target_device_idx=target_device_idx, constant=constant)
        f.check_ready(1)
        f.trigger()
        f.post_trigger()
        value = cpuArray(f.outputs['output'].value)
        np.testing.assert_allclose(value, constant)

    @cpu_and_gpu
    def test_func_generator_sin(self, target_device_idx, xp):
        amp = 1
        freq = 2
        offset = 3
        constant = 4
        f = FuncGenerator('SIN', target_device_idx=target_device_idx, amp=amp, freq=freq, offset=offset, constant=constant)
        f.setup()

        # Test twice in order to test streams capture, if enabled
        for t in [f.seconds_to_t(x) for x in [0.1, 0.2, 0.3]]:
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            value = cpuArray(f.outputs['output'].value)
            np.testing.assert_almost_equal(value, amp * np.sin(freq*2 * np.pi*f.t_to_seconds(t) + offset) + constant)
        

    @cpu_and_gpu
    def test_func_generator_time_history(self, target_device_idx, xp):
        data = xp.arange(12).reshape((3,4))
        time_hist = TimeHistory(data, target_device_idx=target_device_idx)

        f = FuncGenerator('TIME_HIST', target_device_idx=target_device_idx, time_hist=time_hist)
        f.check_ready(1)
        f.trigger()
        f.post_trigger()
        value = f.outputs['output'].value
        np.testing.assert_allclose(cpuArray(value), cpuArray(data[0]))

        # Second iteration
        f.check_ready(2)
        f.trigger()
        f.post_trigger()
        value = f.outputs['output'].value
        np.testing.assert_allclose(cpuArray(value), cpuArray(data[1]))

