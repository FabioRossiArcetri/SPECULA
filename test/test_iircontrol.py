

import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.iir_filter_data import IirFilterData
from specula.processing_objects.iir_filter import IirFilter
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestIirFilter(unittest.TestCase):
   
    # We just check that it goes through.
    @cpu_and_gpu
    def test_iir_filter_instantiation(self, target_device_idx, xp):
        iir_filter = IirFilterData(ordnum=(1,1), ordden=(1,1), num=xp.ones((2,2)), den=xp.ones((2,2)),
                                   target_device_idx=target_device_idx)
        simulParams = SimulParams(time_step=0.001)
        iir_control = IirFilter(simulParams, iir_filter)

