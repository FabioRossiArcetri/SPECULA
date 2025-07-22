
import specula
specula.init(0)  # Default target device

import tempfile
import os
import gc
import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.electric_field_combinator import ElectricFieldCombinator

from test.specula_testlib import cpu_and_gpu

class TestElectricField(unittest.TestCase):

    @cpu_and_gpu
    def test_reset_does_not_reallocate(self, target_device_idx, xp):

        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)

        id_A_before = id(ef.A)
        id_p_before = id(ef.phaseInNm)

        ef.reset()

        id_A_after = id(ef.A)
        id_p_after = id(ef.phaseInNm)

        assert id_A_before == id_A_after
        assert id_p_before == id_p_after

    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):

        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)

        id_A_before = id(ef.A)
        id_p_before = id(ef.phaseInNm)

        ef.set_value([xp.ones(100).reshape(10,10), xp.zeros(100).reshape(10,10)])

        id_A_after = id(ef.A)
        id_p_after = id(ef.phaseInNm)

        assert id_A_before == id_A_after
        assert id_p_before == id_p_after

    @cpu_and_gpu
    def test_ef_combinator(self, target_device_idx, xp):
        pixel_pitch = 0.1
        pixel_pupil = 10
        simulParams = SimulParams(pixel_pupil=pixel_pupil,pixel_pitch=pixel_pitch)
        ef1 = ElectricField(pixel_pupil,pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef2 = ElectricField(pixel_pupil,pixel_pupil, pixel_pitch, S0=2, target_device_idx=target_device_idx)

        A1 = xp.ones((pixel_pupil, pixel_pupil))
        ef1.A = A1
        ef1.phaseInNm = 1 * xp.ones((pixel_pupil, pixel_pupil))
        A2 = xp.ones((pixel_pupil, pixel_pupil))
        A2[0, 0] = 0
        A2[9, 9] = 0
        ef2.A = A2
        ef2.phaseInNm = 3 * xp.ones((pixel_pupil, pixel_pupil))

        ef_combinator = ElectricFieldCombinator(
            simul_params=simulParams,
            target_device_idx=target_device_idx
        )

        ef_combinator.inputs['in_ef1'].set(ef1)
        ef_combinator.inputs['in_ef2'].set(ef2)

        t = 1
        ef1.generation_time = t
        ef2.generation_time = t

        ef_combinator.check_ready(t)
        ef_combinator.setup()
        ef_combinator.trigger()
        ef_combinator.post_trigger()

        out_ef = ef_combinator.outputs['out_ef']

        assert np.allclose(out_ef.A, ef1.A * ef2.A)
        assert np.allclose(out_ef.phaseInNm, ef1.phaseInNm + ef2.phaseInNm)
        assert np.allclose(out_ef.S0, ef1.S0 + ef2.S0)

    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        S0 = 1.23

        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=S0, target_device_idx=target_device_idx)
        ef.A = xp.arange(pixel_pupil * pixel_pupil, dtype=ef.dtype).reshape(pixel_pupil, pixel_pupil)
        ef.phaseInNm = xp.arange(pixel_pupil * pixel_pupil, dtype=ef.dtype).reshape(pixel_pupil, pixel_pupil) * 0.5

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "ef_test.fits")
            ef.save(filename)

            # Restore from file
            ef2 = ElectricField.restore(filename, target_device_idx=target_device_idx)

            # Check that the restored object has the data as expected
            assert np.allclose(cpuArray(ef.A), cpuArray(ef2.A))
            assert np.allclose(cpuArray(ef.phaseInNm), cpuArray(ef2.phaseInNm))
            assert ef.pixel_pitch == ef2.pixel_pitch
            assert ef.S0 == ef2.S0

            # Force cleanup for Windows
            del ef2
            gc.collect()