import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cp
from specula.base_value import BaseValue


class TestBaseDataObj(unittest.TestCase):

    def test_copy_from_cpu_to_cpu(self):
        '''
        Test that copyTo() from CPU to CPU works fine
        '''
        a = BaseValue(value=np.arange(2), target_device_idx=-1)
        b = a.copyTo(target_device_idx=-1)

        assert type(b.value) == np.ndarray
        assert b.target_device_idx == -1
        np.testing.assert_array_equal(b.value, [0, 1])

    def test_update_from_cpu_to_cpu(self):
        '''
        Test that transferDataTo() from CPU to CPU works fine
        '''
        a = BaseValue(value=np.arange(2), target_device_idx=-1)
        b = BaseValue(value=np.zeros(2), target_device_idx=-1)
        a.transferDataTo(b)

        assert type(b.value) == np.ndarray
        assert b.target_device_idx == -1
        np.testing.assert_array_equal(b.value, [0, 1])

    def test_update_scalar_value_cpu_to_cpu(self):
        '''
        Regression test: transferDataTo() must actually update a BaseValue
        constructed with an initial scalar (value=1.0 below): __init__
        builds .value via self.dtype(value), a genuine numpy/cupy scalar
        (e.g. numpy.float32), not an ndarray. array_types only lists true
        ndarray types, so this used to be silently skipped by the copy
        loop entirely -- the destination stayed frozen at whatever value
        it captured on the first copyTo(), forever, even though
        generation_time kept updating on every subsequent
        transferDataTo() call. (A different case -- a BaseValue that
        starts at value=None and is only ever updated via set_value(),
        which produces a *0-d array* rather than a scalar type -- is
        covered separately below; it hit a different bug in the same
        method.)
        '''
        a = BaseValue(value=1.0, target_device_idx=-1)
        b = BaseValue(value=0.0, target_device_idx=-1)
        a.transferDataTo(b)
        self.assertEqual(float(b.value), 1.0)

        a.set_value(2.0)
        a.transferDataTo(b)
        self.assertEqual(float(b.value), 2.0)

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_copy_from_cpu_to_gpu(self):
        '''
        Test that copyTo() with target_device_idx >= 0
        allocates a new object on the GPU with the correct contents
        '''
        a = BaseValue(value=np.arange(2), target_device_idx=-1)
        b = a.copyTo(target_device_idx=0)

        assert type(b.value) == cp.ndarray
        assert b.target_device_idx == 0
        np.testing.assert_array_equal(b.value.get(), [0, 1])

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_update_from_cpu_to_gpu(self):
        '''
        Test that transferDataTo() with a GPU object
        correctly updates the contents
        '''
        a1 = BaseValue(value=np.arange(2), target_device_idx=-1)
        a2 = BaseValue(value=np.arange(2)+2, target_device_idx=-1)
        b = a1.copyTo(target_device_idx=0)
        _ = a2.transferDataTo(b)

        assert type(b.value) == cp.ndarray
        np.testing.assert_array_equal(b.value.get(), [2, 3])
        assert b.target_device_idx == 0

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_copy_from_gpu_to_cpu(self):
        '''
        Test that copyTo() with target_device_idx == -1
        allocates a new object on the CPU with the correct contents
        '''
        a = BaseValue(value=cp.arange(2), target_device_idx=0)
        b = a.copyTo(target_device_idx=-1)

        np.testing.assert_array_equal(b.value, [0, 1])
        assert type(b.value) == np.ndarray
        assert b.target_device_idx == -1

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_update_from_gpu_to_cpu(self):
        '''
        Test that transferDataTo() with a CPU object
        correctly updates the contents
        '''
        a1 = BaseValue(value=cp.arange(2), target_device_idx=0)
        a2 = BaseValue(value=cp.arange(2)+2, target_device_idx=0)
        b = a1.copyTo(target_device_idx=-1)
        _ = a2.transferDataTo(b)

        assert type(b.value) == np.ndarray
        np.testing.assert_array_equal(b.value, [2, 3])
        assert b.target_device_idx == -1

    @unittest.skipIf(cp is None or cp.cuda.runtime.getDeviceCount() < 2, 'at least 2 GPUs are needed')
    def test_copy_from_gpu_to_gpu(self):
        '''
        Test that copyTo() from a GPU device to another
        allocates a new object on the target GPU with the correct contents
        '''        
        a = BaseValue(value=cp.arange(2), target_device_idx=0)
        b = a.copyTo(target_device_idx=1)

        np.testing.assert_array_equal(b.value.get(), [0, 1])
        assert type(b.value) == cp.ndarray
        assert b.target_device_idx == 1

    @unittest.skipIf(cp is None or cp.cuda.runtime.getDeviceCount() < 2, 'at least 2 GPUs are needed')
    def test_update_from_gpu_to_gpu(self):
        '''
        Test that transferDataTo() from a GPU device to another
        correctly updates the contents
        '''
        a1 = BaseValue(value=cp.arange(2), target_device_idx=0)
        a2 = BaseValue(value=cp.arange(2)+2, target_device_idx=0)
        b = a1.copyTo(target_device_idx=1)
        _ = a2.transferDataTo(b)

        assert type(b.value) == cp.ndarray
        np.testing.assert_array_equal(b.value.get(), [2, 3])
        assert b.target_device_idx == 1

    @unittest.skipIf(cp is None or cp.cuda.runtime.getDeviceCount() < 2, 'at least 2 GPUs are needed')
    def test_update_scalar_value_from_gpu_to_gpu(self):
        '''
        Same regression as test_update_scalar_value_cpu_to_cpu, but for
        the actual cross-device scenario that surfaced it in production:
        a Conv2dNetTrainer output on one GPU (target_device_idx=1) read by
        a PlotDisplay on another (target_device_idx=0, BaseDisplay's
        default -- it doesn't accept/forward target_device_idx at all).
        '''
        a1 = BaseValue(value=1.0, target_device_idx=0)
        a2 = BaseValue(value=2.0, target_device_idx=0)
        b = a1.copyTo(target_device_idx=1)
        a2.transferDataTo(b)

        self.assertEqual(float(b.value), 2.0)
        assert b.target_device_idx == 1

    def test_update_0d_array_value_cpu_to_cpu(self):
        '''
        Regression test for the actual production bug (found via
        Conv2dNetTrainer's val_rms_first10_modes output feeding a
        PlotDisplay on a different device): a BaseValue that starts at
        value=None and is only ever updated via set_value() ends up with
        a *0-d array* (built by set_value()'s to_xp() call the first
        time), not a scalar type -- so it IS in array_types, and takes
        the normal in-place-copy path (dest_attr[:] = self_attr). But
        `[:]` raises IndexError on a 0-d array ("too many indices"). The
        DtD/HtH branches' fallback for that used to just rebind the local
        `dest_attr` variable instead of calling setattr(destobj, ...) --
        so the destination silently kept whatever value it had from the
        first copyTo() (which goes through force_reallocation=True and so
        doesn't hit this path), forever, on every later transferDataTo().
        '''
        # Two independently-seeded BaseValues, not copyTo()'s clone: cupy/
        # numpy's asarray() returns the *same object* when no device
        # change is needed (verified separately), so a clone obtained via
        # copyTo() can end up aliasing the source's array -- which would
        # make later updates "work" via the alias regardless of whether
        # transferDataTo() itself is actually correct, defeating the
        # point of this regression test.
        a = BaseValue(target_device_idx=-1)
        a.set_value(1.0)
        b = BaseValue(target_device_idx=-1)
        b.set_value(0.0)

        a.transferDataTo(b)
        self.assertEqual(float(b.value), 1.0)

        a.set_value(2.0)
        a.transferDataTo(b)
        self.assertEqual(float(b.value), 2.0)

    @unittest.skipIf(cp is None or cp.cuda.runtime.getDeviceCount() < 2, 'at least 2 GPUs are needed')
    def test_update_0d_array_value_from_gpu_to_gpu(self):
        '''
        Same as test_update_0d_array_value_cpu_to_cpu, but for the actual
        cross-device scenario that surfaced it in production (see there,
        and test_update_scalar_value_from_gpu_to_gpu). Also avoids
        copyTo()'s aliasing for the same reason (see the comment in
        test_update_0d_array_value_cpu_to_cpu).
        '''
        a = BaseValue(target_device_idx=0)
        a.set_value(1.0)
        b = BaseValue(target_device_idx=1)
        b.set_value(0.0)

        a.transferDataTo(b)
        self.assertEqual(float(b.value), 1.0)

        a.set_value(2.0)
        a.transferDataTo(b)
        self.assertEqual(float(b.value), 2.0)
        assert b.target_device_idx == 1
        assert b.target_device_idx == 1
