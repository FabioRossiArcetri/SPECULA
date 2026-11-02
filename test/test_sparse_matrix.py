import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula import cpuArray
from iteras.lib.sparse_matrix import SparseMatrix, mostly_zeros
from test.specula_testlib import cpu_and_gpu


def sparse_random(rng, shape, fill, dtype):
    m = rng.normal(size=shape).astype(dtype)
    m[rng.random(shape) > fill] = 0
    return m


class TestSparseMatrix(unittest.TestCase):

    @cpu_and_gpu
    def test_matvec_equals_dense(self, target_device_idx, xp):
        rng = np.random.default_rng(1)
        for dtype in (np.float32, np.float64):
            m = sparse_random(rng, (300, 70), 0.05, dtype)
            m[17] = 0                       # an empty row
            v = rng.normal(size=70).astype(dtype)
            sparse = SparseMatrix(xp.asarray(m), xp)
            self.assertEqual(sparse.nnz, np.count_nonzero(m))
            np.testing.assert_allclose(cpuArray(sparse.matvec(xp.asarray(v))), m @ v,
                                       rtol=1e-5 if dtype == np.float32 else 1e-12, atol=1e-6)

    @cpu_and_gpu
    def test_transposed_view(self, target_device_idx, xp):
        # As DM: the influence functions (actuators x pixels), one row per pixel
        rng = np.random.default_rng(2)
        ifunc = sparse_random(rng, (40, 500), 0.03, np.float32)
        cmd = rng.normal(size=40).astype(np.float32)
        sparse = SparseMatrix(xp.asarray(ifunc).T, xp)
        self.assertEqual(sparse.shape, (500, 40))
        np.testing.assert_allclose(cpuArray(sparse.matvec(xp.asarray(cmd))), cmd @ ifunc, rtol=1e-5, atol=1e-6)

    @cpu_and_gpu
    def test_captured_in_cuda_graph(self, target_device_idx, xp):  # pragma: gpu
        """The product can be captured in a CUDA graph, as the dense ones (see graph_matmul)"""
        if xp is np:
            self.skipTest('CUDA graphs are GPU only')
        rng = np.random.default_rng(5)
        m = sparse_random(rng, (200, 50), 0.05, np.float32)
        stream = xp.cuda.Stream(non_blocking=True)
        # All the work on the same stream: a non-blocking stream is not ordered
        # with the default one, where the arrays would otherwise be initialized
        with stream:
            sparse = SparseMatrix(xp.asarray(m), xp)
            v = xp.asarray(rng.normal(size=50).astype(np.float32))
            out = xp.zeros(200, dtype=np.float32)
            out[:] = sparse.matvec(v)       # compiles the kernel outside the capture
            stream.begin_capture()
            out[:] = sparse.matvec(v)
            graph = stream.end_capture()
            out[:] = 0
            v *= 2
            graph.launch(stream)
            stream.synchronize()
        np.testing.assert_allclose(cpuArray(out), m @ cpuArray(v), rtol=1e-5, atol=1e-6)

    @cpu_and_gpu
    def test_mostly_zeros(self, target_device_idx, xp):
        rng = np.random.default_rng(3)
        self.assertTrue(mostly_zeros(xp.asarray(sparse_random(rng, (100, 100), 0.05, np.float32)), xp))
        self.assertFalse(mostly_zeros(xp.asarray(rng.normal(size=(100, 100))), xp))


if __name__ == '__main__':
    unittest.main()
