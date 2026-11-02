"""
Sparse matrix @ vector, for matrices whose elements are mostly exact zeros
(e.g. the zonal influence functions of a DM with many actuators, where each
actuator moves only the pixels around it).

A dense product reads the whole matrix at each call, and is limited by that
memory traffic; the sparse one reads only the non-zero elements. The result
is the same sum of the same products, in another order: it differs from the
dense one only by rounding.

On GPU the product is a raw kernel, which can be captured in a CUDA graph
(see graph_matmul), like the dense products of the simulations.
"""
import numpy as np

_CSR_MATVEC_SRC = r'''
// Y[r] = sum_j data[j] X[indices[j]] over the elements j of row r (CSR format):
// one warp per row. The threads of the warp read consecutive elements of the
// row, then add their sums with warp shuffles, in a fixed order (deterministic).
template<typename T>
__global__ void csr_matvec(const int* indptr, const int* indices, const T* data,
                           const T* X, T* Y, int R) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= R) return;
    T acc = 0;
    for (int j = indptr[row] + lane; j < indptr[row + 1]; j += 32)
        acc += data[j] * X[indices[j]];
    for (int off = 16; off > 0; off /= 2)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0)
        Y[row] = acc;
}
'''
_csr_matvec_kernels = {}


def mostly_zeros(matrix, xp, max_fill=0.2):
    '''True if at most *max_fill* of the elements of *matrix* are not zero'''
    return int(xp.count_nonzero(matrix)) <= max_fill * matrix.size


class SparseMatrix:
    '''
    The non-zero elements of a dense 2-d *matrix* (any layout, e.g. a
    transposed view), row by row (CSR format), for products with vectors.
    '''

    def __init__(self, matrix, xp):
        self.xp = xp
        self.shape = matrix.shape
        self.dtype = matrix.dtype
        rows, cols = xp.nonzero(matrix)      # row by row, columns in increasing order
        self.data = xp.ascontiguousarray(matrix[rows, cols])
        self.indices = cols.astype(np.int32)
        self.indptr = xp.zeros(self.shape[0] + 1, dtype=np.int32)
        self.indptr[1:] = xp.cumsum(xp.bincount(rows, minlength=self.shape[0]))
        if xp is np:
            from scipy.sparse import csr_matrix
            self._csr = csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)

    @property
    def nnz(self):
        return self.data.size

    def matvec(self, v):
        '''matrix @ v, for a vector v'''
        if self.xp is np:
            return self._csr @ v
        return self._matvec_gpu(v)  # pragma: gpu

    def _matvec_gpu(self, v):  # pragma: gpu
        xp = self.xp
        key = (np.dtype(self.dtype).str, xp.cuda.Device().id)
        if key not in _csr_matvec_kernels:
            ctype = 'float' if self.dtype == np.float32 else 'double'
            mod = xp.RawModule(code=_CSR_MATVEC_SRC, name_expressions=[f'csr_matvec<{ctype}>'])
            _csr_matvec_kernels[key] = mod.get_function(f'csr_matvec<{ctype}>')
        v = xp.ascontiguousarray(v, dtype=self.dtype)
        r = self.shape[0]
        y = xp.empty(r, dtype=self.dtype)
        _csr_matvec_kernels[key](((r * 32 + 255) // 256,), (256,),
                                 (self.indptr, self.indices, self.data, v, y, np.int32(r)))
        return y
