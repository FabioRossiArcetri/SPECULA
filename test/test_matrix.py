'''
The generic Matrix data object of iteras, and SPECULA's Recmat derived from it.
'''
import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest

import numpy as np

from iteras import cpuArray
from iteras.calib_manager import CalibManager
from iteras.data_objects.matrix import Matrix
from iteras.lib import storage
from specula.data_objects.recmat import Recmat

from test.specula_testlib import cpu_and_gpu

FORMATS = ['fits', 'npz', 'npy', 'hdf5']


class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = np.arange(6, dtype=np.float32).reshape(2, 3)

    def _path(self, name, data_format):
        return os.path.join(self.tmp.name, name + storage.EXTENSIONS[data_format])

    @cpu_and_gpu
    def test_values(self, target_device_idx, xp):
        m = Matrix(self.data, target_device_idx=target_device_idx)
        self.assertIsInstance(m.matrix, xp.ndarray)
        self.assertEqual((m.nrows, m.ncols), (2, 3))
        m.set_value(self.data * 2)
        np.testing.assert_array_equal(cpuArray(m.get_value()), self.data * 2)
        self.assertIs(m.array_for_display(), m.matrix)
        with self.assertRaises(AssertionError):
            m.set_value(np.zeros((3, 2)))

    @cpu_and_gpu
    def test_save_restore(self, target_device_idx, xp):
        m = Matrix(self.data, target_device_idx=target_device_idx)
        for data_format in FORMATS:
            with self.subTest(data_format=data_format):
                path = self._path(f'm{target_device_idx}', data_format)
                m.save(path)
                restored = Matrix.restore(path, target_device_idx=target_device_idx)
                self.assertIs(type(restored), Matrix)
                np.testing.assert_array_equal(cpuArray(restored.matrix), self.data)
                with storage.read_object(path) as f:
                    self.assertEqual(f.header['OBJ_TYPE'], 'Matrix')

    def test_wrong_version(self):
        path = self._path('v2', 'npz')
        storage.write_object(path, {'VERSION': 2}, {'matrix': self.data})
        with self.assertRaisesRegex(ValueError, 'version'):
            Matrix.restore(path)

    def test_calib_subdir(self):
        # 'matrix/' in iteras, 'rec/' in SPECULA, where matrices are reconstruction matrices
        cm = CalibManager('/root')
        self.assertEqual(cm.filename('Matrix', 'x'), os.path.join('/root', 'rec/', 'x.fits'))


class TestRecmatIsAMatrix(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = np.arange(6, dtype=np.float32).reshape(3, 2)

    def test_recmat_attribute_is_the_matrix(self):
        rec = Recmat(self.data, norm_factor=2.0, target_device_idx=-1)
        self.assertIsInstance(rec, Matrix)
        self.assertIs(rec.recmat, rec.matrix)
        self.assertEqual(rec.nmodes, 3)
        rec.recmat = np.zeros((2, 2), dtype=np.float32)
        self.assertEqual(rec.matrix.shape, (2, 2))
        # Not an array of its own: copied once by transferDataTo()
        self.assertNotIn('recmat', vars(rec))

    def test_recmat_files_read_as_matrix(self):
        '''A Recmat file is a Matrix file, with the reconstruction information'''
        rec = Recmat(self.data, norm_factor=2.0, target_device_idx=-1)
        for data_format in FORMATS:
            with self.subTest(data_format=data_format):
                path = os.path.join(self.tmp.name, 'rec' + storage.EXTENSIONS[data_format])
                rec.save(path)
                np.testing.assert_array_equal(cpuArray(Matrix.restore(path, target_device_idx=-1).matrix),
                                              self.data)
                restored = Recmat.restore(path, target_device_idx=-1)
                self.assertIs(type(restored), Recmat)
                self.assertEqual(restored.norm_factor, 2.0)
                with storage.read_object(path) as f:
                    self.assertEqual(f.header['OBJ_TYPE'], 'Recmat')

    def test_recmat_files_of_previous_versions(self):
        '''The matrix was named 'recmat', and FITS files had no extension names'''
        from astropy.io import fits
        path = os.path.join(self.tmp.name, 'old.npz')
        storage.write_object(path, {'VERSION': 1, 'NORMFACT': 0.5}, {'recmat': self.data})
        restored = Recmat.restore(path, target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(restored.recmat), self.data)
        self.assertEqual(restored.norm_factor, 0.5)

        path = os.path.join(self.tmp.name, 'old.fits')
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NORMFACT'] = 0.5
        fits.writeto(path, np.zeros(2), hdr)
        fits.append(path, self.data)
        for klass in (Recmat, Matrix):
            with self.subTest(klass=klass.__name__):
                restored = klass.restore(path, target_device_idx=-1)
                np.testing.assert_array_equal(cpuArray(restored.matrix), self.data)


if __name__ == '__main__':
    unittest.main()
