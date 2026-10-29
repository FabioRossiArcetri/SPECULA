'''
Data objects can be saved and restored in npz, npy and HDF5 files, besides
FITS (see iteras.lib.storage).
'''
import specula
specula.init(0)  # Default target device

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from iteras.lib import storage
from iteras.calib_manager import CalibManager
from specula import cpuArray
from specula.data_objects.ifunc import IFunc
from specula.data_objects.recmat import Recmat

FORMATS = ['npz', 'npy', 'hdf5']

TEST_DIR = Path(__file__).resolve().parent


def _env_without_format():
    return {k: v for k, v in os.environ.items()
            if k not in ('ITERAS_DATA_FORMAT', 'SPECULA_DATA_FORMAT')}


class TestAllClasses(unittest.TestCase):

    def test_existing_save_tests_in_all_formats(self):
        '''
        Run the tests that save data objects with ITERAS_TEST_CHECK_OBJECT_FORMATS=1
        (see conftest.py): each object saved to FITS is also saved to npz, npy and
        HDF5, and the objects restored from the four files must be equal.
        '''
        files = sorted(str(p) for p in TEST_DIR.glob('test_*.py')
                       if p.name != Path(__file__).name and '.save(' in p.read_text(encoding='utf-8'))
        env = dict(os.environ, ITERAS_TEST_CHECK_OBJECT_FORMATS='1')
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *files],
                           capture_output=True, text=True, env=env, cwd=TEST_DIR.parent)
        self.assertEqual(r.returncode, 0, r.stdout[-5000:] + r.stderr[-2000:])
        match = re.search(r'Object formats checked: (.*)', r.stdout)
        self.assertIsNotNone(match, r.stdout[-2000:])
        checked = {tuple(item.strip().split('/')) for item in match.group(1).split(',')}
        classes = {name for name, _ in checked}
        for data_format in FORMATS:
            self.assertTrue({(name, data_format) for name in classes} <= checked)
        # Most data object classes are saved by their tests
        for name in ['BaseValue', 'ElectricField', 'Intmat', 'Recmat', 'Pixels', 'Slopes',
                     'SubapData', 'PupData', 'M2C', 'IFuncInv', 'IirFilterData', 'SsrFilterData',
                     'Pupilstop', 'Layer', 'ConvolutionKernel', 'SpatioTempArray']:
            self.assertIn(name, classes)

    def test_ifunc_round_trip(self):
        # IFunc is not saved by its tests
        mask = np.ones((4, 4))
        mask[0, 0] = 0
        ifunc = IFunc(ifunc=np.arange(45, dtype=np.float32).reshape(3, 15), mask=mask,
                      target_device_idx=-1)
        with tempfile.TemporaryDirectory() as tmpdir:
            for data_format in FORMATS:
                with self.subTest(data_format=data_format):
                    path = os.path.join(tmpdir, 'ifunc' + storage.EXTENSIONS[data_format])
                    ifunc.save(path)
                    restored = IFunc.restore(path, target_device_idx=-1)
                    np.testing.assert_array_equal(cpuArray(restored.influence_function),
                                                  cpuArray(ifunc.influence_function))
                    np.testing.assert_array_equal(cpuArray(restored.mask_inf_func),
                                                  cpuArray(ifunc.mask_inf_func))


class TestFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.recmat = Recmat(np.arange(6, dtype=np.float32).reshape(3, 2), target_device_idx=-1)

    def test_file_contents(self):
        '''The arrays and the header are stored with their names, readable without iteras'''
        expected = cpuArray(self.recmat.recmat)
        # FITS: same layout as before, with named extensions
        path = os.path.join(self.tmp.name, 'rec.fits')
        self.recmat.save(path)
        from astropy.io import fits
        with fits.open(path) as hdul:
            self.assertEqual(hdul[1].name, 'RECMAT')
            np.testing.assert_array_equal(hdul[1].data, expected)
            self.assertEqual(hdul[0].header['NORMFACT'], self.recmat.norm_factor)
        # npz
        path = os.path.join(self.tmp.name, 'rec.npz')
        self.recmat.save(path)
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f['recmat'], expected)
            self.assertIn('NORMFACT', str(f['__header__']))
        # npy: a directory with one file per array and the header
        path = os.path.join(self.tmp.name, 'rec.npy')
        self.recmat.save(path)
        np.testing.assert_array_equal(np.load(os.path.join(path, 'recmat.npy')), expected)
        self.assertTrue(os.path.exists(os.path.join(path, 'header.json')))
        # HDF5
        import h5py
        path = os.path.join(self.tmp.name, 'rec.h5')
        self.recmat.save(path)
        with h5py.File(path, 'r') as f:
            np.testing.assert_array_equal(f['recmat'][()], expected)
            self.assertEqual(f.attrs['NORMFACT'], self.recmat.norm_factor)

    def test_overwrite(self):
        path = os.path.join(self.tmp.name, 'rec.npz')
        self.recmat.save(path)
        with self.assertRaisesRegex(OSError, 'already exists'):
            self.recmat.save(path, overwrite=False)
        self.recmat.save(path, overwrite=True)

    def test_path_objects(self):
        path = Path(self.tmp.name) / 'rec.npz'
        self.recmat.save(path)
        restored = Recmat.restore(path, target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(restored.recmat), cpuArray(self.recmat.recmat))


class TestStorageAPI(unittest.TestCase):
    '''write_object() and read_object()'''

    def setUp(self):
        from astropy.io import fits
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.header = fits.Header()
        self.header['VERSION'] = 2
        self.header['NAME'] = ('test', 'a comment')
        self.header.add_comment('a comment card')
        self.a = np.arange(6, dtype=np.float64).reshape(2, 3)
        self.b = np.arange(4, dtype=np.int32)

    def _path(self, name, data_format):
        return os.path.join(self.tmp.name, name + storage.EXTENSIONS[data_format])

    def test_fits_layouts(self):
        from astropy.io import fits
        arrays = {'a': self.a, 'b': self.b}
        for layout, primary_shape, n_hdus in [('extensions', None, 3), ('primary', (2, 3), 2),
                                              ('placeholder', (2,), 3)]:
            with self.subTest(layout=layout):
                path = self._path(layout, 'fits')
                storage.write_object(path, self.header, arrays, layout=layout)
                with fits.open(path) as hdul:
                    self.assertEqual(len(hdul), n_hdus)
                    shape = None if hdul[0].data is None else hdul[0].data.shape
                    self.assertEqual(shape, primary_shape)
                    self.assertEqual(hdul[-1].name, 'B')
                    self.assertEqual(hdul[0].header['NAME'], 'test')
                with storage.read_object(path) as f:
                    # The array in the primary HDU has no name in FITS files
                    np.testing.assert_array_equal(f.read('a', 0 if layout == 'primary' else 1), self.a)
                    np.testing.assert_array_equal(f.read('b'), self.b)
        with self.assertRaisesRegex(ValueError, 'layout'):
            storage.write_object(self._path('x', 'fits'), self.header, arrays, layout='other')

    def test_headers_and_arrays_in_all_formats(self):
        for data_format in storage.OBJECT_FORMATS:
            with self.subTest(data_format=data_format):
                path = self._path('obj', data_format)
                storage.write_object(path, self.header, {'a': self.a, 'b': self.b, 'none': None})
                with storage.read_object(path) as f:
                    self.assertEqual(f.header['VERSION'], 2)
                    self.assertEqual(f.header['NAME'], 'test')
                    self.assertEqual(f.header.comments['NAME'], 'a comment')
                    self.assertIn('a comment card', str(f.header['COMMENT']))
                    self.assertTrue(f.has('a', 1))
                    self.assertFalse(f.has('none', 3))
                    np.testing.assert_array_equal(f.read('a', 1), self.a)
                    self.assertEqual(f.read('b', 2).dtype, np.int32)
                    self.assertTrue(f.read('a', 1).dtype.isnative)
                    with self.assertRaises(KeyError):
                        f.read('missing', 5)

    def test_old_fits_files_by_index(self):
        '''FITS files without extension names, as written by older versions'''
        from astropy.io import fits
        path = self._path('old', 'fits')
        fits.writeto(path, np.zeros(2), self.header)
        fits.append(path, self.a)
        with storage.read_object(path) as f:
            np.testing.assert_array_equal(f.read('a', 1), self.a)
            np.testing.assert_array_equal(f.read(None, 1), self.a)
            self.assertFalse(f.has('b', 2))

    def test_load_on_device(self):
        targets = [-1] + ([0] if specula.cp is not None else [])
        for data_format in storage.OBJECT_FORMATS:
            path = self._path('load', data_format)
            storage.write_object(path, self.header, {'a': self.a, 'b': self.b})
            for target in targets:
                with self.subTest(data_format=data_format, target=target), storage.read_object(path) as f:
                    a32 = f.load('a', 1, target_device_idx=target, precision=1)
                    a64 = f.load('a', 1, target_device_idx=target, precision=0)
                    b = f.load('b', 2, target_device_idx=target)
                    self.assertEqual((a32.dtype, a64.dtype, b.dtype),
                                     (np.float32, np.float64, np.int32))
                    if target >= 0:
                        self.assertIsInstance(a32, specula.cp.ndarray)
                    np.testing.assert_array_equal(cpuArray(a64), self.a)
                    np.testing.assert_array_equal(cpuArray(b), self.b)
                    with self.assertRaisesRegex(ValueError, 'integer'):
                        f.load('b', 2, target_device_idx=target, precision=1)

    def test_npy_does_not_replace_other_directories(self):
        path = os.path.join(self.tmp.name, 'other.npy')
        os.makedirs(path)
        with self.assertRaisesRegex(OSError, 'not a directory written'):
            storage.write_object(path, self.header, {'a': self.a})


class TestCalibrationFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cm = CalibManager(self.tmp.name)
        os.makedirs(os.path.join(self.tmp.name, 'rec'))
        self.recmat = Recmat(np.arange(6, dtype=np.float32).reshape(3, 2), target_device_idx=-1)

    def test_extension_of_new_files(self):
        with patch.dict(os.environ, _env_without_format(), clear=True):
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('.fits'))
            os.environ['SPECULA_DATA_FORMAT'] = 'npz'
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('.npz'))
            # pickle is only for DataStore: calibration files stay FITS
            os.environ['SPECULA_DATA_FORMAT'] = 'pickle'
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('.fits'))
        # An explicit extension is kept
        self.assertTrue(self.cm.filename('Recmat', 'r1.h5').endswith('r1.h5'))

    def test_existing_file_found_in_any_format(self):
        self.recmat.save(os.path.join(self.tmp.name, 'rec', 'r1.npz'))
        with patch.dict(os.environ, _env_without_format(), clear=True):
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('r1.npz'))
            # With both, the file of the default format wins
            self.recmat.save(os.path.join(self.tmp.name, 'rec', 'r1.fits'))
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('r1.fits'))
            os.environ['ITERAS_DATA_FORMAT'] = 'npz'
            self.assertTrue(self.cm.filename('Recmat', 'r1').endswith('r1.npz'))

    def test_simul_restores_tag_from_npz(self):
        from iteras.simul import Simul
        self.recmat.save(os.path.join(self.tmp.name, 'rec', 'r1.npz'))
        simul = Simul('dummy.yml')
        simul.cm = self.cm
        restored = simul.restore_object(Recmat, 'r1', target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(restored.recmat), cpuArray(self.recmat.recmat))

    def test_raw_data_files(self):
        os.makedirs(os.path.join(self.tmp.name, 'data'))
        data = np.arange(4.0)
        with patch.dict(os.environ, _env_without_format(), clear=True):
            os.environ['ITERAS_DATA_FORMAT'] = 'npz'
            self.cm.write_data('d1', data)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, 'data', 'd1.npz')))
        np.testing.assert_array_equal(self.cm.read_data('d1'), data)


if __name__ == '__main__':
    unittest.main()
