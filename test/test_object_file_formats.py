'''
Data objects can be saved and restored in npz and HDF5 files, besides FITS
(see iteras.lib.storage and BaseDataObj).
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

try:
    import h5py  # noqa: F401
    FORMATS = ['npz', 'hdf5']
except ImportError:
    FORMATS = ['npz']

TEST_DIR = Path(__file__).resolve().parent


def _env_without_format():
    return {k: v for k, v in os.environ.items()
            if k not in ('ITERAS_DATA_FORMAT', 'SPECULA_DATA_FORMAT')}


class TestAllClasses(unittest.TestCase):

    def test_existing_save_tests_in_all_formats(self):
        '''
        Run the tests that save data objects with ITERAS_TEST_CHECK_OBJECT_FORMATS=1
        (see conftest.py): each object saved to FITS is also saved to npz and HDF5,
        and the objects restored from the three files must be equal.
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

    def test_files_are_npz_and_hdf5(self):
        for data_format in FORMATS:
            path = os.path.join(self.tmp.name, 'rec' + storage.EXTENSIONS[data_format])
            self.recmat.save(path)
            if data_format == 'npz':
                with np.load(path, allow_pickle=False) as f:
                    self.assertIn('hdu1_data', f.files)
            else:
                import h5py
                with h5py.File(path, 'r') as f:
                    self.assertIn('hdu1', f)
            hdul = storage.read_hdulist(path, data_format)
            np.testing.assert_array_equal(hdul[1].data, cpuArray(self.recmat.recmat))

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
