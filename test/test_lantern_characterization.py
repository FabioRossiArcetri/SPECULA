import os
import unittest

import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.data_objects.lantern_characterization import LanternCharacterization

from test.specula_testlib import cpu_and_gpu


N_MESH = 7
N_MODES = 4
N_PORTS = 3
WL = [1150.0, 1160.0, 1170.0]


def make_char(n_wl=3, n_mesh=N_MESH, n_modes=N_MODES, n_ports=N_PORTS,
              pixel_scale_um=None, target_device_idx=None, **kwargs):
    '''A small characterization with arbitrary (but reproducible) matrices'''
    rng = np.random.default_rng(0)
    wavelengths_nm = np.asarray(WL[:n_wl], float)
    mesh = np.stack([np.linspace(-10, 10, n_mesh), np.linspace(5, -5, n_mesh)], axis=1)
    projector = rng.normal(size=(n_wl, n_modes, n_mesh)) + 1j * rng.normal(size=(n_wl, n_modes, n_mesh))
    channels = rng.normal(size=(n_wl, n_ports, n_modes)) + 1j * rng.normal(size=(n_wl, n_ports, n_modes))
    if pixel_scale_um is None:
        pixel_scale_um = np.linspace(0.98, 1.07, n_wl)
    return LanternCharacterization(
        wavelengths_nm=wavelengths_nm,
        mesh_xy_um=mesh,
        mode_projector=projector,
        channel_matrix=channels,
        pixel_scale_um=pixel_scale_um,
        focal_plane_norm=np.full(n_wl, 1.6e6),
        pupil_mask=np.ones((4, 4)),
        lantern='pl_test',
        pad_factor=4,
        design_wavelength_nm=1200.0,
        psf_fill_factor=0.5,
        target_device_idx=target_device_idx,
        **kwargs)


class TestLanternCharacterization(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_pl_char.fits')

    def tearDown(self):
        for f in (self.filename, self.filename.replace('.fits', '.npz')):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    @cpu_and_gpu
    def test_shapes(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        self.assertEqual((char.n_wl, char.n_ports, char.n_modes, char.n_mesh), (3, 3, 4, 7))
        self.assertEqual(char.mode_projector.shape, (3, 4, 7))
        self.assertEqual(char.channel_matrix.shape, (3, 3, 4))
        self.assertEqual(char.complex_dtype, char.mode_projector.dtype)
        self.assertEqual(char.dtype, char.mesh_xy_um.dtype)

    @cpu_and_gpu
    def test_precision_is_respected(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx, precision=0)
        self.assertEqual(char.mesh_xy_um.dtype, xp.float64)
        self.assertEqual(char.mode_projector.dtype, xp.complex128)
        char = make_char(target_device_idx=target_device_idx, precision=1)
        self.assertEqual(char.mesh_xy_um.dtype, xp.float32)
        self.assertEqual(char.mode_projector.dtype, xp.complex64)

    @cpu_and_gpu
    def test_reference_pupil_diameter(self, target_device_idx, xp):
        # lambda / (pad_factor * pixel_scale_um), with lambda in um, gives metres
        char = make_char(target_device_idx=target_device_idx,
                         pixel_scale_um=np.array([1.0253, 1.0253, 1.0253]))
        d = cpuArray(char.reference_pupil_diameter_m)
        expected = cpuArray(char.wavelengths_nm) * 1e-3 / (4 * 1.0253)
        np.testing.assert_allclose(d, expected, rtol=1e-6)
        # the pixel scale of a characterization scales with the wavelength, so the
        # equivalent pupil diameter is the same at every wavelength
        char = make_char(target_device_idx=target_device_idx,
                         pixel_scale_um=cpuArray(make_char(target_device_idx=target_device_idx)
                                                 .wavelengths_nm) * 0.98 / 1150.0)
        d = cpuArray(char.reference_pupil_diameter_m)
        np.testing.assert_allclose(d, d[0] * np.ones(3), rtol=1e-6)

    @cpu_and_gpu
    def test_wavelength_index(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        self.assertEqual(char.wavelength_index(1150.0), 0)
        self.assertEqual(char.wavelength_index(1170.0), 2)
        self.assertEqual(char.indexes_for([1160.0, 1150.0]), [1, 0])
        # within the tolerance (float32 round-off of a stored wavelength)
        self.assertEqual(char.wavelength_index(1160.0 + 0.01), 1)
        with self.assertRaises(ValueError):
            char.wavelength_index(1200.0)

    @cpu_and_gpu
    def test_subset(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        sub = char.subset([0, 2])
        self.assertEqual(sub.n_wl, 2)
        np.testing.assert_allclose(cpuArray(sub.wavelengths_nm), [1150.0, 1170.0])
        np.testing.assert_allclose(cpuArray(sub.channel_matrix),
                                   cpuArray(char.channel_matrix[[0, 2]]))
        # the mesh is shared, not copied
        np.testing.assert_allclose(cpuArray(sub.mesh_xy_um), cpuArray(char.mesh_xy_um))
        self.assertEqual(sub.lantern, 'pl_test')
        self.assertEqual(sub.pad_factor, 4)
        with self.assertRaises(ValueError):
            char.subset([])

    @cpu_and_gpu
    def test_validation(self, target_device_idx, xp):
        rng = np.random.default_rng(1)
        proj = rng.normal(size=(2, 4, N_MESH)) + 0j
        chan = rng.normal(size=(2, N_PORTS, N_MODES)) + 0j
        mesh = np.zeros((N_MESH, 2))
        kw = dict(mesh_xy_um=mesh, mode_projector=proj, channel_matrix=chan,
                  pixel_scale_um=[1.0, 1.0], focal_plane_norm=[1.0, 1.0],
                  target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):        # no wavelength
            LanternCharacterization(wavelengths_nm=[], **kw)
        with self.assertRaises(ValueError):        # not increasing
            LanternCharacterization(wavelengths_nm=[1160.0, 1150.0], **kw)
        with self.assertRaises(ValueError):        # wrong mesh points
            LanternCharacterization(wavelengths_nm=[1150.0, 1160.0],
                               **{**kw, 'mode_projector': proj[:, :, :3]})
        with self.assertRaises(ValueError):        # mode count mismatch
            LanternCharacterization(wavelengths_nm=[1150.0, 1160.0],
                               **{**kw, 'channel_matrix': chan[:, :, :2]})
        with self.assertRaises(ValueError):        # non-positive pixel scale
            LanternCharacterization(wavelengths_nm=[1150.0, 1160.0],
                               **{**kw, 'pixel_scale_um': [1.0, 0.0]})
        with self.assertRaises(ValueError):        # bad pad factor
            LanternCharacterization(wavelengths_nm=[1150.0, 1160.0], pad_factor=0, **kw)
        with self.assertRaises(ValueError):        # one norm per wavelength
            LanternCharacterization(wavelengths_nm=[1150.0, 1160.0],
                               **{**kw, 'focal_plane_norm': [1.0]})

    @cpu_and_gpu
    def test_roundtrip_fits(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        char.save(self.filename, overwrite=True)
        restored = LanternCharacterization.restore(self.filename, target_device_idx=target_device_idx)
        np.testing.assert_allclose(cpuArray(restored.wavelengths_nm), cpuArray(char.wavelengths_nm))
        np.testing.assert_allclose(cpuArray(restored.mesh_xy_um), cpuArray(char.mesh_xy_um))
        np.testing.assert_allclose(cpuArray(restored.mode_projector), cpuArray(char.mode_projector),
                                   rtol=1e-6)
        np.testing.assert_allclose(cpuArray(restored.channel_matrix), cpuArray(char.channel_matrix),
                                   rtol=1e-6)
        np.testing.assert_allclose(cpuArray(restored.pixel_scale_um), cpuArray(char.pixel_scale_um),
                                   rtol=1e-6)
        np.testing.assert_allclose(cpuArray(restored.pupil_mask), cpuArray(char.pupil_mask))
        self.assertEqual(restored.lantern, 'pl_test')
        self.assertEqual(restored.pad_factor, 4)
        self.assertAlmostEqual(restored.design_wavelength_nm, 1200.0)
        self.assertAlmostEqual(restored.psf_fill_factor, 0.5)
        self.assertEqual(restored.channel_matrix.dtype, char.channel_matrix.dtype)

    @cpu_and_gpu
    def test_roundtrip_npz(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        filename = self.filename.replace('.fits', '.npz')
        char.save(filename, overwrite=True)
        restored = LanternCharacterization.restore(filename, target_device_idx=target_device_idx)
        np.testing.assert_allclose(cpuArray(restored.mode_projector), cpuArray(char.mode_projector),
                                   rtol=1e-6)
        self.assertEqual(restored.lantern, 'pl_test')

    @cpu_and_gpu
    def test_restore_checks_object_type(self, target_device_idx, xp):
        from astropy.io import fits
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Something else'
        fits.writeto(self.filename, np.zeros((2, 2)), hdr, overwrite=True)
        with self.assertRaises(ValueError):
            LanternCharacterization.restore(self.filename)

    @cpu_and_gpu
    def test_set_value_shape_check(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        with self.assertRaises(AssertionError):
            char.set_value(xp.zeros((2, 2)))
        char.set_value(xp.zeros((3, N_MODES, N_MESH)))
        np.testing.assert_allclose(cpuArray(char.mode_projector), 0.0)


if __name__ == '__main__':
    unittest.main()
