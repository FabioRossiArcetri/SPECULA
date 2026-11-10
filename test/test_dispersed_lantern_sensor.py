import os
import unittest

import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.lantern_characterization import LanternCharacterization
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.dispersed_lantern_sensor import DispersedLanternSensor

from test.specula_testlib import cpu_and_gpu


PUPIL = 8
PIXEL_PITCH = 0.1
CHAR_PAD = 2
WL = [1200.0, 1210.0]


def make_simul_params(pixel_pupil=PUPIL, pixel_pitch=PIXEL_PITCH):
    return SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch,
                       root_dir='.', total_time=0.1, time_step=0.001)


def make_char(mesh_xy=None, n_modes=1, n_ports=1, wavelengths=WL, pad=CHAR_PAD,
              pixel_scale_um=None, projector=None, channels=None, pupil_mask=None,
              target_device_idx=None):
    '''
    A characterization with the pupil diameter of the simulation (D_ref = D_sim),
    so that the mesh positions map without rescaling, and identity-like matrices
    by default.
    '''
    wavelengths = np.asarray(wavelengths, float)
    n_wl = len(wavelengths)
    if mesh_xy is None:
        mesh_xy = np.zeros((1, 2))
    mesh_xy = np.asarray(mesh_xy, float)
    n_mesh = mesh_xy.shape[0]
    # D_ref = lambda / (pad * pixel_scale_um) must equal the simulation diameter
    d_sim = PUPIL * PIXEL_PITCH
    if pixel_scale_um is None:
        pixel_scale_um = wavelengths * 1e-3 / (pad * d_sim)
    pixel_scale_um = np.broadcast_to(np.asarray(pixel_scale_um, float), (n_wl,))
    if projector is None:
        projector = np.zeros((n_wl, n_modes, n_mesh), dtype=complex)
        for m in range(min(n_modes, n_mesh)):
            projector[:, m, m] = 1.0
    if channels is None:
        channels = np.zeros((n_wl, n_ports, n_modes), dtype=complex)
        for m in range(min(n_modes, n_ports)):
            channels[:, m, m] = 1.0
    return LanternCharacterization(
        wavelengths_nm=wavelengths,
        mesh_xy_um=mesh_xy,
        mode_projector=projector,
        channel_matrix=channels,
        pixel_scale_um=pixel_scale_um,
        focal_plane_norm=np.ones(n_wl),
        pupil_mask=pupil_mask,
        lantern='test',
        pad_factor=pad,
        target_device_idx=target_device_idx)


def make_ef(phase_nm=0.0, amp=1.0, t=1, target_device_idx=None):
    ef = ElectricField(PUPIL, PUPIL, PIXEL_PITCH, target_device_idx=target_device_idx)
    ef.A[:] = amp
    ef.phaseInNm[:] = phase_nm
    ef.generation_time = t
    return ef


def run(obj, ef, t=1):
    obj.inputs['in_ef'].set(ef)
    obj.setup()
    obj.check_ready(t)
    obj.trigger()
    obj.post_trigger()


class TestDispersedLanternSensor(unittest.TestCase):

    @cpu_and_gpu
    def test_uniform_pupil_gives_the_fft_sum(self, target_device_idx, xp):
        # one mode, one port, one node at the focus center: the signal is the
        # square modulus of the sum of the pupil field, i.e. (N_pupil)**2
        char = make_char(target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        run(obj, make_ef(target_device_idx=target_device_idx))
        signals = cpuArray(obj.outputs['out_signals'].value)
        np.testing.assert_allclose(signals, PUPIL ** 4, rtol=1e-5)
        np.testing.assert_allclose(cpuArray(obj.outputs['out_throughput'].value),
                                   PUPIL ** 4, rtol=1e-5)

    @cpu_and_gpu
    def test_global_piston_does_not_change_the_signals(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        run(obj, make_ef(phase_nm=0.0, target_device_idx=target_device_idx))
        ref = cpuArray(obj.outputs['out_signals'].value).copy()
        for piston in (37.0, 300.0, -120.0):
            run(obj, make_ef(phase_nm=piston, target_device_idx=target_device_idx))
            np.testing.assert_allclose(cpuArray(obj.outputs['out_signals'].value), ref, rtol=1e-5)

    @cpu_and_gpu
    def test_amplitude_normalisation(self, target_device_idx, xp):
        # the signals are fractions of the incident power: scaling the pupil
        # amplitude must leave them unchanged (default) or scale them by amp**2
        char = make_char(target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        run(obj, make_ef(amp=1.0, target_device_idx=target_device_idx))
        ref = cpuArray(obj.outputs['out_signals'].value).copy()
        run(obj, make_ef(amp=3.0, target_device_idx=target_device_idx))
        np.testing.assert_allclose(cpuArray(obj.outputs['out_signals'].value), ref, rtol=1e-5)

        raw = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                normalize_amplitude=False, target_device_idx=target_device_idx)
        run(raw, make_ef(amp=2.0, target_device_idx=target_device_idx))
        np.testing.assert_allclose(cpuArray(raw.outputs['out_signals'].value), 4 * ref, rtol=1e-5)

    @cpu_and_gpu
    def test_pupil_size_scaling(self, target_device_idx, xp):
        # a node at a known offset is sampled at u * D_ref / D_sim: with the
        # characterization's D_ref equal to the simulation diameter it is sampled
        # at u itself, and the expected grid index follows from the FFT sampling
        offset_um = 2.5
        char = make_char(mesh_xy=[[offset_um, 0.0]], target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        run(obj, make_ef(target_device_idx=target_device_idx))

        lam_m = WL[0] * 1e-9
        padded = PUPIL * CHAR_PAD
        dtheta = lam_m / (padded * PIXEL_PITCH)
        expected = [offset_um * 1e-6 / dtheta + padded / 2.0 for _ in WL]

        iy0, iy1, ix0, ix1, w00, w01, w10, w11, n_inside = obj._bilinear[0]
        self.assertEqual(n_inside, 1)
        self.assertEqual(int(cpuArray(ix0)[0]), int(np.floor(expected[0])))
        self.assertAlmostEqual(float(cpuArray(w01)[0]), expected[0] - np.floor(expected[0]), places=5)
        # the y axis is not offset
        self.assertEqual(int(cpuArray(iy0)[0]), padded // 2)

    @cpu_and_gpu
    def test_signals_are_flat_when_the_pupil_moves_off_the_lantern(self, target_device_idx, xp):
        # a uniform pupil has a delta-like focal spot: moving the sampled node far
        # away must give essentially no power
        char = make_char(mesh_xy=[[0.0, 0.0], [400.0, 0.0]], n_modes=2, n_ports=2,
                         target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        run(obj, make_ef(target_device_idx=target_device_idx))
        signals = cpuArray(obj.outputs['out_signals'].value)
        total_center = PUPIL ** 4
        self.assertGreater(signals[0, 0], 0.9 * total_center)
        self.assertLess(signals[0, 1], 1e-6 * total_center)

    @cpu_and_gpu
    def test_outputs_and_generation_time(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        run(obj, make_ef(t=7, target_device_idx=target_device_idx), t=7)
        self.assertEqual(obj.outputs['out_signals'].value.shape, (len(WL), 1))
        self.assertEqual(obj.outputs['out_mode_amplitudes'].value.shape, (len(WL), 1))
        self.assertEqual(obj.outputs['out_throughput'].value.shape, (len(WL),))
        for name in ('out_signals', 'out_mode_amplitudes', 'out_throughput'):
            self.assertEqual(obj.outputs[name].generation_time, 7)

    @cpu_and_gpu
    def test_wavelength_subset(self, target_device_idx, xp):
        char = make_char(wavelengths=[1150.0, 1200.0, 1250.0], target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                wavelength_indexes=[0, 2], target_device_idx=target_device_idx)
        self.assertEqual(obj.pl_char.n_wl, 2)
        np.testing.assert_allclose(cpuArray(obj.pl_char.wavelengths_nm), [1150.0, 1250.0])
        run(obj, make_ef(target_device_idx=target_device_idx))
        self.assertEqual(obj.outputs['out_signals'].value.shape, (2, 1))

    @cpu_and_gpu
    def test_validation(self, target_device_idx, xp):
        char = make_char(target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):     # no characterization
            DispersedLanternSensor(make_simul_params(), None)
        with self.assertRaises(ValueError):     # bad pad factor
            DispersedLanternSensor(make_simul_params(), char, pad_factor=0)
        # a mesh node outside the focal-plane grid
        far = make_char(mesh_xy=[[1e5, 0.0]], target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), far, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            run(obj, make_ef(target_device_idx=target_device_idx))
        # a non-square field
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        ef = ElectricField(PUPIL, PUPIL + 2, PIXEL_PITCH, target_device_idx=target_device_idx)
        ef.generation_time = 1
        with self.assertRaises(ValueError):
            run(obj, ef)

    @cpu_and_gpu
    def test_pupil_mask_warning(self, target_device_idx, xp):
        # a characterization pupil that does not match the simulation one: only a
        # warning, the signals are then approximate
        char = make_char(pupil_mask=np.ones((PUPIL, PUPIL)), target_device_idx=target_device_idx)
        obj = DispersedLanternSensor(make_simul_params(), char, pad_factor=CHAR_PAD,
                                target_device_idx=target_device_idx)
        ef = make_ef(target_device_idx=target_device_idx)
        ef.A[:] = 0
        ef.A[:2, :2] = 1            # a pupil much smaller than the characterization one
        with self.assertLogs('specula', level='WARNING') as logs:
            run(obj, ef)
        self.assertTrue(any('characterization pupil' in line for line in logs.output))

    def test_names(self):
        self.assertIn('in_ef', DispersedLanternSensor.input_names())
        for name in ('out_signals', 'out_mode_amplitudes', 'out_throughput'):
            self.assertIn(name, DispersedLanternSensor.output_names())


if __name__ == '__main__':
    unittest.main()
