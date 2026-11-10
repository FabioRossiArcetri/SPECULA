import logging
import numpy as np

from specula import xp_fft
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.lantern_characterization import LanternCharacterization
from specula.data_objects.simul_params import SimulParams


class DispersedLanternSensor(BaseProcessingObj):
    """
    Dispersed photonic-lantern piston sensor.

    Propagates the input ElectricField through the lantern at every wavelength of
    a :class:`~specula.data_objects.lantern_characterization.LanternCharacterization`, and
    outputs the power in each output channel at each wavelength. The detector the
    lantern output is dispersed onto is *not* modelled: the output is the ideal
    per-wavelength photometry, ready for a
    :class:`~specula.processing_objects.lantern_cubic_reconstructor.LanternCubicReconstructor`.

    At each wavelength the object

    1. builds the pupil field ``E = A_pupil * exp(i * 2*pi*phase_nm/lambda)``,
       normalized so that its mean intensity over the pupil is 1 (the
       characterization's power fractions are fractions of the incident power);
    2. zero-pads it by ``pad_factor`` and Fourier-transforms it, giving the
       focal-plane field on a grid with ``pad_factor`` pixels per ``lambda/D``;
    3. samples it, with bilinear interpolation, at the lantern input-face mesh
       nodes of the characterization, rescaled so that the diffraction pattern
       has the same size relative to the lantern as in the characterization
       (see the ``psf_scale`` parameter);
    4. projects the sampled field onto the lantern's LP modes,
       ``g = mode_projector @ E_mesh``;
    5. propagates through the lantern, ``c = channel_matrix @ g``, and takes
       ``|c|**2`` as the power in each channel.

    Everything is linear per wavelength, so the whole chain is two matrix products
    plus one FFT.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters, used for the pupil size (the physical scale of the
        diffraction pattern) and the pupil pixel pitch.
    pl_char : LanternCharacterization
        The characterization: matrices, mesh and focal-plane sampling per
        wavelength.
    pad_factor : int [1]
        Zero-padding factor of the focal-plane FFT (default 4, the value the
        characterizations were computed with). It sets the sampling of the focal
        plane, ``pad_factor`` pixels per ``lambda/D`` at the design wavelength of
        the characterization; it does not need to match the characterization.
    psf_scale : float [1], optional
        Multiplier applied to the rescaling of the mesh positions onto the
        simulation pupil: 1.0 (default) means that the diffraction pattern has the
        same size relative to the lantern as in the characterization, which is
        obtained from the equivalent pupil diameter stored in the
        characterization. Override only for a non-standard focal length; the
        signals then no longer match the characterization.
    wavelength_indexes : list of int, optional
        Subset of the characterization's wavelengths to propagate (default: all).
    normalize_amplitude : bool
        Divide the pupil field by the root-mean-square of its amplitude over the
        pupil (default True), so that the signals are fractions of the incident
        power as the characterization defines them. Set to False to use the
        simulation's absolute amplitude.
    target_device_idx : int [1], optional
        Target device for computation (CPU/GPU). Default is None
        (uses global setting).
    precision : int [1], optional
        Precision for computation (0 for double, 1 for single). Default is None
        (uses global setting).

    Outputs
    -------
    out_signals : BaseValue
        ``(n_wl, n_ports)`` power in each output channel at each wavelength, as a
        fraction of the incident power.
    out_mode_amplitudes : BaseValue
        ``(n_wl, n_modes)`` complex LP-mode amplitudes of each wavelength, as
        projected from the focal plane (diagnostics).
    out_throughput : BaseValue
        ``(n_wl,)`` total power over the output channels at each wavelength,
        i.e. the fraction of the incident power that coupled into the lantern.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 pl_char: LanternCharacterization,
                 pad_factor: int = 4,
                 psf_scale: float = None,
                 wavelength_indexes=None,
                 normalize_amplitude: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if pl_char is None:
            raise ValueError('DispersedLanternSensor needs a pl_char')
        if pad_factor is None or pad_factor < 1:
            raise ValueError('DispersedLanternSensor pad_factor must be >= 1')

        self.simul_params = simul_params
        self.pl_char = pl_char if wavelength_indexes is None \
            else pl_char.subset(wavelength_indexes)
        self.pad_factor = int(pad_factor)
        self.psf_scale = None if psf_scale is None else float(psf_scale)
        self.normalize_amplitude = bool(normalize_amplitude)

        # Filled in setup()
        self._fsize = None
        self._pad = None
        self._center = (0.0, 0.0)
        self._padded = None
        self._amp_mask = None
        self._mask_count = 0
        self._bilinear = []

        self.signals = BaseValue('power in each lantern output channel, per wavelength',
                                 target_device_idx=target_device_idx, precision=precision)
        self.mode_amplitudes = BaseValue('LP-mode amplitudes of each wavelength',
                                         target_device_idx=target_device_idx, precision=precision)
        self.throughput = BaseValue('total power coupled into the lantern, per wavelength',
                                    target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_signals'] = self.signals
        self.outputs['out_mode_amplitudes'] = self.mode_amplitudes
        self.outputs['out_throughput'] = self.throughput

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field from the telescope pupil')}

    @classmethod
    def output_names(cls):
        return {
            'out_signals': OutputDesc(BaseValue, 'Power in each lantern output channel, per wavelength'),
            'out_mode_amplitudes': OutputDesc(BaseValue, 'LP-mode amplitudes of each wavelength'),
            'out_throughput': OutputDesc(BaseValue, 'Total power coupled into the lantern, per wavelength'),
        }

    def _sampling(self, wavelength_index, in_ef):
        '''
        Bilinear interpolation indices and weights of the mesh nodes on the padded
        focal-plane grid, at one wavelength.

        The characterization fixes the size of the diffraction pattern relative to
        the lantern: a pupil of diameter ``D_ref`` produces the same pattern as the
        one the characterization was computed with. A simulation pupil of diameter
        ``D_sim`` gives a narrower pattern, so the mesh node at ``u`` [um] must be
        sampled at the angle ``u * 1e-6 * D_ref / D_sim`` [rad].
        '''
        xp = self.xp
        char = self.pl_char
        w = wavelength_index

        d_ref = float(char.reference_pupil_diameter_m[w])
        d_sim = self.simul_params.pixel_pupil * self.simul_params.pixel_pitch
        scale = d_ref / d_sim
        if self.psf_scale is not None:
            scale *= self.psf_scale

        wavelength_m = float(char.wavelengths_nm[w]) * 1e-9
        # FFT of the padded (ny, nx) grid with pitch dp gives lambda / (n * dp) per pixel
        dtheta_y = wavelength_m / (self._padded_size[0] * self.simul_params.pixel_pitch)
        dtheta_x = wavelength_m / (self._padded_size[1] * self.simul_params.pixel_pitch)

        mesh_x = char.mesh_xy_um[:, 0]
        mesh_y = char.mesh_xy_um[:, 1]
        pos_x = (mesh_x * 1e-6 * scale) / dtheta_x + self._center
        pos_y = (mesh_y * 1e-6 * scale) / dtheta_y + self._center

        iy0 = xp.floor(pos_y).astype(xp.int64)
        ix0 = xp.floor(pos_x).astype(xp.int64)
        wy = pos_y - iy0
        wx = pos_x - ix0
        valid = ((iy0 >= 0) & (iy0 + 1 < self._padded_size[0]) &
                 (ix0 >= 0) & (ix0 + 1 < self._padded_size[1]))
        iy0 = xp.clip(iy0, 0, self._padded_size[0] - 1)
        ix0 = xp.clip(ix0, 0, self._padded_size[1] - 1)
        iy1 = xp.clip(iy0 + 1, 0, self._padded_size[0] - 1)
        ix1 = xp.clip(ix0 + 1, 0, self._padded_size[1] - 1)
        valid = valid.astype(self.dtype)
        return (iy0, iy1, ix0, ix1,
                (1 - wy) * (1 - wx) * valid, (1 - wy) * wx * valid,
                wy * (1 - wx) * valid, wy * wx * valid,
                int(xp.sum(valid)))

    def setup(self):
        super().setup()

        in_ef = self.local_inputs['in_ef']
        if in_ef is None:
            raise ValueError("'in_ef' must be given as an input")
        if self.simul_params is None or self.simul_params.pixel_pitch is None:
            raise ValueError('DispersedLanternSensor needs simul_params with pixel_pitch')

        self._fsize = (int(in_ef.size[0]), int(in_ef.size[1]))
        pixel_pupil = self.simul_params.pixel_pupil
        if pixel_pupil is not None and self._fsize[0] != pixel_pupil:
            self.logger.warning(f'the input field is {self._fsize[0]} px while pixel_pupil is '
                                f'{pixel_pupil}: the pupil diameter of the simulation is taken '
                                f'from pixel_pupil, the FFT from the field')
        if self._fsize[0] != self._fsize[1]:
            raise ValueError(f'DispersedLanternSensor needs a square pupil, not {self._fsize}')

        self._pad = (self.pad_factor * self._fsize[0] - self._fsize[0]) // 2
        size = self._fsize[0] + 2 * self._pad
        self._padded_size = (size, size)
        self._center = (size - 1) / 2.0 if size % 2 else size / 2.0

        n_wl, n_ports, n_modes = self.pl_char.n_wl, self.pl_char.n_ports, self.pl_char.n_modes
        self._wavelengths_nm = [float(self.pl_char.wavelengths_nm[w]) for w in range(n_wl)]
        self.signals.value = self.xp.zeros((n_wl, n_ports), dtype=self.dtype)
        # the wavelengths of the rows of out_signals, for the reconstructor to match
        self.signals.wavelengths_nm = list(self._wavelengths_nm)
        self.mode_amplitudes.value = self.xp.zeros((n_wl, n_modes), dtype=self.complex_dtype)
        self.throughput.value = self.xp.zeros(n_wl, dtype=self.dtype)

        self._padded = self.xp.zeros(self._padded_size, dtype=self.complex_dtype)
        # The pupil amplitude is only known at the first trigger: an upstream
        # AtmoPropagation applies the pupilstop while it runs, so at setup time the
        # field can still be a uniform amplitude. The mask is therefore filled by
        # _build_amplitude_mask() at the first trigger (see prepare_trigger()).
        self._amp_mask = None
        self._mask_count = 0

        self._bilinear = [self._sampling(w, in_ef) for w in range(n_wl)]
        outside = [n for (_, _, _, _, _, _, _, _, n) in self._bilinear]
        if any(n == 0 for n in outside):
            raise ValueError('some mesh nodes fall outside the focal-plane grid: '
                             'check pad_factor and the pupil size')
        if any(n < self.pl_char.n_mesh for n in outside):
            self.logger.warning(f'mesh nodes outside the focal-plane grid at some wavelengths '
                                f'({outside} of {self.pl_char.n_mesh} inside)')

    def _build_amplitude_mask(self):
        '''Pupil amplitude mask, from the field of the first trigger'''
        amp = self.local_inputs['in_ef'].A
        mask = (amp > 0).astype(self.dtype)
        count = int(self.xp.sum(mask))
        if count == 0:
            raise ValueError('the input pupil amplitude is zero everywhere')
        self._amp_mask = mask
        self._mask_count = count
        pupil_mask = self.pl_char.pupil_mask
        if pupil_mask is not None:
            overlap = float(self.xp.sum((pupil_mask > 0) & (amp > 0))) / \
                max(float(self.xp.sum(pupil_mask > 0)), 1.0)
            if overlap < 0.99:
                self.logger.warning(f'the simulation pupil overlaps the characterization pupil '
                                    f'by {100 * overlap:.1f}%: the signals are then only '
                                    f'approximate')

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        if self._amp_mask is None:
            self._build_amplitude_mask()

    def trigger_code(self):
        xp = self.xp
        in_ef = self.local_inputs['in_ef']
        char = self.pl_char
        fft = xp_fft(xp)
        amp = in_ef.A
        n_pad = self._pad
        ny, nx = self._fsize

        if self.normalize_amplitude:
            rms = xp.sqrt(xp.sum(amp * amp * self._amp_mask) / self._mask_count)
            scale = 1.0 / rms
        else:
            scale = self.dtype(1.0)

        for w in range(char.n_wl):
            phase = in_ef.phi_at_lambda(self._wavelengths_nm[w])
            E = amp * scale * xp.exp(self.complex_dtype(1j) * phase)
            self._padded[...] = 0
            self._padded[n_pad:n_pad + ny, n_pad:n_pad + nx] = E
            # the pupil field is centered in its array, the FFT wants it at the origin
            focal = fft.fftshift(fft.fft2(fft.ifftshift(self._padded)))

            iy0, iy1, ix0, ix1, w00, w01, w10, w11, _ = self._bilinear[w]
            e_mesh = (focal[iy0, ix0] * w00 + focal[iy0, ix1] * w01 +
                      focal[iy1, ix0] * w10 + focal[iy1, ix1] * w11)

            g = char.mode_projector[w] @ e_mesh
            c = char.channel_matrix[w] @ g
            self.mode_amplitudes.value[w] = g
            self.signals.value[w] = xp.abs(c) ** 2

    def post_trigger(self):
        super().post_trigger()
        self.throughput.value[:] = self.xp.sum(self.signals.value, axis=1)
        self.signals.generation_time = self.current_time
        self.mode_amplitudes.generation_time = self.current_time
        self.throughput.generation_time = self.current_time
