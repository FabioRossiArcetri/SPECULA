from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from iteras.lib import storage


def to_real_imag(array, xp):
    '''
    Complex array as a real array with a trailing dimension of 2 (real part,
    imaginary part): FITS images cannot store complex data, so the complex
    matrices of a PL characterization are always written this way.
    '''
    return xp.stack((array.real, array.imag), axis=-1)


def from_real_imag(array):
    '''Inverse of :func:`to_real_imag` (numpy or cupy array in, numpy out)'''
    array = array.get() if hasattr(array, 'get') else array
    return array[..., 0] + 1j * array[..., 1]


class LanternCharacterization(BaseDataObj):
    """
    Photonic-lantern characterization data object.

    Holds, for a series of wavelengths, the two matrices that make the lantern a
    *linear* sensor from the telescope focal plane to its output channels:

    * ``mode_projector``, shape ``(n_modes, n_mesh)`` per wavelength: the projection
      of the field sampled at the lantern input face (``mesh_xy_um``) onto the
      lantern's LP modes. Row k is ``conj(v_k)^T B``, with ``B`` the
      finite-element mass matrix of the mesh, so that
      ``g = mode_projector @ E_mesh`` are the LP-mode amplitudes;
    * ``channel_matrix``, shape ``(n_ports, n_modes)`` per wavelength: the linear
      propagation from those mode amplitudes to the complex amplitude of each
      output channel, divided by the incident-power normalization. The power in
      each channel is then ``|channel_matrix @ g|**2``, i.e. the fraction of the
      incident power coupled into it.

    Both matrices come from the same cbeam characterization and are only
    meaningful together, and only at their own wavelength: the mode basis is
    solved per wavelength and its gauge (signs, mixing within degenerate groups)
    is fixed independently at each wavelength, so ``channel_matrix`` is not
    comparable across wavelengths. Only the channel *powers* are, and they are
    what the reconstructors are trained on.

    Parameters
    ----------
    wavelengths_nm : array_like [nm]
        Wavelengths of the characterization, increasing.
    mesh_xy_um : array_like
        Shape ``(n_mesh, 2)``: physical position [um] of the mesh nodes on the
        lantern input face, the same for every wavelength.
    mode_projector : array_like
        Shape ``(n_wl, n_modes, n_mesh)``, complex.
    channel_matrix : array_like
        Shape ``(n_wl, n_ports, n_modes)``, complex.
    pixel_scale_um : array_like [um/pixel]
        Focal-plane sampling used by the characterization at each wavelength.
        It fixes the equivalent pupil diameter of the characterization, see
        :attr:`reference_pupil_diameter_m`.
    focal_plane_norm : array_like
        Incident-power normalization of each wavelength, as used by the
        characterization to turn channel amplitudes into power fractions.
    pupil_mask : array_like, optional
        2D pupil amplitude mask the characterization was computed with. Used to
        check that a simulation uses a compatible pupil.
    lantern : str
        Name of the lantern (e.g. ``'pl19_1200'``), for bookkeeping.
    pad_factor : int [1]
        Zero-padding factor used for the focal-plane FFT by the characterization.
        Together with ``pixel_scale_um`` it defines the equivalent pupil
        diameter; it is *not* the padding factor of the simulation.
    design_wavelength_nm : float [nm], optional
        Design wavelength the lantern geometry was specified at.
    psf_fill_factor : float [1], optional
        Airy radius / cladding radius ratio the characterization was computed
        with (0.50 for the reference designs), for bookkeeping.
    target_device_idx : int [1], optional
        Target device for computation (CPU/GPU). Default is None
        (uses global setting).
    precision : int [1], optional
        Precision for computation (0 for double, 1 for single). Default is None
        (uses global setting).
    """

    def __init__(self,
                 wavelengths_nm=None,
                 mesh_xy_um=None,
                 mode_projector=None,
                 channel_matrix=None,
                 pixel_scale_um=None,
                 focal_plane_norm=None,
                 pupil_mask=None,
                 lantern: str='',
                 pad_factor: int=4,
                 design_wavelength_nm: float=None,
                 psf_fill_factor: float=None,
                 target_device_idx: int=None,
                 precision: int=None):

        super().__init__(precision=precision, target_device_idx=target_device_idx)

        wavelengths_nm = self.xp.asarray(wavelengths_nm, dtype=self.dtype).ravel()
        n_wl = wavelengths_nm.shape[0]

        if n_wl < 1:
            raise ValueError('LanternCharacterization needs at least one wavelength')
        if self.xp.any(self.xp.diff(wavelengths_nm) <= 0):
            raise ValueError('LanternCharacterization wavelengths must be increasing')

        mesh_xy_um = self.xp.asarray(mesh_xy_um, dtype=self.dtype)
        if mesh_xy_um.ndim != 2 or mesh_xy_um.shape[1] != 2:
            raise ValueError(f'mesh_xy_um must have shape (n_mesh, 2), not {mesh_xy_um.shape}')
        n_mesh = mesh_xy_um.shape[0]

        mode_projector = self.to_xp(mode_projector, dtype=self.complex_dtype)
        channel_matrix = self.to_xp(channel_matrix, dtype=self.complex_dtype)

        if mode_projector.ndim != 3 or mode_projector.shape[0] != n_wl:
            raise ValueError(f'mode_projector must have shape (n_wl, n_modes, n_mesh) with '
                             f'n_wl={n_wl}, not {mode_projector.shape}')
        if mode_projector.shape[2] != n_mesh:
            raise ValueError(f'mode_projector needs {n_mesh} mesh points, '
                             f'not {mode_projector.shape[2]}')
        if channel_matrix.ndim != 3 or channel_matrix.shape[0] != n_wl:
            raise ValueError(f'channel_matrix must have shape (n_wl, n_ports, n_modes) with '
                             f'n_wl={n_wl}, not {channel_matrix.shape}')
        if channel_matrix.shape[2] != mode_projector.shape[1]:
            raise ValueError(f'channel_matrix has {channel_matrix.shape[2]} modes while '
                             f'mode_projector has {mode_projector.shape[1]}')

        pixel_scale_um = self.xp.asarray(pixel_scale_um, dtype=self.dtype).ravel()
        focal_plane_norm = self.xp.asarray(focal_plane_norm, dtype=self.dtype).ravel()
        for name, value in (('pixel_scale_um', pixel_scale_um),
                            ('focal_plane_norm', focal_plane_norm)):
            if value.shape[0] != n_wl:
                raise ValueError(f'{name} has {value.shape[0]} values for {n_wl} wavelengths')
        if self.xp.any(pixel_scale_um <= 0):
            raise ValueError('LanternCharacterization pixel_scale_um must be positive')

        if pad_factor is None or pad_factor <= 0:
            raise ValueError('LanternCharacterization pad_factor must be positive')

        self.wavelengths_nm = wavelengths_nm
        self.mesh_xy_um = mesh_xy_um
        self.mode_projector = mode_projector
        self.channel_matrix = channel_matrix
        self.pixel_scale_um = pixel_scale_um
        self.focal_plane_norm = focal_plane_norm
        self.pupil_mask = None if pupil_mask is None else self.to_xp(pupil_mask, dtype=self.dtype)
        self.lantern = lantern
        self.pad_factor = int(pad_factor)
        self.design_wavelength_nm = design_wavelength_nm
        self.psf_fill_factor = psf_fill_factor

    @property
    def n_wl(self):
        '''Number of wavelengths in the characterization'''
        return self.wavelengths_nm.shape[0]

    @property
    def n_ports(self):
        '''Number of output channels of the lantern'''
        return self.channel_matrix.shape[1]

    @property
    def n_modes(self):
        '''Number of LP modes tracked at the lantern input'''
        return self.channel_matrix.shape[2]

    @property
    def n_mesh(self):
        '''Number of nodes of the lantern input-face mesh'''
        return self.mesh_xy_um.shape[0]

    @property
    def reference_pupil_diameter_m(self):
        '''
        Equivalent pupil diameter of the characterization [m], one value per
        wavelength.

        The characterization fixes the ratio between the diffraction pattern and
        the lantern geometry: with ``lambda / D = pad_factor * pixel_scale_um``
        at the focal length of 1 m the characterization implicitly uses, a pupil
        of diameter ``D`` produces the same pattern as the characterization. A
        simulation with a different pupil diameter must rescale the mesh
        positions accordingly (see the dispersed PL sensor).
        '''
        # wavelengths_nm * 1e-3 = wavelength in um, so that the ratio is in metres
        return self.wavelengths_nm * 1e-3 / (self.pad_factor * self.pixel_scale_um)

    def wavelength_index(self, wavelength_nm, atol: float=0.5):
        '''
        Index of *wavelength_nm* in the characterization.

        Raises ValueError if the wavelength is not part of the characterization:
        the matrices are computed per wavelength and cannot be interpolated.
        '''
        diff = self.xp.abs(self.wavelengths_nm - float(wavelength_nm))
        idx = int(self.xp.argmin(diff))
        if float(diff[idx]) > atol:
            raise ValueError(f'wavelength {wavelength_nm} nm is not in the PL characterization '
                             f'({self.wavelengths_nm.min()}-{self.wavelengths_nm.max()} nm)')
        return idx

    def indexes_for(self, wavelengths_nm):
        '''Indexes of a sequence of wavelengths, see :meth:`wavelength_index`'''
        return [self.wavelength_index(w) for w in wavelengths_nm]

    def subset(self, indexes):
        '''
        New characterization with only the given wavelength *indexes*, e.g. the
        ones a reconstructor was trained at. The arrays are not copied on the
        device, so this stays cheap.
        '''
        indexes = [int(i) for i in indexes]
        if not indexes:
            raise ValueError('LanternCharacterization.subset() needs at least one index')
        return LanternCharacterization(
            wavelengths_nm=self.wavelengths_nm[indexes],
            mesh_xy_um=self.mesh_xy_um,
            mode_projector=self.mode_projector[indexes],
            channel_matrix=self.channel_matrix[indexes],
            pixel_scale_um=self.pixel_scale_um[indexes],
            focal_plane_norm=self.focal_plane_norm[indexes],
            pupil_mask=self.pupil_mask,
            lantern=self.lantern,
            pad_factor=self.pad_factor,
            design_wavelength_nm=self.design_wavelength_nm,
            psf_fill_factor=self.psf_fill_factor,
            target_device_idx=self.target_device_idx,
            precision=self.precision,
        )

    def get_value(self):
        return self.mode_projector

    def set_value(self, v):
        '''Set a new mode projector. Arrays are not reallocated.'''
        assert v.shape == self.mode_projector.shape, \
            f'Error: input array shape {v.shape} does not match {self.mode_projector.shape}'
        self.mode_projector[:] = self.to_xp(v)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'LanternCharacterization'
        hdr['LANTERN'] = self.lantern
        hdr['NWL'] = self.n_wl
        hdr['NPORTS'] = self.n_ports
        hdr['NMODES'] = self.n_modes
        hdr['NMESH'] = self.n_mesh
        hdr['PADFACT'] = self.pad_factor
        if self.design_wavelength_nm is not None:
            hdr['DESWL'] = float(self.design_wavelength_nm)
        if self.psf_fill_factor is not None:
            hdr['PSFFILL'] = float(self.psf_fill_factor)
        return hdr

    def save(self, filename, overwrite=False):
        storage.write_object(filename, self.get_fits_header(),
                             {'wavelengths_nm': self.wavelengths_nm,
                              'mesh_xy_um': self.mesh_xy_um,
                              'mode_projector_ri': to_real_imag(self.mode_projector, self.xp),
                              'channel_matrix_ri': to_real_imag(self.channel_matrix, self.xp),
                              'pixel_scale_um': self.pixel_scale_um,
                              'focal_plane_norm': self.focal_plane_norm,
                              'pupil_mask': None if self.pupil_mask is None
                              else self.pupil_mask.astype(self.xp.uint8)},
                             overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
            # 'PLCharacterization' is the name earlier versions of the exporter wrote
            if hdr.get('OBJ_TYPE') not in ('LanternCharacterization', 'PLCharacterization'):
                raise ValueError(f'Error: file {filename} does not contain a LanternCharacterization object')
            version = int(hdr.get('VERSION', 0))
            if version != 1:
                raise ValueError(f'Error: unknown version {version} in file {filename}')
            pupil_mask = f.load('pupil_mask', target_device_idx=-1) \
                if f.has('pupil_mask') else None
            return LanternCharacterization(
                wavelengths_nm=f.load('wavelengths_nm', target_device_idx=-1),
                mesh_xy_um=f.load('mesh_xy_um', target_device_idx=-1),
                mode_projector=from_real_imag(f.read('mode_projector_ri')),
                channel_matrix=from_real_imag(f.read('channel_matrix_ri')),
                pixel_scale_um=f.load('pixel_scale_um', target_device_idx=-1),
                focal_plane_norm=f.load('focal_plane_norm', target_device_idx=-1),
                pupil_mask=pupil_mask,
                lantern=hdr.get('LANTERN', ''),
                pad_factor=int(hdr.get('PADFACT', 4)),
                design_wavelength_nm=hdr.get('DESWL', None),
                psf_fill_factor=hdr.get('PSFFILL', None),
                target_device_idx=target_device_idx)

    def array_for_display(self):
        '''Figure of merit for display: total channel power at every wavelength'''
        return self.xp.sum(self.xp.abs(self.channel_matrix) ** 2, axis=1)
