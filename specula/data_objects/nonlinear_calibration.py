import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class NonlinearCalibration(BaseDataObj):
    """
    Per-mode nonlinear response calibration for a wavefront sensor
    (typically a non-modulated pyramid).

    Holds, for each mode, a sampled curve of "projected response" versus
    "true injected amplitude", obtained by pushing each mode in isolation
    across a range of amplitudes and recording how much of the resulting
    signal projects onto that mode's small-signal (linear-regime)
    interaction vector. In the linear regime this projected response equals
    the injected amplitude; at larger amplitudes it saturates, which is the
    "parallel channel" nonlinearity described for non-modulated pyramid
    WFSs.

    All modes share the same amplitude grid, so the calibration is stored
    as a single 1D amplitude vector plus a (nmodes, namplitudes) response
    matrix.

    Note
    ----
    This is a first, deliberately simple nonparametric (lookup-table)
    calibration meant to generalize the single global scalar of
    :class:`~specula.processing_objects.optical_gain_estimator.OpticalGainEstimator`
    to a per-mode curve. It assumes each mode's response depends only on its
    own amplitude (no cross-mode coupling in the nonlinear regime). It is
    intended to be swapped for a more rigorous, coupled analytical model as
    that becomes available.
    """

    def __init__(self,
                 amplitudes,
                 responses,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters
        ----------
        amplitudes : array-like [nsamples]
            Amplitude grid shared by every mode. Must be sorted in strictly
            increasing order.
        responses : array-like [nmodes, nsamples]
            Projected response of each mode at each amplitude in
            `amplitudes`. For a well-behaved (monotonic) sensor this is
            increasing along the `nsamples` axis for every mode.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        amplitudes = np.asarray(cpuArray(amplitudes), dtype=float)
        responses = np.asarray(cpuArray(responses), dtype=float)

        if amplitudes.ndim != 1:
            raise ValueError(f'amplitudes must be 1D, got shape {amplitudes.shape}')
        if responses.ndim != 2 or responses.shape[1] != amplitudes.shape[0]:
            raise ValueError(
                f'responses must have shape (nmodes, {amplitudes.shape[0]}), '
                f'got {responses.shape}'
            )
        if np.any(np.diff(amplitudes) <= 0):
            raise ValueError('amplitudes must be strictly increasing')

        self.amplitudes = self.to_xp(amplitudes, dtype=self.dtype)
        self.responses = self.to_xp(responses, dtype=self.dtype)

    @property
    def nmodes(self):
        return self.responses.shape[0]

    @property
    def nsamples(self):
        return self.amplitudes.shape[0]

    def get_value(self):
        '''
        Get the per-mode response calibration as a numpy/cupy array
        '''
        return self.responses

    def set_value(self, v):
        '''
        Set new values for the per-mode response calibration
        Arrays are not reallocated
        '''
        assert v.shape == self.responses.shape, \
            f"Error: input array shape {v.shape} does not match responses shape {self.responses.shape}"
        self.responses[:] = self.to_xp(v)

    def forward(self, mode_indices, amplitude):
        """
        Predict the (saturated) response that `amplitude` would produce,
        for each mode in `mode_indices`, via monotonic linear interpolation
        of the calibrated curve (extrapolating with the boundary value
        outside the calibrated range). This is the forward counterpart of
        :meth:`invert`.

        Parameters
        ----------
        mode_indices : array-like [n]
            Mode index for each entry of `amplitude`.
        amplitude : array-like [n]
            True amplitude for each entry.

        Returns
        -------
        response : ndarray [n]
            Predicted (possibly saturated) response for each entry.
        """
        mode_indices = cpuArray(mode_indices)
        amplitude = cpuArray(amplitude)
        amplitudes = cpuArray(self.amplitudes)
        responses = cpuArray(self.responses)

        out = np.empty(len(mode_indices), dtype=float)
        for i, mode in enumerate(mode_indices):
            out[i] = np.interp(amplitude[i], amplitudes, responses[mode])
        return self.to_xp(out, dtype=self.dtype)

    def invert(self, mode_indices, measured_response):
        """
        Estimate the true amplitude that produced `measured_response`,
        for each mode in `mode_indices`, by inverting that mode's
        calibrated response curve via monotonic linear interpolation
        (extrapolating with the boundary value outside the calibrated
        range).

        Parameters
        ----------
        mode_indices : array-like [n]
            Mode index for each entry of `measured_response`.
        measured_response : array-like [n]
            Observed (linear-estimate) response for each mode.

        Returns
        -------
        amplitude : ndarray [n]
            Estimated true amplitude for each entry.
        """
        mode_indices = cpuArray(mode_indices)
        measured_response = cpuArray(measured_response)
        amplitudes = cpuArray(self.amplitudes)
        responses = cpuArray(self.responses)

        out = np.empty(len(mode_indices), dtype=float)
        for i, mode in enumerate(mode_indices):
            out[i] = np.interp(measured_response[i], responses[mode], amplitudes)
        return self.to_xp(out, dtype=self.dtype)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NMODES'] = self.nmodes
        hdr['NSAMPLES'] = self.nsamples
        return hdr

    def save(self, filename, overwrite=False):
        if not filename.endswith('.fits'):
            filename += '.fits'
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.amplitudes), name='AMPLITUDES'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.responses), name='RESPONSES'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError

    @staticmethod
    def restore(filename, target_device_idx=None):
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = int(hdr['VERSION'])
            if version != 1:
                raise ValueError(f'Error: unknown version {version} in file {filename}')
            amplitudes = hdul['AMPLITUDES'].data.copy()
            responses = hdul['RESPONSES'].data.copy()
        return NonlinearCalibration(amplitudes, responses, target_device_idx=target_device_idx)
