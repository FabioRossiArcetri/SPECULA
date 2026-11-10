from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from iteras.lib import storage


class LanternPistonModel(BaseDataObj):
    """
    Common interface of the per-wavelength piston reconstructors of a dispersed
    photonic lantern.

    A model turns the channel powers measured at one wavelength into a *wrapped*
    piston estimate, in ``[-period/2, period/2)``, because a piston is only
    observable modulo the wavelength. The estimates of all wavelengths are then
    fused into the absolute piston by a
    :class:`~specula.processing_objects.spectral_piston_unwrapper.SpectralPistonUnwrapper`.

    Every strategy (the sparse cubic regression of
    :class:`~specula.data_objects.lantern_cubic_model.LanternCubicModel`, the
    analytical forward model of
    :class:`~specula.data_objects.lantern_closed_form_model.LanternClosedFormModel`,
    and any future one) shares the state kept here -- the wavelength grid, the
    periods, the standardisation of the input signals, the per-wavelength errors
    used to weight the unwrap, and its settings -- and implements
    :meth:`predict_pistons`. A simulation chooses the strategy by naming the class
    of its ``pl_model`` entry in the configuration, nothing else changes.

    Parameters
    ----------
    wavelengths_nm : array_like [nm]
        Wavelengths the models were fitted at, increasing.
    periods_nm : array_like [nm]
        Wrap period of each model (the wavelength itself, unless the fit used a
        fraction of it), one per wavelength.
    input_mean, input_std : array_like
        Shape ``(n_wl, n_inputs)``: standardisation of the input signals.
    sigma_nm : array_like
        Shape ``(n_wl, n_outputs)``: wrapped-piston error of each model on its
        training set [nm], used to weight the wavelengths in the unwrap.
    span_nm : float [nm], optional
        Largest piston the unwrap searches for (its default capture range).
    conf_min : float [1]
        Confidence below which the unwrap ignores a wavelength.
    huber_c : float [1]
        Huber threshold of the robust unwrap refinement.
    n_irls : int [1]
        Number of robust refinement iterations of the unwrap.
    pl_name : str
        Lantern name, for bookkeeping.
    """

    def __init__(self,
                 wavelengths_nm=None,
                 periods_nm=None,
                 input_mean=None,
                 input_std=None,
                 sigma_nm=None,
                 span_nm: float = None,
                 conf_min: float = 0.15,
                 huber_c: float = 2.0,
                 n_irls: int = 12,
                 pl_name: str = '',
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(precision=precision, target_device_idx=target_device_idx)

        wavelengths_nm = self.xp.asarray(wavelengths_nm, dtype=self.dtype).ravel()
        periods_nm = self.xp.asarray(periods_nm, dtype=self.dtype).ravel()
        n_wl = wavelengths_nm.shape[0]

        if n_wl < 1:
            raise ValueError('a lantern piston model needs at least one wavelength')
        if self.xp.any(self.xp.diff(wavelengths_nm) <= 0):
            raise ValueError('the wavelengths of a piston model must be increasing')
        if periods_nm.shape[0] != n_wl:
            raise ValueError(f'periods_nm has {periods_nm.shape[0]} values for {n_wl} wavelengths')
        if self.xp.any(periods_nm <= 0):
            raise ValueError('the periods of a piston model must be positive')

        input_mean = self.xp.asarray(input_mean, dtype=self.dtype)
        input_std = self.xp.asarray(input_std, dtype=self.dtype)
        sigma_nm = self.xp.asarray(sigma_nm, dtype=self.dtype)
        if input_mean.ndim != 2 or input_mean.shape[0] != n_wl:
            raise ValueError(f'input_mean must have shape (n_wl, n_inputs), not {input_mean.shape}')
        if input_std.shape != input_mean.shape:
            raise ValueError(f'input_std must have shape {input_mean.shape}, not {input_std.shape}')
        if self.xp.any(input_std <= 0):
            raise ValueError('the input standardisation must have positive deviations')
        if sigma_nm.ndim != 2 or sigma_nm.shape[0] != n_wl:
            raise ValueError(f'sigma_nm must have shape (n_wl, n_outputs), not {sigma_nm.shape}')

        self.wavelengths_nm = wavelengths_nm
        self.periods_nm = periods_nm
        self.input_mean = input_mean
        self.input_std = input_std
        self.sigma_nm = sigma_nm
        self.span_nm = None if span_nm is None else float(span_nm)
        self.conf_min = float(conf_min)
        self.huber_c = float(huber_c)
        self.n_irls = int(n_irls)
        self.pl_name = pl_name

    # -- shared state -------------------------------------------------------
    @property
    def n_wl(self):
        '''Number of wavelengths (one model each)'''
        return self.wavelengths_nm.shape[0]

    @property
    def n_inputs(self):
        '''Number of input signals of each model (the lantern channels)'''
        return self.input_mean.shape[1]

    @property
    def n_outputs(self):
        '''Number of reconstructed pistons'''
        return self.sigma_nm.shape[1]

    def wavelength_index(self, wavelength_nm, atol: float = 0.5):
        '''
        Index of the model fitted at (or nearest to) *wavelength_nm*.

        Raises ValueError if no model is close enough: the models are fitted per
        wavelength and cannot be interpolated.
        '''
        diff = self.xp.abs(self.wavelengths_nm - float(wavelength_nm))
        idx = int(self.xp.argmin(diff))
        if float(diff[idx]) > atol:
            raise ValueError(f'no model at {wavelength_nm} nm (models: '
                             f'{self.wavelengths_nm.min()}-{self.wavelengths_nm.max()} nm)')
        return idx

    def standardise(self, signals, wavelength_index: int):
        '''Signals of one wavelength, standardised as the model was fitted'''
        signals = self.xp.asarray(signals, dtype=self.dtype).ravel()
        if signals.shape[0] != self.n_inputs:
            raise ValueError(f'expected {self.n_inputs} signals, got {signals.shape[0]}')
        return ((signals - self.input_mean[wavelength_index]) /
                self.input_std[wavelength_index])

    # -- strategy -----------------------------------------------------------
    def predict_pistons(self, signals, wavelength_index: int):
        '''
        Wrapped piston estimate (length ``n_outputs``, in
        ``[-period/2, period/2)``) and its confidence, from the *signals* of one
        wavelength (length ``n_inputs``). Implemented by each strategy.
        '''
        raise NotImplementedError('a lantern piston model must implement predict_pistons()')

    def get_value(self):
        return self.sigma_nm

    def set_value(self, v):
        assert v.shape == self.sigma_nm.shape, \
            f'Error: input array shape {v.shape} does not match {self.sigma_nm.shape}'
        self.sigma_nm[:] = self.to_xp(v)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'LanternPistonModel'
        hdr['PLNAME'] = self.pl_name
        hdr['NWL'] = self.n_wl
        hdr['NINPUTS'] = self.n_inputs
        hdr['NOUTPUTS'] = self.n_outputs
        if self.span_nm is not None:
            hdr['SPAN'] = self.span_nm
        hdr['CONFMIN'] = self.conf_min
        hdr['HUBERC'] = self.huber_c
        hdr['NIRLS'] = self.n_irls
        return hdr

    def array_for_display(self):
        '''Figure of merit for display: the per-wavelength error of the models'''
        return self.sigma_nm


def shared_headers(hdr):
    '''(wavelengths, periods, standardisation, sigma, span, settings) from a header'''
    return dict(
        span_nm=hdr.get('SPAN', None),
        conf_min=hdr.get('CONFMIN', 0.15),
        huber_c=hdr.get('HUBERC', 2.0),
        n_irls=hdr.get('NIRLS', 12),
        pl_name=hdr.get('PLNAME', ''))


def write_model(filename, header, arrays, overwrite=False):
    '''Write a piston model in the npz layout, real arrays only (see the exporters)'''
    storage.write_object(filename, header, arrays, overwrite=overwrite)
