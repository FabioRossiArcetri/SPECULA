from astropy.io import fits

from specula.data_objects.lantern_piston_model import LanternPistonModel, shared_headers
from iteras.lib import storage


class LanternCubicModel(LanternPistonModel):
    """
    Sparse polynomial reconstructor for a dispersed photonic lantern.

    One model per wavelength, mapping the lantern's output channel powers at that
    wavelength to the *phasor* of each piston, ``[sin, cos](2*pi*p/period)``.
    Regressing the phasor instead of the piston removes the jump at
    ``+-period/2``: a piston is only observable modulo the wavelength, so each
    per-wavelength model returns a wrapped estimate, which a
    :class:`~specula.processing_objects.spectral_piston_unwrapper.SpectralPistonUnwrapper`
    fuses across wavelengths into the absolute piston.

    Only the non-zero terms of the sparse fit are stored: for each wavelength a
    slice of a flat table of monomials, given by their exponents, with one weight
    per target, plus an offset. The input standardisation and the standardisation
    of the monomials are folded into the weights, so the evaluation is

    ``phasors = monomials((signals - input_mean) / input_std)[terms] @ weights + offset``

    i.e. a gather, a product and a small matrix product per wavelength, with no
    dependency on the fitting library that produced it.

    Parameters
    ----------
    wavelengths_nm, periods_nm, input_mean, input_std, sigma_nm, span_nm,
    conf_min, huber_c, n_irls, pl_name
        As in :class:`~specula.data_objects.lantern_piston_model.LanternPistonModel`.
    n_terms : array_like [1]
        Number of terms of each model, one per wavelength; the terms of
        wavelength ``w`` are ``powers[start:stop]``, with ``start`` the cumulative
        sum of the previous ones.
    powers : array_like
        Shape ``(total_terms, n_inputs)``: the monomial exponents of each term.
    weights : array_like
        Shape ``(total_terms, 2 * n_outputs)``: weight of each term for each
        target, in the order ``[sin_0, ..., sin_{M-1}, cos_0, ..., cos_{M-1}]``.
    offsets : array_like
        Shape ``(n_wl, 2 * n_outputs)``: the model offset of each target.

    Outputs
    -------
    predict_pistons(signals, wavelength_index) -> (pistons, confidence)
        Wrapped pistons [nm] and the confidence of each estimate (the phasor
        length, about 1 when the model is sure).
    """

    def __init__(self,
                 wavelengths_nm=None,
                 periods_nm=None,
                 n_terms=None,
                 powers=None,
                 weights=None,
                 offsets=None,
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

        super().__init__(wavelengths_nm=wavelengths_nm, periods_nm=periods_nm,
                         input_mean=input_mean, input_std=input_std, sigma_nm=sigma_nm,
                         span_nm=span_nm, conf_min=conf_min, huber_c=huber_c,
                         n_irls=n_irls, pl_name=pl_name,
                         target_device_idx=target_device_idx, precision=precision)

        n_terms = self.xp.asarray(n_terms).astype(self.xp.int32).ravel()
        powers = self.xp.asarray(powers).astype(self.xp.int8)
        weights = self.xp.asarray(weights, dtype=self.dtype)
        offsets = self.xp.asarray(offsets, dtype=self.dtype)

        if n_terms.shape[0] != self.n_wl:
            raise ValueError(f'n_terms has {n_terms.shape[0]} values for {self.n_wl} wavelengths')
        if self.xp.any(n_terms < 1):
            raise ValueError('n_terms must be at least 1 per wavelength')

        if powers.ndim != 2:
            raise ValueError(f'powers must have shape (total_terms, n_inputs), not {powers.shape}')
        total_terms = powers.shape[0]
        if int(self.xp.sum(n_terms)) != total_terms:
            raise ValueError(f'the models claim {int(self.xp.sum(n_terms))} terms, powers has {total_terms}')
        if powers.shape[1] != self.n_inputs:
            raise ValueError(f'powers has {powers.shape[1]} inputs, the standardisation has '
                             f'{self.n_inputs}')
        if self.xp.any(powers < 0):
            raise ValueError('the monomial exponents must be non-negative')

        if weights.shape != (total_terms, 2 * self.n_outputs):
            raise ValueError(f'weights must have shape ({total_terms}, {2 * self.n_outputs}), '
                             f'not {weights.shape}')
        if offsets.shape != (self.n_wl, 2 * self.n_outputs):
            raise ValueError(f'offsets must have shape ({self.n_wl}, {2 * self.n_outputs}), '
                             f'not {offsets.shape}')

        self.n_terms = n_terms
        self.powers = powers
        self.weights = weights
        self.offsets = offsets

    @property
    def degree(self):
        '''Degree of the polynomial in the input signals'''
        return int(self.xp.max(self.xp.sum(self.powers, axis=1)))

    def term_slice(self, wavelength_index: int):
        '''(start, stop) of the terms of model *wavelength_index* in the flat tables'''
        start = int(self.xp.sum(self.n_terms[:wavelength_index]))
        return start, start + int(self.n_terms[wavelength_index])

    def predict_phasors(self, signals, wavelength_index: int):
        '''
        Phasors ``[sin, cos]`` (length ``2 * n_outputs``) of the pistons from the
        *signals* of one wavelength (length ``n_inputs``, powers or any monotone
        scaling of them).
        '''
        x = self.standardise(signals, wavelength_index)
        start, stop = self.term_slice(wavelength_index)
        # monomials: x ** powers (0**0 = 1, so a constant term stays constant)
        monomials = self.xp.prod(x ** self.powers[start:stop].astype(self.dtype), axis=1)
        return monomials @ self.weights[start:stop] + self.offsets[wavelength_index]

    def decode_phasors(self, phasors, wavelength_index: int):
        '''
        Wrapped pistons [nm] (length ``n_outputs``, in ``[-period/2, period/2)``)
        and their confidences (the phasor length: ~1 when the model is sure),
        from the *phasors* of one wavelength.
        '''
        phasors = self.xp.asarray(phasors, dtype=self.dtype).ravel()
        if phasors.shape[0] != 2 * self.n_outputs:
            raise ValueError(f'expected {2 * self.n_outputs} phasors, got {phasors.shape[0]}')
        n = self.n_outputs
        sin, cos = phasors[:n], phasors[n:]
        period = self.periods_nm[wavelength_index]
        pistons = self.xp.arctan2(sin, cos) * (period / (2 * self.xp.pi))
        return pistons, self.xp.hypot(sin, cos)

    def predict_pistons(self, signals, wavelength_index: int):
        return self.decode_phasors(self.predict_phasors(signals, wavelength_index),
                                   wavelength_index)

    def get_value(self):
        return self.weights

    def set_value(self, v):
        assert v.shape == self.weights.shape, \
            f'Error: input array shape {v.shape} does not match {self.weights.shape}'
        self.weights[:] = self.to_xp(v)

    def get_fits_header(self):
        hdr = super().get_fits_header()
        # 'PLCharacterization'/'PLCubicModel' are the names earlier exporters wrote
        hdr['OBJ_TYPE'] = 'LanternCubicModel'
        hdr['TOTTERMS'] = self.powers.shape[0]
        return hdr

    def save(self, filename, overwrite=False):
        storage.write_object(filename, self.get_fits_header(),
                             {'wavelengths_nm': self.wavelengths_nm,
                              'periods_nm': self.periods_nm,
                              'n_terms': self.n_terms,
                              'powers': self.powers,
                              'weights': self.weights,
                              'offsets': self.offsets,
                              'input_mean': self.input_mean,
                              'input_std': self.input_std,
                              'sigma_nm': self.sigma_nm},
                             overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
            # 'PLCubicModel' is the name earlier versions of the exporter wrote
            if hdr.get('OBJ_TYPE') not in ('LanternCubicModel', 'PLCubicModel'):
                raise ValueError(f'Error: file {filename} does not contain a LanternCubicModel object')
            version = int(hdr.get('VERSION', 0))
            if version != 1:
                raise ValueError(f'Error: unknown version {version} in file {filename}')
            return LanternCubicModel(
                wavelengths_nm=f.load('wavelengths_nm', target_device_idx=-1),
                periods_nm=f.load('periods_nm', target_device_idx=-1),
                n_terms=f.load('n_terms', target_device_idx=-1),
                powers=f.load('powers', target_device_idx=-1),
                weights=f.load('weights', target_device_idx=-1),
                offsets=f.load('offsets', target_device_idx=-1),
                input_mean=f.load('input_mean', target_device_idx=-1),
                input_std=f.load('input_std', target_device_idx=-1),
                sigma_nm=f.load('sigma_nm', target_device_idx=-1),
                target_device_idx=target_device_idx,
                **{k: v for k, v in shared_headers(hdr).items()
                   if k in ('span_nm', 'conf_min', 'huber_c', 'n_irls', 'pl_name')})

    def array_for_display(self):
        '''Figure of merit for display: number of terms of each fitted model'''
        return self.n_terms
