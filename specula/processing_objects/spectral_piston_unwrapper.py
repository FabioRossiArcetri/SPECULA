from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.lantern_piston_model import LanternPistonModel


class SpectralPistonUnwrapper(BaseProcessingObj):
    """
    Absolute pistons from per-wavelength wrapped estimates.

    A piston is only observable modulo the wavelength, so a
    :class:`~specula.processing_objects.lantern_cubic_reconstructor.LanternCubicReconstructor`
    returns a wrapped estimate at each wavelength, in ``[-period/2, period/2)``,
    with a confidence. This object fuses them into the absolute piston, the
    multi-wavelength analogue of synthetic-wavelength interferometry.

    The estimate of each piston is the circular maximum-likelihood solution: every
    wavelength votes for the pistons that agree with its wrapped estimate,
    ``J(p) = -sum_w W_w cos(phi_w - 2*pi*p/period_w)``, with the weight
    ``W_w = conf_w * (period_w / (2*pi*sigma_w))**2`` built from the confidence and
    from the per-wavelength error ``sigma_w`` of the model, low-confidence
    wavelengths gated out. The cost is minimised on a coarse grid over the whole
    capture range, then on a fine grid around the coarse minimum, and finally
    polished by iteratively reweighted least squares (Huber) where each wavelength
    may only change the piston by an integer number of its own periods, so a
    single wavelength with the wrong fringe does not bias the result.

    Parameters
    ----------
    piston_model : LanternPistonModel
        The model the wrapped estimates come from: it carries the periods, the
        per-wavelength errors, the capture range and the unwrap settings they were
        fitted for.
    pspan : float [nm], optional
        Capture range: pistons are searched in ``[-pspan, pspan]``. Default is the
        value stored in the model (:attr:`LanternCubicModel.span_nm`), which is the one
        the model was fitted for; override it only for a smaller, better-conditioned
        range.
    conf_min : float [1], optional
        Estimates whose confidence is below this are ignored (default from the model).
    huber_c : float [1], optional
        Huber threshold of the robust refinement, in units of the per-sample
        residual scatter (default from the model).
    n_irls : int [1], optional
        Number of robust refinement iterations (default from the model).
    target_device_idx : int [1], optional
        Target device for computation (CPU/GPU). Default is None
        (uses global setting).
    precision : int [1], optional
        Precision for computation (0 for double, 1 for single). Default is None
        (uses global setting).

    Outputs
    -------
    out_pistons : BaseValue
        ``(n_outputs,)`` absolute piston estimates [nm].
    out_agreement : BaseValue
        ``(n_outputs,)`` agreement score in [0, 1]: the fraction of the used
        wavelengths consistent with the estimate within a quarter period. Low
        values mean the estimate is ambiguous or the model is out of its range.
    """

    def __init__(self,
                 piston_model: LanternPistonModel,
                 pspan: float = None,
                 conf_min: float = None,
                 huber_c: float = None,
                 n_irls: int = None,
                 sigma_nm: float = None,
                 sigma_floor_nm: float = None,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if piston_model is None:
            raise ValueError('SpectralPistonUnwrapper needs a piston_model')
        self.model = piston_model

        if pspan is None:
            pspan = piston_model.span_nm
        if pspan is None or pspan <= 0:
            raise ValueError('SpectralPistonUnwrapper needs a positive pspan (the capture range), '
                             'neither given nor stored in the model')
        self.pspan = float(pspan)
        self.conf_min = float(piston_model.conf_min if conf_min is None else conf_min)
        self.huber_c = float(piston_model.huber_c if huber_c is None else huber_c)
        self.n_irls = int(piston_model.n_irls if n_irls is None else n_irls)
        if sigma_nm is not None and sigma_nm <= 0:
            raise ValueError('SpectralPistonUnwrapper sigma_nm must be positive')
        if sigma_floor_nm is not None and sigma_floor_nm < 0:
            raise ValueError('SpectralPistonUnwrapper sigma_floor_nm must be >= 0')
        self.sigma_nm = None if sigma_nm is None else float(sigma_nm)
        self.sigma_floor_nm = 0.0 if sigma_floor_nm is None else float(sigma_floor_nm)
        self._sigma_used = None

        self.pistons = BaseValue('absolute piston estimates',
                                 target_device_idx=target_device_idx, precision=precision)
        self.agreement = BaseValue('agreement of the wavelengths with the piston estimate',
                                   target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_pistons_wrapped'] = InputValue(type=BaseValue)
        self.inputs['in_confidence'] = InputValue(type=BaseValue)
        self.outputs['out_pistons'] = self.pistons
        self.outputs['out_agreement'] = self.agreement

    @classmethod
    def input_names(cls):
        return {
            'in_pistons_wrapped': InputDesc(BaseValue,
                                            'Wrapped piston estimates, one column per wavelength'),
            'in_confidence': InputDesc(BaseValue,
                                       'Confidence of each wrapped piston estimate'),
        }

    @classmethod
    def output_names(cls):
        return {
            'out_pistons': OutputDesc(BaseValue, 'Absolute piston estimates'),
            'out_agreement': OutputDesc(BaseValue, 'Agreement score of each piston estimate'),
        }

    def setup(self):
        super().setup()
        wrapped = self.local_inputs['in_pistons_wrapped']
        confidence = self.local_inputs['in_confidence']
        if wrapped is None or confidence is None:
            raise ValueError("both 'in_pistons_wrapped' and 'in_confidence' must be given")
        expected = (self.model.n_outputs, self.model.n_wl)
        for name, value in (('in_pistons_wrapped', wrapped), ('in_confidence', confidence)):
            if value.value is None or tuple(value.value.shape) != expected:
                raise ValueError(f'{name} must have shape {expected}, not '
                                 f'{None if value.value is None else value.value.shape}')
        self.pistons.value = self.xp.zeros(self.model.n_outputs, dtype=self.dtype)
        if self.sigma_nm is not None:
            self._sigma_used = self.xp.full((self.model.n_outputs, self.model.n_wl),
                                            self.sigma_nm, dtype=self.dtype)
        self.agreement.value = self.xp.zeros(self.model.n_outputs, dtype=self.dtype)

        lam = self.model.periods_nm
        self._lam = lam
        self._wn = 2 * self.xp.pi / lam
        mp = float(self.xp.mean(lam))
        self._mp = mp

        # coarse grid over the whole capture range
        step = max(float(self.xp.min(lam)) / 4, 2 * self.pspan / 6000)
        self._grid = self.xp.arange(-self.pspan, self.pspan + step, step, dtype=self.dtype)
        # fine grid, +-2.5 mean periods around the coarse minimum
        window = 2.5 * mp
        fine_step = max(float(self.xp.min(lam)) / 30, mp / 400)
        self._fine = self.xp.arange(-window, window + fine_step, fine_step, dtype=self.dtype)

    def _circular_cost(self, phi, base, candidates):
        '''
        ``J = -sum_w base_w cos(phi_w - 2*pi*candidate/period_w)`` for every piston
        and every candidate: ``(n_outputs, n_candidates)``.
        '''
        cosines = self.xp.cos(phi[:, :, None] - self._wn[None, :, None] * candidates[None, None, :])
        return -self.xp.sum(base[:, :, None] * cosines, axis=1)

    def trigger_code(self):
        xp = self.xp
        ghat = self.local_inputs['in_pistons_wrapped'].value
        conf = self.local_inputs['in_confidence'].value
        lam = self._lam
        model = self.model

        phi = 2 * xp.pi * ghat / lam[None, :]
        # per-wavelength weight: confidence times the inverse variance of the
        # wrapped estimate. The error is the model's fitted one, raised to
        # sigma_floor_nm when the actual wavefront is worse than the one the model
        # was fitted on (the fit residual would otherwise make every wavelength
        # equally, and absurdly, trusted); sigma_nm overrides it entirely.
        if self._sigma_used is None:
            sigma = xp.maximum(model.sigma_nm.T, max(self.sigma_floor_nm, 1e-6))
        else:
            sigma = self._sigma_used
        kappa = (lam[None, :] / (2 * xp.pi * sigma)) ** 2
        # a phasor length above 1 means the model extrapolated, not that it is sure
        weight = xp.minimum(conf, 1.0)
        base = xp.where(conf >= self.conf_min, weight, 0.0) * kappa
        total = xp.sum(base, axis=1, keepdims=True)
        base = xp.where(total > 0, base, kappa)      # all gated out: use all of them

        # coarse grid over the whole capture range
        cost = self._circular_cost(phi, base, self._grid)
        p = self._grid[xp.argmin(cost, axis=1)]

        # fine grid around the coarse minimum
        candidates = xp.clip(p[:, None] + self._fine[None, :], -self.pspan, self.pspan)
        cosines = xp.cos(phi[:, :, None] - self._wn[None, :, None] * candidates[:, None, :])
        cost = -xp.sum(base[:, :, None] * cosines, axis=1)
        p = candidates[xp.arange(p.shape[0]), xp.argmin(cost, axis=1)]

        # robust refinement, one integer number of periods per wavelength
        for _ in range(self.n_irls):
            shifted = ghat + xp.round((p[:, None] - ghat) / lam[None, :]) * lam[None, :]
            residual = shifted - p[:, None]
            scatter = 1.4826 * xp.median(xp.abs(residual), axis=1, keepdims=True) + 1e-9
            weight = base * xp.minimum(1.0, self.huber_c * scatter /
                                       xp.maximum(xp.abs(residual), 1e-9))
            denominator = xp.sum(weight, axis=1)
            p_new = xp.where(denominator > 0,
                             xp.sum(weight * shifted, axis=1) / xp.maximum(denominator, 1e-12), p)
            p = p_new

        # agreement: fraction of the used wavelengths consistent within a quarter period
        shifted = ghat + xp.round((p[:, None] - ghat) / lam[None, :]) * lam[None, :]
        used = base > 0
        consistent = (xp.abs(shifted - p[:, None]) < lam[None, :] / 4) & used
        self.agreement.value = xp.sum(consistent, axis=1) / xp.maximum(xp.sum(used, axis=1), 1)
        self.pistons.value = xp.clip(p, -1.5 * self.pspan, 1.5 * self.pspan)

    def post_trigger(self):
        super().post_trigger()
        self.pistons.generation_time = self.current_time
        self.agreement.generation_time = self.current_time
