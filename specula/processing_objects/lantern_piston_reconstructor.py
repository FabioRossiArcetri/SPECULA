from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.lantern_piston_model import LanternPistonModel


class LanternPistonReconstructor(BaseProcessingObj):
    """
    Per-wavelength photonic-lantern piston reconstructor.

    Applies a per-wavelength piston model to the lantern photometry: for every
    wavelength it turns the output channel powers into a *wrapped* piston
    estimate, ``[-period/2, period/2)``, and a confidence. A piston is only
    observable modulo the wavelength, so these estimates are deliberately wrapped:
    fuse them across wavelengths with a
    :class:`~specula.processing_objects.spectral_piston_unwrapper.SpectralPistonUnwrapper`.

    The strategy is chosen by the model itself, so this object is the same whatever
    the reconstructor is: the sparse cubic regression of
    :class:`~specula.data_objects.lantern_cubic_model.LanternCubicModel`, the
    analytical forward model inverted per wavelength of
    :class:`~specula.data_objects.lantern_closed_form_model.LanternClosedFormModel`,
    or any other :class:`~specula.data_objects.lantern_piston_model.LanternPistonModel`.
    Selecting a different strategy is a one-line change in the configuration (the
    class of the ``pl_model`` entry).

    The input signals are ``(n_rows, n_ports)``, one row per wavelength, as
    produced by :class:`~specula.processing_objects.dispersed_lantern_sensor.DispersedLanternSensor`.
    When the signals object carries a ``wavelengths_nm`` attribute (the sensor
    sets it), the rows are matched to the model's wavelengths by value, so the
    sensor may propagate more wavelengths than the model was fitted at; otherwise
    the rows are taken in order and their counts must agree.

    Parameters
    ----------
    piston_model : LanternPistonModel
        The per-wavelength reconstructor, with the input standardisation it was
        fitted with.
    target_device_idx : int [1], optional
        Target device for computation (CPU/GPU). Default is None
        (uses global setting).
    precision : int [1], optional
        Precision for computation (0 for double, 1 for single). Default is None
        (uses global setting).

    Outputs
    -------
    out_pistons_wrapped : BaseValue
        ``(n_outputs, n_wl)`` wrapped piston estimates [nm], in ``[-period/2, period/2)``.
    out_confidence : BaseValue
        ``(n_outputs, n_wl)`` confidence of each estimate.
    """

    def __init__(self,
                 piston_model: LanternPistonModel,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if piston_model is None:
            raise ValueError('LanternPistonReconstructor needs a piston_model')
        self.model = piston_model

        self.pistons = BaseValue('wrapped piston estimates',
                                 target_device_idx=target_device_idx, precision=precision)
        self.confidence = BaseValue('confidence of the wrapped piston estimates',
                                    target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_signals'] = InputValue(type=BaseValue)
        self.outputs['out_pistons_wrapped'] = self.pistons
        self.outputs['out_confidence'] = self.confidence

    @classmethod
    def input_names(cls):
        return {'in_signals': InputDesc(BaseValue,
                                       'Lantern channel powers, one row per wavelength')}

    @classmethod
    def output_names(cls):
        return {
            'out_pistons_wrapped': OutputDesc(BaseValue, 'Wrapped piston estimates, per wavelength'),
            'out_confidence': OutputDesc(BaseValue, 'Confidence of each wrapped piston estimate'),
        }

    def _rows_for(self, signals):
        '''
        Index of the signal row of every model wavelength: by value if the signals
        carry their wavelengths, by position otherwise.
        '''
        wavelengths = getattr(signals, 'wavelengths_nm', None)
        n_rows = signals.value.shape[0]
        if wavelengths is None:
            if n_rows != self.model.n_wl:
                raise ValueError(f'the input has {n_rows} rows while the model was fitted at '
                                 f'{self.model.n_wl} wavelengths, and it does not declare its '
                                 f'wavelengths: the rows cannot be matched')
            return list(range(n_rows))
        rows = []
        for w in range(self.model.n_wl):
            wl = float(self.model.wavelengths_nm[w])
            matches = [i for i, value in enumerate(wavelengths) if abs(value - wl) < 0.5]
            if not matches:
                raise ValueError(f'the model wavelength {wl:.1f} nm is not among the input ones '
                                 f'({len(wavelengths)} rows)')
            rows.append(matches[0])
        return rows

    def setup(self):
        super().setup()
        signals = self.local_inputs['in_signals']
        if signals is None:
            raise ValueError("'in_signals' must be given as an input")
        value = signals.value
        if value is None or len(value.shape) != 2:
            raise ValueError(f'in_signals must be a 2D (n_wl, n_ports) array, not '
                             f'{None if value is None else value.shape}')
        if value.shape[1] != self.model.n_inputs:
            raise ValueError(f'the signals have {value.shape[1]} channels while the model was '
                             f'fitted on {self.model.n_inputs}')
        self._rows = self._rows_for(signals)
        self.pistons.value = self.xp.zeros((self.model.n_outputs, self.model.n_wl), dtype=self.dtype)
        self.confidence.value = self.xp.zeros((self.model.n_outputs, self.model.n_wl), dtype=self.dtype)

    def trigger_code(self):
        signals = self.local_inputs['in_signals'].value
        model = self.model
        for w in range(model.n_wl):
            pistons, confidence = model.predict_pistons(signals[self._rows[w]], w)
            self.pistons.value[:, w] = pistons
            self.confidence.value[:, w] = confidence

    def post_trigger(self):
        super().post_trigger()
        self.pistons.generation_time = self.current_time
        self.confidence.generation_time = self.current_time
