from specula.processing_objects.base_modalrec import BaseModalrec
from specula.data_objects.recmat import Recmat
from specula.data_objects.intmat import Intmat
from specula.data_objects.nonlinear_calibration import NonlinearCalibration


class PyramidNonlinearRec(BaseModalrec):
    """
    Nonlinear modal reconstructor for a wavefront sensor (typically a
    non-modulated pyramid), generalizing the plain linear
    :class:`~specula.processing_objects.modalrec.Modalrec` with a
    per-mode saturation correction ("parallel channel") plus a linear
    correction on the residual signal it does not explain ("orthogonal
    channel").

    Reconstruction, each step, is:

    1. ``a_lin = recmat @ slopes``: the ordinary linear estimate (this is
       exactly what :class:`Modalrec` returns).
    2. ``a_parallel = calibration.invert(a_lin)``: de-saturate each mode's
       linear estimate using its calibrated response curve (see
       :class:`~specula.data_objects.nonlinear_calibration.NonlinearCalibration`).
    3. ``predicted_response = calibration.forward(a_parallel)``: re-apply the
       *same* calibrated (possibly saturating) curve forward, to predict how
       much of the signal `a_parallel` actually accounts for. Using the
       calibrated curve here (instead of the linear interaction matrix
       directly) is what keeps step 4 from undoing step 2: a mode fully
       explained by its own single-mode saturation must leave zero residual.
    4. ``predicted_slopes = intmat @ predicted_response`` and
       ``a_orthogonal = recmat @ (slopes - predicted_slopes)``: a single
       linear correction step on whatever signal is left over once each
       mode's own calibrated nonlinearity has been accounted for -- this is
       the "orthogonal shape" information a per-mode saturation curve cannot
       capture (e.g. genuine cross-mode coupling).
    5. ``out_modes = a_parallel + a_orthogonal``.

    With an identity (linear) calibration curve, `a_parallel == a_lin ==
    predicted_response` and `a_orthogonal` is the correction from one
    Gauss-Newton-style relinearization step; if `intmat` and `recmat` are
    exact pseudo-inverses of each other, that correction is zero and this
    reduces exactly to `Modalrec`.

    Note
    ----
    This decomposition assumes each mode's nonlinearity depends only on its
    own amplitude (no cross-mode coupling), and that a single relinearization
    step is sufficient to capture the "orthogonal shape" information. Both
    are deliberate first-version simplifications, meant to be replaced by a
    more rigorous, jointly-calibrated model once available.
    """

    def __init__(self,
                 recmat: Recmat,
                 intmat: Intmat,
                 nonlinear_calib: NonlinearCalibration,
                 nmodes: int = None,
                 apply_orthogonal_correction: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.recmat = recmat
        self.intmat = intmat
        self.nonlinear_calib = nonlinear_calib
        self.apply_orthogonal_correction = apply_orthogonal_correction

        rec_nmodes = self.recmat.recmat.shape[0]
        self.nmodes = nmodes if nmodes is not None else rec_nmodes
        if self.nmodes > rec_nmodes:
            raise ValueError(f'Requested nmodes={self.nmodes} exceeds recmat nmodes={rec_nmodes}')
        if self.nonlinear_calib.nmodes < self.nmodes:
            raise ValueError(
                f'nonlinear_calib covers {self.nonlinear_calib.nmodes} modes, '
                f'fewer than the requested nmodes={self.nmodes}'
            )

        self._mode_indices = self.xp.arange(self.nmodes)
        self.modes.value = self.xp.zeros(self.nmodes, dtype=self.dtype)

    def trigger_code(self):
        a_lin = (self.recmat.recmat[:self.nmodes, :] @ self.slopes)

        a_parallel = self.nonlinear_calib.invert(self._mode_indices, a_lin)

        if self.apply_orthogonal_correction:
            predicted_response = self.nonlinear_calib.forward(self._mode_indices, a_parallel)
            predicted_slopes = self.intmat.intmat[:, :self.nmodes] @ predicted_response
            residual_slopes = self.slopes - predicted_slopes
            a_orthogonal = self.recmat.recmat[:self.nmodes, :] @ residual_slopes
            out_modes = a_parallel + a_orthogonal
        else:
            out_modes = a_parallel

        self.modes.value[:] = out_modes
        self.modes.generation_time = self.current_time
