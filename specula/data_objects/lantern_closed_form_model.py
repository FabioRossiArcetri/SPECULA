from itertools import combinations

from astropy.io import fits

from specula.data_objects.lantern_piston_model import LanternPistonModel, shared_headers
from iteras.lib import storage


class LanternClosedFormModel(LanternPistonModel):
    """
    Analytical forward model of the lantern's response to petal pistons, inverted
    wavelength by wavelength.

    With piston-only aberrations and monochromatic light the field at every output
    core is linear in the six petal phasors ``exp(i 2*pi*p_j/lambda)``, so each
    output power is a Hermitian quadratic form in them::

        N_c(p) = a_c + sum_{j<k} [A_cjk cos D_jk + B_cjk sin D_jk],
        D_jk = 2*pi*(p_j - p_k)/lambda,   petal ref at p = 0

    i.e. ``a + z(p) @ W.T`` with the 30 pair features ``z = [cos D, sin D]``, and,
    if the model was fitted with the coupling-efficiency normalisation of the
    camera signals, divided by ``1 + z(p) @ q``. The constants come from PLREC's
    ``forward.fit_forward`` at each wavelength (directly, or through its
    first-principles ``derivation`` module).

    The reconstructor is the *inversion* of that closed form, as in PLREC's
    ``forward.invert_fixed``: the pair features are recovered from the lifted
    linear system ``(W - s q^T) z = s - a`` and refined by alternating projections
    onto the rank-1 set of petal phasors, which gives two starting points (the
    leading and the second eigenvector of the 6x6 pair matrix); further starts are
    the twin image and a fixed Sobol grid of the +-period/2 cube; from each, a
    fixed number of Gauss-Newton steps with the analytic Jacobian; per sample the
    solution with the lowest residual is kept and wrapped into
    ``[-period/2, period/2)``.

    The pair phases use the *wavelength* (the physics); the *period* is only the
    wrap period of the returned piston.

    Parameters
    ----------
    wavelengths_nm, periods_nm : array_like [nm]
        As in :class:`~specula.data_objects.lantern_piston_model.LanternPistonModel`.
    cf_a : array_like
        Shape ``(n_wl, n_cores)``: the constant term of each core.
    cf_w : array_like
        Shape ``(n_wl, n_cores, 2 * n_pairs)``: the cosine and sine coefficients.
    cf_q : array_like
        Shape ``(n_wl, 2 * n_pairs)``: the denominator coefficients (zero for a
        model fitted on channel powers, not normalised).
    twin : array_like, optional
        Shape ``(n_pistons, n_pistons)``: the twin matrix of the pupil symmetry,
        used as one of the inversion starts.
    n_grid : int
        Number of fixed Sobol starting points (default 32).
    steps : int
        Gauss-Newton iterations from each start (default 5).
    n_proj : int
        Alternating projections of the spectral start (default 5).
    second : bool
        Also start from the second eigenvector (default True).
    reg : float
        Regularisation of the lifted least-squares problem (default 1e-6).
    input_mean, input_std, sigma_nm, span_nm, conf_min, huber_c, n_irls, pl_name
        As in :class:`~specula.data_objects.lantern_piston_model.LanternPistonModel`.

    Outputs
    -------
    predict_pistons(signals, wavelength_index) -> (pistons, confidence)
        Wrapped pistons [nm] and a confidence of 1 (the inversion always returns
        its best estimate; the unwrap weights the wavelengths with ``sigma_nm``).
    """

    def __init__(self,
                 wavelengths_nm=None,
                 periods_nm=None,
                 cf_a=None,
                 cf_w=None,
                 cf_q=None,
                 twin=None,
                 input_mean=None,
                 input_std=None,
                 sigma_nm=None,
                 span_nm: float = None,
                 conf_min: float = 0.15,
                 huber_c: float = 2.0,
                 n_irls: int = 12,
                 n_grid: int = 32,
                 steps: int = 5,
                 n_proj: int = 5,
                 second: bool = True,
                 reg: float = 1e-6,
                 pl_name: str = '',
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(wavelengths_nm=wavelengths_nm, periods_nm=periods_nm,
                         input_mean=input_mean, input_std=input_std, sigma_nm=sigma_nm,
                         span_nm=span_nm, conf_min=conf_min, huber_c=huber_c,
                         n_irls=n_irls, pl_name=pl_name,
                         target_device_idx=target_device_idx, precision=precision)

        n_cores, n_out = self.n_inputs, self.n_outputs
        n_petals = n_out + 1
        self.pairs = list(combinations(range(n_petals), 2))
        n_pairs = len(self.pairs)
        self.ref = n_petals - 1

        cf_a = self.xp.asarray(cf_a, dtype=self.dtype)
        cf_w = self.xp.asarray(cf_w, dtype=self.dtype)
        cf_q = self.xp.asarray(cf_q, dtype=self.dtype)
        if cf_a.shape != (self.n_wl, n_cores):
            raise ValueError(f'cf_a must have shape ({self.n_wl}, {n_cores}), not {cf_a.shape}')
        if cf_w.shape != (self.n_wl, n_cores, 2 * n_pairs):
            raise ValueError(f'cf_w must have shape ({self.n_wl}, {n_cores}, {2 * n_pairs}), '
                             f'not {cf_w.shape}')
        if cf_q.shape != (self.n_wl, 2 * n_pairs):
            raise ValueError(f'cf_q must have shape ({self.n_wl}, {2 * n_pairs}), not {cf_q.shape}')
        if twin is not None:
            twin = self.xp.asarray(twin, dtype=self.dtype)
            if twin.shape != (n_out, n_out):
                raise ValueError(f'twin must have shape ({n_out}, {n_out}), not {twin.shape}')
        if n_grid < 0 or steps < 0 or n_proj < 0:
            raise ValueError('n_grid, steps and n_proj must be >= 0')

        self.cf_a = cf_a
        self.cf_w = cf_w
        self.cf_q = cf_q
        self.twin = twin
        self.n_grid = int(n_grid)
        self.steps = int(steps)
        self.n_proj = int(n_proj)
        self.second = bool(second)
        self.reg = float(reg)
        self._j = self.xp.array([j for j, _ in self.pairs])
        self._k = self.xp.array([k for _, k in self.pairs])
        self._keep = [k for k in range(n_petals) if k != self.ref]
        self._grid = self._sobol_grid(n_petals) if self.n_grid else None

    # -- closed form --------------------------------------------------------
    def _sobol_grid(self, n_petals):
        '''Fixed quasi-random starting points in the +-1/2 cube (host, once)'''
        try:
            from scipy.stats import qmc
            unit = qmc.Sobol(n_petals - 1, scramble=False).random(self.n_grid)
        except ImportError:                      # deterministic fallback
            g = (self.xp.arange(1, n_petals) * 0.6180339887) % 1.0
            unit = (self.xp.arange(self.n_grid)[:, None] * g[None, :]) % 1.0
        return self.xp.asarray((unit - 0.5), dtype=self.dtype)

    def _full_pistons(self, pistons):
        '''(n, n_out) pistons -> (n, n_petals) with the reference petal at 0'''
        n = pistons.shape[0]
        full = self.xp.zeros((n, self.ref + 1), dtype=self.dtype)
        full[:, :self.ref] = self.xp.asarray(pistons, dtype=self.dtype)   # host input on a GPU model
        return full

    def _pair_phases(self, pistons, wavelength_index):
        full = self._full_pistons(pistons)
        return (2 * self.xp.pi * (full[:, self._j] - full[:, self._k]) /
                self.wavelengths_nm[wavelength_index])

    def _pair_features(self, pistons, wavelength_index):
        d = self._pair_phases(pistons, wavelength_index)
        return self.xp.concatenate([self.xp.cos(d), self.xp.sin(d)], axis=1)

    def _evaluate(self, pistons, wavelength_index):
        '''Channel powers of a batch of piston vectors: (n, n_cores)'''
        z = self._pair_features(pistons, wavelength_index)
        numerator = self.cf_a[wavelength_index][None, :] + z @ self.cf_w[wavelength_index].T
        denominator = 1 + z @ self.cf_q[wavelength_index]
        return numerator / denominator[:, None]

    def _jacobian(self, pistons, wavelength_index):
        '''(n, n_cores, n_out) analytic derivative of the powers w.r.t. the pistons'''
        xp = self.xp
        w = self.cf_w[wavelength_index]
        d = self._pair_phases(pistons, wavelength_index)
        cos_d, sin_d = xp.cos(d), xp.sin(d)
        z = xp.concatenate([cos_d, sin_d], axis=1)
        dz_dd = xp.concatenate([-sin_d, cos_d], axis=1)              # (n, 2 n_pairs)

        n_pairs = len(self.pairs)
        dd_dp = xp.zeros((n_pairs, self.n_outputs), dtype=self.dtype)     # D_jk = c (p_j - p_k)
        for i, (j, k) in enumerate(self.pairs):
            if j != self.ref:
                dd_dp[i, self._keep.index(j)] += 1
            if k != self.ref:
                dd_dp[i, self._keep.index(k)] -= 1
        dd_dp = dd_dp * (2 * xp.pi / self.wavelengths_nm[wavelength_index])
        dd_dp = xp.concatenate([dd_dp, dd_dp])
        dz_dp = dz_dd[:, :, None] * dd_dp[None]                       # (n, 2 n_pairs, n_out)

        num = self.cf_a[wavelength_index][None, :] + z @ w.T
        den = 1 + z @ self.cf_q[wavelength_index]
        dnum = xp.einsum('cf,nfp->ncp', w, dz_dp)
        dden = xp.einsum('f,nfp->np', self.cf_q[wavelength_index], dz_dp)
        return (dnum * den[:, None, None] - num[:, :, None] * dden[:, None, :]) / \
            den[:, None, None] ** 2

    def _gauss_newton(self, signals, starts, wavelength_index):
        '''Gauss-Newton from each start; returns (pistons, residual norm) per start'''
        xp = self.xp
        p = xp.array(starts, dtype=self.dtype)
        for _ in range(self.steps):
            residual = signals[None, :] - self._evaluate(p, wavelength_index)
            jac = self._jacobian(p, wavelength_index)                  # (n, n_cores, n_out)
            jt = xp.transpose(jac, (0, 2, 1))
            normal = jt @ jac + 1e-14 * xp.eye(self.n_outputs, dtype=self.dtype)[None]
            step = xp.linalg.solve(normal, (jt @ residual[:, :, None]))
            p = p + step[:, :, 0]
        residual = signals[None, :] - self._evaluate(p, wavelength_index)
        return p, xp.linalg.norm(residual, axis=1)

    def _spectral_start(self, signals, wavelength_index):
        '''Spectral initialisation: min-norm solution of the lifted system, then
        alternating projections onto the rank-1 set (Candes et al. 2015)'''
        xp = self.xp
        a = self.cf_a[wavelength_index]
        w = self.cf_w[wavelength_index]
        q = self.cf_q[wavelength_index]
        # (W - s q^T) z = s - a
        system = w[None, :, :] - signals[:, None] * q[None, None, :]
        rhs = signals - a
        gram = system @ xp.transpose(system, (0, 2, 1)) + self.reg * xp.eye(self.n_inputs, dtype=self.dtype)
        z = (xp.transpose(system, (0, 2, 1)) @
             xp.linalg.solve(gram, rhs[None, :, None]))[:, :, 0]
        vectors = None
        for _ in range(self.n_proj + 1):
            eigenvalues, eigenvectors = xp.linalg.eigh(self._pair_matrix(z))
            vectors = eigenvectors
            p1 = self._pistons_of(vectors[:, :, -1], wavelength_index)
            z_refined = self._pair_features(p1, wavelength_index)
            residual = (rhs[None, :] - (system @ z_refined[:, :, None])[:, :, 0])
            z = z_refined + (xp.transpose(system, (0, 2, 1)) @
                             xp.linalg.solve(gram, residual[:, :, None]))[:, :, 0]
        p1 = self._pistons_of(vectors[:, :, -1], wavelength_index)
        p2 = self._pistons_of(vectors[:, :, -2], wavelength_index)
        return p1[0], p2[0]

    def _pair_matrix(self, z):
        '''(n, 2 n_pairs) pair features -> (n, n_petals, n_petals) Hermitian matrix'''
        xp = self.xp
        n_petals = self.ref + 1
        m = xp.zeros((z.shape[0], n_petals, n_petals), dtype=self.complex_dtype)
        idx = xp.arange(n_petals)
        m[:, idx, idx] = 1
        e = z[:, :len(self.pairs)] + 1j * z[:, len(self.pairs):]
        m[:, self._j, self._k] = e
        m[:, self._k, self._j] = xp.conj(e)
        return m

    def _pistons_of(self, vectors, wavelength_index):
        '''(n, n_petals) phasors -> (n, n_out) pistons, wrapped at the period'''
        xp = self.xp
        phase = xp.angle(vectors) - xp.angle(vectors[:, [self.ref]])
        p = xp.delete(phase, self.ref, axis=1) * self.wavelengths_nm[wavelength_index] / (2 * xp.pi)
        period = self.periods_nm[wavelength_index]
        return p - period * xp.round(p / period)

    # -- interface ----------------------------------------------------------
    def predict_pistons(self, signals, wavelength_index: int):
        xp = self.xp
        s = self.standardise(signals, wavelength_index)
        period = self.periods_nm[wavelength_index]

        p1, p2 = self._spectral_start(s, wavelength_index)
        starts = [p1[None, :]]
        if self.second:
            starts.append(p2[None, :])
        if self.twin is not None:
            starts.append((p1 @ self.twin.T)[None, :])
        if self._grid is not None:
            starts.append(self._grid * period)
        candidates = xp.concatenate(starts, axis=0)

        pistons, residual = self._gauss_newton(s, candidates, wavelength_index)
        best = int(xp.argmin(residual))
        p = pistons[best]
        p = p - period * xp.round(p / period)
        return p, xp.ones(self.n_outputs, dtype=self.dtype)

    def get_fits_header(self):
        hdr = super().get_fits_header()
        hdr['OBJ_TYPE'] = 'LanternClosedFormModel'
        hdr['NGRID'] = self.n_grid
        hdr['STEPS'] = self.steps
        hdr['NPROJ'] = self.n_proj
        hdr['SECOND'] = int(self.second)
        hdr['REG'] = self.reg
        return hdr

    def save(self, filename, overwrite=False):
        storage.write_object(filename, self.get_fits_header(),
                             {'wavelengths_nm': self.wavelengths_nm,
                              'periods_nm': self.periods_nm,
                              'cf_a': self.cf_a,
                              'cf_w': self.cf_w,
                              'cf_q': self.cf_q,
                              'twin': self.twin,
                              'input_mean': self.input_mean,
                              'input_std': self.input_std,
                              'sigma_nm': self.sigma_nm},
                             overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
            if hdr.get('OBJ_TYPE') != 'LanternClosedFormModel':
                raise ValueError(f'Error: file {filename} does not contain a '
                                 f'LanternClosedFormModel object')
            version = int(hdr.get('VERSION', 0))
            if version != 1:
                raise ValueError(f'Error: unknown version {version} in file {filename}')
            twin = f.load('twin', target_device_idx=-1) if f.has('twin') else None
            return LanternClosedFormModel(
                wavelengths_nm=f.load('wavelengths_nm', target_device_idx=-1),
                periods_nm=f.load('periods_nm', target_device_idx=-1),
                cf_a=f.load('cf_a', target_device_idx=-1),
                cf_w=f.load('cf_w', target_device_idx=-1),
                cf_q=f.load('cf_q', target_device_idx=-1),
                twin=twin,
                input_mean=f.load('input_mean', target_device_idx=-1),
                input_std=f.load('input_std', target_device_idx=-1),
                sigma_nm=f.load('sigma_nm', target_device_idx=-1),
                n_grid=int(hdr.get('NGRID', 32)),
                steps=int(hdr.get('STEPS', 5)),
                n_proj=int(hdr.get('NPROJ', 5)),
                second=bool(int(hdr.get('SECOND', 1))),
                reg=float(hdr.get('REG', 1e-6)),
                target_device_idx=target_device_idx,
                **{k: v for k, v in shared_headers(hdr).items() if k in
                   ('span_nm', 'conf_min', 'huber_c', 'n_irls', 'pl_name')})
