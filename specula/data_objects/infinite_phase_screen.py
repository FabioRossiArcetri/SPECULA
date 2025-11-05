from seeing.integrator import evaluateFormula, cpulib
from symao.turbolence import createTurbolenceFormulary, ft_phase_screen0, ft_ift2

from scipy import fft

turbolenceFormulas = createTurbolenceFormulary()

from specula.base_data_obj import BaseDataObj
from specula import ASEC2RAD, RAD2ASEC, cpuArray, np

def seeing_to_r0(seeing, wvl=500.e-9):
    return 0.9759*wvl/(seeing* ASEC2RAD)

def cn2_to_r0(cn2, wvl=500.e-9):
    r0=(0.423*(2*np.pi/wvl)**2*cn2)**(-3./5.)
    return r0

def r0_to_seeing(r0, wvl=500.e-9):
    return (0.9759*wvl/r0)*RAD2ASEC

def cn2_to_seeing(cn2, wvl=500.e-9):
    r0 = cn2_to_r0(cn2,wvl)
    seeing = r0_to_seeing(r0,wvl)
    return seeing

def ft_phase_screen_vect(f1d, PSD_phi_1D, N, delta, seed=321):
    np.random.seed(seed)
    frq_range = 1.0 / delta
    del_f = frq_range / N    
    fx = np.arange(-N / 2., N / 2.) * del_f
    (fx, fy) = np.meshgrid(fx, fx)
    f = np.sqrt(fx**2 + fy**2)    
    PSD_phi_2D = np.interp(f, f1d, PSD_phi_1D)
    M1 = np.random.normal(size=( int(N), int(N)))
    M2 = np.random.normal(size=( int(N), int(N)))
    cn = ( M1 + 1j * M2 ) *  np.sqrt(PSD_phi_2D) *  del_f
    phs = ft_ift2(cn, 1).real
    return phs, PSD_phi_2D, del_f

def compute_covariance_from_PSD_vect( f_vect, psd_vect, P=5, Q=5, points=10000):    
    fd = np.logspace(-P, Q, points)  # Input evaluation points    
    fln = np.log(fd[1]/fd[0])        # Step size    
    PSD_f = np.interp(fd, f_vect, psd_vect) * fd    
    offset_f = fft.fhtoffset(fln, initial= -(P-Q)*np.log(10), mu=0, bias=0)
    rd = np.exp(offset_f)/fd[::-1]*(10**(P-Q))   # Output evaluation points    
    fht = 2*np.pi * fft.fht(PSD_f, fln, mu=0, offset=offset_f, bias = 0)/rd
    fht -= np.min(fht) - 1e-6
    return fht, rd


class InfinitePhaseScreen(BaseDataObj):

    def __init__(self, mx_size, pixel_scale, r0, L0, random_seed=None,                                                        
                 psd1d_freq_data = None,
                 psd1d_data = None,
                 stencil_size_factor=1, xp=None,
                 target_device_idx=None, precision=0):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.psd1d_freq_data = psd1d_freq_data
        self.psd1d_data = psd1d_data
        if not self.psd1d_data is None:
            self.cov_1D_data, self.cov_1D_rd = compute_covariance_from_PSD_vect( self.psd1d_freq_data,self.psd1d_data, P=4, Q=4, points=10000)
            
        self.random_data_col = None
        self.random_data_row = None
        self.requested_mx_size = int(mx_size)
        self.mx_size = 2 ** (int( np.ceil(np.log2(mx_size)))) + 1
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        if xp is not None:
            self.xp = xp
        self.stencil_size_factor = stencil_size_factor

        # stencil size must be odd and >= 257
        base_stencil_size = int(stencil_size_factor * self.mx_size/2)*2 + 1
        min_stencil_size = 257
        self.stencil_size = max(base_stencil_size, min_stencil_size)

        self.stencil = None
        self.stencil_coords = None
        self.stencil_positions = None
        self.n_stencils = 0
        self.cov_mat = None
        self.cov_mat_zz = None
        self.cov_mat_xx = None
        self.cov_mat_zx = None
        self.cov_mat_xz = None
        self.full_scrn = None
        self.A_mat = None
        self.B_mat = None
        self.lastmax = 1
        self.first = True
        
        if random_seed is None:
            raise ValueError("random_seed must be provided")
        else:
            self.random_seed = int(random_seed)
        self.rng = self.xp.random.default_rng(self.random_seed)

        #self.set_stencil_coords_basic()
        self.set_stencil_coords()
        self.setup()

    def phase_covariance(self, r, r0, L0):
        
        r = cpuArray(r)
        # Get rid of any zeros
        r += 1e-40
        
        if not self.psd1d_data is None:
            self.cov_1D_data, self.cov_1D_rd
            cov = np.interp(r, self.cov_1D_rd, self.cov_1D_data)
            print('np.max(cov)', np.min(cov), np.max(cov))
        else:
            r0 = float(r0)
            L0 = float(L0)
            exprCf = turbolenceFormulas['phaseVarianceVonKarman0'].rhs
            (_, cov) = evaluateFormula( exprCf, {'r_0': r0, 'L_0': L0}, ['r'] , [r], cpulib)

    #        A = (L0 / r0) ** (5. / 3)
    #        B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (self.xp.pi ** (8. / 3))
    #        B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)
    #        C = (((2 * self.xp.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * self.xp.pi * r) / L0)
    #        cov = A * B1 * B2 * C / 2

        cov = self.to_xp(cov)

        return cov

    def set_stencil_coords_basic(self):
        self.stencil = self.xp.zeros((self.stencil_size, self.stencil_size))
        self.stencil[:2,:] = 1
        self.stencil_coords = self.to_xp(self.xp.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale
        self.n_stencils = self.stencil_coords.shape[0]

    def set_stencil_coords(self):
        self.stencil = np.zeros((self.stencil_size, self.stencil_size))
        self.stencilF = np.zeros((self.stencil_size, self.stencil_size))
        max_n = int( np.floor(np.log2(self.stencil_size)))
        # the head of stencil (basiaccaly all of it for us)
        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1
            coords = np.round(np.linspace(0, self.stencil_size - 1, n_points)).astype('int32')
            self.stencil[col - 1][coords] = 1
            self.stencilF[self.stencil_size - col][coords] = 1
        # the tail of stencil
        for n in range(1, self.stencil_size_factor + 1):
            col = n * self.mx_size - 1
            self.stencil[col, self.stencil_size // 2] = 1
            self.stencilF[self.stencil_size-col-1, self.stencil_size // 2] = 1
        self.stencil = self.to_xp(self.stencil)
        self.stencilF = self.to_xp(self.stencilF)
        self.stencil_coords = []
        self.stencil_coords.append(self.to_xp(self.xp.where(self.stencil == 1)).T)
        self.stencil_coords.append(self.to_xp(self.xp.where(self.stencilF == 1)).T)
        self.stencil_positions = []
        self.stencil_positions.append(self.stencil_coords[0] * self.pixel_scale)
        self.stencil_positions.append(self.stencil_coords[1] * self.pixel_scale)
        self.n_stencils = self.stencil_coords[0].shape[0]

    def AB_from_positions(self, positions):
        seperations = self.xp.zeros((len(positions), len(positions)))
        px, py = positions[:,0], positions[:,1]
        delta_x_grid_a, delta_x_grid_b = self.xp.meshgrid(px, px)
        delta_y_grid_a, delta_y_grid_b = self.xp.meshgrid(py, py)
        delta_x_grid = delta_x_grid_a - delta_x_grid_b
        delta_y_grid = delta_y_grid_a - delta_y_grid_b
        seperations = self.xp.sqrt(delta_x_grid ** 2 + delta_y_grid ** 2)
        self.cov_mat = self.phase_covariance(seperations, self.r0, self.L0)
        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]
        # Cholesky solve can fail - so do brute force inversion
        cf = self._lu_factor(self.cov_mat_zz)
        inv_cov_zz = self._lu_solve(cf, self.xp.identity(self.cov_mat_zz.shape[0]))
        A_mat = self.cov_mat_xz.dot(inv_cov_zz)
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - A_mat.dot(self.cov_mat_zx)
        # Then do SVD to get B matrix
        u, W, ut = self.xp.linalg.svd(BBt)
        L_mat = self.xp.zeros((self.stencil_size, self.stencil_size))
        self.xp.fill_diagonal(L_mat, self.xp.sqrt(W))
        # Now use sqrt(eigenvalues) to get B matrix
        B_mat = u.dot(L_mat)
        return A_mat, B_mat

    def setup(self):
        # set X coords
        self.new_col_coords1 = self.xp.zeros((self.stencil_size, 2))
        self.new_col_coords1[:, 0] = -1
        self.new_col_coords1[:, 1] = self.xp.arange(self.stencil_size)
        self.new_col_positions1 = self.new_col_coords1 * self.pixel_scale
        # calc separations
        positions1 = self.xp.concatenate((self.stencil_positions[0], self.new_col_positions1), axis=0)
        self.A_mat, self.B_mat = [], []
        A_mat, B_mat = self.AB_from_positions(positions1)
        self.A_mat.append(A_mat)
        self.B_mat.append(B_mat)
        self.A_mat.append(self.xp.fliplr(self.xp.flipud(A_mat)))
        self.B_mat.append(B_mat)
        # make initial screen
        if not self.psd1d_data is None:
            tmp, _, _ = ft_phase_screen_vect( self.psd1d_freq_data, self.psd1d_data, self.stencil_size, self.pixel_scale, seed=self.random_seed)
            self.full_scrn = self.to_xp(tmp)
        else:
            # print('Using formula covariance')
            tmp, _, _ = ft_phase_screen0( turbolenceFormulas, self.r0, self.stencil_size, self.pixel_scale, self.L0, seed=self.random_seed)
            self.full_scrn = self.to_xp(tmp)
            self.full_scrn *= (2 * np.pi) ** (11/6) # this is to compensate SYMAO bug that uses PSD(k) instead of PSD(f)

        self.full_scrn -= self.xp.mean(self.full_scrn[:self.requested_mx_size, :self.requested_mx_size])
        # print(self.full_scrn.shape)

    def prepare_random_data_col(self):
        if self.random_data_col is None:
#            print('generating new random data col')
            self.random_data_col = self.rng.standard_normal(size=self.stencil_size)
        else:
            pass
#            print('using old random data col')

    def prepare_random_data_row(self):
        if self.random_data_row is None:
#            print('generating new random data row')
            self.random_data_row = self.rng.standard_normal(size=self.stencil_size)
        else:
            pass
#            print('using old random data row')

    def get_new_line(self, row, after):
        if row:
            self.prepare_random_data_row()
            stencil_data = self.to_xp(self.full_scrn[self.stencil_coords[after][:, 1], self.stencil_coords[after][:, 0]])
            new_line = self.A_mat[after].dot(stencil_data) + self.B_mat[after].dot(self.random_data_row)
        else:
            self.prepare_random_data_col()
            stencil_data = self.to_xp(self.full_scrn[self.stencil_coords[after][:, 0], self.stencil_coords[after][:, 1]])
            new_line = self.A_mat[after].dot(stencil_data) + self.B_mat[after].dot(self.random_data_col)
        return new_line

    def add_line(self, row, after, flush=True):
        new_line = self.get_new_line(row, after)        
        row_rms = self.xp.max(self.xp.abs(new_line))
        new_line -= self.xp.mean(new_line)
        new_line /= row_rms

        if self.first:
            self.lastmax = row_rms
            self.first = False

        #while row_rms > 5*self.lastmax:
        #    self.random_data_row = None
        #    new_line = self.get_new_line(row, after)
        #    #new_line /= 1.5 * row_rms / self.lastmax            
        #    row_rms = self.xp.max(self.xp.abs(new_line))
        #    print(np.max(np.abs(new_line)))

        if row_rms > 2*self.lastmax:
            print("There was a big rms in InfinitePhase screen new line. There was a big rms in InfinitePhase screen new line. There was a big rms in InfinitePhase screen new line")
            print(np.max(np.abs(new_line)))

        self.lastmax = self.lastmax * (0.95) + row_rms * (0.05)

        if row:
            new_line = new_line[:,self.xp.newaxis]
            if after:
                self.full_scrn = self.xp.concatenate((self.full_scrn, new_line), axis=row)[:self.stencil_size, 1:]
            #    self.ndimage_shift(self.full_scrn, [-1, 0], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[-1, :] = new_line
            else:
                self.full_scrn = self.xp.concatenate((new_line, self.full_scrn), axis=row)[:self.stencil_size, :self.stencil_size]
            #    self.ndimage_shift(self.full_scrn, [1, 0], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[0, :] = new_line
        else:
            new_line = new_line[self.xp.newaxis, :]
            if after:
                self.full_scrn = self.xp.concatenate((self.full_scrn, new_line), axis=row)[1:, :self.stencil_size]
            #    self.ndimage_shift(self.full_scrn, [0, -1], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[:, -1] = new_line
            else:
                self.full_scrn = self.xp.concatenate((new_line, self.full_scrn), axis=row)[:self.stencil_size, :self.stencil_size]
            #    self.ndimage_shift(self.full_scrn, [0, 1], self.full_scrn, order=0, mode='constant', cval=0.0, prefilter=False)
            #    self.full_scrn[:, 0] = new_line
        if flush:
            self.random_data_col = None
            self.random_data_row = None

    @property
    def scrn(self):
        return cpuArray(self.full_scrn[:self.requested_mx_size, :self.requested_mx_size])

    @property
    def scrnRaw(self):
        return self.full_scrn[:self.requested_mx_size, :self.requested_mx_size]

    @property
    def scrnRawAll(self):
        return self.full_scrn
