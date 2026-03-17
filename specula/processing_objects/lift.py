import logging
import math
from collections import namedtuple

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray, np
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pixels import Pixels
from specula.lib.interp2d import Interp2D


WFS_Settings = namedtuple('WFS_Settings',
                         'sampling_ratio fft_sampling fft_padding fft_size actual_fov fft_res')
WFS_Settings.__doc__ = '''
WFS settings

Contains the input parameters used to calculate the wfs
internal array geometries.
'''

class Lift(BaseProcessingObj):
    """
    LIFT algorithm processing object.
    Implements the LIFT algorithm for phase estimation from a focal plane image,
    as described in Meimon et al. 2010.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 defocus_amp: float,
                 nPistons: int,
                 nZern: int, 
                 wavelengthInNm: float,
                 pix_scale: float,
                 npix_side: int,
                 cropped_size: int,
                 ifunc: IFunc=None,
                 n_iter: int=20,
                 fft_res: int=2,
                 quiet: bool=True,
                 fix: bool=False,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        # Store parameters
        self.nPistons = int(nPistons)
        self.nZern = int(nZern)
        self.nmodes = self.nZern + self.nPistons
        self.airef = self.xp.zeros(self.nmodes, dtype=self.dtype)
        self.airef[self.nPistons + 2] = self.dtype(defocus_amp)
        self.n_iter = int(n_iter)
        self.wavelengthInNm = wavelengthInNm
        self.pix_scale = pix_scale
        self.npix_side = npix_side
        self.cropped_size = cropped_size
        self.fft_res = fft_res
        self.quiet = quiet
        self.fix = bool(fix)

        # Derived parameters
        self.modes = None
        self.phase_ref = None
        self._img_norm = None
        self.ref_tip = 0.0
        self.ref_tilt = 0.0
        self.padded = None
        self.radians_per_pixel = None

        self.inputs['in_pixels'] = InputValue(type=Pixels)

        self.out_modes = BaseValue(
            value=self.xp.zeros(self.nmodes, dtype=self.dtype),
            target_device_idx=target_device_idx
        )
        self.out_modes.value = self.xp.zeros(self.nmodes, dtype=self.dtype)

        self.outputs['out_modes'] = self.out_modes

        # self.outputs['phase_estimate'] = None

        if self.verbose:
            logging.info(f"[{self.name}] LIFT initialized with {self.nmodes} modes")

        self.ifunc = ifunc
        mask = self.ifunc.mask_inf_func

        self.set_modalbase(self.ifunc.influence_function,
                           mask,
                           diameter=self.simul_params.pixel_pupil * self.simul_params.pixel_pitch)


    def ft_ft2(self, x):
        pad = (self.fftSize - self.gridSize) // 2
        if self.padded is None:
            self.padded = self.xp.zeros((self.fftSize, self.fftSize), dtype=x.dtype)
        self.padded[pad:pad+self.gridSize, pad:pad+self.gridSize] = x
        result = self.xp.fft.fftshift(self.xp.fft.fft2(self.padded)) * (1.0 / self.fftSize)
        return result[pad:pad+self.gridSize, pad:pad+self.gridSize]


    def computeCoG(self, frame, thFactor = 0.05):
        thValue = thFactor * self.xp.max(frame)
        thImage = self.xp.where( frame < thValue, 0., frame)
        return self.ndimage_center_of_mass(thImage)


    def computeReconstructor(self, H, Rdiag):
        Rinv = 1 / Rdiag
        htrinv = H.T * Rinv.T
        return Rinv, self.xp.linalg.inv(htrinv @ H) @ htrinv


    def setRefTT(self, center_x, center_y, image_size):
        image_center = 0.5 * image_size
        self.ref_tip = (center_y - image_center) * self.radians_per_pixel
        self.ref_tilt = (center_x - image_center) * self.radians_per_pixel


    def calcCenter(self, frame):
        if self.fix:
            return (0.5 * frame.shape[0], 0.5 * frame.shape[1])
        yc, xc = self.computeCoG(frame)
        return (xc, yc)


    def crop(self, frame, center, side=None):
        if side is None:
            side = self.cropped_size
        row_start = int(math.ceil(float(center[1]) - side))
        row_end = int(math.ceil(float(center[1]) + side))
        col_start = int(math.ceil(float(center[0]) - side))
        col_end = int(math.ceil(float(center[0]) + side))
        return frame[row_start:row_end, col_start:col_end]


    def calcCroppedFlux(self, frame, center):
        return self.crop(frame, center).sum()


    @staticmethod
    def calc_geometry(phase_sampling, pixel_pitch, wavelengthInNm,
                      pix_scale, npix_side, fft_res=2.0, quiet=False):
        """Calculate WFS geometry"""
        rad2arcsec = 206264.806247
        wanted_fov = pix_scale * npix_side
        D = phase_sampling * pixel_pitch
        lmbda = wavelengthInNm * 1e-9
        fov_internal = (lmbda / D) * (D / pixel_pitch) * rad2arcsec
        sampling_ratio = wanted_fov / fov_internal

        fft_sampling = round(phase_sampling * sampling_ratio)
        fft_size = round(fft_sampling * fft_res) // 2 * 2
        fft_res = fft_size / float(fft_sampling)
        fft_padding = fft_size - fft_sampling
        actual_fov = (lmbda / D) * (D / (D / fft_sampling)) * rad2arcsec

        return WFS_Settings(sampling_ratio, fft_sampling, fft_padding, fft_size, actual_fov, fft_res)


    def set_modalbase(self, modalbase, mask2d, diameter):
        """Preload modal base and resize to FFT grid"""
        settings = self.calc_geometry(
            mask2d.shape[0],
            diameter / mask2d.shape[0],
            self.wavelengthInNm,
            pix_scale=self.pix_scale,
            npix_side=self.npix_side,
            fft_res=self.fft_res,
            quiet=self.quiet
        )

        self.fftSize = settings.fft_sampling + settings.fft_padding
        self.gridSize = settings.fft_sampling
        self.radians_per_pixel = float(np.pi / (4.0 * settings.fft_res))

        mask2d = cpuArray(mask2d)
        modalbase = cpuArray(modalbase)
        cpu_dtype = np.float32 if self.precision == 1 else np.float64
        resize_interp = Interp2D(
            mask2d.shape,
            (self.gridSize, self.gridSize),
            dtype=cpu_dtype,
            xp=np
        )

        valid_idx = np.nonzero(mask2d)
        f = np.zeros_like(mask2d)

        self.modes = []
        for i in range(self.nmodes):
            f[valid_idx] = modalbase[i, :]
            f2 = resize_interp.interpolate(f.astype(cpu_dtype, copy=False))
            self.modes.append(self.xp.array(f2, dtype=self.dtype))
        self.modesCube = self.xp.stack(self.modes)

        mask2d = resize_interp.interpolate(mask2d.astype(cpu_dtype, copy=False))
        self.mask = self.xp.array(mask2d, dtype=self.dtype)
        self._img_norm = self.dtype(1.0) / self.xp.sqrt(self.mask.sum(dtype=self.dtype))
        self.phase_ref = self.phaseFromCoeffs(self.airef)

        if self.verbose:
            logging.info(f"[{self.name}] Modal base set, gridSize={self.gridSize}, fftSize={self.fftSize}")


    def phaseFromCoeffs(self, coeffs):
        """Phase reconstruction from modal coefficients"""
        coeffs = self.to_xp(coeffs, dtype=self.dtype)
        cv = coeffs[:, None, None]
        return (cv * self.modesCube).sum(axis=0)


    def phaseLIFT(self, p):
        return p + self.phase_ref + \
            self.modes[0 + self.nPistons] * self.ref_tip + \
            self.modes[1 + self.nPistons] * self.ref_tilt


    def abs2(self, x):
        return x.real**2 + x.imag**2


    def IK_prime(self, index, Pd, conjPdTilde, center):
        resultFull = 2.0 * (conjPdTilde *
                               self.ft_ft2(Pd * self.modes[index])).real
        return self.crop(resultFull, center)


    def complexField(self, phase):
        phase_lift = self.phaseLIFT(phase)
        complexField = self.mask * self.xp.exp(self.complex_dtype(1j) * phase_lift)
        complexFieldFFT = self.ft_ft2(complexField)
        return complexField, complexFieldFFT


    def focalPlaneImageFromFFT(self, complexFieldFFT, set_flux=None):
        if set_flux is not None:
            # If a certain total flux must be set,
            # there is no need for normalization since
            # it would be overridden anyway
            img = self.abs2(complexFieldFFT)
            img *= set_flux / img.sum()
        else:
            img = self.abs2(complexFieldFFT * self._img_norm)
        return img


    def calcDerivatives(self, complexField, complexFieldFFT, roi):
        # Precalculate some data for IK_prime
        conjPdTilde = self.xp.conj(complexFieldFFT) * self.complex_dtype(1j)
        IK_p_list = []
        for i in range(self.nmodes):
            IK_p_list.append(self.IK_prime(i, complexField, conjPdTilde, roi).ravel())
        H = self.xp.vstack(IK_p_list).transpose()
        return H


    def computeNoiseCovarianceDiag(self, image):
        '''
        Compute noise covariance matrix.
        Only return the diagonal to avoid allocating a big matrix.
        '''
        nCov = image.ravel()
        cMinTh = nCov.max() * 1e-6
        nCovDiag = self.xp.where(nCov < cMinTh, cMinTh, nCov)
        return nCovDiag


    def applyReconstructor(self, P_ML, DeltaI):
        return P_ML @ DeltaI.ravel()


    def getError(self, DeltaI, Rinv):
        return (DeltaI.ravel()**2 * Rinv).sum() / (self.gridSize**2)


    def setPsf(self, psf):
        psf = self.xp.array(psf)
        tMax = self.xp.max(psf)
        tmpFrame = self.xp.where(psf < 0.05*tMax, 0., psf)
        center = self.computeCoG(tmpFrame)
        frame = self.crop_or_enlarge_around_peak(psf, int(self.gridSize),
                                                 peak_index=(int(center[0]), int(center[1])))
        return self.xp.array(frame)


    def phaseEstimation(self, psf_orig, relTol=1e-3, absTol=1e-3):
        if not self.modes:
            raise Exception('Modal base has not been set yet')

        psf = self.setPsf(psf_orig)
        total_A_ML = self.xp.zeros(self.nmodes, dtype=self.dtype)
        currentPhaseEstimate = self.xp.zeros_like(self.mask)
        olderror = float(1e18)
        errors = []
        currentPhaseEstimates = []
        total_A_MLs = []
        errors.append(olderror)
        currentPhaseEstimates.append(currentPhaseEstimate)
        total_A_MLs.append(total_A_ML)
        # Estimate initial ROI and flux
        center = self.calcCenter(psf)
        flux = self.calcCroppedFlux(psf, center)
        # Set reference TT based on initial ROI
        self.setRefTT(float(center[0]), float(center[1]), float(psf.shape[0]))
        for i in range(self.n_iter):
            # Store current ROI from original PSF
            psfRoi = self.crop(psf, center)
            # Compute PSF from the input phase. Keep intermediate results
            complexField, complexFieldFFT = self.complexField(currentPhaseEstimate)
            I0 = self.focalPlaneImageFromFFT(complexFieldFFT, set_flux=flux)
            # Update ROI based on new image
            newCenter = self.calcCenter(I0)
            I0roi = self.crop(I0, newCenter)
            # Renormalize flux and EF based on new ROI
            flux *= self.calcCroppedFlux(psf, newCenter) / I0roi.sum()
            norm = self.xp.sqrt(flux) / self.xp.sqrt(self.abs2(complexFieldFFT).sum())
            complexField *= norm
            complexFieldFFT *= norm
            # Calculate derivatives and reconstructor
            H = self.calcDerivatives(complexField, complexFieldFFT, newCenter)
            Rdiag = self.computeNoiseCovarianceDiag(I0roi)
            Rinv, P_ML = self.computeReconstructor(H, Rdiag)
            # Use reconstructor to get the delta zernike estimate
            DeltaI = psfRoi - I0roi
            DeltaZ = self.applyReconstructor(P_ML, DeltaI)
            currentPhaseEstimate += self.phaseFromCoeffs(DeltaZ)
            # Accumulate delta modes
            DeltaZ_cpu = cpuArray(DeltaZ)
            total_A_ML += self.to_xp(DeltaZ_cpu, dtype=self.dtype)
            # Update PSF window
            center = newCenter
            # Compute convergence criteria
            newerror = float(cpuArray(self.getError(DeltaI, Rinv)))
            improvement = (olderror - newerror) / olderror
            # Store convergence history
            errors.append(newerror)
            currentPhaseEstimates.append(currentPhaseEstimate.copy())
            total_A_MLs.append(total_A_ML.copy())
            if logging.root.level <= logging.INFO:
                logging.debug(f"Iteration: {i}")
                logging.debug(f"estimated flux: {flux}")
                logging.debug(f"center_x, center_y: {center[0]}, {center[1]}")
                logging.debug(f"min R, max R: {Rdiag.min()}, {Rdiag.max()}")
                logging.debug(f"Current step of modal coefficients: {DeltaZ_cpu}")
                logging.debug(f"Current estimate of modal coefficients: {total_A_ML}")
                logging.debug(f"Criterion value: {newerror}")
            if abs(improvement) < relTol or np.mean(np.abs(DeltaZ_cpu)) < absTol:
                logging.debug(f'criterion reached at the {i+1} iteration')
                break

        lastAML = self.to_xp(total_A_MLs[-1], dtype=self.dtype, force_copy=True)
        lastAML[0 + self.nPistons] += self.ref_tip - 0.5 * self.radians_per_pixel
        lastAML[1 + self.nPistons] += self.ref_tilt - 0.5 * self.radians_per_pixel
        return currentPhaseEstimates[-1], lastAML * self.wavelengthInNm/(2*np.pi), len(total_A_MLs)


    def focalPlaneImageLIFT(self, phase, set_flux=None):
        '''
        Backward compatibility, called from example/test
        '''
        phase = self.xp.array(phase)
        complexField, complexFieldFFT = self.complexField(phase)
        return self.focalPlaneImageFromFFT(complexFieldFFT, set_flux=set_flux)


    def crop_or_enlarge_around_peak(self, in_array, desired_width, peak_index=None):
        '''
        Function to crop a PSF frame or enlarge it around the peak,
        depending on whether it is larger or smaller than the desired width.        

        peak_index: tuple with the (r,c) integer coordinates of the
                    image center. If None, it will be found with np.argmax
        '''
        # Find the peak's location
        if peak_index is None:
            peak_index = self.xp.unravel_index(self.xp.argmax(in_array, axis=None), in_array.shape)
        # Calculate half the desired width, rounded down
        half_width = desired_width // 2
        # Calculate the top-left corner of the crop box
        start_row = max(0, peak_index[0] - half_width)
        start_col = max(0, peak_index[1] - half_width)
        # Calculate the bottom-right corner of the crop box
        end_row = min(in_array.shape[0], peak_index[0] + half_width + 1)
        end_col = min(in_array.shape[1], peak_index[1] + half_width + 1)
        # Crop the in_array
        cropped = in_array[start_row:end_row, start_col:end_col]
        # Check if the cropped area is smaller than desired and needs to be enlarged
        if cropped.shape[0] < desired_width or cropped.shape[1] < desired_width:
            # Calculate padding to add to each side
            padding_row = (desired_width - cropped.shape[0]) // 2
            extra_row = (desired_width - cropped.shape[0]) % 2
            padding_col = (desired_width - cropped.shape[1]) // 2
            extra_col = (desired_width - cropped.shape[1]) % 2
            # Apply padding
            cropped = self.xp.pad(cropped, ((padding_row, padding_row + extra_row),
                                    (padding_col, padding_col + extra_col)),
                            'constant', constant_values=0)
        return cropped


    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.in_pixels = self.local_inputs['in_pixels']


    def trigger(self):
        if self.in_pixels is None:
            return

        psf = self.in_pixels.get_value()
        currentPhaseEstimate, coeffs, niters = self.phaseEstimation(psf) 

        self.outputs['out_modes'].value = self.to_xp(coeffs)
        self.outputs['out_modes'].generation_time = self.current_time

        # self.outputs["phase_estimate"] = currentPhaseEstimate
        if self.verbose:
            logging.info(f"[{self.name}] Trigger done, coeffs={coeffs[:5]}...")


    def finalize(self):
        if self.verbose:
            logging.info(f"[{self.name}] Finalized")
