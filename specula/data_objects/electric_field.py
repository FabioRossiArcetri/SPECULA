from astropy.io import fits
import numpy as np

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class ElectricField(BaseDataObj):
    '''Electric field'''

    def __init__(self,
                 dimx: int,
                 dimy: int,
                 pixel_pitch: float,
                 S0: float=0.0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(precision=precision, target_device_idx=target_device_idx)
        dimx = int(dimx)
        dimy = int(dimy)
        self.pixel_pitch = pixel_pitch
        self.S0 = S0
        self.A = self.xp.ones((dimx, dimy), dtype=self.dtype)
        self.phaseInNm = self.xp.zeros((dimx, dimy), dtype=self.dtype)

    def __str__(self):
        return str(self.A)+str(self.phaseInNm)

    def set_value(self, v):
        '''
        Set new values for phase and amplitude
        
        Arrays are not reallocated
        '''
        self.A[:]= self.to_xp(v[0], dtype=self.dtype)
        self.phaseInNm[:] = self.to_xp(v[1], dtype=self.dtype)

    def get_value(self):
        return self.xp.stack((self.A, self.phaseInNm))

    def reset(self):
        '''
        Reset to zero phase and unitary amplitude
        
        Arrays are not reallocated
        '''
        self.A *= 0
        self.A += 1
        self.phaseInNm *= 0

    @property
    def size(self):
        return self.A.shape

    def checkOther(self, ef2, subrect=None):
        if not isinstance(ef2, ElectricField):
            raise ValueError(f'{ef2} is not an ElectricField instance')
        if subrect is None:
            subrect = [0, 0]
        diff0 = self.size[0] - subrect[0]
        diff1 = self.size[1] - subrect[1]
        if ef2.size[0] != diff0 or ef2.size[1] != diff1:
            raise ValueError(f'{ef2} has size {ef2.size} instead of the required ({diff0}, {diff1})')
        return subrect

    def phi_at_lambda(self, wavelengthInNm, slicey=None, slicex=None):
        if slicey is None:
            slicey = np.s_[:]
        if slicex is None:
            slicex = np.s_[:]
        return self.phaseInNm[slicey, slicex] * ((2 * self.xp.pi) / wavelengthInNm)

    def ef_at_lambda(self, wavelengthInNm, slicey=None, slicex=None, out=None):
        if slicey is None:
            slicey = np.s_[:]
        if slicex is None:
            slicex = np.s_[:]
        phi = self.phi_at_lambda(wavelengthInNm, slicey=slicey, slicex=slicex)
        ef = self.xp.exp(1j * phi, dtype=self.complex_dtype, out=out)
        ef *= self.A[slicey, slicex]
        return ef

    def product(self, ef2, subrect=None):
#        subrect = self.checkOther(ef2, subrect=subrect)    # TODO check subrect from atmo_propagation, even in PASSATA it does not seem right
        x2 = subrect[0] + self.size[0]
        y2 = subrect[1] + self.size[1]
        self.A *= ef2.A[subrect[0] : x2, subrect[1] : y2]
        self.phaseInNm += ef2.phaseInNm[subrect[0] : x2, subrect[1] : y2]

    def area(self):
        return self.A.size * (self.pixel_pitch ** 2)

    def masked_area(self):
        tot = self.xp.sum(self.A)
        return (self.pixel_pitch ** 2) * tot

    def square_modulus(self, wavelengthInNm):
        ef = self.ef_at_lambda(wavelengthInNm)
        return self.xp.real( ef * self.xp.conj(ef) )

    def sub_ef(self, xfrom=None, xto=None, yfrom=None, yto=None, idx=None):
        if idx is not None:
            idx = self.xp.unravel_index(idx, self.A.shape)
            xfrom, xto = self.xp.min(idx[0]), self.xp.max(idx[0] +1)
            yfrom, yto = self.xp.min(idx[1]), self.xp.max(idx[1] +1)
        sub_ef = ElectricField(xto - xfrom + 1, yto - yfrom + 1, self.pixel_pitch)
        sub_ef.A = self.A[xfrom:xto, yfrom:yto]
        sub_ef.phaseInNm = self.phaseInNm[xfrom:xto, yfrom:yto]
        sub_ef.S0 = self.S0
        return sub_ef

    def compare(self, ef2):
        return not (self.xp.array_equal(self.A, ef2._A) and self.xp.array_equal(self.phaseInNm, ef2._phaseInNm))

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'ElectricField'
        hdr['DIMX'] = self.A.shape[0]
        hdr['DIMY'] = self.A.shape[1]
        hdr['PIXPITCH'] = self.pixel_pitch
        hdr['S0'] = self.S0
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.A), name='AMPLITUDE'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.phaseInNm), name='PHASE'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        pitch = hdr['PIXPITCH']
        S0 = hdr['S0']
        ef = ElectricField(dimx, dimy, pitch, S0, target_device_idx=target_device_idx)
        return ef

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'ElectricField':
            raise ValueError(f"Error: file {filename} does not contain an ElectricField object")
        ef = ElectricField.from_header(hdr, target_device_idx=target_device_idx)
        with fits.open(filename) as hdul:
            ef.A = ef.to_xp(hdul[1].data.copy())
            ef.phaseInNm = ef.to_xp(hdul[2].data.copy())
        return ef

    def array_for_display(self):
        frame = self.phaseInNm * (self.A > 0).astype(float)
        idx = self.xp.where(self.A > 0)[0]
        # Remove average phase
        frame[idx] -= self.xp.mean(frame[idx])
        return frame
