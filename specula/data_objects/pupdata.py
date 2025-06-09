import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj

class PupData(BaseDataObj):
    '''
    TODO change to have the pupil index in the second index
    (for compatibility with existing PASSATA data)

    TODO change by passing all the initializing arguments as __init__ parameters,
    to avoid the later initialization (see test/test_slopec.py for an example),
    where things can be forgotten easily
    '''
    def __init__(self,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.radius = self.xp.zeros(4, dtype=self.dtype)
        self.cx = self.xp.zeros(4, dtype=self.dtype)
        self.cy = self.xp.zeros(4, dtype=self.dtype)
        self.ind_pup = self.xp.empty((0, 4), dtype=int)
        self.framesize = np.zeros(2, dtype=int)
        
    @property
    def n_subap(self):
        return self.ind_pup.shape[1] // 4

    def zcorrection(self, indpup):
        tmp = indpup.copy()
        tmp[:, 2], tmp[:, 3] = indpup[:, 3], indpup[:, 2]
        return tmp

    @property
    def display_map(self):
        mask = self.single_mask()
        return self.xp.ravel_multi_index(self.xp.where(mask), mask.shape)

    def single_mask(self):
        f = self.xp.zeros(self.framesize[0]*self.framesize[1], dtype=self.dtype)
        self.xp.put(f, self.ind_pup[:, 0], 1)
        f2d = f.reshape(self.framesize)
        return f2d[:self.framesize[0]//2, self.framesize[1]//2:]

    def complete_mask(self):
        f = self.xp.zeros(self.framesize, dtype=self.dtype)
        for i in range(4):
            f.flat[self.ind_pup[:, i]] = 1
        return f

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 2
        hdr['FSIZEX'] = self.framesize[0]
        hdr['FSIZEY'] = self.framesize[1]

        super().save(filename, hdr)

        fits.append(filename, self.ind_pup)
        fits.append(filename, self.radius)
        fits.append(filename, self.cx)
        fits.append(filename, self.cy)

    def read(self, filename, hdr=None, exten=1):
        super().read(filename)
        self.ind_pup = self.to_xp(fits.getdata(filename, ext=exten))
        self.radius = self.to_xp(fits.getdata(filename, ext=exten + 1))
        self.cx = self.to_xp(fits.getdata(filename, ext=exten + 2))
        self.cy = self.to_xp(fits.getdata(filename, ext=exten + 3))

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version > 2:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        p = PupData(target_device_idx=target_device_idx)
        if version >= 2:
            p.framesize = [int(hdr['FSIZEX']), int(hdr['FSIZEY'])]

        p.read(filename, hdr)
        return p
