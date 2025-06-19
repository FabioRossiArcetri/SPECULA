

import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from specula.base_value import BaseValue


class Slopes(BaseDataObj):
    def __init__(self, 
                 length: int=None, 
                 slopes=None, 
                 interleave: bool=False, 
                 target_device_idx: int=None, 
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        if slopes is not None:
            self.slopes = slopes
        else:
            self.slopes = self.xp.zeros(length, dtype=self.dtype)
        self.interleave = interleave
        self.single_mask = None
        self.display_map = None

        if self.interleave:
            self.indicesX = self.xp.arange(0, self.size // 2) * 2
            self.indicesY = self.indicesX + 1
        else:
            self.indicesX = self.xp.arange(0, self.size // 2)
            self.indicesY = self.indicesX + self.size // 2

    # TODO needed to support late SlopeC-derived class initialization
    # Replace with a full initialization in base class?
    def resize(self, new_size):
        self.slopes = self.xp.zeros(new_size, dtype=self.dtype)
        if self.interleave:
            self.indicesX = self.xp.arange(0, self.size // 2) * 2
            self.indicesY = self.indicesX + 1
        else:
            self.indicesX = self.xp.arange(0, self.size // 2)
            self.indicesY = self.indicesX + self.size // 2

    @property
    def size(self):
        return self.slopes.size

    @property
    def xslopes(self):
        return self.slopes[self.indicesX]

    @xslopes.setter
    def xslopes(self, value):
        self.slopes[self.indicesX] = value

    @property
    def yslopes(self):
        return self.slopes[self.indicesY]

    @yslopes.setter
    def yslopes(self, value):
        self.slopes[self.indicesY] = value

    def indx(self):
        return self.indicesX

    def indy(self):
        return self.indicesY

    def sum(self, s2, factor):
        self.slopes += s2.slopes * factor

    def subtract(self, s2):
        if isinstance(s2, Slopes):
            if s2.slopes.size > 0:
                self.slopes -= s2.slopes
            else:
                print('WARNING (slopes object): s2 (slopes) is empty!')
        elif isinstance(s2, BaseValue):  # Assuming BaseValue is another class
            if s2.value.size > 0:
                self.slopes -= s2.value
            else:
                print('WARNING (slopes object): s2 (base_value) is empty!')

    def x_remap2d(self, frame, idx):
        if len(idx.shape) == 1:
            frame.flat[idx] = self.slopes[self.indx()]
        elif len(idx.shape) == 2:
            frame[idx] = self.slopes[self.indx()]
        else:
            raise ValueError('Frame index must be either 1d for flattened indexes or 2d')

    def y_remap2d(self, frame, idx):
        if len(idx.shape) == 1:
            frame.flat[idx] = self.slopes[self.indy()]
        elif len(idx.shape) == 2:
            frame[idx] = self.slopes[self.indy()]
        else:
            raise ValueError('Frame index must be either 1d for flattened indexes or 2d')

    def get2d(self):
        if self.single_mask is None:
            raise ValueError('Slopes single_mask has not been set')
        if self.display_map is None:
            raise ValueError('Slopes display_map has not been set')
        mask = self.single_mask
        idx = self.display_map
        fx = self.xp.zeros_like(mask, dtype=self.dtype)
        fy = self.xp.zeros_like(mask, dtype=self.dtype)
        self.x_remap2d(fx, idx)
        self.y_remap2d(fy, idx)
        return self.to_xp([fx, fy], dtype=self.dtype)

    def rotate(self, angle, flipx=False, flipy=False):
        sx = self.xslopes
        sy = self.yslopes
        alpha = self.xp.arctan2(sy, sx) + self.xp.radians(angle)
        modulus = self.xp.sqrt(sx**2 + sy**2)
        signx = -1 if flipx else 1
        signy = -1 if flipy else 1
        self.xslopes = self.xp.cos(alpha) * modulus * signx
        self.yslopes = self.xp.sin(alpha) * modulus * signy

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 2
        hdr['INTRLVD'] = int(self.interleave)
        if hasattr(self, 'pupdata_tag') and self.pupdata_tag is not None:
            hdr['PUPD_TAG'] = self.pupdata_tag
        if hasattr(self, 'subapdata_tag') and self.subapdata_tag is not None:
            hdr['SUBAP_TAG'] = self.subapdata_tag
        fits.writeto(filename, np.zeros(2), hdr)
        fits.append(filename, cpuArray(self.slopes))

    def read(self, filename, hdr=None, exten=1):
        super().read(filename)
        self.slopes = fits.getdata(filename, ext=exten)

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version > 2:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        s = Slopes(length=1, target_device_idx=target_device_idx)
        s.interleave = bool(hdr['INTRLVD'])
        if version >= 2:
            # Read optional tags if present
            for tag_key, attr_name in [('PUPD_TAG', 'pupdata_tag'), ('SUBAP_TAG', 'subapdata_tag')]:
                if tag_key in hdr:
                    try:
                        tag_value = str(hdr[tag_key]).strip()
                        if tag_value:  # Non-empty
                            setattr(s, attr_name, tag_value)
                    except (ValueError, TypeError):
                        pass  # Skip invalid tag values
        s.read(filename, hdr)
        return s

    def array_for_display(self):
        return self.xp.hstack(self.get2d())
