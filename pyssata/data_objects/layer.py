import numpy as np
from astropy.io import fits

class Layer(ElectricField):
    def __init__(self, dimx, dimy, pixel_pitch, height, GPU=False, objname="layer", objdescr="layer object", precision=0, type_str=None):
        self._height = height
        self._shiftXYinPixel = np.array([0.0, 0.0])
        self._rotInDeg = 0.0
        self._magnification = 1.0

        super().__init__(dimx, dimy, pixel_pitch, GPU=GPU, objname=objname + ' ef', objdescr=objdescr, precision=precision, type_str=type_str)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    @property
    def shiftXYinPixel(self):
        return self._shiftXYinPixel

    @shiftXYinPixel.setter
    def shiftXYinPixel(self, value):
        self._shiftXYinPixel = value

    @property
    def rotInDeg(self):
        return self._rotInDeg

    @rotInDeg.setter
    def rotInDeg(self, value):
        self._rotInDeg = value

    @property
    def magnification(self):
        return self._magnification

    @magnification.setter
    def magnification(self, value):
        self._magnification = value

    def cleanup(self):
        super().cleanup()

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['HEIGHT'] = self._height
        super().save(filename, hdr)

    def read(self, filename, hdr=None, exten=0):
        super().read(filename, hdr, exten)

    @staticmethod
    def restore(filename):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        dimx = int(hdr['DIMX'])
        dimy = int(hdr['DIMY'])
        height = float(hdr['HEIGHT'])
        pitch = float(hdr['PIXPITCH'])

        layer = Layer(dimx, dimy, pitch, height)
        layer.read(filename, hdr)
        return layer

    def revision_track(self):
        return '$Rev$'