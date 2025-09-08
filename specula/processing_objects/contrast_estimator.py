import numpy as np
import logging

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels


class ContrastEstimator(BaseProcessingObj):
    """
    SPECULA ProcessingObject for computing PSF radial profiles and contrast.

    Inputs
    ------
    psf : 2D array
        PSF image (numpy or GPU array depending on target_device_idx)

    Outputs
    -------
    radial_profile : 1D array
        Mean radius values
    contrast : 1D array
        Normalized azimuthal average PSF profile
    lowest_profile : 1D array
        Normalized minimum values per annulus
    highest_profile : 1D array
        Normalized maximum values per annulus
    """

    def __init__(self,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.inputs['in_psf'] = InputValue(type=BaseValue)                

        # Internal storage        
        self._radial_profile = BaseValue(value=self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)     
        self._psf_profile = BaseValue(value=self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self._lowest_profile = BaseValue(value=self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self._highest_profile = BaseValue(value=self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)
        self._contrast = BaseValue(value=self.xp.zeros(1, dtype=self.dtype), target_device_idx=self.target_device_idx)

        self.outputs["radial_profile"] = self._radial_profile
        self.outputs["psf_profile"] = self._psf_profile
        self.outputs["contrast"] = self._contrast
        self.outputs["lowest_profile"] = self._lowest_profile
        self.outputs["highest_profile"] = self._highest_profile

    def prepare_trigger(self, t):
        """Fetch local inputs"""
        super().prepare_trigger(t)
        self._psf_pixels = self.local_inputs['in_psf'].get_value()

    def _compute_radial_profiles(self):
        sh = self._psf_pixels.shape
        x, y = self.xp.indices(sh)
        x -= sh[0] // 2
        y -= sh[1] // 2
        r = self.xp.sqrt(x**2 + y**2)
        self._r_int = self.xp.round(r).astype(int)
        self._r_uni = self.xp.unique(self._r_int)        
        radial_profile = []
        psf_profile = []
        lowest_profile = []
        highest_profile = []
        for i in self._r_uni:
            # Radial profile
            radial_profile.append(self.xp.mean(r[self._r_int == i]))
            # Contrast (normalized psf profile)
            psf_profile.append(self.xp.mean(self._psf_pixels[self._r_int == i]))
            # Lowest profile
            lowest_profile.append(self.xp.min(self._psf_pixels[self._r_int == i]))
            # Highest profile
            highest_profile.append(self.xp.max(self._psf_pixels[self._r_int == i]))
        
        self._radial_profile.value = self.xp.array(radial_profile)        
        self._psf_profile.value = self.xp.array(psf_profile)
        self._lowest_profile.value = self.xp.array(lowest_profile)
        self._highest_profile.value = self.xp.array(highest_profile)

    def trigger(self):        
        self._compute_radial_profiles()        
        self._contrast.value = self.xp.log(self._psf_profile.value / self._psf_profile.value[0])
        self._lowest_profile.value = self.xp.log(self._lowest_profile.value / self._lowest_profile.value[0])        
        self._highest_profile.value = self.xp.log(self._highest_profile.value / self._highest_profile.value[0])

    def post_trigger(self):
        # Store outputs
        self.outputs["radial_profile"] = self._radial_profile
        self.outputs["psf_profile"] = self._psf_profile
        self.outputs["contrast"] = self._contrast
        self.outputs["lowest_profile"] = self._lowest_profile
        self.outputs["highest_profile"] = self._highest_profile

        self.outputs['radial_profile'].generation_time = self.current_time
        self.outputs['psf_profile'].generation_time = self.current_time
        self.outputs['contrast'].generation_time = self.current_time
        self.outputs['lowest_profile'].generation_time = self.current_time
        self.outputs['highest_profile'].generation_time = self.current_time

        if self.verbose:
            logging.info(f"[{self.name}] Contrast computed, peak contrast={self._contrast[0]}")

    def finalize(self):
        if self.verbose:
            logging.info(f"[{self.name}] Finalized")
