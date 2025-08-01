import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

class ELTM1IFuncCalculator:
    """
    Class to calculate the modal basis for the ELT M1 segments.
    The modal basis is generated from the segmentation map of the M1 mirror.
    The segmentation map is a FITS file that contains the segmentation of the M1 mirror.
    The modal basis is generated by creating a piston, tip, and tilt for each segment.
    The piston is normalized to 1 RMS, and the tip and tilt are normalized to 1 RMS.
    The class also provides a method to save the results in a FITS file and a method to plot the results.
    """

    def __init__(self, dim=512, dtype=np.float32):

        self.dim = dim
        self.dtype = dtype
        self.ifs_cube = None
        self.mask = None
        self.rsegmentation = None

    def load_segmentation_map(self):
        # Load segmentation map
        seg_path = os.path.join(os.path.dirname(__file__), '../data/EltM1SegmMap1015pix38570mm.fits')
        self.size_in_m = 38570.0 / 1e3
        with fits.open(seg_path) as hdul:
            self.segmentation = hdul[0].data.copy()
        self.size_in_pix = self.segmentation.shape[0]
        rescalingFactor = self.segmentation.shape[0] / self.dim
        coord = np.round(np.arange(self.dim) * rescalingFactor).astype(int)
        rsegmentation = np.zeros((self.dim, self.dim))
        for a in range(self.dim):
            for b in range(self.dim):
                rsegmentation[a, b] = self.segmentation[coord[a], coord[b]]
        self.rsegmentation = rsegmentation.astype(self.dtype)

    def compute_mask(self):
        if self.rsegmentation is None:
            self.load_segmentation_map()
        self.mask = (self.rsegmentation > 0).astype(self.dtype)

    def M1_modal_base(self):
        if self.mask is None:
            self.compute_mask()

        pupil = self.mask
        segmentation = self.segmentation
        segm = self.rsegmentation
        idx_pupil = np.where(pupil) 
        tilt,tip = np.meshgrid(np.linspace(-1,1,segm.shape[0]),
                            np.linspace(-1,1,segm.shape[1]))

        M1Base=[]
        for s in range(int(np.max(segmentation))):
            #generate the piston
            pist_s = segm==(s+1)
            idx_s = np.where(pist_s)
            pist_s = pist_s.astype(self.dtype)
            pist_s[idx_s] = pist_s[idx_s] / np.sqrt(np.mean((pist_s[idx_s])**2))
            M1Base.append(pist_s[idx_pupil].copy())

            #generate the segment tip
            tip_s = tip*pist_s
            tip_s[idx_s]*= 1/np.std(tip_s[idx_s])#normalise to 1 (choose your unit) RMS
            tip_s[idx_s]-= np.mean(tip_s[idx_s])#remove the average offset
            M1Base.append(tip_s[idx_pupil].copy())

            #generate the segment tilt
            tilt_s = tilt*pist_s
            tilt_s[idx_s]*=1/np.std(tilt_s[idx_s])#normalise to 1 (choose your unit) RMS
            tilt_s[idx_s]-=np.mean(tilt_s[idx_s])#remove the average offset
            M1Base.append(tilt_s[idx_pupil].copy())

        self.ifs_cube = np.asarray(M1Base, dtype=self.dtype)

    def save_results(self, filename):
        self.M1_modal_base()

        # Save the mask and ifs_2d using IFunc class methods
        ifunc = IFunc(ifunc=self.ifs_cube, mask=self.mask, target_device_idx=-1, precision=0)
        ifunc.save(filename)

    def save_mask(self, filename):
        # Compute mask if not already done
        if self.mask is None:
            self.compute_mask()
        # Calculate pitch from FITS file name: 38570mm / 1015 pixels
        pitch = self.size_in_m / self.dim  # m per pixel        
        simul_params = SimulParams(self.dim, pitch)        
        pupilstop = Pupilstop(simul_params, input_mask=self.mask, mask_diam=self.size_in_m, magnification=1.0)
        pupilstop.save(filename)
        
    def plot_results(self):
        #Nr of total modes = 3x798 (= ptt for each segment)
        n_modes = self.ifs_cube.shape[1]
        mb = np.zeros((self.dim, self.dim, n_modes)) 
        mb[np.where(self.mask>0)] = self.ifs_cube

        fig, ax = plt.subplots(3, 3)
        fig.suptitle('First three segments')
        for i, a in enumerate(ax.flatten()):
            a.imshow(mb[:, :, i], origin='lower')

# Example usage of the class
# from specula.dm.ELT_M1_ifunc_calculator import ELTM1IFuncCalculator
# dim = 480  # Pupil dimension
# calculator = ELTM1IFuncCalculator(dim)
# calculator.save_results('~/ifunc_elt_m1.fits')
# calculator.plot_results()