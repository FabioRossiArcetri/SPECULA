"""Assemble config/ANDES/andes_pl_piston_pyr.yml from the ideal-corrector config.

The only difference: the high-order stage is driven by a modulated pyramid (the real
ANDES SCAO sensor) through a modal reconstructor, instead of by a perfect modal
analysis of the residual. The DM, the atmosphere, the petal loop and the whole
lantern chain are untouched. Text surgery on the YAML keeps every comment of the
original file intact.
"""
import io

SRC = 'config/ANDES/andes_pl_piston.yml'
DST = 'config/ANDES/andes_pl_piston_pyr.yml'

CHAIN = r"""# ---------------------------------------------------------------------------
# The real ANDES SCAO sensor: a modulated pyramid, configured as in
# config/ANDES/params_andes.yml (100 subapertures across the pupil, 120 spacing,
# fov 2.1", modulation 3 lambda/D, 240-pixel detector), sensing the same wavefront
# the ideal-corrector configuration analysed perfectly, and driving the same
# 79-actuator high-order DM through a modal reconstructor.
#
# Calibrations for this geometry (generated here, not part of the upstream ANDES
# data set): calib/pupdata/pl19_1200_pyr_pupdata from calib_pl_pyr_pupdata.yml, and
# calib/rec/pl19_1200_pyr_rec from calib_pl_pyr_rec.yml (500 modes).
# ---------------------------------------------------------------------------
pyramid:
  class:             'ModulatedPyramid'
  simul_params_ref:  'main'
  pup_diam:          100.0                   # pupil diameter in subapertures
  pup_dist:          120.0                   # pupil centre separation
  fov:               2.1                     # [arcsec]
  fov_errinf:        0.1
  fov_errsup:        9.0
  mod_amp:           3.0                     # [lambda/D]
  output_resolution: 240                     # [pixel]
  fft_res:           4.0
  wavelengthInNm:    1200                    # same band the lantern senses
  rotAnglePhInDeg:   15.0
  inputs:
    in_ef: 'prop.out_wfs_source_ef'
  outputs: ['out_i']

detector:
  class:             'CCD'
  simul_params_ref:  'main'
  size:              [240, 240]
  dt:                0.002
  bandw:             400                     # [nm]
  binning:           1
  photon_noise:      true
  readout_noise:     true
  readout_level:     0.5
  quantum_eff:       1.0
  photon_seed:       2
  readout_seed:      2
  excess_noise:      true
  excess_seed:       2
  background_noise:  false
  darkcurrent_noise: false
  inputs:
    in_i: 'pyramid.out_i'
  outputs: ['out_pixels']

slopec:
  class:             'PyrSlopec'
  pupdata_object:    'pl19_1200_pyr_pupdata'
  slopes_from_intensity: true
  inputs:
    in_pixels: 'detector.out_pixels'
  outputs: ['out_slopes', 'out_pupdata']

modalrec:
  class:             'Modalrec'
  recmat_object:     'pl19_1200_pyr_rec'
  inputs:
    in_slopes: 'slopec.out_slopes'
  outputs: ['out_modes']

# 500 of the 4932 DM modes: the interaction matrix is calibrated for these, so the
# high-order residual contains the fitting error of the remaining ones, as in a
# real SCAO loop that does not have the mirror's full authority.
control_ho:
  class:             'Integrator'
  int_gain:          [0.4]
  n_modes:           [500]
  delay:             1
  inputs:
    delta_comm:      'modalrec.out_modes'
  outputs: ['out_comm']

"""

src = io.open(SRC, encoding='utf-8').read()
i = src.index('\nmodal_analysis_ho:')
j = src.index('\ndm_ho:')
out = src[:i + 1] + CHAIN + src[j + 1:]

# the high-order DM is now commanded by the pyramid loop
assert "'integrator_ho.out_comm'" in out
out = out.replace("'integrator_ho.out_comm'", "'control_ho.out_comm'")

# the header note about the stand-in no longer describes this file
out = out.replace(
    '# High-order stage: the analogue of the M4 + pyramid/MMSE block of the other\n'
    '# ANDES configurations (their calibration files are not part of this repository).\n'
    '# A zonal DM, 79 actuators across the ANDES pupil (the M4 scale), closed by a modal\n'
    '# analysis of the residual wavefront. Its influence functions are orthogonal to\n'
    '# the petal pistons, so the high orders are corrected and the pistons are left to\n'
    '# the lantern: the two loops control disjoint degrees of freedom.\n'
    '# Built by tools/andes/make_high_order_dm.py.\n',
    '# High-order correcting DM: 79 actuators across the ANDES pupil (the M4 scale), with\n'
    '# influence functions orthogonal to the petal pistons, so the two loops control\n'
    '# disjoint degrees of freedom. Built by tools/andes/make_high_order_dm.py.\n'
    '# In THIS file it is driven by the pyramid chain above, not by a modal analysis.\n')

io.open(DST, 'w', encoding='utf-8', newline='\n').write(out)
print(f'{DST}: {out.count(chr(10))} lines, pyramid chain inserted')
