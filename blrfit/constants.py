"""
Every number that defines the model, a class, a flag or a selection, with the
reason for its value. Tolerances of the fitting machinery (penalty scales,
pixel-count thresholds, starting amplitudes) stay next to the code they serve.

The values are those of the frozen production fitter that produced the DESI
offset-line catalogue. They were set on synthetic spectra, on the SDSS spectra
of the published offset-line samples and on a small number of DESI test objects
before the catalogue was built; where a value was set by a test rather than
taken from the literature, the test is described next to it and the object is
named where it mattered. The constants are bound by name when the package is
imported and are not meant to be changed at run time: changing any value
invalidates the validation numbers quoted in the README (the synthetic suite,
the Runnoe et al. 2015 and the Liu et al. 2014 same-spectrum comparisons).

Units: velocities and widths in km/s, wavelengths in vacuum Angstrom.
"""
from pathlib import Path

import numpy as np

C_KMS = 299792.458
S2F = 2.0 * np.sqrt(2.0 * np.log(2.0))       # sigma -> FWHM for a Gaussian, 2.3548
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# ----------------------------------------------------------------------------
# Line list (vacuum wavelengths, Angstrom) and fixed doublet ratios
# ----------------------------------------------------------------------------
LAM = dict(
    Hbeta=4862.68, OIII4959=4960.30, OIII5007=5008.24, HeII4686=4687.02,
    Halpha=6564.61, NII6548=6549.86, NII6584=6585.27,
    SII6716=6718.29, SII6731=6732.67,
    MgII=2798.75, MgII2796=2796.35, MgII2803=2803.53,
)
R_OIII = 2.98        # [O III] 5007/4959 (Storey & Zeippen 2000)
R_NII = 2.96         # [N II] 6584/6548 (as in Shen et al. 2011)

# The three complexes that can be fitted (rest-frame windows, Angstrom) and the
# line each one is measured against.
COMPLEX_WINDOW = {"Hbeta": (4700., 5100.), "Halpha": (6400., 6800.),
                  "MgII": (2700., 2900.)}
COMPLEX_LINE = {"Hbeta": "Hbeta", "Halpha": "Halpha", "MgII": "MgII"}
COMPLEX_PREFIX = {"Halpha": "Ha", "Hbeta": "Hb", "MgII": "Mg"}

# Line-free continuum windows of Shen et al. (2011), also used by PyQSOFit.
CONTI_WINDOWS = [(1150, 1170), (1275, 1290), (1350, 1360), (1445, 1465),
                 (1690, 1705), (1770, 1810), (1970, 2400), (2480, 2675),
                 (2925, 3400), (3775, 3832), (4000, 4050), (4200, 4230),
                 (4435, 4640), (5100, 5535), (6005, 6035), (6110, 6250),
                 (6800, 7000), (7160, 7180), (7500, 7800), (8050, 8150)]

# ----------------------------------------------------------------------------
# Pre-processing and continuum
# ----------------------------------------------------------------------------
ERR_FLOOR = 0.02          # fractional flux error added in quadrature (see fit_spectrum)
PL_PIVOT = 3000.0         # power-law pivot wavelength, Angstrom
PL_ALPHA_MIN, PL_ALPHA_MAX = -5.0, 3.0
FE_FWHM_MIN, FE_FWHM_MAX = 1200.0, 10000.0   # Fe II broadening range
FE_SHIFT_MAX = 0.01                          # fractional velocity shift of the Fe II templates
FE_INTRINSIC_FWHM = 900.0                    # intrinsic width of the I Zw 1 templates
N_GAL_MAX = 5              # galaxy eigenspectra, stepped down until non-negative
MIN_HOST_FRAC = 0.10       # host kept only above this fraction of the 4200-5000 A flux (Shen et al. 2011)
HOST_CONTINUUM_ONLY = True # host contributes continuum only under the emission lines
HOST_LINE_HALFWIDTH_KMS = 900.0   # +/- window interpolated over in the host model
HOST_ZMAX = 1.2            # no host decomposition above this redshift (an adopted limit of the production run)
CLIP_LO, CLIP_HI = -3.0, 5.0      # outlier clipping of the continuum fit, one iteration

# ----------------------------------------------------------------------------
# Narrow lines
# ----------------------------------------------------------------------------
# Narrow FWHM <= 1200 km/s, broad FWHM >= 1200 km/s: the conventional division
# (Shen et al. 2011). Both are stored as sigma.
SIG_NARROW_MAX = 1200.0 / S2F      # 509.6
SIG_NARROW_MIN = 25.0
SIG_BROAD_MIN = 1200.0 / S2F
SIG_BROAD_MAX = 15000.0 / S2F      # FWHM up to 15,000 km/s

# Window for the narrow-line velocity about the input redshift. The pipeline
# redshift of a broad-line object is driven by the broad lines themselves, that
# is, by the displacements under study: SDSS J102106.04+452331.8 (Eracleous et
# al. 2012) has its narrow lines 880 km/s from its catalogue redshift and a
# +/-700 km/s window could not reach them. The narrow groups are multi-line
# patterns ([N II] doublet + narrow Halpha; [O III] doublet), so the wide window
# is safe for them; a group at the bound is flagged. Mg II has one narrow line
# only and keeps the tight window.
V_NARROW_MAX = 1500.0
V_NARROW_MAX_MGII = 700.0

# [S II] kinematics in the Halpha complex: own velocity and width, tied softly
# to narrow Halpha + [N II] by Gaussian priors of these widths. A hard tie (all
# narrow lines sharing one velocity and width) leaves residuals at [S II] in
# high signal-to-noise host galaxies that the fit absorbs with a spurious broad
# Gaussian at +7000 km/s; fully free [S II] removes the anchor that defines the
# narrow width in quasars, and narrow Hbeta residuals are then absorbed as
# minimum-width broad components. The soft tie behaves as the hard tie where
# [S II] is weak and lets a strong [S II] depart at a small cost in chi-square.
SII_PRIOR_SIG_FRAC = 0.20
SII_PRIOR_V_KMS = 60.0

# Narrow-line-region wing: a second narrow component under every narrow line of
# the Halpha complex (one amplitude fraction, one velocity, one width shared by
# all lines; carried into narrow Hbeta from the Halpha fit). AGN narrow lines
# carry broad bases with 10-40 per cent of the flux (Heckman et al. 1981;
# Whittle 1985); with one Gaussian per line the information criterion buys
# 1200-2000 km/s "broad" components on these bases, which then contaminate the
# broad profile: a synthetic weak broad Halpha at +800 km/s came back at +379
# without the wing and at +803 with it. Pinned by [N II] and [S II], which have
# no broad counterpart, the wing cannot masquerade as broad Halpha; it is
# excluded from every broad-profile measure.
NLR_WING = True
NW_V_PRIOR_KMS = 150.0    # wing velocity within this of the core velocity (prior width)
NW_F_MAX = 0.5            # wing amplitude fraction of each core
NW_F_PRIOR = 0.25         # weak prior pulling the fraction to zero: it decides only where
#                           [N II]/[S II] are too weak to pin the wing (quasars), where the
#                           wing could otherwise take broad-Halpha core flux

# [O III] core plus blueshifted wing (Heckman et al. 1981) in the Hbeta complex.
SIG_O3_CORE_MAX = 800.0
SIG_WING_MIN, SIG_WING_MAX = 250.0, 2500.0
V_WING_MIN, V_WING_MAX = -2500.0, 500.0     # [O III] wings are blue
# With overlapping width ranges the two components can exchange roles (the wing
# takes the line, the core wanders to a bound) and the systemic velocity is then
# unconstrained. Two hinge penalties order them: the wing at least as wide as
# the core, and the core at least as tall as the wing. A blue component taller
# than the core is the rare outflow-dominated [O III]; such objects show a poor
# [O III] fit and are flagged. The amplitude ordering was added after the core
# of SDSS J143123.52+392501.4 shrank to nothing while the wing took the line.
# The hard split of PyQSOFit (core <= 510, wing > 510 km/s) was tried and misfit
# genuine intermediate-width wings.
O3_ORDER_WIDTH = True
O3_ORDER_AMPLITUDE = True
O3_AMP_ORDER_SCALE = 0.05  # penalty scale for the amplitude ordering, fraction of core + wing

# Systemic velocity of the Halpha complex: the low-ionisation narrow group. A
# preliminary fit of the Hbeta complex provides an [O III] velocity from which
# the Halpha group is started as well as from zero, and a Gaussian prior of
# width SYS_PRIOR_KMS pulls the group toward it when the [O III] core is
# detected at S/N >= SYS_PRIOR_MIN_SNR. A start alone is not enough: under a very
# broad Halpha with weak narrow lines, mistaking [N II] 6584 for narrow Halpha
# (a 940 km/s displacement) can have marginally lower chi-square (the DESI
# spectrum of SDSS J101438.27+031211.1: [O III] at -86 km/s, the Halpha group
# slid to -714 km/s without the prior). Low-ionisation
# lines and the [O III] core agree to ~200 km/s in essentially all AGN, so the
# prior costs nothing where the narrow lines are strong and decides the
# assignment where they are not. A remaining disagreement is flagged.
O3_INFORMED_START = True
O3_START_MIN_SNR = 5.0      # [O III] pre-fit S/N needed to add its velocity as a start
SYS_PRIOR_KMS = 150.0
SYS_PRIOR_MIN_SNR = 10.0
SYS_DISAGREE_KMS = 400.0    # flag 'sys_disagree' beyond this |v([O III]) - v_sys| (used through DEFAULT_THRESH)
SYS_DISAGREE_MIN_SNR = 5.0  # ... when the [O III] core has at least this S/N
NARROW_PRIOR_MIN_SNR = 3.0  # Halpha narrow peak S/N needed to fix the Hbeta narrow kinematics

# ----------------------------------------------------------------------------
# Broad components
# ----------------------------------------------------------------------------
# Centre of each broad component within +/- V_BROAD_MAX of the line. Individual
# components of the disk-like profiles of Eracleous et al. (2012) reach -6300
# km/s; a +/-5000 km/s bound truncated the wings of double-peaked profiles and
# roughly halved their measured widths.
V_BROAD_MAX = 8000.0
# A broad component far from the line must itself be broad: FWHM >= slope * |v|
# (hinge penalty). At the signal-to-noise ratio of DESI host galaxies, small
# imperfections of the narrow-line model are significant and a minimum-width
# "broad" Gaussian would otherwise be parked on [S II] (+7020 and +7680 km/s
# from Halpha) or on [O III] 4959 (+6020 km/s from Hbeta, in the model's linear
# velocity convention). With slope 0.4 a
# component at +7000 km/s must be at least 2800 km/s wide, while the displaced
# components of genuine disk emitters (FWHM 3000-5000 at |v| 5000-8000 km/s in
# the Eracleous et al. 2012 sample) are untouched.
BROAD_WIDTH_SLOPE = 0.4
BROAD_WIDTH_PENALTY_SCALE = 10.0   # km/s of sigma deficit per unit residual
BROAD_STARTS_KMS = (0.0, 1500.0, -1500.0, 3000.0, -3000.0)   # first-component starts
MAX_BROAD = 3
DBIC = 10.0               # a more complex model must improve the BIC by this much
# Coverage: a complex is fitted only if the data extend at least CORE_COVER_KMS
# beyond the line centre on both sides; broad-component centres are confined to
# the covered range less EDGE_MARGIN_KMS; coverage below EDGE_FLAG_KMS on either
# side is flagged 'edge'. Where one flank of the line runs off the detector
# (Halpha at z >~ 0.477 in DESI), an unconstrained fit places components outside
# the data and manufactures offsets of +4000 to +8000 km/s.
CORE_COVER_KMS = 3500.0
EDGE_MARGIN_KMS = 1000.0
EDGE_FLAG_KMS = 6000.0
MIN_COMPLEX_PIXELS = 60   # pixels needed inside the complex window
MIN_COMPLEX_COVER = 0.6   # ... and fraction of the window that must be covered
MAX_NFEV_COMPLEX = 600    # least-squares evaluation budget per start
MAX_NFEV_CONTI = 400
MAX_NFEV_CONTI_HOST = 500

# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------
PROFILE_GRID_KMS = 5.0            # velocity grid on which the broad model is measured
PROFILE_GRID_MAX_KMS = 25000.0
PEAK_HEIGHT_FRAC = 0.20           # local maxima above this fraction of the profile maximum ...
PEAK_PROMINENCE_FRAC = 0.05       # ... with at least this prominence count as peaks
DATA_PEAK_FRAC = 0.80             # parabola fitted above this fraction of the smoothed data peak
DATA_PEAK_SMOOTH_KMS = 250.0
DATA_PEAK_GUARD_HBETA_KMS = 400.0 # narrow components masked to this half-width for the data peak (Hbeta only)
DATA_SMOOTH_KMS = 150.0           # smoothing of the data-side cross-check profile

# ----------------------------------------------------------------------------
# Classification thresholds (see classify.py for the rules)
# ----------------------------------------------------------------------------
DEFAULT_THRESH = dict(
    min_broad_flux_snr=5.0, # integrated S/N of the broad profile below which there is no broad line (E)
    min_broad_peak_snr=1.5, # and its peak must stand above the noise at all (E)
    min_fwhm=1200.0,        # broad-line definition (E below)
    min_narrow_snr=3.0,     # systemic reference must be detected (X below)
    class_fwhm=2000.0,      # below this the broad component is too narrow to classify (W)
    class_flux_snr=8.0,     # ... or too weak (W)
    dpe_min_fwhm=3000.0,    # two-peak (B) rule only for genuinely broad profiles
    offset_kms=300.0,       # |c(1/2) - v_sys| for a significant shift ...
    offset_nsig=3.0,        # ... and at least this many sigma when an error is available
    tilt_frac=0.10,         # |c(1/4) - c(3/4)| / FWHM below this = straight bisector
    ai_max=0.12,            # |A.I.| below this = symmetric
    peak_cen_frac=0.20,     # |v_peak - centroid| / FWHM below this = symmetric
    ki_boxy=0.60,           # K.I. above this = flat-topped (Gaussian 0.456); reported, not used in a rule
    dpe_sep_frac=0.30,      # reported with the B rule (the rule uses max(0.4 FWHM, 1500 km/s))
    dpe_dip=0.08,           # dip between two peaks deeper than this = double-peaked
    flag_peak_snr=5.0,      # flag 'low_peak_snr' below this peak S/N per pixel
    flag_host_frac=0.80,    # flag 'host_dominated' at/above this host fraction
    sii_disagree_kms=150.0, # flag 'sii_disagree' if |v([S II]) - v_sys| exceeds this
    sys_disagree_kms=SYS_DISAGREE_KMS,  # flag 'sys_disagree' if |v([O III] core) - v_sys| exceeds this
    extreme_kms=4000.0,     # flag 'extreme_offset' beyond this |c(1/2) - v_sys|
)
VERY_BROAD_FWHM = 8000.0          # flag 'very_broad'
POOR_FIT_CHI2 = 2.5               # flag 'poor_fit' (reduced chi-square)
LOW_SNR_FLUX = 10.0               # flag 'low_snr' (integrated broad S/N)
DPE_SEP_MIN_KMS = 1500.0          # two peaks must be separated by more than max(0.4 FWHM, this)
VERY_BROAD_B_FWHM = 7000.0        # very broad + asymmetric/flat = B
VERY_BROAD_B_AI = 0.20
VERY_BROAD_B_KI = 0.50
PEAK_DISAGREE_FRAC = 0.25         # flag 'peak_disagree' if model and data peaks differ by this * FWHM

# Selection definitions of the DESI catalogue, applied to the class and flags.
MEASURABLE_CLASSES = ("A", "B", "C", "F")
MEASURABLE_MIN_SNR = 8.0
MEASURABLE_MIN_FWHM = 2000.0
STRONG_OFFSET_KMS = (1000.0, 4000.0)

# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------
# Per-measurement error of Delta v from 8377 pairs of independent DESI spectra
# of the same objects, as a function of the integrated broad-line S/N; pairs
# involving a strong offset scatter 1.5 times more.
ERR_MODEL_NORM_KMS = 650.0
ERR_MODEL_FLOOR_KMS = 45.0
ERR_MODEL_STRONG_FACTOR = 1.5
MC_DEFAULT_N = 30
MC_MIN_SAMPLES = 5

# ----------------------------------------------------------------------------
# Cross-correlation between epochs (rv.py)
# ----------------------------------------------------------------------------
CCF_VMAX_KMS = 2000.0             # default search range
CCF_WIN_FWHM = 1.5                # window +/- this * FWHM about c(1/2) ...
CCF_WIN_MIN_KMS = 2000.0          # ... but at least this
CCF_NSUB_FRAC = 0.07              # narrow-subtraction uncertainty, fraction of the narrow model
CCF_CLIP_SIGMA = 5.0              # outlier clip at the first-pass minimum
CCF_DCHI2_99 = 6.63               # Delta chi-square for a 99 per cent interval on one parameter
CCF_SIG_FROM_99 = 2.576           # 99 per cent two-sided <-> 1 sigma
CCF_SYS_KMS = {"Halpha": 155.0, "Hbeta": 79.0}      # on-sky floors from consecutive DESI epochs
CCF_SYS_HBETA_LOWSNR_KMS = 157.0                    # Hbeta at S/N proxy < 8
CCF_DIR_CUT_KMS = {"Halpha": 466.0, "Hbeta": 238.0} # direction-mismatch cut of the reliable tier
CCF_PROFILE_Z_MAX = 5.0                             # profile-stability cut of the reliable tier
# Profile-stability grades for pairs across surveys (an SDSS spectrum against a
# DESI template). z_prof is a significance: for DESI-DESI pairs its median is
# -14 and 3 per cent of the pairs exceed 5, for SDSS-DESI pairs 43 per cent of
# the Halpha points exceed 5 and the value rises with signal-to-noise, because
# resolution, aperture and calibration differences between the surveys become
# significant as the noise shrinks. The velocity scatter of such points does
# not grow until z_prof is well above 5, so instead of the hard cut they are
# graded and their error floor inflated. Measured on the SDSS-to-DESI velocity
# change of the non-candidate objects of the DESI catalogue (direction cut
# passed; Halpha 582 / 145 / 339 points, Hbeta 781 / 69 / 87 points in the
# three grades): NMAD 143, 161, 262 km/s (Halpha) and 147, 208, 221 km/s (Hbeta).
PROFILE_GRADE_Z = (("stable", 5.0), ("mild", 10.0), ("changed", np.inf))   # upper bounds of z_prof
PROFILE_GRADE_INFLATION = {"Halpha": {"stable": 1.0, "mild": 1.15, "changed": 1.85},
                           "Hbeta": {"stable": 1.0, "mild": 1.4, "changed": 1.5}}
CCF_NULL_KMS = {"Halpha": 143.0, "Hbeta": 147.0}  # cross-survey floor of a stable-grade point (as above)
