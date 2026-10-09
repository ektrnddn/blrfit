# Method

blrfit fits one optical spectrum at a time: a continuum, the narrow lines and the broad
lines of the Hα and Hβ regions (and, experimentally, Mg II). The broad profile is then
measured without reference to the Gaussians it was built from. Every number that
defines the model, a class or a flag is in `blrfit/constants.py`, with the reason for
its value.

## Pre-processing

- **Extinction.** Galactic extinction is removed with the Cardelli, Clayton & Mathis
  (1989) law and the O'Donnell (1994) optical coefficients, R_V = 3.1. Where E(B−V)
  comes from is described in [inputs](inputs.md#galactic-extinction).
- **Masks.** A pixel is unusable when its flux or error is not finite or its inverse
  variance is zero (survey masks are applied by the readers). The two pixels on either
  side of every unusable pixel are excluded as well: single-pixel artefacts next to
  masked pixels otherwise survive the mask. In the DESI example such an artefact at
  6140 Å raised the reduced χ² of Hβ from 1.9 to 2.9.
- **Error floor.** A fractional error of 2 per cent of the flux is added in quadrature
  to the pixel errors that weight the fit. The narrow-line cores of bright galaxies
  reach per-pixel signal-to-noise ratios of several hundred, where the residuals of any
  line model are limited by the spectrophotometric calibration; without the floor they
  dominate χ² and drag the broad components.
- **Rest frame.** The spectrum is moved to the rest frame of the input redshift,
  f_rest(λ/(1+z)) = (1+z) f_obs(λ).

## Continuum

The continuum is a power law A(λ/3000 Å)^α with −5 ≤ α ≤ 3, the optical Fe II template
of Boroson & Green (1992), the ultraviolet composite of Vestergaard & Wilkes (2001),
Salviander et al. (2007) and Tsuzuki et al. (2006) assembled as in Shen et al. (2011),
each convolved to a free FWHM between 1200 and 10 000 km/s with a free normalisation and
a small velocity shift, and a host galaxy built from up to five galaxy eigenspectra of
Yip et al. (2004). All components are fitted together by weighted least squares in the
line-free windows of Shen et al. (2011) and in every line-free pixel inside the range of
the galaxy templates, so that the 4000 Å break and the stellar absorption anchor the
host. Pixels more than 3σ below or 5σ above the first solution are masked once and the
fit is repeated.

- **Several starts.** The fit starts from α = −1.5 and −3.0, each with the host at 0.3
  and 0.9 of the template scale. Every start is fitted on the same pixels, and the
  earliest start within Δχ² = 1 of the lowest χ² is kept. A single start can stop in a
  local minimum: in SDSS J001224.01−102226.5 it ended at χ² 7913 without a host, against
  5333 with a host of 39 per cent, and the Hα offset moved by 500 km/s between the two.
  All starts are recorded in `continuum_solver['starts']`.
- **Host.** The number of eigenspectra is stepped down (5 → 3 → 2 → 1 → 0) until at most
  max(50, 2 per cent) of the pixels in the host range are negative; those are set to
  zero. The host is kept only if it contributes at least 10 per cent of the 4200–5000 Å
  flux. This rule is our choice: Shen et al. (2011) did not decompose individual
  spectra, but corrected L5100 for the host statistically.
- **Host under the lines.** The eigenspectra are principal components of real galaxy
  spectra and contain emission lines whose strengths the line-free fit does not
  constrain; subtracting the full host displaced the systemic velocity by up to
  450 km/s in strongly star-forming DESI hosts. Within ±900 km/s of every catalogued
  line the host model is therefore replaced by a linear interpolation.
- **Known weakness.** With a host, the power law often becomes steeper than an
  accretion disc (α < −7/3 for a thin disc): 59 per cent of the host fits of the SDSS
  comparison sample have α < −2.5, and no fit without a host does. Such fits carry the
  flag `pl_unphysical`, and `continuum_luminosity()` warns, because the luminosity at
  5100 Å then depends on the decomposition. The line measurements are much less
  sensitive to it.

## Narrow lines

Two complexes are fitted after continuum subtraction: Hα (rest 6400–6800 Å: narrow and
broad Hα, [N II] λλ6548, 6584, [S II] λλ6716, 6731) and Hβ (4700–5100 Å: narrow and broad
Hβ, [O III] λλ4959, 5007). The model's narrow He II λ4686 lies just blueward of the Hβ
window, so the data do not constrain it. Doublet ratios are fixed, [N II] 6584/6548 = 2.96
and [O III] 5007/4959 = 2.98; the [S II] λ6716/λ6731 ratio is free and not restricted to
its physical range (about 0.44–1.45). Narrow widths are 25 ≤ σ ≤ 510 km/s
(FWHM ≤ 1200 km/s).

- **Ties.** Narrow Hα and [N II] share one velocity v_n and one width. [S II] has its
  own, tied softly by Gaussian priors (σ within 20 per cent, v within 60 km/s): a hard
  tie leaves residuals at [S II] in high signal-to-noise host galaxies, which the fit
  absorbs with a spurious broad Gaussian at +7000 km/s, and a free [S II] removes the
  anchor that defines the narrow width in quasars.
- **Narrow-line wing.** A second Gaussian under every narrow line of the Hα complex,
  with a common amplitude fraction f_w ≤ 0.5 (weak prior σ = 0.25 towards zero), one
  velocity tied to v_n by a prior of 150 km/s and one width between σ_n and 510 km/s,
  absorbs the non-Gaussian bases of AGN narrow lines (Heckman et al. 1981; Whittle
  1985). Without it, the information criterion buys "broad" components of FWHM
  1200–2000 km/s on those bases: in a synthetic spectrum with 25 per cent of the narrow
  flux in a base three times wider than the core, a broad Hα injected at +800 km/s came
  back at +379 km/s without the wing and at +803 km/s with it. Pinned by [N II] and
  [S II], which have no broad counterpart, the wing cannot imitate broad Hα; it is
  excluded from every broad-profile measure.
- **[O III].** The doublet is a core (σ ≤ 800 km/s) plus a blueshifted wing (σ
  250–2500 km/s, v between −2500 and +500 km/s), ordered by two hinge penalties (wing at
  least as wide as the core, core at least as tall as the wing) so that the two cannot
  exchange roles. The core starts at the smoothed data peak near 5007 Å.
- **Hβ narrow lines.** When the Hα narrow lines are detected (peak S/N ≥ 3), the
  velocity and width of narrow Hβ, He II and the wing are fixed to the Hα values;
  otherwise they are tied to the [O III] core, as in Shen et al. (2011).

## Systemic velocity

v_n is the velocity of narrow Hα + [N II] (with [S II] attached), the narrow-line system
least affected by outflows and available for every object. It may lie within
±1500 km/s of the input redshift, because the pipeline redshift of a broad-line object
can be driven by the broad lines (SDSS J102106.04+452331.8 of Eracleous et al. 2012 has
its narrow lines 880 km/s from its catalogue redshift). The wide window allows
misassociation: [N II] λ6584 taken for narrow Hα displaces v_n by 940 km/s, and under a
very broad Hα the wrong solution can have marginally lower χ². A preliminary Hβ fit
therefore provides an [O III] velocity from which the Hα group is also started, and, when
the [O III] core has S/N ≥ 10, a Gaussian prior of 150 km/s pulls v_n towards it. Strong
narrow lines override the prior; a remaining disagreement above 400 km/s is flagged. In
the DESI sample the reference systems agree: [S II] deviates from v_n by +3 ± 17 km/s and
the [O III] core (where its S/N ≥ 10) by −4 ± 28 km/s (median ± NMAD).

## Broad lines

One to three Gaussians with 1200 ≤ FWHM ≤ 15 000 km/s and centres within ±8000 km/s of
the line. Individual components of the disc-like profiles of Eracleous et al. (2012)
reach −6300 km/s; a ±5000 km/s bound truncated the wings of double-peaked profiles and
roughly halved their widths.

- **Number of components.** Chosen with the Bayesian information criterion,
  BIC = χ² + k ln N: the fewest components unless a more complex model improves the BIC
  by more than 10.
- **Starts.** Each fit is repeated from starting velocities 0, ±1500 and ±3000 km/s of
  the first component (and, for Hα, from two starts of the narrow group); the solution
  with the lowest χ² is kept. The solver is scipy's bounded trust-region least squares.
- **Width–velocity hinge.** A penalty enforces FWHM ≥ 0.4 |v|, so that a component at
  +7000 km/s must be at least 2800 km/s wide and cannot sit on [S II] (+7020 and
  +7680 km/s from Hα) or on [O III] λ4959 (+6020 km/s from Hβ) to absorb narrow-line
  residuals; disc-emitter components (FWHM 3000–5000 km/s at |v| 5000–8000 km/s) are not
  affected.
- **Coverage.** A complex is fitted only if the data extend 3500 km/s beyond the line on
  both sides; the broad centres are confined to the covered range less 1000 km/s, and
  coverage below ±6000 km/s is flagged `edge`. Hα at z ≳ 0.477 leaves the red end of
  DESI; unconstrained fits there placed components outside the data and manufactured
  offsets of +4000 to +8000 km/s.

## Measurements

The summed broad model P(v), evaluated on a 5 km/s grid, is reduced to the
non-parametric quantities of Marziani et al. (1996) and Eracleous et al. (2012):

- the peak velocity and the flux-weighted centroid;
- the bisector centres c(f), the midpoints between the two outermost crossings of the
  profile with f times its peak, and the widths W(f), for f = 1/4, 1/2, 3/4 and 0.9
  (FWHM ≡ W(1/2));
- the asymmetry index A.I. = [v_R(1/4) + v_B(1/4) − 2 v_peak] / W(1/4) and the kurtosis
  index K.I. = W(3/4)/W(1/4), 0.456 for a Gaussian;
- the number of resolved peaks (local maxima above 20 per cent of the maximum with a
  prominence of 5 per cent), their separation and the dip between them;
- the broad flux, its equivalent width against the total continuum and against the AGN
  continuum alone, its luminosity, and the integrated and peak signal-to-noise ratios.

The **primary offset** is Δv = c(1/2) − v_n, the displacement of the half-maximum
bisector from the narrow lines. The peak of a multi-component model is unstable for
flat-topped or two-humped profiles, and the centroid weighs the faint wings on which
different decompositions of the same profile legitimately disagree; c(1/2) is the most
stable of the three, and the others are reported alongside. When the profile has a
shoulder close to half maximum, c(1/2) can still jump between decompositions; the
degeneracy check below detects this. As a check on the model, the peak of the
continuum- and narrow-subtracted data is measured as in Eracleous et al. (2012), by a
parabola through the smoothed profile above 80 per cent of its maximum, and compared
with the model peak (flag `peak_disagree`).

**Conventions.** Velocities are optical-convention velocities, v = c(λ/λ₀ − 1), and
offsets are their differences, v − v_n. The exact relative velocity,
(v − v_n)/(1 + v_n/c), differs from it by at most 0.5 per cent for |v_n| ≤ 1500 km/s.
Widths are as observed, not corrected for the instrumental resolution, a correction
below 0.3 per cent for FWHM ≥ 2000 km/s at SDSS and DESI resolution. The signal-to-noise
ratios use the per-pixel noise with the 2 per cent floor included, so for bright spectra
they reach a ceiling set by the floor rather than by the photon noise.

**Degeneracy check.** Every end point of the multi-start line fit with the selected
number of components and χ² within 4 max(1, χ²_ν) of the best is measured as well. If
their values of c(1/2) − v_n span more than 100 km/s, the line is flagged `degenerate`
and the span is reported as `dv_spread`: the data do not choose between decompositions
with different offsets.

## Classes

The rules are applied in this order; a profile takes the first class whose condition
holds. Class A uses the Monte Carlo error of Δv when one is available, and the 300 km/s
threshold alone otherwise, so a class can differ between point fits and fits with
errors.

| Class | Rule |
|---|---|
| **E** no broad line | FWHM < 1200 km/s, or integrated S/N < 5, or peak S/N < 1.5 |
| **X** no systemic reference | narrow-line reference not detected (S/N < 3) |
| **W** not classifiable | FWHM < 2000 km/s or integrated S/N < 8: something broad is there, but narrow-line residuals dominate such profiles |
| **B** double-peaked or disc-like | two resolved peaks separated by more than max(0.4 FWHM, 1500 km/s) with a dip deeper than 8 per cent and FWHM ≥ 3000 km/s; or FWHM ≥ 7000 km/s with \|A.I.\| ≥ 0.20 or K.I. ≥ 0.50 |
| **A** bulk shift | \|Δv\| > 300 km/s (at more than 3σ when an error is available) and a symmetric profile: \|c(1/4) − c(3/4)\| < 0.10 FWHM, \|A.I.\| < 0.12, \|v_peak − centroid\| < 0.20 FWHM |
| **C** asymmetric | single-peaked and not symmetric |
| **F** normal | symmetric, no significant offset |

B and C describe shapes. The code fits no disc model and does not decide what produces
either shape.

Two selections used by our DESI search are provided as functions:
*measurable* = class A, B, C or F, integrated S/N ≥ 8, FWHM ≥ 2000 km/s and no `edge` flag;
*strong offset* = measurable and 1000 ≤ |Δv| ≤ 4000 km/s. Hβ can inherit its narrow-line
reference from Hα, so the two offsets of one spectrum are not independent.

## Flags

Flags mark measurements to inspect; their absence does not certify a measurement. E, X
and W profiles are returned before most checks.

| Flag | Condition |
|---|---|
| `very_broad` | FWHM ≥ 8000 km/s |
| `poor_fit` | reduced χ² ≥ 2.5 |
| `low_snr` | integrated broad S/N < 10 |
| `low_peak_snr` | broad peak < 5σ per pixel: a component significant only by integration over thousands of km/s is degenerate with continuum residuals |
| `host_dominated` | host ≥ 80 per cent of the 4200–5000 Å light: template mismatch at the few per cent level imitates a very broad line |
| `pl_at_bound` | power-law slope within 0.1 of its bounds, −5 and 3 |
| `pl_unphysical` | power-law slope α < −2.5, steeper than an accretion disc |
| `peak_disagree` | model and data peaks differ by more than 0.25 FWHM |
| `sii_disagree` | [S II] velocity more than 150 km/s from v_n |
| `sys_disagree` | v_n and the [O III] core more than 400 km/s apart ([O III] S/N ≥ 5) |
| `narrow_at_bound` | \|v_n\| ≥ 1425 km/s, within 5 per cent of the ±1500 km/s window |
| `edge` | data cover less than ±6000 km/s around the line |
| `degenerate` | equally good decompositions differ in c(1/2) by more than 100 km/s |
| `param_at_bound` | a line velocity ended on its bound (`params_at_bound` lists every parameter on a bound) |
| `residual_outliers` | three or more pixels more than 5σ off the model, farther than 600 km/s from every narrow line |
| `extreme_offset` | \|Δv\| > 4000 km/s for classes A, C and F: a quality threshold, not a physical limit |
| `mc_multimodal`, `mc_basin_switch`, `mc_too_few`, `insufficient_offset_samples` | the Monte Carlo draws split between solutions, moved away from the fit, or were too few; the errors of the line are withheld |
| `ebv_assumed_zero` | no Galactic E(B−V) was available and 0 was used (command line only) |

## Errors

With `--nmc N` (at least 25; we use 200) the spectrum is perturbed N times with Gaussian
noise from its pixel errors, before the 2 per cent floor, and refitted (continuum and
lines), with the host model, the redshift, the extinction, the masks and the number of
broad components held fixed. The error of each quantity is half the 16th–84th percentile
range of the draws, as in Shen et al. (2013) and Liu et al. (2014). These are statistical
errors conditional on the model: they do not include the choice of host, continuum or
number of components.

The draws are tested for separate solutions. When they split into two groups (a
median-separation test on v_n and Δv), move as a whole away from the fit, or are too
few, the line is flagged and its errors are withheld (`null`); the percentiles stay in
`mc_info`. The evidence for these errors is on the [validation page](validation.md).

## Velocity changes between epochs

`blrfit pair` measures, for each broad line, how far the profile of one epoch has moved
relative to another epoch of the same object. The difference of two separately fitted
offsets would mix a bulk motion with a change of profile, with the decomposition each fit
chose and with each fit's own narrow-line subtraction. Instead, the **template** of
epoch A, its fitted broad Gaussians and nothing else, is evaluated at every trial shift s
from −4000 to +4000 km/s in steps of 10 km/s and compared with the **data** of epoch B,
the continuum-subtracted flux minus B's own narrow model; at each shift the template's
flux scale (held non-negative), an offset and a slope are solved linearly. The reach
exceeds the largest change the method was tested against (2500 km/s); the slope is free
because a linear baseline the fit missed biased the shift by up to 180 km/s without it;
the scale is held non-negative because an inverted template is not a model of the line.

- **Pixels.** The usable pixels of B within max(1.5 FWHM, 3000 km/s) of the model c(1/2)
  of either epoch (so that a large shift keeps both profiles inside), where A's fitted
  window covers every trial shift; at least 20 of them.
- **Errors.** The statistical inverse variance of B's spectrum, carried pixel by pixel
  through the same preprocessing as the fit (mask, de-reddening, rest frame) and never the
  fit weights, which include the 2 per cent calibration floor: that floor is a systematic
  of the absolute calibration and does not scatter one pixel against the next. The weights
  are rebuilt from the statistical variance and the floor and compared with the fit's own
  (`weights_consistent`).
- **Minimum.** A parabola through the contiguous grid points within Δχ² = 2 of the
  minimum (at least five, else the three-point formula) gives the shift and, from its
  curvature, the error, scaled by √χ²_ν where the reduced χ² exceeds one. A minimum within
  one step of the grid edge is `at_bound`; a second local minimum within Δχ² = 6.63 and more
  than 500 km/s away makes the pair `ambiguous`.
- **Both directions.** B's template over A's data gives s′. Under no change s ≈ −s′; the
  change of the pair is (s − s′)/2 in the common input-redshift frame (`s_common`), with the
  error hypot(σ, σ′): each direction's curvature sees the noise of its data epoch only, the
  template epoch's noise enters through its fitted profile, and the two directions are
  nearly anti-correlated, so the two curvature errors add in quadrature (with half that
  error the estimate covered the truth 38 per cent of the time at a nominal 68).
- **Shape.** On the same pixels and with the same linear terms, the χ² of A's template at
  its best shift minus the χ² of B's own broad model is the shape statistic; its value per
  pixel, the larger of the two directions, is `shape_max`. It does not depend on the shift
  itself, so a moved profile gives zero and a changed profile a positive value. A pair
  with `shape_max` ≤ 0.5 has a **stable** profile: on pairs with no expected change the
  95th and 99th percentiles are 0.38 and 0.72.
- **Frame.** A's narrow model (cores and wings) is slid across B's broad-subtracted data
  over ±800 km/s with the same code, in both directions: `dv_narrow` is the shift of the
  narrow lines between the epochs. It does not correct the broad shift (on SDSS pairs the
  corrected shift was no tighter than the uncorrected one); a narrow shift beyond 200 km/s
  vetoes the pair (`frame_offset_large`), as do the fits' own systemic velocities when they
  differ by more than 200 km/s in the tiers.
- **Systematic term.** On real pairs with no expected change the statistical error falls
  short by a term that is added in quadrature: `err_total` = hypot(`err`, 45 km/s) for Hα
  and hypot(`err`, 36 km/s) for Hβ, measured on consecutive DESI nights of the same
  targets ([validation](validation.md#velocity-changes)). Mg II has no term and is flagged
  `uncalibrated_line`.

A pair is **retained** unless a direction is at bound, has a flat minimum, too few pixels
or a flux scale outside 0.25–4; the product of the two scales is outside 0.5–2 (the two
directions must fit reciprocal factors); or the two directions disagree, |s + s′| above
three times the pair error and above two grid steps (on pairs with no change the 99th
percentile of |s + s′|/err is 2.6–3.0). A pair that fails keeps its direction values in
the table with `retained` false and the shift withheld. Every flag is listed with its
meaning on the [outputs page](outputs.md#pair-table).

The pairs of an object are the reference epoch (the highest broad-line S/N) against every
other epoch, and consecutive epochs against each other. Positive `s_common` means the
later epoch is redder.

## Candidate tiers

`blrfit tiers` gives one verdict per object from its pairs, with the pair, the line and
the numbers behind it as the reason. Only pairs retained, with both epochs at an
integrated broad-line S/N of at least 8 (below it a template matches noise), are counted,
each with its total error. The tiers are tried in this order:

| Tier | Rule |
|---|---|
| **disk** | a double-peaked (class B) broad profile in the reference spectrum: a disc emitter, whose peaks move with the disc; the largest change is reported |
| **binary** | a pair whose change is at least 4σ, with a stable profile in both directions, no frame flag, the other Balmer line agreeing within 2σ of the combined error where it is measured, no class B epoch in the pair, no reversal of sign at 4σ twice over the epochs, and a change that an orbit at the virial mass allows |
| **platinum** | the binary rule with the other line measured and agreeing (the two-line criterion of Guo et al. 2019), `shape_max` ≤ 0.3, the two directions within 2σ of each other and a broad peak S/N ≥ 8 in both epochs: a changed profile can slip under the 0.5 shape screen, and very broad low-contrast lines give direction mismatches of 2–3σ without reaching 3 |
| **almost** | a change of at least 4σ that fails exactly one of the shape, two-line and frame conditions; a marginal change (3–4σ) that passes them all; or a change that no orbit allows or that reverses sign |
| **profile** | the strongest change comes with a changed profile and no clean significant pair exists: variability of the line, not a bulk motion |
| **stable** | retained pairs, none above 3σ: an upper limit on the change |
| **none** | no retained pair with both epochs at the S/N floor |

The 4σ threshold comes from the noise alone: of 330 SDSS pairs with no expected change,
one exceeded 3σ (0.9 expected) and none 4σ.

**Orbital bound.** For a circular binary in which the active black hole of virial mass
M carries the broad-line region and the companion has mass M/q, the closest orbit that
keeps the broad-line region inside the active hole's Roche lobe (Paczyński 1971) has the
separation a_min = R_BLR / f(q) and, seen edge-on, the active hole's line-of-sight
amplitude v_max = √(G M_tot / a_min) / (1 + q) at the period P_min; a wider orbit is
slower, v(P) = v_max (P / P_min)^(−1/3), so the largest change of the line-of-sight
velocity over a time Δt is 2 v(P) |sin(π Δt / P)| at the most favourable P ≥ P_min. The
mass is the Hα virial mass of Greene & Ho (2005) where broad Hα is measurable, else the
Hβ mass of Vestergaard & Peterson (2006) with λL_λ(5100) of the power law, and R_BLR
follows Bentz et al. (2013). With q = 0.1 (a heavier companion makes the active hole move
fastest) a change that exceeds the bound at 2σ cannot be the active hole's orbital motion;
otherwise the longest period that still allows it is reported for q = 0.1 and q = 1. The
bound is a label on the candidate, not a model of the object.

## References

Bentz M. C. et al. 2013, ApJ, 767, 149 ·

Boroson T. A., Green R. F. 1992, ApJS, 80, 109 ·
Cardelli J. A., Clayton G. C., Mathis J. S. 1989, ApJ, 345, 245 ·
Eracleous M., Boroson T. A., Halpern J. P., Liu J. 2012, ApJS, 201, 23 ·
Greene J. E., Ho L. C. 2005, ApJ, 630, 122 ·
Guo H., Shen Y., Wang S. 2018, PyQSOFit, ascl:1809.008 ·
Heckman T. M., Miley G. K., van Breugel W. J. M., Butcher H. R. 1981, ApJ, 247, 403 ·
Liu X., Shen Y., Bian F., Loeb A., Tremaine S. 2014, ApJ, 789, 140 ·
Marziani P., Sulentic J. W., Dultzin-Hacyan D., Calvani M., Moles M. 1996, ApJS, 104, 37 ·
O'Donnell J. E. 1994, ApJ, 422, 158 ·
Paczyński B. 1971, ARA&A, 9, 183 ·
Planck Collaboration 2020, A&A, 641, A6 ·
Salviander S., Shields G. A., Gebhardt K., Bonning E. W. 2007, ApJ, 662, 131 ·
Schlegel D. J., Finkbeiner D. P., Davis M. 1998, ApJ, 500, 525 ·
Shen Y. et al. 2011, ApJS, 194, 45 ·
Shen Y., Liu X., Loeb A., Tremaine S. 2013, ApJ, 775, 49 ·
Tsuzuki Y., Kawara K., Yoshii Y., Oyabu S., Tanabé T., Matsuoka Y. 2006, ApJ, 650, 57 ·
Vestergaard M., Peterson B. M. 2006, ApJ, 641, 689 ·
Vestergaard M., Wilkes B. J. 2001, ApJS, 134, 1 ·
Whittle M. 1985, MNRAS, 216, 817 ·
Yip C. W. et al. 2004, AJ, 128, 585
