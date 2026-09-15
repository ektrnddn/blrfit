# Validation of the cross-correlation, blrfit 0.2.0

This file records what the corrected cross-correlation (`blrfit.rv`) has been measured to do,
on synthetic pairs and on real DESI and SDSS spectra, and what it has not yet been calibrated
for. All numbers were measured on 2026-09-14 on macOS arm64 with numpy 1.26.4 and scipy 1.13.1.
Every number below was measured with the final code of this version (two-stage search, despike
and plausibility checks).

## Synthetic pairs: the cross-correlation alone

Gaussian broad profiles, 40 independent pairs per shift (−600, −150, +300, +900 km/s), noise
added to both epochs (`tests/test_rv_synthetic.py::test_ccf_bias_at_peak_snr_8_to_12`). Per FWHM
and peak S/N: the range of the per-shift medians of the deviation from truth, the range of the
per-shift NMADs, the median statistical error, the range of the pull NMADs, and the number of
pairs (of 160) whose error is unavailable.

| FWHM (km/s) | peak S/N | median bias (km/s) | NMAD (km/s) | median error (km/s) | pull NMAD | errors unavailable |
|---|---|---|---|---|---|---|
| 2500 | 8 | +2 to +14 | 64–83 | 29–32 | 1.7–2.7 | 6 |
| 4000 | 8 | 0 to +28 | 70–103 | 23–28 | 2.0–4.0 | 21 |
| 5000 | 8 | −13 to +26 | 106–142 | 22–26 | 4.0–6.9 | 30 |
| 2500 | 12 | 0 to +13 | 33–48 | 24–26 | 1.3–2.0 | 0 |
| 4000 | 12 | +6 to +18 | 61–73 | 24–26 | 1.7–2.7 | 2 |
| 5000 | 12 | −15 to +30 | 66–81 | 22–25 | 2.5–3.5 | 9 |

The shifts are unbiased within the sampling error of these ensembles. The statistical errors
undercover: by factors of 2 to 7 at peak S/N 8 and 1.3 to 3.5 at peak S/N 12, growing with the
width of the line. Every unavailable error is reported with `err_method = 'unavailable'`; 0.1.0
reported 25 of the 30 at FWHM 5000 km/s and peak S/N 8 and dropped the others through the
argument order of its two-direction combination.

## Synthetic pairs: the whole pipeline

Fit, narrow-line subtraction and cross-correlation of four shifts × 8 pairs
(`test_pipeline_grid`, slow). Pooled Halpha medians of the deviation from truth and pull NMADs:

| FWHM (km/s) | peak S/N 25 median (km/s) | pull NMAD | peak S/N 10 median (km/s) | pull NMAD |
|---|---|---|---|---|
| 2500 | −17.5 | 0.96 | −51.9 | 1.9 |
| 4000 | −11.2 | 1.36 | −32.4 | 2.2 |
| 5000 | +0.6 | 1.60 | −15.2 | 2.95 |

The negative bias at FWHM 2500 km/s comes from the narrow-line-region wing of the fitted narrow
model taking flux from the blue flank of a narrow broad line (the docstring of the test gives the
experiment); it is a known limitation at low S/N, and that case is a strict expected failure.

## Synthetic pairs: the cells 0.1.0 got wrong

`tests/test_rv_cells.py`. A flux ratio of 0.5 or 2 between the epochs at peak S/N 25 against 8
(both ways), identical shapes shifted by +300 km/s, six realisations per cell: z_prof stays below
5 in both directions with a median between −3 and 2 (0.1.0: 5 to 23), the fitted flux factor is
the ratio to 5 per cent and the shift is unbiased to 30 km/s. One, three or ten consecutive pixels
of one epoch masked or dropped, in the blue wing, the core or the red wing of two noiseless copies
of one profile: the shift stays within 1 km/s without resampling (0.1.0: one 30 km/s pixel per
dropped pixel blueward of the window).

## Real spectra: the DR1 bench

29 objects of the catalogue with at least two DESI tile epochs in the public DR1 release (before
June 2022) and 27 of them with SDSS spectra, built by `tools/bench_dr1.py` (103 spectra; one DESI
tile epoch does not fit: the solver reports an infeasible start). Every DESI–DESI and SDSS–DESI
pair was fitted with 0.2.0 and cross-correlated with a 5000 km/s search, the range of the
long-baseline run of the catalogue. Measurable records only: classes A, B, C or F in both epochs
and a broad-line flux S/N of at least 8. The 0.1.0 rows run the frozen 0.1.0 cross-correlation on
the same fits. Pairs of DESI epochs are weeks to months apart (median 25 days), so their true
shift is close to zero and their scatter is the null of the method.

| pairs | line | estimator | selection | records | objects | NMAD (km/s) | median error | pull NMAD | shifts > 1500 km/s | largest (km/s) |
|---|---|---|---|---|---|---|---|---|---|---|
| DESI–DESI | Halpha | 0.1.0 | every finite shift | 40 | 28 | 442 | 33 | 8.94 | 9 | 5019 |
| DESI–DESI | Halpha | 0.1.0 | not at bound, z_prof < 5 | 31 | 24 | 398 | 36 | 8.94 | 6 | 2587 |
| DESI–DESI | Halpha | 0.2.0 | every finite shift | 40 | 28 | 277 | 18 | 6.61 | 8 | 4985 |
| DESI–DESI | Halpha | 0.2.0 | reliable | 17 | 15 | 58 | 17 | 3.40 | 0 | 355 |
| DESI–DESI | Hbeta | 0.1.0 | every finite shift | 16 | 14 | 384 | 84 | 4.28 | 4 | 2445 |
| DESI–DESI | Hbeta | 0.1.0 | not at bound, z_prof < 5 | 14 | 13 | 582 | 84 | 4.28 | 4 | 2445 |
| DESI–DESI | Hbeta | 0.2.0 | every finite shift | 16 | 14 | 198 | 24 | 5.80 | 1 | 3719 |
| DESI–DESI | Hbeta | 0.2.0 | reliable | 7 | 7 | 32 | 22 | 1.80 | 0 | 247 |
| SDSS–DESI | Halpha | 0.1.0 | every finite shift | 72 | 27 | 553 | 102 | 4.06 | 22 | 5064 |
| SDSS–DESI | Halpha | 0.1.0 | not at bound, z_prof < 5 | 46 | 20 | 683 | 125 | 4.36 | 13 | 2480 |
| SDSS–DESI | Halpha | 0.2.0 | every finite shift | 72 | 27 | 587 | 51 | 6.83 | 24 | 5041 |
| SDSS–DESI | Halpha | 0.2.0 | reliable | 13 | 9 | 181 | 44 | 5.68 | 0 | 375 |
| SDSS–DESI | Hbeta | 0.1.0 | every finite shift | 33 | 12 | 393 | 120 | 3.79 | 6 | 5020 |
| SDSS–DESI | Hbeta | 0.1.0 | not at bound, z_prof < 5 | 22 | 10 | 277 | 120 | 3.79 | 4 | 2748 |
| SDSS–DESI | Hbeta | 0.2.0 | every finite shift | 33 | 12 | 247 | 57 | 6.13 | 3 | 4036 |
| SDSS–DESI | Hbeta | 0.2.0 | reliable | 14 | 8 | 164 | 49 | 3.20 | 0 | 1427 |

"Reliable" is `rv.is_reliable`: not at bound, finite error, z_prof < 5, direction mismatch below
the 0.1.0 cut, a good narrow-line frame, plausible flux factors and no second minimum of similar
depth. The SDSS–DESI pairs span 4 to 22 years of objects selected for large offsets, so their
scatter includes real changes.

What the table says. With a 5000 km/s search the 0.1.0 cross-correlation places a sizeable
fraction of pairs thousands of km/s from the truth, including pairs of DESI epochs a few weeks
apart, and its own reliable tier keeps most of them (6 of 31 DESI–DESI and 13 of 46 SDSS–DESI
Halpha records above 1500 km/s). The corrected estimator still finds such minima when it is asked
to search 5000 km/s (8 of 40 DESI–DESI Halpha records among all finite shifts), but it recognises
them: none survives its reliability conditions, whose DESI–DESI null is 58 km/s in Halpha and
32 km/s in Hbeta. The price is a smaller reliable sample (17 of 40 DESI–DESI Halpha records,
against 31 on the 0.1.0 tier), mostly through the ambiguity check. Long-baseline products built
with the 0.1.0 estimator and search range need to be rebuilt before any mover is interpreted. The
corrected statistical errors are still about three times too small for pairs of DESI epochs in
Halpha; an on-sky error floor, measured on the production epoch pairs with the corrected
estimator, remains necessary.

## The search and the plausibility checks

Before these were added, 33 of the 161 measurable records had a shift above 1500 km/s, 19 of them
not at bound, with a good frame and a finite error. They came in two kinds: a flux factor of the
template far from one (the scan matched a sliver of the profile: 16 of the 17 records with a
factor outside 1/3 to 3 had such a shift) and a minimum near the edge of the search, where the
overlap of the two spectra is smallest.

* Two-stage search on common pixel sets. The curve statistic is G = χ² − N_pix; where the two
  profiles differ (reduced χ² above one) every pixel removed lowers G, so a search whose pixel set
  changes with the shift drifts toward the shifts with the smallest overlap. Every scan now uses
  the window pixels with data in both spectra at each of its shifts, so the pixel count is the
  same along the curve (`npix_search`). Stage 1 locates the minimum over ±vmax, with a reduced
  range when fewer than half the window's pixels would remain (`CCF_COMMON_MIN_FRAC`); stage 2
  measures the shift, its error and z_prof within 600 km/s of it (`CCF_REFINE_HALF_KMS`), on the
  common set of that small range, which is nearly the whole window. On the bench the searched range
  reaches a median of 2991 km/s in Halpha (16–84 per cent: 1636–4527; 14 per cent of the pairs
  below 1500 km/s) and 2780 km/s in Hbeta.
* Why not a floor on the overlap. A floor on the overlap fraction, relative to its largest value
  along the search, was tried first. The comparison windows of these broad lines (±1.5 FWHM) reach
  the edges of the fitted spectrum (median width 23,400 km/s in Halpha and 27,900 km/s in Hbeta),
  so every shift loses overlap and a floor becomes a cap on the searchable shift: a median reach
  of 3399, 1794 and 887 km/s in Halpha for floors of 0.8, 0.9 and 0.95, and 85 per cent of the
  Halpha pairs unable to search beyond 1000 km/s at 0.95. The clean appearance of the bench at a
  floor of 0.95 came partly from that cap.
* Despike. A single 250-sigma pixel in an otherwise clean synthetic pair put the first version of
  the two-stage search at its +2010 km/s bound, with the true line core clipped by the outlier mask
  computed at that wrong shift. Pixels more than 5 errors from the median of their five neighbours
  are now masked in both spectra before the search (`n_spikes`; one false flag in 80,200 noise
  pixels around a broad and a narrow line). The narrow-line zero point, whose profile is itself
  sharp, is not despiked.
* `scale_ok`: both directions' flux factors within 1/4 to 4 and their product within 1/2 to 2.
* `ambiguous`: a second local minimum of the stage-1 curve within Δχ² = 6.63 of the first and at
  least 500 km/s or three statistical errors away, the alias of a double-peaked profile matched
  peak onto peak. It flags 41 of the 122 usable records (not at bound, good frame, finite error).

With all of this, 19 of the 122 usable records have a shift above 1500 km/s and none is reliable:
14 fail the flux-factor or ambiguity check, the other 5 the profile or direction conditions.

A synthetic 3900 km/s translation (the size of the largest change in the catalogue) of a FWHM
4300 km/s profile is recovered to 30 km/s by the two-stage search when the data extend beyond the
window (`test_large_true_shift_is_recovered_by_the_two_stage_search`). Whether the real spectra of
that object leave enough data for such a range has to be checked on its production epochs.

## Not yet calibrated

* The on-sky error floors: the 0.1.0 floors (155 and 79 km/s for pairs of DESI epochs, 143 and
  147 km/s across surveys) were measured with the 0.1.0 estimator and are reported as reference
  values only.
* The frame-veto threshold (200 km/s), the range of the flux-factor check, the minimum common
  fraction (0.5) and the refinement range (600 km/s), all set on the evidence above and to be
  re-measured on the production epoch pairs.
* The ambiguity check flags a third of the usable bench pairs; whether it is too strict must be
  judged on the production pairs, where the reliable sample has to stay large enough for the
  population statistics.
* The profile-grade inflation factors, measured with the 0.1.0 estimator.
* The bench is 29 objects selected for strong offsets; it establishes the failure modes and the
  size of the improvement, not the calibration.
