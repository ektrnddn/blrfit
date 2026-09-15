# Changelog

## 0.2.0.dev0 (unreleased)

Corrections of the defects found in the September 2026 audit of the frozen
fitter. Every entry states its effect on results; fixes and policy changes are
listed separately so that the change of any catalogue quantity between 0.1.0
and 0.2.0 can be attributed. The tag `v0.1.0` reproduces the catalogue run.
`docs/DELTAS.md` and `docs/deltas_*.csv` tabulate the per-object effect.

### Fixes (single-epoch fitter)

* Fe II broadening evaluates the requested width directly instead of rounding
  it to a 50 km/s cache key. With the rounded operator the finite-difference
  derivative of the width was zero, and the trust-region solver moved the width
  in roundoff-seeded steps of thousands of km/s: the 0.1.0 width column carries
  no information, and Fe-strong fits were not reproducible across platforms
  (a flux rescaling by 1 + 1e-13 moved the Hbeta offset of the pinned spectrum
  spec-1592 by 346 km/s and changed its class).
  Effect: offsets move by at most 23 km/s on the five pinned spectra, no class
  changes; widths become reproducible to 1 km/s. On the 823 SDSS spectra of the
  Liu et al. (2014) and Eracleous et al. (2012) anchor the Hbeta c50_sys changes
  with an NMAD of 6.4 km/s, 71 Hbeta classes change (8.6 per cent) and 8 of 476
  Halpha classes (1.7 per cent); refitted with the historical operator only 2.1
  per cent of the Hbeta classes change, so most Hbeta class changes come from
  this correction, in objects near a class boundary where the Fe II blends
  under Hbeta move the decomposition. The selection line, Halpha, is barely
  affected.
* Hbeta's systemic label and S/N reflect the constraint that entered the fit;
  with `use_ha_systemic=False` Hbeta no longer inherits Halpha's S/N and label
  (0.1.0 could pass the no-systemic test through that label). Production always
  used the tie, so no catalogue number changes.
* Monte Carlo draws run the same line sequence as the observed spectrum ([O III]
  pre-fit, [O III]-informed narrow start and prior, Halpha-to-Hbeta transfer);
  0.1.0 omitted them in the draws, so a narrow group displaced by more than about
  600 km/s from the input redshift aliased by 943 km/s (Halpha taken for
  [N II] 6584) in the draws. Effect: identical errors where the narrow group sits
  near the input redshift (the bulk); corrected errors for displaced groups.
* Remeasurement (`remeasure`) keeps the luminosity distance, so broad-line
  luminosities survive it; Hbeta's systemic label is rebuilt from the stored
  constraint.
* Non-finite solver attempts can no longer be selected (0.1.0 compared NaN
  objectives; 0 of 820 attempts on the anchor set were non-finite).
* `fit_spectrum` refuses a flux whose median lies outside 1e-2 to 1e4 in the
  expected unit (1e-17 erg/s/cm^2/A) instead of fitting it against amplitude
  bounds set for that unit and returning wrong classes silently; the keyword
  `flux_scale` rescales other units. Identical for every SDSS and DESI spectrum.
* Gzipped SDSS spec files and DESI coadds are recognised by name; the redrock
  sibling of a coadd is found whether either file is compressed.

* A spectrum without a single usable pixel (every pixel with a non-finite flux
  or inverse variance, or an inverse variance <= 0) is refused with a
  ValueError that says so. It used to reach the continuum fit, whose reference
  flux was then the median of an empty set: the power-law bound became NaN and
  the solver refused the start ("Initial guess is outside of provided bounds").
  This happened to 6 of the 1,630 spectra of the September 2026 validation
  subset (three of them DESI tile coadds of zero exposures), none of which has
  a fit in the 0.1.0 catalogue run either.
  Effect: an explicit error instead of a solver error; no spectrum that fitted
  before is affected.
* A continuum fit left with no more pixels outside the line complexes than it
  has free parameters (a spectrum covering little more than one complex)
  raises ValueError. Least squares on no residual returned the starting values
  as a successful continuum: for a synthetic spectrum trimmed to 6410-6790 A
  the continuum came out 2.5 times too low and the broad equivalent width 5.6
  times too high, with no flag.
  Effect: none on spectra with continuum pixels.
* The host fraction is NaN, so the host is not subtracted, when the flux summed
  over 4200-5000 A is not positive. The denominator was clamped to 1e-30, so a
  positive host gave fractions near 1e33 that passed the 0.1 threshold.
  Effect: none when that flux sum is positive.
* Input checks: z must be finite and above -1 and sig_broad_min finite and
  non-negative (z = NaN used to end in "no usable pixel"). Observed wavelengths
  given in descending order are sorted before the fit. A spectrum in which more
  than half of the wavelength steps are zero (every pixel repeated, as when two
  exposures are concatenated) is refused: its median pixel spacing was zero and
  the line measurements divided by it (ZeroDivisionError). Repeats in camera
  overlaps alone are fitted as before.
  Effect: none on increasing wavelength arrays; a descending array now gives
  the fit of the same spectrum in increasing order.
* The starting vector handed to the solver is always inside valid bounds. A
  start is clipped to 1e-9 inside its bounds as before; a NaN start, or an
  infinite one on the side of an infinite bound, is replaced by the middle of
  the bounds (the finite bound when only one is finite) with a RuntimeWarning;
  bounds of a free parameter that are not an increasing pair of numbers raise
  ValueError naming it, which is how the first case above was traced.
  Effect: none on spectra that fitted before (their starting vector is
  unchanged bit for bit).

### Fixes (cross-correlation; on-sky calibration still to be redone)

* The default comparison window of `ccf_shift` is centred on the template's
  c(1/2) relative to the input redshift when its systemic velocity is not
  measured (Mg II fitted without a narrow component); it was centred on
  0 km/s, which cut an offset profile. Synthetic Mg II at +4500 km/s with a true
  change of 200 km/s: 292 +/- 96 km/s before, 200 +/- 5 km/s now. NaN values of
  `nsub_frac`, `mismatch`, `win_fwhm`, `win_min` and `clip` are refused.
  Effect: none when the systemic velocity is measured.

* The variance of the chi-square scan includes the fitted flux scale of the
  template (var_y + a^2 var_x, the profile likelihood of two noisy spectra);
  0.1.0 used var_y + var_x, so a flux ratio of 2 between epochs alone produced a
  profile-change grade and a pair error 1.6 times too large. The scan, the
  clipping pass and the residual statistic use the same variance.
* Placement on the template grid matches coordinates exactly and keeps native
  gaps and masks; 0.1.0 copied arrays by index whenever the median spacings
  agreed to 5 per cent, so a masked pixel dropped from one epoch shifted every
  pixel beyond it by one (30 km/s at Halpha), invisibly to the reliability cuts.
  Interpolation requires two valid bracketing pixels and propagates the
  interpolation weights into the variance.
* Two spectra on one pixel lattice (two SDSS spectra on the logarithmic grid,
  or two DESI spectra) are compared by integer placement with no resampling;
  only spectra on different lattices are interpolated, onto a uniform velocity
  grid. An intermediate correction resampled every SDSS template, which shrank
  the errors of same-lattice pairs by 14-22 per cent through the ignored
  interpolation covariance. On the bundled SDSS pair of J001224 (2001, 2013)
  Hbeta gives -293 +/- 29 km/s with z_prof 0.8 (stable); 0.1.0 graded the same
  pair 'changed' (z_prof 10.5), an artefact of the flux ratio of 2.1 between
  the epochs.
* The statistical error of a shift is graded and labelled (`err_method`): the
  Delta chi-square = 6.63 interval when it brackets the minimum on both sides
  ('delta_chi2'), the bracketed half when only one side is
  ('delta_chi2_one_sided'), the local parabola when neither is ('curvature'),
  otherwise 'unavailable'; a pair whose two directions differ is 'mixed'. On the
  low-S/N synthetic ensemble (FWHM 5000 km/s, peak S/N 8, 160 pairs) 30 errors
  are unavailable, all reported; 0.1.0 reported 25 and dropped 12 more through
  its argument-order rule.
* Non-finite errors cannot pass the validity checks, and the bidirectional
  combination treats a failure in either direction the same way (0.1.0 dropped
  or kept a NaN depending on argument order). `err_method` and the bootstrap
  counts report what was actually used. The two-line diagnostic uses the
  displayed corrected quantities. None of this is calibrated yet: the 0.1.0
  floors are carried as reference values, and at low S/N the errors under-cover
  (pull NMAD 4-7 at FWHM 5000 km/s and peak S/N 8, 1.3-3.5 at peak S/N 12).
* DESI fetches store release, reduction and content-addressed bundles with the
  matching redrock file and exposure provenance, published atomically; 0.1.0
  wrote DR1 and EDR extractions of the same exposures to one file name.

### Policy changes

* Narrow-line frame veto of an epoch pair (policy change of the cross-
  correlation). Every pair gets one zero point from [O III] 5007, or [S II]
  where [O III] is not measurable, and a frame check (`frame_ok`,
  `frame_reason`): a zero point beyond `FRAME_VETO_KMS` = 200 km/s (set on the
  zero points of 1,099 epoch pairs; docs/CCF_VALIDATION.md) or at the search
  bound vetoes the pair
  for both lines, and `is_reliable` requires a good frame. The zero point is
  measured in both directions (swapping the spectra only changes its sign; one
  direction alone differed between the two orders of an SDSS-DESI pair by up to
  720 km/s on the DR1 bench), and a zero point whose directions disagree beyond
  twice their combined error vetoes the frame. Hbeta is corrected by
  the zero point when the frame is good, Halpha never, as the 0.1.0
  calibration decided. This replaces two rules: the 0.1.0 command line applied
  the zero point when |zp| < max(30, 2 err), and the production scripts applied
  it below 500 km/s and ignored it above. A frame offset enters both Balmer lines
  identically and so mimics the coincident two-line change the search rewards;
  one two-line object of the first catalogue run carried a +613 km/s zero point
  on every SDSS epoch.
* Unconverged line solutions are flagged and kept, as in 0.1.0: every attempt
  with a finite objective competes for the lowest chi-square; the selected
  attempt's convergence is recorded (`fit['converged']`, status
  `success_unconverged`, column `{HA,HB,MG}_converged`).
* A continuum solver that stopped short of convergence no longer aborts the
  line fits (as in 0.1.0); the state is recorded as `continuum_status` and per
  line as `continuum_converged`. Host attempts that stopped short are accepted
  as in 0.1.0 and recorded in `host_info['solver_attempts']`. Monte Carlo draws
  with an unconverged continuum are kept and counted (`unconverged_continuum_draws`).
* The ultraviolet Fe II width is fixed at 3000 km/s (`FE_UV_FWHM_FIXED_KMS`)
  when fewer than 300 good continuum-window pixels lie in 2200-3090 A
  (`FE_UV_FREE_MIN_PIXELS`; DESI below z = 0.355, SDSS below z = 0.445): there
  the width is unconstrained and the continuous operator drives it to a bound.
  Effect: two of the five pinned spectra, Hbeta offsets by 0.1 and 0.2 km/s.

* Search and plausibility of a cross-correlation match (policy change). The
  curve statistic chi-square - N_pix favours the shifts that use the fewest
  pixels whenever the two profiles differ, so a search whose pixel set changes
  with the shift drifts toward the edge of its range. The search now runs in
  two stages, each on the window pixels with data in both spectra at every one
  of its shifts: stage 1 locates the minimum over +/- vmax (a reduced range when
  fewer than half the window's pixels would remain; `search_range`,
  `common_frac`), stage 2 measures the shift, its error and z_prof within
  600 km/s of it. Single-pixel artefacts are masked in both spectra before the
  search (more than 5 errors from the median of their five neighbours,
  `n_spikes`), so that one bad pixel cannot choose the shift at which the
  outliers are identified. Two checks are reported and required by
  `is_reliable`: `scale_ok` (both fitted flux factors within 1/4 to 4, their
  product within 1/2 to 2) and `ambiguous` (a second minimum of the stage-1
  curve within Delta chi-square 6.63 of the first, at least 500 km/s or three
  errors away: the alias of a double-peaked profile). The narrow-line zero
  point keeps the single-stage search and is not despiked. A floor on the
  overlap fraction was tried and rejected: the windows of broad lines reach the
  edges of the fitted spectrum, so any floor caps the searchable shift (at
  about 900 km/s for a floor of 0.95). On the DR1 bench (29 objects, 161
  measurable pair records) the searched range reaches a median of 3000 km/s,
  no reliable pair has a shift above 1500 km/s, and the reliable pairs of DESI
  epochs scatter by 58 km/s in Halpha (17 pairs, largest 355 km/s) and 32 km/s
  in Hbeta (7 pairs). The 0.1.0 search, with a 5000 km/s range, gave shifts of
  up to 5019 km/s between epochs weeks apart on the same fits
  (`docs/CCF_VALIDATION.md`).

### Diagnostics and tooling (no change of results)

* Solver records per attempt (success, status, nfev, optimality, objective,
  active bounds), `fit_status` per line (`success`, `success_unconverged`,
  `unusable_window`, `uncovered_core`, `solver_failed`), the data and penalty
  parts of the chi-square stored separately from the penalised selection score
  (the `bic` key keeps the 0.1.0 meaning), and `bic_margin`, the distance of
  the chosen component count to the next in BIC (column `{p}_bic_margin`).
* Per-pair records of the search and plausibility checks: `common_frac`,
  `search_reach`, `search_range` and `npix_search` (per direction), `n_spikes`,
  `scale_ok`, `ambiguous`, `dv_alt_ab`, `dv_alt_ba`,
  `dG_alt_ab`, `dG_alt_ba`, and the failed check named in the reliability reason
  of `blrfit rv`.
* Per-direction records of every epoch pair: fitted flux scales, reduced
  chi-square, profile statistic, errors and error methods, pixel counts, and
  the masked native pixels inside the window (`scale_ab`, `scale_ba`,
  `chi2_red_ab`, ..., `n_masked_a`, `n_masked_b`), in `pair_analysis` and in the
  JSON of `blrfit rv`.
* `blrfit.io.fetch.list_desi_epochs` and `fetch_desi_epochs`: the tile-
  cumulative coadds of one target in a public DESI release, by range requests,
  one bundle per tile with its redrock file and exposure provenance;
  `tools/bench_dr1.py` builds a local set of objects with several DESI epochs
  and their SDSS spectra.
* Continuum parameters at a bound (`continuum_info['at_bound']`, column
  `conti_at_bound`), the fitted Fe widths, and `feuv_fwhm_fixed`.
* Monte Carlo bookkeeping: contributing, failed and unconverged draws, the
  systemic reference used per draw, aligned draw samples of v_sys, c50_sys and
  v_peak_sys, and the flag `mc_multimodal` when the sorted draws of v_sys or
  c50_sys split into two groups more than 400 km/s apart (`MC_ALIAS_KMS`) with
  more than a tenth of the draws on each side (`MC_ALIAS_MAX_FRACTION`); the
  percentile error of a flagged line is not a valid statistical error.
* `blrfit.continuum_luminosity` and `blrfit.physics.lambda_l_lambda`: lambda
  L_lambda from the fitted power law with the documented convention (the fit
  works on (1+z)-scaled rest-frame flux; no further redshift factor).
* Fit settings record the systemic mode, complexes, extinction, Monte Carlo
  seed and count, Mg II options, classification overrides and `flux_scale`.
* Pins of 0.2.0 (`tests/data/pins_0.2.0.json`, `tools/make_pins.py`), the
  delta tables under `docs/`, a reproducibility test (flux rescaled by
  1 + 1e-13 leaves every velocity within 0.1 km/s), the anchor test now
  reporting per-spectrum deltas against the stored 0.1.0 run, and synthetic
  cells of the cross-correlation for a flux ratio at unequal S/N and for masked
  or missing pixels (`tests/test_rv_cells.py`).

### Known open items

* The on-sky floors of the cross-correlation (155/79 km/s DESI-DESI, 143/147
  km/s cross-survey) and the profile-grade inflation factors were calibrated
  with the 0.1.0 estimator and must be re-measured, as must the range of the
  flux-factor check (the frame-veto threshold is set on the zero points of the
  validation subset: docs/CCF_VALIDATION.md, Frame veto). Low-S/N errors under-cover
  (see above). The ambiguity check flags a third of the usable bench pairs;
  whether that is too strict must be judged on the production pairs.
* Host-decomposed fits are reproducible to about 1.3 km/s, not 0.1 km/s, under
  a last-bit flux rescaling: the joint host solver stops at slightly different
  points. `tests/test_reproducibility.py` holds 0.1 km/s on the numerical stack
  of the catalogue run (numpy 1.26, scipy 1.13) and 1.5 km/s with current
  releases, where the J001224 2001 epoch moves by up to 0.32 km/s.
* Frozen calibration constants in `constants.py` are unchanged.

## 0.1.0 (2026-09-13)

First public release.

* Single-epoch model: power law + Fe II + host continuum, narrow-line system with soft [S II]
  tie and narrow-line-region wing, [O III] core + wing, 1-3 broad Gaussians chosen by the BIC,
  width-offset hinge, coverage rule, [O III]-informed systemic velocity. Checked against the
  fitter that produced the DESI offset-line catalogue on 824 SDSS spectra: identical results on
  the same numerical stack (numpy 1.26, scipy 1.13); four spectra pinned in the tests.
* Profile measures (bisectors, widths, asymmetry and kurtosis indices, peaks), classes
  A/B/C/F/E/X/W, twelve quality flags, Monte Carlo errors and the DESI repeat-spectrum error model.
* Cross-correlation of the broad profile between epochs (whole-pixel chi-square, profiled scale
  and baseline, narrow-line down-weighting, fixed outlier mask, bidirectional consistency,
  narrow-line zero-point, profile-stability statistic) with the DESI on-sky floors; profile grades
  (stable, mild, changed) with graded cross-survey error floors, the residual effect size and the
  two-line consistency criterion.
* Readers for SDSS spec files, DESI healpix and tile coadds (astropy only; desispec optional and
  identical), and generic tables with declared columns, units and frame.
* `blrfit fit`, `blrfit rv` and `blrfit fetch` (SDSS through astroquery, DESI public releases
  through range requests to the file server; single-target coadd extraction).
* Tests: regression pins in two parts (the continuum and line models, chi-square, measures and
  classes recomputed from the pinned parameters and checked on every platform; the end point of
  a fresh fit held to the classes, flags, component counts, systemic sources, the primary offset
  within 100 km/s and chi-square within 10 per cent, because the solver's end point depends on
  the platform for degenerate decompositions; bit for bit on the reference stack), the penalty
  terms and the extinction curve pinned, reader and healpix tests, command-line tests, synthetic
  validation suites for the fitter, the cross-correlation and the Monte Carlo errors, and the
  Liu et al. (2014) same-spectrum anchor.
