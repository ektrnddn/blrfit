# Changelog

## 0.1.0 (2026-09-11)

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
  narrow-line zero-point, profile-stability statistic) with the DESI on-sky floors.
* Readers for SDSS spec files, DESI healpix and tile coadds (astropy only; desispec optional and
  identical), and generic tables with declared columns, units and frame.
* `blrfit fit`, `blrfit rv` and `blrfit fetch` (SDSS through astroquery, DESI public releases
  through range requests to the file server; single-target coadd extraction).
* Tests: regression pins, reader and healpix tests, command-line tests, synthetic validation
  suites for the fitter, the cross-correlation and the Monte Carlo errors, and the Liu et al.
  (2014) same-spectrum anchor.
