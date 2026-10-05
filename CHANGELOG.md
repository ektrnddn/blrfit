# Changelog

The full notes of versions up to 0.2.0 are attached to the 0.3.0 release on GitHub, and
each version is preserved by its tag.

## 0.3.0 (unreleased)

Fitted numbers change: on 823 SDSS spectra of the Liu et al. (2014) and Eracleous et al.
(2012) objects, the Hβ class changed for 2.7 per cent and Δv moved by more than
100 km/s for 31 Hβ and 6 Hα fits (`tests/data/deltas_anchor_0.2.0_to_0.3.0.csv`).

- The continuum fit starts from several points (four with a host) and keeps the best; a single
  start could stop in a local minimum without a host.
- Equally good decompositions are all measured; a spread of Δv above 100 km/s is flagged
  `degenerate`.
- New flags: `pl_unphysical`, `param_at_bound` (line velocities), `residual_outliers`,
  `ebv_assumed_zero`; Monte Carlo warnings join the line flags, and the errors of a line
  whose draws split between solutions are withheld.
- Bad pixels are masked with two neighbours on each side; SDSS and generic spectra are
  dereddened with SFD98 when the `dust` extra is installed.
- SDSS spectra are read with the inverse-variance mask by default, as in every validation;
  `--sdss-mask-policy conservative` (the 0.2.0 default) also drops every flagged pixel.
- Several spectra per call (`--list`, `--jobs`) with one catalogue table; readable names for
  public fits; a grouped `--help`. The `rv` command is removed (`blrfit.rv` stays, experimental).
- Python 3.10 or later; documentation rewritten as a method, inputs, outputs and validation page.

## 0.2.0 (2026-10-04)

- DESI is the default input; public spectra are found by TARGETID or position in DESI DR1/EDR
  and SDSS DR17, and generic FITS tables, rows and images are read with declared columns.
- SDSS pixels flagged by the pipeline mask are excluded on the command line.
- Monte Carlo errors use the pixel noise without the fitting floor; they are labelled
  conditional, and the DESI repeat-spectrum formula is no longer reported as an error.
- A 2,640-spectrum study of the Monte Carlo errors: 67 criteria pass, 5 are inconclusive.

## 0.2.0rc1 (2026-09-28)

- Corrections from an audit of the 0.1.0 fitter, each with its effect on the results, and the
  ultraviolet Fe II width policy made explicit.

## 0.1.0 (2026-09-13)

- First public release: the single-epoch model, profile measures, classes and flags, Monte
  Carlo errors, the between-epoch cross-correlation, and readers for SDSS, DESI and tables.
