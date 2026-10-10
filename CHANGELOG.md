# Changelog

The full notes of versions up to 0.2.0 are attached to the v0.3.0rc1 pre-release on GitHub,
and each version is preserved by its tag.

## 0.4.0 (2026-10-10)

The fitter is unchanged from 0.3.0rc1 (the pinned fits of `tests/data/pins_0.3.0.json` hold);
0.4.0 adds the velocity changes between epochs and the candidate tiers, and closes 0.3.0.

- `blrfit pair`: the velocity change of each broad line between the dated spectra of one
  object, by a template cross-correlation: one epoch's fitted profile is slid across the
  other epoch's continuum- and narrow-subtracted data, in both directions, with
  statistical errors from the pixel noise and a calibrated term per line. A shape
  statistic separates a moved profile from a changed one; screens on the two directions,
  the flux scales, a second minimum and the narrow-line frame withhold doubtful pairs.
  Writes pair, target and tier tables and one figure per pair; takes spectra, saved fits
  or a public search.
- `blrfit tiers`: candidate tiers of the objects of pair tables (disk, binary, platinum for
  both lines, almost, profile, stable, none), with an orbital bound at the virial mass.
- `physics`: virial masses (Greene & Ho 2005; Vestergaard & Peterson 2006), the radius of
  the broad-line region (Bentz et al. 2013) and the largest orbital velocity change.
- Tables can be written and read as CSV.
- Python: `measure_pair`, `pair_record`, `enumerate_pairs`, `classify_target`,
  `classify_table`, `target_mass`, `plot_pair`.
- `target_mass`: the 5100 Å luminosity of a fit flagged `pl_unphysical` is the value implied by
  the broad Hα (or Hβ) luminosity through Greene & Ho (2005), not the power law, which is 0.3 dex
  too faint for such fits; `l5100_source` and `l5100_pl` record the choice (`l5100_from_line`).
- `blrfit.rv`, the earlier cross-correlation, is deprecated: it is imported on first use, with a
  warning, and will be removed in 0.5; its 0.1.0 calibration constants are marked superseded.
- Python 3.14 is tested and declared.
- `validation/uncertainty_20261010`: the single-spectrum uncertainty study prepared for repetition
  with this release (the protocol of 2026-10-03; `prepare.py`, `run.sbatch`, `finish.py`).

## 0.3.0rc1 (2026-10-05)

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
