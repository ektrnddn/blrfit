# Velocity curves and verdicts

`blrfit multiepoch` takes the pair tables of `blrfit pair` for many objects and decides, object by
object and line by line, whether the broad line moves. It is the analysis of Dadiani & Palmese
(2026, "Cosmic Pairs"), Sections 4.5–4.8 and 6, and every definition below is theirs.

```
blrfit multiepoch calibrate  --pairs PAIRS.csv --catalog CATALOG                 --out DIR
blrfit multiepoch verdicts   --pairs PAIRS.csv --catalog CATALOG --masses MASSES --out DIR [--facts FACTS.csv --all-fits FITS]
blrfit multiepoch orbits     --run DIR --pairs ... --catalog ... --masses ... --facts FACTS.csv
blrfit multiepoch population --run DIR --pairs ... --catalog ...
blrfit multiepoch fraction   --run DIR --pairs ... --catalog ... --masses ...
```

The same from Python:

```python
from blrfit.multiepoch import Inputs, run_calibration, run_verdicts, run_orbits, run_population, run_fraction

inp = Inputs(pairs="pairs.csv", catalog="catalog.fits", masses="masses.fits", facts="facts.csv")
run_calibration(inp, "run")  # run/CALIBRATION.json, CALIBRATION.md
run_verdicts(
    inp, "run/CALIBRATION.json", "run"
)  # run/verdicts.csv, objects.csv, curves.json, pair_table.csv, ...
run_orbits("run", inp)  # run/orbits.json, orbits.csv
run_population("run", inp)  # run/POPULATION.md
run_fraction("run", inp)  # run/BINARY_FRACTION.json, .md
```

## Inputs

| table | one row per | columns the chain reads |
|---|---|---|
| `--pairs` | pair of dated spectra, line | the columns of `blrfit pair` (`pairs.PAIR_COLUMNS`), and per spectrum `cls_a/b`, `snr_a/b`, `peak_snr_a/b`, `fwhm_a/b`, `mjd_a/b`, `id_a/b`; `dt_days`, `dt_rest_yr`; optional `bpk_a/b` (the branch of the class B rule), `fflags_a/b`, `fstatus_a/b`, and `object`, `tid_a`, `tid_b` where one object has several identifiers. CSV, gzipped or not. |
| `--catalog` | identifier (TARGETID) | `RA`, `DEC`, `z_in`, the coadd fit: `HA_class`, `HA_c50_sys`, `HA_fwhm`, `HA_broad_flux_snr`, `HB_class`, `CLEAN`. Identifiers within 1.5 arcsec are one object. |
| `--masses` | identifier | `LOGMBH`, `R_BLR_LTD`, `VMAX_Q01`, `PMIN_YR_Q01` (`blrfit.physics.target_mass`). |
| `--facts` (optional) | dated spectrum, line | `id`, `line`, `chosen`, `c50_sys`, `class`, `class_reason`, `instrument`, `broad_flux_snr`: the offset scale of the orbit fits, and the two-peak rule for objects without a coadd fit. |
| `--all-fits` (optional) | identifier | `HA_n_peaks`, `HA_peak_sep`, `HA_dip_frac`, `HA_fwhm` and the same for Hβ: the double-peaked objects. |

Instruments are read from the spectrum identifiers and dates: `desi_*` is DESI (a single night; `desi_daily_*` a
daily reduction), an SDSS identifier is SDSS below MJD 55000 and BOSS from it. Two SDSS spectra of the same plate
and fiber are one observation (the later coadd includes the earlier exposures) and are never paired.

## The spectrum check

Before any comparison every dated spectrum and line is `good`, `double-peaked` or `not used` (`spectra.csv`). A
spectrum is not used when the line is not measured, when its class is E (no broad line) or X (no narrow-line
reference), when its integrated broad-line S/N is below 8, when its fit did not converge, when it carries a quality
flag that the no-change pairs show to spoil velocities (`spectra.EXCLUDE_FLAGS`, empty by default), or when it is a
DESI night that disagrees at more than 4 sigma with another night within 30 days that itself agrees with a third.
A class B spectrum with two resolved peaks is `double-peaked`: kept, not used for velocities. Class B by the
very-broad rule is good, tagged `very_broad_B`. Tags: `weak_peak` (broad peak below 5 sigma per pixel), `class_W`,
`daily`, `boss`, `short_pair_unconfirmed`.

## Pairs

A pair is `measured` when the four checks of `blrfit pair` passed (the two directions agree, the two flux scales are
reciprocal, a clean minimum in both directions, the narrow-line frame holds), `not measured` when one failed, and
`not used` when a spectrum is not good or the two spectra are not independent. A measured pair's profiles `match`
when the shape statistic (the larger of the two directions, per pixel) is at most 0.5, else they `do not match`.
Only matching pairs carry a velocity; a change that comes with a changed profile is reported and never counted as a
motion.

## The noise model (`calibrate`)

Calibrated on objects whose broad lines should not move: coadd |Δv(Hα)| < 300 km/s, class A or F and a clean fit
(the `null` population; `broad`, every line with |Δv| < 1000 km/s, is the second model). Every object counts once
in every statistic. Four terms describe the scatter beyond the curvature errors of the pairs:

- `tau`: each spectrum's own jitter, by instrument. DESI's from nights at most 30 days apart (the floor that makes
  the pulls of those pairs NMAD 1, over √2), SDSS's and BOSS's from their same-instrument pairs. Every pair that
  contains the spectrum shares it, so that many pairs through one deviant spectrum cannot add up to a detection.
- the wandering of the broad line: a damped random walk in units of the object's broad FWHM, F, with the variogram
  SF²(Δt) = 2 (κF)² [1 − exp(−Δt/τ_c)] between two spectra Δt apart in the rest frame; κ and τ_c make the pulls NMAD 1
  in every bin of rest-frame gap, all kinds of pairs together.
- `omega`: what SDSS-to-DESI and BOSS-to-DESI pairs still need beyond the jitters and the wandering, an offset of
  one object's SDSS (BOSS) spectra against its DESI ones, shared by all of them.
- `epsilon`: an independent per-pair term from closed loops of three spectra, where the shared terms cancel; an
  object whose own loops close worse than the population's gets its own, larger, value, and one whose loops close
  worse than three times the typical value is tagged `pairs do not add up`.

`CALIBRATION.json` also carries, per kind of pair and line, the offset and the floor that make the pulls of the
pairs NMAD 1 (`adopted`): these serve the display and the per-pair significance, not the curve fits.

## The velocity curve and the verdict (`verdicts`)

For each object and line, one velocity per *time* (the spectra within 30 days of one another, single linkage) by
generalized least squares on all its measured pairs with matching profiles. With M the matrix that maps spectra to
pairs (+1 for the later, −1 for the earlier) the covariance of the pair changes is

    C = M diag(σ_k² + τ_k²) Mᵀ − ½ M Γ Mᵀ + ε² I + Σ_I ω_I² c_I c_Iᵀ,

σ_k the statistical error of spectrum k (the median curvature error of the directions in which it is the data),
Γ_kl = SF²(|t_k − t_l|), and c_I marking the pairs between instrument I and DESI.

The **motion test** compares one velocity for all times with a free velocity at every time; the line is **moving**
when the difference in χ² has p < 2.7 × 10⁻³ (the two-sided 3σ of a single pair), on times − 1 degrees of freedom.
A line that does not move is **constant** when all its pairs have matching profiles, **profile variable** when some
do not, and **not measurable** with fewer than two times joined by a measured pair. `objects.csv` gives the verdict
of both lines together: `double-peaked` for an object whose coadd has two resolved peaks, else `moving: Hα`,
`moving: Hβ` or `moving: Hα and Hβ`, else the first of profile variable, constant and not measurable that applies
to either line, with a `line_tag` saying whether the other line agrees within its errors.

Tags of a moving line (`tags` in `verdicts.csv`):

| tag | meaning |
|---|---|
| `two times` / `N times` | the number of times of the tested curve |
| `steady` / `not steady` | with three or more times over more than a year, a constant acceleration fits the velocities at p ≥ 0.01 (`p_steady`, `accel`, `accel_err`) |
| `allowed by its mass` / `faster than its mass allows` / `no mass` | no change between two times exceeds the largest change an orbit at the virial mass allows over their gap, 2 v(P) \|sin(πΔt/P)\| at the most favorable P ≥ P_min (q = 0.1, edge-on), by more than 2σ; `too_fast` lists the pairs of times that do. `too_fast_generous` and `mass_label` repeat the test at the generous limit of the mass (the BLR radius halved and the mass doubled: P_min × 0.25, v_max × 2) |
| `two spectra or one pair of times` | the line has two times only |
| `disappears when one spectrum is excluded` | the motion test, repeated with each spectrum left out, falls below 3σ for some spectrum (`rests_on` names it) |
| `disappears when one time is excluded` | the same with every spectrum of one time left out, so that a time of several spectra can be removed whole (`rests_on_time`) |
| `two pairs above 4 sigma` | two or more pairs across times change by at least 4σ in the full noise model (`n_pairs_4sig`) |
| `strict` | the best pair has shape ≤ 0.3, its two directions within 2σ of each other and both broad peaks at 8σ or more per pixel |
| `profile does not match in some pairs` | the line also has measured pairs whose profiles do not match |
| `pairs do not add up` | the object's loops close worse than three times the typical value |
| `width changes` | the median broad FWHM across the times of the curve changes by more than a factor 1.5 (`fwhm_ratio`) |
| `weak_peak`, `daily`, `boss`, `class_W`, `very_broad_B` | properties of the spectra of the best pair |
| `moves under both noise models` / `moves only under the ordinary-line noise` | the verdict under the second noise model (`verdict_alt`, `p_move_alt`) |

A **robust mover** is an object with a line that moves at three or more times and stays above 3σ with any one
spectrum excluded; `groups.group_of` sorts them by fixed rules, in this order: a steady drift that the mass allows,
changes faster than the mass allows in every moving line, both Balmer lines moving (not steadily), one line moving
(not steadily). `groups.orbit_group_of` sorts them by the orbit test instead.

## The orbit test (`orbits`)

For every line with three or more times, circular orbits the virial mass allows are fitted to the velocity curve
with its full covariance: every period from P_min, the closest orbit at which the active black hole's Roche lobe
still holds its broad-line region, to 10,000 yr, with the amplitude at most v_max (P/P_min)^(−1/3) (q = 0.1,
edge-on, the most permissive case), the center of mass within ±300 km/s of the narrow lines (`--com-max`), on the
offset scale of the best-measured time (from `--facts`). **An allowed orbit fits** when the mass condition costs at
most Δχ² = 4 against the best orbit without it and, with five or more times, the χ² probability is at least 0.01
(the period counts as a fitted parameter, dof = times − 4); with four times or fewer the orbit can only pass
through the points (`orbit_tested` False). `orbit_label` is `an allowed orbit fits`, `fits only at the generous
limit`, `no allowed orbit fits` or `no mass`. The steady drift in `orbits.json` is the steady test of the verdicts
(the straight line against free velocities, in pair space): one implementation since 0.5.0.

## The population test and the binary fraction

`population`: one statistic per object, the Hα verdict; strong offsets (coadd |Δv| ≥ 1000 km/s, and the clean
class A subset) against objects with |Δv| < 1000 (and < 300) km/s, matched 1:1 without replacement in strata of
coadd log FWHM (0.2 dex), log S/N (0.3 dex), number of times (2, 3+) and rest-frame span (< 5, ≥ 5 yr); McNemar's
exact test on the discordant pairs of each of 200 random matchings, the median reported.

`fraction`: for every testable strong offset with a mass, binaries of each rest-frame period P are drawn (mass
ratio log-uniform in 0.1–1, isotropic inclination, the BLR inside the active hole's Roche lobe, the coadd offset as
the orbital velocity at the DESI epoch); the orbit gives the change of every pair the motion test uses, and with the
full noise model the test's χ² difference is a noncentral χ², so the probability of a `moving` verdict follows
without simulating the noise. The one-sided 95% bound of the excess of movers over the matched objects, divided by
the summed sensitivity, is f95(P), the fraction of the strong offsets that can be binaries of period P.

## Outputs of a run directory

| file | content |
|---|---|
| `CALIBRATION.json`, `.md` | the noise model per population and line, the adopted floors per kind of pair |
| `spectra.csv` | every spectrum and line: status, reason, tags |
| `verdicts.csv` | every object and line: the verdict and the columns of the table above |
| `objects.csv` | both lines together: `combined`, `line_tag`, the tags of each line, `null` (a calibrating object) |
| `curves.json` | the velocity at each time (relative to the first and to the best-measured time), its error, the spectra |
| `pair_table.csv` | every pair: status, reason, change, its statistical and full errors, the shape statistic |
| `members.json` | object key → its identifiers |
| `VERDICTS.md` | the report: counts, tags, false alarms on the calibrating objects, the moving fraction by coadd offset |
| `orbits.json`, `.csv` | the orbit records of every curve (`orbits.ORBIT_COLUMNS`) |
| `POPULATION.md`, `BINARY_FRACTION.json` / `.md` | the matched comparison and the binary-fraction limit |

## Reproducibility

`tests/test_multiepoch_regression.py` runs the chain on a frozen extract of the catalog of the paper (38 objects,
6,092 pair rows, the noise model of the whole catalog) and reproduces the verdicts, tags, curves and object verdicts
of the paper's run to the printed digits, and the orbit summaries of its candidate lines. The names that 0.5.0
changed are `n_clean_pairs` → `n_pairs_4sig` and the tag `more than one clean pair` → `two pairs above 4 sigma`
("clean" is the catalog's fit flag); the dead `marginal` label is gone; the steady test has one implementation (in
the paper's run one line, 39633286363874179 Hα, had p = 0.0098 in the verdicts and 0.018 in the orbit records: it
keeps the verdicts' value).
