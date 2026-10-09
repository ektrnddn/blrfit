# Validation

What has been tested, the numbers, and how to reproduce them. Updated for version 0.4.0
(October 2026).

## Software tests

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not slow and not network" -n auto      # about 540 tests, a few minutes
python -m pytest -m slow -n auto                            # the larger synthetic grids, about 15 minutes
```

The fast suite covers the readers, the command line, every model component, the
classification, the Monte Carlo bookkeeping, the velocity changes between epochs on
synthetic profiles and on a fitted spectrum, the tiers, and synthetic spectra with known answers:
bulk shifts of ±1200 km/s recovered to better than 60 km/s (Hα) and 80 km/s (Hβ);
double-peaked, asymmetric and symmetric profiles classified B, C and F; narrow-line
galaxies classified E; host-dominated spectra; discrepant [S II]; non-Gaussian narrow
lines; a narrow-line system 880 km/s from the input redshift; Hα at the edge of the
spectrum.

**Pinned fits.** `tests/test_pins.py` refits five real spectra (four SDSS, one DESI) and
compares every value of their summary rows with `tests/data/pins_0.3.0.json`. On other
numerical libraries the least-squares solver can end at a slightly different point of a
degenerate decomposition, so the comparison allows such differences; with
`BLRFIT_STRICT_PINS=1` it is exact, bit for bit, on the stack that recorded the pins
(Python 3.12, numpy 1.26.4 with OpenBLAS 0.3.21, scipy 1.13.1, astropy 6.1.3, macOS).
Continuous integration runs the fast suite on Python 3.10–3.13 with current releases,
and weekly on that numpy 1.26 stack.

## Comparison with published offsets

Liu et al. (2014) published, for each of their offset quasars, the SDSS spectrum they
measured and the peak velocity offset of broad Hβ. We fit the same spectra (388 are in
DR16) at their redshifts with the default settings and compare our peak offset
(`HB_v_peak_sys`) with theirs, for the 373 with a measurable broad Hβ:

| Pearson r | median difference | NMAD of the difference | same sign |
|---:|---:|---:|---:|
| 0.924 | +6.1 km/s | 111.7 km/s | 95.7 % |

The test (`tests/test_anchor_liu.py`, slow) requires r ≥ 0.90, |median| ≤ 20 km/s,
NMAD ≤ 115 km/s and sign agreement ≥ 95 per cent. It needs the 824 spectra of the Liu et
al. (2014) and Eracleous et al. (2012) objects (438 MB, not in the repository; the test's
docstring says how to assemble them from the SDSS archive).

## Monte Carlo errors

We tested the errors on 2,640 synthetic spectra: 33 configurations (Fe II and host
cases, weak narrow lines, a range of offsets and widths), two peak S/N values (15 and
30), 20 noise realisations each, on an SDSS-like and a DESI-like wavelength grid, with
200 draws per spectrum. The 72 criteria were fixed before the run: the pooled accuracy of
Δv, FWHM and broad flux, and the fraction of truths inside one and two errors, which had
to lie within 60.3–76.3 per cent and 91.4–99.4 per cent (a pass when the whole 95 per
cent bootstrap interval lies inside, a failure when it lies outside, inconclusive
otherwise).

| Quantity | Accuracy | Coverage |
|---|---|---|
| Δv (`c50_sys`) | 8 of 8 pass | 16 of 16 pass |
| FWHM | 8 of 8 pass | 12 pass, 4 inconclusive |
| Broad flux | 8 of 8 pass | 15 pass, 1 inconclusive |

The five inconclusive checks are all on the DESI-like grid, and each interval crosses the
lower boundary:

| Line, S/N | Quantity, interval | Coverage | 95 % interval | Spectra |
|---|---|---:|---:|---:|
| Hα, 15 | FWHM, two errors | 93.8 % | 91.4–96.1 % | 579 |
| Hβ, 15 | FWHM, one error | 61.5 % | 57.7–65.2 % | 660 |
| Hβ, 30 | broad flux, two errors | 92.3 % | 89.7–94.4 % | 660 |
| Hβ, 30 | FWHM, one error | 64.2 % | 60.2–68.0 % | 660 |
| Hβ, 30 | FWHM, two errors | 92.1 % | 89.8–94.1 % | 660 |

So the offset errors cover as they should on these spectra; the width errors are
may be somewhat small for Hβ on the DESI grid. The spectra are drawn from the same
family of components that the model fits, with Gaussian noise: the study cannot reveal
errors of the model itself (for example a host that the eigenspectra do not describe). It
was run with version 0.2.0; 0.3.0 changes the continuum starts and the masks, and the
study has not yet been repeated with it. The protocol, the decisions and the scripts are
in [`validation/uncertainty_20261003`](../validation/uncertainty_20261003/README.md),
which also says how to regenerate the report from the archived records.

**A known failure.** `tests/test_errors_mc.py::test_monte_carlo_pulls_of_c50_sys` (slow)
fits 20 noise realisations of one configuration (Hα and Hβ at +800 km/s, FWHM
4000 km/s, continuum S/N 12) with 30 draws and requires the median |(Δv − truth)/error|
to stay below 1.11, the 99th percentile for unit Gaussian pulls. Since version 0.2.0,
whose errors use the pixel noise alone (without the 2 per cent fitting floor), Hα gives
1.13: the recovered offsets are biased by +14 km/s against a median error of 15 km/s,
while the spread of the pulls is normal (NMAD 0.85). With the noise of version 0.1.0
(pixel noise plus the floor) the errors are 16 per cent larger and the same fits give
1.01, so the bound passed only because the larger errors absorbed the bias. The result is
the same in 0.2.0 and 0.3.0 and on both numerical stacks. The bound is kept and the test is
marked as an expected failure; it will report when the bias, which is not yet understood,
is removed.

## Velocity changes

The template cross-correlation of `blrfit pair` ([method](method.md#velocity-changes-between-epochs))
was developed on synthetic profiles and on SDSS spectra of the Eracleous et al. (2012) and
Liu et al. (2014) objects, with every rule fixed before it was applied to DESI data, and
then checked on DESI spectra of the targets of our offset-line search: a confirmation half of
1,461 targets chosen by a hash of the TARGETID, which took no part in the development.

**Synthetic profiles** (`tests/test_pairs_synthetic.py`). Single, asymmetric two-Gaussian
and three-Gaussian profiles (FWHM 2000–10 000 km/s) on DESI-like and SDSS-like pixel grids,
shifted by up to ±2500 km/s: recovered to better than 1 km/s without noise, with a flux
scale and a linear baseline absorbed exactly. With noise at a broad peak S/N of 8 and 30
(300 realisations per shift), the curvature error of one direction covers the truth at
68.3 ± 8 and 95.4 ± 4 per cent; with both epochs noisy and each template refitted to its
own data, the symmetric estimate with the error hypot(σ, σ′) covers at the same rates,
while a single direction's error alone does not. A template broadened by 75 km/s, the
largest resolution difference between SDSS and DESI, moves the shift by less than 3 km/s.

**Pairs with no expected change.** 330 consecutive SDSS–SDSS pairs of 456 literature
objects, and 335 pairs of consecutive DESI nights at most 30 days apart of the
confirmation-half targets: on the DESI nights the Hα pulls had an NMAD of 1.11 and the Hβ
pulls 1.03 after the screens (285 and 223 of 335 pairs retained). The systematic term per
line was fitted on a training half of the targets (Hα 45 km/s on 118 pairs, Hβ 36 km/s on
97) and checked on the other half, where it brought the pull NMAD from 1.09 to 0.97 (Hα,
167 pairs) and from 0.98 to 0.89 (Hβ, 126 pairs). SDSS pairs under a year gave 29–30 km/s,
below both. On these pairs the 99th percentile of |s + s′|/err is 2.6–3.0, the threshold of
the direction screen; the shape screen calls 11 per cent of unchanged DESI Hα pairs changed
(1 per cent for Hβ), and 45 per cent (21 per cent) of SDSS–DESI pairs 10–15 years apart,
where resolution, aperture and calibration differ.

**Injected shifts.** Shifts of 0, ±150, ±300, ±600, ±1000, ±1500 and ±2500 km/s injected
into real pairs, either by moving the template epoch's broad model inside its own data
(residual design) or by placing it in the other epoch's data (swap design): on the SDSS
pairs the mean bias is within ±9 km/s (Hβ) and ±14 km/s (Hα) through ±1000 km/s and passes
a ±30 km/s tolerance at every shift to ±2500, with 92–99 per cent of the pairs retained and
at least 90 per cent of shifts of 600 km/s and more recovered. On the DESI–DESI and
SDSS–DESI pairs the Hα bias passes the same tolerance where the precision of single DESI
nights (median error 138 km/s) allows a decision, and is inconclusive elsewhere; no
unchanged pair gave |s| > 3 err and > 300 km/s in any cell. Hβ (median error 337 km/s) is
measured and reported with its errors but its bias test is inconclusive at that precision
and its recovery falls below 90 per cent in some cells: Hβ is a check on Hα, not a primary
line. With both epochs synthetic from one real fit and refitted, the pair error covers the
declared truth at 69–81 per cent (68 nominal) and 95–98 per cent (95 nominal).

The 4σ threshold of the tiers: of the 330 SDSS pairs with no expected change, one exceeded
3σ (0.9 expected) and none 4σ. These numbers are from the analysis behind our DESI
search and are not reproduced by the package tests, which cover the code on synthetic and
bundled data.

## Not validated

- Real-data error calibration of the single-epoch offsets: the comparison with Liu et al.
  tests offsets, not their errors.
- Profiles outside the model family (Lorentzian, disc, shouldered profiles), hosts from
  other stellar-population libraries, other Fe II templates; these tests are planned.
- The earlier `blrfit.rv`; the virial masses and the orbital bound as physics (standard
  calibrations are applied, not tested); Mg II, including Mg II changes.
