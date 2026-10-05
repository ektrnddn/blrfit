# Validation

What has been tested, the numbers, and how to reproduce them. Updated for version 0.3.0
(October 2026).

## Software tests

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not slow and not network" -n auto      # about 540 tests, a few minutes
python -m pytest -m slow -n auto                            # the larger synthetic grids, about 15 minutes
```

The fast suite covers the readers, the command line, every model component, the
classification, the Monte Carlo bookkeeping and synthetic spectra with known answers:
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

## Not validated

- Real-data error calibration: the comparison with Liu et al. tests offsets, not their
  errors.
- Profiles outside the model family (Lorentzian, disc, shouldered profiles), hosts from
  other stellar-population libraries, other Fe II templates; these tests are planned.
- Velocity changes between epochs (`blrfit.rv`), black hole masses and Mg II.
