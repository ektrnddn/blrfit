# Current validation status

Updated October 4, 2026. **Version 0.2.0: single-spectrum Halpha/Hbeta fitting,
explicit input provenance and conditional statistical errors.** The numerical
model, optimizer and MC implementation remain RC1. Public input handling and
output labels are updated; no fitting component, error multiplier or selection
threshold was adjusted to make the confirmation pass.

## Fixed uncertainty confirmation

NERSC job **59295726** completed in 05:20:39 with all **2,640/2,640** synthetic
spectra returned and no execution errors. Each used 200 requested noise refits.
The downloaded archive, all 2,644 indexed members, protocol/roster identities and
integer coverage counts were verified locally. The scientific report reproduces
exactly from retained records. A remote preservation receipt verifies the saved
ordinary fits and their durable copy; those fit/archive bodies were not all
downloaded locally.

The fixed criteria yielded **67 pass, five inconclusive, zero fail**. The overall
scientific outcome remains **inconclusive**. This does not invalidate all point
fits, establish all error bars, or justify changing thresholds after seeing results.

| Quantity | Pooled point accuracy | One-/two-error coverage | Supported interpretation |
|---|---|---|---|
| Narrow-frame broad center (`c50_sys`) | 8/8 pass | 16/16 pass | Conditional pooled performance on this grid; not independent host-systemic or between-epoch validation. |
| Observed broad FWHM | 8/8 pass | 12 pass, 4 inconclusive | Width errors remain qualified, especially the listed DESI-like groups. |
| Integrated broad flux | 8/8 pass | 15 pass, 1 inconclusive | Statistical flux coverage is partly supported; calibration/decomposition systematics remain. |

The five inconclusive checks are all on the **DESI-like linear grid**:

| Line / injected S/N | Quantity / interval | Coverage | 95% cell-bootstrap interval | Eligible |
|---|---|---:|---:|---:|
| Halpha / 15 | FWHM / two errors | 93.782% | 91.379–96.075% | 579/660 |
| Hbeta / 15 | FWHM / one error | 61.515% | 57.727–65.152% | 660/660 |
| Hbeta / 30 | Broad flux / two errors | 92.273% | 89.697–94.394% | 660/660 |
| Hbeta / 30 | FWHM / one error | 64.242% | 60.152–68.030% | 660/660 |
| Hbeta / 30 | FWHM / two errors | 92.121% | 89.848–94.091% | 660/660 |

Each confidence interval crosses the lower acceptance boundary. The one-error
coverage region was fixed at **60.3–76.3%**, and the two-error region at
**91.4–99.4%**. A pass requires the entire interval to lie inside the region;
a failure requires it to lie wholly outside. Overlap is inconclusive. In
particular, the Halpha width result narrowly crosses 91.4%; it remains
inconclusive with no rounding-based relabelling. These results do not establish
the cause of any undercoverage or prove that more draws would fix it.

## Denominators and scope

There are 33 controlled physical configurations, two injected peak S/N levels
(15 and 30), 20 fresh noise realizations per configuration/SNR and two grids.
Each line/grid/SNR pool has 660 planned spectra. Halpha/SNR15 was measurable in
**579/660** DESI-like and **603/660** SDSS-like cases; the other six pools each
had **660/660** measurable cases. The unavailable Halpha outcomes have class X
(no measurable systemic reference). No additional MC-error eligibility loss
occurred among measurable lines for these three quantities. Flags and optimizer
and MC diagnostics remain in all records. Availability is not 100% simply
because all computation records returned.

The accuracy criteria concern the pooled mean center error normalized by
max(30 km/s, 5% of the absolute injected offset), within ±1, and pooled mean
relative FWHM/flux bias within ±10%. They do not bound every object's error.
The 95% intervals resample all 33 configuration blocks, retaining their observed
availability. Coverage requires at least 167 finite MC contributions out of 200,
a positive finite error, and no `mc_too_few` flag. No family-wise confidence
claim across the 72 decisions is made.

The design uses independent Gaussian pixel noise, explicit illustrative masks,
observed Gaussian widths, controlled Fe/host/weak-narrow cases and the existing
measurement selection. It does not simulate full survey covariance, instrument
line-spread functions, all stellar populations, uncertain redshifts or changing
component families. Fresh noise on previously developed configurations is not
an untouched astrophysical population. Hbeta can inherit Halpha's narrow-line
reference; neither this study nor the classes validates a host-galaxy systemic
velocity. Passing pooled criteria does not certify each configuration or survey.

The CLI default remains **`--nmc 0`**. `--nmc 200` selects the evaluated draw count;
it is not a universal calibration setting. Other draw counts remain supported
but do not inherit these results. All-catalogue MC, mass-error calibration,
Mg II and between-epoch velocity validation are outside this release's claims.

## Software and prior evidence

The input candidate passed **517 software tests in each of Python 3.9 and 3.12**,
with one skip, 26 deselections and three expected failures. The selected CI suite
excludes slow and live-network tests. Live retrieval/point-fit examples and input
contracts are recorded in [the input guide](INPUT_INTERFACE.md). Final integration
checks are available in the repository's GitHub Actions history.

The prior 1,980-spectrum MC30 study remains 13 pass and three inconclusive;
its outcomes were not overwritten or pooled into this confirmation. Real-data
preflights and the scoped primary, nightly DESI and staged SDSS point-fit runs
establish execution/accounting for their declared inputs, not general scientific
truth or complete archival coverage. The earlier engineering and scientific
record remains in [RC1 validation history](RC1_VALIDATION_HISTORY.md).

See [the uncertainty guide](UNCERTAINTIES.md), [all 72 decisions and reproduction
instructions](../validation/uncertainty_20261003/README.md), and the original
[historical numerical examples](HISTORICAL_VALIDATION.md). Publication of this
software release does not revise the archived RC1 fits or add error bars to them.
