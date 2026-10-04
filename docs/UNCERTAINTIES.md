# What the reported uncertainties mean

The CLI defaults to `--nmc 0`: a point fit with no Monte Carlo errors. Missing
errors are not zero. Use `--nmc 200 --seed 0` to request the draw count evaluated
in the October 2026 study. This is slower and remains conditional; the default
has not been changed to compute errors for every spectrum automatically.

## Definition

Each draw perturbs the input spectrum using its supplied statistical pixel
errors and repeats continuum and line fitting. The host model, redshift,
extinction, broad-component counts and pixel masks remain fixed. The 2% fitting
floor stays in the fitting weights but is not added to the default simulated
noise. `err` is half the 16th–84th percentile range of contributing draws. It is
not automatically a Gaussian standard deviation, and twice that number is not
a universal 95% confidence interval. The published coverage study tests these
symmetric errors around the ordinary measurement explicitly.

## Evidence by quantity

| Quantity | Fixed 2,640-spectrum / MC200 evidence | Limitation |
|---|---|---|
| Broad center relative to the narrow reference (`c50_sys`) | All 8 pooled accuracy and 16 coverage checks pass. | Conditional on measurable lines and the declared synthetic grid; not a host-systemic or between-epoch calibration. |
| Observed broad FWHM | All 8 accuracy checks pass; 12 coverage checks pass and 4 remain inconclusive. | Width errors cannot be described as generally calibrated. |
| Broad integrated flux | All 8 accuracy checks pass; 15 coverage checks pass and 1 remains inconclusive. | Flux calibration and decomposition systematics are additional uncertainties. |
| Virial mass | Not calibrated by this study. | Statistical line-error propagation does not include intrinsic scatter, virial factors or all luminosity/width correlations. |
| Change between epochs | Not validated by this study. | Single-spectrum center errors do not establish the response or uncertainty of a relative-shift estimator. |

The overall result is **67 pass, five inconclusive, zero fail** under fixed
criteria. All five inconclusive checks are width/flux coverage on the DESI-like
grid and cross a lower acceptance boundary. The [validation report](VALIDATION_STATUS.md)
lists their exact values, denominators, selection and thresholds. This is not
proof that they fail or a reason to choose another error multiplier.

The study uses 33 controlled physical configurations, two injected S/N levels,
20 fresh noise realizations and two sampling grids, with independent Gaussian
noise and illustrative masks. Halpha/SNR15 is unavailable in 81/660 DESI-like
and 57/660 SDSS-like cases; coverage is conditional on measurable outcomes.
Other pools each have 660 measurable cases. Success on this experiment does not
establish real-survey covariance, intrinsic-width deconvolution, host/redshift
systematics or validity for arbitrary component families. The earlier MC30
study remains separate: 13 passing and three inconclusive criteria. Other MC
draw counts and the historical `effective` noise policy do not inherit MC200
input-noise coverage results.

## Reading one fit

Keep class, flags, measurement availability, optimizer status and MC diagnostics
with each value. Finite errors and an A/B label do not override unconverged draws,
insufficient samples, basin switching, multimodality or a weak narrow reference.
A line with no usable systemic reference is not a reliable velocity measurement.
Night-to-night virial-mass estimates are not measurements of changing true mass;
line variability, decomposition and the mass prescription also matter.

The historical DESI-repeat formula is available only as a separately labelled
legacy diagnostic. It is not an SDSS prescription or an empirical correction to
make an inconclusive test pass. The large production catalogues were point-fit
runs and remain so. Any new uncertainty catalogue must identify its own inputs,
method, draw count, seed and supported scope.
