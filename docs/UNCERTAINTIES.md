# What the reported uncertainties mean

The CLI defaults to `--nmc 0`: a point fit with no Monte Carlo error bars.
Missing errors are not zero. `--nmc 30` requests the existing conditional
Monte Carlo procedure; requesting more draws alone does not establish calibration.

For each draw, blrfit perturbs the input spectrum using its supplied statistical
pixel errors and repeats continuum and line fitting. The host model, redshift,
extinction, broad-component counts and pixel masks remain fixed. The 2% fitting
floor stays in the fitting weights but is not added to the default simulated
noise. The reported `err` is half the 16th–84th percentile range of contributing
draws. It is not automatically a Gaussian standard deviation, and twice that
number is not automatically a 95% confidence interval.

| Quantity | Existing evidence | Limitation |
|---|---|---|
| Broad center relative to the fitted narrow reference (`c50_sys`) | Four pooled one-error coverage checks passed at injected peak S/N 15 and 30 in the fixed synthetic experiment. One two-error check passed; three were inconclusive. | Conditional on the declared measurement availability and synthetic grid; not a host-systemic or real-survey calibration. |
| Broad FWHM | Pooled point-accuracy checks passed on the same grid. MC errors are computed if requested. | Point accuracy does not establish width-error coverage. |
| Broad integrated flux | A point estimate and optional MC error are returned. | The prior center-coverage results do not validate flux-error coverage. Flux calibration and decomposition systematics remain. |
| Virial mass | Derived quantities can be calculated from the fitted line properties. | Statistical propagation is separate from intrinsic scatter, virial-factor uncertainty and calibration assumptions. Night-to-night estimates are not measurements of changing true mass. |
| Difference between epochs | Experimental code exists separately. | Single-spectrum errors do not validate the response or uncertainty of a relative-shift estimator. |

The prior confirmation comprised 1,980 synthetic spectra and had 13 formal
passing checks, three inconclusive checks and no formal failure. It is preserved
unchanged. It did not require matching old astrophysical fits.

## Finite confirmation being prepared

The next candidate uses the same numerical fitter and error formula with an
explicit 200-draw budget. The purpose is to reduce sampling noise in the estimated
percentile widths. It is not an error multiplier or a change to the line model.
Its fixed design comprises 33 physical configurations, two signal-to-noise levels,
20 fresh noise realizations per configuration and two wavelength grids: 2,640
synthetic spectra. Center, FWHM and broad-flux accuracy and coverage are reported
separately. Pass, fail and inconclusive outcomes are fixed in advance.

The grids resemble DESI linear and SDSS logarithmic sampling. Independent Gaussian
noise, illustrative fixed masks and observed Gaussian profile widths define the
scope; these are not complete models of survey covariance, the line-spread
function or stellar populations. Success would support the stated conditional
synthetic claims, not universal errors for arbitrary spectra. The experiment is
pending; 200 draws are not yet advertised as a validated default.

## Using results responsibly

Keep classifications, measurement availability, solver status and MC diagnostics
with each value. Unconverged draws, failed draws, too few usable samples, changing
narrow references, basin switches and multimodality are recorded. A finite error
or an A/B class does not override these diagnostics.

The historical DESI repeat-spectrum formula is available only as a separately
labelled legacy diagnostic. It is not a current SDSS error prescription or an
empirical correction to make an inconclusive coverage test pass.

The existing production catalogues were point-fit runs. New uncertainty products
must identify their own method, draw count, seed and validation scope, while
preserving the original point-fit records.
