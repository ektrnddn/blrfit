# blrfit

Fit the broad Hα and Hβ profiles of one AGN spectrum, measure their offsets relative
to the fitted narrow-line reference, and save profile measurements, classes, quality
flags and a diagnostic figure. DESI is the default input survey; SDSS and general
spectral tables/images are explicit alternatives.

**Version 0.2.0** provides the public/local input interface with the numerical
fitting and Monte Carlo core used in
[v0.2.0rc1](https://github.com/ektrnddn/blrfit/releases/tag/v0.2.0rc1).
The existing primary and observing-date catalogue fits remain tied to RC1; this
release does not refit or relabel them. Use the versioned installation below.

**Scientific scope.** Successful execution is not proof that every decomposition or
error bar is reliable. The default `--nmc 0` produces point estimates and **no Monte
Carlo uncertainties**. Optional MC errors are conditional statistical estimates;
real-data and survey-wide calibration remains incomplete. Mg II and between-epoch
velocity analysis are experimental. See [validation status](docs/VALIDATION_STATUS.md).

## Install

Use an isolated environment (Python 3.10 or later):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'blrfit[fetch] @ git+https://github.com/ektrnddn/blrfit.git@v0.2.0'
python -m blrfit --version
```

The tag pins the software version; preserve the numerical-library versions and
input files as well when reproducing a fit. From a checkout use
`python -m pip install '.[fetch]'`. The immutable `v0.2.0rc1` tag remains available
for reproducing the earlier interface and deployments.
Local files need only the base dependencies; public retrieval needs the `fetch` extra.
`dust` adds dust-map access; `test` adds the software test dependencies.

## Fit a public DESI target

No coordinates are needed when its TARGETID is known. Only public DR1 is searched by
default. Each returned survey/program coadd is fitted separately and recorded in a
manifest; none is silently selected as a representative observing night.

```bash
blrfit fit --targetid 39627574082538900 --out result_desi
```

The lookup uses the public [Data Lab DESI catalogue](https://datalab.noirlab.edu/data/desi)
and downloads the matching target from the DESI archive. `--releases edr` selects EDR;
`--releases dr1,edr` searches both, whose observations can overlap. Public retrieval
is limited by those releases and the availability of the catalogue and file servers.
It does not search proprietary DESI data or assemble all nights.

## Search by position, optionally including SDSS

Coordinates are ICRS degrees. The default search is DESI within 1.5 arcsec. Add
`--include-sdss` to retrieve SDSS DR17 spectra within **at most 1.5 arcsec** as well.

```bash
blrfit fit --ra 3.1997148876 --dec -8.7834904331 --include-sdss --out result_position
# Download only, without fitting:
blrfit fetch --targetid 39627574082538900 --include-sdss --out downloaded
# SDSS alone:
blrfit fit --survey sdss --ra 3.1001 --dec -10.374 --out result_sdss_position
```

Multiple DESI TARGETIDs cause a clear stop with the candidate list saved. Choose
`--targetid` or explicitly request `--all-matches` to keep each object separate.
SDSS matches are **positional candidates**, not confirmed associations: review their
coordinates, redshifts and spectra. Repeat products are retained. The search is not
a complete SDSS-V/DR20 census; a local compatible SDSS FITS file can still be read.
A failed query is reported as an error, not as “no spectrum exists.” Use a fresh
output directory for each public search so earlier manifests remain intact.

## Fit a local file

```bash
# DESI is the default. A multi-row file needs an exact TARGETID.
blrfit fit coadd-main-dark-17260.fits --targetid 39627574082538900 --out result_local
# Supply z if there is no matching redrock file beside it:
blrfit fit my_desi.fits --targetid 39627574082538900 --z 0.2203154 --out result_local_z
# SDSS: file contents, not its name, identify the spectral columns.
blrfit fit my_sdss.fits --survey sdss --z 0.2288 --out result_local_sdss
```

A DESI file containing exactly one spectral row can omit TARGETID. An explicit
`--redrock path.fits` is also supported. Multiple rows with the same TARGETID must
be separated into observations before fitting; exposure rows are never flattened
together. The standard public DESI retrieval returns a full-depth coadd, which may
contain multiple nights. Local SDSS `spec` files supply their pipeline coadd from
extension 1; their individual-exposure extensions are not fitted by this command.

SDSS inputs are read on their native vacuum `10**loglam` grid, with flux and IVAR in
the supplied units. By default nonzero native MASK/AND_MASK and invalid statistical
pixels are excluded, matching the conservative project input policy. The Python
reader retains its historical `mask_policy="ivar"` default for compatibility;
the CLI uses `--sdss-mask-policy conservative`. Select `ivar` explicitly to reproduce
the older reader. No extra heliocentric/barycentric or air-to-vacuum correction is
applied to these already reduced SDSS products. No SDSS-wide uncertainty or resolution
calibration is implied. Insufficient continuum support remains an unusable fit;
it is not repaired by inventing data or changing the model.

For DESI, E(B−V) defaults to the FIBERMAP value (SFD98). For SDSS/general files it is
the SFD98 value at the file's coordinates (or `--ra`/`--dec`) when the `dust` extra
and its map are installed; otherwise it is taken as zero, with a printed warning and
the flag `ebv_assumed_zero` on every fitted line. `--ebv VALUE` overrides both.
Redshifts are not remeasured automatically; `--z` overrides the file value. Review
catalogue redshift warnings in the saved metadata.

## Fit several spectra

```bash
# Several files, or a list: one path per line (optionally followed by a TARGETID),
# or a FITS/ECSV/CSV table with a column path and optionally targetid and z
blrfit fit spectra/*.fits --survey sdss --out results
blrfit fit --list targets.csv --jobs 8 --out results
```

Each spectrum is fitted as it would be alone and writes its own `*_fit.json` and
figure. The command prints one line per spectrum and writes the catalogue table
`results/blrfit_summary.fits` (or `--table name.ecsv`): one row per spectrum, with
units, and failed spectra kept with their reason. `--jobs N` fits N spectra at a
time or, for a single spectrum, refits its Monte Carlo draws in N processes; the
results do not depend on N.

## Fit general FITS tables or images

**Wavelength, flux and redshift alone are insufficient for this weighted fit.**
Supply a per-pixel 1σ error or inverse variance; the program does not invent errors.
The fit also needs sufficient wavelength coverage outside the emission-line windows
to constrain the continuum.

```bash
# FITS table (also CSV/ECSV): explicit column names, including the noise column.
blrfit fit spectrum.fits --survey generic --hdu 1 --wave WAVE --flux FLUX --ivar IVAR --z 0.2 --out result_table
# Read one row of a vector-valued FITS table, including its redshift column:
blrfit fit spectra.fits --survey generic --row 2 --wave LAMBDA --flux FLUX --err SIGMA --z-column Z --out result_row
# Separate image extensions, with Z in a FITS header:
blrfit fit arrays.fits --survey generic --wave-hdu WAVE --flux-hdu FLUX --err-hdu ERROR --z-key Z --out result_image
```

Wavelength defaults to observed-frame vacuum Å. Use `--wave-unit nm`, `--frame rest`
or `--air` when appropriate. Flux must be **fλ per observed Å** in units of
10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹; `--flux-scale` converts both flux and statistical noise.
Changing wavelength units alone does not convert the flux-density unit. If the
flux has already been transformed to fλ per rest-frame Å, set `--flux-frame rest`
separately; the reader divides it and its errors by (1+z). This is not a conversion
from luminosity density or fν. Optional `--mask COLUMN` excludes nonzero mask values.

FITS has no universal spectrum layout. Explicit table columns or image-HDU arrays
are supported; arbitrary WCS-only images, cubes and instrument-specific products
need an adapter. Ambiguous multi-spectrum arrays are rejected rather than flattened.
Legacy filename-based selection is available explicitly with `--survey auto`.

## Results and uncertainties

Each successful spectrum writes `*_fit.json` and `*_fit.png`; `--pickle` also saves
its full fit. JSON includes Hα/Hβ classifications, flags, solver diagnostics,
redshift/reader provenance and uncertainty status. A public search writes
`inputs/fetch_manifest.json` and `fit_manifest.json`, and names each fit after its
product (`fits/desi-dr1-main-dark-17260-39627574082538900_fit.json`,
`fits/spec-0651-52141-0072_fit.json`); individual failures remain
listed and the command returns a nonzero exit status for incomplete work.

Add `--nmc 200 --seed 0` to use the draw count evaluated in the latest conditional
uncertainty study (slower than a point fit). Other counts remain supported, but
do not inherit that study’s coverage results. These perturb the supplied
statistical pixel noise and refit under the RC1 assumptions; the 2% fitting floor
is not added to the default simulated noise. They do not include all component-choice,
host, calibration, redshift or instrumental systematics. Coverage remains partly
inconclusive; **do not label them fully calibrated uncertainties**. With `--nmc 0`,
missing errors are JSON `null`, not zero. The old DESI-repeat formula is no longer
printed as an error bar, particularly for SDSS. `--legacy-error-diagnostic` can
retain it for DESI as a separately named, uncalibrated diagnostic.

The model details below describe the unchanged RC1 numerical core. The input
interface does not revise archived catalogue fits.

## What is measured

The summed broad model P(v), evaluated on a 5 km/s grid, is reduced to the non-parametric
quantities of Marziani et al. (1996) and Eracleous et al. (2012): the peak velocity, the
flux-weighted centroid, the bisector centres c(f), the midpoints between the two outermost
crossings of the profile with f times its peak, and the widths W(f) for f = 1/4, 1/2, 3/4 and
0.9 (FWHM ≡ W(1/2)); the asymmetry index A.I. = [v_R(1/4) + v_B(1/4) − 2 v_peak] / W(1/4); the
kurtosis index K.I. = W(3/4)/W(1/4), 0.456 for a Gaussian; the number of resolved peaks (local
maxima above 20 per cent of the maximum with a prominence of 5 per cent), their separation and
the dip between them; the broad flux, its equivalent width against the total continuum, its
luminosity, and the integrated and peak signal-to-noise ratios. All velocities are referred to
the systemic velocity v_n of the same fit.

The primary offset is **Δv ≡ c(1/2) − v_n**, the displacement of the half-maximum bisector. The
peak of a multi-component model is unstable for flat-topped or two-humped profiles, and the
centroid weighs precisely the faint wings on which different decompositions of the same profile
legitimately disagree; we use c(1/2) as the primary reported center, while retaining the other definitions for comparison. The peak- and centroid-based
offsets are reported alongside, together with the peak of the continuum- and narrow-line-
subtracted data measured as in Eracleous et al. (2012) (a parabola through the smoothed profile
above 80 per cent of its maximum), which is compared with the model peak (flag `peak_disagree`).

## The model

The model follows the recipe of the SDSS quasar catalogues and of the earlier single-epoch
searches (Shen et al. 2011; Eracleous et al. 2012; Liu et al. 2014), with a small number of
departures forced by the host-dominated spectra of a DESI broad-line sample. Every line
component is a Gaussian in wavelength, F(λ) = A exp[−(λ − λ_c)²/2σ_λ²] with λ_c = λ₀(1 + v/c)
and σ_λ = λ_c σ/c.

**Pre-processing.** Galactic extinction is removed with the Cardelli, Clayton & Mathis (1989)
law and the O'Donnell (1994) optical coefficients, R_V = 3.1. A fractional error floor of 2 per
cent of the flux is added in quadrature: the narrow-line cores of bright galaxies reach per-pixel
signal-to-noise ratios of several hundred, where the residuals of any line model are limited by
the spectrophotometric calibration and would otherwise dominate χ² and drag the broad components.

**Continuum.** A power law A(λ/3000 Å)^α with −5 ≤ α ≤ 3, the optical Fe II template of Boroson &
Green (1992) and the ultraviolet composite of Vestergaard & Wilkes (2001), Salviander et al.
(2007) and Tsuzuki et al. (2006) assembled as in Shen et al. (2011) (each convolved to a free FWHM
between 1200 and 10 000 km/s, free normalisation, small velocity shift), and a host galaxy
built from up to five galaxy eigenspectra of Yip et al. (2004), fitted jointly by weighted least
squares in the line-free windows of Shen et al. (2011) plus every line-free pixel inside the
galaxy-template range (so that the 4000 Å break and the stellar absorption anchor the host).
Pixels deviating by more than 3σ below or 5σ above the first solution are masked once and the
fit repeated. The number of eigenspectra is stepped down (5 → 3 → 2 → 1 → 0) until the host is
non-negative everywhere; the host is kept only if it contributes at least 10 per cent of the
4200–5000 Å flux (Shen et al. 2011). Departure: the host contributes continuum only underneath
the emission lines. The eigenspectra are principal components of real galaxy spectra and contain
emission lines whose strengths the line-free fit does not constrain; subtracting the full host
displaced the systemic velocity by up to 450 km/s in strongly star-forming DESI hosts. The host
model within ±900 km/s of every catalogued line is therefore replaced by a linear interpolation.

**Narrow lines.** Two complexes are fitted after continuum subtraction: Hα (rest 6400–6800 Å:
narrow and broad Hα, [N II] λλ6548, 6584, [S II] λλ6716, 6731) and Hβ (4700–5100 Å: narrow and
broad Hβ, He II λ4686, [O III] λλ4959, 5007). Doublet ratios are fixed, [N II] 6584/6548 = 2.96
and [O III] 5007/4959 = 2.98; narrow widths are 25 ≤ σ ≤ 510 km/s (FWHM ≤ 1200). Narrow Hα and
[N II] share one velocity v_n and one width; [S II] has its own, tied softly by Gaussian priors
(σ within 20 per cent, v within 60 km/s), because a hard tie leaves residuals at [S II] in
high-signal-to-noise host galaxies that the fit absorbs with a spurious broad Gaussian at
+7000 km/s, and a free [S II] removes the anchor that defines the narrow width in quasars. A
narrow-line-region wing, a second Gaussian under every narrow line of the Hα complex with a
common amplitude fraction f_w ≤ 0.5 (weak prior σ_f = 0.25 towards zero), one velocity tied to v_n by
a Gaussian prior of width 150 km/s and one width between σ_n and 510 km/s, absorbs the non-Gaussian bases that AGN
narrow lines carry (Heckman et al. 1981; Whittle 1985). Without it, the information criterion
buys "broad" components of FWHM 1200–2000 km/s on those bases: in a synthetic spectrum with 25
per cent of the narrow flux in a base three times wider than the core, a broad Hα injected at
+800 km/s came back at +379 km/s without the wing and at +803 km/s with it. Pinned by [N II] and
[S II], which have no broad counterpart, the wing cannot masquerade as broad Hα; it is excluded
from every broad-profile measure. In the Hβ complex the [O III] doublet is a core (σ ≤ 800 km/s)
plus the blueshifted wing (σ 250–2500 km/s, v between −2500 and +500 km/s), ordered by two hinge
penalties (wing at least as wide as the core, core at least as tall as the wing) so that the two
cannot exchange roles; the core is started at the smoothed data peak near 5007 Å rather than at
zero velocity. When the Hα narrow lines are detected (peak S/N ≥ 3), the velocity and width of
narrow Hβ and He II and the wing are fixed to the Hα values; otherwise they are tied to the
[O III] core as in Shen et al. (2011).

**Systemic velocity.** v_n is the velocity of narrow Hα + [N II] (with [S II] attached), the
narrow-line system least affected by outflows and available for every object. It may lie within
±1500 km/s of the input redshift, because the pipeline redshift of a broad-line object is driven
by the broad lines themselves (SDSS J102106.04+452331.8 of Eracleous et al. 2012 has its narrow
lines 880 km/s from its catalogue redshift). Against the misassociation that the wide window
allows ([N II] λ6584 mistaken for narrow Hα displaces v_n by 940 km/s, and under a very broad Hα
the misassigned solution can have marginally lower χ²), a preliminary Hβ fit provides an [O III]
velocity from which the Hα group is started in addition to zero, and, when the [O III] core has
S/N ≥ 10, a Gaussian prior of 150 km/s pulls v_n towards it. Strong narrow lines override the
prior; a remaining disagreement above 400 km/s is flagged. On the sky the reference systems agree:
over the measurable objects of the DESI catalogue sample, [S II] deviates from v_n by +3 ± 17 km/s
and the [O III] core (where it has S/N ≥ 10) by −4 ± 28 km/s (median ± NMAD, catalogue run).

**Broad line.** One to three Gaussians with 1200 ≤ FWHM ≤ 15 000 km/s and centres within
±8000 km/s of the line (individual components of the disk-like profiles of Eracleous et al. 2012
reach −6300 km/s; a ±5000 km/s bound truncated the wings of double-peaked profiles and roughly
halved their widths). Only their sum is measured. The number of components is chosen with the
Bayesian information criterion, BIC = χ² + k ln N: the fewest components unless a more complex
model improves the BIC by more than 10. Each fit is repeated from starting velocities 0, ±1500
and ±3000 km/s of the first component (and, for Hα, from two starts of the narrow group), and the
lowest-χ² solution is kept; the solver is scipy's bounded trust-region least squares. A hinge
penalty enforces FWHM ≥ 0.4 |v|, so that a component at +7000 km/s must be at least 2800 km/s
wide and cannot be parked on [S II] (+7020 and +7680 km/s from Hα) or on [O III] λ4959 (+6020
km/s from Hβ) to absorb narrow-line residuals, while genuine disk-emitter components (FWHM
3000–5000 at |v| 5000–8000 km/s) are untouched. A complex is fitted only if the data extend
3500 km/s beyond the line on both sides, the broad centres are confined to the covered range
less 1000 km/s, and coverage below ±6000 km/s is flagged `edge` (Hα at z ≳ 0.477 runs off the red
end of DESI; unconstrained fits placed components outside the data and manufactured offsets of
+4000 to +8000 km/s).

## Classes and flags

The rules are applied in the order listed: a profile takes the first class whose condition
holds. Class A uses the Monte Carlo center error when that error is finite, and the
300 km/s threshold alone otherwise. A class can therefore differ between point-only
and MC runs. The RC1 primary, DESI-night and staged SDSS campaigns used `nmc=0`;
they do not contain a newly calibrated uncertainty catalogue. The labels describe
the fitted profiles and do not establish orbital motion.

| Class | Rule |
|---|---|
| **E** no broad line | FWHM < 1200 km/s, or integrated S/N < 5, or peak S/N < 1.5 |
| **X** no systemic | narrow-line reference not detected (S/N < 3) |
| **W** not classifiable | FWHM < 2000 km/s or integrated S/N < 8: something broad is there, but narrow-line residuals dominate such profiles |
| **B** double-peaked / disk-like | two resolved peaks separated by > max(0.4 FWHM, 1500 km/s) with a dip > 8 per cent and FWHM ≥ 3000; or FWHM ≥ 7000 with \|A.I.\| ≥ 0.20 or K.I. ≥ 0.50 |
| **A** bulk shift | \|Δv\| > 300 km/s at > 3σ (when an error is available) and a symmetric profile: \|c(1/4) − c(3/4)\| < 0.10 FWHM, \|A.I.\| < 0.12, \|v_peak − centroid\| < 0.20 FWHM |
| **C** asymmetric | single-peaked and not symmetric |
| **F** normal | symmetric, no significant offset |

B and C are descriptive shape classes. This code does not fit a disk model or
identify a unique physical explanation for either class.

| Flag | Condition |
|---|---|
| `very_broad` | FWHM ≥ 8000 km/s |
| `poor_fit` | reduced χ² ≥ 2.5 |
| `low_snr` | integrated broad S/N < 10 |
| `low_peak_snr` | broad peak < 5σ per pixel: a component significant only by integration over thousands of km/s is degenerate with continuum-subtraction residuals |
| `host_dominated` | host ≥ 80 per cent of the 4200–5000 Å light: template mismatch at the few-per-cent level mimics a very broad line |
| `pl_at_bound` | power-law slope ≤ −4.9 or ≥ 2.9, within 0.1 of the bounds −5 and 3 |
| `peak_disagree` | model peak and data peak differ by > 0.25 FWHM |
| `sii_disagree` | [S II] velocity > 150 km/s from v_n |
| `sys_disagree` | v_n and the [O III] core > 400 km/s apart ([O III] S/N ≥ 5) |
| `narrow_at_bound` | \|v_n\| ≥ 1425 km/s, within 5% of the ±1500 km/s search boundary |
| `edge` | data cover less than ±6000 km/s around the line |
| `extreme_offset` | \|Δv\| > 4000 km/s for classes A, C and F; a project quality threshold, not a universal physical ceiling |

The absence of a flag does not guarantee a valid measurement: E, W and X return
before these quality checks, and solver/MC diagnostics must also be inspected.
The project's selection definitions are provided as functions:
*measurable* = class A/B/C/F, integrated S/N ≥ 8, FWHM ≥ 2000 km/s, no `edge`; *strong offset* =
measurable and 1000 ≤ |Δv| ≤ 4000 km/s. Both lines are always fitted when the data cover them;
the Halpha and Hbeta measurements are returned separately. Hbeta can inherit its
narrow-line reference from Halpha, so the two offsets are not necessarily independent.

## Errors

**Monte Carlo** (`--nmc 30`, `fit_spectrum(nmc=30)`): the spectrum is perturbed with Gaussian
noise from its supplied pixel-error array, before the 2% fitting floor, and refitted with the host model held fixed and the number of broad
components fixed to the selected one; the error is half the 16th–84th percentile range (as in
Shen et al. 2013 and Liu et al. 2014). This is the conditional statistical error only.
The fitting weights still include the floor; this change adds no model component.
The fixed 2,640-spectrum study with 200 draws passed all 24 pooled point-accuracy
checks and all 16 center-coverage checks. Four width-coverage checks and one
flux-coverage check remained inconclusive: 67 pass, five inconclusive overall.
This supports only the declared synthetic conditions and measurable-line selection;
it is not universal survey calibration. The older 30-draw evidence is retained
separately. See [uncertainty scope](docs/UNCERTAINTIES.md) and
[current validation status](docs/VALIDATION_STATUS.md).
MC can change error-dependent classes while leaving ordinary fitted profiles,
offsets and widths unchanged.
Use `--mc-noise-policy effective` (or `mc_noise_policy="effective"`) to reproduce
the historical perturbations including the floor. Those errors are labelled
`conditional_effective_noise`. Recomputing MC on an old saved result without a
noise-policy setting keeps that historical policy explicitly. Input-noise MC
requires the saved `ivar_stat_rest`; it never silently substitutes floored weights.

**Historical empirical diagnostic.** Earlier versions reported a DESI-repeat error
formula in `dv_err_model`, including for other instruments. This interface leaves
that field null. `--legacy-error-diagnostic` retains the original formula only for
DESI, under `legacy_desi_repeat_error_diagnostic`; it is not an RC1 or SDSS error
calibration. See the validation status for the supported scope and remaining limitations.

## Validation and release scope

The RC1 numerical core has completed the project's scoped DESI and SDSS point-fit
runs. Software checks, real-data execution and statistical coverage are separate
forms of evidence. No test guarantees that every astrophysical decomposition is unique.

The latest known-truth study retained all 2,640 spectra with no execution errors.
Its 72 criteria yielded 67 passes and five inconclusive results; no criterion
failed under the fixed rules. Halpha was measurable in 579/660 low-S/N DESI-like
cases and 603/660 low-S/N SDSS-like cases; the other six line/grid/SNR groups each
had 660/660 measurable cases. Coverage is conditional on those selections.
The [validation status](docs/VALIDATION_STATUS.md),
[uncertainty guide](docs/UNCERTAINTIES.md), and
[reproduction materials](validation/uncertainty_20261003/README.md) retain the
full denominators, thresholds and limitations. The earlier 1,980-spectrum study
remains separate; it is not relabelled as a pass by the newer result.

This is a single-spectrum software release with explicit scientific limits.
The CLI still defaults to point fits (`--nmc 0`). Existing catalogue/viewer
products have not acquired new error bars through publication of this release.

Between-epoch velocity routines and Mg II remain experimental. The velocity
routines (`blrfit.rv`, from Python only) are retained for reproducibility and
development; their reported uncertainties do not establish significant motion.
Earlier numerical examples are preserved in
[historical validation notes](docs/HISTORICAL_VALIDATION.md).

## What the tool does not do

* It does not fit disk models: classes B and C are shape classes, not physical ones.
* It does not measure redshifts: z is an input (from the file where available), and the narrow
  lines are then sought within ±1500 km/s of it.
* It does not calibrate fluxes: luminosities and equivalent widths inherit the input units and
  the luminosity assumes 1e-17 erg s⁻¹ cm⁻² Å⁻¹ (Planck 2018 cosmology through astropy).
* It does not judge binarity. A displaced broad line is a candidate; the interpretation needs
  the multi-epoch and profile-stability arguments of the papers cited above.
* Mg II is implemented (one narrow line inside ±700 km/s, broad line as above) but was not used
  for the DESI catalogue and is not validated to the same level.
* Narrow-line pedestals wider than the wing model (σ > 510 km/s, i.e. FWHM > 1200 km/s, the
  narrow/broad boundary) are absorbed by the broad components: in the synthetic suite a pedestal
  of σ = 750 km/s under a weak broad Hα at +800 km/s gave a two-peaked class-B profile with
  c(1/2) 130–230 km/s low.

## Output

`<stem>_fit.json` holds `blrfit_version`, `input` (`path`, `kind`, `z` and `z_source`, `ebv` and
`ebv_source`, coordinates, date), `settings`, `continuum` (power-law slope, Fe II, host fraction),
`o3_prefit`, `lines` (per line: `label`, `reasons`, `flags`, `measurable`, `strong_offset`, `dv` =
c(1/2) − v_n, `dv_err_mc`, `dv_err_model`, the measures of the profile and of the narrow lines, and
`fitted = false` with an empty label and the reason when the window is not covered) and
`summary_row`, the flat dictionary of the catalogue (`HA_*`, `HB_*`, `conti_*`). `<stem>_fit.png`
is the diagnostic figure; `--pickle` writes the full result. A fit
returns zero when it completes, including explicitly uncovered lines. Input/fit failures
return nonzero; a public batch also returns nonzero when a download or individual fit fails.

`conti_pl_norm` (the power law at 3000 Å rest) and `conti_feop_norm` are flux densities of the
(1+z)-scaled rest-frame spectrum the fit works on, f_rest(λ_rest) = (1+z) f_obs(λ_obs) at
λ_rest = λ_obs/(1+z), in units of 1e-17 erg s⁻¹ cm⁻² Å⁻¹ per rest-frame ångström (the factor
(1+z) is the wavelength Jacobian). The monochromatic continuum luminosity is therefore
λL_λ(5100) = 4π D_L² × 5100 Å × f_rest(5100) × 1e-17 erg/s with f_rest(5100) =
`conti_pl_norm` (5100/3000)^`conti_pl_alpha` and no further redshift factor; dividing by (1+z)
once more understates it by log10(1+z) dex. `blrfit.continuum_luminosity(res)` (and
`blrfit.lambda_l_lambda(conti, z)` for a `conti` dictionary or a summary row) implements this
with the Planck 2018 luminosity distance, the same one that `broad_lum` uses.

## Citing

If you use blrfit, please cite the software (`CITATION.cff`) and the methodological
references relevant to the quantities you use. The project manuscript is in
preparation; it is not a published validation reference. The Fe II templates and the galaxy eigenspectra were obtained from the
PyQSOFit repository (Guo, Shen & Wang 2018; GPL-3.0 code licence) and are the published data of
the authors listed in `blrfit/templates/README.md`; the MIT licence of this package covers its
code.

## References

Boroson T. A., Green R. F. 1992, ApJS, 80, 109 ·
Cardelli J. A., Clayton G. C., Mathis J. S. 1989, ApJ, 345, 245 ·
Eracleous M., Halpern J. P. 1994, ApJS, 90, 1 ·
Eracleous M., Boroson T. A., Halpern J. P., Liu J. 2012, ApJS, 201, 23 ·
Guo H., Shen Y., Wang S. 2018, PyQSOFit, ascl:1809.008 ·
Guo H. et al. 2019, MNRAS, 482, 3288 ·
Heckman T. M., Miley G. K., van Breugel W. J. M., Butcher H. R. 1981, ApJ, 247, 403 ·
Liu X., Shen Y., Bian F., Loeb A., Tremaine S. 2014, ApJ, 789, 140 ·
Marziani P., Sulentic J. W., Dultzin-Hacyan D., Calvani M., Moles M. 1996, ApJS, 104, 37 ·
Morton D. C. 1991, ApJS, 77, 119 (air–vacuum conversion, as used by SDSS) ·
O'Donnell J. E. 1994, ApJ, 422, 158 ·
Planck Collaboration 2020, A&A, 641, A6 (the cosmology of astropy's `Planck18`) ·
Runnoe J. C. et al. 2015, ApJS, 221, 7 ·
Runnoe J. C. et al. 2017, MNRAS, 468, 1683 ·
Salviander S., Shields G. A., Gebhardt K., Bonning E. W. 2007, ApJ, 662, 131 ·
Shen Y. et al. 2011, ApJS, 194, 45 ·
Shen Y., Liu X., Loeb A., Tremaine S. 2013, ApJ, 775, 49 ·
Storey P. J., Zeippen C. J. 2000, MNRAS, 312, 813 ·
Strateva I. V. et al. 2003, AJ, 126, 1720 ·
Tsuzuki Y., Kawara K., Yoshii Y., Oyabu S., Tanabé T., Matsuoka Y. 2006, ApJ, 650, 57 ·
Vestergaard M., Wilkes B. J. 2001, ApJS, 134, 1 ·
Whittle M. 1985, MNRAS, 216, 817 ·
Yip C. W. et al. 2004a, AJ, 128, 585 (galaxy eigenspectra); 2004b, AJ, 128, 2603 (quasar eigenspectra)

## Licence

MIT. Copyright (c) 2026 Ekaterine Dadiani.
