# Public and local spectrum inputs — development interface

This change builds on commit `3cd37d06c8e5959efa8b0bb41458629fc95a689b`
(`v0.2.0rc1`). The numerical model, optimizer, classification and Monte Carlo
implementation are unchanged. This is a development interface, not a new
scientific calibration or approval of between-epoch measurements.

## Compatibility and scope

- `fit` defaults to DESI. Use `--survey sdss` or `--survey generic` for other
  inputs. `--survey auto` restores the legacy filename-based dispatch.
- `fit --targetid ID` or `fit --ra RA --dec DEC` searches public DESI DR1.
  `--include-sdss` adds a direct SDSS DR17 SpecObjAll cone query, limited to
  1.5 arcsec. `fetch` uses the same discovery without fitting. The old CLI
  `--no-sdss`/`--no-desi` switches are replaced by these explicit choices.
- DESI lookup joins `desi_dr1.zpix` with `desi_dr1.photometry` by exact integer
  TARGETID, or the corresponding EDR tables. Indexed HEALPix locations avoid
  assigning an object's product from the query coordinate's pixel alone.
  Product membership is then checked in the downloaded FIBERMAP. Searches
  require the public catalogue mirror and archive to be available.
- SDSS repeat products are retained by plate/MJD/fiber; the `specobjall`
  table is used rather than the `specobj` view or a nearest-only crossmatch.
  Its catalogue flags and positional association are not a validation of
  the object's identity or pipeline redshift. This is not a DR20 census.
- Coadds returned by these services may combine multiple observing nights.
  Fitting them is not a replacement for the project's separately constructed
  nightly inputs. Per-exposure splitting and nightly construction remain
  outside this interface.
- SDSS spectral tables with `loglam`, `flux`, `ivar`, and a native mask can
  be read regardless of filename, including compatible newer products.
  The CLI excludes nonzero MASK/AND_MASK; the Python reader keeps its legacy
  `mask_policy="ivar"` default. No additional motion correction or line-spread
  function deconvolution is introduced.
- General input supports scalar-column or explicitly selected vector-row
  FITS tables, image arrays in named/numbered HDUs, CSV/ECSV/text. A noise
  array is mandatory. Wavelength and flux-density frames are separate
  declarations; rest-frame fλ and its errors are divided by (1+z) when
  `flux_frame="rest"`. No fν or luminosity-density conversion is inferred.

## Error reporting

Default `nmc=0` means no MC error calculation. `dv_err_mc` stays null when
uncomputed; the JSON records `uncertainty.status="not_computed"`.
With MC requested, errors retain the RC1 conditional assumptions and remain
uncalibrated for general scientific use. The per-line MC output and its
diagnostics determine whether an error is available for that line.

The historical DESI-repeat formula is no longer exposed as a calibrated
`dv_err_model`, especially for SDSS. The legacy key remains null for schema
compatibility. An explicit DESI-only option places it in
`legacy_desi_repeat_error_diagnostic`. It changes neither the core fit nor
classification. Original uncertainty validation outcomes are unchanged:
[RC1 validation status](VALIDATION_STATUS.md).

Missing continuum support is still a fit failure. In public batches it is
recorded against that product while the remaining spectra can finish. The
interface does not turn a failed input into a successful measurement.

## Verification

Offline tests cover exact IDs above 2^53, coordinate wraparound and the radius
cut, multiple-object ambiguity, repeated visits, schema/service failures,
vector-row selection, image-HDU mapping, masks, units and redshift conflicts.
The existing reader, CLI, fetch-provenance and physics-I/O tests also run.
Tests do not seek agreement with historical fits as scientific truth.

Live checks on 2026-10-03 resolved DESI TARGETID `39627574082538900` by ID
and position, downloaded its DR1 spectrum and the two matching DR17 SDSS
products, and fitted Hα/Hβ in all three using point fits. This verifies those
service routes and products, not every public release, instrument or layout.

## Service documentation

- [Data Lab DESI catalogue and data access](https://datalab.noirlab.edu/data/desi)
- [Data Lab SDSS tables, including SpecObjAll](https://datalab.noirlab.edu/data/sdss)
- [Data Lab TAP radial-query syntax](https://datalab.noirlab.edu/help/index.php?qa=770&qa_1=appears-radial-function-recognized-accessing-catalogs-through)
- [SDSS DR17 spectral downloads](https://www.sdss4.org/dr17/data_access/bulk/)

Use the survey acknowledgement and citation requirements when publishing
results based on the downloaded data.
