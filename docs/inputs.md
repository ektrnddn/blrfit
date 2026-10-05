# Inputs

blrfit reads DESI coadds, SDSS spectra, and spectra stored in tables or FITS images, and
downloads public DESI and SDSS spectra. Whatever the source, a fit needs observed-frame
wavelengths, a flux density f_λ in 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹, a per-pixel error and a
redshift, and the spectrum must extend far enough on both sides of the lines to
constrain the continuum.

## DESI coadds (the default)

```bash
blrfit fit coadd-main-dark-17260.fits --targetid 39627574082538900 --out fits
```

- A file holding several targets needs `--targetid`; a file with one target does not.
- The redshift comes from the `redrock-*.fits` file next to the coadd, or from
  `--redrock FILE`; otherwise give `--z`.
- E(B−V) comes from the FIBERMAP.
- A coadd combines all exposures of a target in one survey and program, which may span
  several nights. Rows that repeat a TARGETID are rejected rather than combined.

## SDSS spectra

```bash
blrfit fit spec-0651-52141-0072.fits --survey sdss --z 0.2288 --out fits
```

The contents of the file, not its name, identify it: any table with `loglam`, `flux`,
`ivar` and a pixel mask is read, including later SDSS products. The coadded spectrum
(extension 1) is fitted on its native vacuum wavelength grid. Pixels with a nonzero
`and_mask` are excluded on the command line; in Python, `read_sdss` and `read_spectrum`
exclude them with `mask_policy="conservative"` (their default, `"ivar"`, keeps them, as
versions before 0.2 did). The redshift comes from the file unless `--z` is given.

## Tables and FITS images

```bash
# a FITS table, CSV or ECSV with named columns, including the error column
blrfit fit spectrum.fits --survey generic --wave WAVE --flux FLUX --ivar IVAR --z 0.2 --out fits
# one row of a table whose cells are spectra, with its redshift column
blrfit fit spectra.fits --survey generic --row 2 --wave LAMBDA --flux FLUX --err SIGMA --z-column Z --out fits
# arrays in separate image extensions, with the redshift in a header keyword
blrfit fit arrays.fits --survey generic --wave-hdu WAVE --flux-hdu FLUX --err-hdu ERROR --z-key Z --out fits
```

- A per-pixel error is required: `--err` (1σ) or `--ivar` (inverse variance), as a column
  or an image extension. The fit is weighted and does not invent errors.
- Wavelengths are observed-frame vacuum ångström by default; `--wave-unit nm|um|m`,
  `--frame rest` and `--air` declare otherwise.
- The flux must be f_λ per observed ångström in 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹; `--flux-scale F`
  multiplies flux and error into that unit. A flux per rest-frame ångström is declared
  with `--flux-frame rest`; flux and error are then divided by 1 + z. Nothing converts
  from f_ν or from a luminosity density.
- `--mask COLUMN` excludes the pixels where that column is nonzero.
- Images described only by a WCS, data cubes and arrays holding several spectra without
  a `--row` are not read.

## Public spectra

```bash
blrfit fit --targetid 39627574082538900 --out result                # DESI DR1, by TARGETID
blrfit fit --ra 3.19971 --dec -8.78349 --include-sdss --out result   # DESI and SDSS, by position
blrfit fetch --targetid 39627574082538900 --include-sdss --out spectra  # download only
```

These need the `fetch` extra and network access.

- **DESI.** Targets are found in the public catalogue of the
  [Astro Data Lab](https://datalab.noirlab.edu/data/desi) (DR1 by default; `--releases edr`
  or `dr1,edr`), by exact TARGETID or within `--desi-radius` (1.5″) of a position in ICRS
  degrees. The coadd of every survey and program the target appears in is fitted
  separately; only the target's rows are transferred from the archive. When several
  TARGETIDs match a position, the search stops and lists them: choose one with
  `--targetid`, or keep all with `--all-matches`.
- **SDSS.** With `--include-sdss` (or `--survey sdss`), DR17 spectra within `--radius`
  (at most 1.5″) are found in the `specobjall` table and downloaded from the SDSS archive;
  repeat observations are all kept. They are matches by position: check the identity and
  redshift of each.
- A failed query is reported as an error, not as "no spectrum". Use a new output
  directory for each search; the manifests record every candidate and every file, with
  checksums. Acknowledge DESI and SDSS as their data policies require when you publish
  results based on their spectra.

## Several spectra

```bash
blrfit fit spectra/*.fits --survey sdss --out fits
blrfit fit --list targets.csv --jobs 8 --out fits
```

A list is a text file with one spectrum per line, optionally followed by its TARGETID
(`#` starts a comment), or a table (`.fits`, `.ecsv`, `.csv`) with a column `path` and,
optionally, `targetid` and `z`. Relative paths are taken from the current directory.
Each spectrum is fitted as it would be alone. The command prints one line per spectrum
and writes the catalogue table `fits/blrfit_summary.fits` (or `--table name.ecsv`), with
failed spectra kept and their reasons. `--jobs N` fits N spectra at a time or, for a
single spectrum, refits its Monte Carlo draws in N processes; the results are the same
for any N.

## Redshift

The redshift is an input, not measured: it comes from the redrock file (DESI), the
spectrum file (SDSS; `--z-column` or `--z-key` for tables) or `--z`. The narrow lines are
sought within ±1500 km/s of it.

## Galactic extinction

DESI coadds carry the SFD98 E(B−V) of each target in their FIBERMAP. For SDSS and
generic inputs, blrfit looks up SFD98 at the coordinates of the file (or `--ra`, `--dec`)
when the `dust` extra and its map are installed (fetch the map once with
`python -c "import dustmaps.sfd; dustmaps.sfd.fetch()"`); otherwise it uses 0, prints a
warning and flags every fitted line `ebv_assumed_zero`. `--ebv VALUE` overrides both;
`--ebv sfd` requires the map. In Python, `fit_spectrum(ebv=...)` takes the value you give.

## Python

```python
import blrfit

sp = blrfit.read_spectrum("spec-0651-52141-0072.fits", survey="sdss", mask_policy="conservative")
res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z=0.2288, ebv=0.0)
row = blrfit.summary_row(res)  # the flat row described in outputs.md
fig = blrfit.plot_fit(res)
```

`read_spectrum` returns the wavelength, flux and inverse variance with, when the file has
them, the redshift, coordinates and date. `read_desi` and `read_table` read the other
formats directly.
