# Single-spectrum examples

Run these commands from this directory after installing the current package.
The example FITS/CSV files are included in this repository. Each command uses a
separate output directory; the default is a point fit (`nmc=0`).

```bash
# Local SDSS spectrum, with an explicit input redshift:
blrfit fit data/spec-0651-52141-0072.fits --survey sdss --z 0.2288 --out output/sdss

# Local DESI coadd; the matching redrock file supplies its redshift:
blrfit fit data/coadd-main-dark-17260-39627574082538900.fits --targetid 39627574082538900 --out output/desi

# The SDSS spectrum represented as a generic table:
# Wavelength is rest-frame air nm; flux remains observed-frame f_lambda,
# in units of 1e-16 erg/s/cm²/Å. Scaling by 10 also scales its statistical error.
blrfit fit data/J001224_rest_air_nm.csv --survey generic --wave lambda_nm --flux f_lambda --err sigma \
    --wave-unit nm --frame rest --air --z 0.2288 --flux-scale 10 --out output/table

# Public DESI DR1 lookup by exact TARGETID (requires the fetch extra):
blrfit fit --targetid 39627574082538900 --out output/public_desi

# Search the public DESI/SDSS routes without fitting:
blrfit fetch --ra 3.1997148876 --dec -8.7834904331 --include-sdss --out output/downloaded
```

Local SDSS input uses the conservative mask policy by default. The generic CSV
has statistical errors but does not encode that native mask, so its usable pixels
need not match the default SDSS reading. To reproduce the older IVAR-only input
selection, explicitly use `--sdss-mask-policy ivar` on the SDSS command. Matching
input conventions matters when comparing results; equal catalogue values are not
a scientific acceptance criterion.

For conditional MC estimates, add `--nmc 30 --seed 0`. These retain the
[documented limitations](../docs/UNCERTAINTIES.md); requesting MC does not guarantee
that every line has a finite, calibrated error. The
[main guide](../README.md) documents generic FITS columns, rows and image HDUs,
units, frames, redshift, extinction and statistical-noise requirements.

The public lookup returns available indexed DESI DR1/EDR coadds and SDSS DR17
products. It does not reconstruct all observing nights or search every release.
A coadd may span several nights; products with shared exposures must not be
interpreted as independent temporal measurements.

Between-epoch routines remain experimental and are outside these single-spectrum
examples. The older example commands and numerical outputs are retained in
[historical results](HISTORICAL_RESULTS.md); their reported uncertainties and
reliability labels are not current validation claims.
