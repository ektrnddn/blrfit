# Examples

The spectra in `data/` are public SDSS and DESI spectra of two quasars. The first,
J001224, is a candidate of Eracleous et al. (2012) whose broad Hβ Liu et al. (2014)
measured 1952 km/s blueward of its narrow lines. Run the commands from this directory.

| File | Content |
|---|---|
| `spec-0651-52141-0072.fits`, `spec-7169-56628-0344.fits` | SDSS J001224.01−102226.5, z = 0.2288, observed in 2001 and 2013 |
| `J001224_rest_air_nm.csv` | the 2001 spectrum as a table: rest-frame air wavelengths in nm, flux in 10⁻¹⁶ erg s⁻¹ cm⁻² Å⁻¹, 1σ errors |
| `coadd-main-dark-17260-39627574082538900.fits`, `redrock-...fits` | SDSS J001247.93−084700.5, z = 0.2203: its DESI DR1 coadd, reduced to this target, with the redshift file |
| `spec-0652-52138-0326.fits`, `spec-7169-56628-0665.fits` | the same object observed by SDSS in 2001 and 2013 |

```bash
# an SDSS spectrum at the redshift of Liu et al. (2014)
blrfit fit data/spec-0651-52141-0072.fits --survey sdss --z 0.2288 --out output

# the DESI coadd: the redshift comes from the redrock file, E(B-V) from the FIBERMAP
blrfit fit data/coadd-main-dark-17260-39627574082538900.fits --targetid 39627574082538900 --out output

# the 2001 spectrum as a table, declaring its units and frame
blrfit fit data/J001224_rest_air_nm.csv --survey generic --wave lambda_nm --flux f_lambda --err sigma \
    --wave-unit nm --frame rest --air --z 0.2288 --flux-scale 10 --out output/table

# all SDSS spectra at once, with one catalogue table
blrfit fit data/spec-*.fits --survey sdss --out output/all

# Monte Carlo errors (slower)
blrfit fit data/spec-0651-52141-0072.fits --survey sdss --z 0.2288 --nmc 200 --out output/mc

# the two epochs of J001224: fits, the velocity change of each broad line, the candidate tier
blrfit pair data/spec-0651-52141-0072.fits data/spec-7169-56628-0344.fits --survey sdss --z 0.2288 --out output/pair

# the public spectra of the DESI target, downloaded (needs the fetch extra)
blrfit fit --targetid 39627574082538900 --include-sdss --out output/public
```

The `spec-*.fits` batch runs at the redshift in each file. The same steps in Python are
in [quickstart.ipynb](quickstart.ipynb); the model, the outputs and the validation are
described in [docs](../docs/method.md).
