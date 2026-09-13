# Examples

All commands run from this directory; outputs go to `output/`.

| File | What it is |
|---|---|
| `data/spec-0651-52141-0072.fits` | SDSS DR7 spectrum (2001-08-20) of SDSS J001224.01−102226.5, z = 0.2288, a binary candidate of Eracleous et al. (2012) with broad Hβ displaced by −1952 km/s in the Liu et al. (2014) measurement |
| `data/spec-7169-56628-0344.fits` | BOSS spectrum (2013-12-02) of the same object: a second epoch 12 years later |
| `data/coadd-main-dark-17260-39627574082538900.fits` | DESI DR1 healpix coadd (main survey, dark program, healpix 17260) reduced to TARGETID 39627574082538900 = SDSS J001247.93−084700.5, z = 0.2203, with the matching `redrock-*.fits` next to it. Written by `blrfit fetch`; the full public file is 413 MB |
| `data/spec-0652-52138-0326.fits`, `data/spec-7169-56628-0665.fits` | the two SDSS epochs of the DESI example object (2001 and 2013) |
| `data/J001224_rest_air_nm.csv` | the 2001 spectrum of J001224 rewritten as a generic table: rest-frame air wavelengths in nm, flux in 1e-16 erg/s/cm²/Å, 1σ errors |

```bash
mkdir -p output

# 1. an SDSS spectrum; the redshift is the one Liu et al. (2014) used
blrfit fit data/spec-0651-52141-0072.fits --z 0.2288 --nmc 30 --out output

# 2. a DESI coadd: the redshift comes from the redrock file next to it, E(B-V) from the FIBERMAP
blrfit fit data/coadd-main-dark-17260-39627574082538900.fits --targetid 39627574082538900 --out output

# 3. a generic table with declared columns, units and frame
blrfit fit data/J001224_rest_air_nm.csv --wave lambda_nm --flux f_lambda --err sigma \
    --wave-unit nm --frame rest --air --z 0.2288 --flux-scale 10 --lines Halpha,Hbeta --out output

# 4. the velocity change of broad Hbeta between the two SDSS epochs of J001224
blrfit rv data/spec-0651-52141-0072.fits data/spec-7169-56628-0344.fits --z 0.2288 --line Hbeta --out output

# 5. DESI (2021) against SDSS (2001) for J001247, Halpha; the SDSS spectrum is regridded onto the DESI grid
blrfit rv data/spec-0652-52138-0326.fits data/coadd-main-dark-17260-39627574082538900.fits \
    --targetid 39627574082538900 --z 0.2203 --line Halpha --out output

# 6. everything public at a position (needs blrfit[fetch] and the network)
blrfit fetch --ra 3.1997083 --dec -8.7834722 --out output/spectra
```

Expected results (blrfit 0.1.0 with numpy 1.26.4 and scipy 1.13.1 on macOS; other numerical stacks move the
bisector velocities by up to about 20 km/s, the peaks and centroids by up to about 70 km/s and the widths by
up to about 25 km/s): for J001224 (2001) Hα is class C with
Δv = −1061 km/s and Hβ class C with Δv = −1244 km/s, both strong offsets with no flag; the 2013
epoch is shifted by −294 ± 39 km/s in Hβ relative to 2001 (zero-point −13 km/s), with a profile
change (z_prof ≈ 10, the FWHM of our fits drops from about 3390 to about 2930 km/s) that places
the pair outside the reliable tier. With `--nmc 30` the Monte Carlo errors of the 2001 spectrum
are ±287 km/s (Hα) and ±141 km/s (Hβ), well above the repeat-spectrum model (±68 and ±92): the
realisations switch between three-Gaussian decompositions of this asymmetric profile. The CSV
(example 3) gives the same velocities and classes as example 1 (it is the same spectrum); without
`--flux-scale 10` only the luminosities would be ten times low. Fitting example 1 with the file's
own redshift (0.2244, 1035 km/s from the narrow lines) instead of 0.2288 changes the decomposition
(Hα Δv −1041, Hβ −1262 km/s, FWHM 4550): the offsets are stable, the components are not. For
J001247 (DESI, 2021) Hα is class F with Δv = −37 km/s and no flag; its Hβ is class C at about
+90 km/s with the `poor_fit` flag (a narrow unmodelled feature near 5030 Å rest). With `--line Halpha,Hbeta` the tool also grades the profile change (both lines: grade
"changed"; residuals of about 7 per cent of the peak) and uses the cross-survey floor of that
grade, which makes the shifts 0.8σ (Hα, −219 ± 266) and 1.3σ (Hβ, −282 ± 224); the two lines
agree within 1.6σ (two-line criterion satisfied). Example 5 gives a
shift of +67 ± 11 km/s (±155 with the DESI floor) of the DESI spectrum relative to the 2001 SDSS
spectrum in Hα, regridded, directions agreeing, z_prof 5.5, hence outside the reliable tier.
