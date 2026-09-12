# Test fixtures

| File | Content | Provenance |
|---|---|---|
| `pins.json` | The output of the production fitter (summary row, fitted parameters, chi-square, BIC; `produced_by` block inside) for four SDSS spectra of Liu et al. (2014) objects, fitted at their redshifts with E(B−V) = 0: SDSS J001224.01−102226.5 (spec-0651-52141-0072, classes C/C), J101000.54+074235.5 (spec-1237-52762-0298, Hβ class A, −2131 km/s), J091833.82+315621.1 (spec-1592-52990-0139, Hβ class B) and J140827.51+142233.1 (spec-1704-53178-0562, classes F/F) | written on 2026-09-11 with numpy 1.26.4 / scipy 1.13.1 (macOS arm64) |
| `spec-1237-52762-0298.fits`, `spec-1592-52990-0139.fits`, `spec-1704-53178-0562.fits` | SDSS DR7 spectra (reduction 26) of the last three objects, reduced to the PRIMARY, COADD, SPECOBJ and SPZLINE extensions (the per-exposure extensions of the archive file are dropped; the coadded spectrum and redshift are unchanged); 173–176 KB each | https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/PPPP/spec-PPPP-MMMMM-FFFF.fits |
| `healpix_reference.json` | 200 random positions and 8 boundary positions with their nside 64 nested HEALPix indices | healpy 1.19.0, `ang2pix(64, ra, dec, lonlat=True, nest=True)` |

The first pinned spectrum and the DESI, CSV and second-epoch examples live in `examples/data`.
The Liu et al. (2014) anchor test needs the full set of 388 spectra (300 MB), which is not part
of the repository; see `tests/test_anchor_liu.py`.
