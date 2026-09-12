# Spectral templates

The four files were obtained from the PyQSOFit repository (Guo, Shen & Wang 2018;
https://github.com/legolason/PyQSOFit, whose code is distributed under the GPL-3.0 licence),
which collected them from the original publications listed below. They are redistributed here
unchanged so that the fits are reproducible; they are the published data of the cited authors
and remain under those authors' terms. The MIT licence of this package covers its code.

| File | Content | Source |
|---|---|---|
| `fe_optical.txt` | Optical Fe II template of I Zw 1, columns log10(wavelength/Å), flux; 3686–7484 Å; intrinsic width 900 km/s FWHM | Boroson & Green (1992), ApJS 80, 109 |
| `fe_uv.txt` | Ultraviolet Fe II template, same columns, 1075–3499 Å (the fit uses 1200–3500 Å). A composite, as described in the file header and in Shen et al. (2011): 1000–2200 Å from Vestergaard & Wilkes (2001); 2200–3090 Å from Salviander et al. (2007), an extrapolation of the same template beneath Mg II; 3090–3500 Å from Tsuzuki et al. (2006) | Vestergaard & Wilkes (2001), ApJS 134, 1; Salviander et al. (2007), ApJ 662, 131; Tsuzuki et al. (2006), ApJ 650, 57 |
| `gal_eigenspec_Yip2004.fits` | Galaxy PCA eigenspectra (columns WAVE, PCA), 3450–8000 Å | Yip et al. (2004a), AJ 128, 585 |
| `qso_eigenspec_Yip2004_global.fits` | Quasar PCA eigenspectra, global set | Yip et al. (2004b), AJ 128, 2603 |

`blrfit` uses the two Fe II templates and the first five galaxy eigenspectra. The quasar
eigenspectra are loaded for reference only; the continuum model does not use them (see
`blrfit/model/continuum.py` for the reason).
