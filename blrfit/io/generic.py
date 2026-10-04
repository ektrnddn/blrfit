"""
Any spectrum given as a table: FITS binary table, CSV, ECSV or whitespace text.

The caller states which columns hold the wavelength, the flux and either the
1-sigma error or the inverse variance, and in what units and frame they are.
Everything is converted to what the fitter expects: observed-frame vacuum
wavelength in Angstrom, flux in any linear unit, and inverse variance.

* Wavelength units: 'angstrom' (default), 'nm', 'um' or 'm'.
* Frame: 'obs' (default) or 'rest'; a rest-frame wavelength is multiplied by
  (1 + z).
* Air wavelengths are converted to vacuum with the IAU/SDSS relation of
  Morton (1991), vac = air (1 + 2.735182e-4 + 131.4182 / air^2 + 2.76249e8 / air^4),
  applied to the wavelengths as given, in their own frame, before the shift to
  the observed frame.
* Errors: a 1-sigma column becomes ivar = 1 / sigma^2 (zero where sigma is not
  positive or not finite); an inverse-variance column is used as is.
* ``flux_scale`` multiplies the flux so that it is in 1e-17 erg/s/cm^2/A, the
  unit the luminosities assume; velocities do not depend on it.
"""
from __future__ import annotations

import os

import numpy as np

WAVE_UNITS = {"angstrom": 1.0, "a": 1.0, "aa": 1.0, "ang": 1.0, "nm": 10.0, "um": 1e4, "micron": 1e4, "m": 1e10}


def air_to_vacuum(wave_air_aa):
    """Air to vacuum wavelength (Angstrom), Morton (1991) as used by SDSS."""
    w = np.asarray(wave_air_aa, float)
    return w * (1.0 + 2.735182e-4 + 131.4182 / w**2 + 2.76249e8 / w**4)


def vacuum_to_air(wave_vac_aa):
    """Vacuum to air wavelength (Angstrom), the inverse of ``air_to_vacuum`` by iteration."""
    w = np.asarray(wave_vac_aa, float)
    air = w.copy()
    for _ in range(5):
        air = w / (1.0 + 2.735182e-4 + 131.4182 / air**2 + 2.76249e8 / air**4)
    return air


def _load_table(path, hdu=1, row=None):
    """Return a mapping column name -> array (case-insensitive lookup) and the list of names."""
    p = str(path); ext = os.path.splitext(p)[1].lower()
    if ext in (".fits", ".fit", ".fts") or p.lower().endswith((".fits.gz", ".fit.gz", ".fts.gz")):
        from astropy.io import fits
        with fits.open(p, memmap=False) as h:
            data = h[hdu].data
            if not isinstance(h[hdu], (fits.BinTableHDU, fits.TableHDU)):
                raise ValueError("this HDU is an image; specify the image-HDU selectors")
            cols = {n: np.asarray(data[n]) for n in data.columns.names}
            if row is not None:
                if not 0 <= row < len(data):
                    raise ValueError("FITS row is out of range (row numbers start at zero)")
                cols = {n: np.atleast_1d(a[row]) for n, a in cols.items()}
    elif ext in (".ecsv", ".csv"):
        from astropy.io import ascii
        t = ascii.read(p, format="ecsv" if ext == ".ecsv" else "csv")
        cols = {n: np.asarray(t[n]) for n in t.colnames}
    else:
        try:
            from astropy.io import ascii
            t = ascii.read(p)
            cols = {n: np.asarray(t[n]) for n in t.colnames}
        except Exception:
            arr = np.loadtxt(p)
            arr = np.atleast_2d(arr)
            cols = {f"col{i}": arr[:, i] for i in range(arr.shape[1])}
    lower = {k.lower(): k for k in cols}
    return cols, lower


def _get(cols, lower, name):
    if name is None:
        return None
    if isinstance(name, int) or (isinstance(name, str) and name.isdigit()):
        i = int(name)
        keys = list(cols)
        return np.asarray(cols[keys[i]], float)
    key = lower.get(str(name).lower())
    if key is None:
        raise KeyError(f"column {name!r} not found; available: {list(cols)}")
    return np.asarray(cols[key], float)


def read_table(path, wave="wave", flux="flux", err=None, ivar=None, wave_unit="angstrom",
               frame="obs", air=False, z=None, flux_scale=1.0, hdu=1, row=None,
               z_column=None, z_key=None, wave_hdu=None, flux_hdu=None,
               err_hdu=None, ivar_hdu=None, mask=None, flux_frame="obs"):
    """Read a spectrum from a table with user-declared columns, units and frame.

    Returns dict(wave, flux, ivar, z, path, kind='table') with wave in
    observed-frame vacuum Angstrom. Exactly one of ``err`` (1 sigma) and
    ``ivar`` must be given; ``z`` is required for ``frame='rest'``.
    """
    images = any(x is not None for x in (wave_hdu, flux_hdu, err_hdu, ivar_hdu))
    if images:
        if wave_hdu is None or flux_hdu is None or (err_hdu is None) == (ivar_hdu is None):
            raise ValueError("image FITS needs wave_hdu, flux_hdu and exactly one of err_hdu/ivar_hdu")
        if err is not None or ivar is not None or mask is not None or z_column is not None:
            raise ValueError("do not mix image-HDU selectors with table-column selectors")
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdus:
            def array(ext):
                a = np.array(hdus[ext].data, dtype=float, copy=True)
                if row is not None and a.ndim > 1:
                    if not 0 <= row < len(a):
                        raise ValueError("image row out of range")
                    a = a[row]
                return a
            cols = dict(wave=array(wave_hdu), flux=array(flux_hdu), noise=array(err_hdu if err_hdu is not None else ivar_hdu))
        lower = {k:k for k in cols}
        wave, flux = "wave", "flux"
        err, ivar = ("noise", None) if err_hdu is not None else (None, "noise")
    else:
        cols, lower = _load_table(path, hdu=hdu, row=row)
    if (err is None) == (ivar is None):
        raise ValueError("give exactly one of err= (1-sigma column) or ivar= (inverse-variance column)")
    if sum(x is not None for x in (z, z_column, z_key)) > 1:
        raise ValueError("choose only one of z, z_column or z_key")
    if z_column is not None:
        zs = _get(cols, lower, z_column).ravel()
        if not len(zs) or not np.isfinite(zs).all() or not np.all(zs == zs[0]):
            raise ValueError("redshift column must specify one finite redshift for this spectrum")
        z = float(zs[0])
    if z_key is not None:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdus:
            headers = [hdus[0].header]
            if hdu != 0:
                headers.append(hdus[hdu].header)
            values = [float(header[z_key]) for header in headers if z_key in header]
        if not values or not all(v == values[0] for v in values):
            raise ValueError("redshift keyword missing or conflicting between primary and selected HDU")
        z = values[0]
    if z is not None and (not np.isfinite(z) or z <= -1):
        raise ValueError("redshift must be finite and above -1")
    if not np.isfinite(flux_scale) or flux_scale <= 0:
        raise ValueError("flux_scale must be positive and finite")
    if flux_frame not in ("obs", "rest"):
        raise ValueError("flux_frame must be obs or rest")
    if flux_frame == "rest":
        if z is None:
            raise ValueError("rest-frame flux density needs a redshift")
        flux_scale /= 1 + z
    def vector(a):
        if a.ndim == 2 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 1:
            raise ValueError("multiple spectra in FITS table/image: select one zero-based row explicitly")
        return a
    w = vector(_get(cols, lower, wave))
    cols = dict(cols)
    for selector in (flux, err if err is not None else ivar, mask):
        if selector is not None:
            a = vector(_get(cols, lower, selector))
            if a.shape != w.shape:
                raise ValueError("wavelength, flux, noise and mask arrays must have identical shapes")
            key = list(cols)[int(selector)] if str(selector).isdigit() else lower[str(selector).lower()]
            cols[key] = a
    f = _get(cols, lower, flux) * float(flux_scale)
    if err is not None:
        e = _get(cols, lower, err) * float(flux_scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            iv = np.where(np.isfinite(e) & (e > 0), 1.0 / e**2, 0.0)
    else:
        iv = _get(cols, lower, ivar) / float(flux_scale) ** 2
        iv = np.where(np.isfinite(iv) & (iv > 0), iv, 0.0)
    if mask is not None:
        iv = np.where(_get(cols, lower, mask) == 0, iv, 0.)
    unit = str(wave_unit).lower()
    if unit not in WAVE_UNITS:
        raise ValueError(f"unknown wavelength unit {wave_unit!r}; use one of {sorted(WAVE_UNITS)}")
    w = w * WAVE_UNITS[unit]
    if air:
        w = air_to_vacuum(w)
    frame = str(frame).lower()
    if frame in ("rest", "restframe", "rest-frame"):
        if z is None or not np.isfinite(z):
            raise ValueError("frame='rest' needs the redshift z")
        w = w * (1.0 + float(z))
    elif frame not in ("obs", "observed", "obsframe"):
        raise ValueError(f"unknown frame {frame!r}; use 'obs' or 'rest'")
    order = np.argsort(w)
    w, f, iv = w[order], f[order], iv[order]
    good = np.isfinite(w)
    return dict(wave=w[good], flux=f[good], ivar=iv[good], z=(float(z) if z is not None else np.nan),
                path=str(path), kind="table")


def write_table(path, wave, flux, ivar=None, err=None, names=("wave", "flux", "err")):
    """Write a spectrum as CSV or ECSV (by extension) with the given column names."""
    from astropy.table import Table
    if err is None:
        iv = np.asarray(ivar, float)
        with np.errstate(divide="ignore"):
            err = np.where(iv > 0, 1.0 / np.sqrt(np.where(iv > 0, iv, 1.0)), np.nan)
    t = Table([np.asarray(wave, float), np.asarray(flux, float), np.asarray(err, float)], names=list(names))
    ext = os.path.splitext(str(path))[1].lower()
    t.write(path, format="ascii.ecsv" if ext == ".ecsv" else "ascii.csv", overwrite=True)
    return path
