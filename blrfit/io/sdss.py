"""
SDSS spectra: the ``spec-PLATE-MJD-FIBER.fits`` files of DR7 to DR17.

The first extension holds the coadded spectrum (columns ``flux``, ``loglam``,
``ivar``, ``and_mask``, ...) on a log-wavelength grid in vacuum Angstrom and in
units of 1e-17 erg/s/cm^2/A; the second holds the pipeline redshift. Pixels
flagged in ``and_mask`` are not removed: the pipeline sets their inverse
variance to zero where they are unusable, and the DESI catalogue fits used the
inverse variance alone. The files carry no Galactic E(B-V). Files compressed
with gzip (``spec-*.fits.gz``) are read in place; astropy decompresses them
transparently.
"""
from __future__ import annotations

import numpy as np


def mjd_to_date(mjd):
    """Modified Julian date to 'YYYY-MM-DD' ('' if it cannot be converted)."""
    try:
        from astropy.time import Time
        return Time(float(mjd), format="mjd").iso[:10]
    except Exception:
        return ""


def read_sdss(path):
    """Read an SDSS spec file. Returns dict(wave, flux, ivar, z, zerr, cls, mjd,
    plate, fiber, date, ra, dec, path); wave in vacuum Angstrom (observed frame)."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdul:
        d = hdul[1].data
        wave = 10.0 ** np.asarray(d["loglam"], float)
        flux = np.asarray(d["flux"], float)
        ivar = np.asarray(d["ivar"], float)
        hdr = hdul[0].header
        z = zerr = np.nan
        cls = ""
        for ext in range(2, len(hdul)):
            cols = getattr(hdul[ext], "columns", None)
            if cols is not None and "Z" in cols.names:
                t = hdul[ext].data
                z = float(np.atleast_1d(t["Z"])[0])
                if "Z_ERR" in cols.names:
                    zerr = float(np.atleast_1d(t["Z_ERR"])[0])
                if "CLASS" in cols.names:
                    cls = str(np.atleast_1d(t["CLASS"])[0]).strip()
                break
        mjd = hdr.get("MJD", np.nan)
        ra = hdr.get("PLUG_RA", hdr.get("RA", np.nan)); dec = hdr.get("PLUG_DEC", hdr.get("DEC", np.nan))
        return dict(wave=wave, flux=flux, ivar=ivar, z=z, zerr=zerr, cls=cls,
                    mjd=float(mjd) if mjd is not None else np.nan,
                    plate=hdr.get("PLATEID", hdr.get("PLATE")),
                    fiber=hdr.get("FIBERID"), date=mjd_to_date(mjd),
                    ra=float(ra) if ra is not None else np.nan, dec=float(dec) if dec is not None else np.nan,
                    path=str(path), kind="sdss")


def is_sdss_spec(path):
    """True for a file named like an SDSS spec file, plain or gzipped."""
    import os
    name = os.path.basename(str(path)).lower()
    return name.startswith("spec-") and name.endswith((".fits", ".fits.gz"))
