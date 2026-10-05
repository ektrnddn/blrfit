"""
SDSS optical spectra with loglam/flux/ivar in the first spectral table.

The first extension holds the coadded spectrum (columns ``flux``, ``loglam``,
``ivar``, ``and_mask``, ...) on a log-wavelength grid in vacuum Angstrom and in
units of 1e-17 erg/s/cm^2/A; the second holds the pipeline redshift. Pixels
flagged in ``and_mask`` are preserved under the historical Python default
mask_policy="ivar". The CLI selects "conservative", excluding nonzero MASK or
AND_MASK and invalid flux/IVAR, as in the project temporal input reader. The files carry no Galactic E(B-V). Files compressed
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


def read_sdss(path, *, mask_policy="ivar"):
    """Read an SDSS spec file. Returns dict(wave, flux, ivar, z, zerr, cls, mjd,
    plate, fiber, date, ra, dec, path); wave in vacuum Angstrom (observed frame)."""
    from astropy.io import fits

    if mask_policy not in ("ivar", "conservative"):
        raise ValueError("mask_policy must be ivar or conservative")
    with fits.open(path, memmap=False) as hdul:
        d = hdul[1].data
        names = {n.lower(): n for n in getattr(hdul[1].columns, "names", [])}
        if not {"loglam", "flux", "ivar"}.issubset(names):
            raise ValueError("SDSS input needs a loglam/flux/ivar spectral table in extension 1")
        wave = 10.0 ** np.asarray(d["loglam"], float)
        flux = np.asarray(d["flux"], float)
        ivar = np.asarray(d["ivar"], float)
        if wave.ndim != 1 or wave.shape != flux.shape or wave.shape != ivar.shape:
            raise ValueError(
                "SDSS file must contain one coadded spectrum; use a generic row mapping for vector tables"
            )
        mask_column = "mask" if "mask" in names else "and_mask"
        native_mask = np.asarray(d[names[mask_column]]) if mask_column in names else None
        if mask_policy == "conservative":
            if native_mask is None:
                raise ValueError(
                    "SDSS AND_MASK missing; explicitly choose mask_policy='ivar' to use only IVAR"
                )
            ivar = np.where(
                (native_mask == 0) & np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0), ivar, 0.0
            )
        hdr = hdul[0].header
        z = zerr = np.nan
        zwarn = None
        cls = ""
        for ext in range(2, len(hdul)):
            cols = getattr(hdul[ext], "columns", None)
            cmap = {n.upper(): n for n in cols.names} if cols is not None else {}
            if "Z" in cmap:
                t = hdul[ext].data
                if len(t) != 1:
                    raise ValueError("SDSS metadata has multiple redshift rows; select a spectrum explicitly")
                z = float(np.atleast_1d(t[cmap["Z"]])[0])
                if "Z_ERR" in cmap:
                    zerr = float(np.atleast_1d(t[cmap["Z_ERR"]])[0])
                if "CLASS" in cmap:
                    cls = str(np.atleast_1d(t[cmap["CLASS"]])[0]).strip()
                if "ZWARNING" in cmap:
                    zwarn = int(t[cmap["ZWARNING"]][0])
                break
        mjd = hdr.get("MJD", np.nan)
        ra = hdr.get("PLUG_RA", hdr.get("RA", np.nan))
        dec = hdr.get("PLUG_DEC", hdr.get("DEC", np.nan))
        return dict(
            wave=wave,
            flux=flux,
            ivar=ivar,
            z=z,
            zerr=zerr,
            cls=cls,
            mjd=float(mjd) if mjd is not None else np.nan,
            plate=hdr.get("PLATEID", hdr.get("PLATE")),
            fiber=hdr.get("FIBERID"),
            date=mjd_to_date(mjd),
            ra=float(ra) if ra is not None else np.nan,
            dec=float(dec) if dec is not None else np.nan,
            path=str(path),
            kind="sdss",
            mask=native_mask,
            mask_policy=mask_policy,
            product="pipeline_coadd",
            mask_column=mask_column,
            zwarn=zwarn,
            uncertainty_calibrated=False,
        )


def is_sdss_spec(path):
    """True for a file named like an SDSS spec file, plain or gzipped."""
    import os

    name = os.path.basename(str(path)).lower()
    return name.startswith("spec-") and name.endswith((".fits", ".fits.gz"))
