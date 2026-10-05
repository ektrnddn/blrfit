"""
Readers for the three kinds of input (SDSS spec files, DESI coadds, generic
tables) and ``read_spectrum``, which picks one from the file name.
"""

from __future__ import annotations

from .sdss import read_sdss, is_sdss_spec
from .desi import read_desi, is_desi_coadd, is_desi_spectra, coadd_cameras, read_redrock, write_single_target
from .generic import read_table, air_to_vacuum, vacuum_to_air, write_table


def read_spectrum(path, targetid=None, *, survey="auto", mask_policy="ivar", redrock=None, **table_kw):
    """Read a spectrum from an SDSS spec file, a DESI coadd (needs ``targetid``)
    or a table (needs the column declarations of ``read_table``). Explicit
    survey= selects the format regardless of filename; auto preserves legacy
    dispatch. survey="desi" can infer the ID only when there is one row. Returns the
    reader's dictionary with at least wave, flux, ivar, z, path and kind."""
    if survey not in ("auto", "desi", "sdss", "generic"):
        raise ValueError("survey must be desi, sdss, generic or auto")
    if survey == "sdss" or (survey == "auto" and is_sdss_spec(path)):
        return read_sdss(path, mask_policy=mask_policy)
    if survey == "desi" or (survey == "auto" and is_desi_coadd(path)):
        if targetid is None:
            if survey == "auto":
                raise ValueError("a DESI coadd needs targetid= (survey='desi' can infer a sole row)")
            from astropy.io import fits

            with fits.open(path, memmap=False) as h:
                if "FIBERMAP" not in h:
                    raise ValueError("not a DESI FITS file; specify survey='sdss' or 'generic'")
                ids = h["FIBERMAP"].data["TARGETID"]
                if len(ids) != 1:
                    raise ValueError("a multi-row DESI file needs targetid=")
                targetid = int(ids[0])
        return read_desi(path, targetid, redrock=redrock)
    if is_desi_spectra(path):
        raise ValueError(f"{path} is a per-exposure DESI spectra file; use the coadd-*.fits file instead")
    return read_table(path, **table_kw)


__all__ = [
    "read_spectrum",
    "read_sdss",
    "read_desi",
    "read_table",
    "is_sdss_spec",
    "is_desi_coadd",
    "is_desi_spectra",
    "coadd_cameras",
    "read_redrock",
    "write_single_target",
    "air_to_vacuum",
    "vacuum_to_air",
    "write_table",
]
