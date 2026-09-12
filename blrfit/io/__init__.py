"""
Readers for the three kinds of input (SDSS spec files, DESI coadds, generic
tables) and ``read_spectrum``, which picks one from the file name.
"""
from __future__ import annotations

from .sdss import read_sdss, is_sdss_spec
from .desi import read_desi, is_desi_coadd, is_desi_spectra, coadd_cameras, read_redrock, write_single_target
from .generic import read_table, air_to_vacuum, vacuum_to_air, write_table


def read_spectrum(path, targetid=None, **table_kw):
    """Read a spectrum from an SDSS spec file, a DESI coadd (needs ``targetid``)
    or a table (needs the column declarations of ``read_table``). Returns the
    reader's dictionary with at least wave, flux, ivar, z, path and kind."""
    if is_sdss_spec(path):
        return read_sdss(path)
    if is_desi_coadd(path):
        if targetid is None:
            raise ValueError("a DESI coadd needs targetid=")
        return read_desi(path, targetid)
    if is_desi_spectra(path):
        raise ValueError(f"{path} is a per-exposure DESI spectra file; use the coadd-*.fits file instead")
    return read_table(path, **table_kw)


__all__ = ["read_spectrum", "read_sdss", "read_desi", "read_table", "is_sdss_spec", "is_desi_coadd", "is_desi_spectra",
           "coadd_cameras", "read_redrock", "write_single_target", "air_to_vacuum", "vacuum_to_air",
           "write_table"]
