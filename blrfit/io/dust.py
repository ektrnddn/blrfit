"""
Galactic E(B-V) at a sky position from the SFD98 map, through the optional
``dustmaps`` package.

DESI coadds carry the SFD98 value of every target in ``FIBERMAP['EBV']``,
without a recalibration factor; ``sfd_ebv`` returns the same quantity for any
position, so that SDSS and generic spectra can be dereddened the way DESI
spectra are. The map files are not part of ``dustmaps`` itself: fetch them once
with ``python -c "import dustmaps.sfd; dustmaps.sfd.fetch()"``.
"""

from __future__ import annotations

import numpy as np

_QUERY = None


def sfd_ebv(ra, dec):
    """SFD98 E(B-V) at (ra, dec) in degrees. Returns ``(value, "")``, or
    ``(None, reason)`` when the coordinates are missing, ``dustmaps`` is not
    installed or its SFD map has not been fetched."""
    global _QUERY
    try:
        ra, dec = float(ra), float(dec)
    except (TypeError, ValueError):
        return None, "no coordinates"
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None, "no coordinates"
    try:
        from dustmaps.sfd import SFDQuery
    except ImportError:
        return None, "dustmaps not installed"
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        if _QUERY is None:
            _QUERY = SFDQuery()
        return float(_QUERY(SkyCoord(ra * u.deg, dec * u.deg))), ""
    except Exception as exc:  # the map files missing, or unreadable
        return None, f"SFD map not available ({type(exc).__name__})"
