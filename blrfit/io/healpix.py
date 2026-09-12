"""
HEALPix pixel of a sky position in the NESTED scheme, in pure numpy.

DESI healpix files are organised with nside = 64, nested. The algorithm is the
one of the HEALPix C library (``ang2pix_nest_z_phi``); it is verified against
``healpy`` in the test suite.
"""
from __future__ import annotations

import numpy as np


def _spread_bits(v):
    """Interleave the bits of v with zeros (v in the even bit positions)."""
    v = np.asarray(v, dtype=np.int64)
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def ang2pix_nest(nside, ra_deg, dec_deg):
    """Nested HEALPix pixel index for nside (a power of two) and RA, Dec in degrees."""
    nside = int(nside)
    if nside <= 0 or (nside & (nside - 1)):
        raise ValueError("nside must be a power of two")
    order = int(np.log2(nside))
    ra = np.asarray(ra_deg, float); dec = np.asarray(dec_deg, float)
    scalar = ra.ndim == 0 and dec.ndim == 0
    ra = np.atleast_1d(ra); dec = np.atleast_1d(dec)
    # the same conversions as healpy's ang2pix(lonlat=True): theta = pi/2 - dec,
    # z = cos(theta), phi = ra; sin(theta) is used within 0.01 rad of the poles
    theta = np.pi / 2 - np.radians(dec)
    z = np.cos(theta); sth = np.sin(theta)
    near_pole = (theta < 0.01) | (theta > np.pi - 0.01)
    phi = np.radians(ra)
    za = np.abs(z)
    tt = np.mod(phi * (2.0 / np.pi), 4.0)                # in [0, 4)

    ix = np.zeros(ra.shape, dtype=np.int64); iy = np.zeros(ra.shape, dtype=np.int64)
    face = np.zeros(ra.shape, dtype=np.int64)

    eq = za <= 2.0 / 3.0
    if eq.any():
        temp1 = nside * (0.5 + tt[eq]); temp2 = nside * (z[eq] * 0.75)
        jp = np.floor(temp1 - temp2).astype(np.int64)    # ascending edge line
        jm = np.floor(temp1 + temp2).astype(np.int64)    # descending edge line
        ifp = jp // nside; ifm = jm // nside
        f = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
        face[eq] = f
        ix[eq] = jm & (nside - 1)
        iy[eq] = nside - (jp & (nside - 1)) - 1

    po = ~eq
    if po.any():
        ntt = np.floor(tt[po]).astype(np.int64); ntt = np.minimum(ntt, 3)
        tp = tt[po] - ntt
        zp = za[po]
        tmp = np.where(near_pole[po] & (zp >= 0.99), nside * sth[po] / np.sqrt((1.0 + zp) / 3.0),
                       nside * np.sqrt(3 * (1 - zp)))
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1.0 - tp) * tmp).astype(np.int64)
        jp = np.minimum(jp, nside - 1); jm = np.minimum(jm, nside - 1)
        north = z[po] >= 0
        face[po] = np.where(north, ntt, ntt + 8)
        ix[po] = np.where(north, nside - jm - 1, jp)
        iy[po] = np.where(north, nside - jp - 1, jm)

    pix = (face << (2 * order)) + _spread_bits(ix) + (_spread_bits(iy) << 1)
    return int(pix[0]) if scalar else pix
