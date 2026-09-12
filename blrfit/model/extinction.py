"""
Galactic extinction: Cardelli, Clayton & Mathis (1989) with the O'Donnell (1994)
optical coefficients and R_V = 3.1.

The correction is applied to the observed-frame spectrum before anything is
fitted. It changes the continuum slope and therefore the host and Fe II
decomposition, the continuum-subtracted line profiles and the fitted
velocities (on the SDSS spectrum of J001224 an E(B-V) of 0.03 instead of 0
moves the broad Hbeta c(1/2) by about 200 km/s through a different host
solution). Use the same E(B-V) convention as the measurement being reproduced:
the DESI catalogue used the FIBERMAP value, the SDSS comparisons of the
literature samples used 0.
"""
from __future__ import annotations

import numpy as np


def ccm89_alav(wave_aa, rv=3.1):
    """A_lambda / A_V for wavelengths in Angstrom (valid for 0.3 < 1/micron < 8)."""
    x = 1e4 / np.asarray(wave_aa, float)          # inverse microns
    a = np.zeros_like(x); b = np.zeros_like(x)
    ir = (x >= 0.3) & (x < 1.1)
    a[ir] = 0.574 * x[ir] ** 1.61;  b[ir] = -0.527 * x[ir] ** 1.61
    op = (x >= 1.1) & (x < 3.3)
    y = x[op] - 1.82
    # O'Donnell (1994) update of the optical polynomials
    a[op] = (1 + 0.104 * y - 0.609 * y**2 + 0.701 * y**3 + 1.137 * y**4
             - 1.718 * y**5 - 0.827 * y**6 + 1.647 * y**7 - 0.505 * y**8)
    b[op] = (1.952 * y + 2.908 * y**2 - 3.989 * y**3 - 7.985 * y**4
             + 11.102 * y**5 + 5.491 * y**6 - 10.805 * y**7 + 3.347 * y**8)
    uv = (x >= 3.3) & (x <= 8.0)
    xu = x[uv]
    fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9)**2 - 0.009779 * (xu - 5.9)**3, 0.0)
    fb = np.where(xu >= 5.9, 0.2130 * (xu - 5.9)**2 + 0.1207 * (xu - 5.9)**3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67)**2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62)**2 + 0.263) + fb
    return a + b / rv


def deredden(wave_obs, flux, ivar, ebv, rv=3.1):
    """Remove Milky Way extinction from an observed-frame spectrum.

    Returns the corrected flux and inverse variance; the input arrays are
    returned unchanged when E(B-V) is zero, negative or not finite.
    """
    if not (np.isfinite(ebv) and ebv > 0):
        return flux, ivar
    alam = ccm89_alav(wave_obs, rv) * rv * ebv
    corr = 10 ** (0.4 * alam)
    return flux * corr, ivar / corr**2
