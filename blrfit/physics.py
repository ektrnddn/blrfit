"""
Physical quantities derived from a fit: the monochromatic continuum luminosity.

The fitter works on the rest-frame spectrum f_rest(lam_rest) = (1 + z)
f_obs(lam_obs) at lam_rest = lam_obs / (1 + z), in the flux units of the input
(1e-17 erg/s/cm^2/A for SDSS and DESI). The factor (1 + z) is the Jacobian
d lam_obs / d lam_rest: f_rest is the flux density per rest-frame Angstrom,
and the luminosity density per rest-frame Angstrom is then (Hogg 1999,
equations 22 and 23)

    L_lambda(lam_rest) = 4 pi D_L^2 f_rest(lam_rest) = 4 pi D_L^2 (1 + z) f_obs(lam_obs).

The fitted power law ``pl_norm (lam_rest / 3000 A) ** pl_alpha`` is f_rest, so
lambda L_lambda at a rest wavelength is 4 pi D_L^2 lam_rest f_rest(lam_rest)
1e-17 with no further redshift factor. Dividing by (1 + z) once more, as if
``pl_norm`` were an observed-frame flux density, understates the luminosity by
log10(1 + z) dex (0.10 dex at z = 0.25). The luminosity distance is that of the
Planck 2018 cosmology, as for ``broad_lum`` in ``measure_complex``.
"""
from __future__ import annotations

import numpy as np

from .constants import PL_PIVOT, FLUX_UNIT_CGS, LUM_REF_WAVE
from .model.fit import _lumdist_cm


def lumdist_cm(z):
    """Luminosity distance in cm for the Planck 2018 cosmology; a scalar for a
    scalar redshift, an array for an array (NaN where astropy is missing)."""
    z = np.asarray(z, float)
    if z.ndim == 0:
        return _lumdist_cm(float(z))
    try:
        from astropy.cosmology import Planck18
        import astropy.units as u
        return np.asarray(Planck18.luminosity_distance(z).to(u.cm).value, float)
    except Exception:
        return np.full(z.shape, np.nan)


def _pl_parameter(conti, key):
    """``pl_norm`` / ``pl_alpha`` from a ``conti`` dictionary or from a summary
    row, where they are stored as ``conti_pl_norm`` / ``conti_pl_alpha``."""
    if key in conti:
        return np.asarray(conti[key], float)
    if f"conti_{key}" in conti:
        return np.asarray(conti[f"conti_{key}"], float)
    return np.asarray(np.nan)


def lambda_l_lambda(conti, z, lam_rest=LUM_REF_WAVE):
    """lambda L_lambda of the fitted power law at a rest wavelength, in erg/s.

    conti : the ``conti`` dictionary of a ``fit_spectrum`` result (keys
        ``pl_norm``, ``pl_alpha``) or a ``summary_row`` (``conti_pl_norm``,
        ``conti_pl_alpha``); the values may be arrays of equal shape
    z : the redshift of the fit (``res['z']`` / ``z_in``), scalar or array
    lam_rest : rest wavelength in Angstrom, 5100 A by default (the reference
        wavelength of the Hbeta virial mass estimators)

    ``pl_norm`` is the flux density at 3000 A rest of the (1 + z)-scaled
    rest-frame spectrum the fit works on, in 1e-17 erg/s/cm^2/A per rest-frame
    Angstrom, so the luminosity is

        4 pi D_L^2 * lam_rest * pl_norm (lam_rest / 3000)^pl_alpha * 1e-17

    with no further (1 + z) factor: do not divide by (1 + z) again. The result
    is the power law alone, without Fe II or host light, and is NaN where the
    normalisation is not positive, the redshift is not finite or the input was
    not in 1e-17 erg/s/cm^2/A (the fitter does not know the input units; see
    ``fit_spectrum``).
    """
    norm = _pl_parameter(conti, "pl_norm")
    alpha = _pl_parameter(conti, "pl_alpha")
    lam = np.asarray(lam_rest, float)
    dl = lumdist_cm(z)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_rest = norm * (lam / PL_PIVOT) ** alpha
        lum = 4.0 * np.pi * dl**2 * lam * f_rest * FLUX_UNIT_CGS
        ok = (norm > 0) & (lam > 0) & (dl > 0) & np.isfinite(lum)
    out = np.where(ok, lum, np.nan)
    return float(out) if out.ndim == 0 else out


def continuum_luminosity(res, lam_rest=LUM_REF_WAVE):
    """lambda L_lambda (erg/s) of the power-law continuum of a ``fit_spectrum``
    result at ``lam_rest`` (default 5100 A), from ``res['conti']`` and
    ``res['z']``; see ``lambda_l_lambda`` for the convention. NaN when the
    continuum was not fitted."""
    conti = res.get("conti") or {}
    if "pl_norm" not in conti:
        return np.nan
    return lambda_l_lambda(conti, res["z"], lam_rest=lam_rest)


__all__ = ["lambda_l_lambda", "continuum_luminosity", "lumdist_cm"]
