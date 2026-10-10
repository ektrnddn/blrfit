"""
Physical quantities derived from a fit: the monochromatic continuum luminosity,
the virial black-hole mass, the radius of the broad-line region and the
largest orbital velocity change a binary could show.

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

import warnings

import numpy as np

from .constants import PL_PIVOT, FLUX_UNIT_CGS, LUM_REF_WAVE, PL_ALPHA_BLUE_LIMIT
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
    ``fit_spectrum``). A RuntimeWarning is issued when a slope is bluer than
    PL_ALPHA_BLUE_LIMIT (flag ``pl_unphysical``): such a power law is not an
    AGN continuum and its luminosity is not an AGN luminosity.
    """
    norm = _pl_parameter(conti, "pl_norm")
    alpha = _pl_parameter(conti, "pl_alpha")
    if np.any(np.asarray(alpha) < PL_ALPHA_BLUE_LIMIT):
        warnings.warn(
            f"power-law slope bluer than {PL_ALPHA_BLUE_LIMIT} (pl_unphysical): not an AGN continuum, "
            "its luminosity is not an AGN luminosity",
            RuntimeWarning,
            stacklevel=2,
        )
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


# ----------------------------------------------------------------------------
# Virial masses and the orbital bound
# ----------------------------------------------------------------------------
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN_KG = 1.98892e30
PC_M = 3.0856775814913673e16
LIGHT_DAY_M = 2.99792458e8 * 86400.0
YEAR_S = 3.15576e7
# radius-luminosity relation of the broad Hbeta region, Bentz et al. (2013):
# log(R / light-day) = RL_K + RL_ALPHA log(L5100 / 1e44 erg/s)
RL_K, RL_ALPHA = 1.527, 0.533


def virial_mass_halpha(l_halpha, fwhm_kms):
    """Black-hole mass in solar masses from the broad Halpha luminosity (erg/s)
    and FWHM (km/s), Greene & Ho (2005): M = 2.0e6 (L/1e42)^0.55 (FWHM/1e3)^2.06;
    NaN where the luminosity or the width is not positive."""
    l_halpha, fwhm = np.asarray(l_halpha, float), np.asarray(fwhm_kms, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = 2.0e6 * (l_halpha / 1e42) ** 0.55 * (fwhm / 1e3) ** 2.06
        out = np.where((l_halpha > 0) & (fwhm > 0), m, np.nan)
    return float(out) if out.ndim == 0 else out


def virial_mass_hbeta(l5100, fwhm_kms):
    """Black-hole mass in solar masses from lambda L_lambda(5100) (erg/s) and the
    broad Hbeta FWHM (km/s), Vestergaard & Peterson (2006):
    log M = log[(FWHM/1e3)^2 (L5100/1e44)^0.5] + 6.91."""
    l5100, fwhm = np.asarray(l5100, float), np.asarray(fwhm_kms, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = 10.0 ** (np.log10((fwhm / 1e3) ** 2 * (l5100 / 1e44) ** 0.5) + 6.91)
        out = np.where((l5100 > 0) & (fwhm > 0), m, np.nan)
    return float(out) if out.ndim == 0 else out


def blr_radius_ltd(l5100):
    """Radius of the broad Hbeta region in light-days from lambda L_lambda(5100)
    (erg/s), Bentz et al. (2013); NaN where the luminosity is not positive."""
    l5100 = np.asarray(l5100, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 10.0 ** (RL_K + RL_ALPHA * np.log10(l5100 / 1e44))
        out = np.where(l5100 > 0, r, np.nan)
    return float(out) if out.ndim == 0 else out


def roche_fraction(q):
    """Mean radius of the Roche lobe over the separation for a body whose mass
    ratio to the other body is q = M_this / M_other (Paczynski 1971)."""
    q = np.asarray(q, float)
    return np.where(q > 1.0 / 1.88, 0.38 + 0.2 * np.log10(q), 0.46224 * (q / (1 + q)) ** (1.0 / 3.0))


def orbital_limits(m_active, r_blr_ltd, q=0.1):
    """The closest circular binary that keeps the broad-line region of the
    active black hole (mass ``m_active``, solar masses) inside its Roche lobe,
    for a companion of mass m_active / q: the separation (pc), the line-of-sight
    velocity amplitude of the active hole seen edge-on (km/s) and the period
    (years) at that separation. A wider orbit is slower, v = vmax (P/pmin)^(-1/3),
    so vmax is the largest orbital velocity the active hole can have and pmin the
    shortest period. The permissive bound of the tiers takes q = 0.1, the
    heavier companion that makes the active hole move fastest. Returns
    dict(amin_pc, vmax_kms, pmin_yr), NaN where the inputs are not positive."""
    m, r, q = np.asarray(m_active, float), np.asarray(r_blr_ltd, float), float(q)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_tot = m * (1.0 + 1.0 / q) * M_SUN_KG
        a_min = r * LIGHT_DAY_M / roche_fraction(q)
        vmax = np.sqrt(G_SI * m_tot / a_min) / (1.0 + q) / 1e3
        pmin = 2.0 * np.pi * np.sqrt(a_min**3 / (G_SI * m_tot)) / YEAR_S
        ok = (m > 0) & (r > 0)
    out = dict(
        amin_pc=np.where(ok, a_min / PC_M, np.nan),
        vmax_kms=np.where(ok, vmax, np.nan),
        pmin_yr=np.where(ok, pmin, np.nan),
    )
    if np.ndim(m) == 0 and np.ndim(r) == 0:
        out = {k: float(v) for k, v in out.items()}
    return out


def max_orbital_change(vmax_kms, pmin_yr, dt_yr, n=2000):
    """The largest change of the active hole's line-of-sight velocity over
    ``dt_yr`` for any period at or above pmin: 2 v(P) |sin(pi dt / P)| with
    v(P) = vmax (P/pmin)^(-1/3), searched on a logarithmic grid of periods up to
    1e5 pmin. Returns (change in km/s, the period at which it occurs)."""
    if not (
        np.isfinite(vmax_kms)
        and np.isfinite(pmin_yr)
        and np.isfinite(dt_yr)
        and vmax_kms > 0
        and pmin_yr > 0
        and dt_yr > 0
    ):
        return np.nan, np.nan
    P = pmin_yr * np.logspace(0, 5, n)
    dv = 2.0 * vmax_kms * (P / pmin_yr) ** (-1.0 / 3.0) * np.abs(np.sin(np.pi * dt_yr / P))
    k = int(np.argmax(dv))
    return float(dv[k]), float(P[k])


def longest_period(vmax_kms, pmin_yr, dt_yr, change_kms, n=2000):
    """The longest period at which a change of ``change_kms`` over ``dt_yr`` is
    still possible, in years; NaN when no period allows it."""
    if not (np.isfinite(change_kms) and change_kms >= 0 and np.isfinite(vmax_kms) and vmax_kms > 0):
        return np.nan
    if not (np.isfinite(pmin_yr) and pmin_yr > 0 and np.isfinite(dt_yr) and dt_yr > 0):
        return np.nan
    P = pmin_yr * np.logspace(0, 5, n)
    dv = 2.0 * vmax_kms * (P / pmin_yr) ** (-1.0 / 3.0) * np.abs(np.sin(np.pi * dt_yr / P))
    ok = dv >= change_kms
    return float(P[ok].max()) if ok.any() else np.nan


def target_mass(res):
    """The virial mass and the orbital limits of one fit result: the Halpha mass
    where broad Halpha is measurable (class A, B, C or F with a luminosity and a
    width), else the Hbeta mass; the broad-line radius from the 5100 A continuum
    luminosity; and ``orbital_limits`` for q = 0.1 and 1. Returns a dictionary
    with logmbh, logmbh_ha, logmbh_hb, l5100, r_blr_ltd, vmax_q01, pmin_q01_yr,
    vmax_q1 and pmin_q1_yr, NaN where a quantity is not available."""
    meas, cls = res.get("meas") or {}, res.get("cls") or {}

    def usable(line):
        m, c = meas.get(line) or {}, cls.get(line) or {}
        return (
            c.get("label", "") in ("A", "B", "C", "F")
            and np.isfinite(m.get("fwhm", np.nan))
            and m["fwhm"] > 0
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        l5100 = continuum_luminosity(res)
    m_ha = (
        virial_mass_halpha(meas["Halpha"].get("broad_lum", np.nan), meas["Halpha"]["fwhm"])
        if usable("Halpha")
        else np.nan
    )
    m_hb = virial_mass_hbeta(l5100, meas["Hbeta"]["fwhm"]) if usable("Hbeta") else np.nan
    m_act = m_ha if np.isfinite(m_ha) else m_hb
    r_blr = blr_radius_ltd(l5100)
    out = dict(
        logmbh=float(np.log10(m_act)) if np.isfinite(m_act) and m_act > 0 else np.nan,
        logmbh_ha=float(np.log10(m_ha)) if np.isfinite(m_ha) and m_ha > 0 else np.nan,
        logmbh_hb=float(np.log10(m_hb)) if np.isfinite(m_hb) and m_hb > 0 else np.nan,
        l5100=float(l5100) if np.isfinite(l5100) else np.nan,
        r_blr_ltd=float(r_blr) if np.isfinite(r_blr) else np.nan,
    )
    for q, tag in ((0.1, "q01"), (1.0, "q1")):
        lim = orbital_limits(m_act, r_blr, q) if np.isfinite(m_act) and np.isfinite(r_blr) else {}
        out[f"vmax_{tag}"] = float(lim.get("vmax_kms", np.nan))
        out[f"pmin_{tag}_yr"] = float(lim.get("pmin_yr", np.nan))
    return out


__all__ = [
    "lambda_l_lambda",
    "continuum_luminosity",
    "lumdist_cm",
    "virial_mass_halpha",
    "virial_mass_hbeta",
    "blr_radius_ltd",
    "roche_fraction",
    "orbital_limits",
    "max_orbital_change",
    "longest_period",
    "target_mass",
]
