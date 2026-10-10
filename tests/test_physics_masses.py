"""The virial masses, the broad-line radius and the orbital bound."""

import numpy as np
import pytest

from blrfit import physics as P


def test_virial_masses_and_radius():
    assert P.virial_mass_halpha(1e42, 3000.0) == pytest.approx(2.0e6 * 3.0**2.06)
    assert P.virial_mass_hbeta(1e44, 3000.0) == pytest.approx(9.0 * 10**6.91)
    assert P.blr_radius_ltd(1e44) == pytest.approx(10**1.527)
    assert np.isnan(P.virial_mass_halpha(0.0, 3000.0)) and np.isnan(P.virial_mass_hbeta(1e44, np.nan))
    assert np.isnan(P.blr_radius_ltd(-1.0))
    arr = P.virial_mass_halpha(np.array([1e42, 1e43]), np.array([3000.0, 3000.0]))
    assert arr.shape == (2,) and arr[1] / arr[0] == pytest.approx(10**0.55)


def test_roche_fraction():
    assert P.roche_fraction(1.0) == pytest.approx(0.38)
    assert P.roche_fraction(0.1) == pytest.approx(0.46224 * (0.1 / 1.1) ** (1.0 / 3.0))
    assert P.roche_fraction(0.1) < P.roche_fraction(1.0) < P.roche_fraction(10.0)


def test_orbital_limits_follow_kepler():
    m, r, q = 1e8, 30.0, 0.1
    lim = P.orbital_limits(m, r, q)
    a = r * P.LIGHT_DAY_M / P.roche_fraction(q)
    m_tot = m * (1 + 1 / q) * P.M_SUN_KG
    assert lim["amin_pc"] == pytest.approx(a / P.PC_M)
    assert lim["pmin_yr"] == pytest.approx(2 * np.pi * np.sqrt(a**3 / (P.G_SI * m_tot)) / P.YEAR_S)
    # the active hole's share of the relative orbital velocity is M_comp / M_tot = 1 / (1 + q)
    assert lim["vmax_kms"] == pytest.approx(np.sqrt(P.G_SI * m_tot / a) / (1 + q) / 1e3)
    # a heavier companion (smaller q) moves the active hole faster
    assert P.orbital_limits(m, r, 1.0)["vmax_kms"] < lim["vmax_kms"]
    assert all(np.isnan(v) for v in P.orbital_limits(np.nan, r, q).values())


def test_max_change_and_longest_period():
    lim = P.orbital_limits(1e8, 30.0, 0.1)
    vmax, pmin = lim["vmax_kms"], lim["pmin_yr"]
    dv, p_at = P.max_orbital_change(vmax, pmin, 5.0)
    assert 0 < dv <= 2 * vmax and p_at >= pmin
    assert np.isnan(P.max_orbital_change(vmax, pmin, 0.0)[0])
    p_small = P.longest_period(vmax, pmin, 5.0, 0.2 * dv)
    p_large = P.longest_period(vmax, pmin, 5.0, 0.9 * dv)
    assert p_small > p_large >= pmin
    assert np.isnan(P.longest_period(vmax, pmin, 5.0, 2.0 * dv))


def test_target_mass_from_a_result():
    res = dict(
        z=0.2,
        conti=dict(pl_norm=10.0, pl_alpha=-1.5),
        meas=dict(Halpha=dict(fwhm=3000.0, broad_lum=1e42), Hbeta=dict(fwhm=3500.0, broad_lum=3e41)),
        cls=dict(Halpha=dict(label="A"), Hbeta=dict(label="C")),
    )
    out = P.target_mass(res)
    l5100 = P.continuum_luminosity(res)
    assert out["l5100"] == pytest.approx(l5100) and out["r_blr_ltd"] == pytest.approx(P.blr_radius_ltd(l5100))
    assert out["logmbh"] == pytest.approx(np.log10(P.virial_mass_halpha(1e42, 3000.0)))
    assert out["logmbh_hb"] == pytest.approx(np.log10(P.virial_mass_hbeta(l5100, 3500.0)))
    assert np.isfinite(out["vmax_q01"]) and out["vmax_q01"] > out["vmax_q1"] and out["pmin_q01_yr"] > 0
    res["cls"]["Halpha"]["label"] = "E"
    out2 = P.target_mass(res)
    assert out2["logmbh"] == pytest.approx(out["logmbh_hb"]) and np.isnan(out2["logmbh_ha"])
