"""
Unit test of the Galactic extinction law (``blrfit.model.extinction``), which
the regression pins do not exercise: the four pinned spectra have E(B-V) = 0,
for which ``deredden`` returns its input unchanged.

The curve is Cardelli, Clayton & Mathis (1989) with the O'Donnell (1994)
optical polynomials and R_V = 3.1. The pinned values of A_lambda / A_V are
those of the package on the reference stack (numpy 1.26.4); they follow from
the published coefficients, and the wavelengths sample the three branches
and the far-ultraviolet term: 12500 A (infrared, 0.282070), 7000 (0.753770),
5500 (0.998667: the O'Donnell polynomials do not return exactly 1 at V), 4400
(1.322219; with 5500 this gives 1 / (A_B/A_V - A_5500/A_V) = 3.09 for R_V),
3000 (1.818157), 2000 (ultraviolet, 2.842526) and 1500 A (far-ultraviolet
term, 2.663879). A change of any coefficient of the curve by 0.001 moves at
least one of these by more than 5e-7, several hundred times the tolerance
(the least sensitive is the highest-order optical coefficient).
"""
import numpy as np

from blrfit.model.extinction import ccm89_alav, deredden

# A_lambda / A_V at R_V = 3.1 on the reference stack
ALAV = {12500.0: 0.28206957499417296, 7000.0: 0.7537701382808429, 5500.0: 0.9986671323773466,
        4400.0: 1.322218733774922, 3000.0: 1.8181566974452663, 2000.0: 2.8425264357868345,
        1500.0: 2.663879214713716}


def test_ccm89_alav_pinned_values():
    w = np.array(sorted(ALAV))
    got = ccm89_alav(w)
    for wi, gi in zip(w, got):
        assert abs(gi - ALAV[wi]) <= 1e-9 * ALAV[wi], (wi, gi, ALAV[wi])
    assert np.all(np.diff(got[1:]) < 0)       # more extinction toward the blue up to the 2175 A bump
    assert got[0] < got[1]                    # 1500 A lies beyond the bump, below 2000 A
    assert ccm89_alav(np.array([5500.0]), rv=3.1)[0] == got[list(w).index(5500.0)]


def test_deredden_applies_the_curve():
    wave = np.array([3000.0, 4400.0, 5500.0, 7000.0])
    flux = np.array([1.0, 2.0, 3.0, 4.0]); ivar = np.array([4.0, 4.0, 1.0, 1.0])
    ebv = 0.0323                              # the DESI example's FIBERMAP value
    f, iv = deredden(wave, flux, ivar, ebv)
    corr = 10 ** (0.4 * 3.1 * ebv * ccm89_alav(wave))
    assert np.allclose(f, flux * corr, rtol=1e-12, atol=0)
    assert np.allclose(iv, ivar / corr ** 2, rtol=1e-12, atol=0)
    assert np.all(f > flux) and np.all(iv < ivar)
    assert abs(f[2] / flux[2] - 10 ** (0.4 * 3.1 * ebv * ALAV[5500.0])) < 1e-12


def test_deredden_is_the_identity_without_extinction():
    wave = np.array([4000.0, 5000.0]); flux = np.array([1.0, 2.0]); ivar = np.array([1.0, 4.0])
    for ebv in (0.0, -0.1, np.nan):
        f, iv = deredden(wave, flux, ivar, ebv)
        assert f is flux and iv is ivar
