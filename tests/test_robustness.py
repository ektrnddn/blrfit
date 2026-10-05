"""
Unusable pixels, line parameters on a bound and residual outliers.

* The mask is grown by MASK_GROW_PIX pixels around every unusable pixel. The DESI
  example coadd holds six unmasked pixels between two masked pairs at
  6138-6142 A, 15-20 sigma above their neighbours; without the growth the
  [O III] wing ends on its +500 km/s bound and Hbeta has a reduced chi-square
  of 2.9.
* A free line parameter that ends on a bound is listed in ``params_at_bound``;
  amplitudes and the wing fraction at zero, and the parameters of components
  with no data under them or of zero amplitude, are left out. Only a velocity on
  a bound flags the line ``param_at_bound``.
* Pixels more than RESIDUAL_OUTLIER_SIGMA off the model, away from the narrow
  lines, are counted in ``n_residual_outliers``; RESIDUAL_OUTLIER_MIN_PIX or more
  flag ``residual_outliers``.
"""

import numpy as np
import pytest

import blrfit
from blrfit.constants import C_KMS, LAM, MASK_GROW_PIX, RESIDUAL_OUTLIER_MIN_PIX
from blrfit.io import read_desi, read_sdss
from blrfit.model.fit import grow_mask
from blrfit.model.lines import params_at_bound, unconstrained_parameters
from blrfit.model.params import ParamSet
from conftest import DESI_EXAMPLE, DESI_TARGETID, SDSS_EXAMPLE, Z_J001224
from synth import make_spectrum


def test_grow_mask():
    bad = np.zeros(12, bool)
    bad[5] = True
    assert np.flatnonzero(grow_mask(bad, 2)).tolist() == [3, 4, 5, 6, 7]
    edge = np.zeros(6, bool)
    edge[0] = True
    assert np.flatnonzero(grow_mask(edge, 2)).tolist() == [0, 1, 2]
    assert np.array_equal(grow_mask(bad, 0), bad)
    assert not grow_mask(np.zeros(5, bool), 2).any()


@pytest.fixture(scope="module")
def desi_example():
    sp = read_desi(DESI_EXAMPLE, DESI_TARGETID, use_desispec=False)

    def fit(grow):
        return blrfit.fit_spectrum(
            sp["wave"],
            sp["flux"],
            sp["ivar"],
            sp["z"],
            ebv=sp["ebv"],
            complexes=("Halpha", "Hbeta"),
            mask_grow=grow,
        )

    return fit(0), fit(MASK_GROW_PIX)


def test_desi_artefact_with_and_without_the_grown_mask(desi_example):
    old, new = desi_example
    mo, mn = old["meas"]["Hbeta"], new["meas"]["Hbeta"]
    # without the growth: the wing on its upper bound, the artefact left as residual outliers, a poor fit
    assert "w_v:upper" in mo["params_at_bound"]
    assert mo["n_residual_outliers"] >= RESIDUAL_OUTLIER_MIN_PIX
    assert {"poor_fit", "param_at_bound", "residual_outliers"} <= set(old["cls"]["Hbeta"]["flags"])
    # with it: a good fit and a free wing
    assert mn["chi2_red"] < 2.5 and "poor_fit" not in new["cls"]["Hbeta"]["flags"]
    assert not any(p.startswith("w_v") for p in mn["params_at_bound"])
    assert mn["n_residual_outliers"] < RESIDUAL_OUTLIER_MIN_PIX
    assert new["settings"]["mask_grow"] == MASK_GROW_PIX and old["settings"]["mask_grow"] == 0
    assert np.sum(new["ivar_rest"] == 0) > np.sum(old["ivar_rest"] == 0)


def test_he_ii_outside_the_hbeta_window_is_not_counted():
    sp = read_sdss(SDSS_EXAMPLE)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Hbeta",))
    r = res["fits"]["Hbeta"]
    assert "HeII_n_A" in unconstrained_parameters(r["comps"], r["d"], r["x"])
    assert not any(p.startswith("HeII_n_A") for p in res["meas"]["Hbeta"]["params_at_bound"])


def test_params_at_bound_rules():
    ps = ParamSet()
    ps.add("Ha_b0_A", 1.0, 0.0, 10.0)
    ps.add("nw_f", 0.0, 0.0, 0.5)
    ps.add("w_v", 0.0, -2500.0, 500.0)
    ps.add("n_sig", 100.0, 25.0, 509.6)
    # amplitudes and the wing fraction at zero are normal; a velocity on its bound is not
    assert params_at_bound(ps, np.array([0.0, 0.0, 500.0, 100.0]), {}) == ["w_v:upper"]
    assert params_at_bound(ps, np.array([10.0, 0.5, 0.0, 25.0]), {}) == [
        "Ha_b0_A:upper",
        "nw_f:upper",
        "n_sig:lower",
    ]
    # the solver's active set counts as well, and skipped parameters never do
    assert params_at_bound(ps, np.array([5.0, 0.2, 0.0, 100.0]), {"w_v": 1}) == ["w_v:upper"]
    assert params_at_bound(ps, np.array([10.0, 0.2, 0.0, 100.0]), {}, skip={"Ha_b0_A"}) == []


def test_residual_outliers_away_from_the_narrow_lines():
    z = 0.25
    s = make_spectrum(
        z=z,
        snr=25.0,
        seed=5,
        broad=[dict(line="Halpha", v=0.0, fwhm=4000.0, ew=120.0)],
        narrow=dict(ew_ha=20.0),
    )

    def spiked(v_kms):
        flux = s["flux"].copy()
        lam = LAM["Halpha"] * (1 + z) * (1 + v_kms / C_KMS)
        i = int(np.argmin(np.abs(s["wave"] - lam)))
        flux[i : i + 4] += 25.0 / np.sqrt(s["ivar"][i : i + 4])  # four pixels 25 sigma high
        res = blrfit.fit_spectrum(s["wave"], flux, s["ivar"], z, complexes=("Halpha",))
        return res["meas"]["Halpha"], res["cls"]["Halpha"]["flags"]

    far, far_flags = spiked(-4000.0)
    near, near_flags = spiked(100.0)  # on narrow Halpha
    assert far["n_residual_outliers"] >= RESIDUAL_OUTLIER_MIN_PIX and "residual_outliers" in far_flags
    assert near["n_residual_outliers"] < RESIDUAL_OUTLIER_MIN_PIX and "residual_outliers" not in near_flags


def test_only_velocities_on_a_bound_flag_the_line(desi_example):
    from blrfit.classify import classify

    new = desi_example[1]
    ha = dict(new["meas"]["Halpha"])
    # the narrow-line-region wing fraction on its upper bound is listed but does not flag the line
    assert "nw_f:upper" in ha["params_at_bound"] and "param_at_bound" not in new["cls"]["Halpha"]["flags"]
    for listed, flagged in (
        (["nw_sig:upper", "Ha_b0_sig:upper"], False),
        (["Ha_b1_v:lower"], True),
        (["w_v:upper", "w_sig:lower"], True),
        ([], False),
    ):
        ha["params_at_bound"] = listed
        assert ("param_at_bound" in classify(ha)["flags"]) is flagged, listed
