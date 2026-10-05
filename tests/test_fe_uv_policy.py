"""
The ultraviolet Fe II width policy (``fe_uv_width_policy``, recorded in
``settings`` with the fallback width ``fe_uv_fallback_kms``).

Policy A holds the width at the fallback where fewer than 300 continuum
pixels cover 2200-3090 A, the rule of the 0.2.0 pins; it is the default and
reproduces the pins (bit for bit under BLRFIT_STRICT_PINS). Policy B leaves
the width free everywhere and records its bound state; policy C leaves it
free and refits with the width held at the fallback when the free width ends
on a bound (``continuum_info['feuv_refit']``, the discarded free pass kept as
``feuv_free_fit``). On spec-1237 (390 ultraviolet pixels, the free width on
the 10000 km/s bound under A) policy B is the A fit and policy C refits at the
fallback; on spec-1704 (239 pixels, held under A) policy B frees the width.
"""

import numpy as np
import pytest

import blrfit
from blrfit.constants import (
    FE_FWHM_MAX,
    FE_FWHM_MIN,
    FE_UV_FREE_MIN_PIXELS,
    FE_UV_FWHM_FIXED_KMS,
    FE_UV_WIDTH_POLICY,
)
from blrfit.io import read_sdss
from blrfit.model.continuum import FE_UV_WIDTH_POLICIES, fit_continuum, fit_continuum_host
from conftest import DATA
from synth import make_spectrum
from test_pins import (
    STRICT,
    _bit_exact_departures,
    _current_pins,
    _end_point_departures,
    _read_pinned_spectrum,
    _rest_frame_departures,
)

SPEC_1237 = DATA + "/spec-1237-52762-0298.fits"  # z = 0.479, 390 ultraviolet pixels
Z_1237 = 0.47870001196861267
SPEC_1704 = DATA + "/spec-1704-53178-0562.fits"  # z = 0.318, 239 ultraviolet pixels
Z_1704 = 0.31839999556541443


def _pins():
    return _current_pins()["pins"]


def test_default_policy_is_a():
    assert FE_UV_WIDTH_POLICY == "A" and FE_UV_WIDTH_POLICIES == ("A", "B", "C")


@pytest.mark.parametrize("pin", _pins(), ids=lambda p: p["file"])
def test_policy_a_reproduces_the_pins(pin):
    """The explicit policy A with the default fallback and the host guard is the
    pinned fit: classes, flags, component counts, c50_sys and chi-square within
    the tolerances of test_pins, bit for bit in strict mode."""
    sp = _read_pinned_spectrum(pin)
    res = blrfit.fit_spectrum(
        sp["wave"],
        sp["flux"],
        sp["ivar"],
        pin["z"],
        ebv=pin["ebv"],
        complexes=tuple(pin["complexes"]),
        fe_uv_width_policy="A",
        fe_uv_fallback_kms=FE_UV_FWHM_FIXED_KMS,
        host_guard=True,
    )
    row = blrfit.summary_row(res)
    departures = _end_point_departures(res, row, pin) + _rest_frame_departures(res, sp, pin)
    if STRICT:
        departures += _bit_exact_departures(res, row, pin)
    assert not departures, pin["file"] + ":\n  " + "\n  ".join(departures)
    s = res["settings"]
    assert (
        s["fe_uv_width_policy"] == "A"
        and s["fe_uv_fallback_kms"] == FE_UV_FWHM_FIXED_KMS
        and s["host_guard"] is True
    )
    info = res["continuum_info"]
    assert (
        info["feuv_policy"] == "A"
        and info["feuv_fallback_kms"] == FE_UV_FWHM_FIXED_KMS
        and info["feuv_refit"] is False
    )
    assert info["feuv_fwhm_fixed"] is (info["fe_uv"] and info["n_pix_uv"] < FE_UV_FREE_MIN_PIXELS)
    assert res["host_info"]["host_undetermined"] is False


def _fit(path, z, **kw):
    sp = read_sdss(path)
    return blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z, complexes=("Hbeta",), **kw)


def test_policy_b_on_the_wide_window_is_the_a_fit():
    a = _fit(SPEC_1237, Z_1237)
    b = _fit(SPEC_1237, Z_1237, fe_uv_width_policy="B")
    assert (
        a["continuum_info"]["n_pix_uv"] >= FE_UV_FREE_MIN_PIXELS
        and not a["continuum_info"]["feuv_fwhm_fixed"]
    )
    assert b["settings"]["fe_uv_width_policy"] == "B" and b["continuum_info"]["feuv_policy"] == "B"
    assert not b["continuum_info"]["feuv_fwhm_fixed"] and not b["continuum_info"]["feuv_refit"]
    assert b["conti"] == a["conti"]
    assert b["continuum_info"]["fe_width_state"]["feuv_fwhm"]["at_bound"]  # the 10000 km/s bound


def test_policy_c_refits_the_wide_window_at_the_fallback():
    c = _fit(SPEC_1237, Z_1237, fe_uv_width_policy="C")
    info = c["continuum_info"]
    assert c["settings"]["fe_uv_width_policy"] == "C" and info["feuv_policy"] == "C"
    assert info["feuv_refit"] is True and info["feuv_fwhm_fixed"] is True
    assert c["conti"]["feuv_fwhm"] == FE_UV_FWHM_FIXED_KMS
    assert (
        info["fe_width_state"]["feuv_fwhm"]["state"] == "fixed" and "feuv_fwhm" not in info["at_bound_widths"]
    )
    free = info["feuv_free_fit"]
    assert (
        free["fe_width_state"]["feuv_fwhm"]["at_bound"]
        and free["fe_width_state"]["feuv_fwhm"]["bound"] == "upper"
    )
    assert free["fe_widths"]["feuv_fwhm"] == pytest.approx(FE_FWHM_MAX, abs=1.0)
    assert c["cls"]["Hbeta"]["label"] in "ABCFWEX"


def test_policies_b_and_c_on_the_short_window():
    b = _fit(SPEC_1704, Z_1704, fe_uv_width_policy="B")
    c = _fit(SPEC_1704, Z_1704, fe_uv_width_policy="C")
    for res, pol in ((b, "B"), (c, "C")):
        info = res["continuum_info"]
        assert res["settings"]["fe_uv_width_policy"] == pol and info["feuv_policy"] == pol
        assert info["n_pix_uv"] < FE_UV_FREE_MIN_PIXELS
        assert res["host_info"]["feuv_policy"] == pol
        assert FE_FWHM_MIN <= res["conti"]["feuv_fwhm"] <= FE_FWHM_MAX
    assert not b["continuum_info"]["feuv_fwhm_fixed"] and not b["continuum_info"]["feuv_refit"]
    at_bound = b["continuum_info"]["fe_width_state"]["feuv_fwhm"]["at_bound"]
    assert c["continuum_info"]["feuv_refit"] is at_bound
    if at_bound:
        assert c["continuum_info"]["feuv_fwhm_fixed"] and c["conti"]["feuv_fwhm"] == FE_UV_FWHM_FIXED_KMS
    else:
        assert c["conti"] == b["conti"]


def _short_window(z=0.2, seed=1):
    s = make_spectrum(z=z, snr=20.0, seed=seed, narrow=dict(ew_ha=0.0))
    return s["wave"] / (1 + z), s["flux"] * (1 + z), s["ivar"] / (1 + z) ** 2


@pytest.mark.parametrize("fitter", [fit_continuum, fit_continuum_host])
def test_policies_on_a_synthetic_short_window(fitter):
    wr, fr, ir = _short_window()
    a = fitter(wr, fr, ir, fe_uv_width_policy="A")
    b = fitter(wr, fr, ir, fe_uv_width_policy="B")
    c = fitter(wr, fr, ir, fe_uv_width_policy="C")
    ia, ib, ic = a[-1], b[-1], c[-1]
    assert ia["n_pix_uv"] < FE_UV_FREE_MIN_PIXELS
    assert ia["feuv_fwhm_fixed"] and a[0]["feuv_fwhm"] == FE_UV_FWHM_FIXED_KMS and ia["feuv_policy"] == "A"
    assert (
        not ib["feuv_fwhm_fixed"]
        and ib["feuv_policy"] == "B"
        and not ib["fe_width_state"]["feuv_fwhm"]["fixed"]
    )
    if "ps" in ib:
        assert "feuv_fwhm" in ib["ps"].free_names
    assert ic["feuv_refit"] is ib["fe_width_state"]["feuv_fwhm"]["at_bound"]
    if ic["feuv_refit"]:
        assert ic["feuv_fwhm_fixed"] and c[0]["feuv_fwhm"] == FE_UV_FWHM_FIXED_KMS
        assert ic["feuv_free_fit"]["fe_widths"] == ib["fe_widths"]
    else:
        assert c[0] == b[0]
    # the default of the fitter is policy A
    assert fitter(wr, fr, ir)[0] == a[0]


def test_custom_fallback_is_used_and_recorded():
    wr, fr, ir = _short_window()
    d, model, info = fit_continuum(wr, fr, ir, fe_uv_fallback_kms=2500.0)
    assert info["feuv_fwhm_fixed"] and d["feuv_fwhm"] == 2500.0 and info["feuv_fallback_kms"] == 2500.0
    z = 0.2
    s = make_spectrum(
        z=z,
        snr=20.0,
        seed=1,
        narrow=dict(ew_ha=0.0),
        broad=[dict(line="Hbeta", v=500.0, fwhm=4000.0, ew=60.0)],
    )
    res = blrfit.fit_spectrum(
        s["wave"], s["flux"], s["ivar"], z, host=False, complexes=("Hbeta",), fe_uv_fallback_kms=2500.0
    )
    assert res["settings"]["fe_uv_fallback_kms"] == 2500.0 and res["conti"]["feuv_fwhm"] == 2500.0


@pytest.mark.parametrize(
    "kw",
    [
        dict(fe_uv_width_policy="D"),
        dict(fe_uv_width_policy="a"),
        dict(fe_uv_fallback_kms=np.nan),
        dict(fe_uv_fallback_kms=500.0),
        dict(fe_uv_fallback_kms=20000.0),
    ],
)
def test_invalid_policy_or_fallback_is_refused(kw):
    wr, fr, ir = _short_window()
    with pytest.raises(ValueError):
        fit_continuum(wr, fr, ir, **kw)
    with pytest.raises(ValueError):
        fit_continuum_host(wr, fr, ir, **kw)


@pytest.mark.parametrize("policy", ["A", "B", "C"])
def test_mc_uses_serialized_policy_and_custom_fallback(policy):
    """Refit real draws after a save/load; inspect the continuum actually used."""
    import pickle
    from blrfit import errors

    sp = make_spectrum(z=0.2, snr=20.0, seed=1, broad=[dict(line="Hbeta", v=500.0, fwhm=4000.0, ew=60.0)])
    res = blrfit.fit_spectrum(
        sp["wave"],
        sp["flux"],
        sp["ivar"],
        0.2,
        host=False,
        complexes=("Hbeta",),
        max_broad=1,
        fe_uv_width_policy=policy,
        fe_uv_fallback_kms=2500.0,
    )
    saved = pickle.loads(pickle.dumps(res))
    _, _, info = errors.monte_carlo(saved, nmc=2, seed=17, return_diagnostics=True)
    assert info["continuum_settings"] == dict(
        fe_uv_width_policy=policy, fe_uv_fallback_kms=2500.0, multistart=True
    )
    for draw in info["draws"]:
        assert "exception" not in draw
        actual = draw["continuum_policy"]
        assert actual["feuv_policy"] == policy and actual["feuv_fallback_kms"] == 2500.0
        state = actual["fe_width_state"]["feuv_fwhm"]
        if policy == "A":
            assert actual["feuv_fwhm_fixed"] and state["value"] == 2500.0
        if policy == "B":
            assert not actual["feuv_fwhm_fixed"]
        if policy == "C" and actual["feuv_refit"]:
            assert actual["feuv_fwhm_fixed"] and state["value"] == 2500.0
            assert actual["feuv_free_fit"]["fe_width_state"]["feuv_fwhm"]["at_bound"]


def test_joint_policy_c_keeps_refit_provenance(monkeypatch):
    """Exercise bookkeeping after a successful retained-host refit. Bound
    detection is independently tested above; force the trigger on this host."""
    from blrfit.model import continuum

    sp = make_spectrum(z=0.4, snr=30.0, seed=3, host_frac=0.5)
    wr, fr, ir = sp["wave"] / 1.4, sp["flux"] * 1.4, sp["ivar"] / 1.4**2
    monkeypatch.setattr(continuum, "_uv_at_bound", lambda info: True)
    d, total, host, info = fit_continuum_host(
        wr, fr, ir, fe_uv_width_policy="C", fe_uv_fallback_kms=2500.0, n_gal_max=1
    )
    assert info["applied"] and np.any(host)
    assert info["feuv_refit"] is True and info["feuv_free_fit"] is not None
    assert info["feuv_fwhm_fixed"] and d["feuv_fwhm"] == 2500.0
    assert info["feuv_free_fit"]["fe_width_state"]["feuv_fwhm"]["fixed"] is False
