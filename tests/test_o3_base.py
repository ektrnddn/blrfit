"""
The broad [O III] base of the Hbeta complex (``o3_base``; O3_BASE_* in constants.py).

Where [O III] has a base broader than its wing, the model without the base takes
the base flux between [O III] 4959 and 5007 with a broad Hbeta Gaussian at the
width limit of the width-offset hinge, and the broad profile gains a false red
peak near +7500 km/s. The synthetic spectrum below reproduces that failure
(z = 0.6, the Hbeta complex alone, broad Hbeta of FWHM 4000 km/s at rest, [O III]
ten times narrow Hbeta with a wing at -300 km/s and a base of sigma 900 km/s at
-1200 km/s carrying 40 per cent of the core flux): without the rule it is class B
with a red peak at +7400 km/s and c(1/2) = +3400 km/s, with the rule class F with
c(1/2) within 15 km/s of the true zero. A spectrum without a base must come out
exactly as without the rule.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import blrfit
from blrfit import errors
from blrfit.constants import (
    LAM,
    O3_BASE_DBIC,
    O3_BASE_TRIGGER_KMS,
    O3_BASE_TRIGGER_WIDTH,
    S2F,
    SIG_O3_BASE_MIN,
    SIG_O3_BASE_MAX,
    V_O3_BASE_MIN,
    V_O3_BASE_MAX,
)
from blrfit.measure import broad_profile
from blrfit.model.broad import o3_base_trigger
from blrfit.model.fit import summary_row
from blrfit.model.lines import build_complex
from blrfit.model.narrow import oiii_order_penalty
from scipy.signal import find_peaks
from synth import make_spectrum

Z = 0.6
BROAD = [dict(line="Hbeta", v=0.0, fwhm=4000.0, ew=80.0)]
BASE_NARROW = dict(ew_ha=60.0, o3=10.0, o3_wing=(-300.0, 400.0, 0.3), o3_base=(-1200.0, 900.0, 0.4))
_FITS = {}


def fit(narrow, o3_base):
    """The Hbeta fit of the synthetic spectrum with the given narrow lines, cached."""
    key = (tuple(sorted((k, str(v)) for k, v in narrow.items())), o3_base)
    if key not in _FITS:
        sp = make_spectrum(z=Z, snr=20.0, seed=1, broad=BROAD, narrow=narrow)
        _FITS[key] = blrfit.fit_spectrum(
            sp["wave"], sp["flux"], sp["ivar"], Z, complexes=("Hbeta",), o3_base=o3_base
        )
    return _FITS[key]


def peaks_sys(res):
    """Peak velocities of the broad Hbeta model, km/s from the systemic."""
    r = res["fits"]["Hbeta"]
    vg = np.arange(-25000.0, 25000.01, 5.0)
    P = broad_profile(r, LAM["Hbeta"], vg)
    pk, _ = find_peaks(P, height=0.2 * P.max(), prominence=0.05 * P.max())
    return vg[pk] - r["d"]["n_v"]


# ---------------------------------------------------------------------------
# the parts
# ---------------------------------------------------------------------------
def _broad_comps(*pairs):
    d, comps = {"n_v": 0.0}, []
    for k, (v, fwhm) in enumerate(pairs):
        d.update({f"Hb_b{k}_A": 1.0, f"Hb_b{k}_v": v, f"Hb_b{k}_sig": fwhm / S2F})
        comps.append(
            (f"Hb_broad{k}", LAM["Hbeta"], f"Hb_b{k}_A", f"Hb_b{k}_v", f"Hb_b{k}_sig", "broad", None)
        )
    return d, comps


def test_trigger_needs_a_far_and_narrow_broad_gaussian():
    """Only a broad Gaussian beyond O3_BASE_TRIGGER_KMS from the systemic and narrower
    than O3_BASE_TRIGGER_WIDTH |v| triggers the rule; the narrowest one is reported."""
    v = O3_BASE_TRIGGER_KMS + 1500.0
    d, comps = _broad_comps((0.0, 4000.0), (v, 0.41 * v))
    vs, ratio = o3_base_trigger(d, comps)
    assert vs == pytest.approx(v) and ratio == pytest.approx(0.41)
    wide = (O3_BASE_TRIGGER_WIDTH + 0.05) * v
    assert o3_base_trigger(*_broad_comps((0.0, 4000.0), (v, wide))) is None  # a broad red wing
    near = O3_BASE_TRIGGER_KMS - 500.0
    assert o3_base_trigger(*_broad_comps((0.0, 4000.0), (near, 2000.0))) is None  # not far enough
    assert o3_base_trigger(*_broad_comps((0.0, 4000.0), (-v, 0.41 * v))) is None  # blue side
    d, comps = _broad_comps((v, 0.6 * v), (v + 300.0, 0.45 * (v + 300.0)))
    assert o3_base_trigger(d, comps)[1] == pytest.approx(0.45)
    # the distance is taken from the systemic velocity
    d, comps = _broad_comps((O3_BASE_TRIGGER_KMS + 200.0, 0.41 * (O3_BASE_TRIGGER_KMS + 200.0)))
    d["n_v"] = 400.0
    assert o3_base_trigger(d, comps) is None


def test_base_components_and_bounds():
    """The base is a pair of 'wing' components on both [O III] lines with the 5007/4959
    ratio of the core, inside its bounds; it is absent unless asked for, and the Halpha
    complex ignores the request."""
    wave = np.arange(4700.0, 5100.0, 0.5)
    fsub = np.ones_like(wave)
    ps0, comps0 = build_complex("Hbeta", wave, fsub, n_broad=1)
    assert "wb_sig" not in ps0.names and not any(c[0].startswith("OIII5007b") for c in comps0)
    ps, comps = build_complex("Hbeta", wave, fsub, n_broad=1, oiii_base=True)
    base = {c[0]: c for c in comps if c[0] in ("OIII5007b", "OIII4959b")}
    assert set(base) == {"OIII5007b", "OIII4959b"} and all(c[5] == "wing" for c in base.values())
    assert base["OIII4959b"][6][0] == "OIII5007b_A" and base["OIII4959b"][6][1] == pytest.approx(1 / 2.98)
    i_v, i_s = ps.names.index("wb_v"), ps.names.index("wb_sig")
    assert (ps.lb[i_v], ps.ub[i_v]) == (V_O3_BASE_MIN, V_O3_BASE_MAX)
    assert (ps.lb[i_s], ps.ub[i_s]) == (SIG_O3_BASE_MIN, SIG_O3_BASE_MAX)
    wave_ha = np.arange(6400.0, 6800.0, 0.5)
    psa, _ = build_complex("Halpha", wave_ha, np.ones_like(wave_ha), n_broad=1, oiii_base=True)
    assert "wb_sig" not in psa.names


def test_order_penalty_keeps_the_wing_inside_the_base():
    d = dict(o3_sig=150.0, w_sig=400.0, OIII5007c_A=10.0, OIII5007w_A=3.0)
    base = oiii_order_penalty(d)
    assert oiii_order_penalty(dict(d, wb_sig=900.0)) == base + [0.0]
    assert oiii_order_penalty(dict(d, w_sig=1000.0, wb_sig=900.0))[-1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# the rule, end to end
# ---------------------------------------------------------------------------
def test_without_the_rule_the_base_makes_a_false_red_peak():
    """The failure the rule corrects: without the base, a broad Gaussian between the
    [O III] lines at the hinge limit, a red peak beyond +5500 km/s, class B and a
    red offset of thousands of km/s (truth: one Gaussian at rest)."""
    res = fit(BASE_NARROW, o3_base=False)
    r, m = res["fits"]["Hbeta"], res["meas"]["Hbeta"]
    assert "o3_base" not in r and res["settings"]["o3_base"] is False
    assert not any("o3_base" in k or "c50_sys_alt" in k for k in summary_row(res))  # the columns of 0.5
    trig = o3_base_trigger(r["d"], r["comps"])
    assert trig is not None and trig[1] < 0.45, trig
    assert peaks_sys(res).max() > O3_BASE_TRIGGER_KMS
    assert res["cls"]["Hbeta"]["label"] == "B"
    assert m["c50_sys"] > 1000.0, m["c50_sys"]


def test_with_the_rule_the_base_is_kept_and_the_profile_recovered():
    res = fit(BASE_NARROW, o3_base=True)
    stock = fit(BASE_NARROW, o3_base=False)
    r, m = res["fits"]["Hbeta"], res["meas"]["Hbeta"]
    rec = r["o3_base"]
    assert res["settings"]["o3_base"] is True
    assert rec["triggered"] and rec["kept"] and rec["dbic"] <= O3_BASE_DBIC
    assert any(c[0] == "OIII5007b" for c in r["comps"])
    assert rec["trigger_fwhm_ratio"] < O3_BASE_TRIGGER_WIDTH and rec["trigger_v_sys"] > O3_BASE_TRIGGER_KMS
    assert peaks_sys(res).max() < O3_BASE_TRIGGER_KMS
    assert abs(m["c50_sys"]) < 80.0, m["c50_sys"]  # the truth is 0
    assert abs(m["fwhm"] - 4000.0) < 400.0, m["fwhm"]
    assert res["cls"]["Hbeta"]["label"] == "F", res["cls"]["Hbeta"]
    # the alternative recorded is the fit without the base
    assert rec["c50_sys_alt"] == pytest.approx(stock["meas"]["Hbeta"]["c50_sys"], abs=1e-3)
    assert rec["n_broad_alt"] == stock["fits"]["Hbeta"]["n_broad"]
    row = summary_row(res)
    assert row["HB_o3_base"] == 1.0 and row["HB_o3_base_trigger"] == 1.0
    assert row["HB_o3_base_dbic"] == pytest.approx(rec["dbic"])
    assert row["HB_c50_sys_alt"] == pytest.approx(rec["c50_sys_alt"])
    assert res["fit_status"]["Hbeta"]["o3_base"]["kept"] is True


def test_a_spectrum_without_a_base_is_not_touched():
    """No far Gaussian, no trigger: the fit is the fit without the rule, number for number."""
    narrow = dict(ew_ha=60.0, o3=10.0, o3_wing=(-300.0, 400.0, 0.3))
    res, stock = fit(narrow, o3_base=True), fit(narrow, o3_base=False)
    rec = res["fits"]["Hbeta"]["o3_base"]
    assert not rec["triggered"] and not rec["kept"] and np.isnan(rec["dbic"])
    assert not any(c[0] == "OIII5007b" for c in res["fits"]["Hbeta"]["comps"])
    for k in ("c50_sys", "fwhm", "v_sys", "chi2_red"):
        assert res["meas"]["Hbeta"][k] == stock["meas"]["Hbeta"][k], k
    assert res["meas"]["Hbeta"]["o3_base"] == 0.0 and res["meas"]["Hbeta"]["o3_base_trigger"] == 0.0


def test_monte_carlo_draws_keep_the_base(monkeypatch):
    """The draws refit with the base where the fit kept it: noise-free draws return the
    fit's own c(1/2)."""
    res = fit(BASE_NARROW, o3_base=True)
    monkeypatch.setattr(
        errors.np.random, "default_rng", lambda seed: SimpleNamespace(standard_normal=lambda n: np.zeros(n))
    )
    mc, err, info = errors.monte_carlo(res, nmc=5, return_diagnostics=True)
    assert info["line_settings"].get("oiii_base") is True
    assert info["lines"]["Hbeta"]["n_success"] == 5
    np.testing.assert_allclose(mc["Hbeta"]["c50_sys"], res["meas"]["Hbeta"]["c50_sys"], atol=1e-6, rtol=0)
