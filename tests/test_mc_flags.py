"""
Monte Carlo draw-sample flags: the two-cluster test, the basin switch and the
draw-count guard.

The two-cluster test is the median-separation statistic of ``errors``: the
sorted contributing draws are split at their largest gap, both sides must hold
at least a tenth of the draws, and their medians must differ by at least 400
km/s and at least four times hypot(NMAD_left, NMAD_right). Its operating
curve is measured here on synthetic draw sets (seed 20260916): unimodal
Gaussian, Student-t (nu = 3) and skewed families for the false-flag rate,
80/20 mixtures of two sigma = 100 km/s components for the detection rate, at
30 draws (the production number) and at 25 (the smallest number the test
runs on). The full curve, 20,000 sets per case, is slow; the fast suite runs
2,000 sets per case with wide margins. Every rate is printed with its exact
(Clopper-Pearson) 95 per cent interval. The gates sit 2-3 points below the
measured curve: the flag is a documented diagnostic with these sensitivities,
not a validated detector. Skewed sets are reported, not gated.
"""

import copy

import numpy as np
import pytest
from scipy.stats import beta

import blrfit
from blrfit import errors
from blrfit.constants import MC_ALIAS_KMS
from blrfit.model import fit as fit_module
from synth import make_spectrum

SEED = 20260916
SEP_MIN = MC_ALIAS_KMS
RATIO = errors.MC_SEPARATION_NMAD_RATIO
ALIAS_KMS = 943.5  # narrow Halpha taken for [N II] 6584

# ---------------------------------------------------------------------------
# The statistic on constructed sets
# ---------------------------------------------------------------------------


def test_nmad():
    assert errors.nmad([1.0, 1.0, 1.0]) == 0.0
    assert np.isnan(errors.nmad([]))
    x = np.random.default_rng(0).standard_normal(100000)
    assert abs(errors.nmad(x) - 1.0) < 0.02


def test_two_point_clusters_qualify_by_fraction():
    """24/6 and 27/3 of 30 qualify (3 is a tenth); 28/2 does not."""
    s = errors.cluster_summary([0.0] * 24 + [800.0] * 6)
    assert s["qualifies"] and s["n_left"] == 24 and s["n_right"] == 6
    assert s["gap_kms"] == 800.0 and s["separation_kms"] == 800.0
    assert s["median_left"] == 0.0 and s["median_right"] == 800.0
    assert s["nmad_left"] == 0.0 and s["nmad_right"] == 0.0
    assert s["width_kms"] == errors.MC_NMAD_FLOOR_KMS
    assert errors.is_multimodal([0.0] * 27 + [800.0] * 3)
    assert not errors.is_multimodal([0.0] * 28 + [800.0] * 2)
    # At 25 draws a tenth is 2.5: three on the small side qualify, two do not.
    assert errors.is_multimodal([0.0] * 22 + [800.0] * 3)
    assert not errors.is_multimodal([0.0] * 23 + [800.0] * 2)


def test_separation_thresholds():
    """The medians must differ by SEP_MIN and by RATIO times the cluster width."""
    assert errors.is_multimodal([0.0] * 24 + [SEP_MIN] * 6)
    assert not errors.is_multimodal([0.0] * 24 + [SEP_MIN - 1.0] * 6)
    rng = np.random.default_rng(3)
    # Two clusters 1000 km/s apart of width 100 km/s each: sep / width about 7.
    left, right = rng.normal(0.0, 100.0, 24), rng.normal(1000.0, 100.0, 6)
    s = errors.cluster_summary(np.r_[left, right])
    assert s["qualifies"] and s["separation_kms"] > RATIO * s["width_kms"]
    # The same medians with clusters wide enough that sep < RATIO x width.
    wide = np.r_[np.linspace(-600.0, 600.0, 24), np.linspace(700.0, 1300.0, 6)]
    s = errors.cluster_summary(wide)
    assert s["separation_kms"] >= SEP_MIN
    assert s["separation_kms"] < RATIO * s["width_kms"]
    assert not s["qualifies"]


def test_summary_ignores_non_finite_and_short_samples():
    s = errors.cluster_summary([np.nan, 1.0, np.inf])
    assert s["n"] == 1 and not s["qualifies"] and np.isnan(s["gap_kms"])
    s = errors.cluster_summary([])
    assert s["n"] == 0 and np.isnan(s["nmad_all"])
    s = errors.cluster_summary([0.0] * 10 + [np.nan] * 5 + [800.0] * 5)
    assert s["n"] == 15 and s["n_right"] == 5 and s["qualifies"]


def test_basin_switch_rule():
    draws = np.zeros(30) + 900.0
    b = errors.basin_switch(draws, 0.0)
    assert b["assessed"] and b["switched"]
    assert b["shift_kms"] == 900.0 and b["threshold_kms"] == MC_ALIAS_KMS
    # Below 400 km/s: not a switch. Wide draws: the threshold rises with the NMAD.
    assert not errors.basin_switch(np.zeros(30) + 399.0, 0.0)["switched"]
    rng = np.random.default_rng(4)
    wide = rng.normal(500.0, 400.0, 30)
    b = errors.basin_switch(wide, 0.0)
    assert b["threshold_kms"] == pytest.approx(
        max(MC_ALIAS_KMS, errors.MC_BASIN_SWITCH_NMAD_RATIO * b["nmad"])
    )
    assert b["switched"] == (b["shift_kms"] > b["threshold_kms"])
    # Not assessed: too few draws or no estimate.
    assert not errors.basin_switch(np.zeros(4) + 900.0, 0.0)["assessed"]
    assert not errors.basin_switch(draws, np.nan)["assessed"]


# ---------------------------------------------------------------------------
# Operating curve
# ---------------------------------------------------------------------------

DETECT_SEPS = (500.0, 800.0, 1200.0)
FAMILIES = {
    "gaussian_200": lambda rng, n: rng.normal(0.0, 200.0, n),
    "gaussian_500": lambda rng, n: rng.normal(0.0, 500.0, n),
    "gaussian_1000": lambda rng, n: rng.normal(0.0, 1000.0, n),
    "student_t3_500": lambda rng, n: 500.0 * rng.standard_t(3, n),
    "student_t3_1000": lambda rng, n: 1000.0 * rng.standard_t(3, n),
    "skewed_gamma2_500": lambda rng, n: 500.0 * (rng.gamma(2.0, 1.0, n) - 2.0) / np.sqrt(2.0),
}
GATED_FAMILIES = ("gaussian_200", "gaussian_500", "gaussian_1000", "student_t3_500", "student_t3_1000")
ENDPOINTS = (30, 25)


def mixture(rng, n, sep, sig=100.0, f=0.2):
    k = rng.binomial(n, f)
    return np.concatenate([rng.normal(0.0, sig, n - k), rng.normal(sep, sig, k)])


def clopper_pearson(k, n, conf=0.95):
    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a / 2.0, k + 1, n - k))
    return lo, hi


def measured_rate(gen, ndraw, nsets, case_index):
    """Flag rate of ``errors.is_multimodal`` over ``nsets`` sets of ``ndraw``
    draws from ``gen(rng, ndraw)``, with its exact 95 per cent interval."""
    rng = np.random.default_rng([SEED, case_index])
    hits = sum(errors.is_multimodal(gen(rng, ndraw)) for _ in range(nsets))
    lo, hi = clopper_pearson(hits, nsets)
    return dict(rate=hits / nsets, hits=hits, n=nsets, lo=lo, hi=hi)


def operating_curve(nsets):
    """Detection and false-flag rates at both endpoints; case seeds are
    (SEED, case index) in the fixed order below."""
    out = {}
    i = 0
    for ndraw in ENDPOINTS:
        for sep in DETECT_SEPS:
            out[(ndraw, f"mixture_80_20_sigma100_sep{int(sep)}")] = measured_rate(
                lambda rng, n, s=sep: mixture(rng, n, s), ndraw, nsets, i
            )
            i += 1
        for label, gen in FAMILIES.items():
            out[(ndraw, label)] = measured_rate(gen, ndraw, nsets, i)
            i += 1
    for (ndraw, label), r in out.items():
        print(
            f"{ndraw} draws {label:32s} {100 * r['rate']:6.2f} %  "
            f"[{100 * r['lo']:.2f}, {100 * r['hi']:.2f}]  ({r['hits']} of {r['n']})"
        )
    return out


def check_curve(curve, false_max, detect_min):
    """``detect_min`` maps ndraw -> {sep: minimum rate}; ``false_max`` is the
    ceiling on the gated families at either endpoint."""
    for ndraw in ENDPOINTS:
        for label in GATED_FAMILIES:
            r = curve[(ndraw, label)]
            assert r["rate"] <= false_max, (ndraw, label, r)
        for sep, floor in detect_min[ndraw].items():
            r = curve[(ndraw, f"mixture_80_20_sigma100_sep{int(sep)}")]
            assert r["rate"] >= floor, (ndraw, sep, r)


def test_operating_curve_fast():
    """2,000 sets per case; margins several sampling errors wide."""
    curve = operating_curve(2000)
    check_curve(
        curve, false_max=0.03, detect_min={30: {800.0: 0.84, 1200.0: 0.90}, 25: {800.0: 0.79, 1200.0: 0.84}}
    )


@pytest.mark.slow
def test_operating_curve():
    """The measured curve of the contract: 20,000 sets per case, seed 20260916.
    False flags at most 1.5 per cent on the Gaussian and Student-t families;
    detection at least 88 per cent at 800 km/s and 94 at 1,200 for 30 draws,
    83 and 88 for 25 (measured: 90.9 / 95.6 and 85.5 / 90.0)."""
    curve = operating_curve(20000)
    check_curve(
        curve, false_max=0.015, detect_min={30: {800.0: 0.88, 1200.0: 0.94}, 25: {800.0: 0.83, 1200.0: 0.88}}
    )
    # The 500 km/s separation is reported only (about 42 per cent).
    assert 0.3 < curve[(30, "mixture_80_20_sigma100_sep500")]["rate"] < 0.55


# ---------------------------------------------------------------------------
# Through monte_carlo
# ---------------------------------------------------------------------------

BROAD = [
    dict(line="Halpha", v=2000.0, fwhm=5000.0, ew=150.0),
    dict(line="Hbeta", v=2000.0, fwhm=5000.0, ew=40.0),
]


@pytest.fixture(scope="module")
def fitted():
    sp = make_spectrum(snr=18.0, v_sys=700.0, narrow=dict(ew_ha=1.2, o3=100.0), seed=93, broad=BROAD)
    return blrfit.fit_spectrum(
        sp["wave"],
        sp["flux"],
        sp["ivar"],
        0.25,
        host=False,
        fe=False,
        max_broad=1,
        err_floor=0.0,
        complexes=("Halpha", "Hbeta"),
    )


def _sequence_with_pattern(res, shifts, fail=()):
    """A stand-in for the line sequence returning the unperturbed measures with
    the Halpha offsets displaced by shifts[i] on draw i; draws in ``fail``
    return a solver failure for every line."""
    calls = []

    def sequence(wr, fsub, ir, cmodel, host_model, z, complexes, **kw):
        i = len(calls)
        calls.append(1)
        prefit = dict(res["o3_prefit"], status="success")
        if i in fail:
            return {}, {}, prefit, {name: dict(status="solver_failed", attempts=[]) for name in res["fits"]}
        meas = copy.deepcopy(res["meas"])
        dv, dc = shifts[i % len(shifts)]
        meas["Halpha"]["v_sys"] += dv
        meas["Halpha"]["c50_sys"] += dc
        return {}, meas, prefit, {name: dict(status="success") for name in meas}

    return sequence


def recomputed_flag(line_info):
    """The flag from the persisted cluster summary alone."""
    if not line_info["cluster_test_run"]:
        return False
    hit = False
    for c in line_info["clusters"].values():
        w = max(np.hypot(c["nmad_left"], c["nmad_right"]), errors.MC_NMAD_FLOOR_KMS)
        hit |= bool(
            min(c["n_left"], c["n_right"]) >= errors.MC_CLUSTER_MIN_FRACTION * c["n"]
            and c["separation_kms"] >= MC_ALIAS_KMS
            and c["separation_kms"] >= RATIO * w
        )
    return hit


def test_alias_mixture_is_flagged_and_recomputable(fitted, monkeypatch):
    """Four draws in five in the true basin, one in five at the [N II] alias."""
    pattern = [(0.0, 0.0)] * 4 + [(ALIAS_KMS, -ALIAS_KMS)]
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(fitted, pattern))
    mc, err, info = errors.monte_carlo(fitted, nmc=30, return_diagnostics=True)
    assert info["cluster_test"]["statistic"] == "median_separation"
    assert info["cluster_test"]["min_contributing"] == errors.MC_MIN_CONTRIBUTING
    ha = info["lines"]["Halpha"]
    assert ha["n_success"] == 30 and ha["cluster_test_run"]
    assert ha["alias_count"] == 6 and ha["alias_fraction"] == pytest.approx(0.2)
    assert "mc_multimodal" in ha["flags"] and ha["bimodal"]
    assert "mc_too_few" not in ha["flags"] and "mc_basin_switch" not in ha["flags"]
    for key in ("v_sys", "c50_sys"):
        c = ha["clusters"][key]
        assert c["n"] == 30 and {c["n_left"], c["n_right"]} == {24, 6}
        assert c["separation_kms"] == pytest.approx(ALIAS_KMS)
        assert c["gap_kms"] == pytest.approx(ALIAS_KMS)
        assert c["qualifies"]
    assert recomputed_flag(ha) is True
    assert ha["mc_gap_kms"] == pytest.approx(ALIAS_KMS)
    assert ha["mc_sigma"] == 0.0
    # The percentile error spans both basins and is not a statistical error.
    assert err["Halpha"]["c50_sys"] > MC_ALIAS_KMS
    hb = info["lines"]["Hbeta"]
    assert "mc_multimodal" not in hb["flags"] and not hb["bimodal"]
    assert hb["alias_fraction"] == 0.0 and recomputed_flag(hb) is False


def test_unimodal_scatter_is_not_flagged(fitted, monkeypatch):
    """A wide unimodal scatter (500 km/s) is an honest error: no flag, and
    mc_sigma reports its width."""
    rng = np.random.default_rng(11)
    dc = rng.normal(0.0, 500.0, 30)
    pattern = [(0.0, float(x)) for x in dc]
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(fitted, pattern))
    mc, err, info = errors.monte_carlo(fitted, nmc=30, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["cluster_test_run"] and ha["flags"] == []
    assert not ha["bimodal"] and recomputed_flag(ha) is False
    assert ha["mc_sigma"] == pytest.approx(errors.nmad(dc))
    assert ha["mc_gap_kms"] == pytest.approx(np.max(np.diff(np.sort(dc))))
    assert ha["clusters"]["v_sys"]["gap_kms"] == 0.0
    assert ha["alias_count"] == int(np.sum(np.abs(dc) > MC_ALIAS_KMS))


def test_basin_switch_is_flagged(fitted, monkeypatch):
    """Every draw settles in the alias basin: the median moves, the flag is
    mc_basin_switch, not mc_multimodal."""
    monkeypatch.setattr(
        fit_module, "_fit_line_sequence", _sequence_with_pattern(fitted, [(ALIAS_KMS, -ALIAS_KMS)])
    )
    mc, err, info = errors.monte_carlo(fitted, nmc=30, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert "mc_basin_switch" in ha["flags"] and "mc_multimodal" not in ha["flags"]
    assert ha["alias_fraction"] == 1.0
    for key, sign in (("v_sys", 1.0), ("c50_sys", -1.0)):
        b = ha["basin_switch"][key]
        assert b["assessed"] and b["switched"]
        assert b["estimate"] == fitted["meas"]["Halpha"][key]
        assert b["median"] == pytest.approx(fitted["meas"]["Halpha"][key] + sign * ALIAS_KMS)
        assert b["shift_kms"] == pytest.approx(ALIAS_KMS) and b["threshold_kms"] == MC_ALIAS_KMS
    assert err["Halpha"]["c50_sys"] == 0.0
    assert not info["lines"]["Hbeta"]["basin_switch"]["c50_sys"]["switched"]


def test_too_few_draws_replaces_the_cluster_test(fitted, monkeypatch):
    """Below MC_MIN_CONTRIBUTING contributing draws the two-cluster test does not
    run: mc_too_few, no mc_multimodal, the summary still recorded."""
    pattern = [(0.0, 0.0)] * 4 + [(ALIAS_KMS, -ALIAS_KMS)]
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(fitted, pattern))
    mc, err, info = errors.monte_carlo(fitted, nmc=10, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["n_success"] == 10 and not ha["cluster_test_run"]
    assert ha["n_contributing_required"] == errors.MC_MIN_CONTRIBUTING == 25
    assert "mc_too_few" in ha["flags"] and "mc_multimodal" not in ha["flags"]
    assert not ha["bimodal"]
    assert ha["clusters"]["c50_sys"]["qualifies"]  # the statistic itself, kept for inspection
    assert ha["alias_fraction"] == pytest.approx(0.2)
    # 30 requested, six failed: 24 contribute, one short of the test.
    monkeypatch.setattr(
        fit_module, "_fit_line_sequence", _sequence_with_pattern(fitted, pattern, fail=range(6))
    )
    mc, err, info = errors.monte_carlo(fitted, nmc=30, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["n_success"] == 24 and ha["n_failed"] == 6
    assert "failed_draws" in ha["flags"] and "mc_too_few" in ha["flags"]
    assert "mc_multimodal" not in ha["flags"]
    assert ha["clusters"]["c50_sys"]["n"] == 24
    # Five failed: 25 contribute and the test runs.
    monkeypatch.setattr(
        fit_module, "_fit_line_sequence", _sequence_with_pattern(fitted, pattern, fail=range(5))
    )
    mc, err, info = errors.monte_carlo(fitted, nmc=30, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["n_success"] == 25 and ha["cluster_test_run"]
    assert "mc_too_few" not in ha["flags"] and "mc_multimodal" in ha["flags"]


def test_flags_and_summary_are_persisted_in_fit_result(fitted, monkeypatch):
    """fit_spectrum(nmc=...) carries the per-line summary in res['mc_info']."""
    res = copy.deepcopy(fitted)
    pattern = [(0.0, 0.0)] * 4 + [(ALIAS_KMS, -ALIAS_KMS)]
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(res, pattern))
    mc, err, info = errors.monte_carlo(res, nmc=30, return_diagnostics=True)
    res["mc"], res["err"], res["mc_info"] = mc, err, info
    md = res["mc_info"]["lines"]["Halpha"]
    for key in (
        "alias_fraction",
        "mc_sigma",
        "mc_gap_kms",
        "clusters",
        "basin_switch",
        "cluster_test_run",
        "n_contributing_required",
        "bimodal",
        "flags",
    ):
        assert key in md
    row = fit_module.summary_row(res)
    assert "mc_multimodal" in row["HA_mc_flags"].split(",")
    assert row["HA_mc_n_success"] == 30


def test_nonfinite_successful_draws_do_not_enable_cluster_test(fitted, monkeypatch):
    # Solver status alone does not establish 25 finite values for either key.
    pattern = [(0.0, np.nan)] * 6 + [(0.0, 0.0)] * 18 + [(0.0, 900.0)] * 6
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(fitted, pattern))
    mc, err, info = errors.monte_carlo(fitted, nmc=30, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["n_success"] == 30 and ha["n_finite"]["c50_sys"] == 24
    assert ha["alias_n_assessed"] == 24 and ha["alias_n_unassessed"] == 6
    assert ha["alias_fraction"] == pytest.approx(6 / 24)
    assert not ha["cluster_test_run"] and "mc_too_few" in ha["flags"]
    assert "mc_multimodal" not in ha["flags"]
    res = copy.deepcopy(fitted)
    res.update(mc=mc, err=err, mc_info=info)
    row = blrfit.summary_row(res)
    assert row["HA_mc_n_success"] == row["HA_mc_n_converged_lines"] == 30
    assert row["HA_mc_n_finite_c50_sys"] == 24
    for key in ("alias_fraction", "mc_sigma", "mc_gap_kms"):
        col = key if key.startswith("mc_") else "mc_" + key
        assert row["HA_" + col] == ha[key]
