"""Regression tests for conditional MC/systemic estimator consistency."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import blrfit
from blrfit import errors
from blrfit.constants import MC_ALIAS_KMS, MC_ALIAS_MAX_FRACTION
from blrfit.model import fit as fit_module
from blrfit.model import lines
from blrfit.model import continuum
from synth import make_spectrum

BROAD = [dict(line="Halpha", v=2000., fwhm=5000., ew=150.),
         dict(line="Hbeta", v=2000., fwhm=5000., ew=40.)]
ALIAS_KMS = 943.5   # narrow Halpha taken for [N II] 6584


@pytest.fixture(scope="module")
def weak_narrow_spectrum():
    # OIII anchors a weak, displaced Halpha narrow group under a broad line.
    return make_spectrum(snr=18., v_sys=700., narrow=dict(ew_ha=1.2, o3=100.),
        seed=93, broad=BROAD)


@pytest.fixture(scope="module")
def displaced_weak_narrow_spectrum():
    # The alias regime: narrow group 700 km/s from the input redshift, Halpha
    # narrow peak S/N about 7, broad Halpha at +2000 km/s, continuum S/N 10.
    return make_spectrum(snr=10., v_sys=700., narrow=dict(ew_ha=5.), seed=7, broad=BROAD)


def fit_weak(sp, tied=True):
    return blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], .25,
        host=False, fe=False, max_broad=1, err_floor=0., use_ha_systemic=tied,
        complexes=("Halpha", "Hbeta"))


@pytest.mark.parametrize("tied", [False, True])
def test_unperturbed_mc_matches_production_estimator(weak_narrow_spectrum, monkeypatch, tied):
    """The zero-noise limit must use precisely the original systemic estimator.

    The synthetic truth is weak in Halpha but strong in OIII, so a lost prior
    is consequential. Non-default flags are recovered from saved settings.
    """
    res = fit_weak(weak_narrow_spectrum, tied=tied)
    calls = []
    original_fit = fit_module.fit_complex

    def trace(name, *args, **kw):
        calls.append((name, copy.deepcopy({k: v for k, v in kw.items() if k != "diagnostics"})))
        return original_fit(name, *args, **kw)

    monkeypatch.setattr(fit_module, "fit_complex", trace)
    monkeypatch.setattr(errors.np.random, "default_rng",
                        lambda seed: SimpleNamespace(standard_normal=lambda n: np.zeros(n)))
    mc, err, info = errors.monte_carlo(res, nmc=5, return_diagnostics=True)
    assert info["fe"] is False
    assert info["use_ha_systemic"] is tied
    assert info["fixed"] == ["host_model", "broad_component_counts", "redshift", "extinction",
                              "pixel_mask", "input_error_array"]
    assert info["multimodality_assessed"] is True
    for name in res["fits"]:
        for key in ("c50_sys", "v_sys", "fwhm", "v_peak_sys"):
            np.testing.assert_allclose(mc[name][key], res["meas"][name][key], atol=1e-8, rtol=0)
            assert err[name][key] < 1e-8
        assert info["lines"][name]["n_success"] == 5
        assert info["lines"][name]["n_failed"] == 0
        assert info["lines"][name]["alias_count"] == 0
        assert info["lines"][name]["alias_fraction"] == 0.
        assert info["lines"][name]["n_unconverged_continuum"] == 0
        assert info["lines"][name]["flags"] == []
    if tied:
        assert res["o3_prefit"]["snr"] > 10
        assert abs(res["meas"]["Halpha"]["v_sys"] - 700.) < 150.
        assert [n for n, _ in calls] == ["Hbeta", "Halpha", "Hbeta"] * 5
        for name, kw in calls:
            if name == "Halpha":
                assert kw["n_v_prior"] == pytest.approx((res["o3_prefit"]["v_o3"], 150.))
                assert kw["n_v_starts"][-1] == res["o3_prefit"]["v_o3"]
    else:
        assert [n for n, _ in calls] == ["Halpha", "Hbeta"] * 5
        assert all("v_sys_prior" not in kw and "n_v_prior" not in kw for _, kw in calls)


def test_perturbed_draws_refit_prefit_and_record_failures(weak_narrow_spectrum, monkeypatch):
    res = fit_weak(weak_narrow_spectrum)
    original = errors.fit_continuum
    attempts = []

    def intermittent_failure(*args, **kw):
        attempts.append(1)
        if len(attempts) <= 2:
            raise RuntimeError("injected continuum failure")
        return original(*args, **kw)

    monkeypatch.setattr(errors, "fit_continuum", intermittent_failure)
    mc, err, info = errors.monte_carlo(res, nmc=7, seed=21, return_diagnostics=True)
    assert info["status"] == "partial_failure"
    assert "injected continuum failure" in info["draws"][0]["exception"]
    assert "continuum_converged" not in info["draws"][0]
    assert info["n_unconverged_continuum"] == 0
    prefit_v = [d["o3_prefit"]["v_o3"] for d in info["draws"][2:]]
    assert np.std(prefit_v) > .1
    for name in res["fits"]:
        d = info["lines"][name]
        assert d["n_success"] == 5 and d["n_failed"] == 2
        assert d["n_finite"]["c50_sys"] == 5
        assert "failed_draws" in d["flags"]
        assert d["n_unconverged_continuum"] == 0
        assert np.isnan(d["samples"]["c50_sys"][:2]).all()
        assert err[name]["c50_sys"] > 0.
        np.testing.assert_allclose(mc[name]["c50_sys"],
            np.percentile(d["samples"]["c50_sys"][2:], [16, 50, 84]))


def _failed_solver(fun, x0, **kwargs):
    return SimpleNamespace(success=False, status=0, message="maximum evaluations exceeded",
        nfev=1, optimality=1., x=np.asarray(x0), fun=fun(x0), active_mask=np.zeros(len(x0)))


def test_unconverged_line_solver_is_distinct_from_missing_coverage(weak_narrow_spectrum, monkeypatch):
    """An unconverged line solver keeps its finite end point, flagged; an
    unusable window still yields nothing. In the MC such draws contribute."""
    res = fit_weak(weak_narrow_spectrum)
    wr, fsub, ir = res["wave_rest"], res["flux_sub"], res["ivar_rest"]
    monkeypatch.setattr(lines, "least_squares", _failed_solver)
    diag = {}
    r, all_fits = lines.fit_complex_select("Halpha", wr, fsub, ir, max_broad=1, diagnostics=diag)
    assert r is not None and len(all_fits) == 1
    assert r["converged"] is False
    assert diag["status"] == "success_unconverged" and diag["converged"] is False
    assert diag["components"][1]["n_converged"] == 0
    assert diag["components"][1]["attempts"]
    assert all(a["status"] == 0 and not a["success"] for a in diag["components"][1]["attempts"])
    coverage = {}
    assert lines.fit_complex("Halpha", wr, fsub, np.zeros_like(ir), 1, diagnostics=coverage) is None
    assert coverage["status"] == "unusable_window"
    mc, err, info = errors.monte_carlo(res, nmc=5, return_diagnostics=True)
    assert info["status"] == "complete"
    for name in res["fits"]:
        d = info["lines"][name]
        assert d["n_success"] == 5 and d["n_failed"] == 0
        assert d["n_unconverged_lines"] == 5
        assert "unconverged_line_draws" in d["flags"]
        assert "failed_draws" not in d["flags"]
        assert d["n_finite"]["c50_sys"] == 5
        assert np.isfinite(err[name]["c50_sys"])
        assert info["draws"][0]["lines"][name]["status"] == "success_unconverged"
        assert info["draws"][0]["lines"][name]["diagnostics"]["attempts"]
        assert np.isfinite(d["alias_fraction"])


def test_lowest_chi_square_wins_and_unconverged_selection_is_flagged(weak_narrow_spectrum, monkeypatch):
    """The lowest finite chi-square is kept whether or not its solver converged;
    an unconverged selection is flagged, not passed over."""
    res = fit_weak(weak_narrow_spectrum)
    attempts = []

    def solver(fun, x0, **kw):
        attempts.append(1)
        sol = _failed_solver(fun, x0, **kw)
        if len(attempts) == 1:
            sol.fun = np.zeros_like(sol.fun)  # apparently better but unconverged
        else:
            sol.success, sol.status = True, 1
        return sol

    monkeypatch.setattr(lines, "least_squares", solver)
    r = lines.fit_complex("Halpha", res["wave_rest"], res["flux_sub"], res["ivar_rest"], 1)
    assert r is not None
    assert r["solver"]["selected_attempt"] == 0
    assert r["solver"]["attempts"][0]["objective"] == 0
    assert r["solver"]["attempts"][0]["success"] is False
    assert r["converged"] is False
    assert r["solver"]["status"] == "success_unconverged"
    assert r["solver"]["n_converged"] == len(attempts) - 1
    assert all(a["success"] for a in r["solver"]["attempts"][1:])


def test_native_pixel_mask_retained(weak_narrow_spectrum):
    sp = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in weak_narrow_spectrum.items()}
    bad = np.abs(sp["wave"] / 1.25 - 6600.) < 2.
    sp["ivar"][bad] = 0
    r = fit_weak(sp)["fits"]["Halpha"]
    assert np.count_nonzero(~r["native_mask"]) == np.count_nonzero(bad)
    assert (r["native_w"][~r["native_mask"]] == 0).all()
    np.testing.assert_array_equal(r["native_x"][r["native_mask"]], r["x"])
    assert r['data_chi2'] + sum(r['penalty_chi2'].values()) == pytest.approx(r['chi2'])
    assert r['selection_score'] == r['bic']
    assert r['selection_score_kind'] == 'penalized_chi2_plus_k_log_n'


def test_unconverged_continuum_draws_are_kept_and_flagged(weak_narrow_spectrum, monkeypatch):
    """A draw whose continuum solver stops short is kept and counted, as in 0.1.0.

    The converged model is returned unchanged with the solver record marked
    unsuccessful, so the lines of that draw fit exactly as they would have.
    """
    res = fit_weak(weak_narrow_spectrum)
    original = errors.fit_continuum
    calls = []

    def unconverged_on_some_draws(*args, **kw):
        calls.append(1)
        d, model, cinfo = original(*args, **kw)
        if len(calls) in (1, 3):
            cinfo = dict(cinfo, solver=dict(cinfo["solver"], success=False, status=0))
        return d, model, cinfo

    monkeypatch.setattr(errors, "fit_continuum", unconverged_on_some_draws)
    mc, err, info = errors.monte_carlo(res, nmc=5, seed=3, return_diagnostics=True)
    assert info["status"] == "complete"
    assert info["continuum_convergence_checked"] is True
    assert info["n_unconverged_continuum"] == 2
    assert [d["continuum_converged"] for d in info["draws"]] == [False, True, False, True, True]
    assert all("exception" not in d for d in info["draws"])
    for name in res["fits"]:
        d = info["lines"][name]
        assert d["n_success"] == 5 and d["n_failed"] == 0
        assert d["n_unconverged_continuum"] == 2
        assert "unconverged_continuum_draws" in d["flags"]
        assert "failed_draws" not in d["flags"]
        assert d["n_finite"]["c50_sys"] == 5
        assert np.isfinite(d["samples"]["c50_sys"]).all()
        assert np.isfinite(err[name]["c50_sys"])


def test_failed_continuum_solver_keeps_draws_in_mc(weak_narrow_spectrum, monkeypatch):
    """Every draw unconverged: nothing is discarded, everything is flagged."""
    res = fit_weak(weak_narrow_spectrum)
    monkeypatch.setattr(continuum, 'least_squares', _failed_solver)
    _, errors_out, info = errors.monte_carlo(res, nmc=5, return_diagnostics=True)
    assert info['continuum_convergence_checked']
    assert info['n_unconverged_continuum'] == 5
    assert all(d['continuum_converged'] is False for d in info['draws'])
    assert all('exception' not in d for d in info['draws'])
    for name in res['fits']:
        d = info['lines'][name]
        assert d['n_success'] == 5 and d['n_failed'] == 0
        assert d['n_unconverged_continuum'] == 5
        assert 'unconverged_continuum_draws' in d['flags']
        assert np.isfinite(errors_out[name]['c50_sys'])


def test_displaced_weak_narrow_group_stays_in_its_basin(displaced_weak_narrow_spectrum):
    """The shared sequence keeps the draws of a displaced weak narrow group in one basin.

    Without the [O III] start and prior a perturbed draw of this spectrum can
    settle its narrow group on [N II], 943 km/s away.
    """
    res = fit_weak(displaced_weak_narrow_spectrum)
    m = res["meas"]["Halpha"]
    assert 4. < m["narrow_peak_snr"] < 8.
    assert abs(m["v_sys"] - 700.) < 150.
    assert res["o3_prefit"]["snr"] >= 10.
    mc, err, info = errors.monte_carlo(res, nmc=12, seed=5, return_diagnostics=True)
    assert info["multimodality_assessed"] is True
    for name in res["fits"]:
        d = info["lines"][name]
        assert d["n_success"] == 12
        assert d["alias_reference"] == dict(v_sys=res["meas"][name]["v_sys"],
                                            c50_sys=res["meas"][name]["c50_sys"])
        assert d["alias_count"] == 0
        assert d["alias_fraction"] == 0.
        assert "mc_multimodal" not in d["flags"]
        assert np.abs(np.asarray(d["samples"]["v_sys"]) - res["meas"][name]["v_sys"]).max() < MC_ALIAS_KMS
        assert err[name]["c50_sys"] < MC_ALIAS_KMS


def _sequence_with_pattern(res, shifts):
    """A stand-in for the line sequence returning the unperturbed measures,
    with the Halpha narrow group displaced by shifts[i] on draw i."""
    calls = []

    def sequence(wr, fsub, ir, cmodel, host_model, z, complexes, **kw):
        i = len(calls)
        calls.append(1)
        meas = copy.deepcopy(res["meas"])
        dv, dc = shifts[i % len(shifts)]
        meas["Halpha"]["v_sys"] += dv
        meas["Halpha"]["c50_sys"] += dc
        prefit = dict(res["o3_prefit"], status="success")
        return {}, meas, prefit, {name: dict(status="success") for name in meas}

    return sequence


def test_alias_fraction_undefined_without_contributing_draws(weak_narrow_spectrum, monkeypatch):
    res = fit_weak(weak_narrow_spectrum)

    def nothing_fits(wr, fsub, ir, cmodel, host_model, z, complexes, **kw):
        prefit = dict(res["o3_prefit"], status="success")
        return {}, {}, prefit, {name: dict(status="solver_failed", attempts=[]) for name in res["fits"]}

    monkeypatch.setattr(fit_module, "_fit_line_sequence", nothing_fits)
    _, err, info = errors.monte_carlo(res, nmc=4, return_diagnostics=True)
    assert info["status"] == "partial_failure"
    for name in res["fits"]:
        d = info["lines"][name]
        assert d["n_success"] == 0 and d["n_failed"] == 4
        assert d["n_unconverged_lines"] == 0
        assert d["alias_count"] == 0
        assert np.isnan(d["alias_fraction"])
        assert "mc_multimodal" not in d["flags"] and "failed_draws" in d["flags"]
        assert np.isnan(err[name]["c50_sys"])


def test_alias_bookkeeping_flags_a_bimodal_sample(weak_narrow_spectrum, monkeypatch):
    """Draws alternating between the true basin and the [N II] alias."""
    res = fit_weak(weak_narrow_spectrum)
    monkeypatch.setattr(fit_module, "_fit_line_sequence",
                        _sequence_with_pattern(res, [(0., 0.), (ALIAS_KMS, -ALIAS_KMS)]))
    mc, err, info = errors.monte_carlo(res, nmc=8, return_diagnostics=True)
    assert info["multimodality_assessed"] is True
    ha = info["lines"]["Halpha"]
    assert ha["n_success"] == 8
    assert ha["alias_count"] == 4
    assert ha["alias_fraction"] == 0.5
    assert "mc_multimodal" in ha["flags"]
    # The percentile error of the flagged line spans both basins.
    assert err["Halpha"]["v_sys"] > MC_ALIAS_KMS
    assert err["Halpha"]["c50_sys"] > MC_ALIAS_KMS
    hb = info["lines"]["Hbeta"]
    assert hb["alias_count"] == 0 and hb["alias_fraction"] == 0.
    assert "mc_multimodal" not in hb["flags"]


def test_alias_threshold_applies_to_either_offset_and_fraction_is_strict(weak_narrow_spectrum, monkeypatch):
    res = fit_weak(weak_narrow_spectrum)
    # One draw in ten displaced in c50_sys only, within the threshold in v_sys.
    below = 0.5 * MC_ALIAS_KMS
    above = 1.25 * MC_ALIAS_KMS
    pattern = [(below, 0.)] * 9 + [(0., above)]
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(res, pattern))
    _, _, info = errors.monte_carlo(res, nmc=10, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["alias_count"] == 1
    assert ha["alias_fraction"] == pytest.approx(MC_ALIAS_MAX_FRACTION)
    assert "mc_multimodal" not in ha["flags"]
    # Two in ten, displaced together in v_sys, form a second group and cross the fraction.
    pattern = [(0., 0.)] * 8 + [(above, 0.), (above, 0.)]
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(res, pattern))
    _, _, info = errors.monte_carlo(res, nmc=10, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["alias_count"] == 2
    assert ha["alias_fraction"] == pytest.approx(0.2)
    assert "mc_multimodal" in ha["flags"]
    # Two single outliers on different axes are counted but do not form a second group.
    pattern = [(0., 0.)] * 8 + [(-above, 0.), (0., above)]
    monkeypatch.setattr(fit_module, "_fit_line_sequence", _sequence_with_pattern(res, pattern))
    _, _, info = errors.monte_carlo(res, nmc=10, return_diagnostics=True)
    ha = info["lines"]["Halpha"]
    assert ha["alias_count"] == 2 and not ha["bimodal"]
    assert "mc_multimodal" not in ha["flags"]
