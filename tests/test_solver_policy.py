"""
The solver policy of 0.2.0 and the input-scale guard.

Unconverged line solutions are flagged and kept: every finite end point of the
trust-region solver competes for the lowest chi-square, and the selected one
is marked 'success_unconverged' when the solver did not report convergence
(the production fitter kept such end points without a flag). A continuum
solver that stopped short of convergence does not stop the line fits; its
state is recorded as ``continuum_status`` and ``continuum_converged``. The
chosen component count carries ``bic_margin``, its BIC distance from the next
count, because the A-versus-C split can hinge on the one-versus-two component
choice within a few units of DBIC. The input flux must be of the order of
SDSS and DESI fluxes (1e-17 erg/s/cm^2/A) or declared through ``flux_scale``.

The four pinned spectra of ``tests/data/pins.json`` check that the policy
changes nothing there: every fitted line selects a converged attempt and, for
every component count, the lowest finite objective belongs to a converged
attempt, so the rule that discarded unconverged attempts would have selected
the same solutions; the classes are those of the pins.
"""
import json
import os

import numpy as np
import pytest

import blrfit
from blrfit.constants import FLUX_SCALE_MAX, FLUX_SCALE_MIN
from blrfit.io import read_sdss
from blrfit.model import continuum, lines
from blrfit.model.fit import PREFIX
from conftest import DATA, EXAMPLES, SDSS_EXAMPLE, Z_J001224
from synth import make_spectrum

BROAD = [dict(line="Halpha", v=1200., fwhm=4000., ew=150.),
         dict(line="Hbeta", v=1200., fwhm=4000., ew=50.)]


@pytest.fixture(scope="module")
def synthetic():
    return make_spectrum(snr=30., seed=7, broad=BROAD)


def fit_synthetic(sp, **kw):
    settings = dict(host=False, fe=False, max_broad=2, complexes=("Halpha", "Hbeta"))
    settings.update(kw)
    return blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], .25, **settings)


@pytest.fixture(scope="module")
def example():
    sp = read_sdss(SDSS_EXAMPLE)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Halpha", "Hbeta"))
    return sp, res


def test_converged_fit_is_recorded_as_such(synthetic):
    res = fit_synthetic(synthetic, complexes=("Halpha", "Hbeta", "MgII"))
    row = blrfit.summary_row(res)
    assert res["continuum_status"] == "success"
    assert row["continuum_status"] == "success"
    for name in ("Halpha", "Hbeta"):
        p = PREFIX[name]
        assert res["fit_status"][name]["status"] == "success"
        assert res["fit_status"][name]["converged"] is True
        assert res["fit_status"][name]["continuum_converged"] is True
        assert res["fits"][name]["converged"] is True
        assert row[f"{p}_fit_status"] == "success" and row[f"{p}_converged"] is True
        assert res["cls"][name]["label"] == "A"
    # a complex outside the data has no solution and is not converged
    assert row["MG_fit_status"] == "unusable_window" and row["MG_converged"] is False


def test_unconverged_line_solution_is_kept_and_flagged(synthetic, monkeypatch):
    monkeypatch.setattr(lines, "MAX_NFEV_COMPLEX", 3)
    res = fit_synthetic(synthetic)
    row = blrfit.summary_row(res)
    assert res["continuum_status"] == "success"
    for name in ("Halpha", "Hbeta"):
        p = PREFIX[name]
        r = res["fits"][name]
        assert res["fit_status"][name]["status"] == "success_unconverged"
        assert res["fit_status"][name]["converged"] is False
        assert r["converged"] is False
        assert row[f"{p}_fit_status"] == "success_unconverged"
        assert row[f"{p}_converged"] is False
        solver = r["solver"]
        assert solver["status"] == "success_unconverged" and solver["n_converged"] == 0
        assert solver["attempts"] and all(a["status"] == 0 for a in solver["attempts"])
        assert not any(a["success"] for a in solver["attempts"])
        # the lowest finite objective was kept
        objectives = [a["objective"] for a in solver["attempts"] if a["finite"]]
        assert solver["attempts"][solver["selected_attempt"]]["objective"] == min(objectives)
        assert r["chi2"] == min(objectives)
        # the solution is measured and classified like any other
        assert np.isfinite(res["meas"][name]["c50_sys"])
        assert res["cls"][name]["label"] in ("A", "B", "C", "F", "E", "X", "W")
    # every component count was flagged, and the selection carried the flag on
    for nb, d in res["fit_status"]["Halpha"]["components"].items():
        assert d["status"] == "success_unconverged" and d["converged"] is False


def test_unconverged_continuum_does_not_stop_the_lines(synthetic, monkeypatch):
    monkeypatch.setattr(continuum, "MAX_NFEV_CONTI", 1)
    res = fit_synthetic(synthetic)
    row = blrfit.summary_row(res)
    assert res["continuum_info"]["solver"]["success"] is False
    assert res["continuum_status"] == "unconverged"
    assert row["continuum_status"] == "unconverged"
    assert set(res["fits"]) == {"Halpha", "Hbeta"} == set(res["meas"]) == set(res["cls"])
    for name in ("Halpha", "Hbeta"):
        assert res["fit_status"][name]["continuum_converged"] is False
        assert res["fit_status"][name]["status"] == "success"
        assert row[f"{PREFIX[name]}_converged"] is True
    assert res["mc_info"]["status"] == "not_requested"


def test_bic_margin_is_the_distance_to_the_next_component_count(example, synthetic):
    _, res = example
    row = blrfit.summary_row(res)
    for name in ("Halpha", "Hbeta"):
        r = res["fits"][name]
        b = r["all_bic"]; i = r["n_broad"] - 1
        assert len(b) == 3
        expected = min(abs(b[i] - b[j]) for j in range(len(b)) if j != i)
        assert np.isfinite(r["bic_margin"]) and r["bic_margin"] > 0
        assert r["bic_margin"] == expected
        assert row[f"{PREFIX[name]}_bic_margin"] == expected
    # one component count only: no margin
    single = fit_synthetic(synthetic, max_broad=1)
    assert np.isnan(single["fits"]["Halpha"]["bic_margin"])
    assert np.isnan(blrfit.summary_row(single)["HA_bic_margin"])


def test_input_scale_is_guarded(example):
    sp, _ = example
    with pytest.raises(ValueError, match="flux_scale"):
        blrfit.fit_spectrum(sp["wave"], sp["flux"] * 1e-17, sp["ivar"] * 1e34, Z_J001224,
                            complexes=("Halpha", "Hbeta"))
    with pytest.raises(ValueError, match="flux_scale"):
        blrfit.fit_spectrum(sp["wave"], sp["flux"] * 1e6, sp["ivar"] * 1e-12, Z_J001224,
                            complexes=("Halpha", "Hbeta"))
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="flux_scale"):
            blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, flux_scale=bad)
    median = np.median(sp["flux"][np.isfinite(sp["flux"]) & (sp["flux"] > 0)])
    assert FLUX_SCALE_MIN < median < FLUX_SCALE_MAX


def test_flux_scale_reproduces_the_unscaled_fit(example):
    """A power of two is exact in floating point, so the scaled spectrum is the
    unscaled one bit for bit after ``flux_scale`` and the fit is identical. The
    decimal factor 1e-17 is not exactly invertible (a relative 2e-16 on the
    flux), which moves the solver's end point by 0.08 km/s on this spectrum:
    the classes and the offsets agree to the tolerance of that sensitivity."""
    sp, res = example
    scale = 2.0 ** -56                                     # 1.4e-17: the order of cgs flux
    exact = blrfit.fit_spectrum(sp["wave"], sp["flux"] * scale, sp["ivar"] / scale**2, Z_J001224,
                                complexes=("Halpha", "Hbeta"), flux_scale=1.0 / scale)
    assert exact["settings"]["flux_scale"] == 1.0 / scale and res["settings"]["flux_scale"] == 1.0
    for k in ("wave_rest", "flux_rest", "ivar_rest"):
        assert np.array_equal(exact[k], res[k])
    for name in ("Halpha", "Hbeta"):
        assert exact["cls"][name]["label"] == res["cls"][name]["label"]
        assert exact["fits"][name]["n_broad"] == res["fits"][name]["n_broad"]
        for k in ("c50_sys", "v_sys", "fwhm"):
            assert abs(exact["meas"][name][k] - res["meas"][name][k]) < 1e-6
        # flux-bearing outputs are in the scaled unit, that is, the unscaled one
        for k in ("broad_flux", "broad_lum", "broad_ew"):
            assert exact["meas"][name][k] == pytest.approx(res["meas"][name][k], rel=1e-9)
    decimal = blrfit.fit_spectrum(sp["wave"], sp["flux"] * 1e-17, sp["ivar"] * 1e34, Z_J001224,
                                  complexes=("Halpha", "Hbeta"), flux_scale=1e17)
    for name in ("Halpha", "Hbeta"):
        assert decimal["cls"][name]["label"] == res["cls"][name]["label"]
        assert decimal["fits"][name]["n_broad"] == res["fits"][name]["n_broad"]
        assert abs(decimal["meas"][name]["c50_sys"] - res["meas"][name]["c50_sys"]) < 1.0


def _pins():
    with open(os.path.join(DATA, "pins.json")) as fh:
        return json.load(fh)["pins"]


def _spectrum_path(fn):
    for d in (EXAMPLES, DATA):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


@pytest.mark.parametrize("pin", _pins(), ids=lambda p: p["file"])
def test_pins_select_converged_attempts(pin):
    """On the four pins the flag-and-keep rule selects what the discard rule
    would have: the lowest finite objective of every component count belongs
    to a converged attempt, the continuum converged, and the classes are the
    pinned ones."""
    sp = read_sdss(_spectrum_path(pin["file"]))
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], pin["z"], ebv=pin["ebv"],
                              complexes=tuple(pin["complexes"]))
    row = blrfit.summary_row(res)
    assert res["continuum_status"] == "success"
    assert set(res["fits"]) == set(pin["params"])
    for name in pin["params"]:
        p = PREFIX[name]
        assert row[f"{p}_class"] == pin["summary"][f"{p}_class"]
        assert row[f"{p}_fit_status"] == "success" and row[f"{p}_converged"] is True
        assert res["fits"][name]["converged"] is True
        assert np.isfinite(row[f"{p}_bic_margin"])
        for nb, d in res["fit_status"][name]["components"].items():
            assert d["status"] == "success", (name, nb, d["status"])
            finite = [a for a in d["attempts"] if a.get("finite", False)]
            best = min(finite, key=lambda a: a["objective"])
            assert best["success"], (name, nb)
            assert d["attempts"][d["selected_attempt"]] is best
