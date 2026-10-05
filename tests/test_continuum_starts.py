"""
The continuum fit started from several points.

In host-rich spectra the power-law slope and the host amplitude are degenerate,
and a single start can stop in a local minimum. The documented case is the SDSS
spectrum of J001224 (spec-0651-52141-0072) at E(B-V) = 0: its first start stops
without a host, at a chi-square more than 2,000 above the host solution that the
other starts reach on the same pixels. These tests hold the selection rule, the
records of the starts and that case.
"""

import numpy as np
import pytest

import blrfit
import blrfit.model.continuum as continuum
from blrfit.constants import CONTI_START_ALPHAS, CONTI_START_DCHI2, CONTI_START_HOST_SCALES
from blrfit.io import read_sdss
from blrfit.model.continuum import _select_start, fit_continuum
from conftest import SDSS_EXAMPLE, Z_J001224


def _fits(*chi2):
    return [(c, k, f"ps{k}", f"sol{k}") for k, c in enumerate(chi2)]


def test_select_start_keeps_the_earliest_within_tolerance():
    # starts that reach one minimum differ in the last digits: the first is kept
    assert _select_start(_fits(100.0, 100.0 - 0.5 * CONTI_START_DCHI2))[1] == 0
    # a later start lower by more than the tolerance wins
    assert _select_start(_fits(100.0, 98.9, 98.5))[1] == 1
    # the earliest start within the tolerance of the lowest is kept
    assert _select_start(_fits(5000.0, 4000.0, 4000.5, 3999.6))[1] == 1
    # a non-finite chi-square never wins; with none finite the first start stands
    assert _select_start(_fits(np.nan, 50.0))[1] == 1
    assert _select_start(_fits(np.nan, np.nan))[1] == 0


@pytest.fixture(scope="module")
def j001224():
    sp = read_sdss(SDSS_EXAMPLE)
    return blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Halpha", "Hbeta"))


def test_j001224_reaches_the_host_solution(j001224):
    ci, hi = j001224["continuum_info"], j001224["host_info"]
    starts = ci["starts"]
    assert len(starts) == len(CONTI_START_ALPHAS) * len(CONTI_START_HOST_SCALES)
    assert [s["index"] for s in starts] == list(range(len(starts)))
    assert sum(s["selected"] for s in starts) == 1
    kept = starts[ci["start_selected"]]
    assert ci["start_selected"] != 0 and kept["selected"]
    # the first start (the single start up to 0.2) stops without a host, far above the kept one
    assert starts[0]["host_frac"] < 0.05
    assert starts[0]["chi2"] > kept["chi2"] + 2000.0
    assert hi["applied"] and 0.3 < hi["host_frac_4200_5000"] < 0.5
    # the kept start is the earliest within the tolerance of the lowest chi-square
    lowest = min(s["chi2"] for s in starts if np.isfinite(s["chi2"]))
    within = [s["index"] for s in starts if s["chi2"] <= lowest + CONTI_START_DCHI2]
    assert ci["start_selected"] == min(within)
    assert blrfit.summary_row(j001224)["conti_start"] == ci["start_selected"]


def _rest_frame_sdss():
    sp = read_sdss(SDSS_EXAMPLE)
    good = sp["ivar"] > 0
    z = Z_J001224
    return sp["wave"][good] / (1 + z), sp["flux"][good] * (1 + z), sp["ivar"][good] / (1 + z) ** 2


def test_power_law_fit_records_every_slope_start():
    wr, fr, ir = _rest_frame_sdss()
    d, model, info = fit_continuum(wr, fr, ir)
    assert [s["alpha_start"] for s in info["starts"]] == list(CONTI_START_ALPHAS)
    assert sum(s["selected"] for s in info["starts"]) == 1
    assert info["starts"][info["start_selected"]]["selected"]
    assert np.isfinite(info["chi2"]) and np.all(np.isfinite(model))


def test_a_failing_extra_start_is_recorded_not_fatal(monkeypatch):
    real = continuum.least_squares
    failing = CONTI_START_ALPHAS[1]

    def flaky(fun, x0, **kw):
        if np.any(x0 == failing):
            raise RuntimeError("synthetic failure")
        return real(fun, x0, **kw)

    monkeypatch.setattr(continuum, "least_squares", flaky)
    wr, fr, ir = _rest_frame_sdss()
    d, model, info = fit_continuum(wr, fr, ir)
    failed = info["starts"][1]
    assert failed["success"] is False and "synthetic failure" in failed["error"] and not failed["selected"]
    assert info["start_selected"] == 0 and np.all(np.isfinite(model))


def test_monte_carlo_follows_the_recorded_start_policy(j001224):
    """Draws refit the continuum as the result was fitted: several starts for a
    0.3 result, the single start for a result saved before the setting existed."""
    from blrfit.errors import monte_carlo

    assert j001224["settings"]["conti_multistart"] is True
    _, _, info = monte_carlo(j001224, nmc=2, seed=1, return_diagnostics=True)
    assert info["continuum_settings"]["multistart"] is True
    legacy = dict(j001224, settings={k: v for k, v in j001224["settings"].items() if k != "conti_multistart"})
    _, _, info = monte_carlo(legacy, nmc=2, seed=1, return_diagnostics=True)
    assert info["continuum_settings"]["multistart"] is False
