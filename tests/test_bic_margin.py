"""
The selection margin of the component count.

``select_by_bic`` keeps the simplest component count whose penalised selection
score (the chi-square of the penalised fit plus k ln N, not a textbook BIC)
lies within DBIC of the best. ``bic_margin`` is the smallest change of any
single score of the fitted vector that changes that choice: a simpler count j
is chosen once its score falls by more than s_j - s_best - DBIC, the chosen
count c loses once its score rises by DBIC - (s_c - min_{i != c} s_i). The
margin is a function of the persisted score list alone; ``bic_gap`` keeps the
raw distance min |s_c - s_other| of the earlier releases.

Checked on constructed score vectors, on the pinned score lists of
``tests/data/pins_0.2.0.json`` and on synthetic Halpha windows: the critical
score moved by margin - eps leaves the choice and moved by margin + eps
changes it, every other score moved by margin - eps in either direction
leaves it, and a bisection per score and direction finds no smaller flip; a
second broad component tuned to the edge gives a margin below 1 and the choice
flips between noise realisations of the same sigma; a clean single Gaussian
and a clearly double profile give margins above 20; along a monotone
amplitude sweep of the second component the signed distance to the edge
crosses zero where the chosen count changes and the margin is its modulus.
"""
import json
import os

import numpy as np
import pytest

import blrfit
from blrfit.constants import DBIC, LAM
from blrfit.model import lines
from blrfit.model.broad import select_by_bic, selection_margin, score_gap
from blrfit.model.lines import fit_complex_select
from conftest import DATA
from synth import gauss_v, make_spectrum

EPS = 1e-7


def _flip_checks(s, dbic=DBIC):
    """The margin's own definition: the critical score moved by margin - EPS in
    its flip direction leaves the choice, by margin + EPS changes it; every
    other single-score move of margin - EPS leaves it. Returns the margin."""
    s = np.asarray(s, float)
    c = select_by_bic(s, dbic)
    margin, at, direction = selection_margin(s, dbic)
    assert margin >= 0 and 0 <= at < s.size and direction in (-1, 1)
    moved = s.copy(); moved[at] += direction * (margin + EPS)
    assert select_by_bic(moved, dbic) != c
    if margin > EPS:      # a zero margin leaves no move short of the edge
        moved = s.copy(); moved[at] += direction * (margin - EPS)
        assert select_by_bic(moved, dbic) == c
        for j in range(s.size):
            for sign in (-1, 1):
                if j == at and sign == direction:
                    continue
                moved = s.copy(); moved[j] += sign * (margin - EPS)
                assert select_by_bic(moved, dbic) == c, (j, sign)
    return margin


def _smallest_flip(s, dbic=DBIC, tol=1e-9):
    """Bisection per score and direction for the smallest single-score move that
    changes the choice (the choice is monotone along any such move)."""
    s = np.asarray(s, float)
    c = select_by_bic(s, dbic)
    reach = float(np.ptp(s) + 2 * dbic + 1.0)
    best = np.inf
    for j in range(s.size):
        for sign in (-1, 1):
            moved = s.copy(); moved[j] += sign * reach
            if select_by_bic(moved, dbic) == c:
                continue
            lo, hi = 0.0, reach
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                moved = s.copy(); moved[j] += sign * mid
                if select_by_bic(moved, dbic) == c:
                    lo = mid
                else:
                    hi = mid
            best = min(best, hi)
    return best


VECTORS = {
    "simpler within dbic": ([100.0, 95.0, 96.0], 0, 5.0, 0, 1),
    "best chosen, simpler far": ([120.0, 95.0, 96.0], 1, 11.0, 1, 1),
    "best chosen, simpler nearest": ([114.0, 95.0, 110.0], 1, 9.0, 0, -1),
    "four counts": ([120.0, 95.0, 96.5, 140.0], 1, 11.5, 1, 1),
    "two counts, simpler within": ([100.0, 99.0], 0, 9.0, 0, 1),
    "two counts, simpler best": ([90.0, 100.0], 0, 20.0, 0, 1),
    "third within of the best second": ([130.0, 100.0, 105.0], 1, 15.0, 1, 1),
    "tie on the edge": ([100.0, 90.0, 90.0], 1, 0.0, 0, -1),
}


@pytest.mark.parametrize("case", list(VECTORS), ids=list(VECTORS))
def test_constructed_vectors(case):
    s, chosen, margin, at, direction = VECTORS[case]
    assert select_by_bic(s, DBIC) == chosen
    got = selection_margin(s, DBIC)
    assert got == (margin, at, direction)
    assert _flip_checks(s) == margin
    assert abs(_smallest_flip(s) - margin) < 1e-6


def test_tie_follows_the_strict_inequality():
    # a simpler model exactly dbic above the best is not chosen; the margin is zero
    s = [110.0, 100.0]
    assert select_by_bic(s, DBIC) == 1
    assert selection_margin(s, DBIC) == (0.0, 0, -1)
    assert select_by_bic([110.0 - EPS, 100.0], DBIC) == 0
    # the chosen model exactly dbic above the best would not have been chosen
    s = [100.0, 90.0 + 1e-9]
    assert select_by_bic(s, DBIC) == 0
    margin, at, direction = selection_margin(s, DBIC)
    assert 0 < margin < 1e-8 and (at, direction) == (0, 1)


def test_single_score_has_no_margin():
    assert np.isnan(selection_margin([50.0], DBIC)[0])
    assert selection_margin([50.0], DBIC)[1:] == (-1, 0)
    assert np.isnan(score_gap([50.0], 0))


def test_gap_is_the_raw_distance():
    s = np.array([120.0, 95.0, 96.5, 140.0])
    for i in range(s.size):
        assert score_gap(s, i) == min(abs(s[i] - s[j]) for j in range(s.size) if j != i)


def _pinned_score_lists():
    with open(os.path.join(DATA, "pins_0.2.0.json")) as fh:
        pins = json.load(fh)["pins"]
    return [(pin["file"], name, s) for pin in pins for name, s in pin["all_bic"].items()]


@pytest.mark.parametrize("pinned", _pinned_score_lists(), ids=lambda t: f"{t[0]}:{t[1]}")
def test_pinned_score_lists(pinned):
    """The margin recomputed from a persisted score list satisfies its
    definition and equals the smallest flip found by bisection."""
    fn, name, s = pinned
    margin = _flip_checks(s)
    assert abs(_smallest_flip(s) - margin) < 1e-6
    # the chosen model's own move is bounded by its raw gap plus DBIC
    assert margin <= score_gap(s, select_by_bic(s, DBIC)) + DBIC


# ----------------------------------------------------------------------------
# synthetic Halpha windows: continuum-free rest-frame flux, a narrow group and
# one broad Gaussian, plus an optional offset second broad component
# ----------------------------------------------------------------------------
WR = np.arange(6300.0, 6800.0, 0.8)
CONT = 10.0
SNR = 20.0
NARROW = ((LAM["Halpha"], 40.0), (LAM["NII6584"], 40.0), (LAM["NII6548"], 40.0 / 2.96),
          (LAM["SII6716"], 16.0), (LAM["SII6731"], 12.0))


def window(ew2, seed=0, v2=3000.0, fwhm2=1500.0):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(WR)
    for lam, ew in NARROW:
        y += gauss_v(WR, lam, 0.0, 150.0, ew * CONT)
    y += gauss_v(WR, LAM["Halpha"], 0.0, 4000.0 / 2.3548, 150.0 * CONT)
    if ew2 > 0:
        y += gauss_v(WR, LAM["Halpha"], v2, fwhm2 / 2.3548, ew2 * CONT)
    sig = CONT / SNR
    return y + rng.normal(0.0, sig, WR.size), np.full(WR.size, 1.0 / sig ** 2)


def fit(ew2, seed=0, max_broad=2):
    y, iv = window(ew2, seed)
    r, _ = fit_complex_select("Halpha", WR, y, iv, max_broad=max_broad)
    assert r is not None
    return r


def _check_record(r):
    """The persisted keys of the chosen fit, and the recomputation of the
    margin and the gap from the persisted score list."""
    i = r["all_n_broad"].index(r["n_broad"])
    assert i == select_by_bic(r["all_bic"], DBIC)
    margin, at, direction = selection_margin(r["all_bic"], DBIC)
    assert r["bic_margin"] == margin
    assert r["bic_margin_flip"] == dict(n_broad=r["all_n_broad"][at], direction=direction)
    assert r["bic_gap"] == score_gap(r["all_bic"], i)
    assert r["bic_gap"] == min(abs(r["all_bic"][i] - b) for j, b in enumerate(r["all_bic"]) if j != i)


def test_well_separated_cases():
    # one clean Gaussian: the one-component model is best and the next is far
    r = fit(0.0, max_broad=3)
    assert r["n_broad"] == 1 and r["all_n_broad"] == [1, 2, 3]
    assert r["bic_margin"] > 20
    assert r["bic_margin"] == r["bic_gap"] + DBIC
    _check_record(r)
    # a clearly double profile: the two-component model is far below the single
    r = fit(60.0, max_broad=2)
    assert r["n_broad"] == 2
    assert r["bic_margin"] > 20 and r["bic_margin"] == r["bic_gap"] - DBIC
    _check_record(r)


def test_one_component_count_has_no_margin():
    r = fit(0.0, max_broad=1)
    assert r["n_broad"] == 1 and r["all_n_broad"] == [1]
    assert np.isnan(r["bic_margin"]) and np.isnan(r["bic_gap"]) and r["bic_margin_flip"] is None


def test_failed_count_is_absent_from_the_vector(monkeypatch):
    original = lines.fit_complex

    def without_two(name, wave, fsub, ivar, n_broad, **kw):
        if n_broad == 2:
            diag = kw.get("diagnostics")
            if diag is not None:
                diag.update(status="solver_failed", attempts=[], n_converged=0)
            return None
        return original(name, wave, fsub, ivar, n_broad, **kw)

    monkeypatch.setattr(lines, "fit_complex", without_two)
    r = fit(0.0, max_broad=3)
    assert r["all_n_broad"] == [1, 3] and len(r["all_bic"]) == 2
    assert r["n_broad"] == 1
    _check_record(r)


def _signed_edge(r):
    """Distance of the one-versus-two choice from its edge, positive when two
    components are chosen: s_1 - s_2 - DBIC on a two-count vector."""
    assert r["all_n_broad"] == [1, 2]
    return r["all_bic"][0] - r["all_bic"][1] - DBIC


def test_amplitude_sweep_and_the_edge():
    """A monotone sweep of the second component's equivalent width on one noise
    realisation: the chosen count changes once, the signed distance to the
    edge rises monotonically through zero there, the margin is its modulus;
    bisection to the edge gives a margin below 1, where other noise
    realisations of the same sigma choose either count."""
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    fits = [fit(ew2) for ew2 in grid]
    counts = [r["n_broad"] for r in fits]
    signed = [_signed_edge(r) for r in fits]
    assert counts[0] == 1 and counts[-1] == 2
    assert all(b >= a for a, b in zip(counts, counts[1:]))
    assert all(b > a for a, b in zip(signed, signed[1:]))
    for r, d, n in zip(fits, signed, counts):
        assert n == (2 if d >= 0 else 1)
        assert r["bic_margin"] == pytest.approx(abs(d), abs=1e-9)
        _check_record(r)
    k = counts.index(2)
    lo, hi = grid[k - 1], grid[k]
    r_edge = None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        r = fit(mid)
        if r["bic_margin"] < 1.0:
            r_edge = (mid, r)
            break
        if r["n_broad"] == 1:
            lo = mid
        else:
            hi = mid
    assert r_edge is not None, "no point of the sweep within 1 of the edge"
    ew_edge, r = r_edge
    assert abs(_signed_edge(r)) < 1.0 and r["bic_margin"] == pytest.approx(abs(_signed_edge(r)), abs=1e-9)
    _check_record(r)
    # at the edge, a different noise realisation of the same sigma flips the choice
    seen = {r["n_broad"]}
    for seed in range(1, 9):
        seen.add(fit(ew_edge, seed)["n_broad"])
        if seen == {1, 2}:
            break
    assert seen == {1, 2}


def test_fit_spectrum_carries_the_margin():
    """The full fit persists the score list per line (``bic_all`` of the
    measures), the summary row exports the margin, and both the margin and the
    gap are recomputable from the persisted list with the recorded ``dbic``."""
    sp = make_spectrum(snr=30.0, seed=7, broad=[dict(line="Halpha", v=1200.0, fwhm=4000.0, ew=150.0),
                                                 dict(line="Hbeta", v=1200.0, fwhm=4000.0, ew=50.0)])
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], 0.25, host=False, fe=False,
                              max_broad=3, complexes=("Halpha", "Hbeta"))
    row = blrfit.summary_row(res)
    dbic = res["settings"]["dbic"]
    for name, p in (("Halpha", "HA"), ("Hbeta", "HB")):
        r = res["fits"][name]
        _check_record(r)
        assert res["meas"][name]["bic_all"] == r["all_bic"]
        assert row[f"{p}_bic_margin"] == r["bic_margin"]
        assert r["bic_margin"] == selection_margin(res["meas"][name]["bic_all"], dbic)[0]
        assert r["bic_gap"] == score_gap(res["meas"][name]["bic_all"], r["all_n_broad"].index(r["n_broad"]))
        assert np.isfinite(r["bic_margin"]) and r["bic_margin"] >= 0
