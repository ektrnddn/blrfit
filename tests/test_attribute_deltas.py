"""
The attribution script (tools/attribute_deltas.py): the search over toggle
sets on a stub fitter, where every outcome can be constructed, and the end-to-
end run on three pinned spectra.

The stub fitter returns prepared line records per configuration, so the six
outcomes (unchanged, attributed-single, attributed-joint, numerical by the
edge, numerical by a start perturbation, unresolved) are each checked with
the toggle set, the effect sizes and the complement path they must produce.

The three pinned spectra are the worked example spec-0651 (classes C/C, no
change between the versions above 0.1 km/s), spec-1237 (class A, the
continuous Fe II operator moves Hbeta by -23 km/s) and spec-1704 (classes
F/F, host applied; both lines move by -2.5 km/s). On every platform the run
must produce a row per fitted line with an outcome of the documented set and
the current fit of the 0.2.0 pin within the end-point tolerance of
tests/test_pins.py; on the reference stack (BLRFIT_STRICT_PINS=1) the
baseline must reproduce the 0.1.0 pins and the outcomes are those of
docs/DELTAS.md: spec-0651 unchanged, the other two attributed to the operator
alone, with the complement path agreeing.
"""

import csv
import json
import os
import sys

import numpy as np
import pytest

from conftest import DATA, ROOT

sys.path.insert(0, os.path.join(ROOT, "tools"))
import attribute_deltas as ad  # noqa: E402

STRICT = os.environ.get("BLRFIT_STRICT_PINS", "") not in ("", "0")
END_POINT_KMS = 100.0  # the fresh-fit tolerance of tests/test_pins.py


def rec(cls="A", c50=0.0, n=1, flags=(), margin=20.0):
    return dict(cls=cls, flags=tuple(flags), n_broad=n, c50=float(c50), bic_margin=float(margin))


class Stub:
    """A fitter that answers from a table keyed by (sorted on-toggles, perturbation)."""

    def __init__(self, table, default):
        self.table, self.default, self.calls = table, default, []

    def __call__(self, on, perturb):
        self.calls.append((frozenset(on), perturb))
        key = ("+".join(sorted(on)), perturb)
        return dict(lines={"Hbeta": self.table.get(key, self.default)})


def run(table, default, names=("a", "b", "c"), **kw):
    stub = Stub(table, default)
    search = ad.Search(stub, names)
    return ad.attribute_line(search, "Hbeta", **kw), search, stub


def test_same_and_change_string():
    a = rec("A", 100.0, 1, ("low_snr",))
    assert ad.same(a, rec("A", 101.5, 1, ("low_snr",)))
    assert not ad.same(a, rec("A", 101.6, 1, ("low_snr",)))
    assert not ad.same(a, rec("C", 100.0, 1, ("low_snr",)))
    assert not ad.same(a, rec("A", 100.0, 2, ("low_snr",)))
    assert not ad.same(a, rec("A", 100.0, 1, ()))
    assert not ad.same(a, None) and ad.same(None, None)
    assert ad.same(rec(c50=np.nan), rec(c50=np.nan))
    assert ad.change_string(a, rec("C", 0.0, 2, ("poor_fit",))) == "A>C;n1>2;+poor_fit;-low_snr"
    assert ad.change_string(a, a) == ""
    assert ad.change_string(None, a) == "fitted" and ad.change_string(a, None) == "unfitted"


def test_unchanged():
    base = rec("A", 0.0)
    out, search, stub = run({}, base)
    assert out["outcome"] == "unchanged" and out["toggles"] == []
    assert out["effect"] == {"a": 0.0, "b": 0.0, "c": 0.0}
    # the current fit, the baseline, three singles; the complements are the pairs
    assert search.n_fits == 8 and not any(p for _, p in stub.calls)


def test_attributed_single():
    base, cur = rec("A", 0.0), rec("C", 30.0, 2)
    table = {
        ("a", None): cur,
        ("a+b", None): cur,
        ("a+c", None): cur,
        ("a+b+c", None): cur,
        ("b", None): rec("A", 0.5),
        ("c", None): rec("A", 2.0),
    }
    out, search, _ = run(table, base)
    assert out["outcome"] == "attributed-single" and out["toggles"] == ["a"]
    assert out["effect"] == {"a": 30.0, "b": 0.5, "c": 2.0}
    assert out["change"] == {"a": "A>C;n1>2", "b": "", "c": ""}
    assert out["complement_necessary"] == ["a"] and out["order_independent"] is True
    assert out["edge"] is False and out["perturbation"] == ""


def test_attributed_joint():
    base, cur = rec("A", 0.0), rec("A", 40.0)
    table = {
        ("a", None): rec("A", 10.0),
        ("b", None): rec("A", 12.0),
        ("c", None): base,
        ("a+b", None): cur,
        ("a+c", None): rec("A", 10.0),
        ("b+c", None): rec("A", 12.0),
        ("a+b+c", None): cur,
    }
    out, search, _ = run(table, base)
    assert out["outcome"] == "attributed-joint" and out["toggles"] == ["a", "b"]
    assert out["reproducing_pairs"] == [("a", "b")] and out["reproducing_singles"] == []
    assert out["complement_necessary"] == ["a", "b"] and out["order_independent"] is True
    assert out["effect"] == {"a": 10.0, "b": 12.0, "c": 0.0}


def test_numerical_by_the_edge():
    base, cur = rec("A", 0.0, margin=3.0), rec("C", 40.0, margin=-2.0)
    table = {("a+b+c", None): cur}
    out, search, stub = run(table, base)
    assert out["outcome"] == "numerical" and out["perturbation"] == "edge" and out["edge"] is True
    assert out["toggles"] == [] and out["complement_necessary"] == ["a", "b", "c"]
    assert not any(p for _, p in stub.calls)  # no perturbation needed


def test_numerical_by_a_start_perturbation():
    base, cur = rec("A", 0.0), rec("C", 40.0)
    table = {("a+b+c", None): cur, ("a+b+c", "start"): cur, ("", "start"): cur}
    out, search, stub = run(table, base)
    assert out["outcome"] == "numerical" and out["perturbation"] == "baseline+start"
    assert (frozenset(), "start") in stub.calls
    # the current configuration reaching the baseline under the last-bit perturbation counts as well
    table = {("a+b+c", None): cur, ("a+b+c", "start"): cur, ("a+b+c", "lastbit"): base}
    out, _, _ = run(table, base)
    assert out["outcome"] == "numerical" and out["perturbation"] == "current+lastbit"


def test_unresolved():
    base, cur = rec("A", 0.0), rec("C", 40.0)
    table = {("a+b+c", None): cur, ("a+b+c", "start"): cur, ("a+b+c", "lastbit"): cur}
    out, search, stub = run(table, base)
    assert out["outcome"] == "unresolved" and out["edge"] is False and out["perturbation"] == ""
    assert {p for _, p in stub.calls} == {None, "start", "lastbit"}
    out, _, stub = run(table, base, perturb=False)
    assert out["outcome"] == "unresolved" and not any(p for _, p in stub.calls)


def test_line_not_fitted_in_the_current_configuration():
    stub = Stub({("a+b+c", None): rec()}, rec())
    search = ad.Search(lambda on, p: dict(lines={}), ["a"])
    assert ad.attribute_line(search, "Hbeta") is None


def test_toggles_are_discovered():
    toggles, unavailable = ad.available_toggles()
    names = [t.name for t in toggles]
    assert "fe_operator" in names and "fe_uv_width_policy" in names
    assert set(names) | set(unavailable) == {t.name for t in ad.TOGGLES}
    with ad.configured(toggles, frozenset(), perturb="start") as kw:
        from blrfit.model import broad, continuum

        assert broad.BROAD_STARTS_KMS[0] == ad.PERTURB_KMS
        for t in toggles:
            if t.keyword():
                assert kw[t.name] == t.off
        if not toggles[0].keyword():
            assert continuum.FeTemplate.broadened is ad._historical_broadening
    assert broad.BROAD_STARTS_KMS[0] == 0.0
    assert continuum.FeTemplate.broadened is not ad._historical_broadening


# ----------------------------------------------------------------------------
# three pinned spectra
# ----------------------------------------------------------------------------
PINNED = ("spec-0651-52141-0072.fits", "spec-1237-52762-0298.fits", "spec-1704-53178-0562.fits")


def _pins(name):
    with open(os.path.join(DATA, name)) as fh:
        return {p["file"]: p for p in json.load(fh)["pins"]}


def test_three_pinned_spectra(tmp_path):
    current, legacy = _pins("pins_0.2.0.json"), _pins("pins.json")
    out_csv, out_json = tmp_path / "attribution.csv", tmp_path / "attribution.json"
    ad.main(
        [
            "--pins",
            os.path.join(DATA, "pins_0.2.0.json"),
            "--legacy",
            os.path.join(DATA, "pins.json"),
            "--only",
            *PINNED,
            "--out",
            str(out_csv),
            "--json",
            str(out_json),
        ]
    )
    with open(out_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    names = [t.name for t in ad.available_toggles()[0]]
    assert list(rows[0]) == ad.columns(names)
    with open(out_json) as fh:
        dump = json.load(fh)
    assert [s["file"] for s in dump["spectra"]] == list(PINNED)
    by_file = {}
    for r in rows:
        by_file.setdefault(r["spectrum"], {})[r["line"]] = r
    assert set(by_file) == set(PINNED)
    assert set(by_file[PINNED[0]]) == {"Halpha", "Hbeta"} and set(by_file[PINNED[1]]) == {"Hbeta"}
    assert set(by_file[PINNED[2]]) == {"Halpha", "Hbeta"}
    for fn, lines in by_file.items():
        for line, r in lines.items():
            p = ad.PREFIX[line]
            ref = current[fn]["summary"]
            assert r["outcome"] in ad.OUTCOMES, (fn, line, r["outcome"])
            assert r["class_cur"] == ref[f"{p}_class"] and int(r["n_broad_cur"]) == ref[f"{p}_n_broad"]
            assert abs(float(r["c50_cur"]) - ref[f"{p}_c50_sys"]) < END_POINT_KMS
            assert r["toggles_available"] == "+".join(names) and int(r["n_fits"]) >= 2
            assert r["baseline_vs_legacy"] != ""
            assert set(dump["spectra"][PINNED.index(fn)]["configurations"]) >= {
                "baseline",
                "+".join(sorted(names)),
            }
    if not STRICT:
        return
    # the reference stack: the baseline is the 0.1.0 fit and the deltas are the operator's (docs/DELTAS.md);
    # the CSV rounds to 1e-3 km/s, the JSON dump holds the full values
    exact = {(s["file"], r["line"]): r for s in dump["spectra"] for r in s["rows"]}
    for fn, lines in by_file.items():
        for line, r in lines.items():
            p = ad.PREFIX[line]
            assert r["baseline_vs_legacy"] == "reproduced" and float(r["legacy_dc50"]) == 0.0, (fn, line)
            assert exact[fn, line]["c50_base"] == legacy[fn]["summary"][f"{p}_c50_sys"]
            assert exact[fn, line]["c50_cur"] == current[fn]["summary"][f"{p}_c50_sys"]
    assert {r["outcome"] for r in by_file[PINNED[0]].values()} == {"unchanged"}
    for fn in PINNED[1:]:
        for line, r in by_file[fn].items():
            assert r["outcome"] == "attributed-single" and r["toggles"] == "fe_operator", (fn, line)
            assert r["complement_necessary"] == "fe_operator" and r["order_independent"] == "True"
            assert abs(float(r["dc50_fe_operator"]) - float(r["delta_c50"])) < ad.TOL_KMS
    assert float(by_file[PINNED[1]]["Hbeta"]["delta_c50"]) == pytest.approx(-22.75, abs=0.05)
