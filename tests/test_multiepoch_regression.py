"""The chain on a frozen extract of the catalog of the paper (38 objects, 50 identifiers, 6,092 pair rows of the
all-pairs table of 2026-10-16, with the noise model calibrated on the whole catalog): the verdicts, tags, curves and
object verdicts must reproduce the run of record to the last printed digit, and the orbit summaries of a few lines must
reproduce its orbit records. The extract holds the nineteen robust movers and the seven candidate lines of the paper,
ordinary constant objects, profile-variable, double-peaked and not-measurable ones, and objects of two identifiers.

Column and tag names that 0.5.0 renamed are mapped before the comparison (``n_clean_pairs`` -> ``n_pairs_4sig``,
``more than one clean pair`` -> ``two pairs above 4 sigma``)."""

import csv
import json
import os

import numpy as np
import pytest

from blrfit.multiepoch import Inputs, run_orbits, run_verdicts

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "multiepoch")
RENAMED_COLUMNS = {"n_clean_pairs": "n_pairs_4sig"}
RENAMED_TAGS = {"more than one clean pair": "two pairs above 4 sigma"}


def inputs():
    return Inputs(
        pairs=os.path.join(DATA, "pairs.csv.gz"),
        catalog=os.path.join(DATA, "catalog.fits"),
        masses=os.path.join(DATA, "masses.fits"),
        facts=os.path.join(DATA, "facts.csv"),
        all_fits=os.path.join(DATA, "all_fits.fits"),
    ).check()


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def same(a, b, rtol=1e-9):
    """Equal cells; a number written by both sides is compared to a relative 1e-9 (the last printed digits of a
    chi-square probability of 1e-15 differ between BLAS libraries)."""
    a, b = (
        ("" if str(a) == "nan" else a),
        ("" if str(b) == "nan" else b),
    )  # an empty cell is a NaN written by pandas
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if np.isnan(x) and np.isnan(y):
        return True
    return abs(x - y) <= rtol * max(abs(x), abs(y), 1e-300)


def expected_rows(name):
    rows = read(os.path.join(DATA, name))
    out = []
    for r in rows:
        r = {RENAMED_COLUMNS.get(k, k): v for k, v in r.items()}
        for col in ("tags", "halpha_tags", "hbeta_tags"):
            if col in r:
                r[col] = ";".join(RENAMED_TAGS.get(t, t) for t in r[col].split(";"))
        out.append(r)
    return out


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("multiepoch")
    run_verdicts(inputs(), os.path.join(DATA, "CALIBRATION.json"), out, verbose=False)
    return out


def test_verdicts_reproduce_the_run_of_record(run_dir):
    got = {(r["object"], r["line"]): r for r in read(os.path.join(run_dir, "verdicts.csv"))}
    exp = {(r["object"], r["line"]): r for r in expected_rows("expected_verdicts.csv")}
    assert set(got) == set(exp)
    for key, e in exp.items():
        g = got[key]
        for col, val in e.items():
            assert same(g.get(col, ""), val), (key, col, g.get(col), val)
    verdicts = {r["verdict"] for r in got.values()}
    assert {"moving", "constant", "profile variable", "not measurable"} <= verdicts


def test_objects_reproduce_the_run_of_record(run_dir):
    got = {r["object"]: r for r in read(os.path.join(run_dir, "objects.csv"))}
    exp = {r["object"]: r for r in expected_rows("expected_objects.csv")}
    assert set(got) == set(exp)
    for o, e in exp.items():
        for col, val in e.items():
            assert got[o][col] == val, (o, col, got[o][col], val)
    assert "double-peaked" in {r["combined"] for r in got.values()}


def test_the_time_level_leave_one_out_is_recorded(run_dir):
    rows = read(os.path.join(run_dir, "verdicts.csv"))
    robust = [r for r in rows if r["verdict"] == "moving" and int(float(r["n_times"])) >= 3
              and "disappears when one spectrum is excluded" not in r["tags"]]  # fmt: skip
    assert robust
    tagged = [r for r in robust if "disappears when one time is excluded" in r["tags"]]
    assert all(r["rests_on_time"] for r in tagged)
    assert all(not r["rests_on_time"] for r in robust if r not in tagged)


def test_outputs_are_written(run_dir):
    for name in (
        "spectra.csv",
        "verdicts.csv",
        "objects.csv",
        "curves.json",
        "pair_table.csv",
        "members.json",
        "VERDICTS.md",
    ):
        assert os.path.exists(os.path.join(run_dir, name)), name
    curves = json.load(open(os.path.join(run_dir, "curves.json")))
    assert "39633286363874179|Halpha" in curves
    assert len(curves["39633286363874179|Halpha"]["clusters"]) == 10


@pytest.mark.slow
def test_orbits_reproduce_the_run_of_record(run_dir):
    """The orbit summaries (the allowed and the free orbit, the labels) of the candidate and of a steady drifter, to
    the printed digits; the steady test is the one of the verdicts (pair space), so only its slope is compared."""
    objects = ["39633286363874179", "39627769298029988", "2305843047037929685"]
    got = run_orbits(run_dir, inputs(), objects=objects, nproc=1, verbose=False)
    exp = json.load(open(os.path.join(DATA, "expected_orbits.json")))
    checked = 0
    for key, e in exp.items():
        if key.split("|")[0] not in objects:
            continue
        g = got[key]
        for f in (
            "verdict",
            "n_times",
            "mass_label",
            "orbit_fits",
            "orbit_fits_generous",
            "orbit_label",
            "orbit_tested",
            "v_ref",
            "ref_year",
        ):
            assert g.get(f) == e.get(f), (key, f, g.get(f), e.get(f))
        for part in ("orbit_mass", "orbit_free", "orbit_mass_generous", "orbit_free_rel", "orbit_mass_rel"):
            ge, ee = g.get(part), e.get(part)
            if not isinstance(ee, dict):
                assert ge == ee, (key, part)
                continue
            # a fit through few times passes exactly at many periods (chi2 ~ 0): which of them is the minimum is
            # numerical noise, so only the chi2 and the degrees of freedom are compared there
            exact = ee.get("chi2") is not None and ee["chi2"] < 0.5
            for k, v in ee.items():
                if exact and k not in ("chi2", "dof"):
                    continue
                gv = ge.get(k)
                if isinstance(v, float) and v is not None and gv is not None:
                    assert abs(gv - v) <= 1e-9 * max(1.0, abs(v)), (key, part, k, gv, v)
                else:
                    assert gv == v, (key, part, k, gv, v)
        # the steady slope: 0.5.0 takes it from the verdicts' fit (per-spectrum times, pair space), the run of record
        # from the cluster means; they agree to a few per cent
        assert abs(g["steady"]["accel"] - e["steady"]["accel"]) < 0.05 * abs(e["steady"]["accel"]) + 1.0, key
        for gt, et in zip(g["times"], e["times"]):
            assert {k: v for k, v in gt.items() if k != "spectra"} == et, (key, gt, et)
        checked += 1
    assert checked >= 4


def test_the_data_set_holds_the_paper_objects():
    picked = json.load(open(os.path.join(DATA, "objects.json")))
    assert "39633286363874179" in picked and "39627769298029988" in picked
    assert len(picked) == 38
