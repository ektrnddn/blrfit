"""The pair and tiers commands on the two bundled SDSS epochs of J001224."""

import numpy as np
import pytest

from blrfit.batch import read_rows
from blrfit.cli import main
from blrfit.pairs import PAIR_COLUMNS
from blrfit.tiers import TIERS, TIER_COLUMNS
from conftest import SDSS_EXAMPLE, SDSS_EXAMPLE_2, Z_J001224


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    out = tmp_path_factory.mktemp("pair")
    rc = main(
        [
            "pair",
            SDSS_EXAMPLE,
            SDSS_EXAMPLE_2,
            "--survey",
            "sdss",
            "--z",
            str(Z_J001224),
            "--out",
            str(out),
            "--pickle",
            "--quiet",
        ]
    )
    assert rc == 0
    return out


def test_pair_tables(paired):
    out = paired
    rows = read_rows(out / "blrfit_pairs.ecsv")
    assert [r["line"] for r in rows] == ["Halpha", "Hbeta"]
    assert list(rows[0]) == list(PAIR_COLUMNS)
    r = rows[1]
    assert r["pair"] == "spec-0651-52141-0072__spec-7169-56628-0344" and r["kind"] == "sdss-sdss"
    assert r["role"] == "reference+consecutive" and r["targetid"] == "object"
    assert (
        abs(r["dt_days"] - (56628 - 52141)) < 1e-6
        and abs(r["dt_rest_yr"] - 4487 / 365.25 / (1 + Z_J001224)) < 1e-6
    )
    assert r["sigma_sys"] == 36.0 and r["errors_a"] == "fit result"
    assert isinstance(r["retained"], bool) and isinstance(r["stable_shape"], bool)
    for row in rows:
        if row["retained"]:
            assert np.isfinite(row["s_common"]) and row["err_total"] > row["err"] > 0
        else:
            assert "not_retained" in row["flags"]
    targets = read_rows(out / "blrfit_targets.ecsv")
    assert len(targets) == 1 and targets[0]["reference"] in ("spec-0651-52141-0072", "spec-7169-56628-0344")
    assert targets[0]["n_epochs"] == 2 and targets[0]["class_halpha"] in "ABCFEXW"
    assert np.isfinite(targets[0]["logmbh"]) and np.isfinite(targets[0]["vmax_q01"])
    tiers = read_rows(out / "blrfit_tiers.ecsv")
    assert len(tiers) == 1 and list(tiers[0]) == list(TIER_COLUMNS) and tiers[0]["tier"] in TIERS
    assert (out / "spec-0651-52141-0072__spec-7169-56628-0344_Hbeta_pair.png").exists()
    assert (out / "spec-0651-52141-0072_fit.json").exists() and (
        out / "spec-7169-56628-0344_fit.pkl"
    ).exists()


def test_saved_fits_reproduce_the_pairs(paired, tmp_path):
    out = paired
    rc = main(
        [
            "pair",
            "--fits",
            str(out / "spec-0651-52141-0072_fit.pkl"),
            str(out / "spec-7169-56628-0344_fit.pkl"),
            "--out",
            str(tmp_path),
            "--table",
            str(tmp_path / "pairs.csv"),
            "--name",
            "J001224",
            "--no-figure",
            "--quiet",
        ]
    )
    assert rc == 0
    first = read_rows(out / "blrfit_pairs.ecsv")
    again = read_rows(tmp_path / "pairs.csv")
    assert [r["targetid"] for r in again] == ["J001224", "J001224"]
    for a, b in zip(first, again):
        assert (
            a["line"] == b["line"] and a["mjd_a"] == b["mjd_a"] and str(a["retained"]) == str(b["retained"])
        )
        for k in ("s_fwd", "s_rev", "err_fwd", "scale_fwd", "dchi2_shape_fwd"):
            assert np.isclose(a[k], b[k], rtol=1e-9, equal_nan=True), k
    assert (tmp_path / "pairs_targets.csv").exists() and (tmp_path / "pairs_tiers.csv").exists()
    assert not list(tmp_path.glob("*_pair.png"))


def test_tiers_command(paired, tmp_path, capsys):
    out = paired
    rc = main(["tiers", str(out / "blrfit_pairs.ecsv"), "--out", str(tmp_path / "tiers.ecsv")])
    assert rc == 0
    printed = capsys.readouterr().out
    tiers = read_rows(tmp_path / "tiers.ecsv")
    before = read_rows(out / "blrfit_tiers.ecsv")
    assert tiers[0]["tier"] == before[0]["tier"] and tiers[0]["reason"] == before[0]["reason"]
    assert f"object: {tiers[0]['tier']}" in printed
    assert np.isfinite(tiers[0]["logmbh"]) == np.isfinite(before[0]["logmbh"])


def test_pair_needs_two_epochs(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "pair",
                SDSS_EXAMPLE,
                "--survey",
                "sdss",
                "--z",
                str(Z_J001224),
                "--out",
                str(tmp_path),
                "--quiet",
            ]
        )
