"""The multiepoch command: its stages run from the command line on the frozen extract, and the deprecated tiers
command says so."""

import os
import warnings

import pytest

from blrfit import cli

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "multiepoch")


def tables():
    return [
        "--pairs", os.path.join(DATA, "pairs.csv.gz"),
        "--catalog", os.path.join(DATA, "catalog.fits"),
        "--masses", os.path.join(DATA, "masses.fits"),
        "--facts", os.path.join(DATA, "facts.csv"),
        "--all-fits", os.path.join(DATA, "all_fits.fits"),
    ]  # fmt: skip


def test_verdicts_and_population_from_the_command_line(tmp_path):
    out = str(tmp_path / "run")
    rc = cli.main(
        [
            "multiepoch",
            "verdicts",
            *tables(),
            "--calibration",
            os.path.join(DATA, "CALIBRATION.json"),
            "--out",
            out,
            "--quiet",
        ]
    )
    assert rc == 0
    for name in ("verdicts.csv", "objects.csv", "VERDICTS.md"):
        assert os.path.exists(os.path.join(out, name))
    rc = cli.main(["multiepoch", "population", *tables(), "--run", out, "--seeds", "3", "--quiet"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "POPULATION.md"))


def test_calibrate_from_the_command_line(tmp_path):
    """The calibration on the extract runs end to end (its values are those of 38 objects, not the paper's)."""
    out = str(tmp_path / "cal")
    rc = cli.main(["multiepoch", "calibrate", *tables(), "--out", out, "--nboot", "2", "--quiet"])
    assert rc == 0
    assert os.path.exists(os.path.join(out, "CALIBRATION.json"))


def test_help_lists_the_stages(capsys):
    with pytest.raises(SystemExit):
        cli.main(["multiepoch", "--help"])
    text = capsys.readouterr().out
    for stage in ("calibrate", "verdicts", "orbits", "population", "fraction"):
        assert stage in text


def test_tiers_is_deprecated(capsys):
    import importlib

    import blrfit

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.reload(
            blrfit.tiers
        )  # another test may have imported it already: the warning fires on import
    assert any("deprecated" in str(x.message) for x in w)
    with pytest.raises(SystemExit):
        cli.main(["tiers", "--help"])
    assert "deprecated" in capsys.readouterr().out
