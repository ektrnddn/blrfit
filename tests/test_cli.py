"""The command-line tool on the bundled examples."""

import json

import pytest
from astropy.table import Table

from blrfit.classify import FLAG_TEXT, velocities_at_bound
from blrfit.cli import build_parser, main
from conftest import SDSS_EXAMPLE, SDSS_EXAMPLE_2, DESI_EXAMPLE, DESI_TARGETID, CSV_EXAMPLE, Z_J001224
from test_pins import _current_pins

# the end-point tolerance of tests/test_pins.py: the least-squares solver ends at a platform-dependent
# point of a degenerate decomposition, which moves the primary offset by tens of km/s but not the class
END_POINT_KMS = 100.0
# the pinned summary rows of the SDSS example (read with the IVAR-only mask, as the pins are) and of the DESI example
_PINS = {p["file"]: p for p in _current_pins()["pins"]}
J001224_PIN = _PINS["spec-0651-52141-0072.fits"]["summary"]
DESI_PIN = _PINS["coadd-main-dark-17260-39627574082538900.fits"]["summary"]


def test_fit_sdss_example(tmp_path):
    rc = main(
        [
            "fit",
            SDSS_EXAMPLE,
            "--survey",
            "sdss",
            "--sdss-mask-policy",
            "ivar",
            "--z",
            str(Z_J001224),
            "--lines",
            "Halpha,Hbeta,MgII",
            "--out",
            str(tmp_path),
            "--no-figure",
            "--quiet",
        ]
    )
    assert rc == 0
    doc = json.load(open(tmp_path / "spec-0651-52141-0072_fit.json"))
    assert (
        doc["input"]["kind"] == "sdss"
        and doc["input"]["z"] == Z_J001224
        and doc["input"]["z_source"] == "argument"
    )
    assert doc["input"]["ebv"] == 0.0
    ha, hb, mg = doc["lines"]["Halpha"], doc["lines"]["Hbeta"], doc["lines"]["MgII"]
    # the pinned values, to the end-point tolerance
    assert (
        ha["fitted"]
        and ha["label"] == "C"
        and abs(ha["dv"] - J001224_PIN["HA_c50_sys"]) < END_POINT_KMS
        and ha["strong_offset"]
    )
    assert hb["fitted"] and hb["label"] == "C" and abs(hb["dv"] - J001224_PIN["HB_c50_sys"]) < END_POINT_KMS
    assert hb["systemic_source"] == "Halpha prior"
    assert (
        not mg["fitted"]
        and mg["label"] == ""
        and mg["class_text"] == "not fitted"
        and "not fitted" in mg["reasons"][0]
    )
    assert mg["dv"] is None
    assert ha["dv_err_mc"] is None and ha["dv_err_model"] is None
    assert doc["uncertainty"]["status"] == "not_computed" and not doc["uncertainty"]["calibrated"]
    assert doc["summary_row"]["HA_class"] == "C"
    assert doc["settings"]["complexes"] == ["Halpha", "Hbeta", "MgII"]
    assert doc["blrfit_version"]
    assert not (tmp_path / "spec-0651-52141-0072_fit.png").exists()


def test_fit_writes_figure_and_pickle(tmp_path):
    rc = main(
        [
            "fit",
            SDSS_EXAMPLE,
            "--survey",
            "sdss",
            "--sdss-mask-policy",
            "ivar",
            "--z",
            str(Z_J001224),
            "--lines",
            "Hbeta",
            "--out",
            str(tmp_path),
            "--pickle",
            "--table",
            str(tmp_path / "tables" / "one.ecsv"),
            "--quiet",
        ]
    )
    assert rc == 0
    assert (tmp_path / "spec-0651-52141-0072_fit.png").stat().st_size > 10000
    assert (tmp_path / "spec-0651-52141-0072_fit.pkl").exists()
    doc = json.load(open(tmp_path / "spec-0651-52141-0072_fit.json"))
    assert list(doc["lines"]) == ["Hbeta"]
    row = Table.read(tmp_path / "tables" / "one.ecsv")
    assert len(row) == 1 and row["stem"][0] == "spec-0651-52141-0072" and row["status"][0] == "fitted"
    assert row["HB_c50_sys"][0] == pytest.approx(doc["lines"]["Hbeta"]["dv"], abs=1e-9)
    assert row["ebv_assumed_zero"][0] and str(row["HB_fwhm"].unit) == "km / s"


def test_fit_desi_example_uses_redrock_and_fibermap(tmp_path, capsys):
    rc = main(
        [
            "fit",
            DESI_EXAMPLE,
            "--targetid",
            str(DESI_TARGETID),
            "--lines",
            "Halpha,Hbeta",
            "--out",
            str(tmp_path),
            "--no-figure",
        ]
    )
    assert rc == 0
    doc = json.load(open(tmp_path / "coadd-main-dark-17260-39627574082538900_fit.json"))
    assert doc["input"]["kind"] == "desi" and abs(doc["input"]["z"] - 0.22032) < 1e-4
    assert doc["input"]["ebv_source"] == "FIBERMAP" and abs(doc["input"]["ebv"] - 0.0323) < 1e-3
    ha = doc["lines"]["Halpha"]
    assert (
        ha["fitted"]
        and ha["label"] == DESI_PIN["HA_class"] == "F"
        and abs(ha["dv"] - DESI_PIN["HA_c50_sys"]) < END_POINT_KMS
        and ",".join(ha["flags"]) == DESI_PIN["HA_flags"]
        and ha["measurable"]
    )
    assert doc["input"]["z_source"] == "redrock"
    assert doc["continuum"]["host_applied"]
    # the printed table: one error column, and every flag of the fit explained once below it
    printed = capsys.readouterr().out
    header = next(line for line in printed.splitlines() if line.startswith("line "))
    assert "+/-mc" in header and "+/-mod" not in header
    for name, rec in doc["lines"].items():
        for flag in rec["flags"]:
            assert printed.count(f"  {flag}: {FLAG_TEXT[flag]}") == 1
        for velocity in velocities_at_bound(rec["params_at_bound"]):
            assert f"{name} {velocity}" in printed


def test_fit_help_groups_the_options_and_hides_legacy_settings(capsys):
    with pytest.raises(SystemExit) as e:
        main(["fit", "--help"])
    assert e.value.code == 0
    text = capsys.readouterr().out
    for group in ("input:", "public search (no file given):", "model:", "uncertainties:", "output:"):
        assert f"\n{group}\n" in text
    for legacy in ("--sdss-mask-policy", "--mc-noise-policy", "--legacy-error-diagnostic", "generic,auto"):
        assert legacy not in text
    # hidden, not removed: the settings that reproduce earlier releases still parse
    a = build_parser().parse_args(
        [
            "fit",
            "spectrum.fits",
            "--survey",
            "auto",
            "--sdss-mask-policy",
            "ivar",
            "--mc-noise-policy",
            "effective",
            "--legacy-error-diagnostic",
        ]
    )
    assert a.survey == "auto" and a.sdss_mask_policy == "ivar" and a.mc_noise_policy == "effective"
    assert a.legacy_error_diagnostic


def test_fit_csv_example_matches_sdss_fit(tmp_path):
    main(
        [
            "fit",
            SDSS_EXAMPLE,
            "--survey",
            "sdss",
            "--sdss-mask-policy",
            "ivar",
            "--z",
            str(Z_J001224),
            "--lines",
            "Hbeta",
            "--out",
            str(tmp_path),
            "--no-figure",
            "--quiet",
        ]
    )
    rc = main(
        [
            "fit",
            CSV_EXAMPLE,
            "--survey",
            "generic",
            "--wave",
            "lambda_nm",
            "--flux",
            "f_lambda",
            "--err",
            "sigma",
            "--wave-unit",
            "nm",
            "--frame",
            "rest",
            "--air",
            "--z",
            str(Z_J001224),
            "--flux-scale",
            "10",
            "--lines",
            "Hbeta",
            "--out",
            str(tmp_path),
            "--no-figure",
            "--quiet",
        ]
    )
    assert rc == 0
    a = json.load(open(tmp_path / "spec-0651-52141-0072_fit.json"))["lines"]["Hbeta"]
    b = json.load(open(tmp_path / "J001224_rest_air_nm_fit.json"))["lines"]["Hbeta"]
    assert b["label"] == a["label"] and abs(b["dv"] - a["dv"]) < 2.0
    assert b["broad_flux"] == pytest.approx(a["broad_flux"], rel=1e-3)


def test_fit_missing_redshift_exits():
    with pytest.raises(SystemExit):
        main(
            [
                "fit",
                CSV_EXAMPLE,
                "--survey",
                "generic",
                "--wave",
                "lambda_nm",
                "--flux",
                "f_lambda",
                "--err",
                "sigma",
                "--frame",
                "rest",
                "--quiet",
            ]
        )
    with pytest.raises(SystemExit):
        main(["fit", DESI_EXAMPLE, "--survey", "auto", "--quiet"])  # a coadd needs --targetid


def test_rv_command_points_to_the_python_module(capsys):
    with pytest.raises(SystemExit) as e:
        main(["rv", SDSS_EXAMPLE, SDSS_EXAMPLE_2])
    assert e.value.code == 2 and "blrfit.rv" in capsys.readouterr().err


def test_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
