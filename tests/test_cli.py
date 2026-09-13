"""The command-line tool on the bundled examples."""
import json

import numpy as np
import pytest

from blrfit.cli import main
from conftest import SDSS_EXAMPLE, SDSS_EXAMPLE_2, DESI_EXAMPLE, DESI_TARGETID, CSV_EXAMPLE, Z_J001224

# the end-point tolerance of tests/test_pins.py: the least-squares solver ends at a platform-dependent
# point of a degenerate decomposition, which moves the primary offset by tens of km/s but not the class
END_POINT_KMS = 100.0


def test_fit_sdss_example(tmp_path):
    rc = main(["fit", SDSS_EXAMPLE, "--z", str(Z_J001224), "--out", str(tmp_path), "--no-figure", "--quiet"])
    assert rc == 0
    doc = json.load(open(tmp_path / "spec-0651-52141-0072_fit.json"))
    assert doc["input"]["kind"] == "sdss" and doc["input"]["z"] == Z_J001224 and doc["input"]["z_source"] == "argument"
    assert doc["input"]["ebv"] == 0.0
    ha, hb, mg = doc["lines"]["Halpha"], doc["lines"]["Hbeta"], doc["lines"]["MgII"]
    # the pinned values, to the end-point tolerance
    assert ha["fitted"] and ha["label"] == "C" and abs(ha["dv"] - (-1061.0)) < END_POINT_KMS and ha["strong_offset"]
    assert hb["fitted"] and hb["label"] == "C" and abs(hb["dv"] - (-1243.5)) < END_POINT_KMS
    assert hb["systemic_source"] == "Halpha prior"
    assert not mg["fitted"] and mg["label"] == "" and mg["class_text"] == "not fitted" and "not fitted" in mg["reasons"][0]
    assert mg["dv"] is None
    assert ha["dv_err_mc"] is None and ha["dv_err_model"] == pytest.approx(1.5 * max(650 / np.sqrt(ha["broad_flux_snr"]), 45), rel=1e-6)
    assert doc["summary_row"]["HA_class"] == "C"
    assert doc["settings"]["complexes"] == ["Halpha", "Hbeta", "MgII"]
    assert doc["blrfit_version"]
    assert not (tmp_path / "spec-0651-52141-0072_fit.png").exists()


def test_fit_writes_figure_and_pickle(tmp_path):
    rc = main(["fit", SDSS_EXAMPLE, "--z", str(Z_J001224), "--lines", "Hbeta", "--out", str(tmp_path), "--pickle", "--quiet"])
    assert rc == 0
    assert (tmp_path / "spec-0651-52141-0072_fit.png").stat().st_size > 10000
    assert (tmp_path / "spec-0651-52141-0072_fit.pkl").exists()
    doc = json.load(open(tmp_path / "spec-0651-52141-0072_fit.json"))
    assert list(doc["lines"]) == ["Hbeta"]


def test_fit_desi_example_uses_redrock_and_fibermap(tmp_path):
    rc = main(["fit", DESI_EXAMPLE, "--targetid", str(DESI_TARGETID), "--lines", "Halpha,Hbeta", "--out", str(tmp_path),
               "--no-figure", "--quiet"])
    assert rc == 0
    doc = json.load(open(tmp_path / "coadd-main-dark-17260-39627574082538900_fit.json"))
    assert doc["input"]["kind"] == "desi" and abs(doc["input"]["z"] - 0.22032) < 1e-4
    assert doc["input"]["ebv_source"] == "FIBERMAP" and abs(doc["input"]["ebv"] - 0.0323) < 1e-3
    ha = doc["lines"]["Halpha"]
    assert ha["fitted"] and ha["label"] == "F" and abs(ha["dv"] + 37.0) < END_POINT_KMS and ha["flags"] == [] and ha["measurable"]
    assert doc["input"]["z_source"] == "redrock"
    assert doc["continuum"]["host_applied"]


def test_fit_csv_example_matches_sdss_fit(tmp_path):
    main(["fit", SDSS_EXAMPLE, "--z", str(Z_J001224), "--lines", "Hbeta", "--out", str(tmp_path), "--no-figure", "--quiet"])
    rc = main(["fit", CSV_EXAMPLE, "--wave", "lambda_nm", "--flux", "f_lambda", "--err", "sigma", "--wave-unit", "nm",
               "--frame", "rest", "--air", "--z", str(Z_J001224), "--flux-scale", "10", "--lines", "Hbeta",
               "--out", str(tmp_path), "--no-figure", "--quiet"])
    assert rc == 0
    a = json.load(open(tmp_path / "spec-0651-52141-0072_fit.json"))["lines"]["Hbeta"]
    b = json.load(open(tmp_path / "J001224_rest_air_nm_fit.json"))["lines"]["Hbeta"]
    assert b["label"] == a["label"] and abs(b["dv"] - a["dv"]) < 2.0
    assert b["broad_flux"] == pytest.approx(a["broad_flux"], rel=1e-3)


def test_fit_missing_redshift_exits():
    with pytest.raises(SystemExit):
        main(["fit", CSV_EXAMPLE, "--wave", "lambda_nm", "--flux", "f_lambda", "--err", "sigma", "--frame", "rest", "--quiet"])
    with pytest.raises(SystemExit):
        main(["fit", DESI_EXAMPLE, "--quiet"])          # a coadd needs --targetid


def test_rv_two_sdss_epochs(tmp_path):
    rc = main(["rv", SDSS_EXAMPLE, SDSS_EXAMPLE_2, "--z", str(Z_J001224), "--line", "Hbeta", "--out", str(tmp_path), "--quiet"])
    assert rc == 0
    doc = json.load(open(tmp_path / "spec-0651-52141-0072_vs_spec-7169-56628-0344_rv.json"))
    assert doc["measured"] and doc["line"] == "Hbeta"
    assert doc["baseline_days"] == pytest.approx(56628 - 52141) and doc["baseline_rest_yr"] == pytest.approx(4487 / 365.25 / 1.2288, rel=1e-6)
    assert abs(doc["dv"] - (-294)) < 40 and 10 < doc["err"] < 100
    assert doc["consistent"] and not doc["at_bound"] and not doc["regridded"]
    assert doc["zp_line"] == "OIII" and abs(doc["zp_dv"]) < 40 and doc["zp_applied"]
    assert doc["dv_corrected"] == pytest.approx(doc["dv"] - doc["zp_dv"])
    assert doc["dv_abs_epoch2"] == pytest.approx(doc["epochs"][0]["c50_sys"] + doc["dv_corrected"])
    assert doc["sigma_sys_desi"] == 79.0
    assert doc["profile_z"] > 5 and doc["reliable"] is False     # the profile changed shape between 2001 and 2013
    assert doc["profile_grade"] == "changed" and 0.0 < doc["resid_frac"] < 0.2
    assert doc["error_floor_kind"] == "cross_survey_null_changed" and doc["error_floor"] == pytest.approx(147.0 * 1.5)
    assert doc["err_total"] == pytest.approx(np.hypot(doc["err"], doc["error_floor"]))
    assert doc["lines"]["Hbeta"]["dv"] == doc["dv"]
    assert (tmp_path / "spec-0651-52141-0072_vs_spec-7169-56628-0344_rv.png").stat().st_size > 10000


def test_rv_two_lines_and_the_two_line_criterion(tmp_path):
    rc = main(["rv", SDSS_EXAMPLE, SDSS_EXAMPLE_2, "--z", str(Z_J001224), "--line", "Halpha,Hbeta", "--out", str(tmp_path),
               "--no-figure", "--quiet"])
    assert rc == 0
    doc = json.load(open(tmp_path / "spec-0651-52141-0072_vs_spec-7169-56628-0344_rv.json"))
    assert set(doc["lines"]) == {"Halpha", "Hbeta"}
    ha, hb = doc["lines"]["Halpha"], doc["lines"]["Hbeta"]
    assert ha["measured"] and hb["measured"]
    assert abs(ha["dv"] - (-219)) < 40 and abs(hb["dv"] - (-294)) < 40
    assert ha["profile_grade"] == "changed" and hb["profile_grade"] == "changed"
    assert doc["two_line"]["consistent"] and doc["two_line"]["same_sign"]
    assert doc["line"] == "Halpha" and doc["dv"] == ha["dv"]        # the first line at the top level


def test_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
