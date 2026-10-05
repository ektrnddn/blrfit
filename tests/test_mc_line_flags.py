"""
Monte Carlo flags carried with the line, and errors withheld when the draws
cannot support them (MC_ERROR_INVALID_FLAGS): split solutions, a basin switch,
too few draws to test that, or too few offsets. The percentiles stay in
``res['mc']``; class A then uses the offset threshold alone.
"""

import numpy as np
import pytest

import blrfit
from blrfit.cli import main
from blrfit.constants import MC_ERROR_INVALID_FLAGS, MC_LINE_FLAGS
from blrfit.io import read_sdss
from blrfit.model.fit import classify_lines
from conftest import SDSS_EXAMPLE, Z_J001224
from synth import make_spectrum


def test_j001224_single_start_draws_split_and_errors_are_withheld():
    """Halpha of J001224 with the single-start continuum of 0.2: the draws split
    between the two decompositions of tests/test_degenerate.py."""
    sp = read_sdss(SDSS_EXAMPLE)
    res = blrfit.fit_spectrum(
        sp["wave"],
        sp["flux"],
        sp["ivar"],
        Z_J001224,
        complexes=("Halpha", "Hbeta"),
        conti_multistart=False,
        nmc=25,
        seed=0,
    )
    ha = res["mc_info"]["lines"]["Halpha"]
    assert "mc_multimodal" in ha["flags"] and ha["errors_withheld"] == ["mc_multimodal"]
    assert "mc_multimodal" in res["cls"]["Halpha"]["flags"]
    assert all(np.isnan(v) for v in res["err"]["Halpha"].values())
    assert np.all(np.isfinite(res["mc"]["Halpha"]["c50_sys"]))
    row = blrfit.summary_row(res)
    assert "mc_multimodal" in row["HA_flags"].split(",") and np.isnan(row["HA_e_c50_sys"])
    # Hbeta keeps its error
    assert "errors_withheld" not in res["mc_info"]["lines"]["Hbeta"]
    assert np.isfinite(res["err"]["Hbeta"]["c50_sys"]) and res["err"]["Hbeta"]["c50_sys"] > 0


@pytest.fixture(scope="module")
def shifted_line():
    """A symmetric broad Halpha displaced by +450 km/s: class A on the offset alone."""
    z = 0.25
    s = make_spectrum(
        z=z,
        snr=30.0,
        seed=11,
        broad=[dict(line="Halpha", v=450.0, fwhm=4000.0, ew=120.0)],
        narrow=dict(ew_ha=20.0),
    )
    return blrfit.fit_spectrum(s["wave"], s["flux"], s["ivar"], z, complexes=("Halpha",))


def test_withheld_errors_leave_class_a_to_the_offset_threshold(shifted_line):
    res = shifted_line
    assert res["cls"]["Halpha"]["label"] == "A"
    # a finite error of 200 km/s makes the 450 km/s offset insignificant (less than three errors)
    res["err"] = {"Halpha": {"c50_sys": 200.0, "fwhm": 80.0}}
    res["mc_info"] = {"lines": {"Halpha": {"flags": []}}}
    classify_lines(res)
    assert res["cls"]["Halpha"]["label"] == "F"
    # the same error under a flag that invalidates it is withheld: A again, and the flag is shown
    res["err"] = {"Halpha": {"c50_sys": 200.0, "fwhm": 80.0}}
    res["mc_info"] = {"lines": {"Halpha": {"flags": ["mc_basin_switch", "unconverged_line_draws"]}}}
    classify_lines(res)
    assert res["cls"]["Halpha"]["label"] == "A"
    assert all(np.isnan(v) for v in res["err"]["Halpha"].values())
    assert res["mc_info"]["lines"]["Halpha"]["errors_withheld"] == ["mc_basin_switch"]
    flags = res["cls"]["Halpha"]["flags"]
    assert "mc_basin_switch" in flags and "unconverged_line_draws" not in flags


def test_flag_sets():
    assert set(MC_ERROR_INVALID_FLAGS) <= set(MC_LINE_FLAGS)
    assert {"mc_multimodal", "mc_basin_switch", "mc_too_few"} <= set(MC_ERROR_INVALID_FLAGS)


def test_cli_refuses_too_few_draws(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "fit",
                SDSS_EXAMPLE,
                "--survey",
                "sdss",
                "--z",
                str(Z_J001224),
                "--nmc",
                "10",
                "--out",
                str(tmp_path),
            ]
        )
    assert "at least 25" in str(exc.value)
