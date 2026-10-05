"""
Equally good decompositions of a line complex: the ``degenerate`` flag.

The starts of a complex can end at points whose chi-square differ by less than
the noise while their broad profiles differ. ``measure_complex`` compares the
end points within DEGENERATE_DCHI2 x max(1, reduced chi-square) of the
selected one and reports the span of their c(1/2) - v_sys as ``dv_spread``;
the line is flagged ``degenerate`` above DEGENERATE_DV_KMS. The documented
case is Halpha of the 2001 SDSS spectrum of J001224 with the single-start
continuum of version 0.2: two decompositions 0.8 apart in chi-square put
c(1/2) at -1060 and -1524 km/s.
"""

import numpy as np
import pytest

import blrfit
from blrfit.classify import classify
from blrfit.constants import DEGENERATE_DV_KMS
from blrfit.io import read_sdss
from blrfit.measure import equivalent_end_points
from conftest import SDSS_EXAMPLE, Z_J001224
from synth import make_spectrum


@pytest.fixture(scope="module")
def j001224_single_start():
    sp = read_sdss(SDSS_EXAMPLE)
    return blrfit.fit_spectrum(
        sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Halpha", "Hbeta"), conti_multistart=False
    )


def test_j001224_halpha_is_degenerate_with_the_single_start_continuum(j001224_single_start):
    m, cls = j001224_single_start["meas"]["Halpha"], j001224_single_start["cls"]["Halpha"]
    assert m["n_equivalent"] >= 2
    assert m["dv_spread"] > 400.0 and m["fwhm_spread"] > 400.0
    assert "degenerate" in cls["flags"]
    hb = j001224_single_start["meas"]["Hbeta"]
    assert (
        hb["dv_spread"] < DEGENERATE_DV_KMS
        and "degenerate" not in j001224_single_start["cls"]["Hbeta"]["flags"]
    )
    row = blrfit.summary_row(j001224_single_start)
    assert row["HA_dv_spread"] == m["dv_spread"] and "degenerate" in row["HA_flags"].split(",")


def test_a_single_gaussian_line_is_not_degenerate():
    z = 0.25
    s = make_spectrum(
        z=z,
        snr=25.0,
        seed=7,
        broad=[dict(line="Halpha", v=600.0, fwhm=4000.0, ew=120.0)],
        narrow=dict(ew_ha=20.0),
    )
    res = blrfit.fit_spectrum(s["wave"], s["flux"], s["ivar"], z, complexes=("Halpha",))
    m = res["meas"]["Halpha"]
    assert m["n_equivalent"] >= 1 and m["dv_spread"] < 20.0
    assert "degenerate" not in res["cls"]["Halpha"]["flags"]


def test_threshold_and_missing_end_points(j001224_single_start):
    m = dict(j001224_single_start["meas"]["Halpha"])
    for spread, flagged in ((DEGENERATE_DV_KMS + 1.0, True), (DEGENERATE_DV_KMS, False), (np.nan, False)):
        m["dv_spread"] = spread
        assert ("degenerate" in classify(m)["flags"]) is flagged, spread
    # a fit without end points (written before version 0.3) has no spread
    fit = dict(j001224_single_start["fits"]["Halpha"])
    fit.pop("end_points")
    dv, dw, n = equivalent_end_points(fit, 6564.61, np.arange(-25000.0, 25000.01, 5.0))
    assert np.isnan(dv) and np.isnan(dw) and n == 0
