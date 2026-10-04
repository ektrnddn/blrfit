"""Regression cases for the defects corrected in 0.2.0 (see CHANGELOG.md)."""
import copy

import numpy as np
import pytest
from astropy.table import Table

import blrfit
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from blrfit.model.continuum import fe_templates
from blrfit.constants import S2F
from blrfit.io.generic import read_table
from synth import make_spectrum


@pytest.fixture(scope="module", params=[False, True])
def systemic_fit(request):
    tied = request.param
    s = make_spectrum(snr=30., narrow=dict(o3=0.), seed=7,
        broad=[dict(line="Halpha", v=1200., fwhm=4000., ew=150.),
               dict(line="Hbeta", v=1200., fwhm=4000., ew=50.)])
    r = blrfit.fit_spectrum(s["wave"], s["flux"], s["ivar"], .25,
        host=False, fe=False, max_broad=1, use_ha_systemic=tied)
    return tied, r


def test_systemic_reference_tracks_actual_constraint(systemic_fit):
    tied, r = systemic_fit
    m = r["meas"]["Hbeta"]
    assert m["o3_core_snr"] < 3.
    assert r["settings"]["use_ha_systemic"] is tied
    if tied:
        assert m["systemic_source"] == "Halpha prior"
        assert m["sys_snr"] == r["meas"]["Halpha"]["narrow_peak_snr"]
        assert r["fits"]["Hbeta"]["ps"].fixed["n_v"] == m["v_sys"]
    else:
        assert m["systemic_source"] == "[OIII] core"
        assert m["sys_snr"] == m["o3_core_snr"]
        assert r["cls"]["Hbeta"]["label"] == "X"
        assert "n_v" not in r["fits"]["Hbeta"]["ps"].fixed


def test_remeasure_preserves_luminosity_and_repairs_old_label(systemic_fit):
    tied, original = systemic_fit
    r = copy.deepcopy(original)
    # Older releases mislabeled independent Hbeta fits; fitted parameters
    # retain enough information to recover the reference without a refit.
    r["meas"]["Hbeta"]["systemic_source"] = "Halpha prior"
    blrfit.remeasure(r)
    for line in ("Halpha", "Hbeta"):
        assert r["meas"][line]["broad_lum"] == pytest.approx(original["meas"][line]["broad_lum"])
    assert r["meas"]["Hbeta"]["systemic_source"] == original["meas"]["Hbeta"]["systemic_source"]
    assert r["cls"]["Hbeta"]["label"] == original["cls"]["Hbeta"]["label"]


def test_compressed_fits_table(tmp_path):
    path = tmp_path / "spectrum.fits.gz"
    Table(dict(wave=[4500., 4501., 4502.], flux=[2., 3., 4.], err=[.2, .3, .4])).write(path)
    s = read_table(path, err="err")
    np.testing.assert_array_equal(s["wave"], [4500., 4501., 4502.])
    np.testing.assert_allclose(s["ivar"], 1 / np.array([.2, .3, .4])**2)


@pytest.mark.parametrize('template_index', [0, 1])
def test_fe_width_recovery_from_distant_start(template_index):
    fe = fe_templates()[template_index]
    wave = np.linspace(max(fe.wmin + 100, fe.wave.min()),
                       min(fe.wmax - 100, fe.wave.max()), 2000)
    width = 6473.0
    # Independent direct convolution supplies a non-grid-aligned truth.
    sigma = np.sqrt(width**2 - fe.intrinsic**2) / S2F / fe.pix_kms
    truth = np.interp(wave, fe.wave, gaussian_filter1d(fe.flux, sigma, mode='nearest'))
    result = least_squares(lambda p: fe(wave, p[0], p[1], p[2]) - truth,
                           [0.8, 3000., 0.0001],
                           bounds=([0.1, 1200., -.01], [3., 10000., .01]),
                           x_scale='jac', gtol=1e-10, ftol=1e-10, xtol=1e-10)
    assert result.success
    assert result.x[1] == pytest.approx(width, abs=1.)
    assert result.x[0] == pytest.approx(1., abs=1e-4)
    assert abs(result.x[2]) < 1e-6
    shifted = 3000. * (1 + np.sqrt(np.finfo(float).eps))
    assert np.max(np.abs(fe.broadened(shifted) - fe.broadened(3000.))) > 0
