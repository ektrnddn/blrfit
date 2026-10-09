"""
The statistical-error path of the pairs against a real fit: a synthetic
SDSS-like observed-frame spectrum with a continuum, broad and narrow lines,
Galactic extinction, flux-dependent noise and a run of bad pixels is fitted
with ``fit_spectrum``; then ``epoch_profile`` must find every fit pixel in the
input, rebuild the fit's own weights exactly from the statistical variance and
the floor (``weights_consistent``), give the same errors from the input arrays
and from the result alone, and measure a zero shift for the epoch against
itself.
"""

import numpy as np
import pytest

import blrfit
from blrfit import pairs
from blrfit.constants import LAM
from blrfit.model.params import gauss_lam


def synthetic_sdss_spectrum(z=0.08, seed=11):
    rng = np.random.default_rng(seed)
    wave = 10.0 ** np.arange(np.log10(3800.0), np.log10(9200.0), 1e-4)
    rest = wave / (1.0 + z)
    cont = 20.0 * (rest / 5500.0) ** -1.2
    lines = [
        (LAM["Halpha"], 12.0, 0.0, 1700.0),
        (LAM["Halpha"], 9.0, 0.0, 130.0),
        (LAM["NII6584"], 7.0, 0.0, 130.0),
        (LAM["NII6548"], 7.0 / 2.96, 0.0, 130.0),
        (LAM["SII6716"], 2.5, 0.0, 130.0),
        (LAM["SII6731"], 2.0, 0.0, 130.0),
        (LAM["Hbeta"], 4.0, 0.0, 1700.0),
        (LAM["Hbeta"], 2.5, 0.0, 130.0),
        (LAM["OIII5007"], 14.0, 0.0, 130.0),
        (LAM["OIII4959"], 14.0 / 2.98, 0.0, 130.0),
    ]
    model = cont.copy()
    for l0, A, v, s in lines:
        model += gauss_lam(rest, A, l0, v, s) / (1.0 + z)
    sigma = 0.8 + 0.03 * np.abs(model)
    flux = model + rng.normal(0.0, sigma)
    ivar = 1.0 / sigma**2
    k = np.searchsorted(rest, 6500.0)
    ivar[k : k + 5] = 0.0
    flux[k + 2] = np.nan
    ivar[np.searchsorted(rest, 6650.0)] = 0.0
    return wave, flux, ivar, z


@pytest.fixture(scope="module")
def fitted():
    wave, flux, ivar, z = synthetic_sdss_spectrum()
    res = blrfit.fit_spectrum(wave, flux, ivar, z, ebv=0.04, complexes=("Halpha",), host=False, nmc=0)
    assert res["fits"].get("Halpha") is not None, res.get("fit_status")
    return res, dict(wave=wave, flux=flux, ivar=ivar)


def test_statistical_errors_follow_the_fit(fitted):
    res, inputs = fitted
    ep = pairs.epoch_profile(res, "Halpha", inputs=inputs)
    assert ep["errors_source"] == "input spectrum"
    assert ep["weights_consistent"], ep["weight_max_rel_dev"]
    # the floor matters on the bright pixels: statistical errors are smaller than 1/w there
    ratio = (1.0 / ep["w"][ep["ok"]]) / ep["sig"][ep["ok"]]
    assert ratio.max() > 1.05 and ratio.min() >= 1.0 - 1e-9
    # the fit stored only unmasked pixels: the grown bad run is absent
    assert not np.any((ep["x"] > 6499.0) & (ep["x"] < 6501.5))


def test_result_alone_gives_the_same_errors(fitted):
    """A full fit result keeps the inverse variance before the floor on its grid:
    the errors from it equal those from the input arrays, so saved fits can be
    paired without their spectra."""
    res, inputs = fitted
    ep_in = pairs.epoch_profile(res, "Halpha", inputs=inputs)
    ep_res = pairs.epoch_profile(res, "Halpha")
    assert ep_res["errors_source"] == "fit result" and ep_res["weights_consistent"]
    assert np.allclose(ep_in["sig"], ep_res["sig"], rtol=1e-12, equal_nan=True)
    reduced = dict(res)
    reduced.pop("ivar_stat_rest")
    ep_w = pairs.epoch_profile(reduced, "Halpha")
    assert ep_w["errors_source"].startswith("fit weights") and ep_w["weights_consistent"] is None


def test_self_pair_is_zero(fitted):
    res, inputs = fitted
    rec = pairs.measure_pair(res, res, "Halpha", inputs_a=inputs, inputs_b=inputs)
    assert abs(rec["s_common"]) < 0.5 and np.isfinite(rec["err"]) and rec["err"] > 0
    assert abs(rec["dchi2_shape_fwd"]) < 1.0 and abs(rec["dv_narrow"]) < 0.5
    assert rec["chi2_nu_fwd"] < 2.0 and rec["retained"]
    rec2 = pairs.measure_pair(res, res, "Halpha")
    assert abs(rec2["s_common"] - rec["s_common"]) < 1e-6 and abs(rec2["err"] - rec["err"]) < 1e-6
    assert rec2["cls_a"] == res["cls"]["Halpha"]["label"]


def test_epoch_profile_refuses_mismatched_input(fitted):
    res, inputs = fitted
    with pytest.raises(ValueError, match="no input pixel"):
        pairs.epoch_profile(res, "Halpha", inputs=dict(inputs, wave=inputs["wave"] * (1 + 1e-5)))
    # the same grid with another E(B-V) in the record than in the preprocessing: the weight check fails
    res2 = dict(res, settings=dict(res["settings"], ebv=0.0))
    ep = pairs.epoch_profile(res2, "Halpha", inputs=inputs)
    assert ep["weights_consistent"] is False
    rec = pairs.measure_pair(res2, res, "Halpha", inputs_a=inputs, inputs_b=inputs)
    assert "weights_inconsistent_a" in rec["flags"] and "ebv_differs" in rec["flags"]


def test_line_not_fitted(fitted):
    res, _inputs = fitted
    rec = pairs.measure_pair(res, res, "Hbeta")
    assert rec["flags"] == ["line_not_fitted"] and "s_common" not in rec
    row = pairs.pair_row(rec, res, dict(id="a"), dict(id="b"))
    assert row["flags"] == "line_not_fitted" and not row["retained"] and np.isnan(row["s_common"])
