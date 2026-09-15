"""Inputs that used to be fitted into a silent wrong answer, or to fail later with
a misleading message, are refused or handled where they arise."""
import warnings

import numpy as np
import pytest

from blrfit import rv
from blrfit.model.continuum import fit_continuum_host
from blrfit.model.fit import fit_spectrum
from synth import make_spectrum


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def test_continuum_without_pixels_outside_the_complexes_is_refused():
    """A cut-out of the Halpha complex (6410-6790 A rest) leaves no continuum pixel;
    least squares on no residual used to return the starting values as a
    successful continuum (2.5 times too low, equivalent width 5.6 times too high)."""
    z = 0.1
    sp = make_spectrum(z=z, snr=20, broad=[dict(line="Halpha", v=1500., fwhm=4000., ew=300.)], seed=1)
    wr = sp["wave"] / (1 + z)
    s = (wr > 6410.) & (wr < 6790.)
    with pytest.raises(ValueError, match="outside the line complexes"):
        _quiet(fit_spectrum, sp["wave"][s], sp["flux"][s], sp["ivar"][s], z, complexes=("Halpha",))


def test_host_is_not_subtracted_when_the_blue_flux_sum_is_not_positive():
    """A flux summed over 4200-5000 A that is not positive (sky over-subtraction):
    the clamped denominator gave host fractions near 1e33 that passed the 0.1 cut."""
    z = 0.1
    sp = make_spectrum(z=z, snr=20, host_frac=0.5, broad=[dict(line="Halpha", v=0., fwhm=4000., ew=300.)], seed=3)
    wave, flux, ivar = sp["wave"], sp["flux"].copy(), sp["ivar"].copy()
    wr = wave / (1 + z)
    blue = (wr > 4150) & (wr < 5050)
    flux[blue] -= 3.0 * np.median(flux[blue])
    assert flux[(wr > 4200) & (wr < 5000)].sum() < 0
    d, total, host, info = _quiet(fit_continuum_host, wr, flux * (1 + z), ivar / (1 + z) ** 2)
    assert not info["applied"] and not np.any(host)
    assert info["n_gal"] == 0 or np.isnan(info["host_frac_4200_5000"])


@pytest.mark.parametrize("kw, match", [(dict(z=np.nan), "z must be"), (dict(z=-1.0), "z must be"),
                                       (dict(z=np.inf), "z must be"), (dict(sig_broad_min=np.nan), "sig_broad_min")])
def test_invalid_arguments_of_the_fit_are_refused(kw, match):
    sp = make_spectrum(z=0.1, snr=20, broad=[dict(line="Halpha", v=0., fwhm=4000., ew=300.)], seed=2)
    kw = dict(kw)
    z = kw.pop("z", 0.1)
    with pytest.raises(ValueError, match=match):
        fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z, complexes=("Halpha",), **kw)


def test_fully_repeated_wavelengths_are_refused():
    """Two exposures concatenated: the median pixel spacing is zero and the line
    measurements divided by it (ZeroDivisionError after the whole fit)."""
    sp = make_spectrum(z=0.1, snr=20, broad=[dict(line="Halpha", v=0., fwhm=4000., ew=300.)], seed=2)
    w, f, iv = (np.repeat(sp[k], 2) for k in ("wave", "flux", "ivar"))
    with pytest.raises(ValueError, match="wavelength steps are zero"):
        fit_spectrum(w, f, iv, 0.1, complexes=("Halpha",))


def test_descending_wavelengths_give_the_increasing_fit():
    sp = make_spectrum(z=0.1, snr=20, broad=[dict(line="Halpha", v=300., fwhm=4000., ew=300.)], seed=6)
    up = _quiet(fit_spectrum, sp["wave"], sp["flux"], sp["ivar"], 0.1, complexes=("Halpha",))
    down = _quiet(fit_spectrum, sp["wave"][::-1], sp["flux"][::-1], sp["ivar"][::-1], 0.1, complexes=("Halpha",))
    for k in ("c50_sys", "fwhm", "broad_flux"):
        assert down["meas"]["Halpha"][k] == up["meas"]["Halpha"][k]
    assert down["cls"]["Halpha"]["label"] == up["cls"]["Halpha"]["label"]


def test_repeats_in_camera_overlaps_are_fitted():
    """Camera arrays concatenated without a coadd repeat the wavelengths of the
    overlaps (a few per cent of the pixels); such a spectrum is fitted as before."""
    sp = make_spectrum(z=0.1, snr=20, broad=[dict(line="Halpha", v=300., fwhm=4000., ew=300.)], seed=7)
    n = sp["wave"].size
    rep_idx = np.arange(n // 3, n // 3 + 60)
    order = np.sort(np.concatenate([np.arange(n), rep_idx]), kind="stable")
    w, f, iv = sp["wave"][order], sp["flux"][order], sp["ivar"][order]
    r = _quiet(fit_spectrum, w, f, iv, 0.1, complexes=("Halpha",))
    ref = _quiet(fit_spectrum, sp["wave"], sp["flux"], sp["ivar"], 0.1, complexes=("Halpha",))
    assert abs(r["meas"]["Halpha"]["c50_sys"] - ref["meas"]["Halpha"]["c50_sys"]) < 10.0


def _profile(v, centre, rng, **extra):
    f = np.exp(-0.5 * ((v - centre) / (4000.0 / 2.3548)) ** 2) + rng.normal(0.0, 0.01, v.size)
    return dict(v=v, f=f, e=np.full(v.size, 0.01), ok=np.ones(v.size, bool), nmod=np.zeros(v.size), **extra)


@pytest.mark.parametrize("kw", [dict(nsub_frac=np.nan), dict(mismatch=np.nan), dict(win_fwhm=np.nan),
                                dict(win_min=np.nan), dict(clip=np.nan)])
def test_nan_arguments_of_the_cross_correlation_are_refused(kw):
    rng = np.random.default_rng(4)
    v = np.arange(-15000.0, 15000.01, 30.0)
    t = _profile(v, 0.0, rng, fwhm=4000.0, c50_sys=0.0, v_sys=0.0)
    p = _profile(v, 150.0, rng, fwhm=4000.0, c50_sys=150.0, v_sys=0.0)
    with pytest.raises(ValueError):
        rv.ccf_shift(p, t, **kw)


def test_unused_none_arguments_of_the_cross_correlation_still_work():
    rng = np.random.default_rng(4)
    v = np.arange(-15000.0, 15000.01, 30.0)
    t = _profile(v, 0.0, rng, fwhm=4000.0, c50_sys=0.0, v_sys=0.0)
    p = _profile(v, 150.0, rng, fwhm=4000.0, c50_sys=150.0, v_sys=0.0)
    assert rv.ccf_shift(p, t, clip=None, n_clip=0, despike=False) is not None
    assert rv.ccf_shift(p, t, win_fwhm=None, window=(-5000.0, 5000.0)) is not None


def test_window_follows_c50_when_the_systemic_velocity_is_not_measured():
    """Without v_sys (and so without c50_sys) the window was centred on 0 km/s and
    cut an offset profile; it now follows c(1/2) relative to the input redshift."""
    rng = np.random.default_rng(5)
    v = np.arange(-15000.0, 15000.01, 30.0)
    t = _profile(v, 4500.0, rng, fwhm=2000.0, c50_sys=np.nan, v_sys=np.nan, c50=4500.0)
    p = _profile(v, 4700.0, rng, fwhm=2000.0, c50_sys=np.nan, v_sys=np.nan, c50=4700.0)
    t["f"] = np.exp(-0.5 * ((v - 4500.0) / (2000.0 / 2.3548)) ** 2) + rng.normal(0.0, 0.01, v.size)
    p["f"] = np.exp(-0.5 * ((v - 4700.0) / (2000.0 / 2.3548)) ** 2) + rng.normal(0.0, 0.01, v.size)
    r = rv.ccf_shift(p, t)
    assert np.mean(r["window"]) == pytest.approx(4500.0)
    assert abs(r["dv"] - 200.0) < 20.0
    # with the systemic velocity measured, the window is unchanged: c50_sys + v_sys
    t2 = dict(t, c50_sys=4000.0, v_sys=500.0, c50=np.nan)
    assert np.mean(rv.ccf_shift(p, t2)["window"]) == pytest.approx(4500.0)


def test_offset_mgii_without_a_narrow_component_is_measured_about_its_c50():
    """The case found on synthetic spectra: broad Mg II at +4500 and +4700 km/s fitted
    with mgii_narrow=False (no systemic velocity). Default window: 292 +/- 96 km/s
    before, the true 200 km/s now."""
    z = 1.0
    fits = []
    for seed, v in ((11, 4500.0), (12, 4700.0)):
        sp = make_spectrum(z=z, snr=25, broad=[dict(line="MgII", v=v, fwhm=2000., ew=80.)],
                           narrow=dict(ew_ha=0.0), seed=seed)
        fits.append(_quiet(fit_spectrum, sp["wave"], sp["flux"], sp["ivar"], z, complexes=("MgII",), mgii_narrow=False))
    assert not np.isfinite(fits[0]["meas"]["MgII"]["v_sys"])
    r = _quiet(rv.ccf_shift, rv.broad_profile_data(fits[1], "MgII"), rv.broad_profile_data(fits[0], "MgII"))
    assert abs(r["dv"] - 200.0) < 30.0 and r["err"] < 30.0
