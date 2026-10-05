"""
The host-fraction guard (``host_info['host_undetermined']``).

The host fraction is the host's share of the 4200-5000 A flux. When that
window carries no signal the fraction is undetermined and the host is not
subtracted: the inverse-variance weighted flux sum of the window is not
positive, or lies below 3 sigma of its own noise (the variance summed from
ivar, the pixels taken as independent), or more than half of the window's
pixels are masked. The fit then continues without the host, as it does when
the host is not supported, with ``host_frac_4200_5000`` NaN and the reason in
``host_info['reason']``; the window statistics are stored as
``host_info['host_window']`` in every fit. With ``host_guard=False`` the
fitter behaves as before the guard.

The blue window of a host-rich synthetic spectrum (host fraction 0.5,
continuum S/N 20, z = 0.1) is replaced by a constant flux whose weighted sum
is a chosen multiple of its error: 1e-6 sigma is flagged, 5 sigma is not and
leaves the fit identical to the guard-free one.
"""

import numpy as np
import pytest

import blrfit
from blrfit.constants import HOST_GUARD_MAX_MASKED_FRAC, HOST_GUARD_MIN_SNR
from blrfit.model.continuum import HOST_WINDOW, fit_continuum_host, host_window_statistics, pca_templates
from synth import make_spectrum

Z = 0.1


def _host_rich(seed=3):
    s = make_spectrum(
        z=Z, snr=20.0, seed=seed, host_frac=0.5, broad=[dict(line="Halpha", v=600.0, fwhm=4000.0, ew=200.0)]
    )
    wr = s["wave"] / (1 + Z)
    return s, wr, s["flux"] * (1 + Z), s["ivar"] / (1 + Z) ** 2


def _window(wr):
    return (wr > HOST_WINDOW[0]) & (wr < HOST_WINDOW[1])


def _blue_sum_at(fr, ir, win, nsig):
    """A constant flux over the window whose weighted sum is ``nsig`` times its error."""
    out = fr.copy()
    out[win] = nsig / np.sqrt(np.sum(ir[win]))
    return out


def test_window_statistics_are_the_weighted_sum_and_its_noise():
    s, wr, fr, ir = _host_rich()
    P = pca_templates()
    inhost = (wr > P["gw"].min() + 2) & (wr < P["gw"].max() - 2)
    good = np.isfinite(fr) & (ir > 0)
    w = host_window_statistics(wr, fr, ir, good, inhost)
    win = _window(wr) & inhost
    assert w["n_pix"] == win.sum() == w["n_good"] and w["n_masked"] == 0 and w["masked_frac"] == 0.0
    assert w["flux_sum"] == pytest.approx(np.sum(fr[win]))
    assert w["weighted_sum"] == pytest.approx(np.sum(ir[win] * fr[win]))
    assert w["weighted_sigma"] == pytest.approx(np.sqrt(np.sum(ir[win])))
    assert w["snr"] == pytest.approx(w["weighted_sum"] / w["weighted_sigma"]) and w["snr"] > 100
    assert (
        w["min_snr"] == HOST_GUARD_MIN_SNR == 3.0
        and w["max_masked_frac"] == HOST_GUARD_MAX_MASKED_FRAC == 0.5
    )
    assert not w["undetermined"] and w["reasons"] == []


def test_blue_sum_at_a_millionth_of_its_error_is_flagged_and_not_subtracted():
    s, wr, fr, ir = _host_rich()
    win = _window(wr)
    fr = _blue_sum_at(fr, ir, win, 1e-6)
    assert np.sum(fr[win]) > 0  # the plain sum is positive: the 0.2.0 rule alone would not catch it
    d, total, host, info = fit_continuum_host(wr, fr, ir)
    assert info["host_undetermined"] is True and info["host_guard"] is True
    assert not info["applied"] and info["n_gal"] == 0 and np.isnan(info["host_frac_4200_5000"])
    assert not np.any(host) and np.all(np.isfinite(total)) and info["solver_attempts"] == []
    assert info["reason"].startswith("host undetermined: ") and info["reason"].endswith("; PL+Fe only")
    w = info["host_window"]
    assert w["snr"] == pytest.approx(1e-6, rel=1e-6) and w["weighted_sum"] > 0
    assert w["reasons"] == [f"flux sum over 4200-5000 A at {w['snr']:.2f} sigma (< 3)"]
    # the guard off: the old path, no flag
    d0, total0, host0, info0 = fit_continuum_host(wr, fr, ir, host_guard=False)
    assert info0["host_undetermined"] is False and info0["host_guard"] is False
    assert info0["host_window"] == w and info0["reason"] != info["reason"]


def test_blue_sum_at_five_sigma_is_unchanged():
    s, wr, fr, ir = _host_rich()
    win = _window(wr)
    fr = _blue_sum_at(fr, ir, win, 5.0)
    on = fit_continuum_host(wr, fr, ir, host_guard=True)
    off = fit_continuum_host(wr, fr, ir, host_guard=False)
    assert on[3]["host_window"]["snr"] == pytest.approx(5.0, rel=1e-6)
    assert not on[3]["host_undetermined"] and not on[3]["host_window"]["undetermined"]
    assert on[0] == off[0]
    assert np.array_equal(on[1], off[1]) and np.array_equal(on[2], off[2])
    for k in ("applied", "n_gal", "reason", "n_negative_pix"):
        assert on[3][k] == off[3][k]
    assert np.isnan(on[3]["host_frac_4200_5000"]) == np.isnan(off[3]["host_frac_4200_5000"])


def test_blue_sum_just_below_and_above_the_threshold():
    s, wr, fr, ir = _host_rich()
    win = _window(wr)
    below = fit_continuum_host(wr, _blue_sum_at(fr, ir, win, 2.9), ir)[3]
    above = fit_continuum_host(wr, _blue_sum_at(fr, ir, win, 3.1), ir)[3]
    assert below["host_undetermined"] and not above["host_undetermined"]


def test_negative_blue_sum_is_flagged_as_not_positive():
    s, wr, fr, ir = _host_rich()
    win = _window(wr)
    fr = _blue_sum_at(fr, ir, win, -20.0)
    d, total, host, info = fit_continuum_host(wr, fr, ir)
    assert info["host_undetermined"] and not info["applied"] and not np.any(host)
    assert info["host_window"]["reasons"] == ["flux sum over 4200-5000 A not positive"]
    # without the guard the fraction is undefined and the host is still not subtracted (the 0.2.0 rule)
    d0, total0, host0, info0 = fit_continuum_host(wr, fr, ir, host_guard=False)
    assert not info0["host_undetermined"] and not info0["applied"] and not np.any(host0)


@pytest.mark.parametrize("masked_frac, flagged", [(0.6, True), (0.4, False)])
def test_masked_window(masked_frac, flagged):
    s, wr, fr, ir = _host_rich()
    win = _window(wr)
    idx = np.where(win)[0]
    ir = ir.copy()
    ir[idx[: int(masked_frac * idx.size)]] = 0.0
    d, total, host, info = fit_continuum_host(wr, fr, ir)
    w = info["host_window"]
    assert w["n_masked"] == int(masked_frac * idx.size) and w["masked_frac"] == pytest.approx(
        masked_frac, abs=1e-3
    )
    assert info["host_undetermined"] is flagged
    if flagged:
        assert w["reasons"] == [f"{w['n_masked']} of {w['n_pix']} pixels in 4200-5000 A masked"]
        assert not info["applied"] and np.isnan(info["host_frac_4200_5000"])
    else:
        assert w["reasons"] == [] and w["snr"] > HOST_GUARD_MIN_SNR


def test_fully_masked_window_is_flagged_for_both_reasons():
    s, wr, fr, ir = _host_rich()
    ir = ir.copy()
    ir[_window(wr)] = 0.0
    info = fit_continuum_host(wr, fr, ir)[3]
    w = info["host_window"]
    assert info["host_undetermined"] and w["n_good"] == 0 and w["masked_frac"] == 1.0 and np.isnan(w["snr"])
    assert len(w["reasons"]) == 2 and "not positive" in w["reasons"][0] and "masked" in w["reasons"][1]


def test_fit_spectrum_persists_the_flag_and_the_setting():
    s, wr, fr, ir = _host_rich()
    win = _window(wr)
    flux = _blue_sum_at(s["flux"], s["ivar"], win, 1e-6)  # the same construction in the observed frame
    res = blrfit.fit_spectrum(s["wave"], flux, s["ivar"], Z, complexes=("Halpha",))
    hi = res["host_info"]
    assert hi["host_undetermined"] is True and res["settings"]["host_guard"] is True
    assert not hi["applied"] and np.isnan(hi["host_frac_4200_5000"]) and not np.any(res["host_model"])
    assert res["continuum_info"]["host_undetermined"] is True and "host_window" in res["continuum_info"]
    row = blrfit.summary_row(res)
    assert row["host_applied"] is False and np.isnan(row["host_frac"])
    assert res["meas"]["Halpha"]["host_frac"] == 0.0  # the convention of a fit without the host
    assert res["cls"]["Halpha"]["label"] in "ABCFWEX"
    off = blrfit.fit_spectrum(s["wave"], flux, s["ivar"], Z, complexes=("Halpha",), host_guard=False)
    assert off["settings"]["host_guard"] is False and off["host_info"]["host_undetermined"] is False


def test_host_rich_spectrum_with_signal_keeps_its_host():
    s, wr, fr, ir = _host_rich()
    res = blrfit.fit_spectrum(s["wave"], s["flux"], s["ivar"], Z, complexes=("Halpha",))
    hi = res["host_info"]
    assert hi["host_undetermined"] is False and hi["applied"] and 0.3 < hi["host_frac_4200_5000"] < 0.7
    assert hi["host_window"]["snr"] > 100
