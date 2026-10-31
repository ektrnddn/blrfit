"""
The start of the [O III] core of the Hbeta complex (``oiii_start_velocity``).

On the DR2 Hbeta tier the core of some fits sat on a single-pixel sky residual 1,000-1,400 km/s
from the input redshift, raised by the amplitude ordering to the wing's height, while the wing
fitted the [O III] line; the systemic velocity was then that of the residual. The core started
there: the start was the highest point of the data smoothed over three pixels. The start is now a
matched filter of both [O III] lines weighted by the inverse variance, which a residual without a
partner at 4959 A does not outrank.
"""

import numpy as np

from blrfit.constants import C_KMS, LAM, R_OIII
from blrfit.model.narrow import oiii_start_velocity


def _o3_window(v_line=-150.0, sig_line=250.0, amp=1.0, spike_v=1300.0, spike=None, noise=0.3, seed=3):
    """Rest-frame pixels over the [O III] doublet with the doublet at v_line, a flat pedestal, noise
    and, optionally, a single-pixel spike near 5007 A; returns wave, flux, weights."""
    rng = np.random.default_rng(seed)
    wave = np.arange(4900.0, 5060.0, 0.45)
    y = 0.8 + rng.normal(0.0, noise, wave.size)
    for lam0, f in ((LAM["OIII5007"], 1.0), (LAM["OIII4959"], 1.0 / R_OIII)):
        lc = lam0 * (1 + v_line / C_KMS)
        y += amp * f * np.exp(-0.5 * ((wave - lc) / (lc * sig_line / C_KMS)) ** 2)
    if spike is not None:
        y[int(np.argmin(np.abs(wave - LAM["OIII5007"] * (1 + spike_v / C_KMS))))] += spike
    return wave, y, np.full(wave.size, 1.0 / noise)


def test_start_finds_the_doublet_not_a_spike():
    wave, y, w = _o3_window(spike=6.0)  # a spike 6 times the line's peak (20 sigma), at the same variance
    v = oiii_start_velocity(wave, y, 0.0, w)
    assert abs(v - (-150.0)) < 60.0
    # the smoothed-peak start of 0.6.0.dev0 went to the spike
    sel = np.abs((wave / LAM["OIII5007"] - 1) * C_KMS) < 1500.0
    ys = np.convolve(y[sel], np.ones(3) / 3.0, mode="same")
    assert abs((wave[sel][np.argmax(ys)] / LAM["OIII5007"] - 1) * C_KMS - 1300.0) < 60.0


def test_start_ignores_a_down_weighted_residual():
    wave, y, w = _o3_window(spike=40.0)
    j = int(np.argmax(y))
    w = w.copy()
    w[j] = 1e-3  # a sky residual has a large variance
    assert abs(oiii_start_velocity(wave, y, 0.0, w) - (-150.0)) < 60.0


def test_start_falls_back_without_a_line():
    wave, y, w = _o3_window(amp=0.0)
    assert oiii_start_velocity(wave, y, 123.0, w) == 123.0


def test_start_without_weights_still_matches_the_doublet():
    wave, y, _ = _o3_window(v_line=400.0, spike=None)
    assert abs(oiii_start_velocity(wave, y, 0.0) - 400.0) < 60.0



def test_tags_are_information_only():
    """The [O III] systemic tags reach the summary row and change no class, flag or strong-offset decision."""
    import blrfit
    from blrfit.model.fit import O3_SYS_KEYS, _o3_systemic_tags
    from synth import make_spectrum

    sp = make_spectrum(z=0.6, snr=15.0, seed=4, broad=[dict(line="Hbeta", v=0.0, fwhm=4000.0, ew=80.0)],
                       narrow=dict(ew_ha=60.0, o3=8.0, o3_wing=(-300.0, 500.0, 0.3)))
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], 0.6, complexes=("Hbeta",))
    row = blrfit.summary_row(res)
    for k in O3_SYS_KEYS:
        assert f"HB_{k}" in row
    assert row["HB_o3_match_snr"] > 5 and row["HB_tags"] == ""
    assert abs(row["HB_o3_wing_dv"] + 300.0) < 200.0
    # a red wing and a weak match give the tags and nothing else
    r = dict(res["fits"]["Hbeta"])
    r["d"] = dict(r["d"], w_v=r["d"]["o3_v"] + 800.0)
    t = _o3_systemic_tags(r, dict(res["meas"]["Hbeta"]))
    assert t["tags"] == "o3_wing_red"
    r["y"] = np.zeros_like(np.asarray(r["y"]))
    assert _o3_systemic_tags(r, dict(res["meas"]["Hbeta"]))["tags"] == "o3_weak,o3_wing_red"
    assert _o3_systemic_tags(r, dict(res["meas"]["Hbeta"], systemic_source="Halpha prior")) == {}
    cls = res["cls"]["Hbeta"]
    assert "o3_weak" not in cls["flags"] and "o3_wing_red" not in cls["flags"]
