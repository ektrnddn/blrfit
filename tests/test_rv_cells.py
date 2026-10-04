"""Cells of the cross-correlation that 0.1.0 measured wrongly: a flux ratio between
the epochs at unequal signal-to-noise, and masked or missing pixels at several
positions of one epoch. Both are measured on synthetic Gaussian profiles on a
30 km/s grid (DESI-like at Halpha); the data extend 3000 km/s beyond the
comparison window so that the overlap floor does not act."""
import numpy as np
import pytest

from blrfit import rv

V = np.arange(-9000.0, 9000.01, 30.0)


def gauss(v, centre=0.0, fwhm=4000.0):
    return np.exp(-0.5 * ((v - centre) / (fwhm / 2.3548)) ** 2)


def profile(v, flux, err, ok=None):
    return dict(v=v, f=flux, e=np.full(v.size, err), ok=np.ones(v.size, bool) if ok is None else ok,
                nmod=np.zeros(v.size), fwhm=4000.0, c50_sys=0.0, v_sys=0.0)


@pytest.mark.parametrize("ratio", [0.5, 2.0])
@pytest.mark.parametrize("snr_t,snr_p", [(25.0, 8.0), (8.0, 25.0)])
def test_flux_ratio_at_unequal_snr_keeps_identical_shapes_stable(ratio, snr_t, snr_p):
    """An epoch ``ratio`` times brighter or fainter than the template, the two at
    different peak S/N, same shape shifted by +300 km/s (six noise realisations):
    the profile statistic stays below the 'mild' grade in both directions (0.1.0
    gave z_prof of 5 to 23 for these cells, largest when the fainter spectrum is
    also the noisier one), the fitted flux factor is the ratio, and the shift is
    unbiased."""
    zs, dvs, scales = [], [], []
    for seed in range(6):
        rng = np.random.default_rng(100 + seed)
        et, ep = 1.0 / snr_t, ratio / snr_p
        t = profile(V, gauss(V) + rng.normal(0.0, et, V.size), et)
        p = profile(V, ratio * gauss(V, 300.0) + rng.normal(0.0, ep, V.size), ep)
        pair = rv.shift_bidirectional(p, t)
        zs += [pair["s_ab"]["profile_z"], pair["s_ba"]["profile_z"]]
        dvs.append(pair["dv"] - 300.0)
        scales.append(pair["s_ab"]["scale"])
        assert pair["scale_ok"] and not pair["ambiguous"]
    assert np.max(zs) < 5.0 and -3.0 < np.median(zs) < 2.0, zs
    assert abs(np.median(np.log(np.asarray(scales) / ratio))) < 0.05, scales
    assert abs(np.median(dvs)) < 30.0, dvs


@pytest.mark.parametrize("where", [-2500.0, 0.0, 2500.0])
@pytest.mark.parametrize("width", [1, 3, 10])
@pytest.mark.parametrize("dropped", [False, True])
def test_masked_or_missing_pixels_do_not_shift_an_identical_profile(where, width, dropped):
    """1, 3 or 10 consecutive pixels of one epoch masked (kept with ok False) or
    dropped from its arrays, in the blue wing, the core or the red wing, for two
    noiseless copies of one profile: the shift stays within 1 km/s and no
    resampling happens (0.1.0 placed a gapped epoch by index and moved it by one
    pixel, 30 km/s here, per dropped pixel blueward of the window)."""
    f = gauss(V)
    t = profile(V, f.copy(), 0.01)
    k = int(np.argmin(np.abs(V - where)))
    hole = np.zeros(V.size, bool)
    hole[k:k + width] = True
    if dropped:
        keep = ~hole
        p = profile(V[keep], f[keep].copy(), 0.01)
    else:
        p = profile(V, f.copy(), 0.01, ok=~hole)
    r = rv.ccf_shift(p, t)
    assert abs(r["dv"]) < 1.0 and not r["regridded"], (r["dv"], r["regridded"])
    if not dropped:
        assert r["n_masked_prof"] == width
