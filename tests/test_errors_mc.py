"""
Monte Carlo errors of the single-epoch fitter: pulls of the primary offset.

One configuration is injected into DESI-like spectra with independent noise
realisations, each spectrum is fitted with ``nmc`` Monte Carlo refits, and the
pull (c50_sys - truth) / err is formed per realisation. For a correctly
calibrated error the pulls have unit width; the paper states that they are
consistent with unity. The per-pixel errors used by the fitter include its 2
per cent flux-calibration floor in quadrature, which at the peak of the broad
line (2.6 times the continuum) adds 5 per cent to the 8 per cent noise of a
continuum S/N of 12, so a width somewhat below one is the expected outcome.

Configuration: broad Halpha at +800 km/s, FWHM 4000, equivalent width 150 A
(broad peak S/N 19), broad Hbeta at the same velocity with a third of the
equivalent width, narrow lines with the default equivalent width of 40 A,
continuum S/N 12 per pixel near Halpha, 20 realisations, 30 Monte Carlo refits
each (about 10 s per spectrum). Everything is slow.
"""
import numpy as np
import pytest

import blrfit
from blrfit.errors import MC_KEYS
from synth import make_spectrum

pytestmark = pytest.mark.slow

Z = 0.25
V_BROAD, FWHM = 800.0, 4000.0
SNR, NMC, NREAL = 12.0, 30, 20
BROAD = [dict(line="Halpha", v=V_BROAD, fwhm=FWHM, ew=150.0), dict(line="Hbeta", v=V_BROAD, fwhm=FWHM, ew=45.0)]


def nmad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def nmad_interval_unit_pulls(n, conf=0.99, nsim=20000, seed=0):
    """Central ``conf`` interval of the NMAD of n unit-Gaussian samples (n = 20: 0.41-1.67)."""
    x = np.random.default_rng(seed).standard_normal((nsim, n))
    s = 1.4826 * np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)
    q = 100 * (1 - conf) / 2
    return tuple(np.percentile(s, [q, 100 - q]))


@pytest.fixture(scope="module")
def realisations():
    out = []
    for i in range(NREAL):
        sp = make_spectrum(z=Z, snr=SNR, broad=BROAD, seed=i)
        res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z, complexes=("Halpha", "Hbeta"),
                                  nmc=NMC, seed=i)
        out.append((sp, res))
    return out


def test_monte_carlo_output_structure(realisations):
    """res['err'][line] carries every key of MC_KEYS and res['mc'][line] the
    (p16, p50, p84) triples they derive from, with err = (p84 - p16) / 2."""
    for sp, res in realisations:
        for name in ("Halpha", "Hbeta"):
            assert set(res["err"][name]) == set(MC_KEYS)
            assert set(res["mc"][name]) == set(MC_KEYS)
            for k in MC_KEYS:
                trip = res["mc"][name][k]
                assert isinstance(trip, tuple) and len(trip) == 3
                p16, p50, p84 = trip
                if np.isfinite(p50):
                    assert p16 <= p50 <= p84
                    assert res["err"][name][k] == pytest.approx(0.5 * (p84 - p16))
                else:
                    assert not np.isfinite(res["err"][name][k])
            e = res["err"][name]
            assert np.isfinite(e["c50_sys"]) and e["c50_sys"] > 0
            assert np.isfinite(e["fwhm"]) and e["fwhm"] > 0
            assert np.isfinite(e["v_sys"]) and e["v_sys"] > 0


def test_monte_carlo_pulls_of_c50_sys(realisations):
    """Pulls of c50_sys over 20 realisations. The NMAD must lie within the
    adopted band 0.6-1.6 (for unit-Gaussian pulls at n = 20 the 99 per
    cent interval of the NMAD is 0.41-1.67, so the band is about a 92 per cent
    acceptance region; it is applied to both lines) and the median |pull| must not exceed 1.11, the 99th
    percentile of the median |pull| of 20 unit-Gaussian samples (expected
    0.67). The recovered offsets themselves must be within the paper's 60
    km/s for Halpha in the median (measured below)."""
    pulls, dev, err = {"Halpha": [], "Hbeta": []}, {"Halpha": [], "Hbeta": []}, {"Halpha": [], "Hbeta": []}
    for sp, res in realisations:
        for name in ("Halpha", "Hbeta"):
            truth = sp["truth"]["broad"][name]["c50_sys"]
            m, e = res["meas"][name], res["err"][name]
            assert res["cls"][name]["label"] == "A", res["cls"][name]
            dev[name].append(m["c50_sys"] - truth); err[name].append(e["c50_sys"])
            pulls[name].append((m["c50_sys"] - truth) / e["c50_sys"])
    for name in ("Halpha", "Hbeta"):
        p, d, e = np.array(pulls[name]), np.array(dev[name]), np.array(err[name])
        print(f"MC pulls {name}: n {len(p)} NMAD {nmad(p):.2f} std {np.std(p):.2f} median |pull| {np.median(np.abs(p)):.2f} "
              f"max |pull| {np.max(np.abs(p)):.2f}; c50_sys - truth median {np.median(d):+.1f} NMAD {nmad(d):.1f} km/s, "
              f"err median {np.median(e):.1f} km/s")
    p = np.array(pulls["Halpha"])
    lo, hi = nmad_interval_unit_pulls(NREAL)
    assert 0.6 <= nmad(p) <= 1.6, (nmad(p), lo, hi)
    assert np.median(np.abs(p)) <= 1.11
    assert np.max(np.abs(p)) < 5.0
    # bulk shifts are recovered to better than 60 km/s in Halpha (paper); the median here
    # has a sampling error of about 1.25 NMAD / sqrt(20) = 4 km/s
    assert abs(np.median(dev["Halpha"])) < 60.0
    assert abs(np.median(dev["Hbeta"])) < 80.0
    # Hbeta (peak S/N 12): the same band; measured NMAD 1.08
    assert 0.6 <= nmad(np.array(pulls["Hbeta"])) <= 1.6, nmad(np.array(pulls["Hbeta"]))
