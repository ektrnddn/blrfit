"""
Uncertainties: Monte Carlo refits and the empirical repeat-spectrum error model.

Monte Carlo: the rest-frame spectrum is perturbed with Gaussian noise drawn
from its error array and refitted (continuum and lines), with the host model
held fixed and the number of broad components fixed to the one selected for
the unperturbed spectrum; the half-width of the 16th-84th percentile range
over the realisations is the error. The same approach is used by Shen et al.
(2013) and Liu et al. (2014). It measures the statistical error only.

Empirical model: the total error of Delta v, including the systematics of the
continuum and narrow-line decomposition, was measured on the sky from 8377
pairs of independent DESI spectra of the same objects. The per-measurement
error is max(650 km/s / sqrt(S/N), 45 km/s), with S/N the integrated broad-line
signal-to-noise ratio, inflated by 1.5 for strong offsets (|Delta v| >= 1000
km/s), whose profiles are broader and more complex. The DESI strong-offset
catalogue and its inspection pages apply the factor 1.5 to all their objects,
which are strong offsets by construction, so the two prescriptions coincide
there. The model was calibrated on broad Halpha in DESI spectra; for other
lines and instruments it is an indication, not a measurement.
"""
from __future__ import annotations

import numpy as np

from .constants import (ERR_MODEL_NORM_KMS, ERR_MODEL_FLOOR_KMS, ERR_MODEL_STRONG_FACTOR,
                        STRONG_OFFSET_KMS, MC_DEFAULT_N, MC_MIN_SAMPLES, NARROW_PRIOR_MIN_SNR)
from .model.continuum import fit_continuum
from .model.lines import fit_complex
from .measure import measure_complex

MC_KEYS = ["v_peak", "centroid", "c25", "c50", "c75", "c90", "fwhm", "W25", "W75",
           "sigma_line", "AI", "KI", "v_sys", "sig_sys", "v_o3", "broad_flux", "broad_ew",
           "v_peak_sys", "centroid_sys", "c50_sys", "c25_sys", "c75_sys", "n_peaks",
           "peak_top_sys", "centroid25_sys", "centroid50_sys"]


def empirical_error(broad_flux_snr, dv=np.nan):
    """Per-measurement error of Delta v from the DESI repeat-spectrum calibration."""
    snr = np.asarray(broad_flux_snr, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.maximum(ERR_MODEL_NORM_KMS / np.sqrt(snr), ERR_MODEL_FLOOR_KMS)
    e = np.where(np.isfinite(snr) & (snr > 0), e, np.nan)
    strong = np.isfinite(dv) & (np.abs(dv) >= STRONG_OFFSET_KMS[0])
    e = np.where(strong, ERR_MODEL_STRONG_FACTOR * e, e)
    return float(e) if np.ndim(e) == 0 else e


def monte_carlo(res, nmc=MC_DEFAULT_N, seed=0, fe=True, use_ha_systemic=True, kw=None):
    """Refit noise-perturbed spectra and return (percentiles, errors) per complex:
    ``mc[name][key] = (p16, p50, p84)`` and ``err[name][key] = (p84 - p16) / 2``."""
    rng = np.random.default_rng(seed)
    wr, fr, ir = res["wave_rest"], res["flux_rest"], res["ivar_rest"]
    sig = np.where(ir > 0, 1.0 / np.sqrt(np.where(ir > 0, ir, 1)), 0.0)
    z = res["z"]; host = res["host_model"]; kw = kw or {}
    samples = {n: {k: [] for k in MC_KEYS} for n in res["fits"]}
    for i in range(nmc):
        f_i = fr + rng.standard_normal(fr.size) * sig
        fh = f_i - host                                   # host model held fixed
        cd, cmodel, _ = fit_continuum(wr, fh, ir, fit_fe=fe)
        fsub = fh - cmodel
        prior_v = prior_s = prior_nw = None
        for name in ("Halpha", "Hbeta", "MgII"):
            if name not in res["fits"]:
                continue
            r0 = res["fits"][name]
            kws = dict(kw)
            if name == "Hbeta" and use_ha_systemic and prior_v is not None:
                kws.update(v_sys_prior=prior_v, sig_sys_prior=prior_s, nw_prior=prior_nw)
            r = fit_complex(name, wr, fsub, ir, r0["n_broad"], **kws)
            if r is None:
                continue
            m = measure_complex(r, cmodel + host, wr, z, host_model=host)
            for k in MC_KEYS:
                samples[name][k].append(m.get(k, np.nan))
            if name == "Halpha" and m["narrow_peak_snr"] >= NARROW_PRIOR_MIN_SNR and np.isfinite(m["v_sys"]):
                prior_v, prior_s = m["v_sys"], m["sig_sys"]
                if "nw_f" in r["d"]:
                    prior_nw = (float(r["d"]["nw_f"]), float(r["d"]["nw_v"]), float(r["d"]["nw_sig"]))
    mc, err = {}, {}
    for n, dd in samples.items():
        mc[n], err[n] = {}, {}
        for k, vals in dd.items():
            a = np.array(vals, float); a = a[np.isfinite(a)]
            if a.size >= MC_MIN_SAMPLES:
                p16, p50, p84 = np.percentile(a, [16, 50, 84])
                mc[n][k] = (float(p16), float(p50), float(p84))
                err[n][k] = float(0.5 * (p84 - p16))
            else:
                mc[n][k] = (np.nan, np.nan, np.nan); err[n][k] = np.nan
    return mc, err
