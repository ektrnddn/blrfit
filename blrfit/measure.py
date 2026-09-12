"""
Non-parametric measurements of the fitted broad profile.

The summed broad model P(v) is evaluated on a 5 km/s grid and reduced to the
quantities of Marziani et al. (1996) and Eracleous et al. (2012): the peak
velocity, the flux-weighted centroid, the bisector centres c(f) (midpoint of
the two outermost crossings of the level f times the peak) and widths W(f) for
f = 1/4, 1/2, 3/4 and 0.9, with FWHM = W(1/2); the asymmetry index
A.I. = [v_R(1/4) + v_B(1/4) - 2 v_peak] / W(1/4); the kurtosis index
K.I. = W(3/4) / W(1/4) (0.456 for a Gaussian); the number of resolved peaks,
their separation and the dip between them; fluxes, equivalent widths and
signal-to-noise ratios. Every velocity is then referred to the systemic
velocity of the same fit (``*_sys``); the primary offset is c50_sys, the
displacement of the half-maximum bisector.

As a check on the parametric model, the peak of the continuum- and
narrow-line-subtracted data is measured as in Eracleous et al. (2012), by a
parabola through the smoothed profile above 80 per cent of its maximum
(``data_peak_top``), and lightly smoothed data-side measures are carried
alongside (``data_*``).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .constants import (C_KMS, LAM, COMPLEX_LINE, PROFILE_GRID_KMS, PROFILE_GRID_MAX_KMS,
                        PEAK_HEIGHT_FRAC, PEAK_PROMINENCE_FRAC, DATA_PEAK_FRAC,
                        DATA_PEAK_SMOOTH_KMS, DATA_PEAK_GUARD_HBETA_KMS, DATA_SMOOTH_KMS)
from .model.params import gauss_lam
from .model.lines import eval_components

_trapz = getattr(np, "trapezoid", None) or np.trapz

MEASURE_KEYS = ["v_peak", "centroid", "sigma_line", "skew", "fwhm", "W25", "W75", "W90",
                "c25", "c50", "c75", "c90", "AI", "KI", "n_peaks", "peak_sep", "dip_frac",
                "vB50", "vR50", "vB25", "vR25"]


def _crossings(v, P, level):
    """Outermost velocities where P crosses ``level`` (blue, red), interpolated."""
    above = P >= level
    if not above.any():
        return np.nan, np.nan
    idx = np.where(above)[0]
    i0, i1 = idx[0], idx[-1]

    def interp(i, j):                       # crossing between samples i (below) and j (above)
        if i < 0 or j >= len(v) or P[j] == P[i]:
            return v[j]
        return v[i] + (level - P[i]) * (v[j] - v[i]) / (P[j] - P[i])

    vB = interp(i0 - 1, i0) if i0 > 0 else v[0]
    vR = interp(i1 + 1, i1) if i1 < len(v) - 1 else v[-1]
    return float(vB), float(vR)


def profile_measures(v, P):
    """Shape measures of a non-negative profile P(v) on a uniform velocity grid."""
    out = {k: np.nan for k in MEASURE_KEYS}
    if P is None or not np.isfinite(P).any() or np.nanmax(P) <= 0:
        return out
    P = np.where(np.isfinite(P), P, 0.0)
    Pmax = P.max(); ipk = int(np.argmax(P))
    out["v_peak"] = float(v[ipk])
    for f, lab in ((0.5, "50"), (0.25, "25"), (0.75, "75"), (0.9, "90")):
        vB, vR = _crossings(v, P, f * Pmax)
        out[f"c{lab}"] = 0.5 * (vB + vR)
        out[f"W{lab}"] = vR - vB
        if lab in ("50", "25"):
            out[f"vB{lab}"], out[f"vR{lab}"] = vB, vR
    out["fwhm"] = out["W50"]
    norm = _trapz(P, v)
    if norm > 0:
        cen = _trapz(v * P, v) / norm
        mu2 = _trapz((v - cen) ** 2 * P, v) / norm
        mu3 = _trapz((v - cen) ** 3 * P, v) / norm
        out["centroid"] = float(cen)
        out["sigma_line"] = float(np.sqrt(max(mu2, 0)))
        out["skew"] = float(mu3 / mu2 ** 1.5) if mu2 > 0 else np.nan
    W25 = out["W25"]
    if np.isfinite(W25) and W25 > 0:
        out["AI"] = (out["vR25"] + out["vB25"] - 2 * out["v_peak"]) / W25   # Marziani A.I.(1/4)
        out["KI"] = out["W75"] / W25                                         # Gaussian: 0.456
    pk, props = find_peaks(P, height=PEAK_HEIGHT_FRAC * Pmax, prominence=PEAK_PROMINENCE_FRAC * Pmax)
    out["n_peaks"] = int(len(pk))
    if len(pk) >= 2:
        out["peak_sep"] = float(v[pk[-1]] - v[pk[0]])
        seg = P[pk[0]:pk[-1] + 1]
        out["dip_frac"] = float(1.0 - seg.min() / min(P[pk[0]], P[pk[-1]]))
    return out


def data_peak_top(r, lam0, frac=DATA_PEAK_FRAC, smooth_kms=DATA_PEAK_SMOOTH_KMS, guard_kms=None,
                  vmax=9000.0, support=None):
    """Peak velocity of the continuum- and narrow-subtracted DATA (Eracleous et
    al. 2012): mask +/- guard_kms around every narrow component, smooth, take
    the maximum inside the broad profile's support and fit a parabola to the
    points above ``frac`` of it. Velocity relative to lam0 at the input
    redshift (the systemic is subtracted by the caller).

    The narrow-line guard helps Hbeta (weak narrow Hbeta and He II residuals sit
    under the top) but hurts Halpha, where the narrow complex lies under the
    broad top and interpolating across it biases the parabola: guard 400 km/s
    for Hbeta, none for Halpha and Mg II.
    """
    x, y, d = r["x"], r["y"], r["d"]
    if guard_kms is None:
        guard_kms = DATA_PEAK_GUARD_HBETA_KMS if r.get("name") == "Hbeta" else 0.0
    v = (x / lam0 - 1.0) * C_KMS
    o = np.argsort(v); v, y = v[o], y[o]; x = x[o]
    resid = y - eval_components(x, d, r["comps"], kinds=("narrow", "wing", "nwing"))
    mask = np.zeros_like(v, bool)
    if guard_kms > 0:
        for lab, l0, an, vn, sn, kind, ratio in r["comps"]:
            if kind in ("narrow", "wing") and d[an] > 0:
                vc = ((l0 * (1 + d[vn] / C_KMS)) / lam0 - 1.0) * C_KMS
                mask |= np.abs(v - vc) < guard_kms
    if mask.sum() and (~mask).sum() > 10:
        resid = resid.copy()
        resid[mask] = np.interp(v[mask], v[~mask], resid[~mask])
    pix = float(np.median(np.diff(v))) if len(v) > 2 else 1.0
    sm = gaussian_filter1d(resid, max(smooth_kms / pix, 0.5)) if smooth_kms else resid
    lo, hi = (-vmax, vmax) if support is None else support
    sel = (v > lo) & (v < hi)
    if sel.sum() < 5 or not np.isfinite(sm[sel]).any():
        return np.nan
    vs, ss = v[sel], sm[sel]
    i = int(np.nanargmax(ss)); top = ss[i]
    if not np.isfinite(top) or top <= 0:
        return np.nan
    # contiguous region around the maximum above frac * top
    j0 = i
    while j0 > 0 and ss[j0 - 1] >= frac * top: j0 -= 1
    j1 = i
    while j1 < len(ss) - 1 and ss[j1 + 1] >= frac * top: j1 += 1
    if j1 - j0 >= 4:
        try:
            a, b, c = np.polyfit(vs[j0:j1 + 1], ss[j0:j1 + 1], 2)
            if a < 0:
                vp = -b / (2 * a)
                if vs[j0] <= vp <= vs[j1]:
                    return float(vp)
        except Exception:
            pass
    return float(vs[i])


def centroid_above(v, P, frac):
    """First moment of P over the pixels where P > frac * max(P)."""
    if P is None or np.nanmax(P) <= 0:
        return np.nan
    m = P > frac * np.nanmax(P)
    if m.sum() < 3:
        return np.nan
    return float(_trapz(v[m] * P[m], v[m]) / _trapz(P[m], v[m]))


def broad_profile(r, lam0, vgrid):
    """The summed broad model of fit ``r`` on a velocity grid about lam0."""
    lam = lam0 * (1.0 + vgrid / C_KMS)
    return eval_components(lam, r["d"], r["comps"], kinds=("broad",))


def measure_complex(r, conti_full, wave_rest, z, dl_cm=None, vgrid=None, host_model=None):
    """All measurements of one fitted complex. Velocities are km/s relative to
    the line's rest wavelength at the INPUT redshift, and also relative to the
    narrow-line systemic measured in the same fit (``*_sys``).

    ``conti_full`` is the total continuum (power law + Fe II + host) on
    ``wave_rest``; fluxes inherit the input flux units and ``broad_lum`` assumes
    them to be 1e-17 erg/s/cm^2/A (``dl_cm`` is the luminosity distance).
    """
    name = r["name"]; d = r["d"]
    lam0 = LAM[COMPLEX_LINE[name]]
    if vgrid is None:
        vgrid = np.arange(-PROFILE_GRID_MAX_KMS, PROFILE_GRID_MAX_KMS + 0.01, PROFILE_GRID_KMS)
    P = broad_profile(r, lam0, vgrid)
    m = profile_measures(vgrid, P)

    # systemic reference from the same fit
    v_sys = d.get("n_v", np.nan); sig_sys = d.get("n_sig", np.nan)
    v_o3 = d.get("o3_v", np.nan)
    m["v_sys"] = float(v_sys); m["sig_sys"] = float(sig_sys); m["v_o3"] = float(v_o3)
    m["v_sii"] = float(d.get("s2_v", np.nan)); m["sig_sii"] = float(d.get("s2_sig", np.nan))
    m["nw_f"] = float(d.get("nw_f", np.nan)); m["nw_v"] = float(d.get("nw_v", np.nan))
    m["nw_sig"] = float(d.get("nw_sig", np.nan))
    vc = r.get("v_cover", (np.nan, np.nan))
    m["v_cover_lo"] = float(vc[0]); m["v_cover_hi"] = float(vc[1])
    for k in ("v_peak", "centroid", "c25", "c50", "c75", "c90"):
        m[f"{k}_sys"] = m[k] - v_sys if np.isfinite(v_sys) else np.nan
        if name == "Hbeta":
            m[f"{k}_o3"] = m[k] - v_o3 if np.isfinite(v_o3) else np.nan
    m["z_sys"] = (1 + z) * (1 + v_sys / C_KMS) - 1 if np.isfinite(v_sys) else np.nan

    # fluxes (input flux units) and equivalent widths
    lam = lam0 * (1.0 + vgrid / C_KMS)
    fb = float(_trapz(P, lam))
    m["broad_flux"] = fb
    cont_at = float(np.interp(lam0, wave_rest, conti_full))
    m["conti_at_line"] = cont_at
    # EW against the TOTAL continuum (power law + Fe II + host): what one would
    # measure on the spectrum. In host-dominated galaxies the AGN-only continuum
    # is close to zero and an AGN-only equivalent width is meaningless.
    m["broad_ew"] = fb / cont_at if cont_at > 0 else np.nan
    if host_model is not None:
        cont_agn = float(np.interp(lam0, wave_rest, conti_full - host_model))
        m["conti_agn_at_line"] = cont_agn
        m["broad_ew_agn"] = fb / cont_agn if cont_agn > 0 else np.nan
    if dl_cm is not None and np.isfinite(dl_cm):
        m["broad_lum"] = 4 * np.pi * dl_cm**2 * fb * 1e-17
    for lab, lam0n, an, vn, sn, kind, ratio in r["comps"]:
        if kind == "broad":
            continue
        A = d[an] * (ratio[1] if ratio else 1.0)
        lc = lam0n * (1 + d[vn] / C_KMS); sl = lc * d[sn] / C_KMS
        m[f"flux_{lab}"] = float(A * sl * np.sqrt(2 * np.pi))
    # data-side peak (Eracleous et al. 2012) and centroids above fractional
    # levels (first moments insensitive to the wings)
    sup = None
    if np.isfinite(m["vB25"]) and np.isfinite(m["vR25"]):
        sup = (m["vB25"] - 0.5 * m["W25"], m["vR25"] + 0.5 * m["W25"])
    m["peak_top"] = data_peak_top(r, lam0, support=sup)
    m["peak_top_sys"] = m["peak_top"] - v_sys if np.isfinite(v_sys) else np.nan
    m["centroid25"] = centroid_above(vgrid, P, 0.25)
    m["centroid50"] = centroid_above(vgrid, P, 0.50)
    m["centroid25_sys"] = m["centroid25"] - v_sys if np.isfinite(v_sys) else np.nan
    m["centroid50_sys"] = m["centroid50"] - v_sys if np.isfinite(v_sys) else np.nan

    # signal-to-noise proxies (the Monte Carlo gives the errors)
    noise = float(np.median(1.0 / r["w"])) if len(r["w"]) else np.nan
    m["broad_peak_snr"] = float(P.max() / noise) if noise > 0 else np.nan
    # integrated S/N: flux / (sigma_pix * dlambda * sqrt(N_pix inside W25))
    x_ = r["x"]; dlam = float(np.median(np.diff(np.sort(x_)))) if len(x_) > 2 else np.nan
    if np.isfinite(m["vB25"]) and np.isfinite(m["vR25"]) and noise > 0 and np.isfinite(dlam):
        inw = (x_ > lam0 * (1 + m["vB25"] / C_KMS)) & (x_ < lam0 * (1 + m["vR25"] / C_KMS))
        m["broad_flux_snr"] = fb / (noise * dlam * np.sqrt(max(int(inw.sum()), 1)))
    else:
        m["broad_flux_snr"] = np.nan
    nar_peak = max([d[an] for lab, l0, an, vn, sn, kind, rr in r["comps"]
                    if kind == "narrow" and lab not in ("HeII4686_n",)] + [0.0])
    m["narrow_peak_snr"] = float(nar_peak / noise) if noise > 0 else np.nan
    # S/N of the component that DEFINES the systemic: the [O III] core for Hbeta
    # (when the Halpha prior is not used), the Halpha narrow group for Halpha.
    # fit_spectrum overwrites sys_snr with the Halpha narrow S/N when the prior is used.
    if name == "Hbeta" and "OIII5007c_A" in d:
        m["o3_core_snr"] = float(d["OIII5007c_A"] / noise) if noise > 0 else np.nan
        m["sys_snr"] = m["o3_core_snr"]
        # decomposition-independent [O III] velocity: maximum of the total
        # (core + wing) model of [O III] 5007
        vg = np.arange(-3000.0, 3000.01, 5.0)
        lam_g = LAM["OIII5007"] * (1.0 + vg / C_KMS)
        yo3 = np.zeros_like(vg)
        for lab, l0, an, vn, sn, kind, rr in r["comps"]:
            if lab in ("OIII5007c", "OIII5007w"):
                yo3 += gauss_lam(lam_g, d[an], l0, d[vn], d[sn])
        m["v_o3_peak"] = float(vg[int(np.argmax(yo3))]) if yo3.max() > 0 else np.nan
    else:
        m["sys_snr"] = m["narrow_peak_snr"]
    m["chi2_red"] = r["chi2"] / max(r["npix"] - r["nfree"], 1)
    m["n_broad"] = r["n_broad"]
    m["bic_all"] = r.get("all_bic", [])
    # single-Gaussian centre, for comparison with single-Gaussian pipelines
    if r["n_broad"] == 1:
        m["v_single_gauss"] = float(d[[c[3] for c in r["comps"] if c[5] == "broad"][0]])

    # data-side cross-check: continuum- and narrow-subtracted data, lightly smoothed
    x, y = r["x"], r["y"]
    resid = y - eval_components(x, d, r["comps"], kinds=("narrow", "wing", "nwing"))
    vd = (x / lam0 - 1.0) * C_KMS
    order = np.argsort(vd); vd, resid = vd[order], resid[order]
    if len(vd) > 20:
        pix = np.median(np.diff(vd))
        sm = gaussian_filter1d(resid, max(DATA_SMOOTH_KMS / pix, 0.5))
        md = profile_measures(vd, np.clip(sm, 0, None))
        m["data_v_peak"] = md["v_peak"]; m["data_c50"] = md["c50"]
        m["data_centroid_win"] = np.nan
        if np.isfinite(m["vB25"]) and np.isfinite(m["vR25"]):
            win = (vd > m["vB25"] - 0.25 * m["W25"]) & (vd < m["vR25"] + 0.25 * m["W25"])
            if win.sum() > 5 and _trapz(np.clip(sm[win], 0, None), vd[win]) > 0:
                m["data_centroid_win"] = float(_trapz(vd[win] * np.clip(sm[win], 0, None), vd[win])
                                               / _trapz(np.clip(sm[win], 0, None), vd[win]))
    return m
