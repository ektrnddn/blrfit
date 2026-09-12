"""
Diagnostic figures: the continuum decomposition, one velocity panel per
fitted complex with the components, the bisector points and the class, and
overlays of several epochs of the same object.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .constants import C_KMS, LAM, COMPLEX_LINE
from .model.continuum import fe_templates
from .model.params import gauss_lam
from .model.lines import eval_components
from .classify import LABEL_TEXT


def plot_fit(res, title="", vwin=12000.0, figsize=(18, 9)):
    """Full diagnostic figure of a ``fit_spectrum`` result: continuum
    decomposition on top; below, per complex, the continuum-subtracted data,
    the narrow model, the broad components, the total, the residuals, the
    c(1/4), c(1/2), c(3/4) bisector points and the class. Velocities are
    relative to the systemic velocity of the same fit."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    wr, fr = res["wave_rest"], res["flux_rest"]
    names = [n for n in ("Hbeta", "Halpha", "MgII") if n in res["fits"]]
    ncol = max(len(names), 1)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, ncol, figure=fig, height_ratios=[1.0, 1.3, 0.45], hspace=0.35, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    good = res["ivar_rest"] > 0
    ax.plot(wr[good], fr[good], lw=0.5, color="0.2", label="data")
    if res["host_info"].get("applied"):
        ax.plot(wr, res["host_model"], lw=0.8, color="tab:green", label="host (Yip eigenspectra)")
    ax.plot(wr, res["conti_model"] + res["host_model"], lw=1.0, color="tab:orange",
            label="host + power law + Fe II" if res["host_info"].get("applied") else "power law + Fe II")
    fe_op, fe_uv = fe_templates(); cd = res["conti"]
    fe = np.zeros_like(wr)
    if "feop_norm" in cd: fe += fe_op(wr, cd["feop_norm"], cd["feop_fwhm"], cd["feop_shift"])
    if "feuv_norm" in cd: fe += fe_uv(wr, cd["feuv_norm"], cd["feuv_fwhm"], cd["feuv_shift"])
    if fe.max() > 0:
        ax.plot(wr, fe + res["host_model"], lw=0.7, color="tab:purple", alpha=0.8, label="Fe II")
    if good.any():
        lo, hi = np.nanpercentile(fr[good], [0.5, 99.5]); ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.set_xlim(wr.min(), wr.max()); ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_xlabel("rest wavelength (A)"); ax.set_ylabel(r"$f_\lambda$")
    ax.set_title(title or f"z = {res['z']:.4f}", fontsize=10)

    for k, name in enumerate(names):
        r = res["fits"][name]; m = res["meas"][name]; c = res["cls"].get(name, {})
        lam0 = LAM[COMPLEX_LINE[name]]
        vsys = m["v_sys"] if np.isfinite(m["v_sys"]) else 0.0
        x, y = r["x"], r["y"]
        v = (x / lam0 - 1) * C_KMS - vsys
        d = r["d"]
        tot = eval_components(x, d, r["comps"])
        nar = eval_components(x, d, r["comps"], kinds=("narrow", "wing", "nwing"))
        axp = fig.add_subplot(gs[1, k]); axr = fig.add_subplot(gs[2, k], sharex=axp)
        axp.plot(v, y, lw=0.7, color="0.15", label="data - continuum")
        axp.plot(v, nar, lw=0.8, color="tab:green", label="narrow + wing")
        cols = ["tab:blue", "tab:red", "tab:purple"]
        for j, (lab, l0, an, vn, sn, kind, ratio) in enumerate([c_ for c_ in r["comps"] if c_[5] == "broad"]):
            A = d[an] * (ratio[1] if ratio else 1.0)
            axp.plot(v, gauss_lam(x, A, l0, d[vn], d[sn]), lw=0.9, ls="--", color=cols[j % 3], alpha=0.9)
        axp.plot(v, tot, lw=1.1, color="tab:red", label="total")
        axp.axvline(0, color="k", lw=0.7)
        Pb = eval_components(x, d, r["comps"], kinds=("broad",))
        pm = Pb.max() if Pb.size else 0
        for frac, key, mk, lab_ in ((0.25, "c25_sys", "v", "c(1/4)"), (0.5, "c50_sys", "s", "c(1/2)"), (0.75, "c75_sys", "^", "c(3/4)")):
            if np.isfinite(m.get(key, np.nan)):
                axp.plot([m[key]], [frac * pm], marker=mk, color="k", ms=5, ls="none", label=lab_ if k == 0 else None)
        axp.set_xlim(-vwin, vwin)
        lab = c.get("label", "?")
        axp.set_title(f"{name}: class {lab} ({LABEL_TEXT.get(lab, '')})  n_broad={r['n_broad']}", fontsize=9.5)
        txt = (f"c50-sys {m['c50_sys']:+.0f}  peak {m['v_peak_sys']:+.0f}  cen {m['centroid_sys']:+.0f}\n"
               f"FWHM {m['fwhm']:.0f}  AI {m['AI']:+.2f}  KI {m['KI']:.2f}  npk {m['n_peaks']}\n"
               f"v_sys {m['v_sys']:+.0f} ({m.get('systemic_source', '')})  chi2r {m['chi2_red']:.2f}")
        axp.text(0.02, 0.97, txt, transform=axp.transAxes, va="top", fontsize=7.2, family="monospace")
        if k == 0: axp.legend(fontsize=7, loc="upper right")
        axr.plot(v, (y - tot) * r["w"], lw=0.6, color="0.3"); axr.axhline(0, color="k", lw=0.5)
        axr.set_ylim(-5, 5); axr.set_ylabel(r"resid/$\sigma$", fontsize=8)
        axr.set_xlabel(r"$v - v_{\rm sys}$ (km/s)", fontsize=9)
    return fig


def broad_residual_profile(res, name="Hbeta", vwin=12000.0, smooth_kms=150.0):
    """(v, data minus continuum minus narrow model, broad model) relative to
    the fit's own systemic velocity, for overlaying epochs fitted with the same code."""
    if name not in res["fits"]:
        return None
    r = res["fits"][name]; m = res["meas"][name]
    lam0 = LAM[COMPLEX_LINE[name]]
    vsys = m["v_sys"] if np.isfinite(m["v_sys"]) else 0.0
    x, y = r["x"], r["y"]; d = r["d"]
    v = (x / lam0 - 1.0) * C_KMS - vsys
    nar = eval_components(x, d, r["comps"], kinds=("narrow", "wing", "nwing"))
    bro = eval_components(x, d, r["comps"], kinds=("broad",))
    o = np.argsort(v); v, yy, bro = v[o], (y - nar)[o], bro[o]
    if smooth_kms and len(v) > 5:
        pix = float(np.median(np.diff(v)))
        yy = gaussian_filter1d(yy, max(smooth_kms / pix, 0.5))
    sel = np.abs(v) < vwin
    return v[sel], yy[sel], bro[sel]


def plot_epochs_overlay(results, labels, name="Hbeta", vwin=12000.0, normalize=True,
                        title="", ax=None):
    """Overlay several epochs of one object. ``results`` are ``fit_spectrum``
    outputs in chronological order, ``labels`` their dates. Each epoch is
    plotted relative to its own systemic velocity and normalised to its broad
    peak, so that only the profile shape and velocity matter, not the flux
    calibration or the aperture."""
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = plt.cm.plasma(np.linspace(0.05, 0.85, max(len(results), 2)))
    for k, (res, lab) in enumerate(zip(results, labels)):
        out = broad_residual_profile(res, name, vwin)
        if out is None:
            continue
        v, yy, bro = out
        m = res["meas"][name]
        norm = bro.max() if (normalize and bro.max() > 0) else 1.0
        ax.plot(v, yy / norm, lw=0.9, color=cols[k], alpha=0.85,
                label=f"{lab}  c50={m['c50_sys']:+.0f}  pk={m['v_peak_sys']:+.0f}  FWHM={m['fwhm']:.0f}")
        ax.plot(v, bro / norm, lw=1.0, color=cols[k], ls="--", alpha=0.6)
        ax.axvline(m["c50_sys"], color=cols[k], lw=0.8, ls=":")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-vwin, vwin); ax.set_xlabel(r"$v - v_{\rm sys}$ (km/s)")
    ax.set_ylabel("broad profile / peak" if normalize else r"$f_\lambda$")
    ax.legend(fontsize=7.5); ax.set_title(title, fontsize=10)
    return ax.figure if own else ax


def plot_ccf(pair, title="", ax=None):
    """The chi-square excess curve G(n) of a ``rv.ccf_shift``, ``rv.shift_bidirectional``
    or ``rv.pair_analysis(details=True)`` result, with the minimum and the 99 per cent interval."""
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.5, 4))
    s = pair["details"].get("s_ab") if isinstance(pair.get("details"), dict) else pair.get("s_ab", pair)
    if s is None or "curve" not in s:
        ax.text(0.5, 0.5, "no cross-correlation curve", ha="center", transform=ax.transAxes)
        return ax.figure if own else ax
    dv, G = s["curve"]
    ax.plot(dv, G, ".-", ms=3, lw=0.8, color="0.3")
    ax.axvline(pair["dv"], color="tab:red", lw=1.0, label=f"dv = {pair['dv']:+.0f} +/- {pair['err']:.0f} km/s")
    if np.isfinite(s.get("err_lo", np.nan)) and np.isfinite(s.get("err_hi", np.nan)):
        ax.axvspan(s["err_lo"], s["err_hi"], color="tab:red", alpha=0.12, label="99 per cent interval")
    ax.set_xlabel("shift of epoch relative to template (km/s)")
    ax.set_ylabel(r"$\chi^2 - N_{\rm pix}$")
    ax.legend(fontsize=8); ax.set_title(title, fontsize=10)
    return ax.figure if own else ax


def rv_curve(epochs, name="Hbeta", key="c50_sys", ekey="c50_sys"):
    """Velocity curve of one line from the single-epoch fits of several dated
    spectra: ``epochs`` is a list of (mjd, label, res). Returns arrays of
    rest-frame years since the first epoch, the offset ``key`` and its Monte
    Carlo error, plus a weighted linear (constant-acceleration) fit
    v = v0 + a t, or None with fewer than two dated epochs. The
    cross-correlation of ``rv.py`` is the preferred measure of a change; this
    is the decomposition-based cross-check."""
    rows = []
    for mjd, lab, res in epochs:
        if name not in res["meas"] or not np.isfinite(mjd) or mjd > 1e8:
            continue
        m = res["meas"][name]; e = res["err"].get(name, {})
        rows.append((mjd, lab, m[key], e.get(ekey, np.nan), res["z"], m["fwhm"],
                     res["cls"][name]["label"]))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r[0])
    mjd = np.array([r[0] for r in rows]); v = np.array([r[2] for r in rows])
    e = np.array([r[3] for r in rows]); z = rows[0][4]
    t = (mjd - mjd[0]) / 365.25 / (1 + z)
    w = 1.0 / np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e[np.isfinite(e)]) if np.isfinite(e).any() else 100.0) ** 2
    A = np.vstack([np.ones_like(t), t]).T
    try:
        coef, res_, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], v * np.sqrt(w), rcond=None)
        cov = np.linalg.inv((A * w[:, None]).T @ A)
        a, ea = coef[1], np.sqrt(cov[1, 1])
    except Exception:
        a, ea = np.nan, np.nan
    return dict(t_rest_yr=t, v=v, e=e, labels=[r[1] for r in rows], fwhm=[r[5] for r in rows],
                cls=[r[6] for r in rows], accel=a, e_accel=ea, z=z, mjd=mjd)


def plot_rv_curve(epochs, name="Hbeta", title="", key="c50_sys", ax=None):
    """The velocity curve of ``rv_curve`` with its constant-acceleration fit."""
    import matplotlib.pyplot as plt
    rc = rv_curve(epochs, name=name, key=key, ekey=key)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    if rc is None:
        ax.text(0.5, 0.5, "fewer than 2 dated epochs", ha="center", transform=ax.transAxes)
        return ax.figure if own else ax
    ax.errorbar(rc["t_rest_yr"], rc["v"], yerr=rc["e"], fmt="o", color="k", ms=5, capsize=3)
    for tt, vv, lab, cl in zip(rc["t_rest_yr"], rc["v"], rc["labels"], rc["cls"]):
        ax.annotate(f"{lab}  ({cl})", (tt, vv), textcoords="offset points", xytext=(4, 6), fontsize=7)
    if np.isfinite(rc["accel"]):
        tt = np.linspace(0, rc["t_rest_yr"].max(), 20)
        v0 = np.average(rc["v"] - rc["accel"] * rc["t_rest_yr"], weights=1 / np.clip(rc["e"], 1, None) ** 2)
        ax.plot(tt, v0 + rc["accel"] * tt, "r--", lw=1,
                label=f"a = {rc['accel']:+.0f} +/- {rc['e_accel']:.0f} km/s/yr (rest)")
        ax.legend(fontsize=8)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xlabel("rest-frame years since first epoch"); ax.set_ylabel(f"{name} {key} (km/s)")
    ax.set_title(title, fontsize=10)
    return ax.figure if own else ax
