"""
Radial velocities between epochs of a broad line by chi-square cross-correlation.

The multi-Gaussian decomposition of a broad profile is not unique between two
noisy realisations of the same spectrum, so c(1/2) of a refitted model can jump
with no physical change. Velocity changes between epochs are therefore measured
by cross-correlating the continuum- and narrow-line-subtracted broad profiles
(Eracleous et al. 2012; Shen et al. 2013; Liu et al. 2014; Runnoe et al. 2017;
Guo et al. 2019); the single-epoch model enters only through the subtraction of
the pseudo-continuum and the narrow lines.

Specifics of the implementation:

* Whole-pixel shifts on the template's native grid. Two spectra on the same
  grid (two DESI spectra) are aligned by an integer offset with no
  interpolation; a spectrum on a different grid (SDSS against DESI) is regridded
  once, linearly, and the result is flagged ``regridded``. Sub-pixel
  interpolation of the spectrum correlates the noise between pixels and
  underestimates the errors.
* At each shift the epoch profile is compared with the template scaled by a
  free flux factor and a free linear baseline, solved analytically by weighted
  linear least squares.
* Narrow-line down-weighting: 7 per cent of the narrow-line model is added in
  quadrature to the error of each pixel instead of masking pixels near the
  narrow cores. A masked hole migrates with the shift and creates false
  chi-square minima at low signal-to-noise.
* The curve statistic is the excess G(n) = chi^2(n) - N_pix(n), which removes
  the ramp produced by the varying number of overlapping pixels.
* Outliers (> 5 sigma) are identified once at the first-pass minimum and
  excluded at every shift; clipping separately at each shift removes the
  mismatched line flux at wrong shifts and flattens the chi-square valley.
* A polynomial of degree up to six through the +/-10 pixels around the minimum
  gives the sub-pixel shift; the shifts at which it rises by Delta chi^2 =
  6.63 give the 99 per cent interval, converted to a 1 sigma equivalent,
  (hi - lo) / (2 x 2.576). A minimum within two pixels of the search boundary
  is reported as ``at_bound``.
* ``shift_bidirectional`` measures a-vs-b and b-vs-a; the two must agree in
  magnitude with opposite signs (Runnoe et al. 2017) and the difference is
  recorded as ``dir_mismatch``.
* ``narrow_zeropoint`` cross-correlates the narrow-line spectrum ([O III] 5007
  or [S II]) between the two fits as a wavelength- and flux-calibration
  control (Shen et al. 2013; Runnoe et al. 2015).
* ``profile_z`` = (chi^2_min - nu) / sqrt(2 nu) measures whether the two
  profiles differ in shape beyond a shift, scale and baseline.

Conventions: dv > 0 means the epoch's broad profile is redshifted relative to
the template. Velocities are in the frame of the input redshift; the systemic
cancels between epochs of the same object fitted at the same redshift.

Known limitations of the frozen implementation (kept as calibrated on the sky;
see the test suite for the measurements):

* the Delta chi-square error undercovers, increasingly for broader lines and
  lower signal-to-noise ratios (pull NMAD 1.3-1.6 at peak S/N 25 for FWHM
  4000-5000 km/s, 2-5 at peak S/N 8); the on-sky floors of ``systematic_floor``
  are what make the catalogue errors consistent;
* when the polynomial minimum lies several pixels from the discrete minimum
  and the curve stays within Delta chi-square of it up to the edge of the
  fitted range, the raw-curve fallback can return an inverted interval and the
  error is reported as NaN although the shift is not at bound; and when one
  direction's error is NaN, ``shift_bidirectional`` reports NaN for the pair
  in one argument order and the other direction's error in the other;
* the degree-6 polynomial through an asymmetric chi-square valley displaces
  the sub-pixel minimum by up to 6-7 km/s for FWHM 5000 km/s at shifts of
  900-1500 km/s (below 3 km/s for FWHM <= 4000).
"""
from __future__ import annotations

import numpy as np

from .constants import (C_KMS, LAM, CCF_VMAX_KMS, CCF_WIN_FWHM, CCF_WIN_MIN_KMS, CCF_NSUB_FRAC,
                        CCF_CLIP_SIGMA, CCF_DCHI2_99, CCF_SIG_FROM_99, CCF_SYS_KMS,
                        CCF_SYS_HBETA_LOWSNR_KMS, CCF_DIR_CUT_KMS, CCF_PROFILE_Z_MAX)
from .model.lines import eval_components

DCHI2_99 = CCF_DCHI2_99
SIG_FROM_99 = CCF_SIG_FROM_99


def broad_profile_data(res, name="Halpha", mask_narrow_kms=0.0):
    """Continuum- and narrow-subtracted broad profile of one fit on the fit's
    own pixel grid. Returns dict(v, f, e, ok, nmod, v_sys, fwhm, c50_sys); ``nmod``
    is the narrow (core + wing) model used to down-weight pixels near the narrow
    cores. ``mask_narrow_kms`` > 0 instead excludes the pixels within that distance of
    every narrow core (a hard mask; not used by the catalogue)."""
    r = res["fits"].get(name)
    if r is None:
        return None
    x, y, d, comps = np.asarray(r["x"], float), np.asarray(r["y"], float), r["d"], r["comps"]
    w = np.asarray(r["w"], float)
    lam0 = LAM[name]
    v = (x / lam0 - 1.0) * C_KMS
    narrow = eval_components(x, d, comps, kinds=("narrow", "nwing", "wing"))
    f = y - narrow
    # the fit stores w = sqrt(ivar) = 1 / sigma
    e = np.where(w > 0, 1.0 / np.where(w > 0, w, 1.0), np.inf)
    ok = np.isfinite(f) & np.isfinite(e) & (w > 0)
    for lab, l0, an, vn, sn, kind, rr in comps:
        if kind == "narrow":
            vline = (l0 * (1.0 + d[vn] / C_KMS) / lam0 - 1.0) * C_KMS
            ok &= np.abs(v - vline) > mask_narrow_kms
    m = res["meas"].get(name, {}) or {}
    return dict(v=v, f=f, e=e, ok=ok, nmod=np.abs(narrow),
                v_sys=float(m.get("v_sys", np.nan)), fwhm=float(m.get("fwhm", np.nan)),
                c50_sys=float(m.get("c50_sys", np.nan)))


def narrow_profile_data(res, name="Hbeta", which="OIII"):
    """Broad-subtracted narrow spectrum (data minus broad model) for the
    zero-point check: rest 4985-5035 A around [O III] 5007 (``which='OIII'``,
    Hbeta complex) or 6700-6745 A around the [S II] doublet (``which='SII'``,
    Halpha complex). Same structure as ``broad_profile_data``."""
    r = res["fits"].get(name)
    if r is None:
        return None
    x, y, d, comps = np.asarray(r["x"], float), np.asarray(r["y"], float), r["d"], r["comps"]
    w = np.asarray(r["w"], float)
    lam0 = LAM[name]
    broad = eval_components(x, d, comps, kinds=("broad",))
    f = y - broad
    e = np.where(w > 0, 1.0 / np.where(w > 0, w, 1.0), np.inf)
    lo, hi = (4985.0, 5035.0) if which == "OIII" else (6700.0, 6745.0)
    ok = np.isfinite(f) & np.isfinite(e) & (w > 0) & (x >= lo) & (x <= hi)
    v = (x / lam0 - 1.0) * C_KMS
    return dict(v=v, f=f, e=e, ok=ok, nmod=np.zeros_like(f), v_sys=np.nan, fwhm=np.nan, c50_sys=np.nan)


def _pixel_scale(v):
    """Native pixel size in km/s (DESI rest grids are uniform in v; SDSS log grids to ~1 per cent)."""
    dv = np.diff(np.asarray(v, float))
    return float(np.median(dv))


def _prof_on_template(prof, template, tol_frac=0.05):
    """Represent ``prof`` on the template's pixel grid: integer placement with no
    interpolation for the same native grid, one linear regrid (flagged) otherwise.
    Returns (f, e, ok, nmod arrays aligned to the template, regridded flag)."""
    vT = np.asarray(template["v"], float)
    vP = np.asarray(prof["v"], float)
    dT, dP = _pixel_scale(vT), _pixel_scale(vP)
    n = len(vT)
    fP = np.full(n, np.nan); eP = np.full(n, np.inf); okP = np.zeros(n, bool)
    nmP = np.zeros(n)
    pnm = np.asarray(prof.get("nmod", np.zeros(len(vP))), float)
    same = abs(dP - dT) < tol_frac * dT
    if same:
        k0f = (vP[0] - vT[0]) / dT
        k0 = int(round(k0f))
        if abs(k0f - k0) < tol_frac:
            iT0 = max(0, k0); iP0 = iT0 - k0
            m = min(n - iT0, len(vP) - iP0)
            if m > 0:
                fP[iT0:iT0 + m] = prof["f"][iP0:iP0 + m]
                eP[iT0:iT0 + m] = prof["e"][iP0:iP0 + m]
                okP[iT0:iT0 + m] = prof["ok"][iP0:iP0 + m]
                nmP[iT0:iT0 + m] = pnm[iP0:iP0 + m]
            return fP, eP, okP, nmP, False
    order = np.argsort(vP)
    vs, fs, es, oks = vP[order], prof["f"][order], prof["e"][order], prof["ok"][order]
    inside = (vT >= vs[0]) & (vT <= vs[-1])
    fP[inside] = np.interp(vT[inside], vs, fs)
    eP[inside] = np.interp(vT[inside], vs, es)
    nmP[inside] = np.interp(vT[inside], vs, pnm[order])
    idx = np.clip(np.searchsorted(vs, vT), 0, len(vs) - 1)
    okP = inside & oks[idx] & np.isfinite(fP)
    return fP, eP, okP, nmP, True


def _gls(y, x, gv, sig2, keep, baseline):
    """Weighted linear least squares of y on (scaled x + baseline)."""
    if baseline == "linear":
        X = np.vstack([x, np.ones_like(x), (gv - gv.mean()) / 1000.0]).T
    elif baseline == "const":
        X = np.vstack([x, np.ones_like(x)]).T
    else:
        X = x[:, None]
    wgt = np.where(keep, 1.0 / sig2, 0.0)
    XtW = X.T * wgt
    try:
        beta = np.linalg.solve(XtW @ X, XtW @ y)
    except np.linalg.LinAlgError:
        return None, None
    return beta, X


def ccf_shift(prof, template, vmax=CCF_VMAX_KMS, window=None, win_fwhm=CCF_WIN_FWHM,
              win_min=CCF_WIN_MIN_KMS, baseline="linear", clip=CCF_CLIP_SIGMA, n_clip=2,
              min_pix=30, mismatch=0.0, nsub_frac=CCF_NSUB_FRAC, n_mc=0, seed=0):
    """Chi-square cross-correlation shift of ``prof`` relative to ``template`` by
    whole pixels of the template's grid. dv > 0: prof is redshifted relative to
    the template.

    The window defaults to +/- win_fwhm * FWHM (at least +/- win_min) about the
    template's c(1/2). ``baseline`` is 'linear' (scale, constant and slope),
    'const' or None. ``mismatch`` > 0 adds that fraction of the template flux in
    quadrature to the errors (a model-mismatch term; 0 in the catalogue, whose
    statistical error stands on its own). ``n_mc`` > 0 replaces the Delta
    chi-square error by a parametric bootstrap (both spectra perturbed by their
    errors).

    Returns dict(dv, err, err_dchi2, err_curve, err_lo, err_hi, chi2_red,
    profile_z, npix, scale, at_bound, regridded, dv_pix, n_clipped, snr_proxy,
    curve=(dv_grid, G), window) or None.
    """
    rng = np.random.default_rng(seed)
    vT = np.asarray(template["v"], float)
    dpix = _pixel_scale(vT)
    if window is None:
        c0 = template.get("c50_sys", np.nan)
        vs = template.get("v_sys", np.nan)
        c0 = (0.0 if not np.isfinite(c0) else c0) + (vs if np.isfinite(vs) else 0.0)
        fw = template.get("fwhm", np.nan)
        half = max(win_fwhm * fw, win_min) if np.isfinite(fw) else 3000.0
        window = (c0 - half, c0 + half)
    inwin = (vT >= window[0]) & (vT <= window[1])
    fT = np.asarray(template["f"], float); eT = np.asarray(template["e"], float)
    okT = np.asarray(template["ok"], bool)
    fP, eP, okP, nmP, regridded = _prof_on_template(prof, template, tol_frac=0.05)
    nT = len(vT)
    nmT = np.asarray(template.get("nmod", np.zeros(nT)), float)
    N = int(np.ceil(vmax / dpix))
    ns_full = np.arange(-N, N + 1)

    def scan(ns, fPv, fTv, excl):
        chi2s = np.full(ns.size, np.nan); scales = np.full(ns.size, np.nan)
        npixs = np.zeros(ns.size, int)
        for k, nshift in enumerate(ns):
            i_lo = max(0, nshift); i_hi = min(nT, nT + nshift)
            if i_hi - i_lo < min_pix:
                continue
            sl_p = slice(i_lo, i_hi); sl_t = slice(i_lo - nshift, i_hi - nshift)
            good = (inwin[sl_p] & okP[sl_p] & okT[sl_t] & ~excl[sl_p]
                    & np.isfinite(fPv[sl_p]) & np.isfinite(fTv[sl_t]))
            ng = int(good.sum())
            if ng < min_pix:
                continue
            y = fPv[sl_p][good]; x = fTv[sl_t][good]; gv = vT[sl_p][good]
            sig2 = eP[sl_p][good] ** 2 + eT[sl_t][good] ** 2
            if nsub_frac > 0:
                sig2 = sig2 + (nsub_frac * nmP[sl_p][good]) ** 2 + (nsub_frac * nmT[sl_t][good]) ** 2
            if mismatch > 0:
                sig2 = sig2 + (mismatch * np.abs(x)) ** 2
            beta, X = _gls(y, x, gv, sig2, np.ones(ng, bool), baseline)
            if beta is None:
                continue
            npixs[k] = ng
            chi2s[k] = float(np.sum((y - X @ beta) ** 2 / sig2))
            scales[k] = float(beta[0])
        return chi2s, scales, npixs

    no_excl = np.zeros(nT, bool)
    chi2s, scales, npixs = scan(ns_full, fP, fT, no_excl)
    if not np.isfinite(chi2s).any():
        return None
    G0 = chi2s - npixs
    k0a = int(np.nanargmin(G0))
    # outliers identified ONCE at the first-pass minimum, by template-frame pixel
    # index, and excluded identically at every shift
    excl = np.zeros(nT, bool)
    nshift = int(ns_full[k0a])
    i_lo = max(0, nshift); i_hi = min(nT, nT + nshift)
    sl_p = slice(i_lo, i_hi); sl_t = slice(i_lo - nshift, i_hi - nshift)
    good = (inwin[sl_p] & okP[sl_p] & okT[sl_t]
            & np.isfinite(fP[sl_p]) & np.isfinite(fT[sl_t]))
    if int(good.sum()) >= min_pix:
        y = fP[sl_p][good]; x = fT[sl_t][good]; gv = vT[sl_p][good]
        sig2 = eP[sl_p][good] ** 2 + eT[sl_t][good] ** 2
        if nsub_frac > 0:
            sig2 = sig2 + (nsub_frac * nmP[sl_p][good]) ** 2 + (nsub_frac * nmT[sl_t][good]) ** 2
        if mismatch > 0:
            sig2 = sig2 + (mismatch * np.abs(x)) ** 2
        keep = np.ones(y.size, bool)
        for _ in range(n_clip):
            beta, X = _gls(y, x, gv, sig2, keep, baseline)
            if beta is None:
                break
            resid = (y - X @ beta) / np.sqrt(sig2)
            newkeep = np.abs(resid) < clip
            if newkeep.sum() < min_pix or np.array_equal(newkeep, keep):
                keep = newkeep if newkeep.sum() >= min_pix else keep
                break
            keep = newkeep
        idx_all = np.arange(i_lo, i_hi)[good]
        excl[idx_all[~keep]] = True
    n_clipped = int(excl.sum())
    if n_clipped:
        chi2s, scales, npixs = scan(ns_full, fP, fT, excl)
        if not np.isfinite(chi2s).any():
            return None
    Gs = chi2s - npixs
    k0 = int(np.nanargmin(Gs))
    dv_grid = ns_full * dpix
    okw = inwin & okT & np.isfinite(fT)
    snr_proxy = float(np.nanmax(np.abs(fT[okw])) / np.nanmedian(eT[okw])) if okw.any() else np.nan
    base = dict(npix=int(npixs[k0]), scale=float(scales[k0]), regridded=bool(regridded),
                dv_pix=float(dpix), n_clipped=n_clipped, snr_proxy=snr_proxy,
                curve=(dv_grid, Gs), window=tuple(window))
    nfree = {"linear": 3, "const": 2}.get(baseline, 1) + 1
    dof = max(int(npixs[k0]) - nfree, 1)
    chi2_red = float(chi2s[k0] / dof)
    profile_z = float((chi2s[k0] - dof) / np.sqrt(2.0 * dof))
    if k0 <= 1 or k0 >= ns_full.size - 2:
        return dict(dv=float(dv_grid[k0]), err=np.nan, err_dchi2=np.nan, err_lo=np.nan, err_hi=np.nan,
                    chi2_red=chi2_red, profile_z=profile_z, at_bound=True, **base)

    def sub_min(ns, Gv, kmin, half_n=10):
        sl = slice(max(kmin - half_n, 0), min(kmin + half_n + 1, ns.size))
        xs = (ns[sl] - ns[kmin]).astype(float); ys = Gv[sl]
        okp = np.isfinite(ys)
        if okp.sum() < 5:
            return float(ns[kmin]), np.nan, np.nan, np.nan
        deg = min(6, int(okp.sum()) - 2)
        try:
            coef = np.polyfit(xs[okp], ys[okp], deg)
        except Exception:
            return float(ns[kmin]), np.nan, np.nan, np.nan
        xf = np.linspace(xs[okp].min(), xs[okp].max(), 2001)
        yf = np.polyval(coef, xf)
        jm = int(np.argmin(yf))
        cmin = float(yf[jm]); best = float(ns[kmin] + xf[jm])
        thr = cmin + DCHI2_99
        j_lo, j_hi = jm, jm
        while j_lo > 0 and yf[j_lo - 1] <= thr:
            j_lo -= 1
        while j_hi < len(xf) - 1 and yf[j_hi + 1] <= thr:
            j_hi += 1
        lo, hi = float(ns[kmin] + xf[j_lo]), float(ns[kmin] + xf[j_hi])
        if j_lo == 0:
            lo = _raw_cross(ns.astype(float), Gv, kmin, thr, -1, lo)
        if j_hi == len(xf) - 1:
            hi = _raw_cross(ns.astype(float), Gv, kmin, thr, +1, hi)
        # curve-noise term: scatter of G about the polynomial, converted to a
        # shift uncertainty through the local curvature (delta_n ~ sqrt(s / a))
        resid_s = 1.4826 * np.median(np.abs(ys[okp] - np.polyval(coef, xs[okp])))
        a2 = float(np.polyval(np.polyder(coef, 2), xf[jm] - 0.0)) / 2.0
        extra = float(np.sqrt(resid_s / a2)) if (np.isfinite(resid_s) and a2 > 0 and resid_s > 0) else 0.0
        return best, lo, hi, extra

    n_best, n_lo, n_hi, n_extra = sub_min(ns_full, Gs, k0)
    dv_best = float(n_best * dpix)
    err_dchi2 = float((n_hi - n_lo) * dpix / (2.0 * SIG_FROM_99)) if np.isfinite(n_lo) and np.isfinite(n_hi) and n_hi > n_lo else np.nan
    err_curve = float(n_extra * dpix) if np.isfinite(n_extra) else 0.0

    # err_dchi2 undercovers: on the synthetic pairs of the test suite the pull
    # NMAD is 0.97 / 1.33 / 1.64 for FWHM 2500 / 4000 / 5000 km/s at peak S/N
    # 25 and 1.9 / 3.0 / 4.6 at peak S/N 8. The on-sky floor (``sigma_sys``)
    # absorbs this in the catalogue. err_curve is a diagnostic.
    err = err_dchi2
    err_lo = float(n_lo * dpix) if np.isfinite(n_lo) else np.nan
    err_hi = float(n_hi * dpix) if np.isfinite(n_hi) else np.nan
    if n_mc and n_mc > 0:
        half_ci = (n_hi - n_lo) / 2.0 if (np.isfinite(n_lo) and np.isfinite(n_hi)) else 10.0
        wmc = int(min(max(np.ceil(half_ci) + 4, 12), N))
        lo_n = max(int(ns_full[k0]) - wmc, -N); hi_n = min(int(ns_full[k0]) + wmc, N)
        ns_sub = np.arange(lo_n, hi_n + 1)
        dvs_mc = []
        for m in range(int(n_mc)):
            fPm = fP + rng.normal(0.0, 1.0, nT) * np.where(np.isfinite(eP), np.minimum(eP, 1e30), 0.0)
            fTm = fT + rng.normal(0.0, 1.0, nT) * np.where(np.isfinite(eT), np.minimum(eT, 1e30), 0.0)
            c_m, s_m, n_m = scan(ns_sub, fPm, fTm, excl)
            if not np.isfinite(c_m).any():
                continue
            G_m = c_m - n_m
            km = int(np.nanargmin(G_m))
            if km <= 0 or km >= ns_sub.size - 1:
                continue
            nb, _, _, _ = sub_min(ns_sub, G_m, km, half_n=10)
            dvs_mc.append(nb * dpix)
        if len(dvs_mc) >= max(10, 0.5 * n_mc):
            dvs_mc = np.asarray(dvs_mc)
            med = np.median(dvs_mc)
            err = float(1.4826 * np.median(np.abs(dvs_mc - med)))
            err = max(err, 0.05 * dpix)
            err_lo, err_hi = float(dv_best - SIG_FROM_99 * err), float(dv_best + SIG_FROM_99 * err)
    return dict(dv=dv_best, err=float(err), err_dchi2=err_dchi2, err_curve=err_curve,
                err_lo=err_lo, err_hi=err_hi, chi2_red=chi2_red, profile_z=profile_z,
                at_bound=False, **base)


def _raw_cross(dv_grid, chi2s, k0, thr, direction, fallback):
    """First crossing of ``thr`` on the raw curve, walking from k0."""
    k = k0
    while 0 < k < len(dv_grid) - 1:
        k += direction
        if not np.isfinite(chi2s[k]):
            break
        if chi2s[k] >= thr:
            k_prev = k - direction
            f = (thr - chi2s[k_prev]) / (chi2s[k] - chi2s[k_prev])
            return float(dv_grid[k_prev] + f * (dv_grid[k] - dv_grid[k_prev]))
    return fallback


def shift_bidirectional(prof_a, prof_b, **kw):
    """Measure a-vs-b and b-vs-a; the two shifts must agree in magnitude with
    opposite signs. Returns dict(dv, err, consistent, dir_mismatch, at_bound,
    chi2_red, npix, regridded, s_ab, s_ba) with ``dv`` the shift of a relative
    to b, or None."""
    s_ab = ccf_shift(prof_a, prof_b, **kw)
    s_ba = ccf_shift(prof_b, prof_a, **kw)
    if s_ab is None or s_ba is None:
        return None
    if s_ab.get("at_bound") or s_ba.get("at_bound"):
        return dict(dv=s_ab["dv"], err=np.nan, consistent=False, at_bound=True,
                    chi2_red=max(s_ab["chi2_red"], s_ba["chi2_red"]), s_ab=s_ab, s_ba=s_ba)
    e = float(max(s_ab["err"], s_ba["err"]))          # the two directions are correlated: no 1/sqrt(2)
    mismatch = abs(s_ab["dv"] + s_ba["dv"])
    tol = max(2.0 * np.hypot(s_ab["err"], s_ba["err"]), s_ab["dv_pix"])
    return dict(dv=float(0.5 * (s_ab["dv"] - s_ba["dv"])), err=e,
                consistent=bool(mismatch <= tol), dir_mismatch=float(mismatch),
                at_bound=False, chi2_red=float(max(s_ab["chi2_red"], s_ba["chi2_red"])),
                npix=int(min(s_ab["npix"], s_ba["npix"])), regridded=bool(s_ab["regridded"] or s_ba["regridded"]),
                s_ab=s_ab, s_ba=s_ba)


def narrow_zeropoint(res_a, res_b, prefer=("OIII", "SII"), vmax=800.0, **kw):
    """Wavelength/flux zero-point between two fits from the narrow lines.
    Returns dict(dv, err, line, at_bound, ...) of a relative to b, or None."""
    for which in prefer:
        cplx = "Hbeta" if which == "OIII" else "Halpha"
        pa = narrow_profile_data(res_a, name=cplx, which=which)
        pb = narrow_profile_data(res_b, name=cplx, which=which)
        if pa is None or pb is None or pa["ok"].sum() < 12 or pb["ok"].sum() < 12:
            continue
        lo = min(pa["v"][pa["ok"]].min(), pb["v"][pb["ok"]].min())
        hi = max(pa["v"][pa["ok"]].max(), pb["v"][pb["ok"]].max())
        s = ccf_shift(pa, pb, vmax=vmax, window=(lo, hi), baseline="const",
                      min_pix=10, **kw)
        if s is not None:
            s["line"] = which
            return s
    return None


def pair_analysis(res_a, res_b, name="Halpha", zp=True, details=False, **kw):
    """Full comparison of two fits of the same object at the same redshift:
    bidirectional broad-line shift of a relative to b plus the narrow-line
    zero-point. Returns dict(name, dv, err, consistent, at_bound, chi2_red,
    profile_z, npix, regridded, snr_proxy, dir_mismatch, zp_dv, zp_err,
    zp_line, zp_ok) or None; ``details=True`` adds the two one-directional
    ``ccf_shift`` results (including their curves) under 'details'."""
    pa = broad_profile_data(res_a, name=name)
    pb = broad_profile_data(res_b, name=name)
    if pa is None or pb is None:
        return None
    s = shift_bidirectional(pa, pb, **kw)
    if s is None:
        return None
    out = dict(name=name, dv=s["dv"], err=s["err"], consistent=s.get("consistent", False),
               at_bound=s.get("at_bound", False), chi2_red=s.get("chi2_red", np.nan),
               profile_z=max(s.get("s_ab", {}).get("profile_z", np.nan), s.get("s_ba", {}).get("profile_z", np.nan))
               if s.get("s_ab") else np.nan,
               npix=s.get("npix", 0), regridded=s.get("regridded", False),
               snr_proxy=min(s.get("s_ab", {}).get("snr_proxy", np.nan), s.get("s_ba", {}).get("snr_proxy", np.nan))
               if s.get("s_ab") else np.nan,
               dir_mismatch=s.get("dir_mismatch", np.nan))
    if details:
        out["details"] = dict(s_ab=s.get("s_ab"), s_ba=s.get("s_ba"))
    if zp:
        z = narrow_zeropoint(res_a, res_b)
        out.update(zp_dv=(z["dv"] if z else np.nan), zp_err=(z["err"] if z else np.nan),
                   zp_line=(z["line"] if z else ""),
                   zp_ok=bool(z is not None and not z.get("at_bound") and np.isfinite(z.get("err", np.nan))
                              and abs(z["dv"]) < max(30.0, 2.0 * z["err"])))
    return out


def epoch_shifts(epoch_results, template_res, name="Halpha", bidirectional=True, **kw):
    """Shift of each epoch against a template fit. ``epoch_results`` is a list
    of (label, mjd, res). Returns a list of dicts."""
    tp = broad_profile_data(template_res, name=name)
    out = []
    if tp is None:
        return out
    for lab, mjd, res in epoch_results:
        p = broad_profile_data(res, name=name)
        if p is None:
            continue
        s = shift_bidirectional(p, tp, **kw) if bidirectional else ccf_shift(p, tp, **kw)
        if s is None:
            continue
        s = dict(s)
        s.pop("s_ab", None); s.pop("s_ba", None)
        s.update(label=lab, mjd=mjd, fwhm=p["fwhm"], c50_sys=p["c50_sys"], name=name)
        out.append(s)
    return out


def systematic_floor(name, snr_proxy=np.nan):
    """The on-sky error floor of the DESI calibration for one line: 155 km/s
    (Halpha) and 79 km/s (Hbeta; 157 km/s at a signal-to-noise proxy below 8).
    Measured on consecutive DESI epochs; for other instruments it is indicative."""
    if name == "Hbeta" and np.isfinite(snr_proxy) and snr_proxy < 8:
        return CCF_SYS_HBETA_LOWSNR_KMS
    return CCF_SYS_KMS.get(name, np.nan)


def is_reliable(pair):
    """The reliable tier of the DESI calibration: not at bound, profile_z < 5 and
    direction mismatch below the per-line cut (466 km/s Halpha, 238 km/s Hbeta)."""
    if pair is None or pair.get("at_bound"):
        return False
    cut = CCF_DIR_CUT_KMS.get(pair.get("name"), np.nan)
    return (np.isfinite(pair.get("profile_z", np.nan)) and pair["profile_z"] < CCF_PROFILE_Z_MAX
            and np.isfinite(pair.get("dir_mismatch", np.nan)) and pair["dir_mismatch"] < cut)
