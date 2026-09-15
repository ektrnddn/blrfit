"""Velocity translations of continuum/narrow-subtracted broad profiles.

The estimator follows the profile-comparison approach of Eracleous et al.
(2012), Shen et al. (2013), Liu et al. (2014), Runnoe et al. (2017), and Guo
et al. (2019). The corrected implementation profiles a positive flux factor
and baseline with noise in BOTH spectra; see ``_profile_eiv`` for its model.
It uses whole pixels of an explicitly uniform optical-velocity grid and a
local polynomial for sub-pixel refinement. Native gaps and masks are retained;
incompatible grids are explicitly interpolated with squared-weight variance.

This DEVELOPMENT estimator is not a calibrated production likelihood:
* Regridding's off-diagonal covariance is omitted in the objective (native
  noise perturbations do preserve it in the optional conditional bootstrap).
* Continuum/narrow subtraction covariance and instrumental resolution changes
  are not modeled by the diagonal 7% narrow subtraction term.
* Q-N compensates a varying overlap count heuristically; it is not a likelihood
  ratio for one fixed data set. Clipping and polynomial refinement also require
  empirical coverage tests. Delta-Q intervals/profile_z are diagnostics.
* Native noise perturbations condition on fitted subtraction and the observed
  profiles; they do not establish unconditional uncertainty coverage.
* The published 0.1.0 error floors/grade thresholds need fresh calibration.

``dv > 0`` means prof is redshifted relative to template, in the optical
coordinate v=c*(lambda/lambda0-1). A shift is an additive translation in that
coordinate; this API does not fit a relativistic wavelength dilation.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from .constants import (C_KMS, LAM, CCF_VMAX_KMS, CCF_WIN_FWHM, CCF_WIN_MIN_KMS, CCF_NSUB_FRAC,
                        CCF_CLIP_SIGMA, CCF_DCHI2_99, CCF_SIG_FROM_99, CCF_SYS_KMS,
                        CCF_SYS_HBETA_LOWSNR_KMS, CCF_DIR_CUT_KMS, CCF_PROFILE_Z_MAX,
                        PROFILE_GRADE_Z, PROFILE_GRADE_INFLATION, CCF_NULL_KMS, FRAME_VETO_KMS,
                        CCF_SCALE_RANGE, CCF_SCALE_PRODUCT_RANGE, CCF_ALT_MIN_SEP_KMS,
                        CCF_COMMON_MIN_FRAC, CCF_REFINE_HALF_KMS, CCF_REFINE_HALF_PIX)
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
    x = np.asarray(r.get("native_x", r["x"]), float)
    y = np.asarray(r.get("native_y", r["y"]), float)
    d, comps = r["d"], r["comps"]
    w = np.asarray(r.get("native_w", r["w"]), float)
    lam0 = LAM[name]
    v = (x / lam0 - 1.0) * C_KMS
    narrow = eval_components(x, d, comps, kinds=("narrow", "nwing", "wing"))
    f = y - narrow
    # the fit stores w = sqrt(ivar) = 1 / sigma
    e = np.where(w > 0, 1.0 / np.where(w > 0, w, 1.0), np.inf)
    ok = np.isfinite(f) & np.isfinite(e) & (w > 0)
    if "native_mask" in r:
        ok &= np.asarray(r["native_mask"], bool)
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
    x = np.asarray(r.get("native_x", r["x"]), float)
    y = np.asarray(r.get("native_y", r["y"]), float)
    d, comps = r["d"], r["comps"]
    w = np.asarray(r.get("native_w", r["w"]), float)
    lam0 = LAM[name]
    broad = eval_components(x, d, comps, kinds=("broad",))
    f = y - broad
    e = np.where(w > 0, 1.0 / np.where(w > 0, w, 1.0), np.inf)
    lo, hi = (4985.0, 5035.0) if which == "OIII" else (6700.0, 6745.0)
    ok = np.isfinite(f) & np.isfinite(e) & (w > 0)
    if "native_mask" in r:
        ok &= np.asarray(r["native_mask"], bool)
    ok &= (x >= lo) & (x <= hi)
    v = (x / lam0 - 1.0) * C_KMS
    return dict(v=v, f=f, e=e, ok=ok, nmod=np.zeros_like(f), v_sys=np.nan, fwhm=np.nan, c50_sys=np.nan)


def _pixel_scale(v):
    """Median spacing of finite, strictly increasing velocity coordinates."""
    v = np.asarray(v, float)
    if v.ndim != 1 or len(v) < 2 or not np.isfinite(v).all() or np.any(np.diff(v) <= 0):
        raise ValueError("Profile velocities must be finite and strictly increasing")
    return float(np.median(np.diff(v)))


def _validated_profile(prof):
    v = np.asarray(prof["v"], float)
    _pixel_scale(v)
    out = dict(prof, v=v)
    for key in ("f", "e", "ok", "nmod"):
        arr = np.asarray(prof.get(key, np.zeros(v.size)), bool if key == "ok" else float)
        if arr.shape != v.shape:
            raise ValueError("Profile arrays must have the same one-dimensional shape")
        out[key] = arr
    out["ok"] = out["ok"] & np.isfinite(out["f"]) & np.isfinite(out["e"]) & (out["e"] > 0) & np.isfinite(out["nmod"])
    return out


def _lattice_tolerance(vP, vT):
    """Coordinate equality: 1e-6 of the smaller median pixel size, km/s."""
    return min(_pixel_scale(vT), _pixel_scale(vP)) * 1e-6


def _on_lattice(vP, vT):
    """True when every coordinate of ``vP`` inside the range of ``vT`` coincides
    with a coordinate of ``vT``: the two spectra share one pixel lattice (with
    or without holes) and can be compared by integer placement. False when
    there is no overlap or any pixel falls between two template pixels."""
    vP, vT = np.asarray(vP, float), np.asarray(vT, float)
    atol = _lattice_tolerance(vP, vT)
    src_inside = (vP >= vT[0] - atol) & (vP <= vT[-1] + atol)
    if not src_inside.any():
        return False
    n = len(vT)
    tr = np.searchsorted(vT, vP[src_inside])
    ti1 = np.clip(tr, 0, n - 1); ti0 = np.clip(tr - 1, 0, n - 1)
    distances = np.minimum(np.abs(vP[src_inside] - vT[ti0]), np.abs(vP[src_inside] - vT[ti1]))
    return bool(np.all(distances <= atol))


def _prof_on_template(prof, template, tol_frac=None):
    """Place by actual coordinates or interpolate valid adjacent native pixels.

    ``tol_frac`` is retained for compatibility but no longer permits unequal
    spacings. Coordinate equality uses 1e-6 of the smaller median pixel size.
    Missing native coordinates remain missing. Interpolation requires BOTH
    valid bracketing pixels and a bracket no wider than 1.5 native spacings.
    Independent input variances propagate with squared interpolation weights;
    off-diagonal output covariance is not returned by this helper.
    """
    prof = _validated_profile(prof)
    vT, vP = np.asarray(template["v"], float), prof["v"]
    atol = _lattice_tolerance(vP, vT)
    n = len(vT)
    fP = np.full(n, np.nan); eP = np.full(n, np.inf)
    okP = np.zeros(n, bool); nmP = np.zeros(n)
    right = np.searchsorted(vP, vT)
    i1 = np.clip(right, 0, len(vP) - 1)
    i0 = np.clip(right - 1, 0, len(vP) - 1)
    nearest = np.where(np.abs(vT - vP[i0]) <= np.abs(vT - vP[i1]), i0, i1)
    exact = np.abs(vT - vP[nearest]) <= atol
    fP[exact], eP[exact] = prof["f"][nearest[exact]], prof["e"][nearest[exact]]
    okP[exact], nmP[exact] = prof["ok"][nearest[exact]], prof["nmod"][nearest[exact]]
    # Check every source coordinate in the overlap. A compressed array with
    # holes on the same lattice must be placed, never interpolated across holes.
    if _on_lattice(vP, vT):
        return fP, eP, okP, nmP, False
    dP = _pixel_scale(vP)
    interp = ~exact & (right > 0) & (right < len(vP))
    interp &= (vP[i1] - vP[i0] <= 1.5 * dP) & prof["ok"][i0] & prof["ok"][i1]
    jj = np.flatnonzero(interp)
    a = (vT[jj] - vP[i0[jj]]) / (vP[i1[jj]] - vP[i0[jj]])
    b = 1.0 - a
    fP[jj] = b * prof["f"][i0[jj]] + a * prof["f"][i1[jj]]
    eP[jj] = np.sqrt(b*b * prof["e"][i0[jj]]**2 + a*a * prof["e"][i1[jj]]**2)
    nmP[jj] = b * prof["nmod"][i0[jj]] + a * prof["nmod"][i1[jj]]
    okP[jj] = True
    return fP, eP, okP, nmP, True


def _regular_template(template):
    """Use an explicit uniform optical-velocity grid, retaining native holes.

    The convention remains v=c*(lambda/lambda0-1); a logarithmic wavelength
    array is resampled here rather than treated as a linear grid. Called by
    ``ccf_shift`` only when the epoch is off the template's lattice: two
    spectra on one lattice are compared by integer placement instead.
    This is a translation in optical velocity, not a relativistic Doppler fit.
    """
    template = _validated_profile(template)
    v = template["v"]; dp = _pixel_scale(v)
    if np.allclose(np.diff(v), dp, rtol=0.0, atol=dp * 1e-6):
        return template, False
    grid = v[0] + np.arange(int(np.floor((v[-1] - v[0]) / dp + 1e-6)) + 1) * dp
    f, e, ok, nm, regridded = _prof_on_template(template, dict(v=grid))
    return dict(template, v=grid, f=f, e=e, ok=ok, nmod=nm), regridded


def _profile_eiv(y, x, gv, var_y, var_x, keep, baseline):
    """Profile two independent noisy measurements of latent pixel fluxes.

    x_i ~ N(mu_i, var_x_i), y_i ~ N(a*mu_i+B_i*b, var_y_i).
    Profiling the unknown mu_i gives Q(a,b)=sum((y-a*x-B*b)^2 /
    (var_y+a^2*var_x)). There is NO log(var_y+a^2*var_x) term:
    the joint likelihood's original known variances are parameter independent.
    Such a term instead belongs to a different, marginalized model.

    Baseline b is solved at each positive scale a; a is optimized, including
    its effect on weights. Normalizing each input makes the minimization and
    clipping invariant to independent changes of flux/error units.
    """
    if baseline == "linear":
        B = np.column_stack((np.ones_like(x), (gv - gv.mean()) / 1000.0))
    elif baseline == "const":
        B = np.ones((len(x), 1))
    elif baseline is None:
        B = np.empty((len(x), 0))
    else:
        raise ValueError("baseline must be 'linear', 'const', or None")
    if not np.any(keep):
        return None, None, None
    ux = float(np.sqrt(np.mean(x[keep]**2 + var_x[keep])))
    uy = float(np.sqrt(np.mean(y[keep]**2 + var_y[keep])))
    if not (np.isfinite(ux) and ux > 0 and np.isfinite(uy) and uy > 0):
        return None, None, None
    xn, yn, vx, vy = x / ux, y / uy, var_x / ux**2, var_y / uy**2

    def evaluate(log_a, output=False):
        a = np.exp(log_a)
        variance = vy + a*a * vx
        weights = np.where(keep, 1.0 / variance, 0.0)
        delta = yn - a*xn
        try:
            b = np.linalg.solve((B.T * weights) @ B, (B.T * weights) @ delta) if B.shape[1] else np.empty(0)
        except np.linalg.LinAlgError:
            return (None, None, None) if output else np.inf
        residual = delta - B @ b
        q = float(np.sum(weights * residual**2))
        if output:
            beta = np.r_[a * uy / ux, b * uy]
            return beta, np.column_stack((x, B)), variance * uy**2
        return q

    # Inspect the full bounded domain and refine each resolved local minimum;
    # normalized scales at the bounds indicate an unidentified flux factor.
    grid = np.linspace(-12.0, 12.0, 17)
    values = np.array([evaluate(z) for z in grid])
    candidates = []
    for j in range(1, len(grid) - 1):
        if values[j] <= values[j-1] and values[j] <= values[j+1]:
            sol = minimize_scalar(evaluate, bounds=(grid[j-1], grid[j+1]), method="bounded",
                                  options={"xatol": 1e-8})
            if sol.success and np.isfinite(sol.fun):
                candidates.append((sol.fun, sol.x))
    if not candidates:
        return None, None, None
    value, log_a = min(candidates)
    if value >= min(values[0], values[-1]):
        return None, None, None
    return evaluate(log_a, output=True)


def ccf_shift(prof, template, vmax=CCF_VMAX_KMS, window=None, win_fwhm=CCF_WIN_FWHM,
              win_min=CCF_WIN_MIN_KMS, baseline="linear", clip=CCF_CLIP_SIGMA, n_clip=2,
              min_pix=30, mismatch=0.0, nsub_frac=CCF_NSUB_FRAC, n_mc=0, seed=0, two_stage=True,
              despike=None):
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

    Grids. When the two spectra share one pixel lattice (every epoch pixel
    inside the template's range coincides with a template pixel, to 1e-6 of a
    pixel) the comparison is by integer placement with no resampling and
    ``regridded`` is False, whatever the lattice: on a logarithmic wavelength
    lattice (SDSS) an integer shift is then a constant wavelength ratio and the
    quoted velocity is the median pixel size times the shift, which differs
    from an optical-velocity translation by the fraction |v|/c across the
    window (below 1 per cent at 3000 km/s). Only when the lattices differ is
    the template resampled onto a uniform optical-velocity grid (if it is not
    uniform already) and the epoch interpolated onto it, with ``regridded``
    True and ``resampling`` naming what was interpolated.

    Search. The curve statistic G = chi-square - N_pix compares shifts; where
    the two profiles differ, every pixel removed lowers it, so a search over
    shifts that use different pixel sets drifts toward the shifts with the
    smallest overlap. With ``two_stage`` (the default) every scan uses the
    window pixels that have data in both spectra at each of its shifts. Stage 1
    locates the minimum over +/- vmax on that common set (the range is reduced
    below vmax when the set would keep less than CCF_COMMON_MIN_FRAC of the
    window's pixels; ``search_range`` is the range searched, ``common_frac``
    the fraction of the window kept, ``npix_search`` the smallest and largest
    pixel count along the stage-1 curve); stage 2 measures the shift, its error
    and the profile statistic over shifts within CCF_REFINE_HALF_KMS of the
    stage-1 minimum, on the common set of that small range, which is nearly the
    whole window. ``two_stage=False`` searches the whole window in one stage
    (the narrow-line zero point, whose window is its whole data range).

    Outliers. With ``despike`` (default: with ``two_stage``) pixels more than
    ``clip`` errors from the median of their five neighbours are masked in both
    spectra before the search (``n_spikes``): otherwise one bad pixel can pull
    the first-pass minimum to a wrong shift, where the outlier mask then clips
    the true line core and locks the wrong shift in. The zero point, whose
    profile is itself a sharp narrow line, is not despiked. Outliers of the
    profile comparison are then identified once at the first-pass minimum of
    each stage and excluded at every shift of it; ``n_clipped`` counts both.

    Returns dict(dv, err, err_dchi2, err_curve, err_curve_noise, err_lo,
    err_hi, err_method, bracket, refined, chi2_red, profile_z, npix, scale,
    at_bound, regridded, resampling, n_masked_prof, n_masked_template, dv_pix,
    n_clipped, snr_proxy, curve=(dv_grid, G) of the measuring stage,
    curve_global of stage 1, search_range, common_frac, npix_search,
    two_stage, refine_unstable, window) or None; the error
    grades are those of ``_shift_error``.
    """
    rng = np.random.default_rng(seed)
    if not np.isfinite(vmax) or vmax <= 0 or int(n_mc) != n_mc or n_mc < 0:
        raise ValueError("vmax must be positive and n_mc a nonnegative integer")
    if nsub_frac < 0 or mismatch < 0:
        raise ValueError("subtraction and mismatch fractions must be nonnegative")
    prof = _validated_profile(prof)
    native_template = _validated_profile(template)
    ok_prof0, ok_template0 = np.asarray(prof["ok"], bool).copy(), np.asarray(native_template["ok"], bool).copy()
    if (two_stage if despike is None else despike):
        spikes_p = _spikes(prof, clip, nsub_frac)
        spikes_t = _spikes(native_template, clip, nsub_frac)
        prof = dict(prof, ok=ok_prof0 & ~spikes_p)
        native_template = dict(native_template, ok=ok_template0 & ~spikes_t)
    else:
        spikes_p = np.zeros(ok_prof0.size, bool); spikes_t = np.zeros(ok_template0.size, bool)
    if _on_lattice(prof["v"], native_template["v"]):
        # one lattice: integer placement, never a resampled template
        template, template_regridded = native_template, False
    else:
        template, template_regridded = _regular_template(native_template)
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
    resampling = ("template+epoch" if template_regridded else ("epoch" if regridded else "none"))
    regridded = regridded or template_regridded
    # native pixels of each spectrum inside the window that carry no valid flux
    in_p = (prof["v"] >= window[0]) & (prof["v"] <= window[1])
    in_t = (native_template["v"] >= window[0]) & (native_template["v"] <= window[1])
    n_masked_prof = int(np.sum(in_p & ~ok_prof0))
    n_masked_template = int(np.sum(in_t & ~ok_template0))
    n_spikes = int(np.sum(in_p & spikes_p) + np.sum(in_t & spikes_t))
    nT = len(vT)
    nmT = np.asarray(template.get("nmod", np.zeros(nT)), float)
    N = int(np.ceil(vmax / dpix))
    idx = np.arange(nT)
    validP = okP & np.isfinite(fP)
    validT = okT & np.isfinite(fT)
    n_win = int((inwin & validP).sum())

    def common_set(lo_n, hi_n):
        """Window pixels (profile positions on the template grid) with template
        data at every shift from lo_n to hi_n pixels (template index = position
        minus shift)."""
        if not validT.any():
            return np.zeros(nT, bool)
        t_first = int(np.argmax(validT))
        t_last = int(nT - 1 - np.argmax(validT[::-1]))
        return inwin & validP & (idx >= t_first + hi_n) & (idx <= t_last + lo_n)

    def arrays_at(nshift, win, excl, fPv=fP, fTv=fT):
        """(y, x, velocities, var_y, var_x, positions) of the pixels compared at one shift."""
        i_lo = max(0, nshift); i_hi = min(nT, nT + nshift)
        if i_hi - i_lo < min_pix:
            return None
        sl_p = slice(i_lo, i_hi); sl_t = slice(i_lo - nshift, i_hi - nshift)
        good = (win[sl_p] & okP[sl_p] & okT[sl_t] & ~excl[sl_p]
                & np.isfinite(fPv[sl_p]) & np.isfinite(fTv[sl_t]))
        if int(good.sum()) < min_pix:
            return None
        y = fPv[sl_p][good]; x = fTv[sl_t][good]; gv = vT[sl_p][good]
        var_y = eP[sl_p][good] ** 2 + (nsub_frac * nmP[sl_p][good]) ** 2
        var_x = eT[sl_t][good] ** 2 + (nsub_frac * nmT[sl_t][good]) ** 2
        if mismatch > 0:
            var_x = var_x + (mismatch * np.abs(x)) ** 2
        return y, x, gv, var_y, var_x, np.arange(i_lo, i_hi)[good]

    def scan(ns, win, excl, fPv=fP, fTv=fT):
        """chi-square, flux factor and pixel count at each shift of ns."""
        chi2s = np.full(ns.size, np.nan); scales = np.full(ns.size, np.nan)
        npixs = np.zeros(ns.size, int)
        for k, nshift in enumerate(ns):
            a = arrays_at(int(nshift), win, excl, fPv, fTv)
            if a is None:
                continue
            y, x, gv, var_y, var_x, _ = a
            beta, X, sig2 = _profile_eiv(y, x, gv, var_y, var_x, np.ones(y.size, bool), baseline)
            if beta is None:
                continue
            npixs[k] = y.size
            chi2s[k] = float(np.sum((y - X @ beta) ** 2 / sig2))
            scales[k] = float(beta[0])
        return chi2s, scales, npixs

    def outlier_mask(nshift, win):
        """Outliers beyond ``clip`` sigma at one shift, by template-grid
        position; they are excluded identically at every shift of the scan."""
        excl = np.zeros(nT, bool)
        a = arrays_at(nshift, win, excl)
        if a is None:
            return excl
        y, x, gv, var_y, var_x, pos = a
        keep = np.ones(y.size, bool)
        for _ in range(n_clip):
            beta, X, sig2 = _profile_eiv(y, x, gv, var_y, var_x, keep, baseline)
            if beta is None:
                break
            resid = (y - X @ beta) / np.sqrt(sig2)
            newkeep = np.abs(resid) < clip
            if newkeep.sum() < min_pix or np.array_equal(newkeep, keep):
                keep = newkeep if newkeep.sum() >= min_pix else keep
                break
            keep = newkeep
        excl[pos[~keep]] = True
        return excl

    def search(ns, win):
        """First pass, one outlier mask at its minimum, second pass."""
        chi2s, scales, npixs = scan(ns, win, np.zeros(nT, bool))
        if not np.isfinite(chi2s).any():
            return None
        excl = outlier_mask(int(ns[int(np.nanargmin(chi2s - npixs))]), win)
        if excl.any():
            chi2s, scales, npixs = scan(ns, win, excl)
            if not np.isfinite(chi2s).any():
                return None
        return chi2s, scales, npixs, excl

    # stage 1: locate the minimum on the pixels common to every shift searched,
    # reducing the range below vmax when that set would be too small
    N1, win1, use_two = N, inwin, bool(two_stage)
    if use_two:
        need = max(min_pix, CCF_COMMON_MIN_FRAC * n_win)
        while N1 > 0 and int(common_set(-N1, N1).sum()) < need:
            N1 -= 1
        win1 = common_set(-N1, N1)
        if int(win1.sum()) < min_pix:
            N1, win1, use_two = N, inwin, False      # no data beyond the window: one stage
    ns1 = np.arange(-N1, N1 + 1)
    r1 = search(ns1, win1)
    if r1 is None:
        return None
    chi2s1, scales1, npixs1, excl1 = r1
    G1 = chi2s1 - npixs1
    k1 = int(np.nanargmin(G1))
    fin1 = np.flatnonzero(np.isfinite(G1))
    dv_grid1 = ns1 * dpix
    at_bound = bool(k1 <= fin1[0] + 1 or k1 >= fin1[-1] - 1)
    common_frac = float(win1.sum() / n_win) if (use_two and n_win) else 1.0

    # stage 2: measure over shifts near the stage-1 minimum on the common set of
    # that small range (nearly the whole window), recentring if the minimum
    # moves to an inner edge
    ns, chi2s, scales, npixs, excl, Gs, k0, win_m = ns1, chi2s1, scales1, npixs1, excl1, G1, k1, win1
    unstable = False
    if use_two and not at_bound:
        M = int(max(CCF_REFINE_HALF_PIX, np.ceil(CCF_REFINE_HALF_KMS / dpix)))
        centre = int(ns1[k1])
        for _ in range(3):
            lo_n, hi_n = max(centre - M, -N), min(centre + M, N)
            win2 = common_set(lo_n, hi_n)
            ns2 = np.arange(lo_n, hi_n + 1)
            r2 = search(ns2, win2) if int(win2.sum()) >= min_pix else None
            if r2 is None:
                break
            c2, s2, n2, e2 = r2
            G2 = c2 - n2
            k2 = int(np.nanargmin(G2))
            ns, chi2s, scales, npixs, excl, Gs, k0, win_m = ns2, c2, s2, n2, e2, G2, k2, win2
            fin2 = np.flatnonzero(np.isfinite(G2))
            inner_lo = k2 <= fin2[0] + 1 and ns2[fin2[0]] > -N + 1
            inner_hi = k2 >= fin2[-1] - 1 and ns2[fin2[-1]] < N - 1
            if not (inner_lo or inner_hi):
                break
            centre = int(ns2[k2])
        else:
            unstable = True           # three recentrings without an interior minimum
        fin = np.flatnonzero(np.isfinite(Gs))
        at_bound = bool(unstable or k0 <= fin[0] + 1 or k0 >= fin[-1] - 1)

    dv_grid = ns * dpix
    okw = inwin & okT & np.isfinite(fT)
    snr_proxy = float(np.nanmax(np.abs(fT[okw])) / np.nanmedian(eT[okw])) if okw.any() else np.nan
    # size of the profile change at the best shift: rms of the residual after the
    # scale and baseline, as a fraction of the template peak (an effect size, unlike
    # profile_z, which is a significance and grows with signal-to-noise)
    resid_frac = np.nan
    a = arrays_at(int(ns[k0]), win_m, excl)
    if a is not None and okw.any():
        y, x, gv, var_y, var_x, _ = a
        beta, X, sig2 = _profile_eiv(y, x, gv, var_y, var_x, np.ones(y.size, bool), baseline)
        if beta is not None and beta[0] != 0:
            peak = float(np.nanmax(np.abs(fT[okw])))
            resid_frac = float(np.sqrt(np.mean((y - X @ beta) ** 2)) / (abs(beta[0]) * peak))
    finite1 = npixs1[np.isfinite(G1)]
    base = dict(npix=int(npixs[k0]), scale=float(scales[k0]), regridded=bool(regridded),
                resampling=resampling, n_masked_prof=n_masked_prof, n_masked_template=n_masked_template,
                dv_pix=float(dpix), n_clipped=int(excl.sum()) + n_spikes, n_spikes=n_spikes,
                snr_proxy=snr_proxy, resid_frac=resid_frac,
                curve=(dv_grid, Gs), curve_global=(dv_grid1, G1), window=tuple(window),
                search_range=(float(dv_grid1[fin1[0]]), float(dv_grid1[fin1[-1]])),
                common_frac=common_frac, two_stage=use_two, refine_unstable=unstable,
                npix_search=(int(finite1.min()), int(finite1.max())),
                algorithm_version="profile-eiv-v2", covariance_mode="diagonal",
                interpolation_covariance_ignored=bool(regridded), uncertainty_calibrated=False,
                velocity_convention="optical_translation", err_method="delta_chi2", bracket="none", refined=False,
                n_mc_requested=int(n_mc), n_mc_success=0, bootstrap_fallback_reason=None)
    # a second minimum of the stage-1 curve, where every shift uses one pixel set
    dv_alt, dG_alt = _second_minimum(dv_grid1, G1, k1, max(CCF_ALT_MIN_SEP_KMS, 5.0 * dpix))
    base.update(dv_alt=dv_alt, dG_alt=dG_alt, ambiguous=bool(np.isfinite(dG_alt) and dG_alt < DCHI2_99))
    nfree = {"linear": 3, "const": 2}.get(baseline, 1) + 1
    dof = max(int(npixs[k0]) - nfree, 1)
    chi2_red = float(chi2s[k0] / dof)
    profile_z = float((chi2s[k0] - dof) / np.sqrt(2.0 * dof))
    if at_bound:
        base["err_method"] = "unavailable"
        if n_mc:
            base["bootstrap_fallback_reason"] = "minimum_at_search_boundary"
        return dict(dv=float(dv_grid[k0]), err=np.nan, err_dchi2=np.nan, err_curve=np.nan, err_curve_noise=np.nan,
                    err_lo=np.nan, err_hi=np.nan, chi2_red=chi2_red, profile_z=profile_z, at_bound=True, **base)

    ref = _shift_error(ns, Gs, k0, dpix)
    dv_best = ref["dv"]
    err, err_lo, err_hi = ref["err"], ref["err_lo"], ref["err_hi"]
    base["err_method"], base["bracket"], base["refined"] = ref["err_method"], ref["bracket"], ref["refined"]
    if np.isfinite(ref["err"]) and ref["err"] > 0:
        # a second minimum must lie outside the statistical interval of the first
        dv_alt, dG_alt = _second_minimum(dv_grid1, G1, k1, max(CCF_ALT_MIN_SEP_KMS, 3.0 * ref["err"], 5.0 * dpix))
        base.update(dv_alt=dv_alt, dG_alt=dG_alt, ambiguous=bool(np.isfinite(dG_alt) and dG_alt < DCHI2_99))
    n_lo, n_hi = err_lo / dpix, err_hi / dpix
    if n_mc and n_mc > 0:
        half_ci = (n_hi - n_lo) / 2.0 if (np.isfinite(n_lo) and np.isfinite(n_hi)) else 10.0
        wmc = int(min(max(np.ceil(half_ci) + 4, 12), N))
        c_n = int(ns[k0])
        lo_n = max(c_n - wmc, -N); hi_n = min(c_n + wmc, N)
        ns_sub = np.arange(lo_n, hi_n + 1)
        win_sub = common_set(lo_n, hi_n) if use_two else win_m
        dvs_mc = []
        for m in range(int(n_mc)):
            # Perturb native independent pixels BEFORE interpolation, retaining
            # the interpolation-induced covariance in these conditional draws.
            fp_native = prof["f"] + rng.normal(size=len(prof["v"])) * np.where(prof["ok"], prof["e"], 0.0)
            ft_native = native_template["f"] + rng.normal(size=len(native_template["v"])) * np.where(native_template["ok"], native_template["e"], 0.0)
            fPm = _prof_on_template(dict(prof, f=fp_native), template)[0]
            fTm = _prof_on_template(dict(native_template, f=ft_native), template)[0]
            c_m, s_m, n_m = scan(ns_sub, win_sub, excl, fPm, fTm)
            if not np.isfinite(c_m).any():
                continue
            G_m = c_m - n_m
            km = int(np.nanargmin(G_m))
            if km <= 0 or km >= ns_sub.size - 1:
                continue
            dvs_mc.append(_refine_minimum(ns_sub, G_m, km)["best"] * dpix)
        base["n_mc_success"] = len(dvs_mc)
        if len(dvs_mc) >= max(10, 0.5 * n_mc):
            base["err_method"] = "bootstrap"
            dvs_mc = np.asarray(dvs_mc)
            med = np.median(dvs_mc)
            err = float(1.4826 * np.median(np.abs(dvs_mc - med)))
            err = max(err, 0.05 * dpix)
            err_lo, err_hi = float(dv_best - SIG_FROM_99 * err), float(dv_best + SIG_FROM_99 * err)
        else:
            base["bootstrap_fallback_reason"] = "insufficient_successful_draws"
    if not (np.isfinite(err) and err > 0):
        base["err_method"] = "unavailable"
    return dict(dv=dv_best, err=float(err), err_dchi2=ref["err_dchi2"], err_curve=ref["err_curve"],
                err_curve_noise=ref["err_curve_noise"], err_lo=err_lo, err_hi=err_hi,
                chi2_red=chi2_red, profile_z=profile_z, at_bound=False, **base)

def _raw_cross(dv_grid, chi2s, k_start, thr, direction):
    """First crossing of ``thr`` on the raw curve, walking outward from
    ``k_start`` (that point itself counts when it is already at or above the
    threshold); NaN when the curve never reaches the threshold on that side,
    or stops being finite first, so that an unbracketed side is never
    mistaken for one."""
    k = k_start
    if np.isfinite(chi2s[k]) and chi2s[k] >= thr:
        return float(dv_grid[k])
    while 0 < k < len(dv_grid) - 1:
        k += direction
        if not np.isfinite(chi2s[k]):
            break
        if chi2s[k] >= thr:
            k_prev = k - direction
            f = (thr - chi2s[k_prev]) / (chi2s[k] - chi2s[k_prev])
            return float(dv_grid[k_prev] + f * (dv_grid[k] - dv_grid[k_prev]))
    return np.nan


def _refine_minimum(ns, Gv, kmin, half_n=10):
    """Sub-pixel minimum of the chi-square excess curve G(n) from a polynomial
    (degree up to 6) through the +/- ``half_n`` pixels about the grid minimum
    ``kmin``. Returns dict(best, lo, hi, curvature, noise, refined), all in
    pixels: ``lo``/``hi`` are the crossings of G = G_min + DCHI2_99 on either
    side (from the polynomial, else from the raw curve walked outward from
    the edge of the polynomial's range; NaN where the curve never rises that
    far or the crossing lies on the wrong side of the minimum); ``curvature``
    is the half-width at G_min + 1 of the parabola with the polynomial's
    second derivative at the minimum (the one-sigma error of a quadratic
    minimum; NaN when the curvature is not positive); ``noise`` is the scatter
    of G about the polynomial converted to a shift through that curvature
    (delta_n ~ sqrt(s / a), a diagnostic of a rough curve).

    ``refined`` is False, with the grid minimum as ``best`` and no interval or
    curvature, when the polynomial has no interior minimum (its minimum sits
    at the edge of the fitted range): the curve is not locally parabolic (a
    jagged low signal-to-noise surface with spurious minima, or a minimum
    running into the search bound) and the sub-pixel position, the interval
    and the curvature would describe the polynomial's edge, not the data."""
    out = dict(best=float(ns[kmin]), lo=np.nan, hi=np.nan, curvature=np.nan, noise=np.nan, refined=False)
    sl = slice(max(kmin - half_n, 0), min(kmin + half_n + 1, ns.size))
    xs = (ns[sl] - ns[kmin]).astype(float); ys = np.asarray(Gv[sl], float)
    okp = np.isfinite(ys)
    if okp.sum() < 5:
        return out
    deg = min(6, int(okp.sum()) - 2)
    try:
        coef = np.polyfit(xs[okp], ys[okp], deg)
    except Exception:
        return out
    xf = np.linspace(xs[okp].min(), xs[okp].max(), 2001)
    yf = np.polyval(coef, xf)
    jm = int(np.argmin(yf))
    if jm == 0 or jm == len(xf) - 1:
        return out
    cmin = float(yf[jm]); best = float(ns[kmin] + xf[jm])
    thr = cmin + DCHI2_99
    j_lo, j_hi = jm, jm
    while j_lo > 0 and yf[j_lo - 1] <= thr:
        j_lo -= 1
    while j_hi < len(xf) - 1 and yf[j_hi + 1] <= thr:
        j_hi += 1
    lo, hi = float(ns[kmin] + xf[j_lo]), float(ns[kmin] + xf[j_hi])
    # beyond the polynomial's range the raw curve decides, walked outward
    # from the range edge: a crossing inside the range would contradict the
    # polynomial, which found none there
    if j_lo == 0:
        lo = _raw_cross(ns.astype(float), Gv, int(kmin + round(xs[okp].min())), thr, -1)
    if j_hi == len(xf) - 1:
        hi = _raw_cross(ns.astype(float), Gv, int(kmin + round(xs[okp].max())), thr, +1)
    # a crossing on the wrong side of the polynomial minimum (the raw curve
    # crosses next to the grid minimum while the polynomial minimum sits
    # farther out) does not bracket it
    if not (np.isfinite(lo) and lo < best):
        lo = np.nan
    if not (np.isfinite(hi) and hi > best):
        hi = np.nan
    a2 = float(np.polyval(np.polyder(coef, 2), xf[jm])) / 2.0
    curvature = float(1.0 / np.sqrt(a2)) if (np.isfinite(a2) and a2 > 0) else np.nan
    resid_s = 1.4826 * np.median(np.abs(ys[okp] - np.polyval(coef, xs[okp])))
    noise = float(np.sqrt(resid_s / a2)) if (np.isfinite(resid_s) and a2 > 0 and resid_s > 0) else 0.0
    return dict(best=best, lo=lo, hi=hi, curvature=curvature, noise=noise, refined=True)


def _second_minimum(dv_grid, G, k0, min_sep_kms):
    """The deepest local minimum of the curve G at least ``min_sep_kms`` from the
    global minimum at index k0. Returns (dv_alt, dG_alt) with dG_alt = G(alt) -
    G(k0), or (nan, nan) when there is none. A second minimum within Delta
    chi-square 6.63 of the first makes the shift ambiguous: one peak of a
    double-peaked or flat-topped profile can be matched onto the other with a
    changed flux factor."""
    G = np.asarray(G, float); dv_grid = np.asarray(dv_grid, float)
    n = G.size
    best = (np.nan, np.nan)
    for k in range(1, n - 1):
        if k == k0 or not np.isfinite(G[k]) or abs(dv_grid[k] - dv_grid[k0]) < min_sep_kms:
            continue
        near = G[max(0, k - 2):min(n, k + 3)]
        if np.all(np.isfinite(near)) and G[k] <= np.min(near):
            d = float(G[k] - G[k0])
            if not np.isfinite(best[1]) or d < best[1]:
                best = (float(dv_grid[k]), d)
    return best


def _spikes(prof, nsigma, nsub_frac=0.0, half=2):
    """Pixels whose flux deviates from the median of the 2*half+1 valid pixels
    around them (itself included) by more than nsigma times their error, the
    error including the narrow-subtraction term of the scan. A single-pixel
    artefact (a cosmic ray, a sky residual, a bad column) deviates from its
    own neighbours whatever the shift."""
    ok = np.asarray(prof["ok"], bool)
    f = np.where(ok, np.asarray(prof["f"], float), np.nan)
    n = f.size
    if n < 2 * half + 1:
        return np.zeros(n, bool)
    nm = np.asarray(prof.get("nmod", np.zeros(n)), float)
    sig = np.sqrt(np.asarray(prof["e"], float) ** 2 + (nsub_frac * nm) ** 2)
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(f, half, mode="edge"), 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(windows, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.abs(f - med) / np.where(sig > 0, sig, np.inf)
    return ok & np.isfinite(dev) & (dev > nsigma)


def _shift_error(ns, Gv, kmin, dpix, half_n=10):
    """Graded statistical error of the shift, each grade named in ``err_method``:

    * 'delta_chi2': the Delta chi-square = 6.63 (99 per cent) interval brackets
      the minimum on both sides; err is its half-width divided by 2.576.
    * 'delta_chi2_one_sided': one side is bracketed; err is that side's
      distance to the minimum divided by 2.576 (``bracket`` names the side).
    * 'curvature': neither side is bracketed; err is the one-sigma half-width
      of the local parabola, when its curvature is finite and positive.
    * 'unavailable': none of the above, including every curve whose polynomial
      has no interior minimum (``refined`` False); err is NaN.

    Returns dict(dv, err, err_dchi2, err_curve, err_curve_noise, err_lo,
    err_hi, err_method, bracket, refined) in km/s (``dv`` from the refined
    minimum, or the grid minimum when not refined; ``err_lo``/``err_hi`` are
    the bracket edges, NaN on an unbracketed side)."""
    ref = _refine_minimum(ns, Gv, kmin, half_n=half_n)
    best, lo, hi = ref["best"], ref["lo"], ref["hi"]
    lo_ok, hi_ok = bool(np.isfinite(lo) and lo < best), bool(np.isfinite(hi) and hi > best)
    err_curve = float(ref["curvature"] * dpix) if np.isfinite(ref["curvature"]) else np.nan
    err_curve_noise = float(ref["noise"] * dpix) if np.isfinite(ref["noise"]) else np.nan
    err_dchi2 = float((hi - lo) * dpix / (2.0 * SIG_FROM_99)) if (lo_ok and hi_ok) else np.nan
    if lo_ok and hi_ok:
        err, method, bracket = err_dchi2, "delta_chi2", "both"
    elif lo_ok or hi_ok:
        half = (best - lo) if lo_ok else (hi - best)
        err, method, bracket = float(half * dpix / SIG_FROM_99), "delta_chi2_one_sided", ("lo" if lo_ok else "hi")
    elif np.isfinite(err_curve) and err_curve > 0:
        err, method, bracket = err_curve, "curvature", "none"
    else:
        err, method, bracket = np.nan, "unavailable", "none"
    return dict(dv=float(best * dpix), err=float(err), err_dchi2=err_dchi2, err_curve=err_curve,
                err_curve_noise=err_curve_noise, err_lo=float(lo * dpix) if lo_ok else np.nan,
                err_hi=float(hi * dpix) if hi_ok else np.nan, err_method=method, bracket=bracket,
                refined=bool(ref["refined"]))


def _finite_extreme(values, which=max):
    """Never hide a failed directional diagnostic by dropping its NaN."""
    return float(which(values)) if all(np.isfinite(v) for v in values) else np.nan


def _scales_ok(scale_ab, scale_ba):
    """True when both fitted flux factors lie in CCF_SCALE_RANGE and their
    product in CCF_SCALE_PRODUCT_RANGE: the two directions of an honest match
    fit reciprocal factors of a plausible size."""
    lo, hi = CCF_SCALE_RANGE
    plo, phi = CCF_SCALE_PRODUCT_RANGE
    if not (np.isfinite(scale_ab) and np.isfinite(scale_ba)):
        return False
    return bool(lo <= scale_ab <= hi and lo <= scale_ba <= hi and plo <= scale_ab * scale_ba <= phi)


def shift_bidirectional(prof_a, prof_b, **kw):
    """Measure both directions; their errors are correlated, so use the larger.

    Both shifts and both positive errors must be finite. A failed direction
    gives a NaN pair error regardless of argument order. Historical profile
    grades/floors have not been calibrated for the corrected objective.
    """
    s_ab = ccf_shift(prof_a, prof_b, **kw)
    s_ba = ccf_shift(prof_b, prof_a, **kw)
    if s_ab is None or s_ba is None:
        return None
    shifts = (s_ab["dv"], s_ba["dv"])
    errors = (s_ab["err"], s_ba["err"])
    bounded = bool(s_ab.get("at_bound") or s_ba.get("at_bound"))
    valid = bool(not bounded and all(np.isfinite(v) for v in shifts)
                 and all(np.isfinite(e) and e > 0 for e in errors))
    dv = float(0.5 * (shifts[0] - shifts[1])) if all(np.isfinite(v) for v in shifts) else np.nan
    mismatch = float(abs(shifts[0] + shifts[1])) if all(np.isfinite(v) for v in shifts) else np.nan
    tolerance = max(2.0 * np.hypot(*errors), s_ab["dv_pix"], s_ba["dv_pix"]) if valid else np.nan
    methods = (s_ab.get("err_method", "unavailable"), s_ba.get("err_method", "unavailable"))
    reasons = sorted(set(s.get("bootstrap_fallback_reason") for s in (s_ab, s_ba) if s.get("bootstrap_fallback_reason")))
    return dict(dv=dv, err=float(max(errors)) if valid else np.nan,
                consistent=bool(valid and mismatch <= tolerance), dir_mismatch=mismatch,
                at_bound=bounded, statistically_valid=valid,
                chi2_red=_finite_extreme((s_ab["chi2_red"], s_ba["chi2_red"])),
                npix=int(min(s_ab["npix"], s_ba["npix"])),
                regridded=bool(s_ab["regridded"] or s_ba["regridded"]),
                resid_frac=_finite_extreme((s_ab.get("resid_frac", np.nan), s_ba.get("resid_frac", np.nan))),
                algorithm_version="profile-eiv-v2", covariance_mode="diagonal", uncertainty_calibrated=False,
                interpolation_covariance_ignored=bool(s_ab["regridded"] or s_ba["regridded"]),
                velocity_convention="optical_translation",
                err_method=(methods[0] if methods[0] == methods[1] else "mixed") if valid else "unavailable",
                n_mc_requested=sum(s.get("n_mc_requested", 0) for s in (s_ab, s_ba)),
                n_mc_success=sum(s.get("n_mc_success", 0) for s in (s_ab, s_ba)),
                bootstrap_fallback_reason=";".join(reasons) if reasons else None,
                # per direction: a as profile on template b (ab) and the reverse (ba)
                scale_ab=float(s_ab.get("scale", np.nan)), scale_ba=float(s_ba.get("scale", np.nan)),
                chi2_red_ab=float(s_ab["chi2_red"]), chi2_red_ba=float(s_ba["chi2_red"]),
                profile_z_ab=float(s_ab.get("profile_z", np.nan)), profile_z_ba=float(s_ba.get("profile_z", np.nan)),
                err_ab=float(errors[0]), err_ba=float(errors[1]),
                err_method_ab=methods[0], err_method_ba=methods[1],
                npix_ab=int(s_ab["npix"]), npix_ba=int(s_ba["npix"]),
                # masked native pixels of each spectrum inside the window of the
                # direction in which it is the profile
                n_masked_a=int(s_ab.get("n_masked_prof", 0)), n_masked_b=int(s_ba.get("n_masked_prof", 0)),
                # the search of the two directions: fraction of the window kept in
                # stage 1 and the smaller reach of the searched range (km/s)
                common_frac=_finite_extreme((float(s_ab.get("common_frac", np.nan)),
                                             float(s_ba.get("common_frac", np.nan))), min),
                search_reach=_finite_extreme(tuple(min(abs(x) for x in s.get("search_range", (np.nan, np.nan)))
                                                   for s in (s_ab, s_ba)), min),
                # plausibility of the match: flux factors and a second minimum
                scale_ok=_scales_ok(float(s_ab.get("scale", np.nan)), float(s_ba.get("scale", np.nan))),
                ambiguous=bool(s_ab.get("ambiguous", False) or s_ba.get("ambiguous", False)),
                dv_alt_ab=float(s_ab.get("dv_alt", np.nan)), dv_alt_ba=float(s_ba.get("dv_alt", np.nan)),
                dG_alt_ab=float(s_ab.get("dG_alt", np.nan)), dG_alt_ba=float(s_ba.get("dG_alt", np.nan)),
                s_ab=s_ab, s_ba=s_ba)


def narrow_zeropoint(res_a, res_b, prefer=("OIII", "SII"), vmax=800.0, **kw):
    """Wavelength zero point between two fits of the same object from the
    narrow lines: the shift of the broad-subtracted [O III] 5007 region of a
    relative to b (``line`` 'OIII'), or of the [S II] doublet when [O III] is
    not measurable in both fits ('SII'). One zero point serves the whole pair,
    whichever broad line is analysed.

    Measured in both directions, like the broad-line shift: dv = (dv_ab -
    dv_ba) / 2, so that swapping the two spectra changes only its sign (a
    one-directional zero point of an SDSS-DESI pair differs between the two
    orders by up to hundreds of km/s, because the two spectra have different
    lattices and noise). ``err`` is the larger directional error, ``at_bound``
    is set when either direction ends at the edge of the +/- ``vmax`` search,
    ``dir_mismatch`` is |dv_ab + dv_ba| and ``consistent`` compares it with
    max(2 hypot(err_ab, err_ba), one pixel). Returns that record with ``line``
    and ``source`` (the same name), or None when neither line gives a shift in
    both directions with 12 valid pixels in both fits."""
    for which in prefer:
        cplx = "Hbeta" if which == "OIII" else "Halpha"
        pa = narrow_profile_data(res_a, name=cplx, which=which)
        pb = narrow_profile_data(res_b, name=cplx, which=which)
        if pa is None or pb is None or pa["ok"].sum() < 12 or pb["ok"].sum() < 12:
            continue
        lo = min(pa["v"][pa["ok"]].min(), pb["v"][pb["ok"]].min())
        hi = max(pa["v"][pa["ok"]].max(), pb["v"][pb["ok"]].max())
        # The zero-point window is the whole narrow-line region with no data
        # beyond it: one stage over that window, as in 0.1.0; the +/- vmax
        # (800 km/s) search keeps the minimum from drifting.
        opts = dict(vmax=vmax, window=(lo, hi), baseline="const", min_pix=10, two_stage=False, **kw)
        s_ab = ccf_shift(pa, pb, **opts)
        s_ba = ccf_shift(pb, pa, **opts)
        if s_ab is None or s_ba is None:
            continue
        shifts = (float(s_ab["dv"]), float(s_ba["dv"]))
        errors = (float(s_ab.get("err", np.nan)), float(s_ba.get("err", np.nan)))
        finite_err = all(np.isfinite(e) and e > 0 for e in errors)
        mismatch = float(abs(shifts[0] + shifts[1]))
        tolerance = max(2.0 * float(np.hypot(*errors)), float(s_ab["dv_pix"]), float(s_ba["dv_pix"])) if finite_err else np.nan
        methods = (s_ab.get("err_method", "unavailable"), s_ba.get("err_method", "unavailable"))
        rec = dict(s_ab)
        rec.update(dv=0.5 * (shifts[0] - shifts[1]), err=float(max(errors)) if finite_err else np.nan,
                   err_method=(methods[0] if methods[0] == methods[1] else "mixed") if finite_err else "unavailable",
                   bracket=(s_ab.get("bracket") if s_ab.get("bracket") == s_ba.get("bracket") else "mixed"),
                   at_bound=bool(s_ab.get("at_bound", False) or s_ba.get("at_bound", False)),
                   dv_ab=shifts[0], dv_ba=shifts[1], err_ab=errors[0], err_ba=errors[1],
                   dir_mismatch=mismatch, consistent=bool(finite_err and mismatch <= tolerance),
                   line=which, source=which)
        return rec
    return None

def frame_check(zeropoint, name, dv, err):
    """Narrow-line frame check of a pair and the one zero-point rule.

    A pair is in one wavelength frame (``frame_ok``) when its narrow-line zero
    point is measured (finite shift, not at the search bound) and within
    FRAME_VETO_KMS of zero. A larger zero point means the two spectra do not
    share a frame (a calibration or reduction difference, or an aperture on a
    different part of the narrow-line region); such a pair is vetoed for both
    Balmer lines, because a frame offset enters the broad shifts of both lines
    identically and a coincident Halpha/Hbeta change is what the search
    rewards. Hbeta is corrected by the zero point when the frame is good
    (``dv_corrected`` = dv - zp_dv, the zero-point error added in quadrature);
    Halpha keeps its measured shift: the 0.1.0 on-sky calibration rejected an
    Halpha correction (it removed no scatter from consecutive DESI epochs).

    Returns dict(zp_dv, zp_err, zp_err_method, zp_line, zp_source, zp_at_bound,
    frame_ok, frame_reason, zp_applied, dv_corrected, err_corrected).
    ``frame_reason`` is '' when the frame is good, else 'no narrow zero point',
    'zero point at search bound', 'zero point inconsistent between the two
    directions' or 'zero point NNN km/s exceeds veto'."""
    z = zeropoint
    zp_dv = float(z["dv"]) if (z is not None and np.isfinite(z.get("dv", np.nan))) else np.nan
    zp_err = float(z["err"]) if (z is not None and np.isfinite(z.get("err", np.nan)) and z["err"] > 0) else np.nan
    at_bound = bool(z is not None and z.get("at_bound", False))
    source = z.get("source", z.get("line")) if z is not None else None
    if z is None or not np.isfinite(zp_dv):
        ok, reason = False, "no narrow zero point"
    elif at_bound:
        ok, reason = False, "zero point at search bound"
    elif z.get("consistent") is False:
        ok, reason = False, "zero point inconsistent between the two directions"
    elif abs(zp_dv) > FRAME_VETO_KMS:
        ok, reason = False, f"zero point {zp_dv:+.0f} km/s exceeds veto"
    else:
        ok, reason = True, ""
    applied = bool(ok and name == "Hbeta")
    dv_corr = float(dv - zp_dv) if applied else float(dv)
    err_corr = float(np.hypot(err, zp_err)) if applied else float(err)
    return dict(zp_dv=zp_dv, zp_err=zp_err, zp_err_method=(z.get("err_method", "unavailable") if z is not None else "unavailable"),
                zp_line=(source or ""), zp_source=source, zp_at_bound=at_bound,
                frame_ok=ok, frame_reason=reason, zp_applied=applied, dv_corrected=dv_corr, err_corrected=err_corr)


def pair_analysis(res_a, res_b, name="Halpha", zp=True, details=False, **kw):
    """Full comparison of two fits of the same object at the same redshift:
    bidirectional broad-line shift of a relative to b plus the narrow-line
    zero point and frame check. Returns a flat dict or None; with
    ``details=True`` the two one-directional ``ccf_shift`` results (including
    their curves) are added under 'details'. Keys:

    * name, dv, err, err_method, consistent, at_bound, chi2_red, profile_z,
      profile_grade, resid_frac, npix, regridded, snr_proxy, dir_mismatch and
      the bookkeeping of ``shift_bidirectional``;
    * per direction (a as profile on template b, and the reverse): scale_ab,
      scale_ba, chi2_red_ab, chi2_red_ba, profile_z_ab, profile_z_ba, err_ab,
      err_ba, err_method_ab, err_method_ba, npix_ab, npix_ba; n_masked_a,
      n_masked_b (masked native pixels inside the window);
    * plausibility of the match: scale_ok (both flux factors in
      CCF_SCALE_RANGE, their product in CCF_SCALE_PRODUCT_RANGE), ambiguous (a
      second minimum within Delta chi-square 6.63 of the first, at least
      CCF_ALT_MIN_SEP_KMS or three errors away, in either direction),
      dv_alt_ab, dv_alt_ba, dG_alt_ab, dG_alt_ba; common_frac (the smaller
      fraction of the window kept by the stage-1 search of the two directions)
      and search_reach (the smaller reach, km/s, of their searched ranges: a
      larger shift cannot be measured);
    * the frame check of ``frame_check``: zp_dv, zp_err, zp_err_method,
      zp_line, zp_source, zp_at_bound, frame_ok, frame_reason, zp_applied,
      dv_corrected, err_corrected (with ``zp=False`` no zero point is measured
      and frame_ok is False)."""
    pa = broad_profile_data(res_a, name=name)
    pb = broad_profile_data(res_b, name=name)
    if pa is None or pb is None:
        return None
    s = shift_bidirectional(pa, pb, **kw)
    if s is None:
        return None
    out = dict(name=name, dv=s["dv"], err=s["err"], consistent=s.get("consistent", False),
               at_bound=s.get("at_bound", False), chi2_red=s.get("chi2_red", np.nan),
               profile_z=_finite_extreme((s.get("s_ab", {}).get("profile_z", np.nan), s.get("s_ba", {}).get("profile_z", np.nan)))
               if s.get("s_ab") else np.nan,
               npix=s.get("npix", 0), regridded=s.get("regridded", False),
               snr_proxy=_finite_extreme((s.get("s_ab", {}).get("snr_proxy", np.nan), s.get("s_ba", {}).get("snr_proxy", np.nan)), min)
               if s.get("s_ab") else np.nan,
               dir_mismatch=s.get("dir_mismatch", np.nan), resid_frac=s.get("resid_frac", np.nan))
    for key in ("statistically_valid", "algorithm_version", "covariance_mode", "uncertainty_calibrated",
                "interpolation_covariance_ignored", "velocity_convention", "err_method",
                "n_mc_requested", "n_mc_success", "bootstrap_fallback_reason",
                "scale_ab", "scale_ba", "chi2_red_ab", "chi2_red_ba", "profile_z_ab", "profile_z_ba",
                "err_ab", "err_ba", "err_method_ab", "err_method_ba", "npix_ab", "npix_ba",
                "n_masked_a", "n_masked_b", "common_frac", "search_reach", "scale_ok", "ambiguous", "dv_alt_ab", "dv_alt_ba",
                "dG_alt_ab", "dG_alt_ba"):
        out[key] = s[key]
    out["profile_grade"] = profile_grade(out["profile_z"])
    if details:
        out["details"] = dict(s_ab=s.get("s_ab"), s_ba=s.get("s_ba"))
    out.update(frame_check(narrow_zeropoint(res_a, res_b) if zp else None, name, out["dv"], out["err"]))
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


def profile_grade(profile_z):
    """Profile-stability grade of a pair: 'stable' (z_prof < 5), 'mild' (5 to 10),
    'changed' (10 and above), or 'unknown'. The reliable tier of the DESI
    calibration is the 'stable' grade plus the bound and direction conditions;
    the other grades keep the measurement with an inflated error."""
    if profile_z is None or not np.isfinite(profile_z):
        return "unknown"
    for name, upper in PROFILE_GRADE_Z:
        if profile_z < upper:
            return name
    return "changed"


def error_inflation(name, grade):
    """Factor by which the cross-survey error floor grows with the profile grade."""
    return PROFILE_GRADE_INFLATION.get(name, PROFILE_GRADE_INFLATION["Halpha"]).get(grade, np.nan)


def cross_survey_floor(name, grade):
    """Error floor of a point measured against a template from another survey:
    the null scatter of a stable-grade point times the inflation of its grade."""
    return CCF_NULL_KMS.get(name, np.nan) * error_inflation(name, grade)


def two_line_consistent(pair_a, pair_b, nsig=2.0, covariance=0.0):
    """The two-line criterion of Liu et al. (2014) and Guo et al. (2019): the
    shifts of two lines of the same pair of spectra agree within ``nsig`` times
    their combined error and, where both are significant, in sign. Returns
    dict(consistent, difference, sigma, same_sign) or None when either pair is
    missing, at bound or without a finite error."""
    for p in (pair_a, pair_b):
        if (p is None or p.get("at_bound") or not np.isfinite(p.get("dv", np.nan))
                or not np.isfinite(p.get("err", np.nan)) or p["err"] <= 0):
            return None
    if not np.isfinite(covariance) or abs(covariance) > pair_a["err"] * pair_b["err"]:
        raise ValueError("cross-line covariance must define a positive semidefinite error matrix")
    d = float(pair_a["dv"] - pair_b["dv"])
    e = float(np.sqrt(max(pair_a["err"]**2 + pair_b["err"]**2 - 2.0*covariance, 0.0)))
    sig_a = abs(pair_a["dv"]) > pair_a["err"]; sig_b = abs(pair_b["dv"]) > pair_b["err"]
    same_sign = bool(np.sign(pair_a["dv"]) == np.sign(pair_b["dv"])) if (sig_a and sig_b) else True
    return dict(consistent=bool(abs(d) <= nsig * e and same_sign), difference=d,
                sigma=float(abs(d) / e) if e > 0 else np.nan, same_sign=same_sign)


def is_reliable(pair, dir_cut=None):
    """The reliable tier: a finite shift and positive error, not at bound,
    profile_z < 5, direction mismatch below the per-line cut (466 km/s Halpha,
    238 km/s Hbeta; the 0.1.0 calibration) and a good narrow-line frame
    (``frame_ok``: a measured zero point within FRAME_VETO_KMS), plausible flux
    factors (``scale_ok``) and no second minimum of similar depth
    (``ambiguous``). Records built by hand without the last two keys pass them.

    ``dir_cut`` (km/s, optional) replaces CCF_DIR_CUT_KMS for this call: a
    number is the cut for the pair's line, a mapping {line name: cut} is looked
    up with the pair's ``name`` (a line missing from it has no finite cut, so the
    pair is not reliable). This is how a run applies a recalibrated cut (three
    times the recalibrated floor) without changing the package constant."""
    if (pair is None or pair.get("at_bound") or not np.isfinite(pair.get("dv", np.nan))
            or not np.isfinite(pair.get("err", np.nan)) or pair["err"] <= 0):
        return False
    if dir_cut is None:
        cut = CCF_DIR_CUT_KMS.get(pair.get("name"), np.nan)
    elif hasattr(dir_cut, "get"):
        cut = dir_cut.get(pair.get("name"), np.nan)
    else:
        cut = dir_cut
    cut = float(cut) if cut is not None else np.nan
    return bool(np.isfinite(pair.get("profile_z", np.nan)) and pair["profile_z"] < CCF_PROFILE_Z_MAX
                and np.isfinite(pair.get("dir_mismatch", np.nan)) and pair["dir_mismatch"] < cut
                and pair.get("frame_ok", False) and pair.get("scale_ok", True)
                and not pair.get("ambiguous", False))
