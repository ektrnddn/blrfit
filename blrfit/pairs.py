"""
Velocity changes of a broad line between two spectra of one object.

The change is measured by a template cross-correlation. The broad model of one
epoch (its Gaussians as ``fit_spectrum`` returned them, nothing else) is slid
across the other epoch's continuum- and narrow-subtracted data; at each trial
shift the template's flux scale (held non-negative), an offset and a slope are
solved linearly, and the chi-square curve over the shifts gives the shift, an
error from the curvature at the minimum, and a check for a second minimum. The
other direction, the second epoch's model across the first epoch's data, is
measured in the same way. The change of the pair is half the difference of the
two directions, (s - s')/2, with the error hypot(err, err'): each direction's
curvature sees the noise of its data epoch only, the template epoch's noise
enters through its fitted profile, and the two directions are nearly
anti-correlated, so the two curvature errors add in quadrature (with half that
error the estimate covered the truth 38 per cent of the time at a nominal 68).

A difference of two separately fitted offsets mixes a bulk motion with a change
of profile, with the decomposition each fit chose and with each fit's own
narrow-line subtraction. Sliding one epoch's smooth model across the other
epoch's data asks a narrower question, whether the same profile has moved, and
the shape statistic (``dchi2_shape``, the template against the data epoch's own
broad model on the same pixels) answers separately whether it is still the same
profile. Injected shifts of up to 2,500 km/s are recovered without attenuation;
see the validation page.

Errors come from the statistical inverse variance of the input spectrum, taken
pixel by pixel through the fitter's own preprocessing, never from the fit
weights: those carry the 2 per cent calibration floor, which is a systematic of
the absolute calibration and does not scatter one pixel against the next. A
term per line (``PAIR_SYS_KMS``), measured on pairs of spectra with no expected
change, is added in quadrature to the statistical error of a pair.

The frame of a pair is the common input redshift. The broad shift is measured
in it (``s_common``), and the shift of the narrow lines between the epochs
(``dv_narrow``, measured with the same code) is reported beside it. The narrow
shift does not correct the broad shift: on pairs of SDSS spectra the corrected
shift was no tighter than the uncorrected one, and a narrow shift beyond
FRAME_VETO_KMS vetoes the pair instead (flag ``frame_offset_large``).

``measure_pair`` measures one line of one pair; ``pair_record`` adds the
identity, time and systematic-error columns of the pair table;
``enumerate_pairs`` lists the pairs of several epochs. Velocities are optical
velocities in km/s; a positive change means the later epoch is redder.
"""

from __future__ import annotations

import re

import numpy as np

from .constants import (
    C_KMS,
    ERR_FLOOR,
    FRAME_VETO_KMS,
    LAM,
    MASK_GROW_PIX,
    CCF_DCHI2_99,
    CCF_ALT_MIN_SEP_KMS,
    CCF_NSUB_FRAC,
    CCF_SCALE_RANGE,
    CCF_SCALE_PRODUCT_RANGE,
    PAIR_VMAX_KMS,
    PAIR_STEP_KMS,
    PAIR_EDGE_STEPS,
    PAIR_REFINE_DCHI2,
    PAIR_REFINE_MIN_POINTS,
    PAIR_WIN_FWHM,
    PAIR_WIN_MIN_KMS,
    PAIR_MIN_PIX,
    PAIR_NARROW_VMAX_KMS,
    PAIR_NARROW_STEP_KMS,
    PAIR_NARROW_WIN_KMS,
    PAIR_DIR_NSIG,
    PAIR_SYS_KMS,
    PAIR_SHAPE_MAX,
    PAIR_PRIOR_FRAC,
)
from .model.extinction import deredden
from .model.fit import grow_mask
from .model.lines import eval_components
from .model.params import gauss_lam

ALGORITHM_VERSION = "template-v1"

NARROW_KINDS = ("narrow", "nwing", "wing")
BROAD_KINDS = ("broad",)

# the columns of a pair table, in the order written
PAIR_COLUMNS = (
    "targetid",
    "pair",
    "kind",
    "line",
    "role",
    "id_a",
    "id_b",
    "mjd_a",
    "mjd_b",
    "dt_days",
    "dt_rest_yr",
    "retained",
    "stable_shape",
    "s_common",
    "err",
    "sigma_sys",
    "err_total",
    "s_fwd",
    "err_fwd",
    "s_rev",
    "err_rev",
    "dir_mismatch",
    "dv_narrow",
    "err_narrow",
    "s_corrected",
    "err_corrected",
    "v_sys_diff",
    "dc50_sys",
    "dc50_model",
    "scale_fwd",
    "scale_rev",
    "chi2_nu_fwd",
    "chi2_nu_rev",
    "chi2_own_nu_fwd",
    "chi2_own_nu_rev",
    "dchi2_shape_fwd",
    "dchi2_shape_rev",
    "shape_max",
    "npix_fwd",
    "npix_rev",
    "s_alt_fwd",
    "dchi2_alt_fwd",
    "s_alt_rev",
    "dchi2_alt_rev",
    "fwhm_a",
    "fwhm_b",
    "snr_a",
    "snr_b",
    "peak_snr_a",
    "peak_snr_b",
    "cls_a",
    "cls_b",
    "z_a",
    "z_b",
    "ebv_a",
    "ebv_b",
    "errors_a",
    "errors_b",
    "flags",
)

PAIR_FLAG_TEXT = {
    "at_bound": "chi-square minimum at the edge of the +/- 4000 km/s scan: no shift claimed",
    "flat_minimum": "no curvature at the minimum: no error",
    "ambiguous": "a second minimum within Delta chi-square 6.63, more than 500 km/s away",
    "scale_out_of_range": "template flux scale outside 0.25-4: a sliver of the profile was matched",
    "too_few_pixels": "fewer than 20 usable pixels in the window",
    "dir_inconsistent": "the two directions disagree by more than 3 sigma (an asymmetry of the templates)",
    "scale_product_out_of_range": "the flux scales of the two directions are not reciprocal (product outside 0.5-2)",
    "frame_offset_large": "the narrow lines moved by more than 200 km/s between the epochs: the frame is not shared",
    "no_narrow_template": "no narrow line with positive amplitude in the template epoch",
    "z_differs": "the two fits used different redshifts",
    "ebv_differs": "the two fits used different Galactic E(B-V)",
    "weights_inconsistent_a": "the fit weights of epoch A could not be rebuilt from the given input spectrum",
    "weights_inconsistent_b": "the fit weights of epoch B could not be rebuilt from the given input spectrum",
    "errors_from_fit_weights_a": "no statistical errors for epoch A: the fit weights (with the floor) were used",
    "errors_from_fit_weights_b": "no statistical errors for epoch B: the fit weights (with the floor) were used",
    "line_not_fitted": "the line was not fitted in one of the epochs",
    "uncalibrated_line": "no systematic term is calibrated for this line: err_total is the statistical error",
    "not_retained": "a screen failed: the shift is withheld, the direction values are kept",
    "dependent": "the two spectra are not independent observations",
}


# ----------------------------------------------------------------------------
# grids and small helpers
# ----------------------------------------------------------------------------
def velocity_grid(vmax=PAIR_VMAX_KMS, step=PAIR_STEP_KMS):
    """Symmetric grid of trial shifts, exactly centred on zero."""
    n = int(round(vmax / step))
    return np.arange(-n, n + 1, dtype=float) * step


def statistical_ivar_on_fit_grid(
    x_fit, wave_obs, flux, ivar, z, ebv=0.0, flux_scale=1.0, mask_grow=MASK_GROW_PIX
):
    """The statistical inverse variance (and the total flux) of an input spectrum
    on the pixels ``x_fit`` of its fit, through the same path as
    ``fit_spectrum``: flux scale, ascending order, bad pixels zeroed and the mask
    grown, Galactic de-reddening with the fit's E(B-V), then the rest frame
    (ivar / (1+z)^2, flux * (1+z)). Returns dict(ivar_stat, flux_rest, matched,
    max_rel): ``matched`` marks the fit pixels found in the input (the
    wavelengths must agree to 1e-7 relative)."""
    wave_obs = np.asarray(wave_obs, float)
    flux = np.asarray(flux, float) * float(flux_scale)
    ivar = np.asarray(ivar, float) / float(flux_scale) ** 2
    wv = wave_obs[np.isfinite(wave_obs)]
    if wv.size > 1 and np.any(np.diff(wv) < 0):
        order = np.argsort(wave_obs, kind="stable")
        wave_obs, flux, ivar = wave_obs[order], flux[order], ivar[order]
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    bad = grow_mask(bad, int(mask_grow))
    flux = np.where(bad, 0.0, flux)
    ivar = np.where(bad, 0.0, ivar)
    flux, ivar = deredden(wave_obs, flux, ivar, float(ebv))
    wr = wave_obs / (1.0 + z)
    fr = flux * (1.0 + z)
    ir = ivar / (1.0 + z) ** 2
    x_fit = np.asarray(x_fit, float)
    if x_fit.size == 0:
        return dict(ivar_stat=x_fit, flux_rest=x_fit, matched=np.zeros(0, bool), max_rel=0.0)
    j = np.clip(np.searchsorted(wr, x_fit), 0, wr.size - 1)
    jm = np.clip(j - 1, 0, wr.size - 1)
    pick = np.where(np.abs(wr[jm] - x_fit) < np.abs(wr[j] - x_fit), jm, j)
    rel = np.abs(wr[pick] - x_fit) / x_fit
    matched = rel < 1e-7
    return dict(
        ivar_stat=np.where(matched, ir[pick], 0.0),
        flux_rest=np.where(matched, fr[pick], np.nan),
        matched=matched,
        max_rel=float(rel.max()),
    )


def _ivar_from_result(res, x_fit):
    """The statistical inverse variance and the total flux of a ``fit_spectrum``
    result on the pixels of one of its line fits: the fit keeps ``ivar_stat_rest``
    (the inverse variance before the error floor) on its rest-frame grid, of
    which the line's pixels are a subset."""
    wr = np.asarray(res["wave_rest"], float)
    x_fit = np.asarray(x_fit, float)
    j = np.clip(np.searchsorted(wr, x_fit), 0, wr.size - 1)
    jm = np.clip(j - 1, 0, wr.size - 1)
    pick = np.where(np.abs(wr[jm] - x_fit) < np.abs(wr[j] - x_fit), jm, j)
    if x_fit.size and np.max(np.abs(wr[pick] - x_fit) / x_fit) > 1e-9:
        raise ValueError("the line's pixels are not on the result's wavelength grid")
    return np.asarray(res["ivar_stat_rest"], float)[pick], np.asarray(res["flux_rest"], float)[pick]


def narrow_basis(x, d, comps, kinds=NARROW_KINDS):
    """Unit-amplitude profiles of the narrow components, one per free amplitude
    parameter (components tied by a ratio share their parameter, as in the fit).
    Returns (basis, amplitudes, kinds): the profiles, the fitted amplitude of each
    (a solved correction c keeps the total d[an] + c non-negative) and the kind."""
    basis, amps, kind_of = {}, {}, {}
    for _lab, lam0, an, vn, sn, kind, ratio in comps:
        if kind not in kinds:
            continue
        fac = ratio[1] if ratio else 1.0
        basis[an] = basis.get(an, 0.0) + fac * gauss_lam(x, 1.0, lam0, d[vn], d[sn])
        amps[an] = float(d[an])
        kind_of[an] = kind
    return basis, amps, kind_of


def shifted_components(d, comps, s_kms, kinds=BROAD_KINDS, extra_sigma_kms=0.0):
    """Parameters ``d`` with the velocity of every component of the given kinds
    moved by ``s_kms`` (the component centre is lam0 (1 + v/c)). With
    ``extra_sigma_kms`` the components are broadened in quadrature at constant
    flux, for tests of a resolution difference between the epochs."""
    d2 = dict(d)
    for _lab, _lam0, an, vn, sn, kind, _ratio in comps:
        if kind not in kinds:
            continue
        d2[vn] = float(d[vn]) + float(s_kms)
        if extra_sigma_kms > 0 and d[sn] > 0:
            s_new = float(np.hypot(d[sn], extra_sigma_kms))
            d2[sn] = s_new
            d2[an] = float(d[an]) * float(d[sn]) / s_new
    return d2


def model_halfmax(d, comps, lam0, kinds=BROAD_KINDS, step_kms=2.0, vspan_kms=25000.0):
    """c(1/2) and FWHM of the summed model profile of the given kinds, from the
    outermost half-maximum crossings on a fine velocity grid (km/s from lam0)."""
    v = np.arange(-vspan_kms, vspan_kms + step_kms, step_kms)
    p = eval_components(lam0 * (1.0 + v / C_KMS), d, comps, kinds=kinds)
    if not np.any(p > 0):
        return dict(c50=np.nan, fwhm=np.nan, peak=0.0, v_peak=np.nan)
    k = int(np.argmax(p))
    half = 0.5 * p[k]
    idx = np.where(p >= half)[0]
    i0, i1 = int(idx[0]), int(idx[-1])

    def cross(i, j):
        # linear interpolation of the crossing between grid points i (below) and j (above)
        if i < 0 or j >= v.size or p[j] == p[i]:
            return v[max(i, 0)]
        return v[i] + (half - p[i]) * (v[j] - v[i]) / (p[j] - p[i])

    lo = cross(i0 - 1, i0) if i0 > 0 else v[0]
    hi = cross(i1 + 1, i1) if i1 < v.size - 1 else v[-1]
    return dict(c50=0.5 * (lo + hi), fwhm=hi - lo, peak=float(p[k]), v_peak=float(v[k]))


# ----------------------------------------------------------------------------
# one epoch
# ----------------------------------------------------------------------------
def epoch_profile(res, line, inputs=None):
    """Everything the measurement needs from one epoch's fit of one line, or None
    when the line was not fitted.

    ``res`` is a ``fit_spectrum`` result (or a reduced copy that keeps ``z``,
    ``settings`` and, per line, ``x``, ``y``, ``w``, ``d`` and ``comps``). The
    statistical errors come, in this order, from ``inputs`` = dict(wave, flux,
    ivar), the observed-frame arrays the fitter was given; from the result's own
    ``ivar_stat_rest``; from a record's ``ivar_stat``; or, as a last resort, from
    the fit weights, which include the floor. ``errors_source`` says which. With
    the first two, the fit weights are rebuilt from the statistical variance, the
    total flux and the fit's floor and compared with the saved ``w``:
    ``weights_consistent`` is the proof that the pixels and the preprocessing are
    the fit's own."""
    fits = res.get("fits") or {}
    r = fits.get(line)
    if r is None:
        return None
    x = np.asarray(r["x"], float)
    y = np.asarray(r["y"], float)
    w = np.asarray(r["w"], float)
    d, comps = r["d"], r["comps"]
    m = (res.get("meas") or {}).get(line) or {}
    settings = res.get("settings") or {}
    z = float(res.get("z", np.nan))
    ebv = float(settings.get("ebv", 0.0) or 0.0)
    floor = settings.get("err_floor")
    floor = float(ERR_FLOOR if floor is None else floor)
    lam0 = LAM[line]
    narrow = eval_components(x, d, comps, kinds=NARROW_KINDS)
    broad = eval_components(x, d, comps, kinds=BROAD_KINDS)
    weights_consistent, max_rel_dev, flux_rest = None, np.nan, None
    if inputs is not None:
        mg = settings.get("mask_grow")
        st = statistical_ivar_on_fit_grid(
            x,
            inputs["wave"],
            inputs["flux"],
            inputs["ivar"],
            z,
            ebv,
            flux_scale=float(settings.get("flux_scale", 1.0) or 1.0),
            mask_grow=int(MASK_GROW_PIX if mg is None else mg),
        )
        if not st["matched"].all():
            raise ValueError(
                f"{int((~st['matched']).sum())} of {x.size} fit pixels have no input pixel "
                f"(largest relative wavelength difference {st['max_rel']:.2e})"
            )
        ivar_stat, flux_rest, source = st["ivar_stat"], st["flux_rest"], "input spectrum"
    elif "ivar_stat_rest" in res and "wave_rest" in res:
        ivar_stat, flux_rest = _ivar_from_result(res, x)
        source = "fit result"
    elif "ivar_stat" in r:
        ivar_stat, source = np.asarray(r["ivar_stat"], float), "record"
    else:
        ivar_stat, source = w**2, "fit weights (floor included)"
    if flux_rest is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            var_eff = np.where(ivar_stat > 0, 1.0 / ivar_stat, np.inf) + (floor * np.abs(flux_rest)) ** 2
            w_rebuilt = np.where(np.isfinite(var_eff), 1.0 / np.sqrt(var_eff), 0.0)
        good = w > 0
        max_rel_dev = float(np.max(np.abs(w_rebuilt[good] - w[good]) / w[good])) if good.any() else 0.0
        weights_consistent = bool(max_rel_dev < 1e-6)
    with np.errstate(divide="ignore"):
        sig = np.where(ivar_stat > 0, 1.0 / np.sqrt(np.where(ivar_stat > 0, ivar_stat, 1.0)), np.inf)
    ok = np.isfinite(y) & np.isfinite(sig) & (ivar_stat > 0) & (w > 0)
    hm = model_halfmax(d, comps, lam0)
    basis, amps, kind_of = narrow_basis(x, d, comps)
    lo, hi = r.get("window") or (float(x.min()), float(x.max()))
    cls = ((res.get("cls") or {}).get(line) or {}).get("label", "")
    return dict(
        line=line,
        lam0=lam0,
        x=x,
        v=(x / lam0 - 1.0) * C_KMS,
        y=y,
        f=y - narrow,
        sig=sig,
        w=w,
        ok=ok,
        narrow=narrow,
        broad=broad,
        d=d,
        comps=comps,
        z=z,
        ebv=ebv,
        cls=cls,
        v_sys=float(m.get("v_sys", np.nan)),
        fwhm=float(m.get("fwhm", np.nan)),
        c50=float(m.get("c50", np.nan)),
        c50_sys=float(m.get("c50_sys", np.nan)),
        snr=float(m.get("broad_flux_snr", m.get("snr", np.nan))),
        peak_snr=float(m.get("broad_peak_snr", np.nan)),
        model_c50=hm["c50"],
        model_fwhm=hm["fwhm"],
        model_peak=hm["peak"],
        window_kms=((lo / lam0 - 1.0) * C_KMS, (hi / lam0 - 1.0) * C_KMS),
        errors_source=source,
        weights_consistent=weights_consistent,
        weight_max_rel_dev=max_rel_dev,
        narrow_basis=basis,
        narrow_amp=amps,
        narrow_kind=kind_of,
    )


# ----------------------------------------------------------------------------
# the scan
# ----------------------------------------------------------------------------
def select_window(ep_data, ep_templ, grid, win_fwhm=PAIR_WIN_FWHM, win_min=PAIR_WIN_MIN_KMS):
    """The fixed pixel set of a direction: usable pixels of the data epoch within
    max(win_fwhm * FWHM, win_min) of the model c(1/2) of either epoch (so that a
    large shift keeps both profiles inside), restricted to where the template
    epoch's fitted window covers the data for every trial shift. A boolean mask
    over the data epoch's pixels."""
    fw = np.nanmax([ep_templ["model_fwhm"], ep_data["model_fwhm"]])
    half = max(win_fwhm * fw, win_min) if np.isfinite(fw) else win_min
    centres = [c for c in (ep_templ["model_c50"], ep_data["model_c50"]) if np.isfinite(c)]
    if not centres:
        return np.zeros(ep_data["x"].size, bool)
    v = ep_data["v"]
    lo, hi = ep_templ["window_kms"]
    sel = ep_data["ok"] & (v >= min(centres) - half) & (v <= max(centres) + half)
    sel &= (v - grid.max() >= lo) & (v - grid.min() <= hi)
    return sel


def _design_fixed(ep_data, sel, slope, narrow, free_kinds=NARROW_KINDS):
    """The shift-independent columns: offset, optional slope, optional narrow
    amplitudes of the data epoch (the components of ``free_kinds``). Returns
    (matrix, column names)."""
    x = ep_data["x"][sel]
    cols, names = [np.ones_like(x)], ["offset"]
    if slope:
        cols.append((x - ep_data["lam0"]) / 100.0)
        names.append("slope")
    if narrow in ("solve", "prior"):
        for an, prof in ep_data["narrow_basis"].items():
            if ep_data.get("narrow_kind", {}).get(an, "narrow") not in free_kinds:
                continue
            if not ep_data.get("narrow_amp", {}).get(an, 0.0) > 0:
                continue  # a component the fit found absent stays absent
            col = np.asarray(prof)[sel]
            if np.any(col != 0):
                cols.append(col)
                names.append(an)
    return np.column_stack(cols), names


def _solve_linear(M, fw, lb, ub):
    """Weighted linear least squares with the amplitude columns (template scale,
    narrow amplitudes) held within their bounds: the unconstrained solution when
    it respects them, else the offending columns are held at their bound value
    and the rest re-solved. An inverted template is not a model of the line, and
    a narrow line cannot be subtracted below zero."""
    ncol = M.shape[1]
    active = np.ones(ncol, bool)
    coef = np.zeros(ncol)
    for _ in range(ncol + 1):
        if not active.any():
            break
        rhs = fw - M[:, ~active] @ coef[~active]
        sol, *_ = np.linalg.lstsq(M[:, active], rhs, rcond=None)
        coef[active] = sol
        bad_lo = active & (coef < lb)
        bad_hi = active & (coef > ub)
        if not (bad_lo.any() or bad_hi.any()):
            break
        coef[bad_lo] = lb[bad_lo]
        coef[bad_hi] = ub[bad_hi]
        active &= ~(bad_lo | bad_hi)
    res = fw - M @ coef
    return coef, float(res @ res)


def scan_chi2(
    ep_data,
    ep_templ,
    grid,
    sel,
    slope=True,
    narrow="none",
    kappa=CCF_NSUB_FRAC,
    kappa_prior=PAIR_PRIOR_FRAC,
    free_kinds=NARROW_KINDS,
    template_kinds=BROAD_KINDS,
    extra_sigma_kms=0.0,
):
    """Chi-square of the data epoch's broad-only data against the template
    epoch's model shifted by every grid value, with the linear terms solved at
    each shift.

    narrow: "none" subtracts the data epoch's own narrow model and nothing more;
    "weight" adds (kappa * |narrow model|)^2 to the variance; "solve" frees the
    amplitudes of the data epoch's narrow components (centres and widths fixed);
    "prior" frees them under a Gaussian prior of width kappa_prior times the
    fitted amplitude plus one pixel's noise (a ridge: rows c_k / (kappa_prior A_k
    + sigma_pix) appended to the weighted system; the noise floor keeps a
    negligible line from dominating the conditioning). ``free_kinds`` limits the
    freed components (("narrow",): the cores only). Returns dict(grid, chi2,
    scale, coefs, columns, npix, nlin)."""
    if sel.sum() < 3:
        return dict(
            grid=grid,
            chi2=np.full(grid.size, np.nan),
            scale=np.full(grid.size, np.nan),
            coefs=None,
            columns=[],
            npix=int(sel.sum()),
            nlin=0,
        )
    x = ep_data["x"][sel]
    f = ep_data["f"][sel]
    sig = ep_data["sig"][sel]
    if narrow == "weight":
        sig = np.hypot(sig, kappa * np.abs(ep_data["narrow"][sel]))
    W = 1.0 / sig
    fixed, names = _design_fixed(ep_data, sel, slope, narrow, free_kinds)
    fixed = fixed * W[:, None]
    names = ["scale"] + names
    ncol = len(names)
    # bounds: the template scale >= 0; a narrow correction c keeps d[an] + c >= 0; offset and slope free
    lb = np.array(
        [
            0.0
            if n == "scale"
            else (-np.inf if n in ("offset", "slope") else -ep_data["narrow_amp"].get(n, 0.0))
            for n in names
        ]
    )
    ub = np.full(ncol, np.inf)
    fw = f * W
    prior_rows = None
    if narrow == "prior":
        rows = []
        amp_floor = float(np.median(sig))
        for j, n in enumerate(names):
            if n in ("scale", "offset", "slope"):
                continue
            row = np.zeros(ncol)
            row[j] = 1.0 / (kappa_prior * ep_data["narrow_amp"][n] + amp_floor)
            rows.append(row)
        if rows:
            prior_rows = np.array(rows)
            fw = np.concatenate([fw, np.zeros(len(rows))])
    chi2 = np.full(grid.size, np.nan)
    coefs = np.full((grid.size, ncol), np.nan)
    d, comps = ep_templ["d"], ep_templ["comps"]
    for k, s in enumerate(grid):
        t = eval_components(
            x, shifted_components(d, comps, s, template_kinds, extra_sigma_kms), comps, kinds=template_kinds
        )
        M = np.column_stack([t * W, fixed])
        if prior_rows is not None:
            M = np.vstack([M, prior_rows])
        coef, c2 = _solve_linear(M, fw, lb, ub)
        chi2[k] = c2
        coefs[k] = coef
    return dict(
        grid=grid, chi2=chi2, scale=coefs[:, 0], coefs=coefs, columns=names, npix=int(x.size), nlin=ncol
    )


def refine_minimum(
    grid, chi2, edge_steps=PAIR_EDGE_STEPS, dchi2_fit=PAIR_REFINE_DCHI2, min_points=PAIR_REFINE_MIN_POINTS
):
    """Position, error and depth of the chi-square minimum. A parabola is fitted
    to the contiguous points within ``dchi2_fit`` of the minimum (at least
    ``min_points``, else the three-point formula); the error is the Delta
    chi-square = 1 half-width of that parabola. ``at_bound`` marks a minimum
    within ``edge_steps`` of the grid edge: no shift is claimed (s is the grid
    value, err NaN)."""
    chi2 = np.asarray(chi2, float)
    grid = np.asarray(grid, float)
    n = grid.size
    out = dict(s=np.nan, err=np.nan, chi2min=np.nan, k0=-1, at_bound=True, method="none", curvature=np.nan)
    if n < 3 or np.sum(np.isfinite(chi2)) < 3:
        return out
    k0 = int(np.nanargmin(chi2))
    out.update(k0=k0, chi2min=float(chi2[k0]), s=float(grid[k0]))
    if k0 <= edge_steps or k0 >= n - 1 - edge_steps:
        out.update(at_bound=True, method="at_bound")
        return out
    out["at_bound"] = False
    lo, hi = k0, k0
    while lo - 1 >= 0 and np.isfinite(chi2[lo - 1]) and chi2[lo - 1] - chi2[k0] <= dchi2_fit:
        lo -= 1
    while hi + 1 < n and np.isfinite(chi2[hi + 1]) and chi2[hi + 1] - chi2[k0] <= dchi2_fit:
        hi += 1
    g0 = grid[k0]
    if hi - lo + 1 >= min_points:
        a, b, c = np.polyfit(grid[lo : hi + 1] - g0, chi2[lo : hi + 1], 2)
        method = f"parabola_{hi - lo + 1}"
    else:
        h = grid[k0 + 1] - grid[k0]
        a = (chi2[k0 + 1] - 2.0 * chi2[k0] + chi2[k0 - 1]) / (2.0 * h * h)
        b = (chi2[k0 + 1] - chi2[k0 - 1]) / (2.0 * h)
        c = chi2[k0]
        method = "parabola_3"
    if not (np.isfinite(a) and a > 0):
        out.update(method=method + "_flat")
        return out
    s = g0 - b / (2.0 * a)
    out.update(
        s=float(s),
        err=float(1.0 / np.sqrt(a)),
        chi2min=float(c - b * b / (4.0 * a)),
        method=method,
        curvature=float(a),
    )
    return out


def second_minimum(grid, chi2, k0, min_sep_kms=CCF_ALT_MIN_SEP_KMS):
    """The deepest local minimum of chi2 at least ``min_sep_kms`` from the global
    minimum at index k0. Returns (s_alt, dchi2_alt), or (nan, nan)."""
    chi2 = np.asarray(chi2, float)
    grid = np.asarray(grid, float)
    best = (np.nan, np.nan)
    for k in range(1, grid.size - 1):
        if k == k0 or not np.isfinite(chi2[k]) or abs(grid[k] - grid[k0]) < min_sep_kms:
            continue
        near = chi2[max(0, k - 2) : min(grid.size, k + 3)]
        if np.all(np.isfinite(near)) and chi2[k] <= np.min(near):
            dd = float(chi2[k] - chi2[k0])
            if not np.isfinite(best[1]) or dd < best[1]:
                best = (float(grid[k]), dd)
    return best


def measure_direction(
    ep_data,
    ep_templ,
    grid=None,
    slope=True,
    narrow="none",
    kappa=CCF_NSUB_FRAC,
    kappa_prior=PAIR_PRIOR_FRAC,
    free_kinds=NARROW_KINDS,
    extra_sigma_kms=0.0,
    details=False,
):
    """Shift of the data epoch's broad profile relative to the template epoch's
    (positive: the data epoch is redder), with its error, chi-square
    diagnostics, the shape statistic (the template against the data epoch's own
    broad model on the same pixels) and flags. The curvature error is scaled by
    the square root of the reduced chi-square where that exceeds one."""
    grid = velocity_grid() if grid is None else np.asarray(grid, float)
    sel = select_window(ep_data, ep_templ, grid)
    npix = int(sel.sum())
    rec = dict(s=np.nan, err=np.nan, npix=npix, flags=[])
    if npix < PAIR_MIN_PIX:
        rec["flags"].append("too_few_pixels")
        return rec
    opts = dict(slope=slope, narrow=narrow, kappa=kappa, kappa_prior=kappa_prior, free_kinds=free_kinds)
    sc = scan_chi2(ep_data, ep_templ, grid, sel, extra_sigma_kms=extra_sigma_kms, **opts)
    rf = refine_minimum(grid, sc["chi2"])
    dof = max(npix - sc["nlin"] - 1, 1)
    chi2_nu = rf["chi2min"] / dof if np.isfinite(rf["chi2min"]) else np.nan
    err = rf["err"]
    scaled = np.isfinite(err) and np.isfinite(chi2_nu) and chi2_nu > 1
    err_scaled = err * np.sqrt(chi2_nu) if scaled else err
    own = scan_chi2(ep_data, ep_data, np.array([0.0]), sel, **opts)
    chi2_own = float(own["chi2"][0])
    s_alt, dchi2_alt = second_minimum(grid, sc["chi2"], rf["k0"]) if rf["k0"] >= 0 else (np.nan, np.nan)
    scale = float(sc["scale"][rf["k0"]]) if rf["k0"] >= 0 else np.nan
    flags = []
    if rf["at_bound"]:
        flags.append("at_bound")
    if rf["method"].endswith("_flat"):
        flags.append("flat_minimum")
    if np.isfinite(dchi2_alt) and dchi2_alt < CCF_DCHI2_99:
        flags.append("ambiguous")
    if not (CCF_SCALE_RANGE[0] <= scale <= CCF_SCALE_RANGE[1]):
        flags.append("scale_out_of_range")
    rec.update(
        s=float(rf["s"]) if not rf["at_bound"] else np.nan,
        s_grid=float(rf["s"]),
        err=float(err_scaled) if not rf["at_bound"] else np.nan,
        err_raw=float(err),
        chi2min=float(rf["chi2min"]),
        chi2_nu=float(chi2_nu),
        dof=int(dof),
        nlin=int(sc["nlin"]),
        chi2_own=chi2_own,
        chi2_own_nu=chi2_own / dof,
        dchi2_shape=float(rf["chi2min"] - chi2_own),
        scale=scale,
        s_alt=s_alt,
        dchi2_alt=dchi2_alt,
        at_bound=bool(rf["at_bound"]),
        method=rf["method"],
        flags=flags,
    )
    if details:
        rec["grid"], rec["chi2"], rec["scale_curve"], rec["sel"] = grid, sc["chi2"], sc["scale"], sel
        rec["coef_best"] = sc["coefs"][rf["k0"]] if (sc.get("coefs") is not None and rf["k0"] >= 0) else None
        rec["columns"] = sc["columns"]
    return rec


def model_at(
    ep_data,
    ep_templ,
    s,
    coef,
    sel,
    slope=True,
    narrow="none",
    free_kinds=NARROW_KINDS,
    template_kinds=BROAD_KINDS,
    extra_sigma_kms=0.0,
):
    """The full linear model (scaled shifted template + offset [+ slope] [+ narrow
    amplitudes]) on the pixels ``sel`` for the coefficients ``coef`` of a scan."""
    x = ep_data["x"][sel]
    t = eval_components(
        x,
        shifted_components(ep_templ["d"], ep_templ["comps"], s, template_kinds, extra_sigma_kms),
        ep_templ["comps"],
        kinds=template_kinds,
    )
    fixed, _names = _design_fixed(ep_data, sel, slope, narrow, free_kinds)
    return coef[0] * t + fixed @ np.asarray(coef[1:])


def narrow_shift(
    ep_data,
    ep_templ,
    vmax=PAIR_NARROW_VMAX_KMS,
    step=PAIR_NARROW_STEP_KMS,
    win=PAIR_NARROW_WIN_KMS,
    labels=None,
    details=False,
):
    """Shift of the data epoch's narrow lines relative to the template epoch's:
    the template epoch's narrow model (cores, narrow-line wing, [O III] wing)
    slid across the data epoch's broad-subtracted data over the pixels within
    ``win`` of the template's narrow cores (``labels`` restricts the cores, for
    example ("OIII5007c", "OIII4959c")), with a free scale and offset. Positive:
    the data epoch's narrow lines are redder."""
    grid = velocity_grid(vmax, step)
    d, comps = ep_templ["d"], ep_templ["comps"]
    lam0 = ep_data["lam0"]
    centres, used = [], []
    for lab, l0, an, vn, _sn, kind, ratio in comps:
        if kind != "narrow" or (labels is not None and lab not in labels):
            continue
        amp = d[an] * (ratio[1] if ratio else 1.0)
        if amp > 0:
            centres.append((l0 * (1.0 + d[vn] / C_KMS) / lam0 - 1.0) * C_KMS)
            used.append(lab)
    rec = dict(dv=np.nan, err=np.nan, npix=0, cores=used, flags=[])
    if not centres:
        rec["flags"].append("no_narrow_template")
        return rec
    v = ep_data["v"]
    sel = ep_data["ok"] & np.any([np.abs(v - c) <= win for c in centres], axis=0)
    rec["npix"] = int(sel.sum())
    if rec["npix"] < PAIR_MIN_PIX:
        rec["flags"].append("too_few_pixels")
        return rec
    # the data epoch's narrow-only data: y minus its own broad model
    ep_n = dict(ep_data, f=ep_data["y"] - ep_data["broad"], narrow=np.zeros_like(ep_data["y"]))
    sc = scan_chi2(ep_n, ep_templ, grid, sel, slope=False, narrow="none", template_kinds=NARROW_KINDS)
    rf = refine_minimum(grid, sc["chi2"])
    dof = max(rec["npix"] - sc["nlin"] - 1, 1)
    chi2_nu = rf["chi2min"] / dof if np.isfinite(rf["chi2min"]) else np.nan
    scaled = np.isfinite(rf["err"]) and np.isfinite(chi2_nu) and chi2_nu > 1
    err = rf["err"] * np.sqrt(chi2_nu) if scaled else rf["err"]
    rec.update(
        dv=float(rf["s"]) if not rf["at_bound"] else np.nan,
        err=float(err) if not rf["at_bound"] else np.nan,
        chi2_nu=float(chi2_nu),
        scale=float(sc["scale"][rf["k0"]]) if rf["k0"] >= 0 else np.nan,
        at_bound=bool(rf["at_bound"]),
        flags=["at_bound"] if rf["at_bound"] else [],
    )
    if details:
        rec["grid"], rec["chi2"], rec["sel"] = grid, sc["chi2"], sel
    return rec


# ----------------------------------------------------------------------------
# a pair
# ----------------------------------------------------------------------------
_DISQUALIFYING = ("at_bound", "scale_out_of_range", "flat_minimum", "too_few_pixels")


def measure_pair(
    res_a,
    res_b,
    line,
    inputs_a=None,
    inputs_b=None,
    grid=None,
    slope=True,
    narrow="none",
    kappa=CCF_NSUB_FRAC,
    kappa_prior=PAIR_PRIOR_FRAC,
    free_kinds=NARROW_KINDS,
    extra_sigma_kms=0.0,
    frame_veto_kms=FRAME_VETO_KMS,
    narrow_labels=None,
    frame=True,
    details=False,
):
    """The complete measurement of one line of a pair (A earlier, B later):
    both directions, the symmetric estimate, the narrow-line shift and the
    screens. Returns a flat dictionary.

    s_common, err : shift of B's broad profile relative to A's in the common
        input-redshift frame, (s - s')/2 of the two directions, with the error
        hypot(err_fwd, err_rev); NaN when the pair is not retained
    dv_narrow, err_narrow : shift of B's narrow lines relative to A's
    s_corrected, err_corrected : s_common - dv_narrow, a diagnostic
    dc50_sys, dc50_model : the difference of the two fits' own offsets, and of
        their model c(1/2), diagnostics
    dchi2_shape_fwd, dchi2_shape_rev, npix_fwd, npix_rev : the shape statistic
        of each direction (chi-square units) and its pixel count; the value per
        pixel is compared with PAIR_SHAPE_MAX by ``pair_record``
    retained : False when a direction is at bound, has a scale outside
        CCF_SCALE_RANGE, a flat minimum or too few pixels, when the product of
        the two scales is outside CCF_SCALE_PRODUCT_RANGE, or when the two
        directions disagree (``dir_inconsistent``); the direction values stay
    flags : the list of flags (``PAIR_FLAG_TEXT``), each direction's prefixed
        by fwd: or rev:, the narrow comparison's by narrow_fwd: or narrow_rev:

    ``slope`` and ``narrow="none"`` are the validated configuration (see the
    module docstring); ``frame=False`` skips the narrow-line comparison; with
    ``details`` the epochs, the chi-square curves and the coefficients are
    returned too, for figures."""
    ep_a = epoch_profile(res_a, line, inputs_a)
    ep_b = epoch_profile(res_b, line, inputs_b)
    rec = dict(line=line, version=ALGORITHM_VERSION, slope=slope, narrow=narrow, flags=[])
    if ep_a is None or ep_b is None:
        rec["flags"].append("line_not_fitted")
        return rec
    opts = dict(
        grid=grid,
        slope=slope,
        narrow=narrow,
        kappa=kappa,
        kappa_prior=kappa_prior,
        free_kinds=free_kinds,
        extra_sigma_kms=extra_sigma_kms,
        details=details,
    )
    fwd = measure_direction(ep_b, ep_a, **opts)  # template A, data B: B relative to A
    rev = measure_direction(ep_a, ep_b, **opts)  # template B, data A: A relative to B
    s, sp = fwd.get("s", np.nan), rev.get("s", np.nan)
    e, ep = fwd.get("err", np.nan), rev.get("err", np.nan)
    s_sym = 0.5 * (s - sp)
    e_sym = float(np.hypot(e, ep))
    mismatch = abs(s + sp)
    step = float(np.diff(fwd["grid"])[0]) if "grid" in fwd else PAIR_STEP_KMS
    if frame:
        nf = narrow_shift(ep_b, ep_a, labels=narrow_labels, details=details)
        nr = narrow_shift(ep_a, ep_b, labels=narrow_labels, details=details)
    else:
        nf = nr = dict(dv=np.nan, err=np.nan, cores=[], flags=[])
    dvn = 0.5 * (nf.get("dv", np.nan) - nr.get("dv", np.nan))
    evn = float(np.hypot(nf.get("err", np.nan), nr.get("err", np.nan)))
    flags = []
    for tag, r in (("fwd", fwd), ("rev", rev)):
        flags += [f"{tag}:{f}" for f in r.get("flags", [])]
    if np.isfinite(mismatch) and np.isfinite(e_sym) and mismatch > max(PAIR_DIR_NSIG * e_sym, 2.0 * step):
        flags.append("dir_inconsistent")
    prod = fwd.get("scale", np.nan) * rev.get("scale", np.nan)
    if np.isfinite(prod) and not (CCF_SCALE_PRODUCT_RANGE[0] <= prod <= CCF_SCALE_PRODUCT_RANGE[1]):
        flags.append("scale_product_out_of_range")
    flags += [f"narrow_fwd:{f}" for f in nf.get("flags", [])] + [
        f"narrow_rev:{f}" for f in nr.get("flags", [])
    ]
    if np.isfinite(dvn) and abs(dvn) > frame_veto_kms:
        flags.append("frame_offset_large")
    if np.isfinite(ep_a["z"]) and np.isfinite(ep_b["z"]) and abs(ep_a["z"] - ep_b["z"]) > 1e-6:
        flags.append("z_differs")
    if abs(ep_a["ebv"] - ep_b["ebv"]) > 1e-4:
        flags.append("ebv_differs")
    for tag, epx in (("a", ep_a), ("b", ep_b)):
        if epx["weights_consistent"] is False:
            flags.append(f"weights_inconsistent_{tag}")
        if epx["errors_source"].startswith("fit weights"):
            flags.append(f"errors_from_fit_weights_{tag}")
    retained = not any(
        f in ("dir_inconsistent", "scale_product_out_of_range")
        or (not f.startswith("narrow") and any(t in f for t in _DISQUALIFYING))
        for f in flags
    )
    if not retained:
        flags.append("not_retained")
    corrected = retained and np.isfinite(dvn)
    rec.update(
        retained=retained,
        s_common=float(s_sym) if retained else np.nan,
        err=float(e_sym) if retained else np.nan,
        s_fwd=float(s),
        err_fwd=float(e),
        s_rev=float(sp),
        err_rev=float(ep),
        dir_mismatch=float(mismatch),
        dv_narrow=float(dvn),
        err_narrow=float(evn),
        dv_narrow_fwd=float(nf.get("dv", np.nan)),
        dv_narrow_rev=float(nr.get("dv", np.nan)),
        narrow_cores=nf.get("cores", []),
        s_corrected=float(s_sym - dvn) if corrected else np.nan,
        err_corrected=float(np.hypot(e_sym, evn)) if corrected else np.nan,
        v_sys_diff=float(ep_b["v_sys"] - ep_a["v_sys"]),
        dc50_sys=float(ep_b["c50_sys"] - ep_a["c50_sys"]),
        dc50_model=float(ep_b["model_c50"] - ep_a["model_c50"]),
        scale_fwd=fwd.get("scale", np.nan),
        scale_rev=rev.get("scale", np.nan),
        chi2_nu_fwd=fwd.get("chi2_nu", np.nan),
        chi2_nu_rev=rev.get("chi2_nu", np.nan),
        dchi2_shape_fwd=fwd.get("dchi2_shape", np.nan),
        dchi2_shape_rev=rev.get("dchi2_shape", np.nan),
        chi2_own_nu_fwd=fwd.get("chi2_own_nu", np.nan),
        chi2_own_nu_rev=rev.get("chi2_own_nu", np.nan),
        npix_fwd=fwd.get("npix", 0),
        npix_rev=rev.get("npix", 0),
        s_alt_fwd=fwd.get("s_alt", np.nan),
        dchi2_alt_fwd=fwd.get("dchi2_alt", np.nan),
        s_alt_rev=rev.get("s_alt", np.nan),
        dchi2_alt_rev=rev.get("dchi2_alt", np.nan),
        fwhm_a=ep_a["model_fwhm"],
        fwhm_b=ep_b["model_fwhm"],
        snr_a=ep_a["snr"],
        snr_b=ep_b["snr"],
        peak_snr_a=ep_a["peak_snr"],
        peak_snr_b=ep_b["peak_snr"],
        cls_a=ep_a["cls"],
        cls_b=ep_b["cls"],
        z_a=ep_a["z"],
        z_b=ep_b["z"],
        ebv_a=ep_a["ebv"],
        ebv_b=ep_b["ebv"],
        errors_a=ep_a["errors_source"],
        errors_b=ep_b["errors_source"],
        flags=flags,
    )
    if details:
        rec["fwd"], rec["rev"], rec["narrow_fwd"], rec["narrow_rev"] = fwd, rev, nf, nr
        rec["epoch_a"], rec["epoch_b"] = ep_a, ep_b
    return rec


# ----------------------------------------------------------------------------
# the pair table
# ----------------------------------------------------------------------------
def systematic_term(line):
    """The calibrated systematic term of a line in km/s (``PAIR_SYS_KMS``), NaN
    for a line without one."""
    return float(PAIR_SYS_KMS.get(line, np.nan))


def shape_per_pixel(rec):
    """The shape statistic of a pair per pixel, the larger of the two directions."""
    vals = [
        rec.get("dchi2_shape_fwd", np.nan) / max(rec.get("npix_fwd", 1) or 1, 1),
        rec.get("dchi2_shape_rev", np.nan) / max(rec.get("npix_rev", 1) or 1, 1),
    ]
    return float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else np.nan


def pair_kind(kind_a, kind_b):
    """desi-desi, sdss-desi (either order), sdss-sdss, or the two kinds joined."""
    a, b = (str(kind_a or "other").lower(), str(kind_b or "other").lower())
    if a == b:
        return f"{a}-{b}"
    if {a, b} == {"sdss", "desi"}:
        return "sdss-desi"
    return f"{a}-{b}"


_SDSS_ID = re.compile(r"(?<!\d)(\d{3,5})-(\d{5})-(\d{3,4})(?!\d)")


def _plate_fiber(meta):
    """(plate, fiber) of an SDSS epoch from its metadata, or from an identifier of
    the form spec-PLATE-MJD-FIBER; None otherwise."""
    p, f = meta.get("plate"), meta.get("fiber")
    if p is not None and f is not None:
        return int(p), int(f)
    m = _SDSS_ID.search(str(meta.get("id", "")))
    return (int(m.group(1)), int(m.group(3))) if m else None


def independent(meta_a, meta_b):
    """Whether two epochs are independent observations: not the same spectrum,
    and not two SDSS spectra of the same plate and fibre (a later observation of
    a plate is a coaddition that includes the earlier exposures)."""
    if meta_a.get("id") == meta_b.get("id"):
        return False
    pa, pb = _plate_fiber(meta_a), _plate_fiber(meta_b)
    return not (pa and pb and pa == pb)


def pair_row(rec, res_a, meta_a=None, meta_b=None, role=""):
    """One row of the pair table from a ``measure_pair`` record: the identity and
    time of the epochs (``meta``: id, mjd, kind, targetid, and plate and fiber
    for SDSS), the pair kind, the systematic term and the total error, and the
    shape screen. Flags are joined with ';'."""
    meta_a, meta_b = dict(meta_a or {}), dict(meta_b or {})
    line = rec.get("line", "")
    row = {k: rec.get(k) for k in PAIR_COLUMNS}
    id_a, id_b = str(meta_a.get("id", "A")), str(meta_b.get("id", "B"))
    mjd_a, mjd_b = float(meta_a.get("mjd", np.nan)), float(meta_b.get("mjd", np.nan))
    z = float(res_a.get("z", np.nan))
    dt = mjd_b - mjd_a
    flags = list(rec.get("flags", []))
    if not independent(meta_a, meta_b):
        flags.append("dependent")
    sigma = systematic_term(line)
    err = float(rec.get("err", np.nan))
    if np.isfinite(sigma):
        err_total = float(np.hypot(err, sigma)) if np.isfinite(err) else np.nan
    else:
        err_total = err
        if "line_not_fitted" not in flags:
            flags.append("uncalibrated_line")
    shape = shape_per_pixel(rec)
    row.update(
        targetid=str(meta_a.get("targetid", meta_b.get("targetid", "")) or ""),
        pair=f"{id_a}__{id_b}",
        kind=pair_kind(meta_a.get("kind"), meta_b.get("kind")),
        line=line,
        role=role,
        id_a=id_a,
        id_b=id_b,
        mjd_a=mjd_a,
        mjd_b=mjd_b,
        dt_days=float(dt),
        dt_rest_yr=float(dt / 365.25 / (1.0 + z)) if np.isfinite(z) else np.nan,
        retained=bool(rec.get("retained", False)),
        sigma_sys=sigma,
        err_total=err_total,
        shape_max=shape,
        stable_shape=bool(np.isfinite(shape) and shape <= PAIR_SHAPE_MAX),
        npix_fwd=int(rec.get("npix_fwd", 0) or 0),
        npix_rev=int(rec.get("npix_rev", 0) or 0),
        flags=";".join(flags),
    )
    for k in PAIR_COLUMNS:
        if row[k] is None:
            row[k] = "" if k in ("cls_a", "cls_b", "errors_a", "errors_b") else np.nan
    return row


def pair_record(res_a, res_b, line, meta_a=None, meta_b=None, inputs_a=None, inputs_b=None, role="", **kw):
    """``measure_pair`` followed by ``pair_row``: one row of the pair table.
    Keyword arguments go to ``measure_pair``."""
    rec = measure_pair(res_a, res_b, line, inputs_a=inputs_a, inputs_b=inputs_b, **kw)
    return pair_row(rec, res_a, meta_a, meta_b, role)


def epoch_snr(res, lines=("Halpha", "Hbeta")):
    """The larger integrated broad-line S/N of an epoch over ``lines``, for the
    choice of the reference epoch; -inf when no line was measured."""
    vals = []
    for line in lines:
        v = ((res.get("meas") or {}).get(line) or {}).get("broad_flux_snr", np.nan)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return max(vals) if vals else -np.inf


def enumerate_pairs(epochs, lines=("Halpha", "Hbeta")):
    """The pairs of one object's epochs, each a dict with ``res`` (the fit) and the
    metadata of ``pair_record``: the reference epoch (the highest broad-line S/N
    over ``lines``, the earliest when equal) against every other epoch, and every
    consecutive pair in time; a pair that is both carries the role
    'reference+consecutive'. Pairs of dependent spectra are left out, and so are
    pairs with the reference when the reference is one of them. Returns (pairs,
    reference), each pair a dict(a, b, role) with a the earlier epoch."""
    eps = sorted(
        epochs,
        key=lambda e: float(e.get("mjd", np.nan)) if np.isfinite(float(e.get("mjd", np.nan))) else np.inf,
    )
    if not eps:
        return [], None
    ref = max(eps, key=lambda e: (epoch_snr(e["res"], lines), -eps.index(e)))
    pairs = {}

    def add(a, b, role):
        if not independent(a, b):
            return
        key = (a["id"], b["id"])
        if key in pairs:
            pairs[key]["role"] = "reference+consecutive"
        else:
            pairs[key] = dict(a=a, b=b, role=role)

    for e in eps:
        if e is ref:
            continue
        a, b = (e, ref) if eps.index(e) < eps.index(ref) else (ref, e)
        add(a, b, "reference")
    for a, b in zip(eps[:-1], eps[1:]):
        add(a, b, "consecutive")
    return list(pairs.values()), ref


__all__ = [
    "ALGORITHM_VERSION",
    "PAIR_COLUMNS",
    "PAIR_FLAG_TEXT",
    "velocity_grid",
    "statistical_ivar_on_fit_grid",
    "epoch_profile",
    "measure_direction",
    "narrow_shift",
    "measure_pair",
    "pair_row",
    "pair_record",
    "enumerate_pairs",
    "independent",
    "systematic_term",
    "shape_per_pixel",
    "pair_kind",
    "epoch_snr",
    "model_halfmax",
    "shifted_components",
    "model_at",
]
