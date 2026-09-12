"""
Fitting one line complex: assembly of the narrow and broad components, the
penalty terms, the multi-start least-squares solution and the choice of the
number of broad components.

The model of a complex is evaluated on the continuum-subtracted rest-frame
spectrum inside its window (``COMPLEX_WINDOW``). The weighted residual vector
is extended by penalty terms that act as Gaussian priors or one-sided hinges
(see ``narrow.py`` and ``broad.py``); their order is part of the frozen model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..constants import (C_KMS, LAM, COMPLEX_WINDOW, COMPLEX_LINE, COMPLEX_PREFIX,
                         SIG_BROAD_MIN, BROAD_WIDTH_SLOPE, V_NARROW_MAX, MIN_COMPLEX_PIXELS,
                         MIN_COMPLEX_COVER, MAX_NFEV_COMPLEX, MAX_BROAD, DBIC)
from .params import ParamSet, gauss_lam
from .narrow import (add_halpha_narrow, add_hbeta_narrow, add_mgii_narrow,
                     systemic_prior_penalty, nlr_wing_penalty, sii_soft_tie_penalty,
                     oiii_order_penalty, has_oiii_ordering)
from .broad import (add_broad_block, add_mgii_doublet_block, velocity_bounds, core_covered,
                    covered_velocity_bounds, broad_parameter_pairs, width_offset_penalty,
                    first_component_starts, select_by_bic)


def build_complex(name, wave, fsub, n_broad=1, v_sys_prior=None, sig_sys_prior=None,
                  oiii_wing=True, heii=True, mgii_narrow=True, mgii_doublet=False,
                  sig_broad_min=SIG_BROAD_MIN, nw_prior=None, v_bounds=None):
    """Parameter set and component list of one complex (narrow lines first, then
    the broad block). Components are ``(label, lam0, amp_name, v_name, sig_name,
    kind, ratio_to)``; see ``narrow.py``."""
    ps = ParamSet(); comps = []
    amax = max(float(np.nanmax(np.abs(fsub))) * 5.0, 1e-2)
    vb_lo, vb_hi = velocity_bounds(v_bounds)
    if name == "Halpha":
        add_halpha_narrow(ps, comps, wave, fsub, amax, v_sys_prior, sig_sys_prior)
        add_broad_block(ps, comps, wave, fsub, LAM["Halpha"], "Ha", n_broad, amax, vb_lo, vb_hi,
                        sig_broad_min)
    elif name == "Hbeta":
        add_hbeta_narrow(ps, comps, wave, fsub, amax, v_sys_prior, sig_sys_prior,
                         oiii_wing=oiii_wing, heii=heii, nw_prior=nw_prior)
        add_broad_block(ps, comps, wave, fsub, LAM["Hbeta"], "Hb", n_broad, amax, vb_lo, vb_hi,
                        sig_broad_min)
    elif name == "MgII":
        if mgii_narrow:
            add_mgii_narrow(ps, comps, wave, fsub, amax)
        if mgii_doublet:
            add_mgii_doublet_block(ps, comps, wave, fsub, n_broad, amax, vb_lo, vb_hi, sig_broad_min)
        else:
            add_broad_block(ps, comps, wave, fsub, LAM["MgII"], "Mg", n_broad, amax, vb_lo, vb_hi,
                            sig_broad_min)
    else:
        raise ValueError(name)
    return ps, comps


def eval_components(wave, d, comps, kinds=None):
    """Sum of the components of the given kinds ('broad', 'narrow', 'nwing',
    'wing'; all if ``kinds`` is None) for the parameter dictionary ``d``."""
    y = np.zeros_like(wave, dtype=float)
    for lab, lam0, an, vn, sn, kind, ratio in comps:
        if kinds is not None and kind not in kinds:
            continue
        A = d[an] * (ratio[1] if ratio else 1.0)
        y += gauss_lam(wave, A, lam0, d[vn], d[sn])
    return y


def fit_complex(name, wave, fsub, ivar, n_broad, **kw):
    """Fit one complex with ``n_broad`` broad Gaussians, from several starting
    velocities of the first broad component (and optionally of the narrow
    group). Returns the fit dictionary of the lowest-chi-square solution, or
    None if the window is not usable (too few pixels, or the line core not
    covered on both sides).

    Keyword arguments beyond those of ``build_complex``: ``n_v_starts`` (list of
    starting velocities of the narrow group, Halpha) and ``n_v_prior``
    ((velocity, width) of a Gaussian prior on the narrow-group velocity).
    """
    n_v_starts = kw.pop("n_v_starts", None) or [None]
    n_v_prior = kw.pop("n_v_prior", None)
    lo, hi = COMPLEX_WINDOW[name]
    m = (wave > lo) & (wave < hi) & np.isfinite(fsub) & (ivar > 0)
    cover = m.sum() / max(np.sum((wave > lo) & (wave < hi)), 1)
    if m.sum() < MIN_COMPLEX_PIXELS or cover < MIN_COMPLEX_COVER:
        return None
    x, y, w = wave[m], fsub[m], np.sqrt(ivar[m])
    lam_c = LAM[COMPLEX_LINE[name]]
    vcov = (x / lam_c - 1.0) * C_KMS
    v_lo, v_hi = float(vcov.min()), float(vcov.max())
    if not core_covered(v_lo, v_hi):
        return None
    kw = dict(kw); kw["v_bounds"] = covered_velocity_bounds(v_lo, v_hi)
    ps, comps = build_complex(name, x, y, n_broad=n_broad, **kw)
    prefix = COMPLEX_PREFIX[name]

    soft = (name == "Halpha" and "s2_sig" in ps.names)
    order = has_oiii_ordering(name, ps)
    nprior = (n_v_prior is not None and "n_v" in ps.names and "n_v" not in ps.fixed
              and "n_v" not in ps.tie)
    nwing = (name == "Halpha" and "nw_sig" in ps.names and "nw_sig" not in ps.fixed)
    broad_vs = broad_parameter_pairs(comps)

    def resid(p):
        d = ps.full(p)
        r = (y - eval_components(x, d, comps)) * w
        if BROAD_WIDTH_SLOPE > 0 and broad_vs:   # far broad components must be broad
            r = np.concatenate([r, width_offset_penalty(d, broad_vs)])
        if nprior:   # narrow-group velocity toward the [O III] pre-fit
            r = np.concatenate([r, systemic_prior_penalty(d, n_v_prior)])
        if nwing:    # narrow-line-region wing
            r = np.concatenate([r, nlr_wing_penalty(d)])
        if soft:     # [S II] soft tie
            r = np.concatenate([r, sii_soft_tie_penalty(d)])
        if order:    # [O III] core / wing ordering
            r = np.concatenate([r, oiii_order_penalty(d)])
        return r

    vlo_b, vhi_b = kw["v_bounds"]
    starts = first_component_starts(vlo_b, vhi_b, n_broad)
    best = None
    for nv0 in n_v_starts:
        for v0 in starts:
            vals = {f"{prefix}_b0_v": v0}
            if nv0 is not None and "n_v" in ps.names and "n_v" not in ps.fixed and "n_v" not in ps.tie:
                vals["n_v"] = float(np.clip(nv0, -V_NARROW_MAX + 1, V_NARROW_MAX - 1))
                if "s2_v" in ps.names:
                    vals["s2_v"] = vals["n_v"]
            ps.set_values(vals)
            try:
                sol = least_squares(resid, ps.p0(), bounds=ps.bounds(), x_scale="jac",
                                    max_nfev=MAX_NFEV_COMPLEX, loss="linear")
            except Exception:
                continue
            c2 = float(np.sum(sol.fun**2))
            if best is None or c2 < best[0]:
                best = (c2, sol.x.copy())
    if best is None:
        return None
    chi2, pbest = best
    d = ps.full(pbest)
    ps.set_values(d)
    n = int(m.sum()); k = len(ps.free_names)
    return dict(name=name, ps=ps, comps=comps, d=d, chi2=chi2, npix=n, nfree=k,
                bic=chi2 + k * np.log(n), n_broad=n_broad, window=(lo, hi),
                x=x, y=y, w=w, v_cover=(v_lo, v_hi))


def fit_complex_select(name, wave, fsub, ivar, max_broad=MAX_BROAD, dbic=DBIC, **kw):
    """Fit with 1..max_broad broad components and keep the simplest model within
    ``dbic`` of the best BIC. Returns (chosen fit, list of all fits)."""
    fits = []
    for nb in range(1, max_broad + 1):
        r = fit_complex(name, wave, fsub, ivar, nb, **kw)
        if r is not None:
            fits.append(r)
    if not fits:
        return None, []
    bics = np.array([f["bic"] for f in fits])
    r = fits[select_by_bic(bics, dbic)]
    r["all_bic"] = [float(b) for b in bics]
    r["all_chi2"] = [float(f["chi2"]) for f in fits]
    return r, fits
