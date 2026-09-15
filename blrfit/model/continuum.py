"""
The pseudo-continuum: power law, Fe II templates and host galaxy.

Everything under the emission lines that is not line emission is modelled here
and subtracted before the line complexes are fitted. The recipe is that of the
SDSS quasar catalogues (Shen et al. 2011; PyQSOFit): a power law and the
optical and ultraviolet Fe II templates of Boroson & Green (1992) and
Vestergaard & Wilkes (2001) fitted in line-free windows, with outliers clipped
once. The host galaxy is added to the same fit as a non-negative combination of
the first galaxy eigenspectra of Yip et al. (2004), anchored by the 4000 A
break and the stellar absorption features, and kept only if it contributes at
least 10 per cent of the 4200-5000 A flux.

Two departures from the common recipe, both forced by the host-dominated
spectra that make up most of a DESI broad-line sample:

* The host is fitted jointly with the power law and Fe II, not as a separate
  galaxy + quasar eigenspectrum decomposition. The quasar eigenspectra carry
  fixed emission-line shapes, which is exactly wrong for the offset and
  double-peaked profiles this package is about, and an unconstrained
  decomposition of a noisy spectrum readily produces a negative "galaxy". The
  number of eigenspectra is therefore stepped down (5, 3, 2, 1, 0) until the
  host is non-negative.
* The host contributes continuum only underneath the emission lines
  (``host_continuum_only``). The eigenspectra are principal components of real
  galaxy spectra and contain emission lines of their own; fitted to line-free
  pixels, the strengths of those template lines are unconstrained, and
  subtracting the full host removed narrow-line flux of arbitrary strength and
  displaced the systemic velocity by up to 450 km/s in strongly star-forming
  hosts.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from ..constants import (C_KMS, S2F, TEMPLATE_DIR, LAM, COMPLEX_WINDOW, CONTI_WINDOWS,
                         PL_PIVOT, PL_ALPHA_MIN, PL_ALPHA_MAX, FE_FWHM_MIN, FE_FWHM_MAX,
                         FE_SHIFT_MAX, FE_INTRINSIC_FWHM, N_GAL_MAX, MIN_HOST_FRAC,
                         HOST_CONTINUUM_ONLY, HOST_LINE_HALFWIDTH_KMS, CLIP_LO, CLIP_HI,
                         MAX_NFEV_CONTI, MAX_NFEV_CONTI_HOST, FE_UV_FWHM_FIXED_KMS,
                         FE_UV_FREE_MIN_PIXELS)
from .params import ParamSet


# ----------------------------------------------------------------------------
# Fe II templates (PyQSOFit convention: columns log10 wavelength, flux)
# ----------------------------------------------------------------------------
def _solver_record(sol):
    return dict(success=bool(sol.success and np.all(np.isfinite(sol.x))
                             and np.all(np.isfinite(sol.fun))),
                status=int(sol.status), message=str(sol.message), nfev=int(sol.nfev),
                optimality=float(sol.optimality), objective=float(np.sum(sol.fun**2)))


def _at_bound(sol, ps, rtol=1e-6):
    """Names of the free parameters that ended on a bound: those the solver
    reports as active plus those within ``rtol`` of a bound (relative to the
    bound, absolute for bounds below unity, as the solver measures it). A
    width or a shift on a bound is not a measurement; the list is stored so
    that downstream code can flag it."""
    lb, ub = ps.bounds()
    x = np.asarray(sol.x, float)
    hit = np.asarray(sol.active_mask) != 0
    hit |= (x - lb) <= rtol * np.maximum(1.0, np.abs(lb))
    hit |= (ub - x) <= rtol * np.maximum(1.0, np.abs(ub))
    return [n for n, h in zip(ps.free_names, hit) if h]


def _fe_widths(d):
    return {k: float(d[k]) for k in ("feop_fwhm", "feuv_fwhm") if k in d}


def _uv_window_pixels(wave, good, inwin):
    """Good pixels of the fitting windows that cover the ultraviolet Fe II template."""
    return int(np.sum(good & inwin & (wave > 2200) & (wave < 3090)))


def _take_fallback(info, cinfo):
    """Carry the diagnostics of a PL+Fe fallback fit into the host-fit info."""
    for k in ("solver", "at_bound", "fe_widths", "feuv_fwhm_fixed", "n_pix_uv"):
        info[k] = cinfo[k]


class FeTemplate:
    """One Fe II template with Gaussian broadening in log wavelength.

    The I Zw 1 templates have an intrinsic width of 900 km/s (FWHM); a requested
    FWHM is reached by convolving with sqrt(FWHM^2 - 900^2). Evaluate at the
    requested width: quantizing widths makes finite-difference optimizer
    derivatives vanish and can freeze the width at its initial value.
    """

    def __init__(self, path, wmin, wmax, intrinsic_fwhm=FE_INTRINSIC_FWHM):
        d = np.loadtxt(path)
        self.logw = d[:, 0]
        self.wave = 10.0 ** d[:, 0]
        self.flux = d[:, 1] * 1e15                 # bring the template to order unity
        self.flux[~np.isfinite(self.flux)] = 0.0
        self.wmin, self.wmax = wmin, wmax
        self.intrinsic = intrinsic_fwhm
        # template pixel in km/s (the grid is uniform in log wavelength)
        self.pix_kms = np.median(np.diff(self.logw)) * np.log(10.0) * C_KMS

    def broadened(self, fwhm_kms):
        f = max(float(fwhm_kms), self.intrinsic + 10.0)
        sig_conv = np.sqrt(max(f**2 - self.intrinsic**2, 10.0**2)) / S2F
        out = gaussian_filter1d(self.flux, sig_conv / self.pix_kms, mode="nearest")
        return out

    def __call__(self, wave_rest, norm, fwhm_kms, shift):
        """Template flux on ``wave_rest``; ``shift`` is fractional (dlambda / lambda)."""
        y = np.zeros_like(wave_rest, dtype=float)
        if norm <= 0:
            return y
        w = wave_rest * (1.0 + shift)
        m = (w > self.wmin) & (w < self.wmax)
        if not m.any():
            return y
        y[m] = norm * np.interp(w[m], self.wave, self.broadened(fwhm_kms))
        return y


_FE_OP = _FE_UV = None


def fe_templates():
    """The optical (3686-7484 A) and ultraviolet (1200-3500 A) Fe II templates, loaded once."""
    global _FE_OP, _FE_UV
    if _FE_OP is None:
        _FE_OP = FeTemplate(TEMPLATE_DIR / "fe_optical.txt", 3686.0, 7484.0)
        _FE_UV = FeTemplate(TEMPLATE_DIR / "fe_uv.txt", 1200.0, 3500.0)
    return _FE_OP, _FE_UV


# ----------------------------------------------------------------------------
# Host-galaxy eigenspectra (Yip et al. 2004)
# ----------------------------------------------------------------------------
_PCA = None


def pca_templates():
    """Galaxy and quasar eigenspectra as dict(gw, gp, qw, qp), loaded once.
    Only the galaxy set enters the model; the quasar set is kept for reference."""
    global _PCA
    if _PCA is None:
        from astropy.io import fits
        g = fits.open(TEMPLATE_DIR / "gal_eigenspec_Yip2004.fits")[1].data
        q = fits.open(TEMPLATE_DIR / "qso_eigenspec_Yip2004_global.fits")[1].data
        gw = np.asarray(g["WAVE"]).ravel().astype(float)
        gp = np.asarray(g["PCA"])[0].astype(float).reshape(-1, gw.size)
        qw = np.asarray(q["WAVE"]).ravel().astype(float)
        qp = np.asarray(q["PCA"])[0].astype(float).reshape(-1, qw.size)
        _PCA = dict(gw=gw, gp=gp, qw=qw, qp=qp)
    return _PCA


def line_mask(wave_rest, broad_halfwidth_kms=9000.0, narrow_halfwidth_kms=900.0):
    """True where emission lines live: the three broad complexes plus the strong
    narrow lines. These pixels are kept out of the host fit."""
    m = np.zeros_like(wave_rest, bool)
    for lo, hi in COMPLEX_WINDOW.values():
        m |= (wave_rest > lo) & (wave_rest < hi)
    for lam in (3728.48, 3869.86, 3426.85, 4102.89, 4341.68, 6302.05, 6365.54,
                LAM["OIII5007"], LAM["OIII4959"], LAM["SII6716"], LAM["SII6731"]):
        m |= np.abs(wave_rest / lam - 1.0) * C_KMS < narrow_halfwidth_kms
    return m


def narrow_line_mask(wave_rest, halfwidth_kms=HOST_LINE_HALFWIDTH_KMS):
    """True within +/- halfwidth of the catalogued narrow lines and of the narrow
    cores of the Balmer and Mg II lines. Narrower than ``line_mask``: it does not
    cover the broad-line complexes, only the narrow features."""
    m = np.zeros_like(wave_rest, bool)
    for lam in (3728.48, 3869.86, 3426.85, 4102.89, 4341.68, 4364.44, 4687.02, 5877.25,
                6302.05, 6365.54, 7137.77, 7321.0, 7331.7, 9071.1, 9533.2,
                LAM["Hbeta"], LAM["Halpha"], LAM["MgII"], LAM["OIII5007"], LAM["OIII4959"],
                LAM["NII6548"], LAM["NII6584"], LAM["SII6716"], LAM["SII6731"]):
        m |= np.abs(wave_rest / lam - 1.0) * C_KMS < halfwidth_kms
    return m


def host_continuum_only(wave_rest, host):
    """Replace the host model inside the narrow-line masks by a linear
    interpolation from the surrounding continuum.

    The eigenspectra contain emission lines whose strengths the line-free fit
    does not constrain; the host may only contribute stellar continuum under
    the lines, the lines themselves belong to the emission-line fit. The
    stellar Balmer absorption within +/-900 km/s of Hbeta and Halpha is
    interpolated over as well; that affects the narrow-line fluxes slightly and
    the broad-line velocities negligibly.
    """
    m = narrow_line_mask(wave_rest)
    if not m.any() or (~m).sum() < 10:
        return host
    out = host.copy()
    out[m] = np.interp(wave_rest[m], wave_rest[~m], host[~m])
    return out


# ----------------------------------------------------------------------------
# Power law + Fe II
# ----------------------------------------------------------------------------
def conti_model(wave, d, fe_op, fe_uv):
    """Power law plus whichever Fe II templates are in the parameter dictionary."""
    pl = d["pl_norm"] * (wave / PL_PIVOT) ** d["pl_alpha"]
    y = pl.copy()
    if "feop_norm" in d:
        y += fe_op(wave, d["feop_norm"], d["feop_fwhm"], d["feop_shift"])
    if "feuv_norm" in d:
        y += fe_uv(wave, d["feuv_norm"], d["feuv_fwhm"], d["feuv_shift"])
    return y


def _add_pl_fe(ps, fref, fit_fe, cov_op, cov_uv, pl_start, feuv_fixed=False):
    """Power-law and Fe II parameters in the frozen order.

    With ``feuv_fixed`` the ultraviolet width is held at FE_UV_FWHM_FIXED_KMS
    instead of being fitted: where the spectrum covers fewer than
    FE_UV_FREE_MIN_PIXELS pixels of the ultraviolet windows the width has no
    leverage and a free one runs to a bound.
    """
    ps.add("pl_norm", pl_start, 0.0, 1e4 * fref)
    ps.add("pl_alpha", -1.5, PL_ALPHA_MIN, PL_ALPHA_MAX)
    if fit_fe and cov_op:
        ps.add("feop_norm", 0.1 * fref, 0.0, 1e3 * fref)
        ps.add("feop_fwhm", 3000.0, FE_FWHM_MIN, FE_FWHM_MAX)
        ps.add("feop_shift", 0.0, -FE_SHIFT_MAX, FE_SHIFT_MAX)
    if fit_fe and cov_uv:
        ps.add("feuv_norm", 0.1 * fref, 0.0, 1e3 * fref)
        if feuv_fixed:
            ps.add("feuv_fwhm", FE_UV_FWHM_FIXED_KMS, FE_FWHM_MIN, FE_FWHM_MAX, fixed=True)
        else:
            ps.add("feuv_fwhm", 3000.0, FE_FWHM_MIN, FE_FWHM_MAX)
        ps.add("feuv_shift", 0.0, -FE_SHIFT_MAX, FE_SHIFT_MAX)


def fit_continuum(wave, flux, ivar, windows=CONTI_WINDOWS, fit_fe=True, clip=True):
    """Power law + Fe II in the line-free windows, no host.

    Used directly when no host is wanted, as the fallback of the joint fit, and
    for every Monte Carlo realisation (where the host is held fixed).
    Returns (parameter dict, model on ``wave``, info).
    """
    fe_op, fe_uv = fe_templates()
    good = np.isfinite(flux) & (ivar > 0)
    inwin = np.zeros_like(good)
    for lo, hi in windows:
        inwin |= (wave >= lo) & (wave <= hi)
    m = good & inwin
    info = dict(n_pix=int(m.sum()), fe_op=False, fe_uv=False, fallback=False)
    ps = ParamSet()
    fref = np.nanmedian(flux[m]) if m.sum() else np.nanmedian(flux[good])
    fref = max(fref, 1e-3)
    # the Fe II templates enter only where the spectrum covers them
    cov_op = np.sum(good & (wave > 4435) & (wave < 5535)) > 40
    cov_uv = np.sum(good & (wave > 2200) & (wave < 3090)) > 40
    n_uv = _uv_window_pixels(wave, good, inwin)
    feuv_fixed = bool(fit_fe and cov_uv and n_uv < FE_UV_FREE_MIN_PIXELS)
    _add_pl_fe(ps, fref, fit_fe, cov_op, cov_uv, pl_start=fref, feuv_fixed=feuv_fixed)
    info["fe_op"] = bool(fit_fe and cov_op); info["fe_uv"] = bool(fit_fe and cov_uv)
    info["feuv_fwhm_fixed"] = feuv_fixed; info["n_pix_uv"] = n_uv
    if m.sum() < 40:
        # too few window pixels: a power law to everything outside the complexes
        info["fallback"] = True
        m = good.copy()
        for lo, hi in COMPLEX_WINDOW.values():
            m &= ~((wave > lo) & (wave < hi))
        for k in list(ps.names):
            if k.startswith("fe"):
                ps.fixed[k] = 0.0 if k.endswith("norm") else ps.val[ps.names.index(k)]

    w = np.sqrt(ivar[m]); x = wave[m]; y = flux[m]

    def resid(p):
        return (y - conti_model(x, ps.full(p), fe_op, fe_uv)) * w

    sol = least_squares(resid, ps.p0(), bounds=ps.bounds(), x_scale="jac", max_nfev=MAX_NFEV_CONTI)
    if clip and m.sum() > 60:
        # one round of outlier clipping (absorption features, residual lines), as Shen et al. 2011
        r = resid(sol.x)
        keep = (r > CLIP_LO) & (r < CLIP_HI)
        if keep.sum() > 40 and keep.sum() < len(r):
            x, y, w = x[keep], y[keep], w[keep]

            def resid2(p):
                return (y - conti_model(x, ps.full(p), fe_op, fe_uv)) * w

            sol = least_squares(resid2, sol.x, bounds=ps.bounds(), x_scale="jac", max_nfev=MAX_NFEV_CONTI)
            info["n_pix"] = int(keep.sum())
    d = ps.full(sol.x)
    ps.set_values(d)
    model = conti_model(wave, d, fe_op, fe_uv)
    info["chi2"] = float(np.sum(sol.fun**2)); info["ps"] = ps
    info['solver'] = _solver_record(sol)
    info["at_bound"] = _at_bound(sol, ps); info["fe_widths"] = _fe_widths(d)
    return d, model, info


# ----------------------------------------------------------------------------
# Joint host + power law + Fe II (the default path)
# ----------------------------------------------------------------------------
def fit_continuum_host(wave, flux, ivar, fit_fe=True, n_gal_max=N_GAL_MAX,
                       min_host_frac=MIN_HOST_FRAC, windows=CONTI_WINDOWS):
    """One coherent pseudo-continuum: sum_i g_i E_i(lambda) + power law + Fe II.

    Pixels used: the Shen et al. (2011) windows everywhere, plus every pixel
    inside the galaxy-template range that is outside the emission-line masks,
    so that the 4000 A break and the stellar absorption anchor the host. The
    eigenspectrum count is stepped down (n_gal_max, 3, 2, 1, 0) until the host
    is non-negative; the host is kept only above ``min_host_frac`` of the
    4200-5000 A flux, otherwise the continuum is refitted without it.

    Returns (parameter dict, total continuum incl. host, host model, info).
    """
    fe_op, fe_uv = fe_templates(); P = pca_templates()
    good = np.isfinite(flux) & (ivar > 0)
    inhost = (wave > P["gw"].min() + 2) & (wave < P["gw"].max() - 2)
    inwin = np.zeros_like(good)
    for lo, hi in windows:
        inwin |= (wave >= lo) & (wave <= hi)
    use0 = good & (inwin | (inhost & ~line_mask(wave)))
    info = dict(applied=False, host_frac_4200_5000=np.nan, n_gal=0, n_negative_pix=0,
                reason="", n_pix=int(use0.sum()), fe_op=False, fe_uv=False, solver_attempts=[])
    if use0.sum() < 40:
        d, model, cinfo = fit_continuum(wave, flux, ivar, fit_fe=fit_fe)
        info.update(reason="too few line-free pixels; PL+Fe only", n_pix=cinfo["n_pix"])
        _take_fallback(info, cinfo)
        return d, model, np.zeros_like(flux), info

    Gfull = [np.where(inhost, np.interp(wave, P["gw"], P["gp"][i], left=0, right=0), 0.0)
             for i in range(n_gal_max)]
    fref = max(float(np.nanmedian(flux[use0])), 1e-3)
    cov_op = np.sum(good & (wave > 4435) & (wave < 5535)) > 40
    cov_uv = np.sum(good & (wave > 2200) & (wave < 3090)) > 40
    # the galaxy templates start at 3450 A, so inside the ultraviolet windows
    # the joint fit uses the same pixels as the plain fit
    n_uv = _uv_window_pixels(wave, good, inwin)
    feuv_fixed = bool(fit_fe and cov_uv and n_uv < FE_UV_FREE_MIN_PIXELS)
    info["feuv_fwhm_fixed"] = feuv_fixed; info["n_pix_uv"] = n_uv
    gscale = fref / max(float(np.nanmedian(Gfull[0][inhost])) if inhost.sum() else 1.0, 1e-6)

    def make_ps(ng):
        ps = ParamSet()
        _add_pl_fe(ps, fref, fit_fe, cov_op, cov_uv, pl_start=0.7 * fref, feuv_fixed=feuv_fixed)
        for i in range(ng):
            # eigenspectrum 0 is the mean galaxy: its coefficient must be non-negative
            ps.add(f"gal{i}", 0.3 * gscale if i == 0 else 0.0,
                   0.0 if i == 0 else -50.0 * gscale, 50.0 * gscale)
        return ps

    def host_of(d, ng):
        h = np.zeros_like(wave)
        for i in range(ng):
            h += d[f"gal{i}"] * Gfull[i]
        return h

    best = None
    for ng in [n for n in (n_gal_max, 3, 2, 1, 0) if n <= n_gal_max]:
        ps = make_ps(ng)
        use = use0 if ng > 0 else (good & inwin if (good & inwin).sum() >= 40 else use0)
        y = flux[use]; w = np.sqrt(ivar[use])
        idx = np.where(use)[0]

        def resid(p, idx=idx, y=y, w=w, ng=ng):
            d = ps.full(p)
            m = conti_model(wave[idx], d, fe_op, fe_uv)
            for i in range(ng):
                m = m + d[f"gal{i}"] * Gfull[i][idx]
            return (y - m) * w

        try:
            sol = least_squares(resid, ps.p0(), bounds=ps.bounds(), x_scale="jac", max_nfev=MAX_NFEV_CONTI_HOST)
            r = resid(sol.x)
            keep = (r > CLIP_LO) & (r < CLIP_HI)
            if keep.sum() > 40 and keep.sum() < len(r):
                idx2, y2, w2 = idx[keep], y[keep], w[keep]

                def resid2(p, idx=idx2, y=y2, w=w2, ng=ng):
                    d = ps.full(p)
                    m = conti_model(wave[idx], d, fe_op, fe_uv)
                    for i in range(ng):
                        m = m + d[f"gal{i}"] * Gfull[i][idx]
                    return (y - m) * w

                sol = least_squares(resid2, sol.x, bounds=ps.bounds(), x_scale="jac", max_nfev=MAX_NFEV_CONTI_HOST)
        except Exception as exc:
            info['solver_attempts'].append(dict(n_gal=ng, success=False, status='exception',
                                               message=f'{type(exc).__name__}: {exc}'))
            continue
        # an attempt that stopped short of convergence is recorded, not
        # discarded: the solver record and the at-bound list say so
        solver = _solver_record(sol)
        info['solver_attempts'].append(dict(n_gal=ng, **solver))
        d = ps.full(sol.x); ps.set_values(d)
        host = host_of(d, ng)
        n_neg = int(np.sum(host[inhost] < -1e-3 * max(np.nanmax(np.abs(host)), 1e-9)))
        sel = inhost & (wave > 4200) & (wave < 5000) & good
        frac = float(np.sum(host[sel]) / max(np.sum(flux[sel]), 1e-30)) if sel.sum() > 20 else np.nan
        cand = dict(d=d, ps=ps, ng=ng, host=host, n_neg=n_neg, frac=frac,
                    chi2=float(np.sum(sol.fun ** 2)), solver=solver, at_bound=_at_bound(sol, ps))
        if ng == 0 or n_neg <= max(50, 0.02 * inhost.sum()):
            best = cand
            break
    if best is None:
        d, model, cinfo = fit_continuum(wave, flux, ivar, fit_fe=fit_fe)
        info.update(reason="joint fit failed; PL+Fe only")
        _take_fallback(info, cinfo)
        return d, model, np.zeros_like(flux), info

    d, ng, host, frac = best["d"], best["ng"], best["host"], best["frac"]
    info.update(n_gal=ng, n_negative_pix=best["n_neg"], host_frac_4200_5000=frac,
                fe_op="feop_norm" in d, fe_uv="feuv_norm" in d, chi2=best["chi2"], solver=best['solver'],
                at_bound=best["at_bound"], fe_widths=_fe_widths(d))
    if ng > 0 and (not np.isfinite(frac) or frac < min_host_frac):
        # host too weak to trust (Shen et al. 2011): refit without it
        info["reason"] = f"host fraction {frac:.2f} < {min_host_frac}; PL+Fe only"
        d2, model2, cinfo = fit_continuum(wave, flux, ivar, fit_fe=fit_fe)
        _take_fallback(info, cinfo)
        return d2, model2, np.zeros_like(flux), info
    host = np.clip(host, 0, None)
    if HOST_CONTINUUM_ONLY:
        host = host_continuum_only(wave, host)
    total = conti_model(wave, d, fe_op, fe_uv) + host
    info["applied"] = ng > 0
    if ng == 0:
        info["reason"] = "no host component needed"
    return d, total, host, info
