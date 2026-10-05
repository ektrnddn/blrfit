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

Each continuum fit is started from several points (``CONTI_START_ALPHAS`` and,
with a host, ``CONTI_START_HOST_SCALES``), because the power-law slope and the
host amplitude are degenerate in host-rich spectra and one start can stop in a
local minimum. The starts are compared on the same pixels before the outlier
clip and the lowest chi-square is kept; ``continuum_info['starts']`` records
every start and ``start_selected`` the one kept (0 is the start used up to
version 0.2).

Two guards of the continuum fit are recorded with every result. The host
fraction is undetermined, and the host is not fitted, when the 4200-5000 A
window that defines it carries no signal (``host_window_statistics``;
``host_info['host_undetermined']``). The ultraviolet Fe II width follows one
of three policies (``fe_uv_width_policy``: held where the window is short,
always free, or free and refitted at a fallback width when it ends on a
bound), and the bound state of each Fe II width is kept apart from a zero
Fe II norm (``fe_width_states``; ``continuum_info['fe_width_state']``,
``fe_norm_zero``, ``at_bound_widths``).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from ..constants import (
    C_KMS,
    S2F,
    TEMPLATE_DIR,
    LAM,
    COMPLEX_WINDOW,
    CONTI_WINDOWS,
    PL_PIVOT,
    PL_ALPHA_MIN,
    PL_ALPHA_MAX,
    FE_FWHM_MIN,
    FE_FWHM_MAX,
    FE_SHIFT_MAX,
    FE_INTRINSIC_FWHM,
    N_GAL_MAX,
    MIN_HOST_FRAC,
    HOST_CONTINUUM_ONLY,
    HOST_LINE_HALFWIDTH_KMS,
    CLIP_LO,
    CLIP_HI,
    MAX_NFEV_CONTI,
    MAX_NFEV_CONTI_HOST,
    FE_UV_FWHM_FIXED_KMS,
    FE_UV_FREE_MIN_PIXELS,
    FE_UV_WIDTH_POLICY,
    HOST_GUARD_MIN_SNR,
    HOST_GUARD_MAX_MASKED_FRAC,
    CONTI_START_ALPHAS,
    CONTI_START_HOST_SCALES,
    CONTI_START_DCHI2,
)
from .params import ParamSet

# An Fe II width counts as at a bound when it lies within this fraction of the
# bound's span, or this many km/s, of either end (whichever is larger: 88 km/s
# for the 1200-10000 km/s range), or when the solver's active set holds it.
# The solver's own criterion (``_at_bound``, 1e-6 relative) misses widths that
# stopped a few tens of km/s short of a bound on a flat chi-square.
FE_WIDTH_BOUND_TOL_FRAC = 0.01
FE_WIDTH_BOUND_TOL_KMS = 10.0
# An Fe II norm at or below this (template units) multiplies nothing: the
# width then carries no information and is not counted as at a bound.
FE_NORM_ZERO = 1e-6
FE_UV_WIDTH_POLICIES = ("A", "B", "C")
FE_WIDTH_STATES = ("not_covered", "fixed", "unconstrained", "solver_active", "near_bound", "interior")
FE_WIDTH_NORM = {"feop_fwhm": "feop_norm", "feuv_fwhm": "feuv_norm"}
HOST_WINDOW = (4200.0, 5000.0)  # the window that defines the host fraction (Shen et al. 2011)


# ----------------------------------------------------------------------------
# Fe II templates (PyQSOFit convention: columns log10 wavelength, flux)
# ----------------------------------------------------------------------------
def _solver_record(sol):
    return dict(
        success=bool(sol.success and np.all(np.isfinite(sol.x)) and np.all(np.isfinite(sol.fun))),
        status=int(sol.status),
        message=str(sol.message),
        nfev=int(sol.nfev),
        optimality=float(sol.optimality),
        objective=float(np.sum(sol.fun**2)),
    )


def _at_bound(sol, ps, rtol=1e-6):
    """Names of the free parameters that ended on a bound: those the solver
    reports as active plus those within ``rtol`` of a bound (relative to the
    bound, absolute for bounds below unity, as the solver measures it). A
    width or a shift on a bound is not a measurement; the list is stored so
    that downstream code can flag it. The Fe II widths get the wider
    tolerance of ``fe_width_states`` in addition."""
    lb, ub = ps.bounds()
    x = np.asarray(sol.x, float)
    hit = np.asarray(sol.active_mask) != 0
    hit |= (x - lb) <= rtol * np.maximum(1.0, np.abs(lb))
    hit |= (ub - x) <= rtol * np.maximum(1.0, np.abs(ub))
    return [n for n, h in zip(ps.free_names, hit) if h]


def _fe_widths(d):
    return {k: float(d[k]) for k in ("feop_fwhm", "feuv_fwhm") if k in d}


def fe_width_states(ps, d, active_mask=None):
    """The state of each Fe II width after a fit, one record per template.

    ``state`` is one of FE_WIDTH_STATES, decided in this order:
      not_covered    the template is not in the fit (no coverage, or Fe II off);
      unconstrained  the template's norm is zero (at or below FE_NORM_ZERO):
                     the width multiplies nothing and is unidentified;
      fixed          the width was held (the ultraviolet width where the window
                     is short, or at the fallback width);
      solver_active  the solver's active set holds the width on a bound;
      near_bound     the width lies within max(FE_WIDTH_BOUND_TOL_FRAC of the
                     span, FE_WIDTH_BOUND_TOL_KMS) of a bound without being
                     active;
      interior       none of the above.
    The record also carries ``value`` (km/s), ``fixed``, ``norm_zero``,
    ``active`` and ``near`` (the two bound criteria separately), ``bound``
    ('lower', 'upper' or ''), ``tol_kms`` and ``at_bound``, which is
    ``active or near`` for a free width with a non-zero norm and False
    otherwise: a zero norm is never counted as a width at a bound.
    ``active_mask`` is the solver's mask over the free parameters (None when
    no solver ran, as in a fallback with every Fe II parameter fixed)."""
    free = ps.free_names
    mask = np.zeros(len(free), int) if active_mask is None else np.asarray(active_mask)
    out = {}
    for w, n in FE_WIDTH_NORM.items():
        rec = dict(
            state="not_covered",
            value=np.nan,
            fixed=False,
            norm_zero=False,
            active=False,
            near=False,
            bound="",
            tol_kms=np.nan,
            at_bound=False,
        )
        if w in d:
            i = ps.names.index(w)
            lo, hi, val = ps.lb[i], ps.ub[i], float(d[w])
            tol = max(FE_WIDTH_BOUND_TOL_FRAC * (hi - lo), FE_WIDTH_BOUND_TOL_KMS)
            fixed = w in ps.fixed
            norm_zero = bool(float(d.get(n, 0.0)) <= FE_NORM_ZERO)
            active = bool(w in free and mask[free.index(w)] != 0)
            dlo, dhi = val - lo, hi - val
            near = bool(not fixed and (dlo <= tol or dhi <= tol))
            bound = ("lower" if dlo <= dhi else "upper") if (active or near) else ""
            state = (
                "unconstrained"
                if norm_zero
                else "fixed"
                if fixed
                else "solver_active"
                if active
                else "near_bound"
                if near
                else "interior"
            )
            rec.update(
                state=state,
                value=val,
                fixed=fixed,
                norm_zero=norm_zero,
                active=active,
                near=near,
                bound=bound,
                tol_kms=float(tol),
                at_bound=bool((active or near) and not norm_zero and not fixed),
            )
        out[w] = rec
    return out


def _width_bookkeeping(info, ps, d, sol=None):
    """Fill the Fe II width diagnostics of a continuum fit into ``info``:
    ``fe_widths``, ``fe_width_state``, ``fe_norm_zero`` (per template, False
    where the template is absent) and ``at_bound_widths`` (the widths at a
    bound by the wider criterion, zero norms excluded)."""
    states = fe_width_states(ps, d, None if sol is None else sol.active_mask)
    info["fe_widths"] = _fe_widths(d)
    info["fe_width_state"] = states
    info["fe_norm_zero"] = {w[:4]: bool(s["norm_zero"]) for w, s in states.items()}
    info["at_bound_widths"] = [w for w, s in states.items() if s["at_bound"]]


def _uv_window_pixels(wave, good, inwin):
    """Good pixels of the fitting windows that cover the ultraviolet Fe II template."""
    return int(np.sum(good & inwin & (wave > 2200) & (wave < 3090)))


def _check_uv_policy(policy, fallback_kms):
    if policy not in FE_UV_WIDTH_POLICIES:
        raise ValueError(f"fe_uv_width_policy must be one of {FE_UV_WIDTH_POLICIES}, got {policy!r}")
    f = float(fallback_kms)
    if not (np.isfinite(f) and FE_FWHM_MIN <= f <= FE_FWHM_MAX):
        raise ValueError(
            f"fe_uv_fallback_kms must lie in [{FE_FWHM_MIN:g}, {FE_FWHM_MAX:g}] km/s, got {fallback_kms!r}"
        )
    return policy, f


def _uv_policy_record(info, policy, fallback_kms, n_uv, feuv_fixed):
    info["feuv_policy"] = policy
    info["feuv_fallback_kms"] = float(fallback_kms)
    info["feuv_fwhm_fixed"] = bool(feuv_fixed)
    info["feuv_refit"] = False
    info["feuv_free_fit"] = None
    info["n_pix_uv"] = int(n_uv)


def _uv_at_bound(info):
    """Whether the fitted ultraviolet width is at a bound by the width criterion."""
    return bool(info["fe_width_state"]["feuv_fwhm"]["at_bound"])


FALLBACK_KEYS = (
    "solver",
    "at_bound",
    "fe_widths",
    "fe_width_state",
    "fe_norm_zero",
    "at_bound_widths",
    "feuv_fwhm_fixed",
    "feuv_policy",
    "feuv_fallback_kms",
    "feuv_refit",
    "feuv_free_fit",
    "n_pix_uv",
    "starts",
    "start_selected",
)


def _take_fallback(info, cinfo):
    """Carry the diagnostics of a PL+Fe fallback fit into the host-fit info."""
    for k in FALLBACK_KEYS:
        info[k] = cinfo[k]


def host_window_statistics(wave, flux, ivar, good, inhost):
    """The signal of the window that defines the host fraction (HOST_WINDOW,
    rest frame, inside the galaxy-template range).

    Returns dict(n_pix, n_good, n_masked, masked_frac, flux_sum, weighted_sum,
    weighted_sigma, snr, undetermined, reasons). ``flux_sum`` is the plain sum
    over the good pixels, the denominator of the host fraction;
    ``weighted_sum`` is sum(ivar f) with variance sum(ivar) (the pixels taken
    as independent), so ``snr`` = sum(ivar f) / sqrt(sum ivar). The fraction
    is ``undetermined`` when the weighted sum is not positive, when ``snr``
    is below HOST_GUARD_MIN_SNR, or when more than HOST_GUARD_MAX_MASKED_FRAC
    of the window's pixels are masked (no pixel at all counts as both)."""
    win = inhost & (wave > HOST_WINDOW[0]) & (wave < HOST_WINDOW[1])
    sel = win & good
    n_pix, n_good = int(win.sum()), int(sel.sum())
    n_masked = n_pix - n_good
    masked_frac = float(n_masked / n_pix) if n_pix else 1.0
    fsum = float(np.sum(flux[sel]))
    wsum = float(np.sum(ivar[sel] * flux[sel]))
    wsig = float(np.sqrt(np.sum(ivar[sel]))) if n_good else 0.0
    snr = wsum / wsig if wsig > 0 else np.nan
    reasons = []
    if not wsum > 0:
        reasons.append(f"flux sum over {HOST_WINDOW[0]:.0f}-{HOST_WINDOW[1]:.0f} A not positive")
    elif not snr >= HOST_GUARD_MIN_SNR:
        reasons.append(
            f"flux sum over {HOST_WINDOW[0]:.0f}-{HOST_WINDOW[1]:.0f} A at {snr:.2f} sigma "
            f"(< {HOST_GUARD_MIN_SNR:g})"
        )
    if masked_frac > HOST_GUARD_MAX_MASKED_FRAC:
        reasons.append(f"{n_masked} of {n_pix} pixels in {HOST_WINDOW[0]:.0f}-{HOST_WINDOW[1]:.0f} A masked")
    return dict(
        n_pix=n_pix,
        n_good=n_good,
        n_masked=n_masked,
        masked_frac=masked_frac,
        flux_sum=fsum,
        weighted_sum=wsum,
        weighted_sigma=wsig,
        snr=float(snr),
        min_snr=HOST_GUARD_MIN_SNR,
        max_masked_frac=HOST_GUARD_MAX_MASKED_FRAC,
        undetermined=bool(reasons),
        reasons=reasons,
    )


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
        self.flux = d[:, 1] * 1e15  # bring the template to order unity
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
    for lam in (
        3728.48,
        3869.86,
        3426.85,
        4102.89,
        4341.68,
        6302.05,
        6365.54,
        LAM["OIII5007"],
        LAM["OIII4959"],
        LAM["SII6716"],
        LAM["SII6731"],
    ):
        m |= np.abs(wave_rest / lam - 1.0) * C_KMS < narrow_halfwidth_kms
    return m


def narrow_line_mask(wave_rest, halfwidth_kms=HOST_LINE_HALFWIDTH_KMS):
    """True within +/- halfwidth of the catalogued narrow lines and of the narrow
    cores of the Balmer and Mg II lines. Narrower than ``line_mask``: it does not
    cover the broad-line complexes, only the narrow features."""
    m = np.zeros_like(wave_rest, bool)
    for lam in (
        3728.48,
        3869.86,
        3426.85,
        4102.89,
        4341.68,
        4364.44,
        4687.02,
        5877.25,
        6302.05,
        6365.54,
        7137.77,
        7321.0,
        7331.7,
        9071.1,
        9533.2,
        LAM["Hbeta"],
        LAM["Halpha"],
        LAM["MgII"],
        LAM["OIII5007"],
        LAM["OIII4959"],
        LAM["NII6548"],
        LAM["NII6584"],
        LAM["SII6716"],
        LAM["SII6731"],
    ):
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


def _add_pl_fe(
    ps,
    fref,
    fit_fe,
    cov_op,
    cov_uv,
    pl_start,
    feuv_fixed=False,
    feuv_fixed_kms=FE_UV_FWHM_FIXED_KMS,
    alpha_start=CONTI_START_ALPHAS[0],
):
    """Power-law and Fe II parameters in the frozen order.

    With ``feuv_fixed`` the ultraviolet width is held at ``feuv_fixed_kms``
    (FE_UV_FWHM_FIXED_KMS unless a fallback width is given) instead of being
    fitted: where the spectrum covers fewer than FE_UV_FREE_MIN_PIXELS pixels
    of the ultraviolet windows the width has no leverage and a free one runs
    to a bound. ``alpha_start`` is the starting power-law slope.
    """
    ps.add("pl_norm", pl_start, 0.0, 1e4 * fref)
    ps.add("pl_alpha", alpha_start, PL_ALPHA_MIN, PL_ALPHA_MAX)
    if fit_fe and cov_op:
        ps.add("feop_norm", 0.1 * fref, 0.0, 1e3 * fref)
        ps.add("feop_fwhm", 3000.0, FE_FWHM_MIN, FE_FWHM_MAX)
        ps.add("feop_shift", 0.0, -FE_SHIFT_MAX, FE_SHIFT_MAX)
    if fit_fe and cov_uv:
        ps.add("feuv_norm", 0.1 * fref, 0.0, 1e3 * fref)
        if feuv_fixed:
            ps.add("feuv_fwhm", float(feuv_fixed_kms), FE_FWHM_MIN, FE_FWHM_MAX, fixed=True)
        else:
            ps.add("feuv_fwhm", 3000.0, FE_FWHM_MIN, FE_FWHM_MAX)
        ps.add("feuv_shift", 0.0, -FE_SHIFT_MAX, FE_SHIFT_MAX)


def _select_start(fits):
    """The kept start among ``(chi2, index, ps, sol)`` fits: the earliest whose
    chi-square lies within CONTI_START_DCHI2 of the lowest finite one, or the
    first start that ran when no chi-square is finite (as with a single start)."""
    finite = [f for f in fits if np.isfinite(f[0])]
    if not finite:
        return fits[0]
    lowest = min(f[0] for f in finite)
    return min((f for f in finite if f[0] <= lowest + CONTI_START_DCHI2), key=lambda f: f[1])


def _uv_fixed_first(policy, fit_fe, cov_uv, n_uv):
    """Whether the first pass of a fit holds the ultraviolet width: policy A
    where the window is short, never under B and C."""
    return bool(policy == "A" and fit_fe and cov_uv and n_uv < FE_UV_FREE_MIN_PIXELS)


def fit_continuum(
    wave,
    flux,
    ivar,
    windows=CONTI_WINDOWS,
    fit_fe=True,
    clip=True,
    fe_uv_width_policy=FE_UV_WIDTH_POLICY,
    fe_uv_fallback_kms=FE_UV_FWHM_FIXED_KMS,
    multistart=True,
):
    """Power law + Fe II in the line-free windows, no host.

    Used directly when no host is wanted, as the fallback of the joint fit, and
    for every Monte Carlo realisation (where the host is held fixed).
    Returns (parameter dict, model on ``wave``, info).

    ``fe_uv_width_policy`` ("A", "B" or "C", see FE_UV_WIDTH_POLICY) decides
    the treatment of the ultraviolet Fe II width; ``fe_uv_fallback_kms`` is
    the width it is held at under A (short window) and C (free width at a
    bound, then refitted). ``info`` records the policy (``feuv_policy``), the
    fallback (``feuv_fallback_kms``), whether the width was held
    (``feuv_fwhm_fixed``), whether a refit happened (``feuv_refit``), the
    covered ultraviolet pixels (``n_pix_uv``), the parameters at a bound by the
    solver's criterion (``at_bound``) and the state of each Fe II width
    (``fe_width_state``, ``fe_norm_zero``, ``at_bound_widths``; see
    ``fe_width_states``). With ``multistart`` (the default) the fit is
    started from every slope of CONTI_START_ALPHAS and ``info['starts']``
    records each start; without it only the first start is fitted, as up to
    version 0.2.
    """
    policy, fallback = _check_uv_policy(fe_uv_width_policy, fe_uv_fallback_kms)
    alphas = CONTI_START_ALPHAS if multistart else CONTI_START_ALPHAS[:1]
    fe_op, fe_uv = fe_templates()
    good = np.isfinite(flux) & (ivar > 0)
    if not good.any():
        # the reference flux below would be the median of an empty set (NaN)
        raise ValueError("fit_continuum: no usable pixel (no finite flux with a positive inverse variance)")
    inwin = np.zeros_like(good)
    for lo, hi in windows:
        inwin |= (wave >= lo) & (wave <= hi)
    m0 = good & inwin
    fref = np.nanmedian(flux[m0]) if m0.sum() else np.nanmedian(flux[good])
    fref = max(fref, 1e-3)
    # the Fe II templates enter only where the spectrum covers them
    cov_op = np.sum(good & (wave > 4435) & (wave < 5535)) > 40
    cov_uv = np.sum(good & (wave > 2200) & (wave < 3090)) > 40
    n_uv = _uv_window_pixels(wave, good, inwin)

    def solve(feuv_fixed):
        m = m0
        info = dict(n_pix=int(m.sum()), fe_op=False, fe_uv=False, fallback=False)
        pl_only = m.sum() < 40
        if pl_only:
            # too few window pixels: a power law to everything outside the complexes
            info["fallback"] = True
            m = good.copy()
            for lo, hi in COMPLEX_WINDOW.values():
                m &= ~((wave > lo) & (wave < hi))

        def build(alpha0):
            ps = ParamSet()
            _add_pl_fe(
                ps,
                fref,
                fit_fe,
                cov_op,
                cov_uv,
                pl_start=fref,
                feuv_fixed=feuv_fixed,
                feuv_fixed_kms=fallback,
                alpha_start=alpha0,
            )
            if pl_only:
                for k in list(ps.names):
                    if k.startswith("fe"):
                        ps.fixed[k] = 0.0 if k.endswith("norm") else ps.val[ps.names.index(k)]
            return ps

        info["fe_op"] = bool(fit_fe and cov_op)
        info["fe_uv"] = bool(fit_fe and cov_uv)
        _uv_policy_record(info, policy, fallback, n_uv, feuv_fixed)
        n_free = len(build(CONTI_START_ALPHAS[0]).free_names)
        if m.sum() <= n_free:
            # every usable pixel lies inside the line complexes: least squares on
            # no residual would return the starting values as a solution
            raise ValueError(
                f"fit_continuum: {int(m.sum())} usable pixels outside the line complexes "
                f"for {n_free} free continuum parameters"
            )
        w = np.sqrt(ivar[m])
        x = wave[m]
        y = flux[m]

        def residual(ps, x, y, w):
            def resid(p):
                return (y - conti_model(x, ps.full(p), fe_op, fe_uv)) * w

            return resid

        # every start on the same pixels; the lowest chi-square is kept
        starts, fits, first_error = [], [], None
        for k, alpha0 in enumerate(alphas):
            ps_k = build(alpha0)
            rec = dict(index=k, alpha_start=float(alpha0), selected=False)
            try:
                sol_k = least_squares(
                    residual(ps_k, x, y, w),
                    ps_k.p0(),
                    bounds=ps_k.bounds(),
                    x_scale="jac",
                    max_nfev=MAX_NFEV_CONTI,
                )
            except Exception as exc:
                first_error = first_error or exc
                rec.update(chi2=np.nan, pl_alpha=np.nan, success=False, error=f"{type(exc).__name__}: {exc}")
                starts.append(rec)
                continue
            chi2 = float(np.sum(sol_k.fun**2))
            rec.update(chi2=chi2, pl_alpha=float(ps_k.full(sol_k.x)["pl_alpha"]), success=bool(sol_k.success))
            starts.append(rec)
            fits.append((chi2, k, ps_k, sol_k))
        if not fits:
            raise first_error
        _, k_best, ps, sol = _select_start(fits)
        for rec in starts:
            rec["selected"] = rec["index"] == k_best

        if clip and m.sum() > 60:
            # one round of outlier clipping (absorption features, residual lines), as Shen et al. 2011
            r = residual(ps, x, y, w)(sol.x)
            keep = (r > CLIP_LO) & (r < CLIP_HI)
            if keep.sum() > 40 and keep.sum() < len(r):
                x, y, w = x[keep], y[keep], w[keep]
                sol = least_squares(
                    residual(ps, x, y, w), sol.x, bounds=ps.bounds(), x_scale="jac", max_nfev=MAX_NFEV_CONTI
                )
                info["n_pix"] = int(keep.sum())
        d = ps.full(sol.x)
        ps.set_values(d)
        model = conti_model(wave, d, fe_op, fe_uv)
        info["chi2"] = float(np.sum(sol.fun**2))
        info["ps"] = ps
        info["solver"] = _solver_record(sol)
        info["at_bound"] = _at_bound(sol, ps)
        info["starts"] = starts
        info["start_selected"] = int(k_best)
        _width_bookkeeping(info, ps, d, sol)
        return d, model, info

    d, model, info = solve(_uv_fixed_first(policy, fit_fe, cov_uv, n_uv))
    if policy == "C" and _uv_at_bound(info):
        # the free width ended on a bound: hold it at the fallback and refit
        first = info
        d, model, info = solve(True)
        info["feuv_refit"] = True
        info["feuv_free_fit"] = _free_fit_record(first)
    return d, model, info


def _free_fit_record(info):
    """What the discarded free-width pass of policy C found."""
    return dict(
        fe_widths=info["fe_widths"],
        fe_width_state=info["fe_width_state"],
        at_bound=info["at_bound"],
        solver=info["solver"],
        chi2=info["chi2"],
    )


# ----------------------------------------------------------------------------
# Joint host + power law + Fe II (the default path)
# ----------------------------------------------------------------------------
def fit_continuum_host(
    wave,
    flux,
    ivar,
    fit_fe=True,
    n_gal_max=N_GAL_MAX,
    min_host_frac=MIN_HOST_FRAC,
    windows=CONTI_WINDOWS,
    fe_uv_width_policy=FE_UV_WIDTH_POLICY,
    fe_uv_fallback_kms=FE_UV_FWHM_FIXED_KMS,
    host_guard=True,
    multistart=True,
):
    """One coherent pseudo-continuum: sum_i g_i E_i(lambda) + power law + Fe II.

    Pixels used: the Shen et al. (2011) windows everywhere, plus every pixel
    inside the galaxy-template range that is outside the emission-line masks,
    so that the 4000 A break and the stellar absorption anchor the host. The
    eigenspectrum count is stepped down (n_gal_max, 3, 2, 1, 0) until the host
    is non-negative; the host is kept only above ``min_host_frac`` of the
    4200-5000 A flux, otherwise the continuum is refitted without it.

    With ``host_guard`` the 4200-5000 A window is checked first
    (``host_window_statistics``): when its inverse-variance weighted flux sum
    is not positive or below HOST_GUARD_MIN_SNR sigma of its noise, or more
    than HOST_GUARD_MAX_MASKED_FRAC of its pixels are masked, the host
    fraction is undetermined, the host is not fitted, ``host_frac_4200_5000``
    is NaN and ``host_undetermined`` is True with the reason in ``reason``.
    The window statistics are stored as ``host_window`` in every case.

    ``fe_uv_width_policy`` and ``fe_uv_fallback_kms`` are those of
    ``fit_continuum``; under policy C the joint fit is repeated with the
    ultraviolet width held at the fallback when the free width ends on a
    bound (``feuv_refit``). With ``multistart`` (the default) every count of
    eigenspectra is started from each pair of CONTI_START_ALPHAS and
    CONTI_START_HOST_SCALES (slopes alone without a host); without it only the
    first pair, as up to version 0.2.

    Returns (parameter dict, total continuum incl. host, host model, info).
    """
    policy, fallback = _check_uv_policy(fe_uv_width_policy, fe_uv_fallback_kms)
    conti_kw = dict(
        fit_fe=fit_fe, fe_uv_width_policy=policy, fe_uv_fallback_kms=fallback, multistart=multistart
    )
    alphas = CONTI_START_ALPHAS if multistart else CONTI_START_ALPHAS[:1]
    scales = CONTI_START_HOST_SCALES if multistart else CONTI_START_HOST_SCALES[:1]
    fe_op, fe_uv = fe_templates()
    P = pca_templates()
    good = np.isfinite(flux) & (ivar > 0)
    inhost = (wave > P["gw"].min() + 2) & (wave < P["gw"].max() - 2)
    inwin = np.zeros_like(good)
    for lo, hi in windows:
        inwin |= (wave >= lo) & (wave <= hi)
    use0 = good & (inwin | (inhost & ~line_mask(wave)))
    window = host_window_statistics(wave, flux, ivar, good, inhost)
    info = dict(
        applied=False,
        host_frac_4200_5000=np.nan,
        n_gal=0,
        n_negative_pix=0,
        reason="",
        n_pix=int(use0.sum()),
        fe_op=False,
        fe_uv=False,
        solver_attempts=[],
        host_guard=bool(host_guard),
        host_undetermined=False,
        host_window=window,
    )
    if use0.sum() < 40:
        d, model, cinfo = fit_continuum(wave, flux, ivar, **conti_kw)
        info.update(reason="too few line-free pixels; PL+Fe only", n_pix=cinfo["n_pix"])
        _take_fallback(info, cinfo)
        return d, model, np.zeros_like(flux), info
    if host_guard and window["undetermined"]:
        # the window that defines the host fraction carries no signal: the
        # fraction is undetermined and the host is not fitted
        d, model, cinfo = fit_continuum(wave, flux, ivar, **conti_kw)
        info.update(
            reason="host undetermined: " + "; ".join(window["reasons"]) + "; PL+Fe only",
            host_undetermined=True,
        )
        _take_fallback(info, cinfo)
        return d, model, np.zeros_like(flux), info

    Gfull = [
        np.where(inhost, np.interp(wave, P["gw"], P["gp"][i], left=0, right=0), 0.0) for i in range(n_gal_max)
    ]
    fref = max(float(np.nanmedian(flux[use0])), 1e-3)
    cov_op = np.sum(good & (wave > 4435) & (wave < 5535)) > 40
    cov_uv = np.sum(good & (wave > 2200) & (wave < 3090)) > 40
    # the galaxy templates start at 3450 A, so inside the ultraviolet windows
    # the joint fit uses the same pixels as the plain fit
    n_uv = _uv_window_pixels(wave, good, inwin)
    gscale = fref / max(float(np.nanmedian(Gfull[0][inhost])) if inhost.sum() else 1.0, 1e-6)

    def make_ps(ng, feuv_fixed, alpha0=CONTI_START_ALPHAS[0], host_scale=CONTI_START_HOST_SCALES[0]):
        ps = ParamSet()
        _add_pl_fe(
            ps,
            fref,
            fit_fe,
            cov_op,
            cov_uv,
            pl_start=0.7 * fref,
            feuv_fixed=feuv_fixed,
            feuv_fixed_kms=fallback,
            alpha_start=alpha0,
        )
        for i in range(ng):
            # eigenspectrum 0 is the mean galaxy: its coefficient must be non-negative
            ps.add(
                f"gal{i}",
                host_scale * gscale if i == 0 else 0.0,
                0.0 if i == 0 else -50.0 * gscale,
                50.0 * gscale,
            )
        return ps

    def host_of(d, ng):
        h = np.zeros_like(wave)
        for i in range(ng):
            h += d[f"gal{i}"] * Gfull[i]
        return h

    # the window that defines the host fraction; a flux sum that is not positive
    # leaves the fraction undefined (NaN), and the host is then not subtracted
    sel_frac = inhost & (wave > 4200) & (wave < 5000) & good
    fsum = float(np.sum(flux[sel_frac]))

    def host_fraction(host):
        return float(np.sum(host[sel_frac]) / fsum) if sel_frac.sum() > 20 and fsum > 0 else np.nan

    def residual(ps, ng, idx, y, w):
        def resid(p):
            d = ps.full(p)
            m = conti_model(wave[idx], d, fe_op, fe_uv)
            for i in range(ng):
                m = m + d[f"gal{i}"] * Gfull[i][idx]
            return (y - m) * w

        return resid

    def starts_for(ng):
        """(slope, host amplitude) starts; without a host the slopes alone."""
        if ng == 0:
            return [(a, None) for a in alphas]
        return [(a, s) for a in alphas for s in scales]

    def attempt(feuv_fixed):
        """The first non-negative host over the stepped-down eigenspectrum
        counts, or None when every attempt failed. For each count every start
        is fitted on the same pixels and the lowest chi-square is clipped and
        refitted."""
        for ng in [n for n in (n_gal_max, 3, 2, 1, 0) if n <= n_gal_max]:
            use = use0 if ng > 0 else (good & inwin if (good & inwin).sum() >= 40 else use0)
            y = flux[use]
            w = np.sqrt(ivar[use])
            idx = np.where(use)[0]
            starts, fits, first_error = [], [], None
            for k, (alpha0, scale) in enumerate(starts_for(ng)):
                ps_k = make_ps(ng, feuv_fixed, alpha0, CONTI_START_HOST_SCALES[0] if scale is None else scale)
                rec = dict(
                    index=k,
                    alpha_start=float(alpha0),
                    host_start=None if scale is None else float(scale),
                    selected=False,
                )
                try:
                    sol_k = least_squares(
                        residual(ps_k, ng, idx, y, w),
                        ps_k.p0(),
                        bounds=ps_k.bounds(),
                        x_scale="jac",
                        max_nfev=MAX_NFEV_CONTI_HOST,
                    )
                except Exception as exc:
                    first_error = first_error or exc
                    rec.update(
                        chi2=np.nan,
                        pl_alpha=np.nan,
                        host_frac=np.nan,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    starts.append(rec)
                    continue
                d_k = ps_k.full(sol_k.x)
                chi2 = float(np.sum(sol_k.fun**2))
                rec.update(
                    chi2=chi2,
                    pl_alpha=float(d_k["pl_alpha"]),
                    host_frac=host_fraction(host_of(d_k, ng)),
                    success=bool(sol_k.success),
                )
                starts.append(rec)
                fits.append((chi2, k, ps_k, sol_k))
            try:
                if not fits:
                    raise first_error
                _, k_best, ps, sol = _select_start(fits)
                for rec in starts:
                    rec["selected"] = rec["index"] == k_best
                r = residual(ps, ng, idx, y, w)(sol.x)
                keep = (r > CLIP_LO) & (r < CLIP_HI)
                if keep.sum() > 40 and keep.sum() < len(r):
                    sol = least_squares(
                        residual(ps, ng, idx[keep], y[keep], w[keep]),
                        sol.x,
                        bounds=ps.bounds(),
                        x_scale="jac",
                        max_nfev=MAX_NFEV_CONTI_HOST,
                    )
            except Exception as exc:
                info["solver_attempts"].append(
                    dict(
                        n_gal=ng,
                        success=False,
                        status="exception",
                        message=f"{type(exc).__name__}: {exc}",
                        feuv_fixed=feuv_fixed,
                        starts=starts,
                    )
                )
                continue
            # an attempt that stopped short of convergence is recorded, not
            # discarded: the solver record and the at-bound list say so
            solver = _solver_record(sol)
            info["solver_attempts"].append(
                dict(n_gal=ng, feuv_fixed=feuv_fixed, start_selected=int(k_best), starts=starts, **solver)
            )
            d = ps.full(sol.x)
            ps.set_values(d)
            host = host_of(d, ng)
            n_neg = int(np.sum(host[inhost] < -1e-3 * max(np.nanmax(np.abs(host)), 1e-9)))
            frac = host_fraction(host)
            cand = dict(
                d=d,
                ps=ps,
                ng=ng,
                host=host,
                n_neg=n_neg,
                frac=frac,
                chi2=float(np.sum(sol.fun**2)),
                solver=solver,
                at_bound=_at_bound(sol, ps),
                sol=sol,
                feuv_fixed=feuv_fixed,
                starts=starts,
                start=int(k_best),
            )
            if ng == 0 or n_neg <= max(50, 0.02 * inhost.sum()):
                return cand
        return None

    def take(cand):
        info.update(
            n_gal=cand["ng"],
            n_negative_pix=cand["n_neg"],
            host_frac_4200_5000=cand["frac"],
            fe_op="feop_norm" in cand["d"],
            fe_uv="feuv_norm" in cand["d"],
            chi2=cand["chi2"],
            solver=cand["solver"],
            at_bound=cand["at_bound"],
            starts=cand["starts"],
            start_selected=cand["start"],
        )
        _uv_policy_record(info, policy, fallback, n_uv, cand["feuv_fixed"])
        _width_bookkeeping(info, cand["ps"], cand["d"], cand["sol"])

    best = attempt(_uv_fixed_first(policy, fit_fe, cov_uv, n_uv))
    free_fit = None
    if best is not None and policy == "C":
        take(best)
        if _uv_at_bound(info):
            # the free width ended on a bound: hold it at the fallback and
            # refit; the free result stands if the refit fails
            first = _free_fit_record(info)
            refit = attempt(True)
            if refit is not None:
                best = refit
                free_fit = first
    if best is None:
        d, model, cinfo = fit_continuum(wave, flux, ivar, **conti_kw)
        info.update(reason="joint fit failed; PL+Fe only")
        _take_fallback(info, cinfo)
        return d, model, np.zeros_like(flux), info

    d, ng, host, frac = best["d"], best["ng"], best["host"], best["frac"]
    take(best)
    if free_fit is not None:
        info["feuv_refit"] = True
        info["feuv_free_fit"] = free_fit
    if ng > 0 and (not np.isfinite(frac) or frac < min_host_frac):
        # host too weak to trust (Shen et al. 2011): refit without it
        info["reason"] = f"host fraction {frac:.2f} < {min_host_frac}; PL+Fe only"
        d2, model2, cinfo = fit_continuum(wave, flux, ivar, **conti_kw)
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
