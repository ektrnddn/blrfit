"""
Synthetic epochs for the pair tests: a fit-like record built from known
Gaussian components on a DESI-like or SDSS-like pixel grid, with optional
noise, so that a test knows exactly what the estimator should return. The
record carries the truth as its components unless a fit error is asked for,
and ``refit_broad`` replaces the broad parameters by a least-squares fit to the
record's own noisy data, so that a template carries its epoch's noise as a
real fit does.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from blrfit import pairs
from blrfit.constants import COMPLEX_WINDOW, LAM, R_NII, R_OIII
from blrfit.model.lines import eval_components
from blrfit.model.params import gauss_lam

PREFIX = {"Halpha": "Ha", "Hbeta": "Hb"}


def pixel_grid(line, z, kind="desi"):
    """Rest-frame pixel wavelengths of a complex window: DESI-like (0.8 A
    observed, linear) or SDSS-like (1e-4 dex, 69 km/s)."""
    lo, hi = COMPLEX_WINDOW[line]
    if kind == "desi":
        return np.arange(lo, hi, 0.8 / (1.0 + z))
    if kind == "sdss":
        return 10.0 ** np.arange(np.log10(lo), np.log10(hi), 1e-4)
    raise ValueError(kind)


def broad_comps(line, params):
    """Component tuples and parameters of broad Gaussians ``params`` =
    [(A, v_kms, sig_kms), ...] in the fitter's naming (Ha_b0_A, Ha_b0_v, ...)."""
    p = PREFIX[line]
    comps, d = [], {}
    for k, (A, v, s) in enumerate(params):
        an, vn, sn = f"{p}_b{k}_A", f"{p}_b{k}_v", f"{p}_b{k}_sig"
        comps.append((f"{p}_b{k}", LAM[line], an, vn, sn, "broad", None))
        d[an], d[vn], d[sn] = float(A), float(v), float(s)
    return comps, d


def narrow_comps(line, amp, v=0.0, sig=150.0):
    """The narrow lines of a complex as the fitter ties them: Halpha + [N II]
    (6548 tied by 1/R_NII) + [S II]; or Hbeta + [O III] (4959 tied by 1/R_OIII)."""
    if line == "Halpha":
        comps = [
            ("Halpha_n", LAM["Halpha"], "Halpha_n_A", "n_v", "n_sig", "narrow", None),
            ("NII6584", LAM["NII6584"], "NII6584_A", "n_v", "n_sig", "narrow", None),
            ("NII6548", LAM["NII6548"], "NII6584_A", "n_v", "n_sig", "narrow", ("NII6584_A", 1.0 / R_NII)),
            ("SII6716", LAM["SII6716"], "SII6716_A", "s2_v", "s2_sig", "narrow", None),
            ("SII6731", LAM["SII6731"], "SII6731_A", "s2_v", "s2_sig", "narrow", None),
        ]
        d = dict(
            Halpha_n_A=amp,
            NII6584_A=0.8 * amp,
            SII6716_A=0.3 * amp,
            SII6731_A=0.25 * amp,
            n_v=v,
            n_sig=sig,
            s2_v=v,
            s2_sig=sig,
        )
    elif line == "Hbeta":
        comps = [
            ("Hbeta_n", LAM["Hbeta"], "Hbeta_n_A", "n_v", "n_sig", "narrow", None),
            ("OIII5007c", LAM["OIII5007"], "OIII5007c_A", "o3_v", "o3_sig", "narrow", None),
            (
                "OIII4959c",
                LAM["OIII4959"],
                "OIII5007c_A",
                "o3_v",
                "o3_sig",
                "narrow",
                ("OIII5007c_A", 1.0 / R_OIII),
            ),
        ]
        d = dict(Hbeta_n_A=amp, OIII5007c_A=3.0 * amp, n_v=v, n_sig=sig, o3_v=v, o3_sig=sig)
    else:
        raise ValueError(line)
    return comps, {k: float(x) for k, x in d.items()}


def make_epoch(
    line="Halpha",
    z=0.1,
    grid="desi",
    broad=((10.0, 0.0, 1500.0),),
    narrow=None,
    snr_peak=None,
    rng=None,
    shift=0.0,
    narrow_shift=0.0,
    flux_scale=1.0,
    baseline=(0.0, 0.0),
    fit_narrow_error=0.0,
    sigma_floor=None,
):
    """One synthetic epoch as a fit-like record.

    broad: [(A, v, sig_kms), ...] in the common frame; ``shift`` is added to
        every broad velocity (the truth of the test); ``narrow`` = dict(amp=...,
        v=..., sig=...) adds the narrow lines, moved by ``narrow_shift`` (a frame
        offset);
    snr_peak: peak S/N of the broad profile (None: no noise; the errors are
        then nominal, 1e-3 of the peak, so that chi-square is still defined);
    flux_scale, baseline: the data are flux_scale * model + b0 + b1 (x -
        lam0)/100; the record's amplitudes carry the flux scale, the baseline
        stays in the data (a continuum the fit missed);
    fit_narrow_error: fractional error of the narrow amplitudes in the record
        (the data keep the truth): a narrow-subtraction residual.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x = pixel_grid(line, z, grid)
    comps, d = broad_comps(line, [(A, v + shift, s) for A, v, s in broad])
    if narrow:
        cn, dn = narrow_comps(line, **narrow)
        for key in ("n_v", "s2_v", "o3_v"):
            if key in dn:
                dn[key] += narrow_shift
        comps, d = comps + cn, {**d, **dn}
    model = eval_components(x, d, comps)
    truth = flux_scale * model + baseline[0] + baseline[1] * (x - LAM[line]) / 100.0
    peak = flux_scale * float(eval_components(x, d, comps, kinds=("broad",)).max())
    if snr_peak:
        sigma = peak / float(snr_peak)
        noise = rng.normal(0.0, sigma, x.size)
    else:
        sigma = max(peak, 1.0) * 1e-3 if sigma_floor is None else sigma_floor
        noise = np.zeros(x.size)
    y = truth + noise
    ivar = np.full(x.size, 1.0 / sigma**2)
    d_fit = dict(d)
    for _lab, _l0, an, _vn, _sn, kind, _r in comps:
        if kind == "broad":
            d_fit[an] = d[an] * flux_scale
        else:
            d_fit[an] = d[an] * flux_scale * (1.0 + fit_narrow_error)
    hm = pairs.model_halfmax(d_fit, comps, LAM[line])
    v_sys = float(d.get("n_v", 0.0)) if narrow else 0.0
    return dict(
        z=float(z),
        fits={
            line: dict(
                name=line,
                d=d_fit,
                comps=comps,
                x=x,
                y=y,
                w=np.sqrt(ivar),
                ivar_stat=ivar,
                window=tuple(COMPLEX_WINDOW[line]),
                n_broad=len(broad),
            )
        },
        meas={
            line: dict(
                v_sys=v_sys,
                fwhm=hm["fwhm"],
                c50=hm["c50"],
                c50_sys=hm["c50"] - v_sys,
                broad_flux_snr=float(snr_peak) if snr_peak else np.inf,
            )
        },
        cls={line: dict(label="F")},
        settings=dict(ebv=0.0, err_floor=0.0, complexes=[line], nmc=0),
        truth=dict(
            shift=float(shift),
            narrow_shift=float(narrow_shift),
            flux_scale=float(flux_scale),
            baseline=tuple(baseline),
            sigma=float(sigma),
            peak=peak,
        ),
    )


def refit_broad(rec, line, rng=None, start_jitter_kms=0.0):
    """Replace the record's broad parameters by a least-squares fit of the same
    number of Gaussians to the record's (noisy) data, the narrow model held at
    the record's values: a template that carries the epoch's own noise. Starts
    from the truth (optionally jittered); this tests the error propagation, not
    the fitter's convergence."""
    r = rec["fits"][line]
    x, y, w = r["x"], r["y"], r["w"]
    comps, d = r["comps"], dict(r["d"])
    narrow = eval_components(x, d, comps, kinds=pairs.NARROW_KINDS)
    bro = [(an, vn, sn) for _lab, _l0, an, vn, sn, kind, _r in comps if kind == "broad"]
    p0 = np.array([v for an, vn, sn in bro for v in (d[an], d[vn], d[sn])])
    if start_jitter_kms and rng is not None:
        p0[1::3] += rng.normal(0.0, start_jitter_kms, len(bro))
    lo = np.array([v for _ in bro for v in (0.0, -1.5e4, 50.0)])
    hi = np.array([v for _ in bro for v in (np.inf, 1.5e4, 2.0e4)])

    def resid(p):
        m = np.zeros_like(x)
        for k in range(len(bro)):
            m += gauss_lam(x, p[3 * k], LAM[line], p[3 * k + 1], p[3 * k + 2])
        return (y - narrow - m) * w

    sol = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac")
    for k, (an, vn, sn) in enumerate(bro):
        d[an], d[vn], d[sn] = (float(v) for v in sol.x[3 * k : 3 * k + 3])
    out = dict(rec)
    out["fits"] = {line: dict(r, d=d)}
    hm = pairs.model_halfmax(d, comps, LAM[line])
    out["meas"] = {
        line: dict(
            rec["meas"][line], fwhm=hm["fwhm"], c50=hm["c50"], c50_sys=hm["c50"] - rec["meas"][line]["v_sys"]
        )
    }
    out["refit"] = dict(success=bool(sol.success), nfev=int(sol.nfev), chi2=float(np.sum(sol.fun**2)))
    return out
