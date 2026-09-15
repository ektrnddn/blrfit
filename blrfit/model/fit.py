"""
The end-to-end fit of one spectrum.

Steps, in order:
  1. bad pixels zeroed, Galactic de-reddening, 2 per cent error floor in
     quadrature, shift to the rest frame at the input redshift;
  2. pseudo-continuum (power law + Fe II + host) and its subtraction;
  3. a preliminary Hbeta fit for the [O III] velocity that starts and, at high
     [O III] signal-to-noise, constrains the Halpha narrow group;
  4. each requested complex, Halpha first: 1-3 broad Gaussians chosen by the
     BIC, from several starting velocities; the Halpha narrow kinematics are
     passed to the Hbeta complex when the Halpha narrow lines are detected;
  5. profile measurements, optional Monte Carlo errors, classification.

Result dictionary: z, wave_rest, flux_rest, ivar_rest, host_model, host_info,
conti (parameters), conti_model (power law + Fe II), flux_sub, o3_prefit,
settings, fits{name: fit dict}, meas{name: measures}, cls{name: class dict},
mc{name: {key: (p16, p50, p84)}}, err{name: {key: error}}.
``fit_status`` distinguishes missing coverage from line-solver failure and
records, per line, whether the selected solution converged (``converged``)
and whether the continuum solver did (``continuum_converged``);
``continuum_status`` ('success', 'unconverged' or 'unknown') is the end state
of the continuum solver, which does not stop the line fits: a continuum that
stopped short of convergence is used and flagged, as the production fitter
did. ``mc_info`` describes the conditional uncertainty model and draw
diagnostics.

Every velocity reported is a difference between two quantities measured in
the same fit, so the input redshift cancels from the offsets to first order.
It still decides where the narrow lines are sought (+/-1500 km/s of z; +/-700
for Mg II: a redshift wrong by more gives class X, not a shifted velocity),
which complexes the data cover, whether the host is fitted (z < 1.2), and the
systemic redshift and luminosity that are derived from it.

``settings`` records the frozen configuration for every result. The keys
``sii_mode`` ('soft'), ``o3_mode`` ('order') and ``o3_split`` (False) name the
fixed choices for the [S II] tie and the [O III] core/wing ordering, whose
alternatives were rejected during development; they are kept so that results
written by earlier versions of the fitter can be compared.
"""
from __future__ import annotations

import numpy as np

from ..constants import (ERR_FLOOR, HOST_ZMAX, SIG_BROAD_MIN, MAX_BROAD, DBIC, V_BROAD_MAX,
                         HOST_CONTINUUM_ONLY, O3_ORDER_AMPLITUDE, O3_INFORMED_START,
                         O3_START_MIN_SNR, SYS_PRIOR_KMS, SYS_PRIOR_MIN_SNR, V_NARROW_MAX,
                         BROAD_WIDTH_SLOPE, NLR_WING, NARROW_PRIOR_MIN_SNR,
                         FLUX_SCALE_MIN, FLUX_SCALE_MAX)
from .extinction import deredden
from .continuum import fit_continuum, fit_continuum_host
from .lines import fit_complex, fit_complex_select
from ..measure import measure_complex
from ..classify import classify

SUMMARY_KEYS = ("v_sys", "sig_sys", "v_o3", "v_sii", "sig_sii", "z_sys", "v_peak", "centroid", "c25", "c50", "c75",
                "c90", "fwhm", "W25", "W75", "sigma_line", "skew", "AI", "KI", "n_peaks",
                "peak_sep", "dip_frac", "v_peak_sys", "centroid_sys", "c50_sys", "c25_sys",
                "c75_sys", "peak_top", "peak_top_sys", "centroid25_sys", "centroid50_sys",
                "broad_flux", "broad_ew", "broad_ew_agn", "conti_at_line", "broad_lum",
                "broad_peak_snr", "broad_flux_snr",
                "narrow_peak_snr", "sys_snr", "o3_core_snr", "v_o3_peak", "v_o3_pre", "o3_pre_snr",
                "nw_f", "nw_v", "nw_sig", "v_cover_lo", "v_cover_hi",
                "chi2_red", "n_broad", "v_single_gauss",
                "data_v_peak", "data_c50", "data_centroid_win")
PREFIX = {"Halpha": "HA", "Hbeta": "HB", "MgII": "MG"}


def _lumdist_cm(z):
    """Luminosity distance for the Planck 2018 cosmology, in cm (NaN if astropy is missing)."""
    try:
        from astropy.cosmology import Planck18
        import astropy.units as u
        return float(Planck18.luminosity_distance(z).to(u.cm).value)
    except Exception:
        return np.nan


def _fit_line_sequence(wr, fsub, ir, cmodel, host_model, z, complexes,
                       use_ha_systemic=True, max_broad=MAX_BROAD, dbic=DBIC,
                       fixed_n_broad=None, kw=None, dl=np.nan):
    """The same systemic estimator for the observed spectrum and each MC draw.

    MC conditions on the selected component counts and host model. All data-
    dependent OIII starts/priors and Halpha-to-Hbeta transfers are recomputed.
    """
    prior_v = prior_s = prior_snr = prior_nw = None
    order = [c for c in ("Halpha", "Hbeta", "MgII") if c in complexes]
    fits, meas, fit_status = {}, {}, {}
    base_kw = dict(kw or {})
    # [O III]-informed start and prior for the Halpha narrow group
    v_o3_pre, o3_pre_snr = np.nan, 0.0
    pre_diag = dict(status="disabled")
    if O3_INFORMED_START and "Halpha" in order and "Hbeta" in order and use_ha_systemic:
        try:
            pre_kw = {k: base_kw[k] for k in ("oiii_wing", "heii", "sig_broad_min") if k in base_kw}
            rpre = fit_complex("Hbeta", wr, fsub, ir, 1, diagnostics=pre_diag, **pre_kw)
            if rpre is not None and "OIII5007c_A" in rpre["d"]:
                npre = float(np.median(1.0 / rpre["w"]))
                v_o3_pre = float(rpre["d"]["o3_v"])
                o3_pre_snr = float(rpre["d"]["OIII5007c_A"] / npre) if npre > 0 else 0.0
        except Exception as exc:
            pre_diag.update(status="exception", message=f"{type(exc).__name__}: {exc}")
    prefit = dict(v_o3=v_o3_pre, snr=o3_pre_snr, **pre_diag)
    for name in order:
        if fixed_n_broad is not None and name not in fixed_n_broad:
            continue
        kws = dict(base_kw)
        if name == "Halpha" and np.isfinite(v_o3_pre) and o3_pre_snr >= O3_START_MIN_SNR:
            kws["n_v_starts"] = [0.0, v_o3_pre] if abs(v_o3_pre) > 50 else [v_o3_pre]
            if o3_pre_snr >= SYS_PRIOR_MIN_SNR:
                kws["n_v_prior"] = (v_o3_pre, SYS_PRIOR_KMS)
        if name == "Hbeta" and use_ha_systemic and prior_v is not None:
            kws.update(v_sys_prior=prior_v, sig_sys_prior=prior_s, nw_prior=prior_nw)
        diag = {}
        if fixed_n_broad is None:
            r, allfits = fit_complex_select(name, wr, fsub, ir, max_broad=max_broad,
                                           dbic=dbic, diagnostics=diag, **kws)
        else:
            r = fit_complex(name, wr, fsub, ir, fixed_n_broad[name], diagnostics=diag, **kws)
            allfits = []
        fit_status[name] = diag
        if r is None:
            continue
        if fixed_n_broad is None:
            r["all_fits"] = allfits
        m = measure_complex(r, cmodel + host_model, wr, z, dl_cm=dl, host_model=host_model)
        ha_prior_used = name == "Hbeta" and use_ha_systemic and prior_v is not None
        m["systemic_source"] = ("Halpha prior" if ha_prior_used
                                else ("[OIII] core" if name == "Hbeta" else "own narrow group"))
        if ha_prior_used:
            m["sys_snr"] = prior_snr
        if name == "Halpha":
            m["v_o3_pre"] = v_o3_pre; m["o3_pre_snr"] = o3_pre_snr
        fits[name] = r; meas[name] = m
        if name == "Halpha" and m["narrow_peak_snr"] >= NARROW_PRIOR_MIN_SNR and np.isfinite(m["v_sys"]):
            prior_v, prior_s, prior_snr = m["v_sys"], m["sig_sys"], m["narrow_peak_snr"]
            dd = r["d"]
            if "nw_f" in dd:
                prior_nw = (float(dd["nw_f"]), float(dd["nw_v"]), float(dd["nw_sig"]))

    return fits, meas, prefit, fit_status


def fit_spectrum(wave_obs, flux, ivar, z, ebv=0.0, host=True, fe=True,
                 complexes=("Halpha", "Hbeta", "MgII"), max_broad=MAX_BROAD, dbic=DBIC,
                 use_ha_systemic=True, oiii_wing=True, heii=True,
                 mgii_narrow=True, mgii_doublet=False, nmc=0, seed=0,
                 thresholds=None, sig_broad_min=SIG_BROAD_MIN, err_floor=ERR_FLOOR,
                 flux_scale=1.0):
    """Fit one spectrum end to end; see the module docstring for the result keys.

    wave_obs : observed-frame vacuum wavelength, Angstrom
    flux, ivar : flux in 1e-17 erg/s/cm^2/A (the unit of SDSS and DESI spectra, for
        which the amplitude bounds and starting values of the model are set) and its
        inverse variance; a median positive flux outside FLUX_SCALE_MIN-FLUX_SCALE_MAX
        raises ValueError
    flux_scale : factor that brings the flux to 1e-17 erg/s/cm^2/A (1e17 for a spectrum
        in erg/s/cm^2/A); the flux is multiplied by it and the inverse variance divided
        by its square before anything else, and every flux-bearing output (rest-frame
        arrays, continuum parameters, amplitudes, fluxes, equivalent widths,
        luminosities) is then in the scaled unit; recorded in ``settings``
    z : redshift at which the narrow lines are sought (+/-1500 km/s)
    ebv : Galactic E(B-V); 0 skips the de-reddening
    host, fe : include the host galaxy / the Fe II templates in the continuum
    complexes : any of "Halpha", "Hbeta", "MgII"; a complex outside the data is skipped
    nmc, seed : Monte Carlo realisations for the errors (0 = none)
    thresholds : overrides of the classification thresholds (``DEFAULT_THRESH``)
    """
    complexes = tuple(complexes)
    flux_scale = float(flux_scale)
    if not (np.isfinite(flux_scale) and flux_scale > 0):
        raise ValueError(f"flux_scale must be a positive finite number, got {flux_scale!r}")
    wave_obs = np.asarray(wave_obs, float); flux = np.asarray(flux, float) * flux_scale
    ivar = np.asarray(ivar, float) / flux_scale**2
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    flux = np.where(bad, 0.0, flux); ivar = np.where(bad, 0.0, ivar)
    # Input-scale guard: the model is set up for fluxes of order 1-1000 in
    # 1e-17 erg/s/cm^2/A (see FLUX_SCALE_MIN); another unit has to be declared
    # through flux_scale rather than silently fitted against the wrong floors.
    positive = flux[~bad & (flux > 0)]
    if positive.size:
        median_flux = float(np.median(positive))
        if not FLUX_SCALE_MIN <= median_flux <= FLUX_SCALE_MAX:
            raise ValueError(
                f"median positive flux {median_flux:.3g} is outside {FLUX_SCALE_MIN:g}-{FLUX_SCALE_MAX:g}: "
                "pass the flux in 1e-17 erg/s/cm^2/A (the unit of SDSS and DESI spectra), or give "
                "flux_scale, the factor that brings it to that unit (1e17 for erg/s/cm^2/A); the "
                "amplitude bounds and starting values of the model are set for that unit")
    flux, ivar = deredden(wave_obs, flux, ivar, ebv)
    # Error floor: a fractional flux-calibration/model term in quadrature.
    # The narrow-line cores of bright galaxies reach per-pixel S/N of several
    # hundred, where the residuals of any line model are limited by the
    # spectrophotometric calibration; without the floor they dominate
    # chi-square and drag the broad components.
    if err_floor and err_floor > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(ivar > 0, 1.0 / ivar, 0.0) + (err_floor * np.abs(flux)) ** 2
            ivar = np.where(ivar > 0, 1.0 / var, 0.0)
    wr = wave_obs / (1 + z); fr = flux * (1 + z); ir = ivar / (1 + z) ** 2

    res = dict(z=z, wave_rest=wr, flux_rest=fr, ivar_rest=ir, fits={}, meas={}, cls={},
               mc={}, err={}, mc_info=dict(status="not_requested"))
    if host and z < HOST_ZMAX:
        cd, ctotal, host_model, hinfo = fit_continuum_host(wr, fr, ir, fit_fe=fe)
        cmodel = ctotal - host_model                 # power law + Fe II only
        cinfo = hinfo
    else:
        host_model = np.zeros_like(fr)
        hinfo = dict(applied=False, reason="host=False" if not host else f"z >= {HOST_ZMAX}: no host decomposition")
        cd, cmodel, cinfo = fit_continuum(wr, fr, ir, fit_fe=fe)
    res["host_model"] = host_model; res["host_info"] = hinfo
    res["conti"] = cd; res["conti_model"] = cmodel
    res['continuum_info'] = {k: v for k, v in cinfo.items() if k != 'ps'}
    res["settings"] = dict(sig_broad_min=sig_broad_min, max_broad=max_broad, dbic=dbic,
                           oiii_wing=oiii_wing, heii=heii, host=host, fe=fe, err_floor=err_floor,
                           v_broad_max=V_BROAD_MAX, sii_mode="soft",
                           host_continuum_only=HOST_CONTINUUM_ONLY, o3_split=False,
                           o3_mode="order", o3_amp_order=O3_ORDER_AMPLITUDE,
                           o3_informed_start=O3_INFORMED_START, sys_prior_kms=SYS_PRIOR_KMS,
                           v_narrow_max=V_NARROW_MAX, broad_width_slope=BROAD_WIDTH_SLOPE,
                           nlr_wing=NLR_WING, use_ha_systemic=use_ha_systemic,
                           complexes=list(complexes), ebv=ebv, nmc=nmc, seed=seed,
                           mgii_narrow=mgii_narrow, mgii_doublet=mgii_doublet,
                           thresholds=dict(thresholds or {}), flux_scale=flux_scale)
    fsub = fr - host_model - cmodel
    res["flux_sub"] = fsub
    # A continuum solver that stopped short of convergence still leaves a
    # usable pseudo-continuum: the lines are fitted from it and the state is
    # recorded, as the production fitter did, so that the effect of such
    # continua can be counted in the catalogue rather than removed from it.
    solver = cinfo.get("solver")
    if solver is None:
        res["continuum_status"] = "unknown"
    else:
        res["continuum_status"] = "success" if solver.get("success", False) else "unconverged"
    continuum_converged = res["continuum_status"] == "success"
    dl = _lumdist_cm(z)

    line_kw = dict(oiii_wing=oiii_wing, heii=heii, mgii_narrow=mgii_narrow,
                   mgii_doublet=mgii_doublet, sig_broad_min=sig_broad_min)
    res["fits"], res["meas"], res["o3_prefit"], res["fit_status"] = _fit_line_sequence(
        wr, fsub, ir, cmodel, host_model, z, complexes, use_ha_systemic=use_ha_systemic,
        max_broad=max_broad, dbic=dbic, kw=line_kw, dl=dl)
    for status in res["fit_status"].values():
        status["continuum_converged"] = continuum_converged
    for m in res["meas"].values():
        m["host_frac"] = float(hinfo.get("host_frac_4200_5000", np.nan)) if hinfo.get("applied") else 0.0
        m["pl_alpha"] = float(cd.get("pl_alpha", np.nan))

    if nmc and nmc > 0:
        from ..errors import monte_carlo
        res["mc"], res["err"], res["mc_info"] = monte_carlo(
            res, nmc=nmc, seed=seed, return_diagnostics=True)
    for name, m in res["meas"].items():
        res["cls"][name] = classify(m, err=res["err"].get(name), t=thresholds)
    return res


def remeasure(res, thresholds=None):
    """Recompute the measures and classes of a stored result with the current
    ``measure_complex`` and ``classify``, keeping the stored fits."""
    host = res.get("host_model")
    if host is None:
        host = np.zeros_like(res["wave_rest"])
    hinfo = res.get("host_info", {}); cd = res.get("conti", {})
    old_meas = res.get("meas", {})
    res["meas"] = {}; res["cls"] = {}
    prior_snr = None
    for name in ("Halpha", "Hbeta", "MgII"):
        r = res["fits"].get(name)
        if r is None:
            continue
        m = measure_complex(r, res["conti_model"] + host, res["wave_rest"], res["z"],
                            dl_cm=_lumdist_cm(res["z"]), host_model=host)
        m["host_frac"] = float(hinfo.get("host_frac_4200_5000", np.nan)) if hinfo.get("applied") else 0.0
        m["pl_alpha"] = float(cd.get("pl_alpha", np.nan))
        src = old_meas.get(name, {}).get("systemic_source")
        # Prefer the fitted constraint over legacy metadata, which could report
        # an Halpha prior even when use_ha_systemic=False.
        if name == "Hbeta" and "ps" in r:
            src = "Halpha prior" if r["ps"].fixed.get("n_v") is not None else "[OIII] core"
        if src is None:
            src = ("Halpha prior" if (name == "Hbeta" and r["ps"].fixed.get("n_v") is not None)
                   else ("[OIII] core" if name == "Hbeta" else "own narrow group"))
        m["systemic_source"] = src
        if name == "Hbeta" and src == "Halpha prior" and prior_snr is not None:
            m["sys_snr"] = prior_snr
        if name == "Halpha":
            pre = res.get("o3_prefit", {})
            m["v_o3_pre"] = float(pre.get("v_o3", np.nan)); m["o3_pre_snr"] = float(pre.get("snr", 0.0))
        res["meas"][name] = m
        if name == "Halpha" and m["narrow_peak_snr"] >= NARROW_PRIOR_MIN_SNR and np.isfinite(m["v_sys"]):
            prior_snr = m["narrow_peak_snr"]
    for name, m in res["meas"].items():
        res["cls"][name] = classify(m, err=res.get("err", {}).get(name), t=thresholds)
    return res


def summary_row(res, prefix_meta=None):
    """Flatten a result into one dictionary suitable for a table row: continuum
    parameters (``conti_*``), host information, and per line (``HA_``, ``HB_``,
    ``MG_``) every measure, the Monte Carlo errors (``*_e_*``), the class, the
    reasons, the flags and the systemic source.

    The solver bookkeeping of ``fit_spectrum`` is exported when the result
    carries it (a result evaluated from stored parameters does not):
    ``continuum_status``, ``conti_at_bound`` (continuum parameters that ended
    at a bound, comma separated) and ``conti_feuv_fwhm_fixed`` from
    ``continuum_info``; per line ``*_fit_status`` and ``*_converged`` from
    ``fit_status``, and ``*_bic_margin`` (the distance of the chosen
    component count from the next in BIC) from the fit."""
    row = dict(prefix_meta or {})
    row["z_in"] = res["z"]
    row["host_applied"] = bool(res["host_info"].get("applied", False))
    row["host_frac"] = res["host_info"].get("host_frac_4200_5000", np.nan)
    for k in ("pl_alpha", "pl_norm", "feop_norm", "feop_fwhm", "feuv_norm"):
        row[f"conti_{k}"] = res["conti"].get(k, np.nan)
    if "continuum_status" in res:
        row["continuum_status"] = res["continuum_status"]
    cinfo = res.get("continuum_info")
    if cinfo is not None:
        row["conti_at_bound"] = ",".join(str(k) for k in (cinfo.get("at_bound") or []))
        row["conti_feuv_fwhm_fixed"] = bool(cinfo.get("feuv_fwhm_fixed", False))
    for name, status in res.get("fit_status", {}).items():
        row[f"{PREFIX[name]}_fit_status"] = status["status"]
        row[f"{PREFIX[name]}_converged"] = bool(status.get("converged", False))
    for name, m in res["meas"].items():
        p = PREFIX[name]
        for k in SUMMARY_KEYS:
            row[f"{p}_{k}"] = m.get(k, np.nan)
        if "bic_margin" in res["fits"].get(name, {}):
            row[f"{p}_bic_margin"] = res["fits"][name]["bic_margin"]
        for k, e in res["err"].get(name, {}).items():
            row[f"{p}_e_{k}"] = e
        c = res["cls"].get(name, {})
        row[f"{p}_class"] = c.get("label", "")
        row[f"{p}_reason"] = "; ".join(c.get("reasons", []))[:200]
        row[f"{p}_flags"] = ",".join(c.get("flags", []))
        row[f"{p}_systemic_source"] = m.get("systemic_source", "")
        mc_info = res.get("mc_info", {})
        if name in mc_info.get("lines", {}):
            md = mc_info["lines"][name]
            row[f"{p}_mc_n_requested"] = mc_info["n_requested"]
            row[f"{p}_mc_n_success"] = md["n_success"]
            row[f"{p}_mc_n_failed"] = md["n_failed"]
            row[f"{p}_mc_n_finite_c50_sys"] = md["n_finite"]["c50_sys"]
            row[f"{p}_mc_flags"] = ",".join(md["flags"])
            row[f"{p}_mc_uncertainty_model"] = mc_info["uncertainty_model"]
    return row
