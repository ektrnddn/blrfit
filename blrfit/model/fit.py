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
stopped short of convergence still gives a usable pseudo-continuum, and is used
and flagged. ``mc_info`` describes the conditional uncertainty model and draw
diagnostics.

Every velocity reported is a difference between two quantities measured in
the same fit, so the input redshift cancels from the offsets to first order.
It still decides where the narrow lines are sought (+/-1500 km/s of z; +/-700
for Mg II: a redshift wrong by more gives class X, not a shifted velocity),
which complexes the data cover, whether the host is fitted (z < 1.2), and the
systemic redshift and luminosity that are derived from it.

``settings`` records the configuration of every result. The keys
``sii_mode`` ('soft'), ``o3_mode`` ('order') and ``o3_split`` (False) name the
fixed choices for the [S II] tie and the [O III] core/wing ordering, whose
alternatives were rejected during development; they are kept so that results
written by earlier versions of the fitter can be compared. ``fe_uv_width_policy``
and ``fe_uv_fallback_kms`` record the treatment of the ultraviolet Fe II width
and ``host_guard`` whether the host was skipped where its window carries no
signal (``host_info['host_undetermined']``); results without these keys were
fitted with policy "A", the fallback at FE_UV_FWHM_FIXED_KMS and no guard.
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    ERR_FLOOR,
    HOST_ZMAX,
    SIG_BROAD_MIN,
    MAX_BROAD,
    DBIC,
    V_BROAD_MAX,
    HOST_CONTINUUM_ONLY,
    O3_ORDER_AMPLITUDE,
    O3_INFORMED_START,
    O3_START_MIN_SNR,
    SYS_PRIOR_KMS,
    SYS_PRIOR_MIN_SNR,
    V_NARROW_MAX,
    BROAD_WIDTH_SLOPE,
    NLR_WING,
    NARROW_PRIOR_MIN_SNR,
    FLUX_SCALE_MIN,
    FLUX_SCALE_MAX,
    FE_UV_WIDTH_POLICY,
    FE_UV_FWHM_FIXED_KMS,
    MC_ERROR_INVALID_FLAGS,
    MC_LINE_FLAGS,
    MASK_GROW_PIX,
)
from .extinction import deredden
from .continuum import fit_continuum, fit_continuum_host
from .lines import fit_complex, fit_complex_select
from ..measure import measure_complex
from ..classify import classify

SUMMARY_KEYS = (
    "v_sys",
    "sig_sys",
    "v_o3",
    "v_sii",
    "sig_sii",
    "z_sys",
    "v_peak",
    "centroid",
    "c25",
    "c50",
    "c75",
    "c90",
    "fwhm",
    "W25",
    "W75",
    "sigma_line",
    "skew",
    "AI",
    "KI",
    "n_peaks",
    "peak_sep",
    "dip_frac",
    "v_peak_sys",
    "centroid_sys",
    "c50_sys",
    "c25_sys",
    "c75_sys",
    "peak_top",
    "peak_top_sys",
    "centroid25_sys",
    "centroid50_sys",
    "broad_flux",
    "broad_ew",
    "broad_ew_agn",
    "conti_at_line",
    "broad_lum",
    "broad_peak_snr",
    "broad_flux_snr",
    "narrow_peak_snr",
    "sys_snr",
    "o3_core_snr",
    "v_o3_peak",
    "v_o3_pre",
    "o3_pre_snr",
    "nw_f",
    "nw_v",
    "nw_sig",
    "v_cover_lo",
    "v_cover_hi",
    "chi2_red",
    "n_broad",
    "v_single_gauss",
    "data_v_peak",
    "data_c50",
    "data_centroid_win",
    "dv_spread",
    "fwhm_spread",
    "n_equivalent",
    "n_residual_outliers",
)
PREFIX = {"Halpha": "HA", "Hbeta": "HB", "MgII": "MG"}


def grow_mask(bad, n):
    """``bad`` extended by ``n`` pixels on each side of every flagged pixel, in
    the order of the (sorted) spectrum; unchanged for ``n`` = 0."""
    bad = np.asarray(bad, bool)
    if n <= 0 or not bad.any() or bad.all():
        return bad
    out = bad.copy()
    for k in range(1, int(n) + 1):
        out[k:] |= bad[:-k]
        out[:-k] |= bad[k:]
    return out


def _lumdist_cm(z):
    """Luminosity distance for the Planck 2018 cosmology, in cm (NaN if astropy is missing)."""
    try:
        from astropy.cosmology import Planck18
        import astropy.units as u

        return float(Planck18.luminosity_distance(z).to(u.cm).value)
    except Exception:
        return np.nan


def _fit_line_sequence(
    wr,
    fsub,
    ir,
    cmodel,
    host_model,
    z,
    complexes,
    use_ha_systemic=True,
    max_broad=MAX_BROAD,
    dbic=DBIC,
    fixed_n_broad=None,
    kw=None,
    dl=np.nan,
):
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
            r, allfits = fit_complex_select(
                name, wr, fsub, ir, max_broad=max_broad, dbic=dbic, diagnostics=diag, **kws
            )
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
        m["systemic_source"] = (
            "Halpha prior" if ha_prior_used else ("[OIII] core" if name == "Hbeta" else "own narrow group")
        )
        if ha_prior_used:
            m["sys_snr"] = prior_snr
        if name == "Halpha":
            m["v_o3_pre"] = v_o3_pre
            m["o3_pre_snr"] = o3_pre_snr
        fits[name] = r
        meas[name] = m
        if name == "Halpha" and m["narrow_peak_snr"] >= NARROW_PRIOR_MIN_SNR and np.isfinite(m["v_sys"]):
            prior_v, prior_s, prior_snr = m["v_sys"], m["sig_sys"], m["narrow_peak_snr"]
            dd = r["d"]
            if "nw_f" in dd:
                prior_nw = (float(dd["nw_f"]), float(dd["nw_v"]), float(dd["nw_sig"]))

    return fits, meas, prefit, fit_status


def fit_spectrum(
    wave_obs,
    flux,
    ivar,
    z,
    ebv=0.0,
    host=True,
    fe=True,
    complexes=("Halpha", "Hbeta", "MgII"),
    max_broad=MAX_BROAD,
    dbic=DBIC,
    use_ha_systemic=True,
    oiii_wing=True,
    heii=True,
    mgii_narrow=True,
    mgii_doublet=False,
    nmc=0,
    seed=0,
    thresholds=None,
    sig_broad_min=SIG_BROAD_MIN,
    err_floor=ERR_FLOOR,
    flux_scale=1.0,
    fe_uv_width_policy=FE_UV_WIDTH_POLICY,
    fe_uv_fallback_kms=FE_UV_FWHM_FIXED_KMS,
    host_guard=True,
    mc_noise_policy="input",
    conti_multistart=True,
    mask_grow=MASK_GROW_PIX,
    jobs=1,
):
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
    jobs : worker processes for the Monte Carlo draws; the result does not depend on it
        (see ``errors.monte_carlo``), and it is not recorded
    mc_noise_policy : "input" uses the supplied pixel variance for perturbations;
        "effective" includes the fitting variance floor as independent noise for
        historical reproduction. Both retain the same ordinary fitting weights.
    thresholds : overrides of the classification thresholds (``DEFAULT_THRESH``)
    fe_uv_width_policy, fe_uv_fallback_kms : treatment of the ultraviolet Fe II width
        ("A", "B" or "C", see FE_UV_WIDTH_POLICY in constants.py) and the width it is
        held at under A and C; both recorded in ``settings``
    host_guard : skip the host when the 4200-5000 A window carries no signal
        (``host_info['host_undetermined']``; see fit_continuum_host); recorded in ``settings``
    conti_multistart : start the continuum fit from several points and keep the best
        (the default; see CONTI_START_ALPHAS in constants.py); False fits the first start
        only, as up to version 0.2; recorded in ``settings``
    mask_grow : pixels on each side of every unusable pixel that are excluded as well
        (MASK_GROW_PIX; 0 as up to version 0.2); recorded in ``settings``
    """
    complexes = tuple(complexes)
    if mc_noise_policy not in ("input", "effective"):
        raise ValueError("mc_noise_policy must be 'input' or 'effective'")
    flux_scale = float(flux_scale)
    if not (np.isfinite(flux_scale) and flux_scale > 0):
        raise ValueError(f"flux_scale must be a positive finite number, got {flux_scale!r}")
    zf, sbm = float(z), float(sig_broad_min)
    if not (np.isfinite(zf) and zf > -1.0):
        raise ValueError(f"z must be a finite number above -1, got {z!r}")
    if not (np.isfinite(sbm) and sbm >= 0):
        raise ValueError(f"sig_broad_min must be finite and non-negative, got {sig_broad_min!r}")
    wave_obs = np.asarray(wave_obs, float)
    flux = np.asarray(flux, float) * flux_scale
    ivar = np.asarray(ivar, float) / flux_scale**2
    wv = wave_obs[np.isfinite(wave_obs)]
    if wv.size > 1 and np.any(np.diff(wv) < 0):
        # descending (or unordered) wavelengths are fitted in increasing order
        order = np.argsort(wave_obs, kind="stable")
        wave_obs, flux, ivar = wave_obs[order], flux[order], ivar[order]
        wv = wave_obs[np.isfinite(wave_obs)]
    if wv.size > 1 and not np.median(np.diff(wv)) > 0:
        raise ValueError(
            "more than half of the wavelength steps are zero (repeated pixels, e.g. two exposures "
            "concatenated): combine the repeated wavelengths first"
        )
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    if bad.all():
        raise ValueError(
            "no usable pixel: every pixel has a non-finite flux or inverse variance, "
            "or an inverse variance <= 0 (a masked spectrum)"
        )
    bad = grow_mask(bad, mask_grow)
    flux = np.where(bad, 0.0, flux)
    ivar = np.where(bad, 0.0, ivar)
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
                "amplitude bounds and starting values of the model are set for that unit"
            )
    flux, ivar = deredden(wave_obs, flux, ivar, ebv)
    # Preserve measurement noise separately from the modelling floor used to
    # weight the fit. Both arrays undergo the same frame/unit transformation.
    ivar_stat = ivar.copy()
    # Error floor: a fractional flux-calibration/model term in quadrature.
    # The narrow-line cores of bright galaxies reach per-pixel S/N of several
    # hundred, where the residuals of any line model are limited by the
    # spectrophotometric calibration; without the floor they dominate
    # chi-square and drag the broad components.
    if err_floor and err_floor > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(ivar > 0, 1.0 / ivar, 0.0) + (err_floor * np.abs(flux)) ** 2
            ivar = np.where(ivar > 0, 1.0 / var, 0.0)
    wr = wave_obs / (1 + z)
    fr = flux * (1 + z)
    ir = ivar / (1 + z) ** 2

    res = dict(
        z=z,
        wave_rest=wr,
        flux_rest=fr,
        ivar_rest=ir,
        fits={},
        meas={},
        cls={},
        ivar_stat_rest=ivar_stat / (1 + z) ** 2,
        mc={},
        err={},
        mc_info=dict(status="not_requested"),
    )
    conti_kw = dict(
        fit_fe=fe,
        fe_uv_width_policy=fe_uv_width_policy,
        fe_uv_fallback_kms=fe_uv_fallback_kms,
        multistart=bool(conti_multistart),
    )
    if host and z < HOST_ZMAX:
        cd, ctotal, host_model, hinfo = fit_continuum_host(wr, fr, ir, host_guard=host_guard, **conti_kw)
        cmodel = ctotal - host_model  # power law + Fe II only
        cinfo = hinfo
    else:
        host_model = np.zeros_like(fr)
        hinfo = dict(
            applied=False, reason="host=False" if not host else f"z >= {HOST_ZMAX}: no host decomposition"
        )
        cd, cmodel, cinfo = fit_continuum(wr, fr, ir, **conti_kw)
    res["host_model"] = host_model
    res["host_info"] = hinfo
    res["conti"] = cd
    res["conti_model"] = cmodel
    res["continuum_info"] = {k: v for k, v in cinfo.items() if k != "ps"}
    res["settings"] = dict(
        sig_broad_min=sig_broad_min,
        max_broad=max_broad,
        dbic=dbic,
        oiii_wing=oiii_wing,
        heii=heii,
        host=host,
        fe=fe,
        err_floor=err_floor,
        v_broad_max=V_BROAD_MAX,
        sii_mode="soft",
        host_continuum_only=HOST_CONTINUUM_ONLY,
        o3_split=False,
        o3_mode="order",
        o3_amp_order=O3_ORDER_AMPLITUDE,
        o3_informed_start=O3_INFORMED_START,
        sys_prior_kms=SYS_PRIOR_KMS,
        v_narrow_max=V_NARROW_MAX,
        broad_width_slope=BROAD_WIDTH_SLOPE,
        nlr_wing=NLR_WING,
        use_ha_systemic=use_ha_systemic,
        complexes=list(complexes),
        ebv=ebv,
        nmc=nmc,
        seed=seed,
        mgii_narrow=mgii_narrow,
        mgii_doublet=mgii_doublet,
        thresholds=dict(thresholds or {}),
        flux_scale=flux_scale,
        fe_uv_width_policy=cinfo.get("feuv_policy", fe_uv_width_policy),
        fe_uv_fallback_kms=float(cinfo.get("feuv_fallback_kms", fe_uv_fallback_kms)),
        host_guard=bool(host_guard),
        mc_noise_policy=mc_noise_policy,
        conti_multistart=bool(conti_multistart),
        mask_grow=int(mask_grow),
    )
    fsub = fr - host_model - cmodel
    res["flux_sub"] = fsub
    # A continuum solver that stopped short of convergence still leaves a
    # usable pseudo-continuum: the lines are fitted from it and the state is
    # recorded, so that such fits are flagged rather than lost.
    solver = cinfo.get("solver")
    if solver is None:
        res["continuum_status"] = "unknown"
    else:
        res["continuum_status"] = "success" if solver.get("success", False) else "unconverged"
    continuum_converged = res["continuum_status"] == "success"
    dl = _lumdist_cm(z)

    line_kw = dict(
        oiii_wing=oiii_wing,
        heii=heii,
        mgii_narrow=mgii_narrow,
        mgii_doublet=mgii_doublet,
        sig_broad_min=sig_broad_min,
    )
    res["fits"], res["meas"], res["o3_prefit"], res["fit_status"] = _fit_line_sequence(
        wr,
        fsub,
        ir,
        cmodel,
        host_model,
        z,
        complexes,
        use_ha_systemic=use_ha_systemic,
        max_broad=max_broad,
        dbic=dbic,
        kw=line_kw,
        dl=dl,
    )
    for status in res["fit_status"].values():
        status["continuum_converged"] = continuum_converged
    for m in res["meas"].values():
        m["host_frac"] = float(hinfo.get("host_frac_4200_5000", np.nan)) if hinfo.get("applied") else 0.0
        m["pl_alpha"] = float(cd.get("pl_alpha", np.nan))

    if nmc and nmc > 0:
        from ..errors import monte_carlo

        res["mc"], res["err"], res["mc_info"] = monte_carlo(
            res, nmc=nmc, seed=seed, return_diagnostics=True, jobs=jobs
        )
    classify_lines(res, thresholds)
    return res


def classify_lines(res, thresholds=None):
    """Classify every measured line of ``res`` with its Monte Carlo errors.

    A line whose draws raised a flag of MC_ERROR_INVALID_FLAGS has its errors
    withheld: every entry of ``res['err'][name]`` becomes NaN (the percentiles
    stay in ``res['mc']``) and ``mc_info['lines'][name]['errors_withheld']``
    names the flags; class A then uses the offset threshold alone. The flags
    of MC_LINE_FLAGS are appended to the line's own flags."""
    lines = (res.get("mc_info") or {}).get("lines", {})
    for name, m in res["meas"].items():
        info = lines.get(name, {})
        mc_flags = list(info.get("flags", []))
        err = res.get("err", {}).get(name)
        withheld = [f for f in mc_flags if f in MC_ERROR_INVALID_FLAGS]
        if err and withheld:
            err = {k: np.nan for k in err}
            res["err"][name] = err
            info["errors_withheld"] = withheld
        c = classify(m, err=err, t=thresholds)
        c["flags"] = list(c["flags"]) + [f for f in mc_flags if f in MC_LINE_FLAGS and f not in c["flags"]]
        res["cls"][name] = c
    return res


def remeasure(res, thresholds=None):
    """Recompute the measures and classes of a stored result with the current
    ``measure_complex`` and ``classify``, keeping the stored fits."""
    host = res.get("host_model")
    if host is None:
        host = np.zeros_like(res["wave_rest"])
    hinfo = res.get("host_info", {})
    cd = res.get("conti", {})
    old_meas = res.get("meas", {})
    res["meas"] = {}
    res["cls"] = {}
    prior_snr = None
    for name in ("Halpha", "Hbeta", "MgII"):
        r = res["fits"].get(name)
        if r is None:
            continue
        m = measure_complex(
            r,
            res["conti_model"] + host,
            res["wave_rest"],
            res["z"],
            dl_cm=_lumdist_cm(res["z"]),
            host_model=host,
        )
        m["host_frac"] = float(hinfo.get("host_frac_4200_5000", np.nan)) if hinfo.get("applied") else 0.0
        m["pl_alpha"] = float(cd.get("pl_alpha", np.nan))
        src = old_meas.get(name, {}).get("systemic_source")
        # Prefer the fitted constraint over legacy metadata, which could report
        # an Halpha prior even when use_ha_systemic=False.
        if name == "Hbeta" and "ps" in r:
            src = "Halpha prior" if r["ps"].fixed.get("n_v") is not None else "[OIII] core"
        if src is None:
            src = (
                "Halpha prior"
                if (name == "Hbeta" and r["ps"].fixed.get("n_v") is not None)
                else ("[OIII] core" if name == "Hbeta" else "own narrow group")
            )
        m["systemic_source"] = src
        if name == "Hbeta" and src == "Halpha prior" and prior_snr is not None:
            m["sys_snr"] = prior_snr
        if name == "Halpha":
            pre = res.get("o3_prefit", {})
            m["v_o3_pre"] = float(pre.get("v_o3", np.nan))
            m["o3_pre_snr"] = float(pre.get("snr", 0.0))
        res["meas"][name] = m
        if name == "Halpha" and m["narrow_peak_snr"] >= NARROW_PRIOR_MIN_SNR and np.isfinite(m["v_sys"]):
            prior_snr = m["narrow_peak_snr"]
    classify_lines(res, thresholds)
    return res


def summary_row(res, prefix_meta=None):
    """Flatten a result into one dictionary suitable for a table row: continuum
    parameters (``conti_*``), host information, and per line (``HA_``, ``HB_``,
    ``MG_``) every measure, the Monte Carlo errors (``*_e_*``), the class, the
    reasons, the flags and the systemic source.

    The solver bookkeeping of ``fit_spectrum`` is exported when the result
    carries it (a result evaluated from stored parameters does not):
    ``continuum_status``, ``conti_at_bound`` (continuum parameters that ended
    at a bound, comma separated), ``conti_feuv_fwhm_fixed`` and
    ``conti_start`` (the continuum start kept; 0 is the single start used up
    to version 0.2) from ``continuum_info``; per line ``*_fit_status`` and ``*_converged`` from
    ``fit_status``, and ``*_bic_margin`` (the smallest single-score change
    that changes the chosen component count) from the fit. MC counts separate
    contributing, finite and solver-converged draws; flags and their scalar
    diagnostics accompany the percentile errors."""
    row = dict(prefix_meta or {})
    row["z_in"] = res["z"]
    row["host_applied"] = bool(res["host_info"].get("applied", False))
    row["host_frac"] = res["host_info"].get("host_frac_4200_5000", np.nan)
    if "host_undetermined" in res["host_info"]:
        row["host_undetermined"] = bool(res["host_info"]["host_undetermined"])
    for k in ("pl_alpha", "pl_norm", "feop_norm", "feop_fwhm", "feuv_norm"):
        row[f"conti_{k}"] = res["conti"].get(k, np.nan)
    if "continuum_status" in res:
        row["continuum_status"] = res["continuum_status"]
    cinfo = res.get("continuum_info")
    if cinfo is not None:
        row["conti_at_bound"] = ",".join(str(k) for k in (cinfo.get("at_bound") or []))
        row["conti_feuv_fwhm_fixed"] = bool(cinfo.get("feuv_fwhm_fixed", False))
        if "start_selected" in cinfo:
            row["conti_start"] = int(cinfo["start_selected"])
    for name, status in res.get("fit_status", {}).items():
        row[f"{PREFIX[name]}_fit_status"] = status["status"]
        row[f"{PREFIX[name]}_converged"] = bool(status.get("converged", False))
    for name, m in res["meas"].items():
        p = PREFIX[name]
        for k in SUMMARY_KEYS:
            row[f"{p}_{k}"] = m.get(k, np.nan)
        if "params_at_bound" in m:
            row[f"{p}_params_at_bound"] = ",".join(m["params_at_bound"])
        if "bic_margin" in res["fits"].get(name, {}):
            row[f"{p}_bic_margin"] = res["fits"][name]["bic_margin"]
        if "bic_gap" in res["fits"].get(name, {}):
            row[f"{p}_bic_gap"] = res["fits"][name]["bic_gap"]
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
            for key in (
                "alias_fraction",
                "alias_n_assessed",
                "alias_n_unassessed",
                "mc_sigma",
                "mc_gap_kms",
                "n_converged_lines",
                "n_converged_both",
                "n_unconverged_lines",
                "n_unconverged_continuum",
            ):
                if key in md:
                    column = key if key.startswith("mc_") else "mc_" + key
                    row[f"{p}_{column}"] = md[key]
            row[f"{p}_mc_flags"] = ",".join(md["flags"])
            row[f"{p}_mc_uncertainty_model"] = mc_info["uncertainty_model"]
            if "noise_policy" in mc_info:
                row[f"{p}_mc_noise_policy"] = mc_info["noise_policy"]
                row[f"{p}_mc_noise_variance_source"] = mc_info["noise_variance_source"]
    return row
