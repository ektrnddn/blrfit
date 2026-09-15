"""
Uncertainties: Monte Carlo refits and the empirical repeat-spectrum error model.

Monte Carlo: the rest-frame spectrum is perturbed with Gaussian noise drawn
from its error array and refitted (continuum and lines), with the host model
held fixed and the number of broad components fixed to the one selected for
the unperturbed spectrum; the half-width of the 16th-84th percentile range
over the realisations is the error. The same approach is used by Shen et al.
(2013) and Liu et al. (2014). It measures the statistical error only.

Empirical model: the total error of Delta v, including the systematics of the
continuum and narrow-line decomposition, was measured on the sky from 8377
pairs of independent DESI spectra of the same objects. The per-measurement
error is max(650 km/s / sqrt(S/N), 45 km/s), with S/N the integrated broad-line
signal-to-noise ratio, inflated by 1.5 for strong offsets (|Delta v| >= 1000
km/s), whose profiles are broader and more complex. The DESI strong-offset
catalogue and its inspection pages apply the factor 1.5 to all their objects,
which are strong offsets by construction, so the two prescriptions coincide
there. The model was calibrated on broad Halpha in DESI spectra; for other
lines and instruments it is an indication, not a measurement.
"""
from __future__ import annotations

import numpy as np

from .constants import (ERR_MODEL_NORM_KMS, ERR_MODEL_FLOOR_KMS, ERR_MODEL_STRONG_FACTOR,
                        STRONG_OFFSET_KMS, MC_DEFAULT_N, MC_MIN_SAMPLES, MC_ALIAS_KMS,
                        MC_ALIAS_MAX_FRACTION)
from .model.continuum import fit_continuum

MC_KEYS = ["v_peak", "centroid", "c25", "c50", "c75", "c90", "fwhm", "W25", "W75",
           "sigma_line", "AI", "KI", "v_sys", "sig_sys", "v_o3", "broad_flux", "broad_ew",
           "v_peak_sys", "centroid_sys", "c50_sys", "c25_sys", "c75_sys", "n_peaks",
           "peak_top_sys", "centroid25_sys", "centroid50_sys"]
# Line statuses whose draw enters the percentile sample: a finite end point of
# the line solver contributes whether or not the solver reported convergence.
MC_CONTRIBUTING_STATUS = ("success", "success_unconverged")


def empirical_error(broad_flux_snr, dv=np.nan):
    """Per-measurement error of Delta v from the DESI repeat-spectrum calibration."""
    snr = np.asarray(broad_flux_snr, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.maximum(ERR_MODEL_NORM_KMS / np.sqrt(snr), ERR_MODEL_FLOOR_KMS)
    e = np.where(np.isfinite(snr) & (snr > 0), e, np.nan)
    strong = np.isfinite(dv) & (np.abs(dv) >= STRONG_OFFSET_KMS[0])
    e = np.where(strong, ERR_MODEL_STRONG_FACTOR * e, e)
    return float(e) if np.ndim(e) == 0 else e


def monte_carlo(res, nmc=MC_DEFAULT_N, seed=0, fe=None, use_ha_systemic=None,
                kw=None, return_diagnostics=False):
    """Refit perturbed spectra with the same line estimator as ``fit_spectrum``.

    Returns ``(mc, err)`` by default, or ``(mc, err, info)`` with
    ``return_diagnostics=True``. Defaults are inferred from the result's saved
    settings. The host, broad-component counts, redshift, extinction correction,
    pixel masks and input error array are held fixed; PL/Fe and all systemic
    starts, priors and transfers are recomputed in every draw. These conditional
    statistical errors do not include model-selection or host uncertainty.

    ``err`` is the half-width of the 16th-84th percentile range over the draws
    whose line solver returned a finite solution. A draw whose continuum or
    line solver stopped short of convergence is kept, as in 0.1.0, and
    recorded: ``draw['continuum_converged']``, the line status
    'success_unconverged', the per-line counts ``n_unconverged_continuum`` and
    ``n_unconverged_lines`` of such draws in the sample, and the flags
    'unconverged_continuum_draws' and 'unconverged_line_draws'. A non-finite
    continuum model still invalidates the draw.

    Alias diagnostic: in a perturbed spectrum the narrow group of the Halpha
    complex can settle on [N II] instead of narrow Halpha (a 674 or 943 km/s
    displacement), which moves v_sys and c50_sys of that draw into another
    basin; percentiles over a mixture of basins are not a statistical error.
    A draw is counted as aliased when its v_sys or c50_sys lies more than MC_ALIAS_KMS
    from the unperturbed estimate in ``res['meas']``. ``info`` records
    ``alias_count`` and ``alias_fraction`` (of the contributing draws, NaN if
    there are none) per line and raises the flag 'mc_multimodal' when the
    sorted draws split into two groups separated by more than MC_ALIAS_KMS with
    more than MC_ALIAS_MAX_FRACTION of the draws on each side; ``err`` of a flagged line is still the 16-84
    half-range and is not a valid statistical error. ``info`` also records
    failure and finite-sample counts, the actual systemic-reference choices and
    the per-draw offsets, so a small percentile error cannot conceal discarded
    draws or aliases. Percentiles alone are not a coverage calibration.
    """
    from .model.fit import _fit_line_sequence

    if isinstance(nmc, (bool, np.bool_)) or int(nmc) != nmc or nmc < 0:
        raise ValueError("nmc must be a non-negative integer")
    nmc = int(nmc)
    settings = res.get("settings", {})
    fe = settings.get("fe", True) if fe is None else bool(fe)
    use_ha_systemic = (settings.get("use_ha_systemic", True)
                       if use_ha_systemic is None else bool(use_ha_systemic))
    line_kw = {k: settings[k] for k in ("oiii_wing", "heii", "mgii_narrow",
               "mgii_doublet", "sig_broad_min") if k in settings}
    line_kw.update(kw or {})
    rng = np.random.default_rng(seed)
    wr, fr, ir = res["wave_rest"], res["flux_rest"], res["ivar_rest"]
    sig = np.where(ir > 0, 1.0 / np.sqrt(np.where(ir > 0, ir, 1)), 0.0)
    z, host = res["z"], res["host_model"]
    counts = {name: r["n_broad"] for name, r in res["fits"].items()}
    complexes = settings.get("complexes", tuple(counts))
    samples = {name: {k: np.full(nmc, np.nan) for k in MC_KEYS} for name in counts}
    info = dict(status="complete", n_requested=nmc, seed=seed,
                uncertainty_model="conditional_statistical",
                fixed=["host_model", "broad_component_counts", "redshift", "extinction",
                       "pixel_mask", "input_error_array"],
                refitted=["power_law", "Fe_II_if_enabled", "OIII_prefit", "line_parameters",
                          "systemic_starts_priors_and_transfers"],
                component_counts=counts, fe=fe, use_ha_systemic=use_ha_systemic,
                line_settings=line_kw, continuum_convergence_checked=True,
                main_continuum_domain_matched=not bool(res.get('host_info', {}).get('applied')),
                continuum_domain='standard continuum windows; fixed-host conditional refit',
                multimodality_assessed=True, n_unconverged_continuum=0, draws=[], lines={})
    for i in range(nmc):
        draw = dict(index=i, lines={})
        try:
            f_i = fr + rng.standard_normal(fr.size) * sig
            fh = f_i - host
            _, cmodel, cinfo = fit_continuum(wr, fh, ir, fit_fe=fe)
            draw['continuum_solver'] = cinfo.get('solver', {})
            # An unconverged continuum solver keeps its draw and is counted per
            # line below (as in 0.1.0); only a non-finite model invalidates it.
            draw['continuum_converged'] = bool(cinfo.get('solver', {}).get('success', False))
            if not draw['continuum_converged']:
                info['n_unconverged_continuum'] += 1
            if not np.all(np.isfinite(cmodel)):
                raise ValueError("non-finite continuum model")
            draw["continuum_fallback"] = bool(cinfo.get("fallback", False))
            _, measures, prefit, fit_status = _fit_line_sequence(
                wr, fh - cmodel, ir, cmodel, host, z, complexes,
                use_ha_systemic=use_ha_systemic, fixed_n_broad=counts, kw=line_kw)
            draw["o3_prefit"] = {k: prefit[k] for k in ("status", "v_o3", "snr")}
            for name in counts:
                status = fit_status.get(name, dict(status="not_attempted"))
                record = dict(status=status["status"])
                if name in measures:
                    m = measures[name]
                    record["systemic_source"] = m["systemic_source"]
                    for k in MC_KEYS:
                        samples[name][k][i] = m.get(k, np.nan)
                if status["status"] != "success":
                    record["diagnostics"] = status
                draw["lines"][name] = record
        except Exception as exc:
            draw["exception"] = f"{type(exc).__name__}: {exc}"
            # A partially recorded draw must never enter the percentile sample.
            for name in counts:
                for values in samples[name].values():
                    values[i] = np.nan
                draw["lines"][name] = dict(status="exception")
        info["draws"].append(draw)
    mc, err = {}, {}
    for name, dd in samples.items():
        mc[name], err[name] = {}, {}
        statuses = [d["lines"][name]["status"] for d in info["draws"]]
        ok = np.array([s in MC_CONTRIBUTING_STATUS for s in statuses], bool)
        n_success = int(ok.sum())
        line_info = dict(n_success=n_success, n_failed=nmc - n_success,
                         n_finite={}, flags=[], samples={})
        # Unconverged solvers keep their draws (as in 0.1.0) and are counted here.
        line_info["n_unconverged_lines"] = statuses.count("success_unconverged")
        converged = np.array([d.get("continuum_converged", True) for d in info["draws"]], bool)
        line_info["n_unconverged_continuum"] = int(np.sum(ok & ~converged))
        sources = [d["lines"][name].get("systemic_source") for d in info["draws"]]
        line_info["systemic_source_counts"] = {s: sources.count(s) for s in sorted(set(sources) - {None})}
        if n_success < nmc:
            line_info["flags"].append("failed_draws")
        if len(line_info["systemic_source_counts"]) > 1:
            line_info["flags"].append("systemic_reference_changed")
        if line_info["n_unconverged_lines"]:
            line_info["flags"].append("unconverged_line_draws")
        if line_info["n_unconverged_continuum"]:
            line_info["flags"].append("unconverged_continuum_draws")
        # Alias diagnostic: a contributing draw more than MC_ALIAS_KMS from the
        # unperturbed estimate in v_sys or c50_sys sits in another basin.
        ref = res.get("meas", {}).get(name, {})
        ref_v, ref_c = float(ref.get("v_sys", np.nan)), float(ref.get("c50_sys", np.nan))
        with np.errstate(invalid="ignore"):
            aliased = ok & ((np.abs(dd["v_sys"] - ref_v) > MC_ALIAS_KMS)
                            | (np.abs(dd["c50_sys"] - ref_c) > MC_ALIAS_KMS))
        line_info["alias_reference"] = dict(v_sys=ref_v, c50_sys=ref_c)
        line_info["alias_count"] = int(aliased.sum())
        line_info["alias_fraction"] = float(aliased.sum() / n_success) if n_success else np.nan
        # The flag needs two separated groups of draws: a gap wider than
        # MC_ALIAS_KMS in the sorted v_sys or c50_sys with more than
        # MC_ALIAS_MAX_FRACTION of the contributing draws on each side. A wide
        # unimodal scatter (a large but honest error) does not qualify.
        bimodal = False
        for key in ("v_sys", "c50_sys"):
            a = np.sort(dd[key][ok & np.isfinite(dd[key])])
            if a.size >= MC_MIN_SAMPLES:
                gaps = np.diff(a); j = int(np.argmax(gaps))
                if gaps[j] > MC_ALIAS_KMS and min(j + 1, a.size - j - 1) / a.size > MC_ALIAS_MAX_FRACTION:
                    bimodal = True
        line_info["bimodal"] = bimodal
        if bimodal:
            line_info["flags"].append("mc_multimodal")
        for k, values in dd.items():
            a = values[np.isfinite(values)]
            line_info["n_finite"][k] = int(a.size)
            if a.size >= MC_MIN_SAMPLES:
                p16, p50, p84 = np.percentile(a, [16, 50, 84])
                mc[name][k] = (float(p16), float(p50), float(p84))
                err[name][k] = float(0.5 * (p84 - p16))
            else:
                mc[name][k] = (np.nan, np.nan, np.nan)
                err[name][k] = np.nan
        if line_info["n_finite"]["c50_sys"] < MC_MIN_SAMPLES:
            line_info["flags"].append("insufficient_offset_samples")
        # Retain draw alignment across lines for alias inspection and covariance.
        for k in ("v_sys", "c50_sys", "v_peak_sys"):
            line_info["samples"][k] = dd[k].tolist()
        info["lines"][name] = line_info
    if any(d["n_failed"] for d in info["lines"].values()):
        info["status"] = "partial_failure"
    if return_diagnostics:
        return mc, err, info
    return mc, err
