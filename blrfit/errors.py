"""
Uncertainties: Monte Carlo refits and the empirical repeat-spectrum error model.

Monte Carlo: the rest-frame spectrum is perturbed with Gaussian noise drawn
from its supplied statistical error array and refitted (continuum and lines), with the host model
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

from .constants import (
    ERR_MODEL_NORM_KMS,
    ERR_MODEL_FLOOR_KMS,
    ERR_MODEL_STRONG_FACTOR,
    STRONG_OFFSET_KMS,
    MC_DEFAULT_N,
    MC_MIN_SAMPLES,
    MC_ALIAS_KMS,
)
from .model.continuum import fit_continuum

MC_KEYS = [
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
    "AI",
    "KI",
    "v_sys",
    "sig_sys",
    "v_o3",
    "broad_flux",
    "broad_ew",
    "v_peak_sys",
    "centroid_sys",
    "c50_sys",
    "c25_sys",
    "c75_sys",
    "n_peaks",
    "peak_top_sys",
    "centroid25_sys",
    "centroid50_sys",
]
# Line statuses whose draw enters the percentile sample: a finite end point of
# the line solver contributes whether or not the solver reported convergence.
MC_CONTRIBUTING_STATUS = ("success", "success_unconverged")
# The offsets on which the draw sample is tested for several basins.
MC_BASIN_KEYS = ("v_sys", "c50_sys")

# Multimodality of the draw sample.
# The two-cluster test is characterised for 25 to 30 contributing draws; with
# fewer the flag 'mc_too_few' replaces it.
MC_MIN_CONTRIBUTING = 25
# Median-separation statistic: the sorted draws are split at their largest
# gap; the two sides are clusters when each holds at least this fraction of
# the draws and their medians differ by at least MC_ALIAS_KMS and at least
# MC_SEPARATION_NMAD_RATIO times hypot(NMAD_left, NMAD_right). Operating
# curve on 80/20 mixtures of sigma 100 km/s: 42 / 91 / 96 per cent detection
# at separations 500 / 800 / 1200 km/s for 30 draws (41 / 85 / 90 at 25),
# 0.25-0.33 per cent false flags on Gaussian sets of sigma 200-1000 km/s,
# 1.1 per cent on Student-t (nu = 3), 4 per cent on a skewed set. A wide
# unimodal scatter is an honest error and is not flagged.
MC_CLUSTER_MIN_FRACTION = 0.10
MC_SEPARATION_NMAD_RATIO = 4.0
# Floor of the within-cluster width, so that two point-like clusters qualify.
MC_NMAD_FLOOR_KMS = 1e-9
# Basin switch: the median of the draws lies more than max(MC_ALIAS_KMS,
# MC_BASIN_SWITCH_NMAD_RATIO x NMAD of the draws) from the unperturbed estimate.
MC_BASIN_SWITCH_NMAD_RATIO = 3.0


def nmad(x):
    """1.4826 times the median absolute deviation from the median; NaN when empty."""
    a = np.asarray(x, float)
    if a.size == 0:
        return np.nan
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def cluster_summary(draws):
    """Split the finite draws at the largest gap of their sorted values and
    summarise both sides: sizes, medians, NMADs, the gap, the separation of the
    medians, the within-cluster width hypot(NMAD_left, NMAD_right) and whether
    the pair qualifies as two clusters under the median-separation statistic.
    ``nmad_all`` is the NMAD of the whole sample (the export ``mc_sigma``)."""
    a = np.asarray(draws, float)
    a = np.sort(a[np.isfinite(a)])
    n = int(a.size)
    out = dict(
        n=n,
        nmad_all=nmad(a) if n else np.nan,
        gap_kms=np.nan,
        n_left=0,
        n_right=0,
        median_left=np.nan,
        median_right=np.nan,
        nmad_left=np.nan,
        nmad_right=np.nan,
        separation_kms=np.nan,
        width_kms=np.nan,
        min_fraction=np.nan,
        qualifies=False,
    )
    if n < 2:
        return out
    gaps = np.diff(a)
    j = int(np.argmax(gaps))
    left, right = a[: j + 1], a[j + 1 :]
    out.update(
        gap_kms=float(gaps[j]),
        n_left=int(left.size),
        n_right=int(right.size),
        median_left=float(np.median(left)),
        median_right=float(np.median(right)),
        nmad_left=nmad(left),
        nmad_right=nmad(right),
    )
    out["separation_kms"] = abs(out["median_right"] - out["median_left"])
    out["width_kms"] = max(float(np.hypot(out["nmad_left"], out["nmad_right"])), MC_NMAD_FLOOR_KMS)
    out["min_fraction"] = min(left.size, right.size) / n
    out["qualifies"] = bool(
        min(left.size, right.size) >= MC_CLUSTER_MIN_FRACTION * n
        and out["separation_kms"] >= MC_ALIAS_KMS
        and out["separation_kms"] >= MC_SEPARATION_NMAD_RATIO * out["width_kms"]
    )
    return out


def is_multimodal(draws):
    """The median-separation statistic of ``cluster_summary`` on one draw set."""
    return cluster_summary(draws)["qualifies"]


def basin_switch(draws, estimate):
    """Whether the median of the finite draws lies more than
    max(MC_ALIAS_KMS, MC_BASIN_SWITCH_NMAD_RATIO x NMAD) from ``estimate``.
    Assessed only with at least MC_MIN_SAMPLES draws and a finite estimate."""
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    out = dict(
        assessed=False,
        n=int(a.size),
        estimate=float(estimate),
        median=np.nan,
        nmad=np.nan,
        shift_kms=np.nan,
        threshold_kms=np.nan,
        switched=False,
    )
    if a.size < MC_MIN_SAMPLES or not np.isfinite(estimate):
        return out
    out.update(assessed=True, median=float(np.median(a)), nmad=nmad(a))
    out["shift_kms"] = abs(out["median"] - out["estimate"])
    out["threshold_kms"] = max(MC_ALIAS_KMS, MC_BASIN_SWITCH_NMAD_RATIO * out["nmad"])
    out["switched"] = bool(out["shift_kms"] > out["threshold_kms"])
    return out


def empirical_error(broad_flux_snr, dv=np.nan):
    """Per-measurement error of Delta v from the DESI repeat-spectrum calibration."""
    snr = np.asarray(broad_flux_snr, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.maximum(ERR_MODEL_NORM_KMS / np.sqrt(snr), ERR_MODEL_FLOOR_KMS)
    e = np.where(np.isfinite(snr) & (snr > 0), e, np.nan)
    strong = np.isfinite(dv) & (np.abs(dv) >= STRONG_OFFSET_KMS[0])
    e = np.where(strong, ERR_MODEL_STRONG_FACTOR * e, e)
    return float(e) if np.ndim(e) == 0 else e


_MC_CONTEXT = None


def _mc_init(context):
    global _MC_CONTEXT
    _MC_CONTEXT = context


def _mc_task(item):
    return _mc_draw(*item, _MC_CONTEXT)


def _mc_draw(i, deviates, context):
    """Draw ``i`` of ``monte_carlo``: refit the spectrum perturbed by
    ``deviates`` (standard normal, one per pixel) times the noise. Returns the
    draw record and the MC_KEYS values of every measured line ({} when the
    draw raised)."""
    from .model.fit import _fit_line_sequence

    wr, fr, ir, sig, host, z, complexes, counts, fe, continuum_kw, use_ha_systemic, line_kw = context
    draw = dict(index=i, lines={})
    values = {}
    try:
        f_i = fr + deviates * sig
        fh = f_i - host
        _, cmodel, cinfo = fit_continuum(wr, fh, ir, fit_fe=fe, **continuum_kw)
        draw["continuum_policy"] = {
            k: cinfo.get(k)
            for k in (
                "feuv_policy",
                "feuv_fallback_kms",
                "feuv_fwhm_fixed",
                "feuv_refit",
                "feuv_free_fit",
                "fe_width_state",
            )
        }
        draw["continuum_solver"] = cinfo.get("solver", {})
        # An unconverged continuum solver keeps its draw and is counted per
        # line; only a non-finite model invalidates it.
        draw["continuum_converged"] = bool(cinfo.get("solver", {}).get("success", False))
        if not np.all(np.isfinite(cmodel)):
            raise ValueError("non-finite continuum model")
        draw["continuum_fallback"] = bool(cinfo.get("fallback", False))
        _, measures, prefit, fit_status = _fit_line_sequence(
            wr,
            fh - cmodel,
            ir,
            cmodel,
            host,
            z,
            complexes,
            use_ha_systemic=use_ha_systemic,
            fixed_n_broad=counts,
            kw=line_kw,
        )
        draw["o3_prefit"] = {k: prefit[k] for k in ("status", "v_o3", "snr")}
        for name in counts:
            status = fit_status.get(name, dict(status="not_attempted"))
            record = dict(status=status["status"])
            if name in measures:
                m = measures[name]
                record["systemic_source"] = m["systemic_source"]
                values[name] = {k: m.get(k, np.nan) for k in MC_KEYS}
            if status["status"] != "success":
                record["diagnostics"] = status
            draw["lines"][name] = record
    except Exception as exc:
        draw["exception"] = f"{type(exc).__name__}: {exc}"
        # A partially recorded draw must never enter the percentile sample.
        values = {}
        for name in counts:
            draw["lines"][name] = dict(status="exception")
    return draw, values


def monte_carlo(
    res,
    nmc=MC_DEFAULT_N,
    seed=0,
    fe=None,
    use_ha_systemic=None,
    kw=None,
    return_diagnostics=False,
    noise_policy=None,
    jobs=1,
):
    """Refit perturbed spectra with the same line estimator as ``fit_spectrum``.

    Returns ``(mc, err)`` by default, or ``(mc, err, info)`` with
    ``return_diagnostics=True``. Defaults are inferred from the result's saved
    settings. The host, broad-component counts, redshift, extinction correction,
    pixel masks and input error array are held fixed; PL/Fe and all systemic
    starts, priors and transfers are recomputed in every draw. These conditional
    statistical errors do not include model-selection or host uncertainty.
    ``noise_policy='input'`` draws from ``ivar_stat_rest`` (before the fitting
    variance floor). The ordinary fitting weights remain ``ivar_rest`` in every
    draw. ``'effective'`` reproduces the old post-floor perturbation model; it is
    labelled effective-noise sensitivity, not calibrated statistical uncertainty.
    The default reads the saved ``mc_noise_policy``. Old results lacking that
    setting retain the effective policy with explicit legacy metadata. An input
    policy without its saved statistical variance refuses to run. With ``jobs``
    > 1 the draws are refitted in that many worker processes (see
    ``batch.ordered_map``); the deviates are drawn in the same order, so the
    result is that of a serial run.

    ``err`` is the half-width of the 16th-84th percentile range over the draws
    whose line solver returned a finite solution. A draw whose continuum or
    line solver stopped short of convergence is kept and recorded: ``draw['continuum_converged']``, the line status
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
    ``alias_count`` and ``alias_fraction`` per line, using only contributing
    draws with both finite offsets and finite reference estimates. The assessed
    and unassessed counts are recorded; the fraction is NaN if none is assessable.

    Two-cluster test (median-separation statistic, on v_sys and c50_sys of
    the contributing draws): the sorted draws are split at their largest gap;
    the flag 'mc_multimodal' is raised when each side holds at least
    MC_CLUSTER_MIN_FRACTION of the draws and the medians of the two sides
    differ by at least MC_ALIAS_KMS and at least MC_SEPARATION_NMAD_RATIO
    times hypot(NMAD_left, NMAD_right). The test runs with at least
    MC_MIN_CONTRIBUTING contributing draws, where its operating curve is
    measured (``tests/test_mc_flags.py``); with fewer the flag 'mc_too_few' is
    raised instead and the cluster summary is still recorded. The flag is a
    diagnostic with a measured sensitivity (about 91 per cent for an 80/20
    mixture 800 km/s apart at 30 draws), not a validated detector. A basin
    switch of the whole sample, |median of the draws - estimate| greater than
    max(MC_ALIAS_KMS, MC_BASIN_SWITCH_NMAD_RATIO x NMAD), raises
    'mc_basin_switch'. Per line ``info`` keeps ``clusters`` (sizes, medians,
    NMADs, gap, separation and width per key, from which the flag is
    recomputable), ``basin_switch`` per key, ``mc_sigma`` (the NMAD of the
    c50_sys draws) and ``mc_gap_kms`` (their largest gap). ``err`` of a
    flagged line is still the 16-84 half-range and is not a valid statistical
    error. ``info`` also records failure and finite-sample counts, the actual
    systemic-reference choices and the per-draw offsets, so a small percentile
    error cannot conceal discarded draws or aliases. Percentiles alone are not
    a coverage calibration.
    """
    if isinstance(nmc, (bool, np.bool_)) or int(nmc) != nmc or nmc < 0:
        raise ValueError("nmc must be a non-negative integer")
    nmc = int(nmc)
    settings = res.get("settings", {})
    legacy_noise_policy = noise_policy is None and "mc_noise_policy" not in settings
    noise_policy = settings.get("mc_noise_policy", "effective") if noise_policy is None else noise_policy
    if noise_policy not in ("input", "effective"):
        raise ValueError("MC noise policy must be 'input' or 'effective'")
    # Results predating the selectable policy used A and a 3000 km/s fallback,
    # and results predating 0.3 a single continuum start. Resolve those
    # historical defaults explicitly, even if a later release changes its
    # default; every draw refits under the recorded estimator.
    continuum_kw = dict(
        fe_uv_width_policy=settings.get("fe_uv_width_policy", "A"),
        fe_uv_fallback_kms=settings.get("fe_uv_fallback_kms", 3000.0),
        multistart=bool(settings.get("conti_multistart", False)),
    )
    fe = settings.get("fe", True) if fe is None else bool(fe)
    use_ha_systemic = (
        settings.get("use_ha_systemic", True) if use_ha_systemic is None else bool(use_ha_systemic)
    )
    line_kw = {
        k: settings[k]
        for k in ("oiii_wing", "heii", "mgii_narrow", "mgii_doublet", "sig_broad_min")
        if k in settings
    }
    line_kw.update(kw or {})
    rng = np.random.default_rng(seed)
    wr, fr, ir = res["wave_rest"], res["flux_rest"], res["ivar_rest"]
    noise_key = "ivar_stat_rest" if noise_policy == "input" else "ivar_rest"
    if noise_key not in res:
        raise ValueError("input MC noise requires saved ivar_stat_rest; do not infer it from floored weights")
    noise_ivar = np.asarray(res[noise_key], dtype=float)
    if (
        noise_ivar.shape != fr.shape
        or not np.all(np.isfinite(noise_ivar))
        or np.any(noise_ivar < 0)
        or not np.array_equal(noise_ivar > 0, ir > 0)
    ):
        raise ValueError("MC noise inverse variance must be finite, nonnegative and match the fitting mask")
    sig = np.where(noise_ivar > 0, 1.0 / np.sqrt(np.where(noise_ivar > 0, noise_ivar, 1)), 0.0)
    z, host = res["z"], res["host_model"]
    counts = {name: r["n_broad"] for name, r in res["fits"].items()}
    complexes = settings.get("complexes", tuple(counts))
    samples = {name: {k: np.full(nmc, np.nan) for k in MC_KEYS} for name in counts}
    info = dict(
        status="complete",
        n_requested=nmc,
        seed=seed,
        uncertainty_model=(
            "conditional_statistical" if noise_policy == "input" else "conditional_effective_noise"
        ),
        noise_policy=noise_policy,
        noise_variance_source=noise_key,
        fit_weight_variance_source="ivar_rest",
        legacy_noise_policy=legacy_noise_policy,
        fixed=[
            "host_model",
            "broad_component_counts",
            "redshift",
            "extinction",
            "pixel_mask",
            "input_error_array",
        ],
        refitted=[
            "power_law",
            "Fe_II_if_enabled",
            "OIII_prefit",
            "line_parameters",
            "systemic_starts_priors_and_transfers",
        ],
        component_counts=counts,
        fe=fe,
        use_ha_systemic=use_ha_systemic,
        line_settings=line_kw,
        continuum_settings=continuum_kw,
        continuum_convergence_checked=True,
        main_continuum_domain_matched=not bool(res.get("host_info", {}).get("applied")),
        continuum_domain="standard continuum windows; fixed-host conditional refit",
        multimodality_assessed=True,
        cluster_test=dict(
            statistic="median_separation",
            keys=list(MC_BASIN_KEYS),
            min_contributing=MC_MIN_CONTRIBUTING,
            min_fraction=MC_CLUSTER_MIN_FRACTION,
            separation_kms=MC_ALIAS_KMS,
            separation_nmad_ratio=MC_SEPARATION_NMAD_RATIO,
            nmad_floor_kms=MC_NMAD_FLOOR_KMS,
            basin_switch_kms=MC_ALIAS_KMS,
            basin_switch_nmad_ratio=MC_BASIN_SWITCH_NMAD_RATIO,
        ),
        n_unconverged_continuum=0,
        draws=[],
        lines={},
    )
    # The deviates are drawn in the order of the draws whether the draws run
    # here or in worker processes, so the sample does not depend on ``jobs``.
    deviates = (rng.standard_normal(fr.size) for _ in range(nmc))
    context = (wr, fr, ir, sig, host, z, complexes, counts, fe, continuum_kw, use_ha_systemic, line_kw)
    if jobs > 1 and nmc > 1:
        from .batch import ordered_map

        results = ordered_map(_mc_task, enumerate(deviates), jobs, initializer=_mc_init, initargs=(context,))
    else:
        results = (_mc_draw(i, d, context) for i, d in enumerate(deviates))
    for draw, values in results:
        if draw.get("continuum_converged") is False:
            info["n_unconverged_continuum"] += 1
        for name, line_values in values.items():
            for k in MC_KEYS:
                samples[name][k][draw["index"]] = line_values[k]
        info["draws"].append(draw)
    mc, err = {}, {}
    for name, dd in samples.items():
        mc[name], err[name] = {}, {}
        statuses = [d["lines"][name]["status"] for d in info["draws"]]
        ok = np.array([s in MC_CONTRIBUTING_STATUS for s in statuses], bool)
        n_success = int(ok.sum())
        line_info = dict(n_success=n_success, n_failed=nmc - n_success, n_finite={}, flags=[], samples={})
        # Unconverged solvers keep their draws and are counted here.
        line_info["n_unconverged_lines"] = statuses.count("success_unconverged")
        converged = np.array([d.get("continuum_converged", True) for d in info["draws"]], bool)
        line_info["n_unconverged_continuum"] = int(np.sum(ok & ~converged))
        line_info["n_converged_lines"] = statuses.count("success")
        line_info["n_converged_both"] = int(np.sum((np.asarray(statuses) == "success") & converged))
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
        assessed = (
            ok
            & np.isfinite(dd["v_sys"])
            & np.isfinite(dd["c50_sys"])
            & np.isfinite(ref_v)
            & np.isfinite(ref_c)
        )
        with np.errstate(invalid="ignore"):
            aliased = assessed & (
                (np.abs(dd["v_sys"] - ref_v) > MC_ALIAS_KMS) | (np.abs(dd["c50_sys"] - ref_c) > MC_ALIAS_KMS)
            )
        line_info["alias_reference"] = dict(v_sys=ref_v, c50_sys=ref_c)
        line_info["alias_count"] = int(aliased.sum())
        line_info["alias_n_assessed"] = int(assessed.sum())
        line_info["alias_n_unassessed"] = int(n_success - assessed.sum())
        line_info["alias_fraction"] = float(aliased.sum() / assessed.sum()) if assessed.any() else np.nan
        # Two-cluster test on the contributing draws of v_sys and c50_sys
        # (median-separation statistic, see the module constants); it needs
        # MC_MIN_CONTRIBUTING draws, below which 'mc_too_few' stands in for
        # it. The basin switch compares the median of the draws with the
        # unperturbed estimate. Both summaries are kept per key.
        line_info["n_contributing_required"] = MC_MIN_CONTRIBUTING
        line_info["clusters"] = {key: cluster_summary(dd[key][ok]) for key in MC_BASIN_KEYS}
        line_info["cluster_test_run"] = all(
            c["n"] >= MC_MIN_CONTRIBUTING for c in line_info["clusters"].values()
        )
        line_info["basin_switch"] = {
            key: basin_switch(dd[key][ok], {"v_sys": ref_v, "c50_sys": ref_c}[key]) for key in MC_BASIN_KEYS
        }
        line_info["mc_sigma"] = line_info["clusters"]["c50_sys"]["nmad_all"]
        line_info["mc_gap_kms"] = line_info["clusters"]["c50_sys"]["gap_kms"]
        bimodal = line_info["cluster_test_run"] and any(
            c["qualifies"] for c in line_info["clusters"].values()
        )
        line_info["bimodal"] = bool(bimodal)
        if not line_info["cluster_test_run"]:
            line_info["flags"].append("mc_too_few")
        if bimodal:
            line_info["flags"].append("mc_multimodal")
        if any(b["switched"] for b in line_info["basin_switch"].values()):
            line_info["flags"].append("mc_basin_switch")
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
