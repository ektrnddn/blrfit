# Derived explicitly from v17_measure.py for the October 3 confirmation.
# Changes: named linear/log grids and fixed illustrative observer-frame masks.
# Observed Gaussian widths are injected; no intrinsic-width deconvolution claim.
"""V17 known-truth spectra and measurements, without changing fitter defaults.

This module generates no catalogue products and performs no file writes. The
caller freezes the protocol, source/templates/environment and roster before
running it, and persists each returned record. Use separate processes with one
task at a time per process: the three *diagnostic* fits temporarily restrict a
module-local starting-velocity function. The ordinary fit and its 30 MC draws
run first with the production estimator unchanged; a diagnostic never replaces
that result, even when it is closer to truth.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import time
import traceback

import numpy as np
from scipy.ndimage import gaussian_filter1d

import blrfit
from blrfit.classify import is_measurable
from blrfit.constants import C_KMS, DEFAULT_THRESH, ERR_FLOOR, LAM, R_NII, R_OIII, S2F, TEMPLATE_DIR
from blrfit.model.continuum import pca_templates
from blrfit.model import lines as lines_module


LINES = ("Halpha", "Hbeta")
OFFSETS = (0, -300, 300, -1000, 1000, -2500, 2500)
WIDTHS = (3000, 5000, 8000)
SNRS = (8, 15, 30)
DIAGNOSTIC_STARTS = (0.0, 1500.0, -1500.0)
SCHEMA = "blrfit-v17-measurement-2"
# Fixed continuum-template convention, not estimated from V17 fit outcomes.
# Interpolating these intervals also removes stellar absorption inside them;
# this limits the generator's scope and is recorded explicitly in its protocol.
HOST_FEATURE_CENTERS = (3728.48, 3869.86, 3426.85, 4102.89, 4341.68, 4364.44,
                       4687.02, 5877.25, 6302.05, 6365.54, 7137.77, 7321.0,
                       7331.7, 9071.1, 9533.2,
                       LAM["Hbeta"], LAM["Halpha"], LAM["MgII"],
                       LAM["OIII5007"], LAM["OIII4959"], LAM["NII6548"],
                       LAM["NII6584"], LAM["SII6716"], LAM["SII6731"])


def clean_json(value):
    """Keep exact integers and explicitly null nonfinite numerical values."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def document_sha256(doc):
    return hashlib.sha256(json.dumps(clean_json(doc), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha256(*arrays):
    h = hashlib.sha256()
    for values in arrays:
        a = np.asarray(values, dtype="<f8")
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def _cell_id(nuisance, width, offset):
    return f"{nuisance}_w{width}_v{'m' if offset < 0 else 'p'}{abs(offset):04d}"


def _cells():
    cells = []
    for width in WIDTHS:
        for offset in OFFSETS:
            cells.append(dict(cell_id=_cell_id("base", width, offset), nuisance="base",
                              width_kms=width, offset_kms=offset))
    for nuisance in ("fe", "host", "weak_narrow", "all"):
        for offset in (0, 1000, 2500):
            cells.append(dict(cell_id=_cell_id(nuisance, 5000, offset), nuisance=nuisance,
                              width_kms=5000, offset_kms=offset))
    for i, cell in enumerate(cells):
        cell.update(cell_index=i, baseline_cell_id=_cell_id("base", cell["width_kms"],
                                                         cell["offset_kms"]))
    return cells


def make_protocol(seed=20260919):
    """The frozen V17 roster plus explicit, non-astrophysical generator choices."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return dict(
        schema="blrfit-v17-protocol-2", seed=int(seed), cells=_cells(),
        peak_snr=list(SNRS), primary_realizations=[0, 1], mc_draws=30,
        diagnostic_starts_kms=list(DIAGNOSTIC_STARTS),
        required_protocol=dict(n_cells=33, n_primary_spectra=198,
                               primary_mc_draws=5940, followup_realizations_per_trigger=10,
                               pooled_unit="equally weighted declared cells",
                               followups="separate diagnostics; never reweight primary pool"),
        generator_choices=dict(
            redshift=0.25, observed_wave_start=3600.0, observed_wave_stop=9824.0,
            observed_wave_step=0.8, pl_norm_at_3000=10.0, pl_alpha=-1.5,
            flux_units="observed f_lambda in1e-17 erg/s/cm2/observed-A; ivar in inverse squared units",
            pl_normalization_frame="10 observed density at observed wavelength3000*(1+z)",
            rest_transform="f_lambda_rest=(1+z)*f_lambda_observed; ivar_rest=ivar_observed/(1+z)^2",
            broad_profile="single Gaussian in optical velocity C*(lambda/lambda0-1)",
            broad_sampled_peak_flux=10.0,
            peak_snr_definition="sampled broad-only peak / supplied independent pixel-noise sigma",
            equal_balmer_peaks="synthetic design; not an astrophysical line-ratio claim",
            noise="constant sigma=10/requested_peak_snr; independent Gaussian pixels",
            fitter_error_floor="unchanged production default; achieved fitter S/N also recorded",
            host_fraction=0.6, host_reference_angstrom=4700.0,
            host_definition="host/(power law+host) at4700; excludes FeII and emission lines",
            host_template="nonnegative-clipped first Yip galaxy eigenspectrum, known narrow-feature intervals interpolated before normalization",
            host_feature_centers_angstrom=[float(x) for x in HOST_FEATURE_CENTERS],
            host_feature_halfwidth_kms=900.0,
            host_feature_interpolation="linear interpolation from pixels outside the union of fixed narrow-feature intervals",
            host_limit="controlled continuum template; removes stellar absorption in masked intervals too; no general stellar-population validation",
            fe_mean_ratio=0.2, fe_window_angstrom=[4434.0, 4684.0], fe_fwhm_kms=5000.0,
            fe_shift=0.0, fe_definition="mean optical FeII / mean AGN power law in4434-4684; not RFe",
            fe_generator="direct scipy Gaussian convolution of optical template; no fitted Fe operator",
            uv_fe_injected=False, narrow_sigma_kms=150.0,
            narrow_ha_ew_angstrom=40.0, weak_narrow_ha_ew_angstrom=5.0,
            weak_narrow_systemic_kms=700.0,
            displaced_group="all narrow Balmer+[NII]+[SII]+[OIII]; no relative OIII-SII shift",
            broad_center="true narrow-system velocity + requested c50_sys offset",
            narrow_flux_ratios=dict(NII6584_over_Halpha=1.0, NII6584_over_NII6548=float(R_NII),
                                    SII6716_over_Halpha=0.4, SII6716_over_SII6731=1.3,
                                    Halpha_over_Hbeta=3.0, OIII5007_over_Hbeta=8.0,
                                    OIII5007_over_OIII4959=float(R_OIII))),
        seed_policy=dict(noise="SeedSequence([seed,0,cell_index,snr_index,realization])",
                         mc="uint32 from SeedSequence([seed,1,cell_index,snr_index,realization])",
                         independence="independent cells, S/N levels, and realizations",
                         nuisance_comparison="unpaired differences; no common-noise variance claim"),
        estimator=dict(production="ordinary fit_spectrum defaults; Halpha+Hbeta; nmc30",
                       diagnostics="three separate nmc0 fits, one first-broad start each; never select using truth",
                       diagnostic_scope="lines.first_component_starts only; restored in finally; process-serial",
                       coverage="ordinary c50_sys +/-1 or2 times catalogue err_c50_sys, not MC median",
                       mc_error="(p84-p16)/2 of conditional refits; host/component count fixed",
                       independent_frame="actual [OIII] core reference only; tied/own-group reported separately"),
        limitations=["independent Gaussian pixel noise; no resampling correlations",
                     "no instrumental LSF convolution", "no real DESI error-calibration claim",
                     "single broad Gaussian and equal Balmer peaks are synthetic design choices",
                     "conditional MC fixes host and component counts; these uncertainties are not covered"],
        permitted_claim="conditional synthetic-grid behavior; no real-population, completeness, or binary-rate claim")


def _validate_protocol(protocol):
    # Numerical choices are fixed by this version, not freely tuned after a run.
    if protocol != make_protocol(protocol.get("seed")):
        raise ValueError("protocol differs from frozen v17_measure version")


def make_task(cell_id, snr, realization, protocol, *, stage="primary"):
    """Create primary tasks or separately labelled prescribed follow-up tasks."""
    _validate_protocol(protocol)
    found = [c for c in protocol["cells"] if c["cell_id"] == cell_id]
    if len(found) != 1 or snr not in SNRS:
        raise ValueError("unknown V17 cell or peak S/N")
    if isinstance(realization, bool) or not isinstance(realization, (int, np.integer)):
        raise ValueError("realization must be an integer")
    allowed = range(2) if stage == "primary" else range(2, 12) if stage == "followup" else ()
    if realization not in allowed:
        raise ValueError("stage/realization is outside frozen primary or ten-followup range")
    cell = found[0]
    seed_base = [protocol["seed"], 0, cell["cell_index"], SNRS.index(snr), int(realization)]
    mc_sequence = seed_base.copy(); mc_sequence[1] = 1
    mc_seed = int(np.random.SeedSequence(mc_sequence).generate_state(1, dtype=np.uint32)[0])
    return dict(**cell, snr=int(snr), realization=int(realization), stage=stage,
                task_id=f"{cell_id}_snr{snr:02d}_r{realization:02d}_{stage}",
                noise_seed_sequence=seed_base, mc_seed_sequence=mc_sequence, mc_seed=mc_seed)


def build_roster(protocol=None):
    protocol = make_protocol() if protocol is None else protocol
    _validate_protocol(protocol)
    return [make_task(c["cell_id"], snr, r, protocol)
            for c in protocol["cells"] for snr in SNRS for r in range(2)]


def followup_tasks(requests, protocol):
    """Ten distinct, deterministic follow-ups per requested cell/SNR stratum.

    ``requests`` contains dictionaries with cell_id and snr; repeated requests
    from different line/metric triggers do not run the same spectrum twice.
    The caller records why each stratum triggered before using these tasks.
    """
    strata = sorted({(request["cell_id"], request["snr"]) for request in requests})
    return [make_task(cell_id, snr, realization, protocol, stage="followup")
            for cell_id, snr in strata for realization in range(2, 12)]


build_followups = followup_tasks


def _validate_task(task, protocol):
    expected = make_task(task["cell_id"], task["snr"], task["realization"], protocol,
                         stage=task["stage"])
    if task != expected:
        raise ValueError("task differs from its frozen roster/seed identity")


def _gaussian(wave_rest, line, center_kms, sigma_kms):
    velocity = (wave_rest / LAM[line] - 1.0) * C_KMS
    return np.exp(-0.5 * ((velocity - center_kms) / sigma_kms) ** 2)


def _optical_fe(wave_rest, fwhm):
    """Independent direct convolution, rather than the fitter's FeTemplate call."""
    template = np.loadtxt(TEMPLATE_DIR / "fe_optical.txt")
    logwave, values = template[:, 0], template[:, 1] * 1e15
    values = np.where(np.isfinite(values), values, 0.0)
    pixel_kms = float(np.median(np.diff(logwave)) * np.log(10) * C_KMS)
    # The shipped I Zw1 template's declared intrinsic FWHM is900km/s.
    sigma_pixels = math.sqrt(fwhm ** 2 - 900.0 ** 2) / S2F / pixel_kms
    convolved = gaussian_filter1d(values, sigma_pixels, mode="nearest")
    out = np.zeros_like(wave_rest)
    covered = (wave_rest > 3686.0) & (wave_rest < 7484.0)
    out[covered] = np.interp(wave_rest[covered], 10 ** logwave, convolved)
    return out


def _host_continuum(wave_rest, raw_host, centers, halfwidth):
    """Direct generator interpolation; no fit or outcome-selected parameter."""
    masked = np.zeros(len(wave_rest), dtype=bool)
    for center in centers:
        masked |= np.abs(wave_rest / center - 1.0) * C_KMS < halfwidth
    if (~masked).sum() < 10:
        raise ValueError("host feature interpolation has insufficient continuum support")
    clean = np.asarray(raw_host, dtype=float).copy()
    clean[masked] = np.interp(wave_rest[masked], wave_rest[~masked], clean[~masked])
    return clean, masked


def generate_spectrum(task, protocol=None, grid="desi_linear"):
    """Return input arrays, noiseless components, and analytic Gaussian truth.

    Gaussian widths are defined in the input-redshift optical velocity frame,
    so the supplied FWHM and c50_sys are exact independent analytic truths.
    Normalizing sampled peaks only changes amplitudes, never centers or widths.
    """
    protocol = make_protocol() if protocol is None else protocol
    _validate_protocol(protocol); _validate_task(task, protocol)
    g = protocol["generator_choices"]
    if grid == "desi_linear":
        wave = np.arange(3600., 9824.01, 0.8)
    elif grid == "sdss_log":
        wave = 10 ** np.arange(np.log10(3600.), np.log10(9824.), 1e-4)
    else:
        raise ValueError("unknown frozen sampling grid")
    wr = wave / (1 + g["redshift"])
    pl = g["pl_norm_at_3000"] * (wr / 3000.0) ** g["pl_alpha"]
    nuisance = task["nuisance"]
    host = np.zeros_like(wave)
    host_mask = np.zeros(len(wave), dtype=bool)
    host_template_raw = np.zeros_like(wave)
    host_fraction = g["host_fraction"] if nuisance in ("host", "all") else 0.0
    if host_fraction:
        templates = pca_templates()
        host_shape = np.maximum(templates["gp"][0], 0.0)
        host_template_raw = np.interp(wr, templates["gw"], host_shape, left=0, right=0)
        host, host_mask = _host_continuum(wr, host_template_raw,
            g["host_feature_centers_angstrom"], g["host_feature_halfwidth_kms"])
        reference = float(np.interp(g["host_reference_angstrom"], wr, host))
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("host template has no positive reference flux")
        pl_ref = float(np.interp(g["host_reference_angstrom"], wr, pl))
        host *= pl_ref * host_fraction / (1 - host_fraction) / reference
    fe = np.zeros_like(wave)
    fe_ratio = g["fe_mean_ratio"] if nuisance in ("fe", "all") else 0.0
    window = (wr >= g["fe_window_angstrom"][0]) & (wr <= g["fe_window_angstrom"][1])
    if fe_ratio:
        fe = _optical_fe(wr, g["fe_fwhm_kms"])
        reference = float(np.mean(fe[window]))
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("Fe template has no positive normalization flux")
        fe *= fe_ratio * np.mean(pl[window]) / reference
    weak = nuisance in ("weak_narrow", "all")
    v_sys = g["weak_narrow_systemic_kms"] if weak else 0.0
    ew_ha = g["weak_narrow_ha_ew_angstrom"] if weak else g["narrow_ha_ew_angstrom"]
    pl_ha = g["pl_norm_at_3000"] * (LAM["Halpha"] / 3000) ** g["pl_alpha"]
    f_ha = ew_ha * pl_ha
    fluxes = dict(Halpha=f_ha, NII6584=f_ha, NII6548=f_ha / R_NII,
                  SII6716=0.4 * f_ha, SII6731=0.4 * f_ha / 1.3,
                  Hbeta=f_ha / 3, OIII5007=f_ha / 3 * 8, OIII4959=f_ha / 3 * 8 / R_OIII)
    narrow = np.zeros_like(wave)
    for line, integrated_flux in fluxes.items():
        sigma_lambda = LAM[line] * g["narrow_sigma_kms"] / C_KMS
        amplitude = integrated_flux / (sigma_lambda * math.sqrt(2 * math.pi))
        narrow += amplitude * _gaussian(wr, line, v_sys, g["narrow_sigma_kms"])
    sigma = g["broad_sampled_peak_flux"] / task["snr"]
    broad = {}; truths = {}
    center = v_sys + task["offset_kms"]
    for line in LINES:
        profile = _gaussian(wr, line, center, task["width_kms"] / S2F)
        amplitude = g["broad_sampled_peak_flux"] / float(profile.max())
        broad[line] = amplitude * profile
        integrated_flux = amplitude * (LAM[line] * task["width_kms"] / S2F / C_KMS) * math.sqrt(2 * math.pi)
        pl_at_line = g["pl_norm_at_3000"] * (LAM[line] / 3000) ** g["pl_alpha"]
        truths[line] = dict(c50_sys=float(task["offset_kms"]), c50=float(center),
                            fwhm=float(task["width_kms"]), v_sys=float(v_sys),
                            peak_snr=float(broad[line].max() / sigma),
                            analytic_peak_snr=float(amplitude / sigma),
                            broad_flux=float(integrated_flux * (1 + g["redshift"])),
                            broad_flux_definition="integrated observed flux, units1e-17 erg/s/cm2",
                            ew_agn_rest=float(integrated_flux / pl_at_line))
    model = pl + host + fe + narrow + broad["Halpha"] + broad["Hbeta"]
    noise = np.random.default_rng(np.random.SeedSequence(task["noise_seed_sequence"])).standard_normal(wave.size)
    flux = model + sigma * noise
    ivar = np.full_like(wave, 1 / sigma ** 2)
    for low, high in ((5576., 5581.), (6297., 6303.), (7590., 7605.)):
        ivar[(wave >= low) & (wave <= high)] = 0.
    # These fixed demonstration masks and diagonal noise are not a survey noise calibration.
    for line in LINES:
        peak_index = int(np.argmax(broad[line]))
        truths[line].update(
            noise_with_default_floor_at_peak=float(math.hypot(sigma, ERR_FLOOR * abs(flux[peak_index]))),
            peak_snr_after_default_floor_at_injected_peak=float(broad[line][peak_index] /
                math.hypot(sigma, ERR_FLOOR * abs(flux[peak_index]))),
            effective_snr_definition="broad truth at its sampled peak / pixel error after production floor on noisy flux; not fitted broad_peak_snr")
    truth = dict(lines=truths, sigma_pixel=float(sigma), z=g["redshift"],
                 baseline_cell_id=task["baseline_cell_id"], nuisance=nuisance,
                 host_fraction_at4700=host_fraction,
                 achieved_host_fraction_at4700=float(np.interp(4700, wr, host) /
                     np.interp(4700, wr, host + pl)),
                 host_narrow_features_removed=bool(host_fraction),
                 host_interpolated_pixels=int(host_mask.sum()),
                 host_template_raw_sha256=array_sha256(host_template_raw),
                 host_continuum_sha256=array_sha256(host),
                 fe_mean_ratio=float(np.mean(fe[window]) / np.mean(pl[window])),
                 true_narrow_system_kms=float(v_sys), narrow_ha_ew_angstrom=ew_ha,
                 narrow_line_fluxes={k: v * (1 + g["redshift"]) for k, v in fluxes.items()},
                 narrow_flux_units="integrated observed flux in1e-17 erg/s/cm2",
                 independent_generator_truth=True,
                 input_peak_snr_excludes_fitter_error_floor=True)
    return dict(wave=wave, flux=flux, ivar=ivar, truth=truth,
                noise_standard_normal=noise, noiseless=model,
                components=dict(power_law=pl, host=host, host_feature_mask=host_mask,
                                host_template_raw=host_template_raw,
                                fe=fe, narrow=narrow, broad=broad))


@contextmanager
def diagnostic_start(start_kms):
    """Restrict a diagnostic fit's starts; call serially within each process."""
    old = lines_module.first_component_starts
    def one_start(low, high, n_broad):
        return [float(np.clip(start_kms if n_broad >= 1 else 0.0, low + 1, high - 1))]
    lines_module.first_component_starts = one_start
    try:
        yield
    finally:
        lines_module.first_component_starts = old


def _finite(value):
    return value is not None and bool(np.isfinite(value))


def _fit_record(spectrum, protocol, mc_seed, nmc, fitter):
    start = time.monotonic()
    try:
        result = fitter(spectrum["wave"].copy(), spectrum["flux"].copy(), spectrum["ivar"].copy(),
                        spectrum["truth"]["z"], complexes=LINES, nmc=nmc, seed=mc_seed)
        info = result.get("mc_info", {})
        record = dict(status="returned", settings=result.get("settings", {}),
                      host_info=result.get("host_info", {}),
                      continuum_status=result.get("continuum_status", "unknown"),
                      continuum_info=result.get("continuum_info", {}),
                      fit_status=result.get("fit_status", {}), o3_prefit=result.get("o3_prefit", {}),
                      mc_info=info, lines={})
        for line in LINES:
            meas = result.get("meas", {}).get(line, {})
            cls = result.get("cls", {}).get(line, {})
            fit = result.get("fits", {}).get(line, {})
            err = result.get("err", {}).get(line, {})
            status = result.get("fit_status", {}).get(line, {})
            source = meas.get("systemic_source", "unknown")
            tied = source == "Halpha prior"
            # Own narrow-group and soft-tied SII estimates are reported, but
            # are not advertised as an independent [OIII]/[SII] validation.
            independent = (source == "[OIII] core" and _finite(meas.get("v_sys"))
                           and _finite(meas.get("sys_snr"))
                           and meas["sys_snr"] >= DEFAULT_THRESH["min_narrow_snr"])
            record["lines"][line] = dict(
                finite=all(_finite(meas.get(k)) for k in ("c50_sys", "fwhm")),
                measurable=bool(is_measurable(cls.get("label", ""), cls.get("flags", []),
                                              meas.get("broad_flux_snr", np.nan), meas.get("fwhm", np.nan))),
                c50_sys=meas.get("c50_sys"), fwhm=meas.get("fwhm"), v_sys=meas.get("v_sys"),
                label=cls.get("label", ""), flags=cls.get("flags", []),
                err_c50_sys=err.get("c50_sys"), err_fwhm=err.get("fwhm"),
                mc=info.get("lines", {}).get(line, {}), mc_status=info.get("status", "not_requested"),
                mc_n_requested=info.get("n_requested", nmc),
                systemic_source=source, frame_independent=independent, frame_tied=tied,
                frame_independence_reason=("independent OIII core measured" if independent else
                    "Halpha prior transferred to Hbeta" if tied else "own/unknown narrow frame; no independent-frame claim"),
                line_status=status.get("status", "missing"), fit_converged=fit.get("converged"),
                n_broad=fit.get("n_broad"), selection_score=fit.get("selection_score"),
                selection_score_kind=fit.get("selection_score_kind"), bic_margin=fit.get("bic_margin"),
                solver=fit.get("solver", {}), measures=meas, classification=cls,
                mc_percentiles=result.get("mc", {}).get(line, {}))
    except Exception as exc:
        record = dict(status="error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(), lines={})
    record["elapsed_seconds"] = time.monotonic() - start
    return clean_json(record)


def measure_spectrum(task, protocol=None, *, fitter=None):
    """One ordinary 30-MC measurement plus three separately labelled diagnostics.

    All failed and unmeasurable results are retained. This function applies no
    statistical pass criterion, excludes no cell, and writes no files.
    """
    protocol = make_protocol() if protocol is None else protocol
    _validate_protocol(protocol); _validate_task(task, protocol)
    fitter = blrfit.fit_spectrum if fitter is None else fitter
    record = dict(schema=SCHEMA, **task, protocol_sha256=document_sha256(protocol),
                  production={}, basin_diagnostics=[])
    started = time.monotonic()
    try:
        spectrum = generate_spectrum(task, protocol)
    except Exception as exc:
        record.update(status="generator_error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        return clean_json(record)
    record.update(truth=spectrum["truth"], input_sha256=array_sha256(spectrum["wave"], spectrum["flux"], spectrum["ivar"]),
                  noiseless_sha256=array_sha256(spectrum["noiseless"]),
                  noise_sha256=array_sha256(spectrum["noise_standard_normal"]))
    record["production"] = _fit_record(spectrum, protocol, task["mc_seed"], 30, fitter)
    for velocity in DIAGNOSTIC_STARTS:
        with diagnostic_start(velocity):
            diagnostic = _fit_record(spectrum, protocol, task["mc_seed"], 0, fitter)
        diagnostic.update(start_kms=velocity, diagnostic_only=True,
                          cannot_replace_production=True, mc_requested=0)
        record["basin_diagnostics"].append(diagnostic)
    record["status"] = "complete" if record["production"]["status"] == "returned" else "production_error"
    record["n_diagnostic_errors"] = sum(d["status"] == "error" for d in record["basin_diagnostics"])
    record["elapsed_seconds"] = time.monotonic() - started
    return clean_json(record)
