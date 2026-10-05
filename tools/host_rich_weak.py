"""Host-guard accuracy on the declared weak, host-rich constructed spectra.

Run from the package checkout, with a NEW output directory::

    PYTHONPATH=. python tools/host_rich_weak.py --snr-kind sum --line both \
        --n 20 --jobs 4 --out /absolute/path/to/new/host_sum_run

The default is the S/N of the 4200--5000 A sum, which the guard uses.
The model has host fraction 0.3--0.6 at 4700 A and band S/N 1--5.
Every noise realization is fit with the guard, without a host, and with the
host allowed (guard disabled). All fit failures, classifications, solver
states and offsets are retained, including unmeasurable lines. No detection
or accuracy success is inferred from an empty retained sample.

The bootstrap interval is conditional on each fixed constructed spectral
model and independent simulated pixel noise, not an interval across physical
objects. The bias decision uses at least 20 measurable realizations per cell;
other cases are inconclusive. Every cell and line is reported separately.
A failed PL+Fe-only bias interval triggers the declared host-free-refit review;
no aggregate opposite-sign cancellation or population/coverage claim is made.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import scipy

import blrfit
from blrfit.classify import is_measurable
from blrfit.constants import C_KMS, LAM, R_NII, R_OIII, S2F
from blrfit.model.continuum import HOST_WINDOW, pca_templates
from blrfit.validation import tolerance_decision

DESI_WAVE = np.arange(3600.0, 9824.01, 0.8)


def gauss_v(wave_rest, lam0, v, sigma, flux):
    lc = lam0 * (1.0 + v / C_KMS)
    sl = lc * sigma / C_KMS
    return flux / (sl * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wave_rest - lc) / sl) ** 2)


def make_spectrum(
    z,
    host_frac,
    snr,
    snr_kind,
    v_broad,
    fwhm,
    ew,
    rng,
    pl_norm=10.0,
    pl_alpha=-1.5,
    narrow_sigma=150.0,
    ew_ha=40.0,
    noise_profile="uniform",
    red_pixel_snr=15.0,
):
    """One host-rich spectrum on the DESI grid; returns (wave, flux, ivar, truth)."""
    wr = DESI_WAVE / (1 + z)
    pl = pl_norm * (wr / 3000.0) ** pl_alpha
    P = pca_templates()
    e0 = np.clip(np.interp(wr, P["gw"], P["gp"][0], left=0.0, right=0.0), 0, None)
    ref = float(np.interp(4700.0, P["gw"], P["gp"][0]))
    agn_ref = float(pl_norm * (4700.0 / 3000.0) ** pl_alpha)
    host = e0 / ref * agn_ref * host_frac / (1.0 - host_frac)
    model = pl + host
    c_ha = float(np.interp(LAM["Halpha"], wr, pl))
    f_ha = ew_ha * c_ha
    lines = [
        (LAM["Halpha"], f_ha),
        (LAM["NII6584"], f_ha),
        (LAM["NII6548"], f_ha / R_NII),
        (LAM["SII6716"], 0.4 * f_ha),
        (LAM["SII6731"], 0.4 * f_ha / 1.3),
        (LAM["Hbeta"], f_ha / 3.0),
        (LAM["OIII5007"], 8.0 * f_ha / 3.0),
        (LAM["OIII4959"], 8.0 * f_ha / 3.0 / R_OIII),
    ]
    for lam0, f in lines:
        model = model + gauss_v(wr, lam0, 0.0, narrow_sigma, f)
    for line in ("Halpha", "Hbeta"):
        c_line = float(np.interp(LAM[line], wr, pl))
        model = model + gauss_v(wr, LAM[line], v_broad, fwhm / S2F, ew * c_line)
    win = (wr > HOST_WINDOW[0]) & (wr < HOST_WINDOW[1])
    if snr_kind == "pixel":
        noise = float(np.median(model[win])) / snr
    else:
        # sum(f) / sqrt(N sigma^2) = snr over the window
        noise = float(np.sum(model[win])) / (snr * np.sqrt(win.sum()))
    sigma = np.full_like(model, noise)
    red_noise = noise
    if noise_profile == "blue_weak":
        red = (wr > 6800) & (wr < 6900)
        red_noise = float(np.median((pl + host)[red])) / red_pixel_snr
        sigma[~win] = red_noise
    elif noise_profile != "uniform":
        raise ValueError("unknown noise profile")
    flux = model + rng.standard_normal(model.size) * sigma
    ivar = 1.0 / sigma**2
    truth = dict(
        c50_sys=v_broad,
        host_frac_at_4700=host_frac,
        noise=noise,
        requested_band_snr=snr,
        snr_kind=snr_kind,
        n_band_pixels=int(win.sum()),
        expected_sum_snr=float(np.sum(model[win]) / (noise * np.sqrt(win.sum()))),
        fwhm=fwhm,
        equivalent_width=ew,
        noise_profile=noise_profile,
        red_noise=red_noise,
        red_pixel_snr=red_pixel_snr if noise_profile == "blue_weak" else None,
    )
    return DESI_WAVE, flux, ivar, truth


def nmad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


MODES = {"default": {}, "plfe": {"host": False}, "host_free": {"host_guard": False}}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, doc):
    with Path(path).open("x") as f:
        json.dump(clean_json(doc), f, indent=2, allow_nan=False)
        f.write("\n")


def source_identity():
    package = Path(blrfit.__file__).resolve().parent
    paths = [Path(__file__).resolve()] + sorted(package.rglob("*.py"))
    paths += sorted(p for p in (package / "templates").iterdir() if p.is_file())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def array_identity(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.asarray(array, dtype="<f8")
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def run_realization(task):
    config, cell_index, host_frac, snr, realization = task
    seed_parts = [config["seed"], cell_index, realization]
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    wave, flux, ivar, truth = make_spectrum(
        config["z"],
        host_frac,
        snr,
        config["snr_kind"],
        config["v"],
        config["fwhm"],
        config["ew"],
        rng,
        noise_profile=config["noise_profile"],
        red_pixel_snr=config["red_pixel_snr"],
    )
    record = dict(
        cell_index=cell_index,
        realization=realization,
        seed_sequence=seed_parts,
        host_fraction=host_frac,
        requested_snr=snr,
        truth=truth,
        input_sha256=array_identity(wave, flux, ivar),
        fits={},
    )
    lines = ("Halpha", "Hbeta") if config["line"] == "both" else (config["line"],)
    for mode, kwargs in MODES.items():
        start = time.monotonic()
        try:
            res = blrfit.fit_spectrum(wave, flux, ivar, config["z"], complexes=lines, **kwargs)
            hi = res["host_info"]
            fit = dict(
                status="returned",
                settings=res["settings"],
                host_undetermined=bool(hi.get("host_undetermined", False)),
                host_applied=bool(hi.get("applied", False)),
                host_info=hi,
                continuum_status=res["continuum_status"],
                lines={},
            )
            for line in lines:
                meas, cls = res["meas"].get(line, {}), res["cls"].get(line, {})
                fit["lines"][line] = dict(
                    c50_sys=float(meas.get("c50_sys", np.nan)),
                    fwhm=float(meas.get("fwhm", np.nan)),
                    broad_flux_snr=float(meas.get("broad_flux_snr", np.nan)),
                    label=cls.get("label", ""),
                    flags=cls.get("flags", []),
                    line_status=res["fits"].get(line, {}).get("fit_status", "unknown"),
                    measurable=bool(
                        is_measurable(
                            cls.get("label", ""),
                            cls.get("flags", []),
                            meas.get("broad_flux_snr", np.nan),
                            meas.get("fwhm", np.nan),
                        )
                    ),
                )
        except Exception as exc:
            fit = dict(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        fit["elapsed_seconds"] = time.monotonic() - start
        record["fits"][mode] = fit
    return clean_json(record)


def summarize_cell(rows, line, mode, seed, bootstrap=2000, min_measurable=20):
    finite, measurable, labels, guard = [], [], {}, []
    errors = 0
    for row in rows:
        fit = row["fits"][mode]
        if fit["status"] == "error":
            errors += 1
            continue
        m = fit["lines"][line]
        labels[m["label"]] = labels.get(m["label"], 0) + 1
        guard.append(fit["host_undetermined"])
        if m["c50_sys"] is not None:
            delta = m["c50_sys"] - row["truth"]["c50_sys"]
            finite.append(delta)
            if m["measurable"]:
                measurable.append(delta)
    values = np.asarray(measurable)
    estimate = float(np.median(values)) if values.size else None
    interval = None
    if len(values) >= min_measurable:
        rng = np.random.default_rng(seed)
        draws = np.median(rng.choice(values, size=(bootstrap, len(values)), replace=True), axis=1)
        interval = np.percentile(draws, [2.5, 97.5]).tolist()
    tolerance = max(30.0, 0.05 * abs(rows[0]["truth"]["c50_sys"]))
    decision = tolerance_decision(
        estimate,
        interval,
        [-tolerance, tolerance],
        units="km/s",
        n_observations=len(values),
        n_groups=int(bool(values.size)),
        method="percentile bootstrap of independent noise realizations; conditional on one fixed synthetic model",
    )
    if len(values) < min_measurable:
        decision["reason"] = f"fewer than {min_measurable} measurable realizations; accuracy not established"
    return dict(
        mode=mode,
        line=line,
        n_requested=len(rows),
        n_errors=errors,
        n_finite=len(finite),
        n_measurable=len(measurable),
        measurable_fraction=len(measurable) / len(rows),
        labels=labels,
        n_host_undetermined=sum(guard),
        finite_bias_median=float(np.median(finite)) if finite else None,
        finite_bias_nmad=nmad(finite),
        measurable_bias_nmad=nmad(values),
        bias_decision=decision,
        bootstrap_replicates=bootstrap,
        bootstrap_seed=seed,
        minimum_measurable=min_measurable,
        physical_population_validation=False,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-frac", default="0.3,0.45,0.6")
    ap.add_argument("--snr", default="1,2,3,5")
    ap.add_argument("--snr-kind", choices=("pixel", "sum"), default="sum")
    ap.add_argument(
        "--noise-profile",
        choices=("uniform", "blue_weak"),
        default="uniform",
        help="separate follow-up family: weak blue host band with usable red-line data",
    )
    ap.add_argument("--red-pixel-snr", type=float, default=15.0)
    ap.add_argument("--cells", default="", help="optional host:snr pairs; roster order is recorded")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--z", type=float, default=0.1)
    ap.add_argument("--line", default="both", choices=("Halpha", "Hbeta", "both"))
    ap.add_argument("--v", type=float, default=600.0)
    ap.add_argument("--fwhm", type=float, default=4000.0)
    ap.add_argument("--ew", type=float, default=150.0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", required=True, help="NEW output directory (existing paths refused)")
    a = ap.parse_args(argv)
    cells = (
        [(float(h), float(s)) for h, s in (c.split(":") for c in a.cells.split(","))]
        if a.cells
        else [(float(h), float(s)) for h in a.host_frac.split(",") for s in a.snr.split(",")]
    )
    if (
        a.n < 1
        or a.jobs < 1
        or a.seed < 0
        or not np.isfinite(a.z)
        or a.z <= -1
        or not np.isfinite(a.v)
        or not np.isfinite(a.fwhm)
        or a.fwhm <= 0
        or not np.isfinite(a.ew)
        or a.ew <= 0
        or not np.isfinite(a.red_pixel_snr)
        or a.red_pixel_snr <= 0
        or any(not 0 < h < 1 or not np.isfinite(s) or s <= 0 for h, s in cells)
    ):
        ap.error("invalid grid, realization count, seed, jobs or spectrum parameters")
    output = Path(a.out)
    output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    protocol = dict(
        schema="blrfit-host-guard-experiment-1",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arguments=vars(a),
        cells=cells,
        modes=MODES,
        package_version=blrfit.__version__,
        package_import=blrfit.__file__,
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        source_identity=identity,
        bootstrap_replicates=2000,
        minimum_measurable=20,
        decision="95% interval wholly inside/outside +/-max(30 km/s, 5% of truth); otherwise inconclusive",
        estimand="median offset error conditional on measurable spectra of each fixed synthetic model",
        fallback="any failed PL+Fe-only cell triggers host-free-refit-and-report-both policy implementation; inconclusive never establishes safety",
        limitations=[
            "independent Gaussian pixel noise",
            "fixed package-generated host and broad-line models",
            "host fraction defined at 4700 A",
            "not V17 coverage or a physical-population accuracy claim",
        ],
    )
    write_json(output / "PROTOCOL.json", protocol)
    tasks = [(vars(a), i, h, s, j) for i, (h, s) in enumerate(cells) for j in range(a.n)]
    rows = []
    executor = (
        ProcessPoolExecutor(max_workers=a.jobs, mp_context=multiprocessing.get_context("spawn"))
        if a.jobs > 1
        else None
    )
    try:
        results = executor.map(run_realization, tasks) if executor else map(run_realization, tasks)
        with (output / "ROWS.jsonl").open("x") as journal:
            for row in results:
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                rows.append(row)
                print(f"completed {len(rows)}/{len(tasks)} spectra ({len(MODES)} fits each)", flush=True)
    finally:
        if executor:
            executor.shutdown()
    if source_identity() != identity:
        raise RuntimeError(
            "experiment source changed during execution; preserve partial evidence and rerun under one identity"
        )
    lines = ("Halpha", "Hbeta") if a.line == "both" else (a.line,)
    summaries = []
    for i, (h, s) in enumerate(cells):
        selected = [r for r in rows if r["cell_index"] == i]
        for j, line in enumerate(lines):
            for k, mode in enumerate(MODES):
                summary = summarize_cell(selected, line, mode, a.seed + 10000 + i * 100 + j * 10 + k)
                summary.update(cell_index=i, host_fraction=h, requested_snr=s)
                summaries.append(summary)
                print(
                    f"host={h:g} sum/pixel_snr={s:g} {line} {mode}: measurable={summary['n_measurable']}/{len(selected)} "
                    f"bias={summary['bias_decision']['estimate']} outcome={summary['bias_decision']['outcome']}",
                    flush=True,
                )
    failed = [r for r in summaries if r["mode"] == "plfe" and r["bias_decision"]["outcome"] == "fail"]
    inconclusive = [
        r for r in summaries if r["mode"] == "plfe" and r["bias_decision"]["outcome"] == "inconclusive"
    ]
    errors = sum(fit["status"] == "error" for row in rows for fit in row["fits"].values())
    report = dict(
        status="complete",
        n_spectra=len(rows),
        n_fits=len(rows) * len(MODES),
        n_fit_errors=errors,
        rows_sha256=hashlib.sha256((output / "ROWS.jsonl").read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((output / "PROTOCOL.json").read_bytes()).hexdigest(),
        results=summaries,
        fallback_required=bool(failed),
        plfe_accuracy="failed"
        if failed
        else "inconclusive"
        if inconclusive
        else "passed_in_constructed_cells",
        physical_population_accuracy_claim=False,
    )
    write_json(output / "RESULT.json", report)
    print(
        f"wrote {output / 'RESULT.json'}; PL+Fe accuracy {report['plfe_accuracy']}; fit errors {errors}",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
