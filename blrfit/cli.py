"""
Command-line interface.

    blrfit fit   SPECTRUM [--targetid TID] [--z Z] [--ebv E] [...]   one spectrum -> table, JSON, figure
    blrfit fit   --ra RA --dec DEC [--out DIR] [...]                 the public DESI or SDSS spectra of a position
    blrfit fetch --ra RA --dec DEC [--out DIR] [...]                 download them without fitting
    blrfit pair  SPECTRUM SPECTRUM [...] [--out DIR]                 the epochs of one object: fits, velocity changes, tier
    blrfit tiers PAIRS [...] [--out FILE]                            the candidate tier of every object of pair tables

Successful point fits include explicit unavailable-line outcomes. Input or
fitting exceptions return a nonzero status; public batches retain failed products
in their manifests. The earlier between-epoch routines of ``blrfit.rv`` are
deprecated since 0.4.0 (removed in 0.5); ``blrfit pair`` supersedes them.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter

import numpy as np

from . import __version__
from .batch import ordered_map, read_list, read_rows, scalar, summary_table, table_format, write_table
from .constants import COMPLEX_WINDOW, MAX_BROAD, DBIC, ERR_FLOOR
from .io import read_spectrum, is_desi_coadd, is_sdss_spec
from .io.dust import sfd_ebv
from .io.sdss import mjd_to_date
from .model.fit import PREFIX, fit_spectrum, summary_row
from .classify import LABEL_TEXT, FLAG_TEXT, is_measurable, is_strong_offset, velocities_at_bound
from .errors import empirical_error, MC_MIN_CONTRIBUTING
from .pairs import PAIR_FLAG_TEXT, enumerate_pairs, epoch_snr, measure_pair, pair_row
from .physics import target_mass
from .tiers import TIERS, TIER_TEXT, classify_table

LINE_KEYS = (
    "v_peak_sys",
    "centroid_sys",
    "peak_top_sys",
    "centroid25_sys",
    "centroid50_sys",
    "c25_sys",
    "c75_sys",
    "c90_sys",
    "fwhm",
    "W25",
    "W75",
    "W90",
    "sigma_line",
    "skew",
    "AI",
    "KI",
    "n_peaks",
    "peak_sep",
    "dip_frac",
    "n_broad",
    "broad_flux",
    "broad_ew",
    "broad_ew_agn",
    "broad_lum",
    "broad_flux_snr",
    "broad_peak_snr",
    "v_sys",
    "sig_sys",
    "sys_snr",
    "narrow_peak_snr",
    "z_sys",
    "v_sii",
    "sig_sii",
    "v_o3",
    "v_o3_peak",
    "o3_core_snr",
    "nw_f",
    "nw_v",
    "nw_sig",
    "v_cover_lo",
    "v_cover_hi",
    "chi2_red",
    "v_single_gauss",
    "data_v_peak",
    "data_c50",
    "data_centroid_win",
    "dv_spread",
    "fwhm_spread",
    "n_equivalent",
    "n_residual_outliers",
    "params_at_bound",
    "host_frac",
    "pl_alpha",
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _clean(obj):
    """Make an object JSON-serialisable: numpy scalars to Python, NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    return obj


def _fmt(v, w=7, d=0, plus=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-".rjust(w)
    return f"{v:{'+' if plus else ''}{w}.{d}f}"


def _add_table_args(p):
    g = p.add_argument_group("generic table input (--survey generic: FITS table, CSV, ECSV or text)")
    g.add_argument(
        "--wave", default="wave", metavar="COL", help="wavelength column name or index (default wave)"
    )
    g.add_argument("--flux", default="flux", metavar="COL", help="flux column (default flux)")
    g.add_argument("--err", default=None, metavar="COL", help="1-sigma error column")
    g.add_argument("--ivar", default=None, metavar="COL", help="inverse-variance column (instead of --err)")
    g.add_argument("--wave-unit", default="angstrom", metavar="UNIT", help="angstrom (default), nm, um or m")
    g.add_argument(
        "--frame", default="obs", metavar="{obs,rest}", help="obs (default) or rest; rest needs --z"
    )
    g.add_argument("--air", action="store_true", help="wavelengths are in air (converted to vacuum)")
    g.add_argument(
        "--flux-scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="multiply flux and statistical error into 1e-17 erg/s/cm^2/observed-A",
    )
    g.add_argument("--hdu", type=int, default=1, metavar="N", help="extension of a FITS table (default 1)")
    g.add_argument(
        "--row", type=int, metavar="N", help="zero-based spectrum row for a vector table or 2-D image"
    )
    g.add_argument("--z-column", metavar="COL", help="redshift column for a generic table")
    g.add_argument("--z-key", metavar="KEY", help="FITS header redshift keyword (primary or --hdu)")
    g.add_argument("--mask", metavar="COL", help="generic table mask column; nonzero pixels excluded")
    g.add_argument(
        "--flux-frame",
        choices=("obs", "rest"),
        default="obs",
        help="flux-density frame, separately from the wavelength frame (default obs)",
    )
    for name in ("wave", "flux", "err", "ivar"):
        g.add_argument(
            f"--{name}-hdu",
            type=lambda s: int(s) if s.isdigit() else s,
            metavar="HDU",
            help=f"image HDU name or number containing {name}",
        )


def _table_kwargs(a):
    if a.err is None and a.ivar is None and a.err_hdu is None and a.ivar_hdu is None:
        sys.exit(
            "a table needs its error column: give --err <column> (1-sigma) or --ivar <column> (inverse variance)"
        )
    return dict(
        wave=a.wave,
        flux=a.flux,
        err=a.err,
        ivar=a.ivar,
        wave_unit=a.wave_unit,
        frame=a.frame,
        air=a.air,
        z=a.z,
        flux_scale=a.flux_scale,
        hdu=a.hdu,
        row=a.row,
        z_column=a.z_column,
        z_key=a.z_key,
        mask=a.mask,
        flux_frame=a.flux_frame,
        wave_hdu=a.wave_hdu,
        flux_hdu=a.flux_hdu,
        err_hdu=a.err_hdu,
        ivar_hdu=a.ivar_hdu,
    )


def _load(path, a, targetid=None):
    """Read a spectrum and settle the redshift and E(B-V). Returns (dict, z, z_source, ebv, ebv_source)."""
    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")
    try:
        survey = getattr(a, "survey", "auto")
        generic = survey == "generic" or (
            survey == "auto" and not is_desi_coadd(path) and not is_sdss_spec(path)
        )
        sp = read_spectrum(
            path,
            targetid=targetid,
            survey=survey,
            mask_policy=getattr(a, "sdss_mask_policy", "ivar"),
            redrock=getattr(a, "redrock", None),
            **(_table_kwargs(a) if generic else {}),
        )
    except (ValueError, KeyError, OSError) as e:
        sys.exit(f"cannot read {path}: {e}")
    if a.z is not None:
        z, zsrc = float(a.z), "argument"
    elif np.isfinite(sp.get("z", np.nan)):
        z, zsrc = float(sp["z"]), ("redrock" if sp.get("kind") == "desi" else "file")
    elif sp.get("kind") == "desi":
        sys.exit("no redshift: give --z, or place the matching redrock-*.fits file next to the coadd")
    else:
        sys.exit("no redshift: give --z")
    # Galactic E(B-V): the SFD98 value for every survey. DESI coadds carry it in
    # their FIBERMAP; for other inputs it is looked up at the file's coordinates
    # (or --ra/--dec) when dustmaps and its SFD map are installed, and otherwise
    # taken as zero with a warning and the flag ebv_assumed_zero.
    ra = a.ra if getattr(a, "ra", None) is not None else sp.get("ra", np.nan)
    dec = a.dec if getattr(a, "dec", None) is not None else sp.get("dec", np.nan)
    if a.ebv is None:
        if sp.get("kind") == "desi":
            ebv, esrc = float(sp["ebv"]), "FIBERMAP"
        else:
            ebv, why = sfd_ebv(ra, dec)
            ebv, esrc = (ebv, "SFD map") if ebv is not None else (0.0, f"assumed 0: {why}")
    elif str(a.ebv).lower() in ("sfd", "dust", "map"):
        ebv, why = sfd_ebv(ra, dec)
        if ebv is None:
            sys.exit(
                f"--ebv sfd: {why}; give coordinates (--ra, --dec) if the file has none, install dustmaps "
                '(pip install dustmaps) and fetch its map once: python -c "import dustmaps.sfd; dustmaps.sfd.fetch()"'
            )
        esrc = "SFD map"
    else:
        ebv, esrc = float(a.ebv), "argument"
    return sp, z, zsrc, ebv, esrc


def _stem(path, targetid=None):
    """Output stem of a local spectrum: its file name without the extension,
    and the TARGETID when one is given and the name does not contain it."""
    base, stripped = os.path.basename(path), False
    for suffix in (".gz", ".fits", ".fit"):
        if base.lower().endswith(suffix):
            base, stripped = base[: -len(suffix)], True
    if not stripped:
        base = os.path.splitext(base)[0]
    if targetid is not None and str(targetid) not in base:
        base = f"{base}-{int(targetid)}"
    return base


def _line_record(name, res):
    """The JSON record of one line."""
    if name not in res["fits"]:
        lo, hi = COMPLEX_WINDOW[name]
        state = res.get("fit_status", {}).get(name, {})
        status = state.get("status", "unknown")
        reason = (
            f"complex not fitted: {status}"
            if status not in ("unknown", "unusable_window", "uncovered_core")
            else f"complex not fitted: insufficient usable coverage in {lo:.0f}-{hi:.0f} A"
        )
        return dict(
            fitted=False,
            label="",
            class_text="not fitted",
            reasons=[reason],
            fit_status=state,
            flags=[],
            flag_text=[],
            measurable=False,
            strong_offset=False,
            dv=None,
        )
    m = res["meas"][name]
    c = res["cls"][name]
    e = res["err"].get(name, {})
    dv = m.get("c50_sys", np.nan)
    rec = dict(
        fitted=True,
        label=c["label"],
        class_text=LABEL_TEXT.get(c["label"], ""),
        reasons=list(c.get("reasons", [])),
        flags=list(c.get("flags", [])),
        flag_text=[FLAG_TEXT.get(f, f) for f in c.get("flags", [])],
        measurable=bool(
            is_measurable(
                c["label"], c.get("flags", []), m.get("broad_flux_snr", np.nan), m.get("fwhm", np.nan)
            )
        ),
        strong_offset=bool(
            is_strong_offset(
                c["label"], c.get("flags", []), m.get("broad_flux_snr", np.nan), m.get("fwhm", np.nan), dv
            )
        ),
        dv=dv,
        dv_err_mc=e.get("c50_sys", np.nan),
        # the repeat-spectrum model was calibrated on measurable lines: not reported for E, X, W
        dv_err_model=(
            empirical_error(m.get("broad_flux_snr", np.nan), dv)
            if c["label"] in ("A", "B", "C", "F")
            else np.nan
        ),
        systemic_source=m.get("systemic_source", ""),
        v_sii_minus_sys=(m.get("v_sii", np.nan) - m.get("v_sys", np.nan)),
        v_o3_minus_sys=(m.get("v_o3", np.nan) - m.get("v_sys", np.nan))
        if np.isfinite(m.get("v_o3", np.nan))
        else (m.get("v_o3_pre", np.nan) - m.get("v_sys", np.nan)),
        o3_snr=m.get("o3_core_snr", m.get("o3_pre_snr", np.nan)),
    )
    for k in LINE_KEYS:
        rec[k] = m.get(k, np.nan)
    rec["errors_mc"] = dict(e)
    rec["fit_status"] = res.get("fit_status", {}).get(name, {})
    rec["converged"] = bool(res["fits"][name].get("converged", True))
    rec["continuum_status"] = res.get("continuum_status", "unknown")
    rec["bic_margin"] = res["fits"][name].get("bic_margin", np.nan)
    rec["solver"] = res["fits"][name].get("solver", {})
    rec["fit_statistics"] = {
        k: res["fits"][name].get(k)
        for k in (
            "data_chi2",
            "penalty_chi2",
            "selection_score",
            "selection_score_kind",
            "data_score_at_penalized_fit",
        )
    }
    rec["bic_all"] = list(m.get("bic_all", []))
    return rec


def _print_fit_table(lines, res, out):
    hdr = (
        f"{'line':7} {'cls':3} {'dv=c50-sys':>12} {'+/-mc':>6} {'peak':>7} {'cen':>7} {'FWHM':>6} {'nb':>2} "
        f"{'fS/N':>6} {'pS/N':>5} {'A.I.':>6} {'K.I.':>5} {'v_sys':>6} {'sS/N':>5} {'[SII]':>6} {'[OIII]':>7} {'oS/N':>5} flags"
    )
    out(hdr)
    for name in ("Halpha", "Hbeta", "MgII"):
        if name not in lines:
            continue
        L = lines[name]
        if not L["fitted"]:
            out(f"{name:7} {'-':3} {L['reasons'][0]}")
            continue
        out(
            f"{name:7} {L['label']:3} {_fmt(L['dv'], 12, 0, True)} {_fmt(L['dv_err_mc'], 6)} "
            f"{_fmt(L['v_peak_sys'], 7, 0, True)} {_fmt(L['centroid_sys'], 7, 0, True)} {_fmt(L['fwhm'], 6)} {L['n_broad']:>2} "
            f"{_fmt(L['broad_flux_snr'], 6, 1)} {_fmt(L['broad_peak_snr'], 5, 1)} {_fmt(L['AI'], 6, 2, True)} {_fmt(L['KI'], 5, 2)} "
            f"{_fmt(L['v_sys'], 6, 0, True)} {_fmt(L['sys_snr'], 5, 1)} {_fmt(L['v_sii_minus_sys'], 6, 0, True)} "
            f"{_fmt(L['v_o3_minus_sys'], 7, 0, True)} {_fmt(L['o3_snr'], 5, 1)} {','.join(L['flags']) or '-'}"
        )
    for name in ("Halpha", "Hbeta", "MgII"):
        if name in lines and lines[name]["fitted"]:
            L = lines[name]
            out(
                f"  {name}: class {L['label']} ({L['class_text']}); {'; '.join(L['reasons'])}; "
                f"systemic from {L['systemic_source']}; measurable {L['measurable']}, strong offset {L['strong_offset']}"
            )
    hi = res["host_info"]
    n_gal = hi.get("n_gal", 0)
    out(
        f"  continuum: power-law slope {res['conti'].get('pl_alpha', np.nan):+.2f}; host "
        + (
            f"{hi.get('host_frac_4200_5000', np.nan):.2f} of the 4200-5000 A flux "
            f"({n_gal} eigenspectr{'um' if n_gal == 1 else 'a'})"
            if hi.get("applied")
            else f"not used ({hi.get('reason', '')})"
        )
    )
    # each flag once, with its meaning; for a velocity on a bound, which one
    shown = {}
    for name, L in lines.items():
        if not L["fitted"]:
            continue
        for flag, text in zip(L["flags"], L["flag_text"]):
            where = shown.setdefault(flag, (text, []))[1]
            if flag == "param_at_bound":
                where.append(f"{name} " + ", ".join(velocities_at_bound(L["params_at_bound"])))
    for flag, (text, where) in shown.items():
        out(f"  {flag}: {text}" + (f" ({'; '.join(where)})" if where else ""))
    out(
        "  velocities in km/s; dv = c50 - v_sys with c50 the half-maximum bisector c(1/2); "
        "+/-mc from the Monte Carlo (--nmc); columns: blrfit fit --help"
    )


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def _parse_lines(text):
    names = {k.lower(): k for k in COMPLEX_WINDOW}
    lines = [names.get(s.strip().lower(), s.strip()) for s in text.split(",") if s.strip()]
    bad = [s for s in lines if s not in COMPLEX_WINDOW]
    if bad:
        sys.exit(f"unknown line(s) {bad}; choose from {list(COMPLEX_WINDOW)}")
    return lines


def _silent(*args):
    pass


def fit_file(a, path, targetid=None, z=None, stem=None, out=print, jobs=1, results=None):
    """Fit one local spectrum with the command-line options ``a`` and write its
    products. ``targetid`` and ``z`` (None: from the file or ``a.z``) and the
    output ``stem`` are those of this spectrum; ``jobs`` processes refit its
    Monte Carlo draws. Returns the spectrum's catalogue record (identity
    columns and summary row). A dictionary ``results`` receives, under the stem,
    the fit itself with the epoch's identity (``pair`` needs it)."""
    if z is not None:
        a = copy.copy(a)
        a.z = z
    sp, z, zsrc, ebv, esrc = _load(path, a, targetid)
    lines = _parse_lines(a.lines)
    out(
        f"blrfit {__version__}: {path}"
        + (f" TARGETID {int(targetid)}" if targetid else "")
        + f"  z = {z:.5f} ({zsrc})  E(B-V) = {ebv:.4f} ({esrc})  lines {','.join(lines)}"
        + (f"  Monte Carlo {a.nmc}" if a.nmc else "")
    )
    try:
        res = fit_spectrum(
            sp["wave"],
            sp["flux"],
            sp["ivar"],
            z,
            ebv=ebv,
            host=not a.no_host,
            fe=not a.no_fe,
            complexes=tuple(lines),
            max_broad=a.max_broad,
            dbic=a.dbic,
            nmc=a.nmc,
            seed=a.seed,
            err_floor=a.err_floor,
            mc_noise_policy=a.mc_noise_policy,
            jobs=jobs,
        )
    except ValueError as e:
        sys.exit(f"cannot fit {path}: {e}")
    recs = {name: _line_record(name, res) for name in lines}
    for rec in recs.values():
        # The historical DESI-repeat formula was never an SDSS calibration or
        # a validation of this release. Keep it only as an explicitly requested diagnostic.
        legacy = rec.get("dv_err_model")
        rec["dv_err_model"] = np.nan
        if a.legacy_error_diagnostic and sp.get("kind") == "desi":
            rec["legacy_desi_repeat_error_diagnostic"] = legacy
        rec["uncertainty_calibrated"] = False
    ebv_assumed_zero = esrc.startswith("assumed 0")
    if ebv_assumed_zero:
        out(
            f"warning: Galactic E(B-V) taken as 0 ({esrc[len('assumed 0: ') :]}); give --ebv, or install dustmaps and "
            'fetch its SFD map (python -c "import dustmaps.sfd; dustmaps.sfd.fetch()")'
        )
        for rec in recs.values():
            if rec.get("fitted"):
                rec["flags"] = list(rec["flags"]) + ["ebv_assumed_zero"]
                rec["flag_text"] = list(rec["flag_text"]) + [FLAG_TEXT["ebv_assumed_zero"]]
    _print_fit_table(recs, res, out)

    stem = stem or _stem(path, targetid)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, stem)
    hi = res["host_info"]
    flux_unit = (
        "1e-17 erg/s/cm^2/A"
        if sp.get("kind") in ("sdss", "desi") or a.flux_scale != 1.0
        else "as given (luminosities assume 1e-17 erg/s/cm^2/A)"
    )
    row = summary_row(res)
    doc = dict(
        blrfit_version=__version__,
        uncertainty=dict(
            calibrated=False,
            mc_requested=a.nmc,
            status="conditional_mc" if a.nmc else "not_computed",
            scope="pixel-noise perturbations; no survey-wide or between-epoch calibration",
            legacy_empirical_error_used=False,
        ),
        input=dict(
            path=os.path.abspath(path),
            kind=sp.get("kind"),
            targetid=sp.get("targetid"),
            reader_policy=sp.get("mask_policy", "default"),
            catalogue_zwarn=sp.get("zwarn"),
            product=sp.get("product", "DESI coadd" if sp.get("kind") == "desi" else "supplied spectrum"),
            public_source=getattr(a, "public_source", None),
            z=z,
            z_source=zsrc,
            ebv=ebv,
            ebv_source=esrc,
            ebv_assumed_zero=ebv_assumed_zero,
            ra=sp.get("ra", np.nan),
            dec=sp.get("dec", np.nan),
            mjd=sp.get("mjd", np.nan),
            date=mjd_to_date(sp["mjd"]) if np.isfinite(sp.get("mjd", np.nan)) else "",
            flux_unit=flux_unit,
            n_pixels=int(len(sp["wave"])),
            wave_min=float(np.min(sp["wave"])),
            wave_max=float(np.max(sp["wave"])),
        ),
        settings=dict(res["settings"], complexes=lines, nmc=a.nmc, seed=a.seed),
        continuum=dict(
            pl_alpha=res["conti"].get("pl_alpha", np.nan),
            pl_norm=res["conti"].get("pl_norm", np.nan),
            feop_norm=res["conti"].get("feop_norm", np.nan),
            feop_fwhm=res["conti"].get("feop_fwhm", np.nan),
            feuv_norm=res["conti"].get("feuv_norm", np.nan),
            host_applied=bool(hi.get("applied", False)),
            host_frac=hi.get("host_frac_4200_5000", np.nan),
            host_n_gal=hi.get("n_gal", 0),
            host_reason=hi.get("reason", ""),
        ),
        o3_prefit=res.get("o3_prefit", {}),
        mc_info=res.get("mc_info", {}),
        continuum_solver=res.get("continuum_info", {}),
        lines=recs,
        summary_row=row,
    )
    with open(base + "_fit.json", "w") as fh:
        json.dump(_clean(doc), fh, indent=1)
    out(f"-> {base}_fit.json")
    if not a.no_figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plot import plot_fit

        fig = plot_fit(res, title=f"{stem}  z = {z:.4f}")
        fig.savefig(base + "_fit.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        out(f"-> {base}_fit.png")
    if a.pickle:
        import pickle

        with open(base + "_fit.pkl", "wb") as fh:
            pickle.dump(res, fh)
        out(f"-> {base}_fit.pkl")
    if results is not None:
        mjd = sp.get("mjd")
        results[stem] = dict(
            id=stem,
            res=res,
            path=path,
            mjd=float(mjd) if mjd is not None else np.nan,
            kind=sp.get("kind"),
            targetid=sp.get("targetid", targetid),
            plate=sp.get("plate"),
            fiber=sp.get("fiber"),
            z=z,
            ebv=ebv,
        )
    targetid = sp.get("targetid", targetid)
    return dict(
        spectrum=path,
        targetid=None if targetid is None else int(targetid),
        stem=stem,
        status="fitted",
        error="",
        z_source=zsrc,
        ebv=ebv,
        ebv_source=esrc,
        ebv_assumed_zero=ebv_assumed_zero,
        flux_unit=flux_unit,
        **{k: scalar(v) for k, v in row.items()},
    )


def _batch_task(task):
    """One spectrum of a batch (in a worker process or not): its record, or a failed one."""
    a, entry = task
    try:
        return fit_file(a, entry["path"], entry["targetid"], entry["z"], entry["stem"], out=_silent)
    except (Exception, SystemExit) as exc:
        return dict(
            spectrum=entry["path"],
            targetid=entry["targetid"],
            stem=entry["stem"],
            status="failed",
            error=str(exc) or type(exc).__name__,
        )


def _batch_line(rec, lines):
    """One printed line per spectrum of a batch: class and offset of each line, with its flags."""
    if rec["status"] != "fitted":
        return f"{rec['stem']}: failed: {rec['error']}"
    parts = []
    for name in lines:
        p = PREFIX[name]
        if f"{p}_class" not in rec:
            parts.append(f"{name} not fitted")
            continue
        dv = rec.get(f"{p}_c50_sys")
        text = f"{name} {rec[f'{p}_class']} " + (f"{dv:+.0f}" if dv is not None and np.isfinite(dv) else "-")
        flags = rec.get(f"{p}_flags")
        parts.append(text + (f" ({flags})" if flags else ""))
    return (
        f"{rec['stem']}: " + "; ".join(parts) + ("  [E(B-V) taken as 0]" if rec["ebv_assumed_zero"] else "")
    )


def _check_fit_options(a):
    if a.nmc < 0:
        sys.exit("--nmc must be nonnegative")
    if 0 < a.nmc < MC_MIN_CONTRIBUTING:
        sys.exit(
            f"--nmc must be 0 or at least {MC_MIN_CONTRIBUTING}: the test for draws that split between "
            "separate solutions needs that many"
        )
    if a.jobs < 1:
        sys.exit("--jobs must be at least 1")
    if a.table:
        try:
            table_format(a.table)
        except ValueError as e:
            sys.exit(str(e))


def _entries(a):
    """The local spectra of the command line and of --list, as fit entries."""
    entries = [dict(path=p, targetid=a.targetid, z=None) for p in a.spectra]
    if a.list:
        try:
            listed = read_list(a.list)
        except (ValueError, OSError) as e:
            sys.exit(f"cannot read the list {a.list}: {e}")
        entries += [dict(e, targetid=a.targetid if e["targetid"] is None else e["targetid"]) for e in listed]
    return entries


def _name_stems(entries):
    """A distinct output stem for every entry: the same file name in two
    directories, or a spectrum listed twice, must not share products."""
    used = set()
    for e in entries:
        stem = base = _stem(e["path"], e["targetid"])
        n = 2
        while stem in used:
            stem, n = f"{base}-{n}", n + 1
        used.add(stem)
        e["stem"] = stem


def cmd_fit(a):
    _check_fit_options(a)
    entries = _entries(a)
    if not entries:
        from .input_workflow import fit_public

        return fit_public(
            a,
            lambda one: fit_file(
                one, one.spectrum, one.targetid, stem=one.stem, out=(_silent if one.quiet else print)
            ),
        )
    if a.include_sdss:
        sys.exit("--include-sdss is for public queries without a local file")
    lines = _parse_lines(a.lines)
    out = _silent if a.quiet else print
    if len(entries) == 1 and not a.list:
        rec = fit_file(a, entries[0]["path"], entries[0]["targetid"], stem=a.stem, out=out, jobs=a.jobs)
        if a.table:
            write_table(summary_table([rec], meta=_table_meta(a, lines)), a.table)
            out(f"-> {a.table}")
        return 0
    if a.stem:
        sys.exit("--stem names the products of one spectrum; several spectra are named after their files")
    _name_stems(entries)
    table = a.table or os.path.join(a.out, "blrfit_summary.fits")
    out(
        f"blrfit {__version__}: {len(entries)} spectra, lines {','.join(lines)}"
        + (f", Monte Carlo {a.nmc}" if a.nmc else "")
        + (f", {a.jobs} processes" if a.jobs > 1 else "")
    )
    records = []
    for k, rec in enumerate(ordered_map(_batch_task, ((a, e) for e in entries), a.jobs), 1):
        records.append(rec)
        out(f"[{k}/{len(entries)}] " + _batch_line(rec, lines))
    write_table(summary_table(records, meta=_table_meta(a, lines)), table)
    failed = sum(r["status"] != "fitted" for r in records)
    zero = sum(bool(r.get("ebv_assumed_zero")) for r in records)
    if zero:
        out(
            f"warning: Galactic E(B-V) taken as 0 for {zero} of the spectra; give --ebv, or install dustmaps and fetch "
            'its SFD map (python -c "import dustmaps.sfd; dustmaps.sfd.fetch()")'
        )
    out(f"{len(records) - failed} fitted, {failed} failed -> {table}")
    return 1 if failed else 0


def _table_meta(a, lines):
    # FITS header keywords: at most eight characters
    return dict(
        BLRFIT=__version__,
        LINES=",".join(lines),
        NMC=a.nmc,
        SEED=a.seed,
        FLUXUNIT="1e-17 erg/s/cm2/Angstrom as read; see the column flux_unit",
    )


# ----------------------------------------------------------------------------
# pair
# ----------------------------------------------------------------------------
def _pair_fit_task(task):
    """One epoch of a pair run (in a worker process or not): its catalogue record
    and the epoch (the fit with its identity), or a failed record and None."""
    a, entry = task
    store = {}
    try:
        rec = fit_file(
            a, entry["path"], entry["targetid"], entry["z"], entry["stem"], out=_silent, results=store
        )
    except (Exception, SystemExit) as exc:
        failed = dict(
            spectrum=entry["path"],
            targetid=entry["targetid"],
            stem=entry["stem"],
            status="failed",
            error=str(exc) or type(exc).__name__,
        )
        return failed, None
    return rec, store[entry["stem"]]


def _saved_fit(path):
    """An epoch from a saved fit (``blrfit fit --pickle``): the result, and the
    identity and date of the spectrum from the JSON written beside it."""
    import pickle

    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")
    with open(path, "rb") as fh:
        res = pickle.load(fh)
    if not isinstance(res, dict) or "fits" not in res:
        sys.exit(f"{path} is not a saved blrfit fit")
    base = path[: -len("_fit.pkl")] if path.endswith("_fit.pkl") else os.path.splitext(path)[0]
    epoch = dict(
        id=os.path.basename(base),
        res=res,
        path=path,
        mjd=np.nan,
        kind=None,
        targetid=None,
        plate=None,
        fiber=None,
        z=res.get("z"),
        ebv=(res.get("settings") or {}).get("ebv"),
    )
    meta = base + "_fit.json"
    if os.path.exists(meta):
        with open(meta) as fh:
            inp = json.load(fh).get("input") or {}
        mjd = inp.get("mjd")
        epoch.update(
            mjd=float(mjd) if mjd is not None else np.nan, kind=inp.get("kind"), targetid=inp.get("targetid")
        )
    return epoch


def _pair_line(row, width):
    s, e = row["s_common"], row["err_total"]
    sig = abs(s) / e if (np.isfinite(s) and np.isfinite(e) and e > 0) else np.nan
    return (
        f"{row['id_a'] + ' -> ' + row['id_b']:{width}} {row['line']:7} {_fmt(row['dt_rest_yr'], 6, 2)} "
        f"{_fmt(s, 7, 0, True)} {_fmt(e, 5)} {_fmt(sig, 5, 1)}  {_fmt(row['shape_max'], 5, 2)}  "
        f"{_fmt(row['scale_fwd'], 5, 2)} {_fmt(row['scale_rev'], 5, 2)}  {row['flags'] or '-'}"
    )


def _measure_pairs(a, epochs, lines, out):
    """Every pair of the epochs for every line: the rows of the pair table, the
    target row (reference epoch, classes, mass and orbital limits) and, unless
    --no-figure, one figure per pair and line. The epochs are one object, named
    by --name, by their common TARGETID, or 'object'."""
    pairs, ref = enumerate_pairs(epochs, lines)
    tids = {str(e["targetid"]) for e in epochs if e.get("targetid") is not None}
    name = a.name or (tids.pop() if len(tids) == 1 else "object")
    for e in epochs:
        e["targetid"] = name
    out(
        f"blrfit {__version__}: {len(epochs)} epochs of {name}, lines {','.join(lines)}; "
        f"reference {ref['id']} (broad S/N {epoch_snr(ref['res'], lines):.0f}), {len(pairs)} pairs"
    )
    width = max([len(f"{p['a']['id']} -> {p['b']['id']}") for p in pairs] + [4])
    out(
        f"{'pair':{width}} {'line':7} {'dt_yr':>6} {'s':>7} {'+/-':>5} {'sigma':>5}  {'shape':>5}  {'scale':>5} {'rev':>5}  flags"
    )
    rows = []
    for pr in pairs:
        for line in lines:
            rec = measure_pair(pr["a"]["res"], pr["b"]["res"], line, details=not a.no_figure)
            row = pair_row(rec, pr["a"]["res"], pr["a"], pr["b"], role=pr["role"])
            rows.append(row)
            out(_pair_line(row, width))
            if not a.no_figure and "epoch_a" in rec:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                from .plot import plot_pair

                fig = plot_pair(rec, title=f"{pr['a']['id']} -> {pr['b']['id']}")
                fig.savefig(os.path.join(a.out, f"{pr['a']['id']}__{pr['b']['id']}_{line}_pair.png"), dpi=110)
                plt.close(fig)
    shown = {}
    for row in rows:
        for flag in row["flags"].split(";"):
            if flag and flag not in shown:
                shown[flag] = PAIR_FLAG_TEXT.get(flag.split(":")[-1], "")
    for flag, text in shown.items():
        out(f"  {flag}: {text}" if text else f"  {flag}")
    out(
        "  velocities in km/s; s = change of the later epoch's broad profile relative to the earlier (positive: "
        "redder), +/- its total error, sigma = |s| / error; shape = profile-change statistic per pixel (stable at "
        "most 0.5); scale = template flux scale of each direction; columns: blrfit pair --help"
    )
    cls, meas = ref["res"].get("cls") or {}, ref["res"].get("meas") or {}
    target = dict(targetid=name, reference=ref["id"], n_epochs=len(epochs), n_pairs=len(pairs))
    for line in lines:
        target[f"class_{line.lower()}"] = (cls.get(line) or {}).get("label", "")
        target[f"snr_{line.lower()}"] = float((meas.get(line) or {}).get("broad_flux_snr", np.nan))
    target.update(target_mass(ref["res"]))
    return rows, target, name


def _sibling(table, word):
    """The path of a companion table: blrfit_pairs.ecsv -> blrfit_<word>.ecsv, name.csv -> name_<word>.csv."""
    base, ext = os.path.splitext(table)
    if base.lower().endswith(".fits"):
        base, ext = base[:-5], ".fits" + ext
    return (base[:-6] if base.endswith("_pairs") else base) + "_" + word + ext


def cmd_pair(a):
    _check_fit_options(a)
    lines = _parse_lines(a.lines)
    out = _silent if a.quiet else print
    epochs = []
    if a.fits:
        if a.spectra or a.list:
            sys.exit("--fits takes saved fits in place of spectra")
        epochs = [_saved_fit(p) for p in a.fits]
    else:
        entries = _entries(a)
        if not entries:
            from .input_workflow import fit_public

            store = {}
            fit_public(
                a,
                lambda one: fit_file(
                    one,
                    one.spectrum,
                    one.targetid,
                    stem=one.stem,
                    out=(_silent if one.quiet else print),
                    results=store,
                ),
            )
            epochs = list(store.values())
        else:
            if a.include_sdss:
                sys.exit("--include-sdss is for public queries without a local file")
            _name_stems(entries)
            out(
                f"blrfit {__version__}: fitting {len(entries)} spectra, lines {','.join(lines)}"
                + (f", Monte Carlo {a.nmc}" if a.nmc else "")
                + (f", {a.jobs} processes" if a.jobs > 1 else "")
            )
            for k, (rec, epoch) in enumerate(
                ordered_map(_pair_fit_task, ((a, e) for e in entries), a.jobs), 1
            ):
                out(f"[{k}/{len(entries)}] " + _batch_line(rec, lines))
                if epoch is not None:
                    epochs.append(epoch)
    if len(epochs) < 2:
        sys.exit("two fitted epochs are needed")
    os.makedirs(a.out, exist_ok=True)
    rows, target, name = _measure_pairs(a, epochs, lines, out)
    table = a.table or os.path.join(a.out, "blrfit_pairs.ecsv")
    meta = dict(BLRFIT=__version__, LINES=",".join(lines))
    write_table(summary_table(rows, meta=meta), table)
    targets_table, tiers_table = _sibling(table, "targets"), _sibling(table, "tiers")
    write_table(summary_table([target], meta=meta), targets_table)
    classes = {line: target[f"class_{line.lower()}"] for line in lines}
    tiers = classify_table(rows, masses={name: target}, classes={name: classes})
    write_table(summary_table(tiers, meta=meta), tiers_table)
    out(f"-> {table}\n-> {targets_table}\n-> {tiers_table}")
    for t in tiers:
        out(f"{t['targetid']}: {t['tier']} ({TIER_TEXT[t['tier']]}): {t['reason']}")
    return 0


# ----------------------------------------------------------------------------
# tiers
# ----------------------------------------------------------------------------
def _target_tables(a):
    """The target tables: --targets, else the targets table written beside each pair table."""
    if a.targets:
        return list(a.targets)
    return [t for t in (_sibling(p, "targets") for p in a.pairs) if os.path.exists(t)]


def cmd_tiers(a):
    out = _silent if a.quiet else print
    rows = []
    for path in a.pairs:
        if not os.path.exists(path):
            sys.exit(f"file not found: {path}")
        try:
            rows += read_rows(path)
        except (ValueError, OSError) as e:
            sys.exit(f"cannot read {path}: {e}")
    masses, classes = {}, {}
    for path in _target_tables(a):
        try:
            targets = read_rows(path)
        except (ValueError, OSError) as e:
            sys.exit(f"cannot read {path}: {e}")
        for r in targets:
            low = {str(k).lower(): v for k, v in r.items()}
            tid = str(low.get("targetid", ""))
            masses[tid] = low
            classes[tid] = {
                "Halpha": low.get("class_halpha", low.get("ha_class", "")) or "",
                "Hbeta": low.get("class_hbeta", low.get("hb_class", "")) or "",
            }
    tiers = classify_table(rows, masses, classes)
    if not tiers:
        out("no pair rows")
        return 1
    for t in tiers:
        out(f"{t['targetid']}: {t['tier']}: {t['reason']}")
    counts = Counter(t["tier"] for t in tiers)
    out("  " + ", ".join(f"{k} {counts[k]}" for k in TIERS if counts[k]))
    for k in TIERS:
        if counts[k]:
            out(f"  {k}: {TIER_TEXT[k]}")
    table = a.out or os.path.join(os.path.dirname(os.path.abspath(a.pairs[0])), "blrfit_tiers.ecsv")
    write_table(summary_table(tiers, meta=dict(BLRFIT=__version__)), table)
    out(f"-> {table}")
    return 0


# ----------------------------------------------------------------------------
# fetch
# ----------------------------------------------------------------------------
def cmd_fetch(a):
    from .input_workflow import fetch_public

    return fetch_public(a)


def _public_args(p):
    p.add_argument(
        "--include-sdss", action="store_true", help="also take the SDSS DR17 spectra within --radius"
    )
    p.add_argument(
        "--radius",
        type=float,
        default=1.5,
        metavar="ARCSEC",
        help="SDSS match radius in arcsec, at most 1.5 (default 1.5)",
    )
    p.add_argument(
        "--desi-radius",
        type=float,
        default=1.5,
        metavar="ARCSEC",
        help="DESI search radius in arcsec (default 1.5)",
    )
    p.add_argument(
        "--releases",
        default="dr1",
        metavar="LIST",
        help="DESI releases, comma-separated: dr1 (default), edr, or dr1,edr",
    )
    p.add_argument(
        "--all-matches",
        action="store_true",
        help="keep every DESI TARGETID within --desi-radius as a separate object (default: list them and stop)",
    )


FIT_TABLE_HELP = """\
the printed table (velocities in km/s, relative to the narrow-line systemic velocity v_sys):
  cls            class: A bulk shift, B double-peaked, C asymmetric, F normal; E no broad
                 line, X no systemic reference, W broad component too weak or narrow to classify
  dv=c50-sys     offset of the half-maximum bisector c(1/2) of the broad profile
  +/-mc          its Monte Carlo error (with --nmc; withheld when the draws are unreliable)
  peak, cen      peak and flux-weighted centroid of the broad profile
  FWHM, nb       full width at half maximum; number of broad Gaussians
  fS/N, pS/N     broad-line flux S/N; S/N of the broad peak per pixel
  A.I., K.I.     asymmetry index at 1/4 maximum; kurtosis index W(3/4)/W(1/4)
  v_sys, sS/N    systemic velocity relative to the input redshift, and its S/N
  [SII], [OIII]  [S II] and [O III] core velocities; oS/N the S/N of the [O III] core
  flags          quality warnings, explained below the table
"""


PAIR_TABLE_HELP = """\
the printed table (velocities in km/s):
  pair           the earlier epoch -> the later one; the reference epoch (highest broad S/N) is in every pair
  dt_yr          rest-frame years between the epochs
  s, +/-         change of the later epoch's broad profile relative to the earlier (positive: redder), with
                 its total error, statistical plus the calibrated term; withheld (-) when a screen failed
  sigma          |s| divided by the error
  shape          the profile-change statistic per pixel: at most 0.5 for a stable profile
  scale, rev     the template flux scale of each direction (A over B, B over A)
  flags          screens and warnings, explained below the table
then the tier of the object with its reason (blrfit tiers --help).
"""

TIERS_HELP = """\
tiers, in the order they are tried:
  disk           a double-peaked (class B) broad profile in the reference spectrum: a disc emitter
  platinum       a change of at least 4 sigma with a stable profile, both Balmer lines agreeing, the two
                 directions within 2 sigma, broad peak S/N >= 8 in both epochs, and an orbit that allows it
  binary         the same without the conditions on the second line, the directions and the peak S/N
  almost         a significant change that fails one condition, or a marginal one (3-4 sigma) that passes all
  profile        the strongest change comes with a changed profile: variability, not a bulk motion
  stable         no change above 3 sigma: an upper limit
  none           no retained pair with both epochs at the S/N floor of 8
"""


def _add_fit_options(f, pairs=False):
    """The options of a fit, shared by ``fit`` and ``pair``."""
    from .io.public import exact_targetid

    g = f.add_argument_group("input")
    g.add_argument(
        "--list",
        metavar="FILE",
        help="more spectra, listed in a file: one path per line, optionally followed by its TARGETID, "
        "or a table (.fits, .ecsv, .csv) with a column path and optionally targetid and z",
    )
    g.add_argument(
        "--survey",
        choices=("desi", "sdss", "generic", "auto"),
        metavar="{desi,sdss,generic}",
        default="desi",
        help="format of the file: desi (default), sdss, or generic for a table (see below); "
        "without a file, the archive to search (desi or sdss)",
    )
    g.add_argument(
        "--targetid",
        type=exact_targetid,
        default=None,
        help="DESI TARGETID: the target to take from a coadd, or to search for",
    )
    g.add_argument(
        "--redrock", metavar="FILE", help="DESI redrock file (default: the redrock file next to the coadd)"
    )
    g.add_argument(
        "--sdss-mask-policy",
        choices=("ivar", "conservative"),
        default="ivar",
        help="SDSS pixels to exclude: those without inverse variance (ivar, the default and the reading of "
        "the validation), or also every pixel with a mask bit (conservative)",
    )
    g.add_argument("--z", type=float, default=None, help="redshift (default: from the file, when it has one)")
    g.add_argument(
        "--ebv",
        default=None,
        help="Galactic E(B-V): a number, or 'sfd' for the SFD98 map (default: the FIBERMAP value for DESI; "
        "otherwise the SFD98 map if dustmaps is installed, else 0, with a warning and the flag ebv_assumed_zero)",
    )
    g.add_argument(
        "--ra",
        type=float,
        default=None,
        metavar="DEG",
        help="right ascension in degrees, for a public search or the SFD98 map",
    )
    g.add_argument("--dec", type=float, default=None, metavar="DEG", help="declination in degrees")
    _public_args(f.add_argument_group("public search (no file given)"))
    g = f.add_argument_group("model")
    g.add_argument(
        "--lines",
        default="Halpha,Hbeta",
        metavar="LIST",
        help="comma-separated complexes among Halpha, Hbeta and MgII (default Halpha,Hbeta); "
        "complexes outside the data are skipped; MgII is experimental",
    )
    g.add_argument("--no-host", action="store_true", help="no host-galaxy component")
    g.add_argument("--no-fe", action="store_true", help="no Fe II templates")
    g.add_argument(
        "--max-broad",
        type=int,
        default=MAX_BROAD,
        metavar="N",
        help=f"largest number of broad Gaussians (default {MAX_BROAD})",
    )
    g.add_argument(
        "--dbic",
        type=float,
        default=DBIC,
        help=f"BIC decrease required to add a broad Gaussian (default {DBIC:.0f})",
    )
    g.add_argument(
        "--err-floor",
        type=float,
        default=ERR_FLOOR,
        metavar="FRAC",
        help=f"fractional error floor added to the pixel errors (default {ERR_FLOOR})",
    )
    g = f.add_argument_group("uncertainties")
    g.add_argument(
        "--nmc",
        type=int,
        default=0,
        metavar="N",
        help="Monte Carlo realisations for statistical errors: 0 (default, no errors) or at least 25",
    )
    g.add_argument(
        "--seed", type=int, default=0, metavar="N", help="random seed of the Monte Carlo (default 0)"
    )
    g = f.add_argument_group("output")
    g.add_argument(
        "--out", default=".", metavar="DIR", help="output directory (default: the current directory)"
    )
    if not pairs:
        g.add_argument(
            "--stem", default=None, metavar="NAME", help="output file stem (default: from the file name)"
        )
    g.add_argument(
        "--table",
        metavar="FILE",
        help=(
            "the pair table, FITS (.fits), ECSV (.ecsv) or CSV (.csv); the target and tier tables are written "
            "beside it (default <out>/blrfit_pairs.ecsv)"
            if pairs
            else "catalogue table of the summary rows, FITS (.fits), ECSV (.ecsv) or CSV (.csv), with units "
            "(default with several spectra: <out>/blrfit_summary.fits)"
        ),
    )
    g.add_argument(
        "--no-figure",
        action="store_true",
        help="do not write the figures" if pairs else "do not write the figure",
    )
    g.add_argument("--pickle", action="store_true", help="also write the full result as <stem>_fit.pkl")
    g.add_argument("--quiet", action="store_true", help="print nothing")
    g.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="worker processes: several spectra are fitted in parallel, or the Monte Carlo draws of one "
        "(default 1); the results do not depend on N",
    )
    _add_table_args(f)
    # Settings that reproduce earlier releases; they work but are not listed in --help.
    f.add_argument(
        "--mc-noise-policy", choices=("input", "effective"), default="input", help=argparse.SUPPRESS
    )
    f.add_argument("--legacy-error-diagnostic", action="store_true", help=argparse.SUPPRESS)


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="blrfit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Fit the broad Balmer lines of AGN spectra: offsets from the narrow-line reference,\n"
        "profile classes and quality flags; velocity changes between epochs and candidate tiers.",
    )
    p.add_argument("--version", action="version", version=f"blrfit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    from .io.public import exact_targetid

    f = sub.add_parser(
        "fit",
        help="fit spectra",
        usage="%(prog)s [spectrum ...] [options]",
        description="Fit spectra: local files, the files of a list, or the public DESI or SDSS spectra of a\n"
        "position or a DESI TARGETID. Writes <stem>_fit.json and <stem>_fit.png for every spectrum and\n"
        "prints the table below; with several spectra, one line each and a catalogue table of all of them.",
        epilog=FIT_TABLE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    f.add_argument(
        "spectra",
        nargs="*",
        metavar="spectrum",
        help="local spectrum files; leave out to search the public archives (--ra and --dec, or --targetid)",
    )
    _add_fit_options(f)
    f.set_defaults(func=cmd_fit)

    q = sub.add_parser(
        "pair",
        help="velocity changes between the epochs of one object",
        usage="%(prog)s [spectrum spectrum ...] [options]",
        description="Measure the velocity changes of the broad lines between the dated spectra of one object:\n"
        "local files, the files of a list, the public DESI and SDSS spectra of a position or TARGETID, or saved\n"
        "fits (--fits). Every spectrum is fitted as by 'fit'; then the reference epoch (highest broad S/N) is\n"
        "paired with every other and consecutive epochs with each other, each pair measured in both directions\n"
        "with the template cross-correlation. Writes the pair table, a target table (reference, classes, mass,\n"
        "orbital limits), a tier table and one figure per pair and line, and prints the table below.",
        epilog=PAIR_TABLE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    q.add_argument("spectra", nargs="*", metavar="spectrum", help="two or more spectra of one object")
    q.add_argument(
        "--fits",
        nargs="+",
        metavar="PKL",
        help="saved fits (<stem>_fit.pkl of 'fit --pickle', with the JSON beside them) in place of spectra",
    )
    q.add_argument(
        "--name",
        metavar="NAME",
        help="name of the object in the tables (default: its TARGETID, else 'object')",
    )
    _add_fit_options(q, pairs=True)
    q.set_defaults(func=cmd_pair, stem=None)

    t = sub.add_parser(
        "tiers",
        help="candidate tiers from pair tables",
        description="The candidate tier of every object of one or more pair tables written by 'pair' (or by\n"
        "Python, pairs.pair_record): one row per object with the tier, its reason and the pair behind it.",
        epilog=TIERS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    t.add_argument("pairs", nargs="+", metavar="PAIRS", help="pair tables (.fits, .ecsv or .csv)")
    t.add_argument(
        "--targets",
        nargs="+",
        metavar="TABLE",
        help="target tables with the reference classes and masses (default: the table beside each pair table)",
    )
    t.add_argument(
        "--out",
        metavar="FILE",
        help="the tier table (default: blrfit_tiers.ecsv beside the first pair table)",
    )
    t.add_argument("--quiet", action="store_true", help="print nothing")
    t.set_defaults(func=cmd_tiers)

    g = sub.add_parser(
        "fetch",
        help="download public spectra without fitting",
        description="Download the public DESI or SDSS spectra of a position or a DESI TARGETID, "
        "with a manifest of what was found.",
    )
    g.add_argument("--ra", type=float, metavar="DEG", help="right ascension in degrees")
    g.add_argument("--dec", type=float, metavar="DEG", help="declination in degrees")
    g.add_argument("--targetid", type=exact_targetid, help="DESI TARGETID, instead of a position")
    g.add_argument(
        "--survey", choices=("desi", "sdss"), default="desi", help="archive to search (default desi)"
    )
    g.add_argument("--out", default="spectra", metavar="DIR", help="download directory (default spectra)")
    g.add_argument("--quiet", action="store_true", help="print nothing")
    _public_args(g)
    g.set_defaults(func=cmd_fetch)
    return p


def main(argv=None):
    p = build_parser()
    args = sys.argv[1:] if argv is None else list(argv)
    if args[:1] == ["rv"]:
        p.exit(
            2,
            "blrfit: the rv command was removed in version 0.3.0 and blrfit.rv is deprecated since "
            "0.4.0 (removed in 0.5): use blrfit pair\n",
        )
    a = p.parse_args(args)
    try:
        return a.func(a)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        p.exit(1, f"blrfit: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
