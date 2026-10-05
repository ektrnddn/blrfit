"""
Command-line interface.

    blrfit fit   SPECTRUM [--targetid TID] [--z Z] [--ebv E] [...]   one spectrum -> table, JSON, figure
    blrfit rv    EPOCH1 EPOCH2 --z Z --line Halpha [...]           velocity change between two spectra
    blrfit fetch --ra RA --dec DEC [--out DIR] [...]                 public SDSS and DESI spectra of a position

Successful point fits include explicit unavailable-line outcomes. Input or
fitting exceptions return a nonzero status; public batches retain failed products
in their manifests. Between-epoch routines are experimental.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__
from .constants import COMPLEX_WINDOW, MAX_BROAD, DBIC, ERR_FLOOR
from .io import read_spectrum, is_desi_coadd, is_sdss_spec
from .io.dust import sfd_ebv
from .io.sdss import mjd_to_date
from .model.fit import fit_spectrum, summary_row
from .classify import LABEL_TEXT, FLAG_TEXT, is_measurable, is_strong_offset, velocities_at_bound
from .errors import empirical_error, MC_MIN_CONTRIBUTING
from . import rv as RV

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


def _stem(path, sp, targetid=None):
    base = os.path.splitext(os.path.basename(path))[0]
    if base.endswith(".fits"):
        base = base[:-5]
    if sp.get("kind") == "desi" and targetid is not None and str(targetid) not in base:
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
def cmd_fit(a):
    if a.nmc < 0:
        sys.exit("--nmc must be nonnegative")
    if 0 < a.nmc < MC_MIN_CONTRIBUTING:
        sys.exit(
            f"--nmc must be 0 or at least {MC_MIN_CONTRIBUTING}: the test for draws that split between "
            "separate solutions needs that many"
        )
    if a.spectrum is None:
        from .input_workflow import fit_public

        return fit_public(a, cmd_fit)
    if a.include_sdss:
        sys.exit("--include-sdss is for public queries without a local file")
    sp, z, zsrc, ebv, esrc = _load(a.spectrum, a, a.targetid)
    names = {k.lower(): k for k in COMPLEX_WINDOW}
    lines = [names.get(s.strip().lower(), s.strip()) for s in a.lines.split(",") if s.strip()]
    bad = [s for s in lines if s not in COMPLEX_WINDOW]
    if bad:
        sys.exit(f"unknown line(s) {bad}; choose from {list(COMPLEX_WINDOW)}")
    out = (lambda *x: None) if a.quiet else print
    out(
        f"blrfit {__version__}: {a.spectrum}"
        + (f" TARGETID {int(a.targetid)}" if a.targetid else "")
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
        )
    except ValueError as e:
        sys.exit(f"cannot fit {a.spectrum}: {e}")
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

    stem = a.stem or _stem(a.spectrum, sp, a.targetid)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, stem)
    hi = res["host_info"]
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
            path=os.path.abspath(a.spectrum),
            kind=sp.get("kind"),
            targetid=sp.get("targetid"),
            reader_policy=sp.get("mask_policy", "RC1"),
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
            flux_unit=(
                "1e-17 erg/s/cm^2/A"
                if sp.get("kind") in ("sdss", "desi") or a.flux_scale != 1.0
                else "as given (luminosities assume 1e-17 erg/s/cm^2/A)"
            ),
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
        summary_row=summary_row(res),
    )
    with open(base + "_fit.json", "w") as fh:
        json.dump(_clean(doc), fh, indent=1)
    out(f"-> {base}_fit.json")
    if not a.no_figure:
        import matplotlib

        matplotlib.use("Agg")
        from .plot import plot_fit

        fig = plot_fit(res, title=f"{stem}  z = {z:.4f}")
        fig.savefig(base + "_fit.png", dpi=110, bbox_inches="tight")
        out(f"-> {base}_fit.png")
    if a.pickle:
        import pickle

        with open(base + "_fit.pkl", "wb") as fh:
            pickle.dump(res, fh)
        out(f"-> {base}_fit.pkl")
    return 0


# ----------------------------------------------------------------------------
# rv
# ----------------------------------------------------------------------------
def _rv_line(line, res1, res2, sp1, sp2, epochs_meta, a, out):
    """Cross-correlate one line of the two fitted epochs; returns the JSON record (or a
    'measured': False record) and the pair for the figure."""
    epochs = []
    for meta, sp, res in ((epochs_meta[0], sp1, res1), (epochs_meta[1], sp2, res2)):
        m = res["meas"].get(line, {})
        c = res["cls"].get(line, {})
        epochs.append(
            dict(
                meta,
                fitted=line in res["fits"],
                label=c.get("label", ""),
                flags=list(c.get("flags", [])),
                c50_sys=m.get("c50_sys", np.nan),
                v_peak_sys=m.get("v_peak_sys", np.nan),
                fwhm=m.get("fwhm", np.nan),
                broad_flux_snr=m.get("broad_flux_snr", np.nan),
                v_sys=m.get("v_sys", np.nan),
                line=_line_record(line, res),
            )
        )
        out(
            f"  {os.path.basename(meta['path'])}: MJD {_fmt(meta['mjd'], 8, 1)} {line} class {c.get('label', '-')} "
            f"c50-sys {_fmt(m.get('c50_sys', np.nan), 6, 0, True)} FWHM {_fmt(m.get('fwhm', np.nan), 5)} "
            f"flux S/N {_fmt(m.get('broad_flux_snr', np.nan), 5, 1)} flags {','.join(c.get('flags', [])) or '-'}"
        )
    pair = (
        RV.pair_analysis(res2, res1, name=line, vmax=a.vmax, n_mc=a.nmc, details=True)
        if (line in res1["fits"] and line in res2["fits"])
        else None
    )
    doc = dict(line=line, epochs=epochs)
    if pair is None:
        doc.update(
            measured=False, reason="the line is not fitted in both epochs or the cross-correlation failed"
        )
        out(f"  {line}: cross-correlation not possible: " + doc["reason"])
        return doc, None
    from .rv_policy import measurement_policy

    policy = measurement_policy(pair, line, sp1.get("kind"), sp2.get("kind"))
    grade = pair["profile_grade"]
    err_total = policy["err_total"]
    dv_corr = policy["dv_corrected"]
    reliable = policy["reliable"]
    cut = RV.CCF_DIR_CUT_KMS.get(line, np.nan)
    checks = [
        (pair["at_bound"], "at bound"),
        (not pair.get("frame_ok", False), f"narrow-line frame: {pair.get('frame_reason', '')}"),
        (
            not pair.get("scale_ok", True),
            f"implausible flux factors {pair.get('scale_ab', np.nan):.2f} / {pair.get('scale_ba', np.nan):.2f}",
        ),
        (bool(pair.get("ambiguous", False)), "a second cross-correlation minimum of similar depth"),
        (
            not (pair["profile_z"] < RV.CCF_PROFILE_Z_MAX),
            f"profile_z {pair['profile_z']:.1f} >= {RV.CCF_PROFILE_Z_MAX:.0f}",
        ),
        (
            not (pair["dir_mismatch"] < cut),
            f"direction mismatch {pair['dir_mismatch']:.0f} >= {cut:.0f} km/s",
        ),
    ]
    why = next((text for failed, text in checks if failed), "")
    s_ab = (pair.get("details") or {}).get("s_ab") or {}
    doc.update(
        measured=True,
        dv=pair["dv"],
        err=pair["err"],
        err_dchi2=s_ab.get("err_dchi2", np.nan),
        err_method=pair.get("err_method", s_ab.get("err_method", "unknown")),
        n_mc_requested=pair.get("n_mc_requested", a.nmc),
        n_mc_success=pair.get("n_mc_success", 0),
        bootstrap_fallback_reason=pair.get("bootstrap_fallback_reason", ""),
        algorithm_version=pair.get("algorithm_version", "unknown"),
        covariance_mode=pair.get("covariance_mode", "unknown"),
        profile_grade=grade,
        resid_frac=pair["resid_frac"],
        err_total=err_total,
        sigma_sys_desi=RV.systematic_floor(line, pair.get("snr_proxy", np.nan)),
        reliable_reason=why or policy["calibration_status"],
        consistent=pair["consistent"],
        dir_mismatch=pair["dir_mismatch"],
        profile_z=pair["profile_z"],
        chi2_red=pair["chi2_red"],
        at_bound=pair["at_bound"],
        regridded=pair["regridded"],
        npix=pair["npix"],
        snr_proxy=pair["snr_proxy"],
        zp_dv=pair["zp_dv"],
        zp_err=pair["zp_err"],
        zp_line=pair["zp_line"],
        zp_source=pair.get("zp_source"),
        frame_ok=bool(pair.get("frame_ok", False)),
        frame_reason=pair.get("frame_reason", ""),
        zp_applied=bool(pair.get("zp_applied", False)),
        dv_corrected=dv_corr,
        reliable=bool(reliable),
        significance=float(abs(dv_corr) / err_total)
        if (np.isfinite(err_total) and err_total > 0)
        else np.nan,
        c50_difference=epochs[1]["c50_sys"] - epochs[0]["c50_sys"],
        dv_abs_epoch1=epochs[0]["c50_sys"],
        dv_abs_epoch2=epochs[0]["c50_sys"] + dv_corr,
    )
    doc.update(policy)
    out(
        f"  {line}: shift of epoch 2 relative to epoch 1 {pair['dv']:+.0f} +/- {pair['err']:.0f} km/s; "
        f"corrected shift {dv_corr:+.0f} +/- {err_total:.0f} (statistical approximation, error {pair.get('err_method', 'unknown')}); "
        f"calibration pending; directions {'agree' if pair['consistent'] else 'DISAGREE'} "
        f"(mismatch {pair['dir_mismatch']:.0f}); "
        f"{'regridded' if pair['regridded'] else 'same grid'}; {'at bound' if pair['at_bound'] else 'inside search range'}"
    )
    out(
        f"  {line}: profile grade {grade} (z_prof {pair['profile_z']:.1f}, residual {100 * pair['resid_frac']:.1f} per cent of the peak); "
        f"reliable tier {reliable}{(' (' + why + ')') if why else ''}"
    )
    out(
        f"  {line}: narrow-line zero-point ({pair['zp_line'] or 'none'}) {_fmt(pair['zp_dv'], 5, 0, True)} +/- {_fmt(pair['zp_err'], 4)} km/s "
        f"(frame {'ok' if doc['frame_ok'] else 'VETOED, ' + doc['frame_reason']}; {'applied' if doc['zp_applied'] else 'not applied'}) -> dv = {dv_corr:+.0f} km/s ({doc['significance']:.1f} statistical error units, uncalibrated); "
        f"c(1/2) difference of the two fits {doc['c50_difference']:+.0f}; offset from the narrow lines "
        f"epoch 1 {epochs[0]['c50_sys']:+.0f}, epoch 2 {doc['dv_abs_epoch2']:+.0f} km/s"
    )
    return doc, pair


def cmd_rv(a):
    tids = [t.strip() for t in (a.targetid or "").split(",") if t.strip()]
    tid1 = int(tids[0]) if tids else None
    tid2 = int(tids[1]) if len(tids) > 1 else tid1
    sp1, z1, zsrc1, ebv1, esrc1 = _load(a.epoch1, a, tid1)
    a2 = argparse.Namespace(**vars(a))
    a2.z = z1  # both epochs at the same redshift
    sp2, z2, zsrc2, ebv2, esrc2 = _load(a.epoch2, a2, tid2)
    if a.ebv is not None:
        ebv2 = ebv1
    z = z1
    names = {k.lower(): k for k in COMPLEX_WINDOW}
    lines = [names.get(s.strip().lower(), s.strip()) for s in a.line.split(",") if s.strip()]
    bad = [s for s in lines if s not in COMPLEX_WINDOW]
    if bad:
        sys.exit(f"unknown line(s) {bad}; choose from {list(COMPLEX_WINDOW)}")
    if 0 < a.nmc < 10:
        sys.exit(
            "--nmc must be at least 10: the bootstrap error is kept only when at least 10 realisations succeed"
        )
    complexes = tuple(
        c
        for c in ("Halpha", "Hbeta", "MgII")
        if c in lines or (c in ("Halpha", "Hbeta") and set(lines) & {"Halpha", "Hbeta"})
    )
    out = (lambda *x: None) if a.quiet else print
    out(f"blrfit {__version__} rv: {','.join(lines)}, z = {z:.5f} ({zsrc1}), E(B-V) {ebv1:.4f} / {ebv2:.4f}")
    try:
        res1 = fit_spectrum(sp1["wave"], sp1["flux"], sp1["ivar"], z, ebv=ebv1, complexes=complexes)
        res2 = fit_spectrum(sp2["wave"], sp2["flux"], sp2["ivar"], z, ebv=ebv2, complexes=complexes)
    except ValueError as e:
        sys.exit(f"cannot fit the epochs: {e}")
    epochs_meta = [
        dict(
            path=os.path.abspath(path),
            kind=sp.get("kind"),
            targetid=tid,
            mjd=sp.get("mjd", np.nan),
            date=mjd_to_date(sp["mjd"]) if np.isfinite(sp.get("mjd", np.nan)) else "",
        )
        for path, sp, tid in ((a.epoch1, sp1, tid1), (a.epoch2, sp2, tid2))
    ]
    doc = dict(
        blrfit_version=__version__, z=z, convention="dv > 0: epoch 2 redshifted relative to epoch 1", lines={}
    )
    mjd1, mjd2 = epochs_meta[0]["mjd"], epochs_meta[1]["mjd"]
    if np.isfinite(mjd1) and np.isfinite(mjd2):
        doc["baseline_days"] = float(mjd2 - mjd1)
        doc["baseline_rest_yr"] = float((mjd2 - mjd1) / 365.25 / (1 + z))
    pairs = {}
    for line in lines:
        rec, pair = _rv_line(line, res1, res2, sp1, sp2, epochs_meta, a, out)
        doc["lines"][line] = rec
        pairs[line] = pair
    first = doc["lines"][lines[0]]
    doc.update({k: v for k, v in first.items() if k != "lines"})  # the first line's record at the top level
    if "Halpha" in pairs and "Hbeta" in pairs:
        from .rv_policy import two_line_policy

        tl = two_line_policy(doc["lines"]["Halpha"], doc["lines"]["Hbeta"])
        doc["two_line"] = (
            tl if tl is not None else dict(consistent=False, reason="one of the lines has no usable shift")
        )
        if "difference" in tl:
            out(
                f"  two-line criterion (Halpha vs Hbeta): {'consistent' if tl['consistent'] else 'NOT consistent'} "
                f"(difference {tl['difference']:+.0f} km/s, {tl['sigma']:.1f} sigma{'' if tl['same_sign'] else ', opposite signs'})"
            )
    stem = a.stem or f"{_stem(a.epoch1, sp1, tid1)}_vs_{_stem(a.epoch2, sp2, tid2)}"
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, stem)
    with open(base + "_rv.json", "w") as fh:
        json.dump(_clean(doc), fh, indent=1)
    out(f"-> {base}_rv.json")
    if not a.no_figure and any(p is not None for p in pairs.values()):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plot import plot_epochs_overlay, plot_ccf

        shown = [l for l in lines if pairs.get(l) is not None]
        fig, axes = plt.subplots(
            len(shown),
            2,
            figsize=(13, 4.6 * len(shown)),
            squeeze=False,
            gridspec_kw=dict(width_ratios=[1.6, 1]),
        )
        for k, line in enumerate(shown):
            pair = pairs[line]
            plot_epochs_overlay(
                [res1, res2],
                [epochs_meta[0]["date"] or "epoch 1", epochs_meta[1]["date"] or "epoch 2"],
                name=line,
                title=f"{stem}: {line}",
                ax=axes[k, 0],
            )
            plot_ccf(
                pair,
                title=f"{line}: dv = {pair['dv']:+.0f} +/- {pair['err']:.0f} km/s, grade {pair['profile_grade']}",
                ax=axes[k, 1],
            )
        fig.tight_layout()
        fig.savefig(base + "_rv.png", dpi=110)
        out(f"-> {base}_rv.png")
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


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="blrfit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Single-spectrum broad AGN line fitting: narrow-reference offsets, "
        "profile classes and quality flags. Between-epoch routines are experimental.",
    )
    p.add_argument("--version", action="version", version=f"blrfit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser(
        "fit",
        help="fit one spectrum",
        usage="%(prog)s [spectrum] [options]",
        description="Fit one spectrum: a local file, or the public DESI or SDSS spectra of a position or a\n"
        "DESI TARGETID. Prints a table and writes <stem>_fit.json and <stem>_fit.png.",
        epilog=FIT_TABLE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from .io.public import exact_targetid

    f.add_argument(
        "spectrum",
        nargs="?",
        help="local spectrum file; leave out to search the public archives (--ra and --dec, or --targetid)",
    )
    g = f.add_argument_group("input")
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
    g.add_argument(
        "--stem", default=None, metavar="NAME", help="output file stem (default: from the file name)"
    )
    g.add_argument("--no-figure", action="store_true", help="do not write the figure")
    g.add_argument("--pickle", action="store_true", help="also write the full result as <stem>_fit.pkl")
    g.add_argument("--quiet", action="store_true", help="print nothing")
    _add_table_args(f)
    # Settings that reproduce earlier releases; they work but are not listed in --help.
    f.add_argument(
        "--sdss-mask-policy", choices=("conservative", "ivar"), default="conservative", help=argparse.SUPPRESS
    )
    f.add_argument(
        "--mc-noise-policy", choices=("input", "effective"), default="input", help=argparse.SUPPRESS
    )
    f.add_argument("--legacy-error-diagnostic", action="store_true", help=argparse.SUPPRESS)
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser(
        "rv",
        help="experimental between-epoch diagnostics; not validated velocity measurements",
        description="Cross-correlate the broad line of two epochs; writes <stem>_rv.json and <stem>_rv.png.",
    )
    r.add_argument("epoch1")
    r.add_argument("epoch2")
    r.add_argument("--targetid", default=None, help="DESI TARGETID, or two comma-separated (one per epoch)")
    r.add_argument(
        "--z", type=float, default=None, help="redshift used for both epochs (default: from epoch 1's file)"
    )
    r.add_argument(
        "--ebv", default=None, help="Galactic E(B-V) for both epochs (default: per file as in fit)"
    )
    r.add_argument("--ra", type=float, default=None, help="right ascension (deg), for --ebv sfd")
    r.add_argument("--dec", type=float, default=None, help="declination (deg)")
    r.add_argument(
        "--line",
        default="Halpha",
        help="Halpha (default), Hbeta, MgII, or a comma-separated list; "
        "with Halpha,Hbeta the two-line criterion is evaluated",
    )
    r.add_argument(
        "--vmax", type=float, default=2000.0, help="search range of the shift, km/s (default 2000)"
    )
    r.add_argument(
        "--nmc",
        type=int,
        default=0,
        help="bootstrap realisations for the cross-correlation error, at least 10 (default 0 = Delta chi-square error)",
    )
    r.add_argument("--out", default=".", help="output directory")
    r.add_argument("--stem", default=None, help="output file stem")
    r.add_argument("--no-figure", action="store_true", help="do not write the figure")
    r.add_argument("--quiet", action="store_true", help="print nothing")
    _add_table_args(r)
    r.set_defaults(func=cmd_rv)

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
    a = p.parse_args(argv)
    try:
        return a.func(a)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as exc:
        p.exit(1, f"blrfit: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
