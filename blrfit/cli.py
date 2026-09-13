"""
Command-line interface.

    blrfit fit   SPECTRUM [--targetid TID] [--z Z] [--ebv E] [...]   one spectrum -> table, JSON, figure
    blrfit rv    EPOCH1 EPOCH2 --z Z --line Halpha [...]           velocity change between two spectra
    blrfit fetch --ra RA --dec DEC [--out DIR] [...]                 public SDSS and DESI spectra of a position

``fit`` always exits with status 0 when the spectrum could be read: a line
that cannot be fitted or classified is reported as such in the JSON.
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
from .io.sdss import mjd_to_date
from .model.fit import fit_spectrum, summary_row
from .classify import LABEL_TEXT, FLAG_TEXT, is_measurable, is_strong_offset
from .errors import empirical_error
from . import rv as RV

LINE_KEYS = ("v_peak_sys", "centroid_sys", "peak_top_sys", "centroid25_sys", "centroid50_sys", "c25_sys", "c75_sys",
             "c90_sys", "fwhm", "W25", "W75", "W90", "sigma_line", "skew", "AI", "KI", "n_peaks", "peak_sep", "dip_frac", "n_broad",
             "broad_flux", "broad_ew", "broad_ew_agn", "broad_lum", "broad_flux_snr", "broad_peak_snr",
             "v_sys", "sig_sys", "sys_snr", "narrow_peak_snr", "z_sys", "v_sii", "sig_sii", "v_o3", "v_o3_peak",
             "o3_core_snr", "nw_f", "nw_v", "nw_sig", "v_cover_lo", "v_cover_hi", "chi2_red", "v_single_gauss",
             "data_v_peak", "data_c50", "data_centroid_win", "host_frac", "pl_alpha")


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
    g = p.add_argument_group("table input (FITS table, CSV, ECSV or text)")
    g.add_argument("--wave", default="wave", help="wavelength column name or index (default wave)")
    g.add_argument("--flux", default="flux", help="flux column (default flux)")
    g.add_argument("--err", default=None, help="1-sigma error column")
    g.add_argument("--ivar", default=None, help="inverse-variance column (instead of --err)")
    g.add_argument("--wave-unit", default="angstrom", help="angstrom (default), nm, um or m")
    g.add_argument("--frame", default="obs", help="obs (default) or rest; rest needs --z")
    g.add_argument("--air", action="store_true", help="wavelengths are in air (converted to vacuum)")
    g.add_argument("--flux-scale", type=float, default=1.0,
                   help="factor bringing the flux to 1e-17 erg/s/cm^2/A (needed for luminosities only)")
    g.add_argument("--hdu", type=int, default=1, help="extension of a FITS table (default 1)")


def _table_kwargs(a):
    if a.err is None and a.ivar is None:
        sys.exit("a table needs its error column: give --err <column> (1-sigma) or --ivar <column> (inverse variance)")
    return dict(wave=a.wave, flux=a.flux, err=a.err, ivar=a.ivar, wave_unit=a.wave_unit, frame=a.frame,
                air=a.air, z=a.z, flux_scale=a.flux_scale, hdu=a.hdu)


def _load(path, a, targetid=None):
    """Read a spectrum and settle the redshift and E(B-V). Returns (dict, z, z_source, ebv, ebv_source)."""
    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")
    try:
        if is_desi_coadd(path):
            if targetid is None:
                sys.exit("a DESI coadd needs --targetid")
            sp = read_spectrum(path, targetid=int(targetid))
        elif is_sdss_spec(path):
            sp = read_spectrum(path)
        else:
            sp = read_spectrum(path, **_table_kwargs(a))
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
    if a.ebv is None:
        ebv, esrc = (float(sp["ebv"]), "FIBERMAP") if sp.get("kind") == "desi" else (0.0, "default 0")
    elif str(a.ebv).lower() in ("sfd", "dust", "map"):
        ra = a.ra if getattr(a, "ra", None) is not None else sp.get("ra", np.nan)
        dec = a.dec if getattr(a, "dec", None) is not None else sp.get("dec", np.nan)
        ebv, esrc = _ebv_sfd(ra, dec), "SFD map"
    else:
        ebv, esrc = float(a.ebv), "argument"
    return sp, z, zsrc, ebv, esrc


def _ebv_sfd(ra, dec):
    """Galactic E(B-V) from the SFD map through the dustmaps package."""
    if not (np.isfinite(ra) and np.isfinite(dec)):
        sys.exit("--ebv sfd needs coordinates: --ra and --dec, or a file that carries them")
    try:
        from dustmaps.sfd import SFDQuery
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        sys.exit("--ebv sfd needs the dustmaps package (pip install dustmaps; then fetch the SFD map "
                 "with dustmaps.sfd.fetch())")
    return float(SFDQuery()(SkyCoord(ra * u.deg, dec * u.deg)))


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
        return dict(fitted=False, label="", class_text="not fitted",
                    reasons=[f"complex not fitted: the rest-frame window {lo:.0f}-{hi:.0f} A is not covered "
                             f"(the line core must be covered to +/-3500 km/s with at least 60 pixels)"],
                    flags=[], flag_text=[], measurable=False, strong_offset=False, dv=None)
    m = res["meas"][name]; c = res["cls"][name]; e = res["err"].get(name, {})
    dv = m.get("c50_sys", np.nan)
    rec = dict(fitted=True, label=c["label"], class_text=LABEL_TEXT.get(c["label"], ""),
               reasons=list(c.get("reasons", [])), flags=list(c.get("flags", [])),
               flag_text=[FLAG_TEXT.get(f, f) for f in c.get("flags", [])],
               measurable=bool(is_measurable(c["label"], c.get("flags", []), m.get("broad_flux_snr", np.nan), m.get("fwhm", np.nan))),
               strong_offset=bool(is_strong_offset(c["label"], c.get("flags", []), m.get("broad_flux_snr", np.nan), m.get("fwhm", np.nan), dv)),
               dv=dv, dv_err_mc=e.get("c50_sys", np.nan),
               # the repeat-spectrum model was calibrated on measurable lines: not reported for E, X, W
               dv_err_model=(empirical_error(m.get("broad_flux_snr", np.nan), dv)
                             if c["label"] in ("A", "B", "C", "F") else np.nan),
               systemic_source=m.get("systemic_source", ""),
               v_sii_minus_sys=(m.get("v_sii", np.nan) - m.get("v_sys", np.nan)),
               v_o3_minus_sys=(m.get("v_o3", np.nan) - m.get("v_sys", np.nan)) if np.isfinite(m.get("v_o3", np.nan))
               else (m.get("v_o3_pre", np.nan) - m.get("v_sys", np.nan)),
               o3_snr=m.get("o3_core_snr", m.get("o3_pre_snr", np.nan)))
    for k in LINE_KEYS:
        rec[k] = m.get(k, np.nan)
    rec["errors_mc"] = dict(e)
    rec["bic_all"] = list(m.get("bic_all", []))
    return rec


def _print_fit_table(lines, res, out):
    hdr = (f"{'line':7} {'cls':3} {'dv=c50-sys':>12} {'+/-mc':>6} {'+/-mod':>6} {'peak':>7} {'cen':>7} {'FWHM':>6} {'nb':>2} "
           f"{'fS/N':>6} {'pS/N':>5} {'A.I.':>6} {'K.I.':>5} {'v_sys':>6} {'sS/N':>5} {'[SII]':>6} {'[OIII]':>7} {'oS/N':>5} flags")
    out(hdr)
    for name in ("Halpha", "Hbeta", "MgII"):
        if name not in lines:
            continue
        L = lines[name]
        if not L["fitted"]:
            out(f"{name:7} {'-':3} {'not fitted (window not covered)':>12}")
            continue
        out(f"{name:7} {L['label']:3} {_fmt(L['dv'], 12, 0, True)} {_fmt(L['dv_err_mc'], 6)} {_fmt(L['dv_err_model'], 6)} "
            f"{_fmt(L['v_peak_sys'], 7, 0, True)} {_fmt(L['centroid_sys'], 7, 0, True)} {_fmt(L['fwhm'], 6)} {L['n_broad']:>2} "
            f"{_fmt(L['broad_flux_snr'], 6, 1)} {_fmt(L['broad_peak_snr'], 5, 1)} {_fmt(L['AI'], 6, 2, True)} {_fmt(L['KI'], 5, 2)} "
            f"{_fmt(L['v_sys'], 6, 0, True)} {_fmt(L['sys_snr'], 5, 1)} {_fmt(L['v_sii_minus_sys'], 6, 0, True)} "
            f"{_fmt(L['v_o3_minus_sys'], 7, 0, True)} {_fmt(L['o3_snr'], 5, 1)} {','.join(L['flags']) or '-'}")
    for name in ("Halpha", "Hbeta", "MgII"):
        if name in lines and lines[name]["fitted"]:
            L = lines[name]
            out(f"  {name}: class {L['label']} ({L['class_text']}); {'; '.join(L['reasons'])}; "
                f"systemic from {L['systemic_source']}; measurable {L['measurable']}, strong offset {L['strong_offset']}")
    hi = res["host_info"]
    out(f"  continuum: power-law slope {res['conti'].get('pl_alpha', np.nan):+.2f}; host "
        + (f"{hi.get('host_frac_4200_5000', np.nan):.2f} of the 4200-5000 A flux ({hi.get('n_gal', 0)} eigenspectra)"
           if hi.get("applied") else f"not used ({hi.get('reason', '')})"))
    out("  velocities in km/s; dv = c50 - v_sys with c50 the half-maximum bisector c(1/2); "
        "+/-mc from the Monte Carlo (--nmc), +/-mod from the DESI repeat-spectrum error model")


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------
def cmd_fit(a):
    sp, z, zsrc, ebv, esrc = _load(a.spectrum, a, a.targetid)
    names = {k.lower(): k for k in COMPLEX_WINDOW}
    lines = [names.get(s.strip().lower(), s.strip()) for s in a.lines.split(",") if s.strip()]
    bad = [s for s in lines if s not in COMPLEX_WINDOW]
    if bad:
        sys.exit(f"unknown line(s) {bad}; choose from {list(COMPLEX_WINDOW)}")
    out = (lambda *x: None) if a.quiet else print
    if 0 < a.nmc < 10:
        out(f"note: {a.nmc} Monte Carlo realisations give unreliable percentiles; the catalogue used 30")
    out(f"blrfit {__version__}: {a.spectrum}" + (f" TARGETID {int(a.targetid)}" if a.targetid else "")
        + f"  z = {z:.5f} ({zsrc})  E(B-V) = {ebv:.4f} ({esrc})  lines {','.join(lines)}"
        + (f"  Monte Carlo {a.nmc}" if a.nmc else ""))
    res = fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z, ebv=ebv, host=not a.no_host, fe=not a.no_fe,
                       complexes=tuple(lines), max_broad=a.max_broad, dbic=a.dbic, nmc=a.nmc, seed=a.seed,
                       err_floor=a.err_floor)
    recs = {name: _line_record(name, res) for name in lines}
    _print_fit_table(recs, res, out)

    stem = a.stem or _stem(a.spectrum, sp, a.targetid)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, stem)
    hi = res["host_info"]
    doc = dict(blrfit_version=__version__,
               input=dict(path=os.path.abspath(a.spectrum), kind=sp.get("kind"), targetid=(int(a.targetid) if a.targetid else None),
                          z=z, z_source=zsrc, ebv=ebv, ebv_source=esrc,
                          ra=sp.get("ra", np.nan), dec=sp.get("dec", np.nan), mjd=sp.get("mjd", np.nan),
                          date=mjd_to_date(sp["mjd"]) if np.isfinite(sp.get("mjd", np.nan)) else "",
                          flux_unit=("1e-17 erg/s/cm^2/A" if sp.get("kind") in ("sdss", "desi") or a.flux_scale != 1.0
                                     else "as given (luminosities assume 1e-17 erg/s/cm^2/A)"),
                          n_pixels=int(len(sp["wave"])), wave_min=float(np.min(sp["wave"])), wave_max=float(np.max(sp["wave"]))),
               settings=dict(res["settings"], complexes=lines, nmc=a.nmc, seed=a.seed),
               continuum=dict(pl_alpha=res["conti"].get("pl_alpha", np.nan), pl_norm=res["conti"].get("pl_norm", np.nan),
                              feop_norm=res["conti"].get("feop_norm", np.nan), feop_fwhm=res["conti"].get("feop_fwhm", np.nan),
                              feuv_norm=res["conti"].get("feuv_norm", np.nan),
                              host_applied=bool(hi.get("applied", False)), host_frac=hi.get("host_frac_4200_5000", np.nan),
                              host_n_gal=hi.get("n_gal", 0), host_reason=hi.get("reason", "")),
               o3_prefit=res.get("o3_prefit", {}),
               lines=recs, summary_row=summary_row(res))
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
        m = res["meas"].get(line, {}); c = res["cls"].get(line, {})
        epochs.append(dict(meta, fitted=line in res["fits"], label=c.get("label", ""), flags=list(c.get("flags", [])),
                           c50_sys=m.get("c50_sys", np.nan), v_peak_sys=m.get("v_peak_sys", np.nan),
                           fwhm=m.get("fwhm", np.nan), broad_flux_snr=m.get("broad_flux_snr", np.nan),
                           v_sys=m.get("v_sys", np.nan), line=_line_record(line, res)))
        out(f"  {os.path.basename(meta['path'])}: MJD {_fmt(meta['mjd'], 8, 1)} {line} class {c.get('label', '-')} "
            f"c50-sys {_fmt(m.get('c50_sys', np.nan), 6, 0, True)} FWHM {_fmt(m.get('fwhm', np.nan), 5)} "
            f"flux S/N {_fmt(m.get('broad_flux_snr', np.nan), 5, 1)} flags {','.join(c.get('flags', [])) or '-'}")
    pair = (RV.pair_analysis(res2, res1, name=line, vmax=a.vmax, n_mc=a.nmc, details=True)
            if (line in res1["fits"] and line in res2["fits"]) else None)
    doc = dict(line=line, epochs=epochs)
    if pair is None:
        doc.update(measured=False, reason="the line is not fitted in both epochs or the cross-correlation failed")
        out(f"  {line}: cross-correlation not possible: " + doc["reason"])
        return doc, None
    same_desi = sp1.get("kind") == "desi" and sp2.get("kind") == "desi"
    grade = pair["profile_grade"]
    # error floor: the DESI-DESI systematic for two DESI spectra, the graded
    # cross-survey null otherwise (the grade inflates it for changed profiles)
    if same_desi:
        floor = RV.systematic_floor(line, pair.get("snr_proxy", np.nan)); floor_kind = "desi_sys"
    else:
        floor = RV.cross_survey_floor(line, grade); floor_kind = f"cross_survey_null_{grade}"
    err_total = float(np.hypot(pair["err"], floor)) if (np.isfinite(pair["err"]) and np.isfinite(floor)) else np.nan
    zp_ok = bool(pair.get("zp_ok", False))
    dv_corr = pair["dv"] - pair["zp_dv"] if (line == "Hbeta" and zp_ok) else pair["dv"]
    reliable = RV.is_reliable(pair)
    cut = RV.CCF_DIR_CUT_KMS.get(line, np.nan)
    why = ("at bound" if pair["at_bound"] else
           (f"profile_z {pair['profile_z']:.1f} >= {RV.CCF_PROFILE_Z_MAX:.0f}" if not (pair["profile_z"] < RV.CCF_PROFILE_Z_MAX) else
            (f"direction mismatch {pair['dir_mismatch']:.0f} >= {cut:.0f} km/s" if not (pair["dir_mismatch"] < cut) else "")))
    s_ab = (pair.get("details") or {}).get("s_ab") or {}
    doc.update(measured=True, dv=pair["dv"], err=pair["err"], err_dchi2=s_ab.get("err_dchi2", np.nan),
               err_method="bootstrap" if a.nmc >= 10 else "dchi2",
               profile_grade=grade, resid_frac=pair["resid_frac"], error_floor=floor, error_floor_kind=floor_kind,
               err_total=err_total, sigma_sys_desi=RV.systematic_floor(line, pair.get("snr_proxy", np.nan)),
               reliable_reason=why,
               consistent=pair["consistent"], dir_mismatch=pair["dir_mismatch"], profile_z=pair["profile_z"],
               chi2_red=pair["chi2_red"], at_bound=pair["at_bound"], regridded=pair["regridded"], npix=pair["npix"],
               snr_proxy=pair["snr_proxy"], zp_dv=pair["zp_dv"], zp_err=pair["zp_err"], zp_line=pair["zp_line"], zp_ok=zp_ok,
               zp_applied=bool(line == "Hbeta" and zp_ok), dv_corrected=dv_corr, reliable=bool(reliable),
               significance=float(abs(dv_corr) / err_total) if (np.isfinite(err_total) and err_total > 0) else np.nan,
               c50_difference=epochs[1]["c50_sys"] - epochs[0]["c50_sys"],
               dv_abs_epoch1=epochs[0]["c50_sys"], dv_abs_epoch2=epochs[0]["c50_sys"] + dv_corr)
    out(f"  {line}: shift of epoch 2 relative to epoch 1 {pair['dv']:+.0f} +/- {pair['err']:.0f} km/s; "
        f"with the {'DESI floor' if same_desi else 'cross-survey floor'} {floor:.0f} ({floor_kind}): +/- {err_total:.0f}; "
        f"directions {'agree' if pair['consistent'] else 'DISAGREE'} (mismatch {pair['dir_mismatch']:.0f}); "
        f"{'regridded' if pair['regridded'] else 'same grid'}; {'at bound' if pair['at_bound'] else 'inside search range'}")
    out(f"  {line}: profile grade {grade} (z_prof {pair['profile_z']:.1f}, residual {100 * pair['resid_frac']:.1f} per cent of the peak); "
        f"reliable tier {reliable}{(' (' + why + ')') if why else ''}")
    out(f"  {line}: narrow-line zero-point ({pair['zp_line'] or 'none'}) {_fmt(pair['zp_dv'], 5, 0, True)} +/- {_fmt(pair['zp_err'], 4)} km/s "
        f"({'applied' if doc['zp_applied'] else 'not applied'}) -> dv = {dv_corr:+.0f} km/s ({doc['significance']:.1f} sigma); "
        f"c(1/2) difference of the two fits {doc['c50_difference']:+.0f}; offset from the narrow lines "
        f"epoch 1 {epochs[0]['c50_sys']:+.0f}, epoch 2 {doc['dv_abs_epoch2']:+.0f} km/s")
    return doc, pair


def cmd_rv(a):
    tids = [t.strip() for t in (a.targetid or "").split(",") if t.strip()]
    tid1 = int(tids[0]) if tids else None
    tid2 = int(tids[1]) if len(tids) > 1 else tid1
    sp1, z1, zsrc1, ebv1, esrc1 = _load(a.epoch1, a, tid1)
    a2 = argparse.Namespace(**vars(a)); a2.z = z1          # both epochs at the same redshift
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
        sys.exit("--nmc must be at least 10: the bootstrap error is kept only when at least 10 realisations succeed")
    complexes = tuple(c for c in ("Halpha", "Hbeta", "MgII") if c in lines or (c in ("Halpha", "Hbeta") and set(lines) & {"Halpha", "Hbeta"}))
    out = (lambda *x: None) if a.quiet else print
    out(f"blrfit {__version__} rv: {','.join(lines)}, z = {z:.5f} ({zsrc1}), E(B-V) {ebv1:.4f} / {ebv2:.4f}")
    res1 = fit_spectrum(sp1["wave"], sp1["flux"], sp1["ivar"], z, ebv=ebv1, complexes=complexes)
    res2 = fit_spectrum(sp2["wave"], sp2["flux"], sp2["ivar"], z, ebv=ebv2, complexes=complexes)
    epochs_meta = [dict(path=os.path.abspath(path), kind=sp.get("kind"), targetid=tid, mjd=sp.get("mjd", np.nan),
                        date=mjd_to_date(sp["mjd"]) if np.isfinite(sp.get("mjd", np.nan)) else "")
                   for path, sp, tid in ((a.epoch1, sp1, tid1), (a.epoch2, sp2, tid2))]
    doc = dict(blrfit_version=__version__, z=z, convention="dv > 0: epoch 2 redshifted relative to epoch 1", lines={})
    mjd1, mjd2 = epochs_meta[0]["mjd"], epochs_meta[1]["mjd"]
    if np.isfinite(mjd1) and np.isfinite(mjd2):
        doc["baseline_days"] = float(mjd2 - mjd1); doc["baseline_rest_yr"] = float((mjd2 - mjd1) / 365.25 / (1 + z))
    pairs = {}
    for line in lines:
        rec, pair = _rv_line(line, res1, res2, sp1, sp2, epochs_meta, a, out)
        doc["lines"][line] = rec; pairs[line] = pair
    first = doc["lines"][lines[0]]
    doc.update({k: v for k, v in first.items() if k != "lines"})     # the first line's record at the top level
    if "Halpha" in pairs and "Hbeta" in pairs:
        tl = RV.two_line_consistent(pairs["Halpha"], pairs["Hbeta"])
        doc["two_line"] = tl if tl is not None else dict(consistent=False, reason="one of the lines has no usable shift")
        if tl is not None:
            out(f"  two-line criterion (Halpha vs Hbeta): {'consistent' if tl['consistent'] else 'NOT consistent'} "
                f"(difference {tl['difference']:+.0f} km/s, {tl['sigma']:.1f} sigma{'' if tl['same_sign'] else ', opposite signs'})")
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
        fig, axes = plt.subplots(len(shown), 2, figsize=(13, 4.6 * len(shown)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1.6, 1]))
        for k, line in enumerate(shown):
            pair = pairs[line]
            plot_epochs_overlay([res1, res2], [epochs_meta[0]["date"] or "epoch 1", epochs_meta[1]["date"] or "epoch 2"],
                                name=line, title=f"{stem}: {line}", ax=axes[k, 0])
            plot_ccf(pair, title=f"{line}: dv = {pair['dv']:+.0f} +/- {pair['err']:.0f} km/s, grade {pair['profile_grade']}",
                     ax=axes[k, 1])
        fig.tight_layout(); fig.savefig(base + "_rv.png", dpi=110)
        out(f"-> {base}_rv.png")
    return 0


# ----------------------------------------------------------------------------
# fetch
# ----------------------------------------------------------------------------
def cmd_fetch(a):
    from .io import fetch as F
    os.makedirs(a.out, exist_ok=True)
    found = dict(ra=a.ra, dec=a.dec, sdss=[], desi=[])
    searched = 0
    if not a.no_sdss:
        try:
            found["sdss"] = F.fetch_sdss(a.ra, a.dec, a.out, radius_arcsec=a.radius); searched += 1
        except ImportError as e:
            print(f"SDSS lookup skipped: {e} (pip install 'blrfit[fetch]')")
    if not a.no_desi:
        try:
            found["desi"] = F.fetch_desi(a.ra, a.dec, a.out, targetid=a.targetid, radius_arcsec=a.desi_radius,
                                         releases=tuple(r.strip() for r in a.releases.split(",") if r.strip()))
            searched += 1
        except ImportError as e:
            print(f"DESI lookup skipped: {e} (pip install 'blrfit[fetch]')")
    with open(os.path.join(a.out, "fetch_manifest.json"), "w") as fh:
        json.dump(_clean(found), fh, indent=1)
    n = len(found["sdss"]) + len(found["desi"])
    print(f"{n} spectrum(s) in {a.out}; manifest {os.path.join(a.out, 'fetch_manifest.json')}")
    if searched == 0:
        print("nothing was searched: install the fetch extras (pip install 'blrfit[fetch]')")
        return 1
    return 0


# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="blrfit", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Broad AGN emission lines against the narrow-line systemic velocity: "
                                            "offsets, profile classes, quality flags and velocity changes between epochs.")
    p.add_argument("--version", action="version", version=f"blrfit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit one spectrum", description="Fit one spectrum; writes <stem>_fit.json and <stem>_fit.png.")
    f.add_argument("spectrum", help="SDSS spec-*.fits, DESI coadd-*.fits (with --targetid) or a table")
    f.add_argument("--targetid", type=int, default=None, help="DESI TARGETID (coadd input)")
    f.add_argument("--z", type=float, default=None, help="redshift (default: from the file when it has one)")
    f.add_argument("--ebv", default=None, help="Galactic E(B-V); a number, or 'sfd' for a dust-map lookup (default: DESI FIBERMAP value, else 0)")
    f.add_argument("--ra", type=float, default=None, help="right ascension (deg), for --ebv sfd when the file has no coordinates")
    f.add_argument("--dec", type=float, default=None, help="declination (deg)")
    f.add_argument("--lines", default="Halpha,Hbeta,MgII",
                   help="comma-separated complexes among Halpha, Hbeta, MgII (default all three; those outside the data are skipped)")
    f.add_argument("--nmc", type=int, default=0, help="Monte Carlo realisations for the errors (default 0 = none; the catalogue used 30)")
    f.add_argument("--seed", type=int, default=0, help="random seed of the Monte Carlo (default 0)")
    f.add_argument("--no-host", action="store_true", help="no host-galaxy component")
    f.add_argument("--no-fe", action="store_true", help="no Fe II templates")
    f.add_argument("--max-broad", type=int, default=MAX_BROAD, help=f"maximum number of broad Gaussians (default {MAX_BROAD})")
    f.add_argument("--dbic", type=float, default=DBIC, help=f"BIC improvement required for one more component (default {DBIC:.0f})")
    f.add_argument("--err-floor", type=float, default=ERR_FLOOR, help="fractional error floor (default 0.02)")
    f.add_argument("--out", default=".", help="output directory"); f.add_argument("--stem", default=None, help="output file stem")
    f.add_argument("--no-figure", action="store_true", help="do not write the diagnostic figure")
    f.add_argument("--pickle", action="store_true", help="also write the full result as <stem>_fit.pkl")
    f.add_argument("--quiet", action="store_true", help="print nothing")
    _add_table_args(f)
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("rv", help="velocity change between two spectra of one object",
                       description="Cross-correlate the broad line of two epochs; writes <stem>_rv.json and <stem>_rv.png.")
    r.add_argument("epoch1"); r.add_argument("epoch2")
    r.add_argument("--targetid", default=None, help="DESI TARGETID, or two comma-separated (one per epoch)")
    r.add_argument("--z", type=float, default=None, help="redshift used for both epochs (default: from epoch 1's file)")
    r.add_argument("--ebv", default=None, help="Galactic E(B-V) for both epochs (default: per file as in fit)")
    r.add_argument("--ra", type=float, default=None, help="right ascension (deg), for --ebv sfd")
    r.add_argument("--dec", type=float, default=None, help="declination (deg)")
    r.add_argument("--line", default="Halpha", help="Halpha (default), Hbeta, MgII, or a comma-separated list; "
                   "with Halpha,Hbeta the two-line criterion is evaluated")
    r.add_argument("--vmax", type=float, default=2000.0, help="search range of the shift, km/s (default 2000)")
    r.add_argument("--nmc", type=int, default=0,
                   help="bootstrap realisations for the cross-correlation error, at least 10 (default 0 = Delta chi-square error)")
    r.add_argument("--out", default=".", help="output directory"); r.add_argument("--stem", default=None, help="output file stem")
    r.add_argument("--no-figure", action="store_true", help="do not write the figure")
    r.add_argument("--quiet", action="store_true", help="print nothing")
    _add_table_args(r)
    r.set_defaults(func=cmd_rv)

    g = sub.add_parser("fetch", help="download public SDSS and DESI spectra of a position",
                       description="Look a position up in SDSS (astroquery) and the DESI public releases; writes the spectra and fetch_manifest.json.")
    g.add_argument("--ra", type=float, required=True); g.add_argument("--dec", type=float, required=True)
    g.add_argument("--out", default="spectra")
    g.add_argument("--radius", type=float, default=2.0, help="SDSS search radius, arcsec")
    g.add_argument("--desi-radius", type=float, default=1.0, help="DESI position match radius, arcsec")
    g.add_argument("--targetid", type=int, default=None, help="DESI TARGETID (selects the row exactly)")
    g.add_argument("--releases", default="dr1,edr", help="DESI releases to search")
    g.add_argument("--no-sdss", action="store_true"); g.add_argument("--no-desi", action="store_true")
    g.set_defaults(func=cmd_fetch)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
