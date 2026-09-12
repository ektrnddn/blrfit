"""
Profile classes and quality flags: explicit rules on the measured quantities.

Classes
    A  bulk shift: |c(1/2) - v_sys| > 300 km/s at more than 3 sigma (when an
       error is available) and a symmetric profile (straight bisector,
       |A.I.| < 0.12, peak close to the centroid)
    B  double-peaked / disk-like: two resolved peaks separated by more than
       max(0.4 FWHM, 1500 km/s) with a dip deeper than 8 per cent and
       FWHM >= 3000, or FWHM >= 7000 with A.I. >= 0.20 or K.I. >= 0.50
    C  asymmetric single-peaked
    F  normal: symmetric, no significant offset
    E  no significant broad component (FWHM < 1200, integrated S/N < 5 or
       peak S/N < 1.5)
    X  no measurable systemic reference (narrow-line S/N < 3)
    W  a broad component that passes the 1200 km/s boundary but is too narrow
       (FWHM < 2000) or too weak (integrated S/N < 8) to classify: narrow-line
       residuals dominate such profiles

Classes B and C are not cleanly separable from single-epoch shape statistics
alone; the operative distinction is A against the rest.

Flags (a measurement with none of them is "clean")
    very_broad       FWHM >= 8000 km/s
    poor_fit         reduced chi-square >= 2.5
    low_snr          integrated broad S/N < 10
    low_peak_snr     broad peak < 5 sigma per pixel: a component significant only
                     by integration over thousands of km/s is degenerate with
                     continuum-subtraction residuals (host-template mismatch)
    host_dominated   host >= 80 per cent of the 4200-5000 A light: template
                     mismatch at the few-per-cent level mimics a very broad line
    pl_at_bound      power-law slope at a bound: the continuum model is not
                     describing the spectrum
    peak_disagree    model peak and data peak differ by > 0.25 FWHM
    sii_disagree     [S II] velocity > 150 km/s from the systemic: narrow Halpha
                     contaminated by the broad line, or a complex narrow-line region
    sys_disagree     systemic and [O III] core > 400 km/s apart: a misidentified
                     narrow group or a strongly offset [O III]
    narrow_at_bound  narrow group at the edge of its +/-1500 km/s window
    edge             data cover less than +/-6000 km/s around the line
    extreme_offset   |c(1/2) - v_sys| > 4000 km/s: beyond the Roche ceiling of
                     almost any bound binary; a disk-emitter component, an
                     artefact or a misidentified line
"""
from __future__ import annotations

import numpy as np

from .constants import (DEFAULT_THRESH, V_NARROW_MAX, EDGE_FLAG_KMS, VERY_BROAD_FWHM,
                        POOR_FIT_CHI2, LOW_SNR_FLUX, DPE_SEP_MIN_KMS, VERY_BROAD_B_FWHM,
                        VERY_BROAD_B_AI, VERY_BROAD_B_KI, PEAK_DISAGREE_FRAC,
                        SYS_DISAGREE_MIN_SNR, MEASURABLE_CLASSES,
                        MEASURABLE_MIN_SNR, MEASURABLE_MIN_FWHM, STRONG_OFFSET_KMS)

LABEL_TEXT = {"A": "bulk shift", "B": "double-peaked / disk", "C": "asymmetric",
              "F": "normal", "E": "no broad line", "X": "no systemic reference",
              "W": "weak/narrow broad component - not classified"}

FLAG_TEXT = {
    "very_broad": "FWHM >= 8000 km/s",
    "poor_fit": "reduced chi-square >= 2.5",
    "low_snr": "integrated broad S/N < 10",
    "low_peak_snr": "broad peak < 5 sigma per pixel",
    "host_dominated": "host >= 80 per cent of the 4200-5000 A light",
    "pl_at_bound": "power-law slope at a bound",
    "peak_disagree": "model and data peaks differ by > 0.25 FWHM",
    "sii_disagree": "[S II] velocity > 150 km/s from the systemic",
    "sys_disagree": "systemic and [O III] core > 400 km/s apart",
    "narrow_at_bound": "narrow group at the edge of its +/-1500 km/s window",
    "edge": "data cover < +/-6000 km/s around the line",
    "extreme_offset": "|c(1/2) - v_sys| > 4000 km/s",
}


def classify(m, err=None, t=None):
    """Class, reasons, features and flags for the output of ``measure_complex``.

    ``err`` is the dictionary of Monte Carlo errors of the same complex (may be
    empty); ``t`` overrides entries of ``DEFAULT_THRESH``.
    Returns dict(label, reasons, features, flags).
    """
    t = dict(DEFAULT_THRESH, **(t or {}))
    err = err or {}
    R, flags = [], []
    fw = m.get("fwhm", np.nan)
    if (not np.isfinite(fw) or fw < t["min_fwhm"]
            or not (m.get("broad_flux_snr", 0) >= t["min_broad_flux_snr"])
            or not (m.get("broad_peak_snr", 0) >= t["min_broad_peak_snr"])):
        return dict(label="E", reasons=["no significant broad component"], features={}, flags=flags)
    sys_snr = m.get("sys_snr", m.get("narrow_peak_snr", 0))
    if not np.isfinite(m.get("v_sys", np.nan)) or not (sys_snr >= t["min_narrow_snr"]):
        return dict(label="X", reasons=[f"systemic reference not measurable (S/N {sys_snr:.1f})"],
                    features={}, flags=flags)
    if fw < t["class_fwhm"] or m.get("broad_flux_snr", 0) < t["class_flux_snr"]:
        return dict(label="W", reasons=[f"broad component too narrow/weak to classify "
                    f"(FWHM {fw:.0f}, flux S/N {m.get('broad_flux_snr', np.nan):.1f})"], features={}, flags=flags)

    if fw >= VERY_BROAD_FWHM: flags.append("very_broad")
    if m.get("chi2_red", 0) >= POOR_FIT_CHI2: flags.append("poor_fit")
    if m.get("broad_flux_snr", 99) < LOW_SNR_FLUX: flags.append("low_snr")
    if m.get("broad_peak_snr", 99) < t["flag_peak_snr"]: flags.append("low_peak_snr")
    if m.get("host_frac", 0) >= t["flag_host_frac"]: flags.append("host_dominated")
    pa = m.get("pl_alpha", np.nan)
    if np.isfinite(pa) and (pa <= -4.9 or pa >= 2.9): flags.append("pl_at_bound")   # within 0.1 of the bounds -5, 3
    vsys = m.get("v_sys", np.nan)
    if np.isfinite(vsys) and abs(vsys) >= 0.95 * V_NARROW_MAX: flags.append("narrow_at_bound")
    if (m.get("v_cover_lo", -1e9) > -EDGE_FLAG_KMS) or (m.get("v_cover_hi", 1e9) < EDGE_FLAG_KMS):
        flags.append("edge")
    if m.get("systemic_source", "") != "[OIII] core":
        vo3 = m.get("v_o3", np.nan) if np.isfinite(m.get("v_o3", np.nan)) else m.get("v_o3_pre", np.nan)
        so3 = m.get("o3_core_snr", np.nan) if np.isfinite(m.get("o3_core_snr", np.nan)) else m.get("o3_pre_snr", 0.0)
        if (np.isfinite(vo3) and np.isfinite(vsys) and so3 >= SYS_DISAGREE_MIN_SNR
                and abs(vo3 - vsys) > t["sys_disagree_kms"]):
            flags.append("sys_disagree")
    vs2 = m.get("v_sii", np.nan)
    if (np.isfinite(vs2) and np.isfinite(m.get("v_sys", np.nan))
            and m.get("flux_SII6716", 0) + m.get("flux_SII6731", 0) > 0
            and abs(vs2 - m["v_sys"]) > t["sii_disagree_kms"]):
        flags.append("sii_disagree")
    pt = m.get("peak_top_sys", np.nan)
    if np.isfinite(pt) and abs(pt - m["v_peak_sys"]) > PEAK_DISAGREE_FRAC * fw: flags.append("peak_disagree")

    off = m["c50_sys"]; e_off = err.get("c50_sys", np.nan)
    tilt = abs(m["c25"] - m["c75"]) / fw
    pc = abs(m["v_peak"] - m["centroid"]) / fw
    ai = abs(m["AI"]); ki = m["KI"]
    feats = dict(offset=off, e_offset=e_off, tilt=tilt, peak_minus_cen=pc, AI=ai, KI=ki,
                 n_peaks=m["n_peaks"], peak_sep=m.get("peak_sep", np.nan),
                 dip=m.get("dip_frac", np.nan), fwhm=fw)

    if (fw >= t["dpe_min_fwhm"] and m["n_peaks"] >= 2
            and m["peak_sep"] > max(0.4 * fw, DPE_SEP_MIN_KMS) and m["dip_frac"] > t["dpe_dip"]):
        R.append(f"two peaks {m['peak_sep']:.0f} km/s apart, dip {m['dip_frac']:.2f}")
        return dict(label="B", reasons=R, features=feats, flags=flags)
    if fw >= VERY_BROAD_B_FWHM and (ai >= VERY_BROAD_B_AI or (np.isfinite(ki) and ki >= VERY_BROAD_B_KI)):
        R.append(f"very broad ({fw:.0f}) and asymmetric/flat: A.I. {ai:.2f}, K.I. {ki:.2f}")
        return dict(label="B", reasons=R, features=feats, flags=flags)

    if abs(off) > t["extreme_kms"]: flags.append("extreme_offset")
    sig = abs(off) > t["offset_kms"] and (not np.isfinite(e_off) or abs(off) > t["offset_nsig"] * e_off)
    symmetric = (tilt < t["tilt_frac"]) and (ai < t["ai_max"]) and (pc < t["peak_cen_frac"])
    if sig and symmetric:
        R.append(f"c50 offset {off:+.0f} km/s, flat bisector (tilt {tilt:.2f} FWHM), A.I. {ai:.2f}")
        return dict(label="A", reasons=R, features=feats, flags=flags)
    if not symmetric:
        R.append(f"asymmetric: tilt {tilt:.2f} FWHM, A.I. {ai:.2f}, |peak-cen| {pc:.2f} FWHM"
                 + (f", offset {off:+.0f}" if sig else ", no net offset"))
        return dict(label="C", reasons=R, features=feats, flags=flags)
    R.append(f"symmetric, offset {off:+.0f} km/s not significant")
    return dict(label="F", reasons=R, features=feats, flags=flags)


def is_measurable(label, flags, broad_flux_snr, fwhm):
    """The catalogue definition of a measurable offset: class A/B/C/F, integrated
    S/N >= 8, FWHM >= 2000 km/s and no 'edge' flag."""
    return (label in MEASURABLE_CLASSES and np.isfinite(broad_flux_snr) and broad_flux_snr >= MEASURABLE_MIN_SNR
            and np.isfinite(fwhm) and fwhm >= MEASURABLE_MIN_FWHM and "edge" not in (flags or []))


def is_strong_offset(label, flags, broad_flux_snr, fwhm, dv):
    """Measurable and 1000 <= |Delta v| <= 4000 km/s."""
    lo, hi = STRONG_OFFSET_KMS
    return is_measurable(label, flags, broad_flux_snr, fwhm) and np.isfinite(dv) and lo <= abs(dv) <= hi
