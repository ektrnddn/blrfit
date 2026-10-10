"""
Candidate tiers from the measured velocity changes of a target.

One verdict per target with its reasons, from the rows of a pair table
(``pairs.pair_record``): the retained pairs whose two epochs both reach the
S/N floor, with the total error (statistical plus the calibrated term). The
tiers, in the order they are tried:

disk
    a double-peaked (class B) broad profile in the reference spectrum, whatever
    its changes: a disc emitter, whose peaks move with the disc; the largest
    change is reported.
binary, platinum
    at least one pair whose change is at least TIER_SIG_BINARY times its total
    error, with a stable profile in both directions (shape statistic at most
    PAIR_SHAPE_MAX per pixel), no frame flag, the other Balmer line agreeing
    where it is measured, no class B epoch in the pair, a change that an orbit
    at the virial mass allows (where a mass is given), and, with three or more
    epochs, changes relative to the reference that do not reverse sign at that
    significance twice. The platinum tier, the two-line criterion of Guo et al.
    (2019) made strict, also needs the other line measured and agreeing, a shape statistic at most TIER_PLATINUM_SHAPE_MAX, the two
    directions within TIER_PLATINUM_MISMATCH_NSIG sigma of each other and a broad
    peak S/N of at least TIER_PLATINUM_PEAK_SNR in both epochs.
almost
    a significant pair that fails exactly one of the shape, two-line and frame
    conditions; a marginal pair (TIER_SIG_MARGINAL to TIER_SIG_BINARY sigma)
    that passes them all; or a significant change that no orbit at the mass
    allows, or that reverses sign.
profile
    the strongest change comes with a changed profile and no clean significant
    pair exists: variability of the line, not a bulk motion.
stable
    retained pairs, none above TIER_SIG_MARGINAL sigma: an upper limit.
none
    no retained pair with both epochs at the S/N floor.

The thresholds are those of the validation (``constants.py``); a class B epoch
in a pair, a frame flag or a disagreement of the two fits' own systemic
velocities (TIER_VSYS_MAX_KMS) mark the pair, and the reasons name the pair,
the line and the numbers behind every verdict.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .constants import (
    PAIR_SHAPE_MAX,
    TIER_SIG_BINARY,
    TIER_SIG_MARGINAL,
    TIER_TWO_LINE_NSIG,
    TIER_VSYS_MAX_KMS,
    TIER_SNR_MIN,
    TIER_PLATINUM_SHAPE_MAX,
    TIER_PLATINUM_MISMATCH_NSIG,
    TIER_PLATINUM_PEAK_SNR,
    TIER_ORBIT_NSIG,
    CCF_SCALE_PRODUCT_RANGE,
)
from .pairs import independent
from .physics import max_orbital_change, longest_period

TIERS = ("platinum", "binary", "almost", "profile", "disk", "stable", "none")

TIER_TEXT = {
    "platinum": "a significant, clean change in both Balmer lines (the two-line criterion): the strongest binary candidates",
    "binary": "a significant change with a stable profile, consistent directions and an allowed orbit (one line)",
    "almost": "a significant change that fails one condition, or a marginal change that passes them all",
    "profile": "the strongest change comes with a changed profile: variability, not a bulk motion",
    "disk": "a double-peaked (class B) broad profile: a disc emitter",
    "stable": "no change above three sigma: an upper limit",
    "none": "no retained pair with both epochs at the S/N floor",
}

# the columns of the tier table, one row per target
TIER_COLUMNS = (
    "targetid",
    "tier",
    "reason",
    "line",
    "pair",
    "s_kms",
    "err_kms",
    "sigma",
    "shape_max",
    "two_line_agree",
    "dt_rest_yr",
    "orbital_ok",
    "p_max_q01_yr",
    "p_max_q1_yr",
    "p_min_q1_yr",
    "logmbh",
    "n_pairs",
)


def _f(v):
    """A float from a table cell: numbers, numeric strings, empty and masked cells."""
    if v is None or isinstance(v, np.ma.core.MaskedConstant):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _b(v):
    """A boolean from a table cell written as True/False or 1/0."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "t", "yes")


def _s(v):
    if v is None or isinstance(v, np.ma.core.MaskedConstant):
        return ""
    return str(v)


def pair_verdicts(rows):
    """The quantities of the tier rules per (target, pair), both lines side by
    side: one dict per pair with ``lines`` -> line -> dict(s, err, sig,
    retained, stable, shape, frame, classes, dt, role, scale, snr, peak_snr,
    mismatch_nsig). Rows of dependent spectra are left out; ``err`` is the total
    error where the table has one."""
    by = defaultdict(dict)
    for r in rows:
        id_a, id_b = _s(r.get("id_a")), _s(r.get("id_b"))
        if "dependent" in _s(r.get("flags")) or not independent(dict(id=id_a), dict(id=id_b)):
            continue
        by[(_s(r.get("targetid")), _s(r.get("pair")))].setdefault(_s(r.get("line")), r)
    out = []
    for (tid, pair), lines in by.items():
        v = dict(targetid=tid, pair=pair, lines={})
        for line, r in lines.items():
            s = _f(r.get("s_common"))
            e = _f(r.get("err_total"))
            if not np.isfinite(e):
                e = _f(r.get("err"))
            shape = _f(r.get("shape_max"))
            if not np.isfinite(shape):
                shape = max(
                    _f(r.get("dchi2_shape_fwd")) / max(_f(r.get("npix_fwd")), 1),
                    _f(r.get("dchi2_shape_rev")) / max(_f(r.get("npix_rev")), 1),
                )
            flags = _s(r.get("flags"))
            vs = _f(r.get("v_sys_diff"))
            frame_bad = (
                "frame_offset_large" in flags
                or "narrow_fwd:at_bound" in flags
                or "narrow_rev:at_bound" in flags
                or (np.isfinite(vs) and abs(vs) > TIER_VSYS_MAX_KMS)
            )
            prod = _f(r.get("scale_fwd")) * _f(r.get("scale_rev"))
            lo, hi = CCF_SCALE_PRODUCT_RANGE
            retained = _b(r.get("retained")) and (not np.isfinite(prod) or lo <= prod <= hi)
            err_stat = _f(r.get("err"))
            v["lines"][line] = dict(
                s=s,
                err=e,
                sig=abs(s) / e if (np.isfinite(s) and np.isfinite(e) and e > 0) else np.nan,
                retained=retained,
                stable=bool(np.isfinite(shape) and shape <= PAIR_SHAPE_MAX),
                shape=shape,
                frame=frame_bad,
                classes=(_s(r.get("cls_a")), _s(r.get("cls_b"))),
                dt=_f(r.get("dt_rest_yr")),
                role=_s(r.get("role")) or "consecutive",
                scale=(_f(r.get("scale_fwd")), _f(r.get("scale_rev"))),
                snr=(_f(r.get("snr_a")), _f(r.get("snr_b"))),
                peak_snr=(_f(r.get("peak_snr_a")), _f(r.get("peak_snr_b"))),
                mismatch_nsig=(_f(r.get("dir_mismatch")) / err_stat) if err_stat > 0 else np.nan,
            )
        out.append(v)
    return out


def orbital_screen(mass, s, err, dt_yr, nsig=TIER_ORBIT_NSIG):
    """Whether a change ``s`` +/- ``err`` over ``dt_yr`` (rest-frame years) is
    possible for an orbit at the target's virial mass. ``mass`` carries vmax_q01,
    pmin_q01_yr, vmax_q1, pmin_q1_yr and logmbh (``physics.target_mass``; the
    upper-case names of a catalogue table are accepted). The change is taken
    ``nsig`` sigma below its measured value. Returns dict(orbital_ok, dv_max_q01,
    dv_max_q1, p_max_q01_yr, p_max_q1_yr, p_min_q01_yr, p_min_q1_yr, logmbh);
    orbital_ok is None when no bound is available."""
    m = {str(k).lower(): v for k, v in dict(mass).items()}
    s_lo = max(abs(s) - nsig * err, 0.0) if np.isfinite(s) and np.isfinite(err) else np.nan
    out = dict(logmbh=_f(m.get("logmbh")))
    for tag in ("q01", "q1"):
        vmax, pmin = _f(m.get(f"vmax_{tag}")), _f(m.get(f"pmin_{tag}_yr", m.get(f"pmin_yr_{tag}")))
        dvmax, _ = max_orbital_change(vmax, pmin, dt_yr)
        out[f"dv_max_{tag}"] = dvmax
        out[f"p_max_{tag}_yr"] = longest_period(vmax, pmin, dt_yr, s_lo) if np.isfinite(dvmax) else np.nan
        out[f"p_min_{tag}_yr"] = pmin
    if np.isfinite(out["dv_max_q01"]) and np.isfinite(s_lo):
        out["orbital_ok"] = bool(s_lo <= out["dv_max_q01"])
    else:
        out["orbital_ok"] = None
    return out


def _counted(v):
    """The lines of a pair verdict that count: retained, with a significance and
    both epochs at the S/N floor."""
    return {
        k: x
        for k, x in v["lines"].items()
        if x["retained"]
        and np.isfinite(x["sig"])
        and all(np.isfinite(q) and q >= TIER_SNR_MIN for q in x["snr"])
    }


def _fmt_change(c):
    return f"{c['line']} {c['s']:+.0f} +/- {c['err']:.0f} km/s ({c['sig']:.1f} sigma)"


def classify_target(pairs, mass=None, reference_classes=None):
    """The tier of one target from its pair verdicts (``pair_verdicts``).
    ``mass``: the target's masses and orbital limits (``physics.target_mass``
    or a catalogue row) for the orbital condition; ``reference_classes``: the
    classes of the reference spectrum per line, for the disk tier. Returns
    (tier, reasons, best), ``best`` the pair and line behind the verdict or
    None."""
    best, marginal, profile, disk = None, None, None, False
    signed = defaultdict(list)
    if reference_classes and "B" in [str(c) for c in reference_classes.values()]:
        disk = True
        top = max(
            (x for v in pairs for x in _counted(v).values()),
            key=lambda x: x["sig"],
            default=None,
        )
        reason = "double-peaked (class B) broad profile in the reference spectrum"
        if top:
            reason += f"; largest change {top['sig']:.1f} sigma ({top['s']:+.0f} +/- {top['err']:.0f} km/s)"
        return "disk", [reason], top
    for v in pairs:
        counted = _counted(v)
        if any("B" in x["classes"] for x in v["lines"].values()):
            disk = True
        for line, x in counted.items():
            if "reference" in x["role"]:
                signed[line].append((x["dt"], x["s"] / x["err"]))
        for line, x in counted.items():
            other = [y for k, y in counted.items() if k != line]
            if other:
                agree = all(
                    abs(x["s"] - y["s"]) <= TIER_TWO_LINE_NSIG * np.hypot(x["err"], y["err"]) for y in other
                )
            else:
                agree = None
            cand = dict(
                pair=v["pair"],
                line=line,
                s=x["s"],
                err=x["err"],
                sig=x["sig"],
                stable=x["stable"],
                frame=x["frame"],
                agree=agree,
                shape=x["shape"],
                scale=x["scale"],
                snr=x["snr"],
                dt=x["dt"],
                peak_snr=x["peak_snr"],
                mismatch_nsig=x["mismatch_nsig"],
            )
            cand["platinum"] = bool(
                agree is True
                and np.isfinite(x["shape"])
                and x["shape"] <= TIER_PLATINUM_SHAPE_MAX
                and np.isfinite(cand["mismatch_nsig"])
                and cand["mismatch_nsig"] <= TIER_PLATINUM_MISMATCH_NSIG
                and all(np.isfinite(q) and q >= TIER_PLATINUM_PEAK_SNR for q in cand["peak_snr"])
            )
            fails = [
                name
                for name, bad in (
                    ("shape", not x["stable"]),
                    ("two-line", agree is False),
                    ("frame", x["frame"]),
                    ("class B", "B" in x["classes"]),
                )
                if bad
            ]
            cand["fails"] = fails
            if x["sig"] >= TIER_SIG_BINARY:
                if not fails:
                    if best is None or (cand["platinum"], x["sig"]) > (best["platinum"], best["sig"]):
                        best = cand
                elif len(fails) == 1 and (marginal is None or x["sig"] > marginal["sig"]):
                    marginal = cand
                if not x["stable"] and (profile is None or x["sig"] > profile["sig"]):
                    profile = cand
            elif (
                x["sig"] >= TIER_SIG_MARGINAL
                and not fails
                and (marginal is None or x["sig"] > marginal["sig"])
            ):
                marginal = cand
    if best is not None and mass is not None:
        orb = orbital_screen(mass, best["s"], best["err"], best.get("dt", np.nan))
        best["orbit"] = orb
        if orb["orbital_ok"] is False:
            return (
                "almost",
                [
                    f"{_fmt_change(best)} in {best['pair']} exceeds any orbit at log M {orb['logmbh']:.1f} "
                    f"(largest change {orb['dv_max_q01']:.0f} km/s over {best['dt']:.1f} yr, q = 0.1, edge-on)"
                ],
                best,
            )
    if best is not None:
        seq = sorted(signed.get(best["line"], []))
        strong = [np.sign(p) for _, p in seq if abs(p) >= TIER_SIG_BINARY]
        flips = sum(1 for a, b in zip(strong[:-1], strong[1:]) if a != b)
        if flips >= 2:
            return (
                "almost",
                [
                    f"{_fmt_change(best)} in {best['pair']}, but the changes reverse sign {flips} times at that significance"
                ],
                best,
            )
        orb = best.get("orbit") or {}
        ptxt = ""
        if orb.get("orbital_ok"):
            ptxt = (
                f"; P <= {orb['p_max_q01_yr']:.0f} yr (q = 0.1) / <= {orb['p_max_q1_yr']:.0f} yr (q = 1), "
                f"P_min {orb['p_min_q1_yr']:.0f} yr, log M {orb['logmbh']:.1f}"
            )
        tier = "platinum" if best.get("platinum") else "binary"
        return (
            tier,
            [
                f"{_fmt_change(best)}, stable shape (max shape {best['shape']:.2f} per pixel), "
                + ("the other line agrees" if best["agree"] else "the other line not measured")
                + f", |s + s'|/err {best['mismatch_nsig']:.1f}, peak S/N {best['peak_snr'][0]:.0f}/{best['peak_snr'][1]:.0f}, "
                f"in {best['pair']}" + ptxt
            ],
            best,
        )
    if marginal is not None:
        why = (
            "fails " + ", ".join(marginal["fails"])
            if marginal["fails"]
            else f"marginal ({marginal['sig']:.1f} sigma)"
        )
        return "almost", [f"{_fmt_change(marginal)} in {marginal['pair']}: {why}"], marginal
    if profile is not None:
        return (
            "profile",
            [
                f"{_fmt_change(profile)} with a changed profile (max shape {profile['shape']:.2f} per pixel) in {profile['pair']}"
            ],
            profile,
        )
    if disk:
        return "disk", ["a double-peaked (class B) epoch"], None
    usable = [x for v in pairs for x in _counted(v).values()]
    if usable:
        top = max(usable, key=lambda x: x["sig"])
        return (
            "stable",
            [f"largest change {top['sig']:.1f} sigma ({top['s']:+.0f} +/- {top['err']:.0f} km/s)"],
            None,
        )
    return "none", ["no retained pair with both epochs at the S/N floor"], None


def classify_table(rows, masses=None, classes=None):
    """Tiers of every target of a pair table. ``rows``: the pair records (dicts
    or table rows); ``masses``: targetid -> masses (``physics.target_mass``);
    ``classes``: targetid -> {line: class} of the reference spectrum. Returns a
    list of tier rows (``TIER_COLUMNS``), the strongest first."""
    masses, classes = masses or {}, classes or {}
    by_target = defaultdict(list)
    for v in pair_verdicts(rows):
        by_target[v["targetid"]].append(v)
    out = []
    for tid, ps in by_target.items():
        tier, reasons, best = classify_target(ps, masses.get(tid), classes.get(tid))
        b = best or {}
        o = b.get("orbit") or {}

        def num(x):
            return float(x) if x is not None and np.isfinite(_f(x)) else np.nan

        out.append(
            dict(
                targetid=tid,
                tier=tier,
                reason="; ".join(reasons),
                line=b.get("line", ""),
                pair=b.get("pair", ""),
                s_kms=num(b.get("s")),
                err_kms=num(b.get("err")),
                sigma=num(b.get("sig")),
                shape_max=num(b.get("shape")),
                two_line_agree="" if b.get("agree") is None else str(bool(b.get("agree"))),
                dt_rest_yr=num(b.get("dt")),
                orbital_ok="" if o.get("orbital_ok") is None else str(bool(o.get("orbital_ok"))),
                p_max_q01_yr=num(o.get("p_max_q01_yr")),
                p_max_q1_yr=num(o.get("p_max_q1_yr")),
                p_min_q1_yr=num(o.get("p_min_q1_yr")),
                logmbh=num(o.get("logmbh")),
                n_pairs=len(ps),
            )
        )
    out.sort(key=lambda r: (TIERS.index(r["tier"]), -(r["sigma"] if np.isfinite(r["sigma"]) else 0.0)))
    return out


__all__ = [
    "TIERS",
    "TIER_TEXT",
    "TIER_COLUMNS",
    "pair_verdicts",
    "orbital_screen",
    "classify_target",
    "classify_table",
]
