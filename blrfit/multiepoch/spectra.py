"""The spectrum check: every spectrum, per line, is 'good', 'double-peaked' or 'not used' with a reason, plus tags.

Rules:
  not used   line not measured (no broad S/N in the fit) | class E (no broad line) | class X (no narrow-line reference) |
             broad S/N < 8 | the fit did not converge | a blrfit quality flag that the data show spoils velocities
             (EXCLUDE_FLAGS, decided on pairs where no change is expected; the other flags stay on the record) | a DESI night inconsistent with the
             other nights of the same object within 30 d (a bad night: it disagrees at > 4 sigma with a night that itself
             agrees with a third one)
  double-peaked   class B by blrfit's two-peak rule (two resolved peaks): kept, not used for velocities in the main
             analysis. Class B by the very-broad rule (FWHM >= 7000 km/s, asymmetric or flat,
             one peak) is good with the tag very_broad_B. Tables without the branch
             (bpk) keep every class B as double-peaked.
  good       class A, C, F or W with broad S/N >= 8
  tags       weak_peak (broad peak < 5 sigma per pixel; such pairs carry honest errors), daily (a DESI
             daily reduction), boss (a BOSS/eBOSS spectrum), short_pair_unconfirmed (in a <= 30 d pair that disagrees at
             > 4 sigma, with no third night to tell which night is wrong)
"""

from collections import defaultdict

import numpy as np

from . import common as K

STAGE4_SYS = {"Halpha": 45.0, "Hbeta": 36.0}
# blrfit's per-line quality flags that exclude a spectrum for that line: only those that make velocities worse on pairs
# where no change is expected (a flag study on no-change pairs, its rule fixed before looking); every other flag is kept on the record
EXCLUDE_FLAGS = {"Halpha": (), "Hbeta": ()}
CLASS_REASON = {
    "E": "class E: no broad line",
    "X": "class X: no narrow-line reference (the frame check cannot run)",
}
# class W (a broad line 1200-2000 km/s wide, or too weak to classify) is a tag, not a cut: with broad S/N >= 8 such pairs
# scatter no more than A/C/F pairs (no-change DESI nights: pull NMAD 0.60 vs 0.94; long baselines 0.72-1.39 vs 1.48-2.56)


def frame_bad(r):
    vs = K.f(r.get("v_sys_diff"))
    fl = r["flags"]
    return (
        ("frame_offset_large" in fl)
        or ("narrow_fwd:at_bound" in fl)
        or ("narrow_rev:at_bound" in fl)
        or (np.isfinite(vs) and abs(vs) > K.VSYS_MAX)
    )


def measured(r):
    """The four checks of the pair (directions, reciprocal scales, directions agree, frame) passed."""
    prod = K.f(r.get("scale_fwd")) * K.f(r.get("scale_rev"))
    return (
        r["retained"] == "True"
        and (not np.isfinite(prod) or 0.5 <= prod <= 2.0)
        and not frame_bad(r)
        and np.isfinite(K.f(r["s_common"]))
    )


def check(rows, okey, exclude_flags=None, require_converged=True, bad_nights=True):
    """(object, spectrum id, line) -> dict(status, reason, tags, fields...). The object key pools TARGETIDs.
    exclude_flags: line -> flags that exclude (default EXCLUDE_FLAGS; () for none). A flag study asks for no flag
    exclusion and no bad-night step, which would otherwise hide what a flag does to the no-change pairs."""
    excl = (
        EXCLUDE_FLAGS
        if exclude_flags is None
        else (
            exclude_flags if isinstance(exclude_flags, dict) else {ln: tuple(exclude_flags) for ln in K.LINES}
        )
    )
    fields = K.spectrum_fields(rows)
    out = {}
    for (tid, sid, line), s in fields.items():
        key = (okey(tid), sid, line)
        if key in out and np.isfinite(out[key]["snr"]):
            continue
        tags = []
        if np.isfinite(s["peak_snr"]) and s["peak_snr"] < 5:
            tags.append("weak_peak")
        if s["cls"] == "W":
            tags.append("class_W")
        if s["reduction"] == "daily":
            tags.append("daily")
        if s["instrument"] == "BOSS":
            tags.append("boss")
        if not np.isfinite(s["snr"]):
            st, why = "not used", "line not measured"
        elif s["cls"] in CLASS_REASON:
            st, why = "not used", CLASS_REASON[s["cls"]]
        elif s["cls"] == "B" and s.get("bpk") == "broad":
            st, why = ("good", "") if s["snr"] >= K.SNR_MIN else ("not used", "broad S/N below 8")
            tags.append("very_broad_B")
        elif s["cls"] == "B":
            st, why = (
                "double-peaked",
                "class B: two resolved peaks" if s.get("bpk") == "two" else "class B: double-peaked",
            )
        elif s["snr"] < K.SNR_MIN:
            st, why = "not used", "broad S/N below 8"
        elif s["cls"] in K.GOOD_CLASSES + ("W",):
            st, why = "good", ""
        else:
            st, why = "not used", f"class '{s['cls']}' unknown"
        if st in ("good", "double-peaked"):
            bad = [fl for fl in s.get("fflags", "").split(";") if fl and fl in excl.get(line, ())]
            if require_converged and s.get("fstatus") and s["fstatus"] != "success|True":
                st, why = "not used", "the fit did not converge"
            elif bad:
                st, why = "not used", "blrfit flag " + ", ".join(bad)
        out[key] = dict(
            s, targetid=tid, object=okey(tid), spectrum=sid, line=line, status=st, reason=why, tags=tags
        )
    if not bad_nights:
        return out
    # bad nights: DESI nights within 30 d must agree
    short = defaultdict(list)
    for r in rows:
        if K.pair_kind(r) != "desi-desi short" or not measured(r) or not K.independent(r["pair"]):
            continue
        a, b = (
            out.get((okey(r["targetid"]), r["id_a"], r["line"])),
            out.get((okey(r["targetid"]), r["id_b"], r["line"])),
        )
        if not a or not b or a["status"] != "good" or b["status"] != "good" or K.shape(r) > K.SHAPE_MAX:
            continue
        pull = K.f(r["s_common"]) / np.hypot(K.f(r["err"]), STAGE4_SYS[r["line"]])
        short[(okey(r["targetid"]), r["line"])].append((r["id_a"], r["id_b"], pull))
    for (o, line), prs in short.items():
        agree, disagree = defaultdict(int), defaultdict(list)
        for a, b, p in prs:
            if abs(p) <= 3:
                agree[a] += 1
                agree[b] += 1
            elif abs(p) > 4:
                disagree[a].append(b)
                disagree[b].append(a)
        for n, partners in disagree.items():
            vindicated = [p for p in partners if agree[p] > 0]
            if agree[n] == 0 and vindicated:
                rec = out[(o, n, line)]
                rec.update(status="not used", reason="DESI night inconsistent with the nights within 30 d")
            elif agree[n] == 0:
                out[(o, n, line)]["tags"].append("short_pair_unconfirmed")
    return out


def object_fwhm(spec):
    """(object, line) -> the median broad FWHM (km/s) of the object's good spectra: the width that sets the scale of its
    broad-line velocity jitter (calibrate, verdicts)."""
    v = defaultdict(list)
    for (o, _sid, line), s in spec.items():
        if s["status"] == "good" and np.isfinite(s.get("fwhm", np.nan)) and s["fwhm"] > 0:
            v[(o, line)].append(s["fwhm"])
    return {k: float(np.median(x)) for k, x in v.items()}


def pair_label(r, spec, okey):
    """(status, reason) of one pair row for one line, :
    'measured' (reason 'profiles match' / 'profiles do not match'), 'not measured' (which of the four checks failed),
    'not used' (a spectrum is not good, or the two spectra are not independent)."""
    if not K.independent(r["pair"]):
        return "not used", "the same spectrum or a re-observation on the same plate"
    o, line = okey(r["targetid"]), r["line"]
    for e, name in (("a", "earlier"), ("b", "later")):
        x = spec.get((o, r[f"id_{e}"], line))
        if not x:
            return "not used", f"the {name} spectrum has no fit of this line"
        if x["status"] != "good":
            return (
                "not used",
                f"the {name} spectrum is {x['status'] if x['status'] == 'double-peaked' else 'not used: ' + x['reason']}",
            )
    fl = r["flags"]
    prod = K.f(r.get("scale_fwd")) * K.f(r.get("scale_rev"))
    if "dir_inconsistent" in fl:
        return "not measured", "the two directions disagree"
    if "scale_product_out_of_range" in fl or (np.isfinite(prod) and not 0.5 <= prod <= 2.0):
        return "not measured", "the two brightness scales are not reciprocal"
    if r["retained"] != "True" or not np.isfinite(K.f(r["s_common"])):
        return (
            "not measured",
            "a direction gave no clean minimum (edge of the scan, flat, or brightness scale beyond 0.25-4)",
        )
    if frame_bad(r):
        return "not measured", "the frame does not hold (the narrow lines moved)"
    return "measured", "profiles match" if K.shape(r) <= K.SHAPE_MAX else "profiles do not match"
