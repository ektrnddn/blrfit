"""The verdicts of every object and line of a pair table: the spectrum check, the velocity curves and verdicts from all
good spectra, the line tag of each object, and the false-alarm rate on objects that should not move.

    blrfit multiepoch verdicts --pairs PAIRS.csv --catalog CATALOG --masses MASSES --calibration CALIBRATION.json --out DIR

Writes into ``out``: spectra.csv (every spectrum and line: status, reason, tags), verdicts.csv (every object and line),
objects.csv (both lines together), curves.json (the velocity at each time), pair_table.csv (every pair: status, reason,
change, its error in the full noise model), members.json (object key -> identifiers) and VERDICTS.md (the report).
Two noise models (``calibrate``): the primary one, calibrated on ordinary lines (null), gives the verdicts; the second
(alt_population, default broad: every line with |dv| < 1000 km/s, which wanders more) is run too, and a moving line is
tagged 'moves under both noise models' or 'moves only under the ordinary-line noise'. verdicts.csv also carries
fwhm_ratio (the largest over the smallest median broad FWHM across the times of the curve; moving lines above 1.5 carry
the tag 'width changes'), and too_fast_generous and mass_label, the mass test at the generous limit of ``verdicts``.
"""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from . import common as K
from . import spectra as SP
from . import verdicts as V

VERDICT_COLUMNS = [
    "object", "line", "verdict", "profile", "tags", "p_move", "verdict_alt", "p_move_alt", "fwhm", "fwhm_ratio",
    "blr_kms_5yr", "dof", "n_times", "span_yr", "best_pair", "best_change", "best_sigma", "n_pairs_4sig", "strict",
    "p_steady", "accel", "accel_err", "rests_on", "rests_on_time", "too_fast", "closure_p", "pair_eps",
    "pair_eps_object", "n_triangles", "n_match", "n_mismatch", "pairs not measured",
    "pairs with a spectrum not used", "why", "too_fast_generous", "mass_label",
]  # fmt: skip
COMBINED_ORDER = ["profile variable", "constant", "not measurable"]


def match_clusters(c1, c2):
    """Pairs (k1, k2) of clusters of two lines that share a spectrum."""
    out = []
    for k1, a in c1["clusters"].items():
        for k2, b in c2["clusters"].items():
            if set(a["spectra"]) & set(b["spectra"]):
                out.append((k1, k2))
                break
    return out


def line_agreement(c_move, c_other):
    """p of the other line's curve agreeing with the moving line's, on clusters they share (>= 2), else None."""
    if not c_move or not c_other:
        return None, 0
    m = match_clusters(c_move, c_other)
    if len(m) < 2:
        return None, len(m)
    (a0, b0), rest = m[0], m[1:]
    chi = 0.0
    for a, b in rest:
        d1 = c_move["clusters"][a]["u"] - c_move["clusters"][a0]["u"]
        d2 = c_other["clusters"][b]["u"] - c_other["clusters"][b0]["u"]
        var = sum(
            c["clusters"][k]["err"] ** 2 for c, k in ((c_move, a), (c_move, a0), (c_other, b), (c_other, b0))
        )
        chi += (d1 - d2) ** 2 / max(var, 1.0)
    return float(stats.chi2.sf(chi, len(rest))), len(m)


def combined_verdict(ha, hb, double_peaked, curves, o):
    """The verdict of both lines together, and the line tag of a moving object."""
    mv = [ln for ln, x in (("Halpha", ha), ("Hbeta", hb)) if x["verdict"] == "moving"]
    tag = ""
    if mv:
        if len(mv) == 2:
            p, _n = line_agreement(curves.get(f"{o}|Halpha"), curves.get(f"{o}|Hbeta"))
            tag = "both lines move, " + (
                "agree"
                if p is not None and p >= 0.01
                else ("disagree" if p is not None else "no shared times")
            )
        else:
            other = "Hbeta" if mv[0] == "Halpha" else "Halpha"
            p, _n = line_agreement(curves.get(f"{o}|{mv[0]}"), curves.get(f"{o}|{other}"))
            tag = (
                f"{other} not measured at the same times"
                if p is None
                else (f"{other} agrees within errors" if p >= 0.01 else f"{other} disagrees")
            )
    if double_peaked:
        combined = "double-peaked"
    elif mv:
        combined = "moving: " + " and ".join("Hα" if ln == "Halpha" else "Hβ" for ln in mv)
    else:
        combined = min(
            (ha["verdict"], hb["verdict"]),
            key=lambda v: COMBINED_ORDER.index(v) if v in COMBINED_ORDER else 9,
        )
    return combined, tag


def _vc2(res, line):
    return sum(1 for k, r in res.items() if k[1] == line and r.get("verdict_alt") == "moving")


def run_verdicts(inputs, calibration, out, alt_population="broad", verbose=True):
    """The whole verdict stage on ``inputs`` (``common.Inputs``) with the noise model of ``calibration``
    (CALIBRATION.json of ``run_calibration``), written into ``out``. Returns the verdict records,
    (object, line) -> dict."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (
        Path(calibration).resolve() != (out / "CALIBRATION.json").resolve()
    ):  # the run directory is self-contained
        shutil.copy(calibration, out / "CALIBRATION.json")
    rows = inputs.load_pairs()
    C = inputs.load_catalog()
    okey = K.object_key(C)
    masses = inputs.load_masses()
    calib = V.Calibration(calibration)
    calib_alt = V.Calibration(calibration, alt_population) if alt_population else None
    pops = K.populations(C, okey)
    zof = K.redshifts(C)
    spec = SP.check(rows, okey)
    fwhm = SP.object_fwhm(spec)
    with open(out / "spectra.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "targetid", "spectrum", "line", "instrument", "reduction", "mjd", "class", "snr", "peak_snr", "status", "reason", "tags"])  # fmt: skip
        for (o, sid, line), s in sorted(spec.items()):
            w.writerow([o, s["targetid"], sid, line, s["instrument"], s["reduction"], f"{s['mjd']:.3f}", s["cls"], f"{s['snr']:.1f}", f"{s['peak_snr']:.1f}", s["status"], s["reason"], ";".join(s["tags"])])  # fmt: skip
    byobj = K.by_object(rows, okey)
    members = K.members_of(rows, okey)
    res, curves, pair_rows = {}, {}, []
    for o, rs in sorted(byobj.items()):
        mass = K.mass_of(o, masses, members)
        z = K.redshift_of(o, zof, members, rs)
        for line in K.LINES:
            keep = {}
            r, c = V.analyze(o, line, rs, spec, calib, mass, z, keep=keep, fwhm=fwhm)
            full = {}
            for p_ in keep.get("recs", []):
                full[p_["pair"]] = keep["pair_sigma"](
                    dict(p_, floor=keep["eps"]) if p_["prof"] == "match" else p_
                )
            seen_p = set()
            for row in rs:
                if row["line"] != line or row["pair"] in seen_p:
                    continue
                seen_p.add(row["pair"])
                st_, why_ = SP.pair_label(row, spec, okey)
                d_, e_ = K.f(row["s_common"]), K.f(row["err"])
                ef = full.get(row["pair"], np.nan)
                sh = K.shape(row)
                pair_rows.append(
                    dict(
                        object=o, line=line, pair=row["pair"], id_a=row["id_a"], id_b=row["id_b"],
                        mjd_a=f"{K.f(row['mjd_a']):.3f}", mjd_b=f"{K.f(row['mjd_b']):.3f}", kind=K.pair_kind(row),
                        dt_rest_yr=f"{K.f(row['dt_rest_yr']):.3f}", status=st_, reason=why_,
                        change=f"{d_:.1f}" if np.isfinite(d_) else "", err_stat=f"{e_:.1f}" if np.isfinite(e_) else "",
                        err_full=f"{ef:.1f}" if np.isfinite(ef) else "",
                        nsigma=f"{abs(d_) / ef:.2f}" if np.isfinite(ef) and np.isfinite(d_) and ef > 0 else "",
                        shape=f"{sh:.2f}" if np.isfinite(sh) else "", s_fwd=row["s_fwd"][:8], s_rev=row["s_rev"][:8],
                        dv_narrow=row["dv_narrow"][:7],
                    )
                )  # fmt: skip
            if calib_alt is not None:
                ra, _ = V.analyze(o, line, rs, spec, calib_alt, mass, z, fwhm=fwhm)
                r.update(verdict_alt=ra["verdict"], p_move_alt=ra.get("p_move", np.nan))
                if r["verdict"] == "moving":
                    tg = (
                        "moves under both noise models"
                        if ra["verdict"] == "moving"
                        else "moves only under the ordinary-line noise"
                    )
                    r["tags"] = ";".join([t for t in r.get("tags", "").split(";") if t] + [tg])
            res[(o, line)] = r
            if c:
                curves[f"{o}|{line}"] = c
    with open(out / "verdicts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=VERDICT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for k in sorted(res):
            w.writerow({kk: res[k].get(kk, "") for kk in VERDICT_COLUMNS})
    json.dump(curves, open(out / "curves.json", "w"), indent=0)
    json.dump({o: sorted(m) for o, m in members.items()}, open(out / "members.json", "w"))
    with open(out / "pair_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    # objects: the two lines together; the double-peaked group is set aside (two resolved peaks in the coadd)
    dp = K.double_peaked_objects(C, okey, inputs.facts, inputs.all_fits)
    objs = []
    for o in sorted(byobj):
        ha, hb = res[(o, "Halpha")], res[(o, "Hbeta")]
        combined, tag = combined_verdict(ha, hb, o in dp, curves, o)
        objs.append(
            dict(
                object=o, combined=combined, line_tag=tag, halpha=ha["verdict"], hbeta=hb["verdict"],
                halpha_tags=ha.get("tags", ""), hbeta_tags=hb.get("tags", ""), null=o in pops["null"],
            )
        )  # fmt: skip
    with open(out / "objects.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(objs[0].keys()))
        w.writeheader()
        w.writerows(objs)
    report = _report(out, inputs, calib, calib_alt, spec, res, objs, C, okey, dp)
    (out / "VERDICTS.md").write_text("\n".join(report) + "\n")
    if verbose:
        print("\n".join(report[:60]))
        print("->", out / "VERDICTS.md")
    return res


def _report(out, inputs, calib, calib_alt, spec, res, objs, C, okey, dp):
    coadd_b = {x["object"]: x["object"] in dp for x in objs}
    L = [
        "# Multi epoch verdicts — " + str(out.name), "",
        f"`blrfit multiepoch verdicts` on `{Path(inputs.pairs).name}` (primary noise model: {calib.population}"
        + (f"; second: {calib_alt.population}" if calib_alt else "") + ").", "",
    ]  # fmt: skip
    st = Counter((s["line"], s["status"]) for s in spec.values())
    why = Counter((s["line"], s["reason"]) for s in spec.values() if s["status"] == "not used")
    L += ["## Spectrum check", "", "| line | good | double-peaked | not used |", "|---|---|---|---|"]
    for line in K.LINES:
        L.append(
            f"| {line} | {st[(line, 'good')]} | {st[(line, 'double-peaked')]} | {st[(line, 'not used')]} |"
        )
    L += ["", "Not used, by reason: " + "; ".join(f"{ln} {r}: {n}" for (ln, r), n in sorted(why.items())), ""]
    L += ["## Verdicts per line (objects)", "", "| verdict | Hα | Hβ |", "|---|---|---|"]
    vc = Counter((k[1], r["verdict"]) for k, r in res.items())
    for v in ("moving", "profile variable", "constant", "not measurable"):
        L.append(f"| {v} | {vc[('Halpha', v)]} | {vc[('Hbeta', v)]} |")
    if calib_alt is not None:
        both = {
            ln: sum(
                1
                for k, r in res.items()
                if k[1] == ln and r["verdict"] == "moving" and r.get("verdict_alt") == "moving"
            )
            for ln in K.LINES
        }
        L += ["", f"Second noise model ({calib_alt.population}): moving Hα {_vc2(res, 'Halpha')}, Hβ {_vc2(res, 'Hbeta')}; "
              f"of the lines moving under the primary model, moving under both: Hα {both['Halpha']}, Hβ {both['Hbeta']}."]  # fmt: skip
    L += [
        "",
        "## Objects, both lines (double-peaked objects apart)",
        "",
        "| combined | objects |",
        "|---|---|",
    ]
    for k, n in Counter(x["combined"] for x in objs).most_common():
        L.append(f"| {k} | {n} |")
    L += ["", "Line tags of moving objects: " + "; ".join(f"{k}: {n}" for k, n in Counter(x["line_tag"] for x in objs if x["line_tag"]).most_common()), ""]  # fmt: skip
    tg = Counter()
    for (_o, line), r in res.items():
        if r["verdict"] == "moving":
            for t in r.get("tags", "").split(";"):
                if t:
                    tg[(line, t)] += 1
    L += ["Tags of moving lines: " + "; ".join(f"{ln} {t}: {n}" for (ln, t), n in sorted(tg.items())), ""]
    L += ["## Width changes across the times", "",
          "The ratio of the largest to the smallest median broad FWHM over the times of each line's curve (`fwhm_ratio`); a moving "
          f"line above {V.WIDTH_RATIO_MAX} carries the tag `width changes` (a tag, not a cut). Lines of double-peaked objects left out.", "",
          "| verdict | lines with a ratio | median ratio | > 1.5 | > 2 | > 3 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for v in ("moving", "constant"):
        rat = np.array([r["fwhm_ratio"] for (o, line), r in res.items() if r["verdict"] == v and not coadd_b.get(o) and np.isfinite(r.get("fwhm_ratio", np.nan))])  # fmt: skip
        if len(rat):
            L.append(
                f"| {v} | {len(rat)} | {np.median(rat):.2f} | {np.mean(rat > 1.5):.1%} | {np.mean(rat > 2):.1%} | {np.mean(rat > 3):.1%} |"
            )
    L.append("")
    L += ["## The mass test at the generous limit", "",
          f"The tag and `too_fast` keep the nominal mass; `too_fast_generous` and `mass_label` repeat the test with the BLR radius halved and the "
          f"virial mass doubled together (P_min x {V.GENEROUS_PMIN:.2f}, v_max x {V.GENEROUS_VMAX:.0f}). Moving lines by label (double-peaked objects left out):", "",
          "| mass label | Hα | Hβ | with 3 or more times: Hα | Hβ |", "|---|---|---|---|---|"]  # fmt: skip
    for lab in V.MASS_LABELS + ("no mass",):
        cells = []
        for min_t in (0, 3):
            for line in K.LINES:
                cells.append(sum(1 for (o, ln), r in res.items() if ln == line and r["verdict"] == "moving" and not coadd_b.get(o)
                                 and r.get("mass_label") == lab and (r.get("n_times") or 0) >= min_t))  # fmt: skip
        L.append(f"| {lab} | " + " | ".join(str(c) for c in cells) + " |")
    L.append("")
    nul = [x for x in objs if x["null"]]
    L += ["## False alarms on objects that should not move", "",
          f"Null objects (coadd |Δv(Hα)| < 300 km/s, class A/F, clean) with pairs: {len(nul)}; moving in Hα: {sum(x['halpha'] == 'moving' for x in nul)}, "
          f"in Hβ: {sum(x['hbeta'] == 'moving' for x in nul)}. (The null objects calibrated the noise model, so this is a consistency check, not an "
          "independent test.)", ""]  # fmt: skip
    dvo = {}
    col = C["HA_c50_sys"]
    col = np.asarray(col.filled(np.nan) if hasattr(col, "filled") else col, float)
    for t, d in zip(C["TARGETID"], col):
        dvo.setdefault(okey(str(int(t))), d)

    def band(d):
        if not np.isfinite(d):
            return "no coadd Hα"
        return "|Δv| ≥ 1000 (strong)" if abs(d) >= 1000 else ("300–1000" if abs(d) >= 300 else "< 300")

    L += ["## Moving fraction by coadd offset (Hα, testable objects, double-peaked objects apart)", "", "| coadd Δv(Hα) | testable | moving | fraction moving |", "|---|---|---|---|"]  # fmt: skip
    tb, mv_ = Counter(), Counter()
    for x in objs:
        if x["combined"] == "double-peaked" or x["halpha"] == "not measurable":
            continue
        b = band(dvo.get(x["object"], np.nan))
        tb[b] += 1
        mv_[b] += x["halpha"] == "moving"
    for b in ("< 300", "300–1000", "|Δv| ≥ 1000 (strong)", "no coadd Hα"):
        if tb[b]:
            L.append(f"| {b} | {tb[b]} | {mv_[b]} | {mv_[b] / tb[b]:.1%} |")
    L.append("")
    return L
