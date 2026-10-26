"""Population test: do strong offsets move more often than matched objects without one? One statistic per object, the
Halpha verdict from all its pairs (verdicts.csv): 'moving' (p < 2.7e-3, 3 sigma) and, stricter, 'moving at 4 sigma'
(p < 6.3e-5), among objects whose Halpha could be tested (a motion test ran).
Candidates: coadd |dv(Halpha)| >= 1000 km/s (and the clean class A subset); reference pools: |dv| < 1000 km/s and
|dv| < 300 km/s. Matching: strata of log FWHM (0.2 dex) and log S/N (0.3 dex) of the coadd Halpha, plus the number of
times (2, 3+) and the rest-frame span (< 5, >= 5 yr). The 1:1 matching without replacement is repeated with many seeds;
McNemar's exact test on the discordant pairs of each matching (the median p and the median counts are reported).
Double-peaked coadds are left out of both sides.

    blrfit multiepoch population --run DIR --pairs PAIRS.csv --catalog CATALOG
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from . import common as K


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(verdicts_dir, C, okey=None):
    """cat: the coadd facts of every object key; st: the Halpha statistics of every testable object (double-peaked
    coadds apart). C: the catalog table."""
    okey = K.object_key(C) if okey is None else okey
    cat = {}
    for r in C:
        t = str(int(r["TARGETID"]))
        if okey(t) == t:
            cat[t] = dict(
                dv=f(r["HA_c50_sys"]),
                fwhm=f(r["HA_fwhm"]),
                snr=f(r["HA_broad_flux_snr"]),
                cls=str(r["HA_class"]).strip(),
                clean=K._b(r["CLEAN"]),
                z=f(r["z_in"]),
            )
    vd = {
        r["object"]: r
        for r in csv.DictReader(open(Path(verdicts_dir) / "verdicts.csv"))
        if r["line"] == "Halpha"
    }
    ob = {r["object"]: r for r in csv.DictReader(open(Path(verdicts_dir) / "objects.csv"))}
    st = {}
    for o, v in vd.items():
        if (
            o not in cat
            or ob.get(o, {}).get("combined") == "double-peaked"
            or not np.isfinite(f(v["p_move"]))
        ):
            continue
        c = cat[o]
        st[o] = dict(
            move=f(v["p_move"]) < 2.7e-3,
            strict=f(v["p_move"]) < 6.33e-5,
            times=int(f(v["n_times"])),
            span=f(v["span_yr"]) / (1 + c["z"]),
            verdict=v["verdict"],
        )
    return cat, st


def stratum(cat, st):
    """The matching key: log FWHM (0.2 dex), log S/N (0.3 dex) of the coadd Halpha, times (2, 3+), rest span (< 5, >= 5 yr)."""

    def key(o):
        c, s = cat[o], st[o]
        fw = np.log10(c["fwhm"]) if c["fwhm"] > 0 else np.nan
        sn = np.log10(c["snr"]) if c["snr"] > 0 else np.nan
        return (
            round(fw / 0.2) if np.isfinite(fw) else -1,
            round(sn / 0.3) if np.isfinite(sn) else -1,
            min(s["times"], 3),
            int(s["span"] >= 5),
        )

    return key


def matchings(cand, pool, key, seeds):
    """For each seed, the 1:1 matched (candidate, control) pairs, drawn without replacement within each stratum."""
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        by = defaultdict(list)
        for o in pool:
            by[key(o)].append(o)
        for v in by.values():
            rng.shuffle(v)
        pairs = []
        for o in rng.permutation(cand):
            k = key(o)
            if by[k]:
                pairs.append((o, by[k].pop()))
        out.append(pairs)
    return out


def STRONG(c):
    return np.isfinite(c["dv"]) and abs(c["dv"]) >= 1000


def BELOW_1000(c):
    return np.isfinite(c["dv"]) and abs(c["dv"]) < 1000


def run_population(run, inputs, out=None, seeds=200, verbose=True):
    """The matched comparison on the verdicts of ``run`` (verdicts.csv, objects.csv), written to POPULATION.md in
    ``run`` (or ``out``). Returns the lines of the report."""
    a = _Args(verdicts=str(run), out=str(out or Path(run) / "POPULATION.md"), seeds=seeds)
    C = inputs.load_catalog()
    cat, st = load(a.verdicts, C)
    lines = [
        "# Strong offsets against matched objects — one test per object",
        "",
        f"Per object: the Halpha verdict from all its pairs (`{Path(a.verdicts).name}/verdicts.csv`), among {len(st)} objects whose Halpha "
        "could be tested (double-peaked coadds left out). Matching on coadd log FWHM (0.2 dex), log S/N (0.3 dex), number of times "
        f"(2, 3+) and rest-frame span (< 5, ≥ 5 yr); 1:1 without replacement, {a.seeds} random matchings; McNemar's exact test per matching.",
        "",
    ]

    key = stratum(cat, st)

    for label, csel in (
        ("strong offsets (|Δv| ≥ 1000 km/s)", lambda c: np.isfinite(c["dv"]) and abs(c["dv"]) >= 1000),
        (
            "strong offsets, clean class A",
            lambda c: np.isfinite(c["dv"]) and abs(c["dv"]) >= 1000 and c["cls"] == "A" and c["clean"],
        ),
        ("offsets 300–1000 km/s", lambda c: np.isfinite(c["dv"]) and 300 <= abs(c["dv"]) < 1000),
    ):
        for pname, psel in (
            ("|Δv| < 1000", lambda c: np.isfinite(c["dv"]) and abs(c["dv"]) < 1000),
            ("|Δv| < 300", lambda c: np.isfinite(c["dv"]) and abs(c["dv"]) < 300),
        ):
            if label.startswith("offsets 300") and pname == "|Δv| < 1000":
                continue
            cand = [o for o in st if csel(cat[o])]
            pool = [o for o in st if psel(cat[o]) and o not in cand]
            res = defaultdict(list)
            for pairs in matchings(cand, pool, key, a.seeds):
                for stat in ("move", "strict"):
                    yc = np.array([st[c][stat] for c, _ in pairs], dtype=bool)
                    yr = np.array([st[r][stat] for _, r in pairs], dtype=bool)
                    b, c_ = int(np.sum(yc & ~yr)), int(np.sum(~yc & yr))
                    p = binomtest(b, b + c_, 0.5, alternative="greater").pvalue if b + c_ else np.nan
                    res[stat].append(
                        (
                            len(pairs),
                            b,
                            c_,
                            int(np.sum(yc & yr)),
                            int(np.sum(~yc & ~yr)),
                            p,
                            yc.mean() if len(yc) else np.nan,
                            yr.mean() if len(yr) else np.nan,
                        )
                    )
            fc = np.mean([st[o]["move"] for o in cand]) if cand else np.nan
            fp = np.mean([st[o]["move"] for o in pool]) if pool else np.nan
            lines += [
                f"## {label} against {pname}",
                "",
                f"candidates {len(cand)} (moving {sum(st[o]['move'] for o in cand)}, {100 * fc:.1f}%); pool {len(pool)} (moving "
                f"{sum(st[o]['move'] for o in pool)}, {100 * fp:.1f}%), before matching.",
                "",
            ]
            for stat, name in (("move", "moving (3σ)"), ("strict", "moving at 4σ (stricter)")):
                arr = np.array(res[stat], dtype=float)
                med = np.nanmedian(arr, axis=0)
                lines.append(
                    f"- {name}: matched pairs {med[0]:.0f}; candidate only {med[1]:.0f}, control only {med[2]:.0f}, both {med[3]:.0f}, neither {med[4]:.0f}; "
                    f"fractions {100 * med[6]:.1f}% vs {100 * med[7]:.1f}%; McNemar one-sided p: median {med[5]:.3g} "
                    f"(16–84% over matchings {np.nanpercentile(arr[:, 5], 16):.3g}–{np.nanpercentile(arr[:, 5], 84):.3g})"
                )
            lines.append("")
    lines += [
        "Note: an excess of movers among offset-selected objects is expected from regression toward the mean as well as from binaries "
        "(objects chosen for a large DESI offset partly owe it to variability, so their earlier spectra sit closer to zero); the excess is an "
        "upper limit on the binary signal, not a detection, unless the movers also pass the orbit tests.",
        "",
    ]
    Path(a.out).write_text("\n".join(lines) + "\n")
    if verbose:
        print("\n".join(lines))
    return lines


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)
