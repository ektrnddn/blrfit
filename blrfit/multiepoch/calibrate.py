#!/usr/bin/env python
"""The velocity errors of every kind of pair, calibrated on objects that should not move.

Every object counts once in every statistic (weights 1 / its number of pairs in the cell; plan P1b), so an object with
many spectra cannot set a floor. The pairs are the ones the verdicts use: the four checks passed (spectra.measured),
both spectra good in the spectrum check (spectra.check) and of class A/C/F. The populations: 'null' (coadd
|dv(Halpha)| < 300 km/s, class A/F, clean) and 'broad' (|dv(Halpha)| < 1000 km/s). Errors: bootstrap over objects.

1. The table, per kind of pair (DESI nights <= 30 d apart, DESI nights farther apart, SDSS-I/II to DESI, BOSS to DESI,
   SDSS to SDSS, SDSS to BOSS, BOSS to BOSS), line and profile (matching, shape <= 0.5, or not): the offset (median
   change) and the floor (added in quadrature to the statistical error so that the pulls have NMAD 1). The adopted
   floors (null where it has >= MIN_N pairs, else broad; DESI nights <= 30 d keep the calibrated terms of ``blrfit.pairs``,
   45 / 36 km/s) serve the display and the pair tags; the curve fits carry their own model (part 2).
2. The noise model of the curve fits (``verdicts``), per population and line, on matching profiles:
     tau     each spectrum's own jitter, by instrument (km/s): DESI from DESI nights <= 30 d apart (floor / sqrt 2), SDSS
             and BOSS from their same-instrument pairs (with the time term below included);
     BLR     the velocity of a broad line wanders with time: between two spectra Delta t apart (rest frame) it adds
             SF^2 = 2 (kappa F)^2 [1 - exp(-Delta t / tau_c)], F the object's broad FWHM (a damped random walk in units of
             the line width). The data ask for both the width and the gap: the extra scatter of DESI-DESI pairs grows from
             ~0 at <= 30 d to ~0.03 FWHM at 3 yr, and broader lines scatter more in km/s but alike in units of FWHM
             kappa and tau_c make the object-weighted pull NMAD 1 in every rest-frame gap bin
             (GAP_EDGES; objective sum over bins of n_objects (ln NMAD)^2), all kinds of pairs together, with no
             instrument offset;
     omega   what SDSS-to-DESI (BOSS-to-DESI) pairs still need beyond the jitters and the BLR term: one object's SDSS
             (BOSS) spectra offset against its DESI ones;
     eps     the independent per-pair term from triangle closures.
   'components' holds the primary population (PRIMARY), 'components_by_population' both.

    blrfit multiepoch calibrate --pairs PAIRS.csv --catalog CATALOG --out DIR

exclude_objects: object keys left out of every population and of the loop term, so that a calibration can be made on
part of the null objects and tested on the rest (a split-half false-alarm check).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import common as K
from . import spectra as SP

MIN_N = 25
STAGE4_SYS = {"Halpha": 45.0, "Hbeta": 36.0}
KINDS = ("desi-desi short", "desi-desi", "sdss-desi", "boss-desi", "sdss-sdss", "sdss-boss", "boss-boss")
PRIMARY = "null"  # the population of the primary noise model (plan decision)
GAP_EDGES = (
    0.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
)  # rest-frame days: the bins of the BLR-term fit
TAUC_GRID = np.geomspace(30.0, 1.0e5, 48)  # rest-frame days; the top end acts as a random walk over 25 yr
KAPPA_MAX = 0.25


def wquantile(x, w, q):
    """Weighted quantile (midpoint rule)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, c, x))


def wnmad(x, w):
    m = wquantile(x, w, 0.5)
    return 1.4826 * wquantile(np.abs(np.asarray(x, float) - m), w, 0.5)


def object_weights(objs):
    """1 / (number of pairs of the object in this set): every object counts once, however many pairs it has (an object
    with 64 BOSS spectra has 2,016 BOSS-BOSS pairs among all pairs)."""
    from collections import Counter

    n = Counter(objs)
    return np.array([1.0 / n[o] for o in objs])


def solve_floor(dv, err, offset, w=None):
    """The floor (km/s) that makes the object-weighted NMAD of (dv - offset) / hypot(err, floor) equal 1; 0 when it is
    already <= 1."""
    w = np.ones_like(dv) if w is None else w

    def pull(fl):
        return wnmad((dv - offset) / np.hypot(err, fl), w)

    if pull(0.0) <= 1.0:
        return 0.0
    lo, hi = 0.0, 5000.0
    if pull(hi) > 1.0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pull(mid) > 1.0 else (lo, mid)
    return 0.5 * (lo + hi)


def stats(recs, rng, nboot=300):
    """recs: list of (object, change, statistical error, err_total). Offset, floor, their bootstrap 16-84%, NMAD and
    4-sigma tail fraction of the pulls with the Stage 5 errors (err_total) and with the calibrated ones."""
    dv = np.array([r[1] for r in recs])
    err = np.array([r[2] for r in recs])
    etot = np.array([r[3] for r in recs])
    ob = [r[0] for r in recs]
    w = object_weights(ob)
    off = wquantile(dv, w, 0.5)
    fl = solve_floor(dv, err, off, w)
    objs = sorted(set(ob))
    idx = defaultdict(list)
    for i, r in enumerate(recs):
        idx[r[0]].append(i)
    boff, bfl = [], []
    for _ in range(nboot):
        pick = rng.choice(len(objs), len(objs), replace=True)
        ii = np.concatenate([idx[objs[j]] for j in pick])
        wb = np.concatenate([np.full(len(idx[objs[j]]), 1.0 / len(idx[objs[j]])) for j in pick])
        o = wquantile(dv[ii], wb, 0.5)
        boff.append(o)
        bfl.append(solve_floor(dv[ii], err[ii], o, wb))
    p_old = (dv - off) / etot
    p_new = (dv - off) / np.hypot(err, fl) if np.isfinite(fl) else np.full_like(dv, np.nan)

    def q(a):
        return [float(np.nanpercentile(a, 16)), float(np.nanpercentile(a, 84))]

    def wt(m):
        return float(np.sum(w * m) / np.sum(w))

    return dict(
        n=len(recs),
        objects=len(objs),
        offset=round(off, 1),
        offset_16_84=[round(v, 1) for v in q(boff)],
        floor=round(fl, 1) if np.isfinite(fl) else None,
        floor_16_84=[round(v, 1) for v in q(bfl)],
        median_stat_err=round(wquantile(err, w, 0.5), 1),
        nmad_stage5=round(wnmad(p_old, w), 2),
        tail4_stage5=round(wt(np.abs(p_old) > 4), 3),
        nmad_calibrated=round(wnmad(p_new, w), 2) if np.isfinite(fl) else np.nan,
        tail4_calibrated=round(wt(np.abs(p_new) > 4), 3),
        median_change_kms=round(off, 1),
        nmad_change_kms=round(wnmad(dv, w), 1),
    )


def usable_pairs(rows, spec, okey):
    """The pairs the calibration uses, once each: measured, both spectra good and class A/C/F, independent."""
    seen = set()
    for r in rows:
        o, line = okey(r["targetid"]), r["line"]
        if (o, r["pair"], line) in seen or not K.independent(r["pair"]):
            continue
        seen.add((o, r["pair"], line))
        sa, sb = spec.get((o, r["id_a"], line)), spec.get((o, r["id_b"], line))
        if (
            not sa
            or not sb
            or sa["status"] != "good"
            or sb["status"] != "good"
            or sa["cls"] not in K.GOOD_CLASSES
            or sb["cls"] not in K.GOOD_CLASSES
        ):
            continue
        if not SP.measured(r):
            continue
        dv, err, sh = K.f(r["s_common"]), K.f(r["err"]), K.shape(r)
        if np.isfinite(dv) and err > 0 and np.isfinite(sh):
            yield o, r, sa, sb, dv, err, sh


def collect(rows, spec, pops, okey):
    """(population, kind, line, profile) -> list of (object, change, err, err_total)."""
    out = defaultdict(list)
    for o, r, _sa, _sb, dv, err, sh in usable_pairs(rows, spec, okey):
        prof = "match" if sh <= K.SHAPE_MAX else "mismatch"
        for pop, objs in pops.items():
            if o in objs:
                out[(pop, K.pair_kind(r), r["line"], prof)].append((o, dv, err, K.f(r["err_total"])))
    return out


def jitter_records(rows, spec, pops, okey, fwhm):
    """population -> line -> list of dict(o, kind, dv, err, dt (rest days), F, ia, ib) of matching pairs."""
    out = defaultdict(lambda: defaultdict(list))
    for o, r, sa, sb, dv, err, sh in usable_pairs(rows, spec, okey):
        F = fwhm.get((o, r["line"]))
        if sh > K.SHAPE_MAX or not F:
            continue
        rec = dict(
            o=o,
            kind=K.pair_kind(r),
            dv=dv,
            err=err,
            dt=abs(K.f(r["dt_rest_yr"])) * 365.25,
            F=F,
            ia=sa["instrument"],
            ib=sb["instrument"],
        )
        for pop, objs in pops.items():
            if o in objs:
                out[pop][r["line"]].append(rec)
    return out


def sf2(dt_days, kappa, tau_c):
    """The BLR term between two spectra, in units of FWHM^2."""
    return 2.0 * kappa**2 * (1.0 - np.exp(-np.asarray(dt_days, float) / tau_c))


class JitterData:
    """Arrays of one population and line for the BLR-term fit: changes and errors in units of the object's FWHM, the
    per-kind offsets removed, the gap bins with their object weights."""

    def __init__(self, D, tau, keep=None):
        D = [d for d in D if keep is None or d["o"] in keep]
        self.D = D
        self.F = np.array([d["F"] for d in D])
        self.dt = np.array([d["dt"] for d in D])
        self.ob = np.array([d["o"] for d in D])
        kinds = np.array([d["kind"] for d in D])
        dv = np.array([d["dv"] for d in D])
        off = np.zeros_like(dv)
        for k in set(kinds):
            m = kinds == k
            off[m] = wquantile(dv[m], object_weights(list(self.ob[m])), 0.5)
        self.x = (dv - off) / self.F
        self.e2 = np.array([d["err"] ** 2 + tau[d["ia"]] ** 2 + tau[d["ib"]] ** 2 for d in D]) / self.F**2
        b = np.digitize(self.dt, GAP_EDGES[1:-1])
        self.bins = []
        for bi in range(len(GAP_EDGES) - 1):
            ii = np.where(b == bi)[0]
            if len(ii) >= 10:
                self.bins.append((bi, ii, object_weights(list(self.ob[ii])), len(set(self.ob[ii]))))

    def objective(self, kappa, tau_c):
        p = self.x / np.sqrt(self.e2 + sf2(self.dt, kappa, tau_c))
        return sum(n * np.log(max(wnmad(p[ii], w), 1e-6)) ** 2 for _, ii, w, n in self.bins)

    def fit(self):
        """(kappa, tau_c): for each tau_c on the grid, kappa by golden section; the best pair."""
        best = (np.inf, 0.0, TAUC_GRID[-1])
        g = (np.sqrt(5) - 1) / 2
        for tc in TAUC_GRID:
            a, b = 0.0, KAPPA_MAX
            c, d = b - g * (b - a), a + g * (b - a)
            fc, fd = self.objective(c, tc), self.objective(d, tc)
            for _ in range(28):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - g * (b - a)
                    fc = self.objective(c, tc)
                else:
                    a, c, fc = c, d, fd
                    d = a + g * (b - a)
                    fd = self.objective(d, tc)
            k = 0.5 * (a + b)
            Jk, J0 = self.objective(k, tc), self.objective(0.0, tc)
            k, J = (k, Jk) if Jk <= J0 else (0.0, J0)
            if J < best[0] - 1e-12:
                best = (J, k, tc)
        return best[1], best[2], best[0]

    def bin_table(self, kappa, tau_c, rng, nboot=200):
        """Per gap bin: the measured BLR term (sqrt SF^2 in FWHM units that makes the pulls NMAD 1 with the jitters
        included) with its bootstrap 16-84%, the model value, and the NMAD of the pulls without and with the model."""
        out = []
        for bi, ii, w, n in self.bins:
            x, e = self.x[ii], np.sqrt(self.e2[ii])
            k = solve_floor(x, e, 0.0, w)
            objs = sorted(set(self.ob[ii]))
            oi = defaultdict(list)
            for j, o in enumerate(self.ob[ii]):
                oi[o].append(j)
            bk = []
            for _ in range(nboot):
                pk = rng.integers(0, len(objs), len(objs))
                jj = np.concatenate([oi[objs[m]] for m in pk])
                wb = np.concatenate([np.full(len(oi[objs[m]]), 1.0 / len(oi[objs[m]])) for m in pk])
                bk.append(solve_floor(x[jj], e[jj], 0.0, wb))
            dtm = float(wquantile(self.dt[ii], w, 0.5))
            pm = x / np.sqrt(e**2 + sf2(self.dt[ii], kappa, tau_c))
            out.append(
                dict(
                    gap_lo=GAP_EDGES[bi],
                    gap_hi=GAP_EDGES[bi + 1],
                    median_gap_days=round(dtm, 1),
                    pairs=len(ii),
                    objects=n,
                    measured=round(float(k), 4) if np.isfinite(k) else None,
                    measured_16_84=[
                        round(float(np.nanpercentile(bk, 16)), 4),
                        round(float(np.nanpercentile(bk, 84)), 4),
                    ],
                    model=round(float(np.sqrt(sf2(dtm, kappa, tau_c))), 4),
                    nmad_without=round(wnmad(x / e, w), 2),
                    nmad_with=round(wnmad(pm, w), 2),
                    tail4_with=round(float(np.sum(w * (np.abs(pm) > 4)) / np.sum(w)), 3),
                )
            )
        return out


def same_instrument_tau(D, kind, kappa, tau_c):
    """The per-spectrum jitter of one instrument from its same-instrument pairs, the BLR term included: floor / sqrt 2."""
    S = [d for d in D if d["kind"] == kind]
    if len(S) < 5:
        return None, len(S)
    ob = [d["o"] for d in S]
    w = object_weights(ob)
    dv = np.array([d["dv"] for d in S])
    F = np.array([d["F"] for d in S])
    dt = np.array([d["dt"] for d in S])
    err = np.sqrt(np.array([d["err"] for d in S]) ** 2 + F**2 * sf2(dt, kappa, tau_c))
    fl = solve_floor(dv, err, wquantile(dv, w, 0.5), w)
    return (fl / np.sqrt(2) if np.isfinite(fl) else None), len(S)


def cross_omega(D, kind, tau, kappa, tau_c):
    """What one kind of cross-instrument pair needs beyond the jitters and the BLR term (km/s), and its pair count."""
    S = [d for d in D if d["kind"] == kind]
    if len(S) < 5:
        return None, len(S)
    ob = [d["o"] for d in S]
    w = object_weights(ob)
    dv = np.array([d["dv"] for d in S])
    F = np.array([d["F"] for d in S])
    dt = np.array([d["dt"] for d in S])
    err = np.sqrt(
        np.array([d["err"] ** 2 + tau[d["ia"]] ** 2 + tau[d["ib"]] ** 2 for d in S])
        + F**2 * sf2(dt, kappa, tau_c)
    )
    om = solve_floor(dv, err, wquantile(dv, w, 0.5), w)
    return (float(om) if np.isfinite(om) else None), len(S)


def fit_population(D, Dbroad, line, rng, nboot=60):
    """D: the jitter records of one population and line; Dbroad: those of the broad population (fallback for kinds with
    fewer than MIN_N pairs). Returns the components dict."""

    def pick(kind):
        return D if sum(d["kind"] == kind for d in D) >= MIN_N else Dbroad

    # DESI's per-spectrum jitter from nights <= 30 d apart (no BLR term at these gaps)
    S = pick("desi-desi short")
    sh = [d for d in S if d["kind"] == "desi-desi short"]
    ob = [d["o"] for d in sh]
    w = object_weights(ob)
    dv = np.array([d["dv"] for d in sh])
    er = np.array([d["err"] for d in sh])
    fl_s = solve_floor(dv, er, wquantile(dv, w, 0.5), w)
    tau = dict(DESI=fl_s / np.sqrt(2), SDSS=0.0, BOSS=0.0)
    src = dict(DESI="own" if S is D else "broad")
    kappa, tau_c = 0.0, TAUC_GRID[-1]
    for _it in range(2):  # SDSS / BOSS jitters, then the BLR term; once more with the BLR term in the jitters
        for inst, kind in (("SDSS", "sdss-sdss"), ("BOSS", "boss-boss")):
            Dk = pick(kind)
            t, n = same_instrument_tau(Dk, kind, kappa, tau_c)
            tau[inst] = t if t is not None else tau["DESI"]
            src[inst] = ("own" if Dk is D else "broad") if t is not None else "DESI value (too few pairs)"
        JD = JitterData(D, tau)
        kappa, tau_c, Jmin = JD.fit()
    om = {}
    for inst, kind in (("SDSS", "sdss-desi"), ("BOSS", "boss-desi")):
        Dk = pick(kind)
        v, n = cross_omega(Dk, kind, tau, kappa, tau_c)
        om[inst] = round(v, 1) if v is not None else 0.0
        src[f"omega_{inst}"] = "own" if Dk is D else "broad"
    # bootstrap over objects of the BLR term (the jitters held)
    g = defaultdict(list)
    for d in D:
        g[d["o"]].append(d)
    objs = sorted(g)
    bk, bt = [], []
    for _ in range(nboot):
        pk = rng.integers(0, len(objs), len(objs))
        Db = [dict(d, o=f"{d['o']}#{n_}") for n_, m in enumerate(pk) for d in g[objs[m]]]
        k_, t_, _ = JitterData(Db, tau).fit()
        bk.append(k_)
        bt.append(t_)

    def at(yrs):
        return [round(float(np.sqrt(sf2(yrs * 365.25, k_, t_))), 4) for k_, t_ in zip(bk, bt)]

    def q(a):
        return [round(float(np.percentile(a, 16)), 4), round(float(np.percentile(a, 84)), 4)]

    Fmed = float(np.median([g[o][0]["F"] for o in objs]))  # the median object's FWHM
    return dict(
        tau={k: round(float(v), 1) for k, v in tau.items()},
        omega=om,
        sources=src,
        blr=dict(
            kappa=round(float(kappa), 4),
            tau_c_days=round(float(tau_c), 1),
            objective=round(float(Jmin), 3),
            kappa_16_84=q(bk),
            tau_c_days_16_84=[round(float(np.percentile(bt, 16)), 1), round(float(np.percentile(bt, 84)), 1)],
            sf_fwhm_at={
                f"{y} yr": round(float(np.sqrt(sf2(y * 365.25, kappa, tau_c))), 4) for y in (0.1, 1, 3, 5, 15)
            },
            sf_fwhm_at_16_84={f"{y} yr": q(at(y)) for y in (1, 5, 15)},
            median_fwhm_kms=round(Fmed, 0),
            sf_kms_at_median_fwhm={
                f"{y} yr": round(float(Fmed * np.sqrt(sf2(y * 365.25, kappa, tau_c))), 1) for y in (1, 5, 15)
            },
            bins=JitterData(D, tau).bin_table(kappa, tau_c, rng),
        ),
        pairs=len(D),
        objects=len(objs),
    )


def closure_eps(rows, spec, okey, line, exclude=()):
    """The population's independent per-pair term from triangle closures (verdicts.triangles): per object with three or
    more triangles of measured, matching pairs between good spectra, 1.4826 x median |c| / sqrt(3); the median over
    objects (each object once). Objects whose loops do not close get their own, larger value in verdicts.analyze."""
    from . import verdicts as Vd

    by = defaultdict(list)
    for r in rows:
        if (
            r["line"] != line
            or not SP.measured(r)
            or not K.independent(r["pair"])
            or K.shape(r) > K.SHAPE_MAX
        ):
            continue
        o = okey(r["targetid"])
        if o in exclude:
            continue
        sa, sb = spec.get((o, r["id_a"], line)), spec.get((o, r["id_b"], line))
        if not sa or not sb or sa["status"] != "good" or sb["status"] != "good":
            continue
        d = K.f(r["s_common"])
        if np.isfinite(d):
            by[o].append(dict(a=r["id_a"], b=r["id_b"], d=d))
    e = []
    for _o, prs in by.items():
        seen, uniq = set(), []
        for p in prs:
            if (p["a"], p["b"]) not in seen:
                seen.add((p["a"], p["b"]))
                uniq.append(p)
        c = Vd.triangles(uniq)
        if len(c) >= 3:
            e.append(1.4826 * np.median(np.abs(c)) / np.sqrt(3.0))
    return dict(
        pair_eps=round(float(np.median(e)), 1) if e else Vd.PAIR_EPS,
        pair_eps_objects=len(e),
        pair_eps_16_84=[round(float(np.percentile(e, 16)), 1), round(float(np.percentile(e, 84)), 1)]
        if e
        else None,
    )


def run_calibration(inputs, out, nboot=60, exclude_objects=None, seed=20261014, verbose=True):
    """Calibrate the noise model on the pairs of ``inputs`` (``common.Inputs``) and write CALIBRATION.json and
    CALIBRATION.md into ``out``. exclude_objects: object keys left out of every population (a sequence, or a file with
    one key per line). Returns the calibration dict."""
    a = _Args(out=out, nboot=nboot, exclude_objects=exclude_objects)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = inputs.load_pairs()
    C = inputs.load_catalog()
    okey = K.object_key(C)
    pops = K.populations(C, okey)
    excl = set()
    if a.exclude_objects:
        if isinstance(a.exclude_objects, (str, Path)):
            excl = {ln.strip() for ln in open(a.exclude_objects) if ln.strip()}
        else:
            excl = {str(x) for x in a.exclude_objects}
        pops = {k: v - excl for k, v in pops.items()}
        print(
            f"excluding {len(excl)} objects from every population: null {len(pops['null'])}, broad {len(pops['broad'])} objects left",
            flush=True,
        )
    spec = SP.check(rows, okey)
    fwhm = SP.object_fwhm(spec)
    recs = collect(rows, spec, pops, okey)
    table = {}
    for (pop, kind, line, prof), v in sorted(recs.items()):
        if len(v) >= 5:
            table[f"{pop}|{kind}|{line}|{prof}"] = stats(v, rng)
    adopted = {}
    for kind in KINDS:
        for line in K.LINES:
            for prof in ("match", "mismatch"):
                s_null, s_broad = (
                    table.get(f"null|{kind}|{line}|{prof}"),
                    table.get(f"broad|{kind}|{line}|{prof}"),
                )
                src, s = (
                    ("null", s_null)
                    if s_null and s_null["n"] >= MIN_N
                    else (("broad", s_broad) if s_broad else (None, None))
                )
                if s is None:
                    continue
                rec = dict(
                    source=src,
                    n=s["n"],
                    offset=s["offset"],
                    floor=s["floor"],
                    offset_16_84=s["offset_16_84"],
                    floor_16_84=s["floor_16_84"],
                )
                if kind == "desi-desi short":  # the Stage 4 terms stay; the solver's value is a check
                    rec.update(
                        offset=0.0, floor=STAGE4_SYS[line], solver_floor=s["floor"], solver_offset=s["offset"]
                    )
                adopted[f"{kind}|{line}|{prof}"] = rec
    # the noise model of the curve fits, per population (docstring, part 2)
    JR = jitter_records(rows, spec, pops, okey, fwhm)
    by_pop = {}
    for pop in ("null", "broad"):
        by_pop[pop] = {}
        for line in K.LINES:
            comp = fit_population(JR[pop][line], JR["broad"][line], line, rng, a.nboot)

            def fl(kind, line=line):
                return (adopted.get(f"{kind}|{line}|match") or {}).get("floor")

            om, tau = comp["omega"], comp["tau"]
            comp["check_sdss_boss"] = dict(
                predicted_without_blr=round(
                    float(np.sqrt(tau["SDSS"] ** 2 + tau["BOSS"] ** 2 + om["SDSS"] ** 2 + om["BOSS"] ** 2)), 1
                ),
                measured=fl("sdss-boss"),
            )
            comp.update(closure_eps(rows, spec, okey, line, excl))
            by_pop[pop][line] = comp
            print(
                f"{pop} {line}: tau {comp['tau']}, omega {comp['omega']}, BLR kappa {comp['blr']['kappa']} tau_c {comp['blr']['tau_c_days']} d, "
                f"SF/FWHM {comp['blr']['sf_fwhm_at']}",
                flush=True,
            )
    excluded = (
        dict(
            file=a.exclude_objects,
            n=len(excl),
            null_objects_left=len(pops["null"]),
            broad_objects_left=len(pops["broad"]),
        )
        if excl
        else None
    )
    calibration = dict(
        min_n=MIN_N,
        stage4_sys=STAGE4_SYS,
        primary=PRIMARY,
        adopted=adopted,
        components=by_pop[PRIMARY],
        components_by_population=by_pop,
        gap_edges_days=GAP_EDGES,
        table=table,
        excluded_objects=excluded,
    )
    (out / "CALIBRATION.json").write_text(json.dumps(calibration, indent=1))
    lines = [
        "# Velocity-error calibration on objects that should not move (P1)",
        "",
        "`blrfit multiepoch calibrate`. Pairs: the four checks passed, both spectra good in the spectrum check and class A/C/F, "
        "independent spectra. `null`: coadd |Δv(Hα)| < 300 km/s, class A/F, clean; `broad`: |Δv(Hα)| < 1000 km/s. Offset = median change; "
        "floor = the term added in quadrature to the statistical error so that the pulls have NMAD 1. Pulls with the Stage 5 errors "
        "(statistical ⊕ 45/36 km/s) and with the calibrated ones; tail = fraction beyond 4σ (Gaussian: 0.006%).",
        "",
    ]
    if excl:
        lines += [
            f"**Restricted populations** (`--exclude-objects {a.exclude_objects}`): {len(excl)} object keys left out of every population and of the loop "
            f"term; null objects left {len(pops['null'])}, broad {len(pops['broad'])} (all catalog objects of each population, with or without pairs).",
            "",
        ]
    lines += [
        "| population | kind | line | profile | pairs (objects) | offset km/s (16–84%) | floor km/s (16–84%) | median stat. err | NMAD / tail, Stage 5 | NMAD / tail, calibrated |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, s in table.items():
        pop, kind, line, prof = k.split("|")
        lines.append(
            f"| {pop} | {kind} | {line} | {prof} | {s['n']} ({s['objects']}) | {s['offset']:+.0f} ({s['offset_16_84'][0]:+.0f} to {s['offset_16_84'][1]:+.0f}) | "
            f"{s['floor'] if s['floor'] is not None else '—'} ({s['floor_16_84'][0]:.0f}–{s['floor_16_84'][1]:.0f}) | {s['median_stat_err']:.0f} | "
            f"{s['nmad_stage5']:.2f} / {100 * s['tail4_stage5']:.1f}% | {s['nmad_calibrated']:.2f} / {100 * s['tail4_calibrated']:.1f}% |"
        )
    lines += [
        "",
        "## Adopted per kind (null where it has ≥ %d pairs, else broad; DESI nights ≤ 30 d keep the Stage 4 terms): display and pair tags"
        % MIN_N,
        "",
        "| kind | line | profile | source | pairs | offset km/s | floor km/s |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, s in adopted.items():
        kind, line, prof = k.split("|")
        lines.append(
            f"| {kind} | {line} | {prof} | {s['source']} | {s['n']} | {s['offset']:+.0f} | {s['floor']} |"
        )
    lines += [
        "",
        f"## The noise model of the curve fits (primary population: {PRIMARY})",
        "",
        "τ: one spectrum's jitter by instrument (km/s). BLR: the broad-line velocity wanders with time, SF² = 2(κF)²[1 − exp(−Δt/τ_c)], F the object's "
        "FWHM, Δt rest frame; shown as √SF² in units of FWHM after 1, 5 and 15 yr (16–84% over objects). ω: one object's SDSS-I/II or BOSS spectra "
        "offset against its DESI ones, beyond the jitters and the BLR term. ε: the independent per-pair term from loops.",
        "",
        "| population | line | pairs (objects) | τ DESI / SDSS / BOSS | κ | τ_c (d) | √SF²/FWHM 1 / 5 / 15 yr | km/s at the median FWHM, 15 yr | ω SDSS / BOSS | ε |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pop, L in by_pop.items():
        for line, c in L.items():
            b = c["blr"]
            lines.append(
                f"| {pop} | {line} | {c['pairs']} ({c['objects']}) | {c['tau']['DESI']} / {c['tau']['SDSS']} / {c['tau']['BOSS']} | {b['kappa']} ({b['kappa_16_84'][0]}–{b['kappa_16_84'][1]}) | "
                f"{b['tau_c_days']:.0f} ({b['tau_c_days_16_84'][0]:.0f}–{b['tau_c_days_16_84'][1]:.0f}) | "
                + " / ".join(
                    f"{b['sf_fwhm_at'][y]:.3f} ({b['sf_fwhm_at_16_84'][y][0]:.3f}–{b['sf_fwhm_at_16_84'][y][1]:.3f})"
                    for y in ("1 yr", "5 yr", "15 yr")
                )
                + f" | {b['sf_kms_at_median_fwhm']['15 yr']} (F {b['median_fwhm_kms']:.0f}) | {c['omega']['SDSS']} / {c['omega']['BOSS']} | {c['pair_eps']} |"
            )
    lines += [
        "",
        "### The BLR term by gap bin (√SF² in units of FWHM: measured with its 16–84%, the model; pull NMAD without / with the model)",
        "",
        "| population | line | rest gap (d) | pairs (objects) | median gap (d) | measured | model | NMAD without / with | beyond 4σ with |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pop, L in by_pop.items():
        for line, c in L.items():
            for bb in c["blr"]["bins"]:
                lines.append(
                    f"| {pop} | {line} | {bb['gap_lo']:.0f}–{bb['gap_hi']:.0f} | {bb['pairs']} ({bb['objects']}) | {bb['median_gap_days']:.0f} | "
                    f"{bb['measured']} ({bb['measured_16_84'][0]}–{bb['measured_16_84'][1]}) | {bb['model']} | {bb['nmad_without']} / {bb['nmad_with']} | {100 * bb['tail4_with']:.1f}% |"
                )
    (out / "CALIBRATION.md").write_text("\n".join(lines) + "\n")
    if verbose:
        print("\n".join(lines))
    return calibration


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)
