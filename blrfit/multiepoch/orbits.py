"""Orbit tests of the velocity curves.

For every object and line whose velocity curve has two or more times (``verdicts.analyze``), this writes what a figure
or a page needs to show the curve against orbits, from one file:
  - the velocity at each time on the offset scale (the best-measured time carries its own fit's offset), with its error;
  - the straight line in time (a steady drift: the long-period limit of an orbit), the same test as the 'steady' tag of
    ``verdicts`` (the straight line against free velocities, in pair space, p on times - 2 degrees of freedom);
  - for three or more times, the circular orbits the virial mass allows, fitted to all times with their full covariance:
    every period from P_min, the closest orbit at which the active black hole's Roche lobe still holds its broad-line
    region, to 10,000 yr, with the amplitude at most v_max (P/P_min)^(-1/3) (q = 0.1, edge-on: the most permissive case;
    ``orbit_fit``); the binary's center of mass within +-com_max of the narrow lines (the orbit must produce the offsets
    themselves), and, for comparison, the same fit with no mass condition (the periods the data alone prefer). The
    goodness of fit counts the period as a fitted parameter: dof = times - 4;
  - every pair of times from the curve (the velocity difference and its error from the joint fit's covariance), and every
    measured pair of spectra between two times (|change|, its error in the full noise model, profiles match or not), with
    the parameters of the largest change an orbit at the virial mass allows over a gap and of the line's own wandering;
  - the motion test without each spectrum in turn (leave-one-out p values).
The robust movers are grouped by the orbit test (``groups``).

Every allowed-orbit fit is also made at the generous limit of the mass (``verdicts``: P_min x 0.25, v_max x 2):
'orbit_mass_generous' (P range, K, com, chi2, p) and 'orbit_fits_generous' are added, with the three-valued 'orbit_label'
('an allowed orbit fits' / 'fits only at the generous limit' / 'no allowed orbit fits'; 'no mass' for a line without a virial
mass, whose nominal 'orbit_mass' is the free orbit with no cap and whose 'orbit_fits' is True) and, from ``verdicts.analyze``,
the mass test's 'too_fast_generous' and 'mass_label'. The bound on the binary's center of mass is a parameter (com_max,
default ``orbit_fit.COM_MAX`` = 300 km/s; 'com_max' is recorded in every record).

    blrfit multiepoch orbits --run DIR --pairs PAIRS.csv --catalog CATALOG --masses MASSES --facts FACTS.csv
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

from . import common as K
from . import orbit_fit as OF
from . import spectra as SP
from . import verdicts as V

P_FIT = 0.01  # an orbit "fits" when its chi2 probability is at least this
DCHI2_MASS = 4.0  # ... and the mass condition costs at most this much chi2 against the best orbit without it
PERIODS = np.logspace(0, 4, 1200)  # yr, rest frame (orbit_fit.scan's grid)
ORBIT_LABELS = ("an allowed orbit fits", "fits only at the generous limit", "no allowed orbit fits")


def year(mjd):
    return 1858.87 + mjd / 365.25


def clean(x):
    """NaN and infinities to None, recursively (strict JSON for the site)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def summary(chis, n_data, n_par):
    """Best period, its 1-sigma range (delta chi2 <= 2.3) and the chi2 probability with the period counted."""
    chis = np.asarray(chis, float)
    if not np.isfinite(chis).any():
        return dict(chi2=None)
    i = int(np.nanargmin(chis))
    ok = PERIODS[chis <= chis[i] + 2.3]
    dof = n_data - n_par - 1
    return dict(
        chi2=round(float(chis[i]), 3),
        P=round(float(PERIODS[i]), 2),
        P_lo=round(float(ok.min()), 2),
        P_hi=round(float(ok.max()), 2),
        dof=int(dof),
        p=float(stats.chi2.sf(chis[i], dof)) if dof > 0 else None,
    )


def orbit_ok(d, key="orbit_mass"):
    """An orbit the mass allows fits this line: the mass condition costs at most DCHI2_MASS against the best orbit without it,
    and, with at least one degree of freedom left (five or more times), the chi2 probability is at least P_FIT. With four
    times or fewer the orbit cannot be tested: it only says whether an allowed orbit can pass through the points.
    key: 'orbit_mass' (the nominal mass) or 'orbit_mass_generous' (the generous limit)."""
    om, of = d.get(key) or {}, d.get("orbit_free") or {}
    if om.get("chi2") is None or of.get("chi2") is None:
        return None
    if om["chi2"] - of["chi2"] > DCHI2_MASS:
        return False
    return om["p"] >= P_FIT if om.get("p") is not None else True


def orbit_label(fits, fits_generous, has_mass=True):
    """The three-valued label from the nominal and the generous orbit tests (None without an orbit test). Without a virial mass
    the nominal 'orbit_mass' is the free orbit (no cap) and orbit_fits is True, as since 2026-10-15; the label is then 'no mass'."""
    if not has_mass:
        return "no mass"
    if fits is None:
        return None
    if fits:
        return ORBIT_LABELS[0]
    return ORBIT_LABELS[1] if fits_generous else ORBIT_LABELS[2]


def line_orbits(o, line, res, keep, mass, z, facts, com_max=OF.COM_MAX):
    fit, comp, cl_of, t_rest, times = (keep[k] for k in ("fit", "comp", "cl_of", "t_rest", "times"))
    clusters = sorted(fit["clusters"])
    tc = {c: float(np.mean([t_rest[s] for s in comp if cl_of[s] == c])) for c in clusters}
    ty = {c: float(np.mean([year(times[s]) for s in comp if cl_of[s] == c])) for c in clusters}
    V0 = {(i, j): (fit["V"].get((i, j), 0.0) if fit["V"] else 0.0) for i in clusters for j in clusters}
    u0 = {c: fit["u"].get(c, 0.0) for c in clusters}
    cost = {
        r: (
            np.median([V0[(c, c)] + V0[(r, r)] - 2 * V0[(c, r)] for c in clusters if c != r])
            if len(clusters) > 1
            else 0.0
        )
        for r in clusters
    }
    ref = min(cost, key=cost.get)
    u = {c: u0[c] - u0[ref] for c in clusters}
    Vc = {
        (i, j): V0[(i, j)] - V0[(i, ref)] - V0[(j, ref)] + V0[(ref, ref)]
        for i in clusters
        for j in clusters
        if i != ref and j != ref
    }
    err = {c: (float(np.sqrt(max(Vc[(c, c)], 0.0))) if c != ref else 0.0) for c in clusters}
    offs = [
        facts[(s.split("@")[0], line)] for s in comp if cl_of[s] == ref and (s.split("@")[0], line) in facts
    ]
    v_ref = float(np.median(offs)) if offs else np.nan
    s_ref = max(50.0, float(np.std(offs)) if len(offs) > 1 else 50.0) if offs else np.nan
    m = mass or {}
    vmax, pmin = float(m.get("VMAX_Q01", np.nan)), float(m.get("PMIN_YR_Q01", np.nan))
    mg = V.generous(m) or {}
    vmax_g, pmin_g = float(mg.get("VMAX_Q01", np.nan)), float(mg.get("PMIN_YR_Q01", np.nan))
    blr = keep.get("blr")
    n = len(clusters)
    inst = keep["inst"]
    out = dict(
        object=o,
        line=line,
        verdict=res["verdict"],
        tags=res.get("tags", ""),
        p_move=res.get("p_move"),
        n_times=n,
        z=round(float(z), 5),
        ref_year=round(ty[ref], 3),
        v_ref=round(v_ref, 1) if np.isfinite(v_ref) else None,
        times=[
            dict(
                year=round(ty[c], 3),
                t_rest=round(tc[c], 3),
                u=round(u[c], 1),
                err=round(err[c], 1),
                v=round(v_ref + u[c], 1) if np.isfinite(v_ref) else None,
                ref=bool(c == ref),
                inst=sorted({inst[s] for s in comp if cl_of[s] == c}),
                spectra=sorted(s for s in comp if cl_of[s] == c),
            )
            for c in sorted(clusters, key=ty.get)
        ],
        mass=dict(
            log_m=round(float(m.get("LOGMBH", np.nan)), 3) if m else None,
            r_blr_ld=round(float(m.get("R_BLR_LTD", np.nan)), 2) if m else None,
            p_min_yr=round(pmin, 2) if np.isfinite(pmin) else None,
            v_max=round(vmax, 1) if np.isfinite(vmax) else None,
            p_min_yr_generous=round(pmin_g, 2) if np.isfinite(pmin_g) else None,
            v_max_generous=round(vmax_g, 1) if np.isfinite(vmax_g) else None,
        ),
        wander=dict(kF=round(blr[0], 2), tau_yr=round(blr[1], 3)) if blr else None,
        too_fast=res.get("too_fast"),
        too_fast_generous=res.get("too_fast_generous"),
        mass_label=res.get("mass_label"),
        com_max=float(com_max),
        loo=[
            dict(
                s=sid.split("@")[0],
                year=round(year(times[sid]), 3),
                inst=inst.get(sid, ""),
                p=float(pk) if np.isfinite(pk) else None,
            )
            for sid, pk in sorted((keep.get("loo") or {}).items(), key=lambda kv: times.get(kv[0], 0.0))
        ],
    )
    # every measured pair between two times
    pairs = []
    for p in keep["recs"]:
        ca, cb = cl_of.get(p["a"]), cl_of.get(p["b"])
        if ca is None or cb is None or ca == cb:
            continue
        se = keep["pair_sigma"](dict(p, floor=keep["eps"]) if p["prof"] == "match" else p)
        pairs.append(
            dict(
                a=p["a"],
                b=p["b"],
                dt=round(abs(p["dt_rest"]), 4),
                d=round(p["d"], 1),
                se=round(se, 1),
                match=p["prof"] == "match",
                shape=round(p["shape"], 3) if np.isfinite(p["shape"]) else None,
            )
        )
    out["pairs"] = pairs
    # every pair of times, from the curve (the velocity difference and its error from the covariance of the joint fit, built
    # from the pairs whose profiles match): what the mass test compares with the largest change an orbit allows
    tp = []
    for ii, i in enumerate(clusters):
        for j in clusters[ii + 1 :]:
            var = V0[(i, i)] + V0[(j, j)] - 2 * V0[(i, j)]
            a_, b_ = (i, j) if ty[i] <= ty[j] else (j, i)
            tp.append(
                dict(
                    ya=round(ty[a_], 3),
                    yb=round(ty[b_], 3),
                    dt=round(abs(tc[j] - tc[i]), 4),
                    du=round(u0[b_] - u0[a_], 1),
                    se=round(float(np.sqrt(max(var, 0.0))), 1),
                )
            )
    out["time_pairs"] = tp
    if n < 2:
        return out
    orb = (
        OF.scan(tc, u, Vc, ref, vmax, pmin, v_ref, s_ref, periods=PERIODS, com_max=com_max)
        if n >= 3
        else None
    )
    tt = np.linspace(min(ty.values()) - 1.0, max(ty.values()) + 1.0, 60)
    base = v_ref if np.isfinite(v_ref) else 0.0
    # the steady test of verdicts.analyze: the straight line in rest-frame time against free velocities, on the pairs
    # (fit["chi2_lin"] - fit["chi2_free"], times - 2 degrees of freedom), with the slope of that fit
    dof_lin = n - 2
    chi_lin = (
        max(float(fit["chi2_lin"]) - float(fit["chi2_free"]), 0.0)
        if np.isfinite(fit.get("chi2_lin", np.nan))
        else np.nan
    )
    out["steady"] = dict(
        accel=round(float(fit["accel"]), 2) if np.isfinite(fit["accel"]) else None,
        accel_err=round(float(fit["accel_err"]), 2) if np.isfinite(fit["accel_err"]) else None,
        chi2=round(chi_lin, 3) if np.isfinite(chi_lin) else None,
        dof=int(dof_lin),
        p=float(stats.chi2.sf(chi_lin, dof_lin)) if dof_lin > 0 and np.isfinite(chi_lin) else None,
        curve=dict(
            year=[round(x, 3) for x in tt],
            v=[round(base + float(fit["accel"]) * (x - ty[ref]) / (1 + z), 1) for x in tt],
        )
        if np.isfinite(fit["accel"])
        else None,
    )
    if n < 3:
        return out
    out["orbit_free_rel"] = summary(orb["curves"]["rel_free"], n - 1, 2)
    out["orbit_mass_rel"] = summary(orb["curves"]["rel_mass"], n - 1, 2)
    yg = np.linspace(min(ty.values()) - 2.0, max(ty.values()) + 2.0, 300)
    tg = tc[ref] + (yg - ty[ref]) / (1 + z)

    def curve_of(A, B, c, P):
        w = 2 * np.pi / P
        return dict(
            year=[round(x, 3) for x in yg],
            v=[round(float(A * np.sin(w * t) + B * np.cos(w * t) + c), 1) for t in tg],
        )

    if np.isfinite(v_ref):
        out["orbit_free"] = summary(orb["curves"]["abs_free"], n, 3)
        of = out["orbit_free"]
        if (
            of.get("chi2") is not None
        ):  # the orbit the data alone prefer (no mass condition; centre of mass bounded)
            # drawn at the best period, or, when many periods pass through the points exactly (too few times), at the slowest
            P_draw = of["P_hi"] if (of["chi2"] < 0.5 and of["P_hi"] < 9000) else of["P"]
            _c2, A, B, c = OF.best_abs(tc, u, Vc, ref, P_draw, np.nan, np.nan, v_ref, s_ref, com_max=com_max)
            if np.isfinite(A):
                of.update(
                    K=round(float(np.hypot(A, B)), 1),
                    com=round(float(c), 1),
                    P_drawn=round(float(P_draw), 2),
                    curve=curve_of(A, B, c, P_draw),
                )

        def allowed(vm, pm):
            """The best orbit the mass allows (vm, pm) at every period: the summary with K, com and the curve at the best one."""
            fits = [OF.best_abs(tc, u, Vc, ref, P, vm, pm, v_ref, s_ref, com_max=com_max) for P in PERIODS]
            chis = np.array([f[0] for f in fits])
            sm = summary(chis, n, 3)
            if sm.get("chi2") is not None:
                k = int(np.nanargmin(chis))
                _c2, A, B, c = fits[k]
                sm.update(
                    K=round(float(np.hypot(A, B)), 1),
                    com=round(float(c), 1),
                    curve=curve_of(A, B, c, PERIODS[k]),
                )
            return sm

        out["orbit_mass"] = allowed(vmax, pmin)
        if np.isfinite(vmax_g) and np.isfinite(
            pmin_g
        ):  # the generous limit (the module docstring): the same fit at P_min x 0.25, v_max x 2
            out["orbit_mass_generous"] = allowed(vmax_g, pmin_g)
    out["orbit_fits"] = orbit_ok(out)
    out["orbit_fits_generous"] = orbit_ok(out, "orbit_mass_generous")
    out["orbit_label"] = orbit_label(
        out["orbit_fits"], out["orbit_fits_generous"], bool(np.isfinite(vmax) and np.isfinite(pmin))
    )
    out["orbit_tested"] = bool((out.get("orbit_mass") or {}).get("p") is not None)
    return out


G = {}  # the tables the workers share (filled by run_orbits, inherited by fork)


def object_orbits(o):
    """All lines of one object with a curve: 'object|line' -> record."""
    vd, masses, members, zof = G["vd"], G["masses"], G["members"], G["zof"]
    mass = K.mass_of(o, masses, members)
    z = K.redshift_of(o, zof, members)
    rec = {}
    for line in K.LINES:
        v = vd.get((o, line))
        if not v or not v.get("n_times") or int(v["n_times"]) < max(2, G.get("min_times", 2)):
            continue
        if G.get("verdict") and v["verdict"] != G["verdict"]:
            continue
        keep = {}
        res, _curve = V.analyze(
            o, line, G["byobj"][o], G["spec"], G["calib"], mass, z, keep=keep, fwhm=G["fwhm"]
        )
        if "fit" not in keep:
            continue
        if res["verdict"] != v["verdict"]:
            raise RuntimeError(
                f"{o} {line}: verdict {res['verdict']} here against {v['verdict']} in verdicts.csv (rerun on the same inputs)"
            )
        rec[f"{o}|{line}"] = line_orbits(
            o, line, res, keep, mass, z, G["facts"], com_max=G.get("com_max", OF.COM_MAX)
        )
    return rec


ORBIT_COLUMNS = [
    "object",
    "line",
    "verdict",
    "n_times",
    "log_m",
    "p_min_yr",
    "v_max",
    "steady_p",
    "orbit_mass_chi2",
    "orbit_mass_dof",
    "orbit_mass_p",
    "orbit_mass_P",
    "orbit_mass_P_lo",
    "orbit_mass_P_hi",
    "orbit_mass_K",
    "orbit_free_chi2",
    "orbit_free_P_lo",
    "orbit_free_P_hi",
    "orbit_fits",
    "orbit_tested",
    "too_fast",
    "orbit_mass_com",
    "orbit_free_com",
    "com_max",
    "too_fast_generous",
    "mass_label",
    "p_min_yr_generous",
    "v_max_generous",
    "orbit_mass_generous_chi2",
    "orbit_mass_generous_p",
    "orbit_mass_generous_P",
    "orbit_mass_generous_P_lo",
    "orbit_mass_generous_P_hi",
    "orbit_mass_generous_K",
    "orbit_mass_generous_com",
    "orbit_fits_generous",
    "orbit_label",
]


def run_orbits(
    run,
    inputs,
    out=None,
    objects=None,
    verdict=None,
    min_times=2,
    com_max=OF.COM_MAX,
    nproc=None,
    verbose=True,
):
    """The orbit records of every line with a curve of ``run`` (the directory of ``run_verdicts``: verdicts.csv and
    CALIBRATION.json), written to orbits.json and orbits.csv in ``run`` (or ``out``, a JSON path). Returns the records."""
    run = Path(run)
    vd = {(r["object"], r["line"]): r for r in csv.DictReader(open(run / "verdicts.csv"))}
    rows = inputs.load_pairs()
    C = inputs.load_catalog()
    okey = K.object_key(C)
    masses = inputs.load_masses()
    calib = V.Calibration(run / "CALIBRATION.json")
    spec = SP.check(rows, okey)
    fwhm = SP.object_fwhm(spec)
    byobj = K.by_object(rows, okey)
    zof = K.redshifts(C)
    members = K.members_of(rows, okey)
    facts = K.load_facts(inputs.facts)
    todo = sorted(
        {
            o
            for (o, line), v in vd.items()
            if v.get("n_times")
            and int(v["n_times"]) >= max(2, min_times)
            and (not verdict or v["verdict"] == verdict)
        }
    )
    if objects:
        todo = [o for o in todo if o in set(str(x) for x in objects)]
    G.update(
        vd=vd,
        byobj=byobj,
        spec=spec,
        calib=calib,
        fwhm=fwhm,
        masses=masses,
        members=members,
        zof=zof,
        facts=facts,
        com_max=com_max,
        verdict=verdict,
        min_times=min_times,
    )
    if verbose:
        print(
            f"{len(todo)} objects; com_max {com_max:.0f} km/s"
            + (f"; verdict {verdict}" if verdict else "")
            + f"; min times {min_times}",
            flush=True,
        )
    result = {}
    nproc = max(1, (os.cpu_count() or 2) - 1) if nproc is None else nproc
    nproc = max(1, min(nproc, len(todo)))
    if nproc == 1:
        for o in todo:
            result.update(object_orbits(o))
    else:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(nproc) as pool:
            for k, rec in enumerate(pool.imap_unordered(object_orbits, todo, chunksize=4), 1):
                result.update(rec)
                if verbose and k % 200 == 0:
                    print(f"{k}/{len(todo)} objects", flush=True)
    jpath = Path(out) if out else run / "orbits.json"
    result = clean(result)
    jpath.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    cpath = jpath.with_suffix(".csv")
    with open(cpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ORBIT_COLUMNS)
        for _key, d in sorted(result.items()):
            om, of, st, og = (
                d.get("orbit_mass") or {},
                d.get("orbit_free") or {},
                d.get("steady") or {},
                d.get("orbit_mass_generous") or {},
            )
            w.writerow(
                [
                    d["object"],
                    d["line"],
                    d["verdict"],
                    d["n_times"],
                    d["mass"]["log_m"],
                    d["mass"]["p_min_yr"],
                    d["mass"]["v_max"],
                    st.get("p"),
                    om.get("chi2"),
                    om.get("dof"),
                    om.get("p"),
                    om.get("P"),
                    om.get("P_lo"),
                    om.get("P_hi"),
                    om.get("K"),
                    of.get("chi2"),
                    of.get("P_lo"),
                    of.get("P_hi"),
                    d.get("orbit_fits"),
                    d.get("orbit_tested"),
                    bool(d.get("too_fast")),
                    om.get("com"),
                    of.get("com"),
                    d.get("com_max"),
                    bool(d.get("too_fast_generous")),
                    d.get("mass_label"),
                    d["mass"].get("p_min_yr_generous"),
                    d["mass"].get("v_max_generous"),
                    og.get("chi2"),
                    og.get("p"),
                    og.get("P"),
                    og.get("P_lo"),
                    og.get("P_hi"),
                    og.get("K"),
                    og.get("com"),
                    d.get("orbit_fits_generous"),
                    d.get("orbit_label"),
                ]
            )
    if verbose:
        print(f"{len(result)} curves -> {jpath}, {cpath}")
    return result
