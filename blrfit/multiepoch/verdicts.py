"""Each object's velocity curve per line and its verdict, from all its good spectra together.

The data are the measured pairs (the four checks passed) between good spectra (spectra.check). Each pair gives the change
d = v(B) - v(A) of the later spectrum B against the earlier A. Its noise has two parts:
  - each spectrum's own noise: the forward direction (A's model over B's data) sees only B's noise, the reverse only A's
    (``blrfit.pairs``), so sigma_k = the median error of the directions in which spectrum k is the data; a spectrum shared by
    several pairs (the reference night above all) puts the same noise into all of them;
  - each spectrum's jitter tau (by instrument), one object's SDSS-I/II (BOSS) spectra offset against its DESI ones
    (omega), and an independent per-pair term (eps, from triangle closures), all calibrated on objects that should not
    move (``calibrate``);
  - the broad line's own wandering (the BLR term): the velocities of one object at rest-frame times t_k
    follow a damped random walk whose variogram is Gamma_kl = 2 (kappa F)^2 [1 - exp(-|t_k - t_l| / tau_c)], F its broad
    FWHM; for changes this gives the covariance -1/2 M Gamma M^T.
So Cov(d) = M diag(sigma_k^2) M^T - 1/2 M Gamma M^T + diag(eps^2) + sum_I omega_I^2 c_I c_I^T, M the pair-spectrum
incidence matrix (+1 for B, -1 for A), and the fits are generalized least squares. Spectra within 30 d are one time (a
cluster); every cluster has one velocity. A motion is then a change larger than the line's own wandering allows.

Test 1, does it move: chi2 of 'every cluster at the same velocity' minus chi2 of 'one free velocity per cluster', with
(clusters - 1) degrees of freedom; p < 2.7e-3 (the two-sided 3 sigma of one pair) = moving. It does not depend on
which spectrum is the reference. Only pairs whose profiles match (shape <= 0.5) enter;
pairs whose profiles do not match are counted (profile variable) but never claim a motion.
Test 2, steady: a straight line in rest-frame time (nearly constant acceleration) against free velocities, with three or
more clusters spanning more than a year; p < 0.01 = not steady.
Test 3, allowed by the mass: every pair of clusters, |u_i - u_j| - 2 sigma_ij <= the Roche ceiling over their rest-frame
gap (bound.max_change, q = 0.1, edge-on); else 'faster than its mass allows'.
  The generous limit (for a mass uncertain by 0.3-0.4 dex and a BLR radius by 0.19 dex): the same test with the BLR radius halved and the virial mass doubled together,
  treated as independent (the two ~1 sigma systematics in the direction that allows the most motion). ``blrfit.physics`` gives
  P_min ~ R^1.5 M^-0.5 and v_max ~ (M/R)^0.5, so P_min x 0.25 and v_max x 2 (GENEROUS_PMIN, GENEROUS_VMAX). The tag and the
  'too_fast' column keep the nominal test; 'too_fast_generous' and the three-valued 'mass_label' ('faster than its mass allows
  (even at the generous limit)' / 'faster at the nominal mass only' / 'allowed by its mass') are added beside them. The
  generous bound is larger at every gap, so a line too fast at the generous limit is too fast at the nominal mass.
Leave-one-out: test 1 without each spectrum; 'rests_on' lists the spectra without which the motion falls below 3 sigma
(tag: "disappears when one spectrum is excluded"); the same test with every spectrum of one time left out names the
times in 'rests_on_time' (tag: "disappears when one time is excluded").
Width: the median broad FWHM of the spectra at each time of the curve (spectra.check()'s 'fwhm',
as spectra.object_fwhm uses it); fwhm_ratio = the largest over the smallest. A moving line with fwhm_ratio > WIDTH_RATIO_MAX
(1.5) is tagged "width changes": a change of profile that the velocity may follow. A tag, not a cut: no verdict changes.
A line with two or more cross-time pairs above 4 sigma in the full noise model carries the tag "two pairs above 4 sigma"
(the column n_pairs_4sig); 'strict' marks a best pair with shape <= 0.3, the two directions within 2 sigma of each other
and both broad peaks at 8 sigma or more per pixel.
"""

from collections import defaultdict

import numpy as np
from scipy import stats

from . import bound as O
from . import common as K
from . import spectra as SP

P_MOVE = 2 * stats.norm.sf(3.0)  # 2.70e-3: moving (the two-sided 3 sigma of one pair)
P_MARG = P_MOVE  # the leave-one-out threshold: the motion disappears when it falls below 3 sigma
P_STEADY = 0.01
CLUSTER_DAYS = 30.0
STEADY_SPAN_YR = 1.0
MIN_FLOOR = 15.0  # km/s: keeps a pair's calibrated floor (display, strict tag) above zero
PAIR_EPS = (
    10.0  # km/s: the independent per-pair term of the curve fits (the rest is per spectrum / per object)
)
WIDTH_RATIO_MAX = (
    1.5  # the 'width changes' tag: largest / smallest median broad FWHM over the times of the curve
)
# the generous limit of the mass test: the BLR radius halved (R_BLR 0.19 dex uncertain) and the virial mass doubled
# (0.3-0.4 dex) together; P_min ~ R^1.5 M^-0.5 and v_max ~ (M/R)^0.5 (blrfit.physics), so P_min x 0.25 and v_max x 2
GENEROUS_R = 0.5  # R_BLR x 0.5
GENEROUS_M = 2.0  # M x 2
GENEROUS_PMIN = 0.25  # = GENEROUS_R ** 1.5 * GENEROUS_M ** -0.5 (written out so the factor is exact)
GENEROUS_VMAX = 2.0  # = (GENEROUS_M / GENEROUS_R) ** 0.5
assert (
    abs(GENEROUS_PMIN - GENEROUS_R**1.5 * GENEROUS_M**-0.5) < 1e-12
    and abs(GENEROUS_VMAX - (GENEROUS_M / GENEROUS_R) ** 0.5) < 1e-12
)
MASS_LABELS = (
    "allowed by its mass",
    "faster at the nominal mass only",
    "faster than its mass allows (even at the generous limit)",
)


class Calibration:
    def __init__(self, path, population=None):
        """population: whose noise model to use ('null', 'broad'; default the file's primary)."""
        import json

        js = json.load(open(path))
        self.adopted = js["adopted"]
        by = js.get("components_by_population") or {}
        self.population = population or js.get("primary", "null")
        self.components = by.get(self.population) or js.get("components", {})

    def blr(self, line):
        """(kappa, tau_c in years) of the BLR term; (0, 1) when the calibration has none."""
        b = self.components.get(line, {}).get("blr")
        return (float(b["kappa"]), float(b["tau_c_days"]) / 365.25) if b else (0.0, 1.0)

    def tau(self, inst, line):
        return float(self.components[line]["tau"][inst])

    def omega(self, line):
        return {k: float(v) for k, v in self.components[line]["omega"].items()}

    def eps(self, line):
        """The population's independent per-pair term (calibrate, triangle closures); PAIR_EPS if not calibrated."""
        return float(self.components.get(line, {}).get("pair_eps", PAIR_EPS))

    def floor(self, kind, line, prof):
        rec = self.adopted.get(f"{kind}|{line}|{prof}") or self.adopted.get(f"{kind}|{line}|match")
        fl = rec["floor"] if rec and rec.get("floor") is not None else np.nan
        if not np.isfinite(fl):  # a kind without calibration: the largest floor of that line and profile
            fl = max(
                v["floor"]
                for k, v in self.adopted.items()
                if k.endswith(f"|{line}|{prof}") and v.get("floor") is not None
            )
        return max(float(fl), MIN_FLOOR)


def generous(mass):
    """The mass record at the generous limit: PMIN_YR_Q01 x GENEROUS_PMIN, VMAX_Q01 x GENEROUS_VMAX (None without a mass)."""
    if not mass or not np.isfinite(mass.get("VMAX_Q01", np.nan)):
        return None
    return dict(
        mass,
        PMIN_YR_Q01=float(mass["PMIN_YR_Q01"]) * GENEROUS_PMIN,
        VMAX_Q01=float(mass["VMAX_Q01"]) * GENEROUS_VMAX,
    )


def mass_test(fit, tc, vmax, pmin):
    """Test 3 on one fit: the pairs of clusters whose |u_i - u_j| - 2 sigma_ij exceeds the largest change an orbit allows over
    their rest-frame gap (bound.max_change at vmax, pmin), as (|du|, the bound, the gap) rounded; empty when none."""
    fast = []
    for i in fit["clusters"]:
        for j in fit["clusters"]:
            if j <= i:
                continue
            var = (
                fit["V"].get((i, i), 0.0) + fit["V"].get((j, j), 0.0) - 2 * fit["V"].get((i, j), 0.0)
                if fit["V"]
                else 0.0
            )
            du = abs(fit["u"][j] - fit["u"][i])
            dvmax, _ = O.max_change(vmax, pmin, abs(tc[j] - tc[i]))
            if np.isfinite(dvmax) and du - 2 * np.sqrt(max(var, 0.0)) > dvmax:
                fast.append((round(du), round(dvmax), round(abs(tc[j] - tc[i]), 2)))
    return fast


def mass_label(fast, fast_generous):
    """The three-valued label of the mass test from the nominal and the generous lists of too-fast pairs."""
    if fast_generous:
        return MASS_LABELS[2]
    return MASS_LABELS[1] if fast else MASS_LABELS[0]


def clusters_of(times_mjd):
    """spectrum -> cluster index: single linkage of the observation dates, gaps <= CLUSTER_DAYS."""
    items = sorted(times_mjd.items(), key=lambda kv: kv[1])
    cl, c, last = {}, -1, -1e9
    for k, t in items:
        if t - last > CLUSTER_DAYS:
            c += 1
        cl[k] = c
        last = t
    return cl


def covariance(pairs, sig, t_rest, omega=None, inst=None, blr=None):
    """The covariance of the pair changes in the full noise model (module docstring) and the spectra in matrix order."""
    spec = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    si = {k: i for i, k in enumerate(spec)}
    P, N = len(pairs), len(spec)
    Mi = np.zeros((P, N))
    for j, p in enumerate(pairs):
        Mi[j, si[p["b"]]] += 1.0
        Mi[j, si[p["a"]]] -= 1.0
    C = Mi @ np.diag([sig[k] ** 2 for k in spec]) @ Mi.T + np.diag([p["floor"] ** 2 for p in pairs])
    if omega and inst:  # one object's SDSS-I/II (BOSS) spectra share an offset against its DESI ones
        for name, om in omega.items():
            c = np.array([float(inst[p["b"]] == name) - float(inst[p["a"]] == name) for p in pairs])
            if np.any(c):
                C = C + om**2 * np.outer(c, c)
    if blr and blr[0] > 0:  # the line's own wandering: covariance of changes from the variogram
        amp, tc = blr
        tt = np.array([t_rest[k] for k in spec])
        G = 2.0 * amp**2 * (1.0 - np.exp(-np.abs(tt[:, None] - tt[None, :]) / tc))
        C = C - 0.5 * Mi @ G @ Mi.T
    return C, spec


def design(pairs, cl_of, ref_cluster):
    """The matrix that maps the free cluster velocities (every cluster but ref_cluster) to the pair changes."""
    cls = sorted({cl_of[p["a"]] for p in pairs} | {cl_of[p["b"]] for p in pairs})
    free = [c for c in cls if c != ref_cluster]
    ci = {c: i for i, c in enumerate(free)}
    X = np.zeros((len(pairs), len(free)))
    for j, p in enumerate(pairs):
        cb, ca = cl_of[p["b"]], cl_of[p["a"]]
        if cb in ci:
            X[j, ci[cb]] += 1.0
        if ca in ci:
            X[j, ci[ca]] -= 1.0
    return X, cls, free, ci


def motion_matrix(pairs, sig, cl_of, t_rest, omega=None, inst=None, blr=None):
    """Q such that the motion test's chi2 difference is d^T Q d, and its degrees of freedom: with Gaussian noise of
    covariance C a velocity curve that produces the changes d0 gives a noncentral chi2 with noncentrality d0^T Q d0
    (used for the sensitivity to orbits, fraction)."""
    C, _ = covariance(pairs, sig, t_rest, omega, inst, blr)
    Ci = np.linalg.inv(C)
    ref = min(cl_of[p["a"]] for p in pairs)
    X, cls, free, _ = design(pairs, cl_of, ref)
    if not free:
        return None, 0
    A = X.T @ Ci @ X
    Q = Ci @ X @ np.linalg.pinv(A) @ X.T @ Ci
    return 0.5 * (Q + Q.T), len(free)


def gls(pairs, sig, cl_of, t_rest, ref_cluster, omega=None, inst=None, blr=None):
    """pairs: list of dict(a, b, d, floor). blr: (kappa F in km/s, tau_c in years) of the BLR term. Returns the fit: chi2
    of constant / free / linear, the cluster velocities relative to ref_cluster with their covariance, the acceleration."""
    C, spec = covariance(pairs, sig, t_rest, omega, inst, blr)
    P, N = len(pairs), len(spec)
    Ci = np.linalg.inv(C)
    d = np.array([p["d"] for p in pairs])
    X, cls, free, ci = design(pairs, cl_of, ref_cluster)
    chi0 = float(d @ Ci @ d)
    out = dict(n_pairs=P, n_spectra=N, clusters=cls, chi2_const=chi0)
    if free:
        A = X.T @ Ci @ X
        V = np.linalg.pinv(A)
        u = V @ X.T @ Ci @ d
        r = d - X @ u
        out.update(
            chi2_free=float(r @ Ci @ r),
            u={c: float(u[ci[c]]) for c in free},
            V={(a, b): float(V[ci[a], ci[b]]) for a in free for b in free},
        )
    else:
        out.update(chi2_free=chi0, u={}, V={})
    out["u"][ref_cluster] = 0.0
    # straight line in rest-frame time: d = a (t_B - t_A)
    x = np.array([t_rest[p["b"]] - t_rest[p["a"]] for p in pairs])
    if np.any(np.abs(x) > 0):
        va = 1.0 / float(x @ Ci @ x)
        a = va * float(x @ Ci @ d)
        rl = d - a * x
        out.update(accel=a, accel_err=np.sqrt(va), chi2_lin=float(rl @ Ci @ rl))
    else:
        out.update(accel=np.nan, accel_err=np.nan, chi2_lin=np.nan)
    return out


def components(pairs):
    adj = defaultdict(set)
    for p in pairs:
        adj[p["a"]].add(p["b"])
        adj[p["b"]].add(p["a"])
    seen, comps = set(), []
    for s in adj:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def triangles(pairs):
    """Closure residuals c = d_AB + d_BC - d_AC of every triangle of spectra whose three pairs are all measured. The shared
    terms (each spectrum's noise and jitter, the instrument offsets) cancel around a loop, so c measures what is
    independent per pair: template-dependent errors, i.e. profiles that differ between the epochs."""
    d = {(p["a"], p["b"]): p["d"] for p in pairs}
    adj = defaultdict(dict)
    for (a, b), v in d.items():
        adj[a][b] = v
    out = []
    for a in adj:
        for b in adj[a]:
            for c in adj.get(b, {}):
                if c in adj[a]:
                    out.append(adj[a][b] + adj[b][c] - adj[a][c])
    return np.array(out)


def pair_eps(pairs, eps_pop):
    """The independent per-pair term of one object and line: from its own triangles when it has three or more
    (robust: 1.4826 x the median |c| over sqrt(3)), never below the population value; (eps, n_triangles, eps_object)."""
    c = triangles(pairs)
    if len(c) >= 3:
        e_obj = float(1.4826 * np.median(np.abs(c)) / np.sqrt(3.0))
    elif len(c):
        e_obj = float(np.sqrt(np.mean(c**2) / 3.0))
    else:
        e_obj = np.nan
    return (max(eps_pop, e_obj) if np.isfinite(e_obj) else eps_pop), len(c), e_obj


def motion_p(pairs, sig, cl_of, t_rest, omega=None, inst=None, blr=None):
    """Test 1 on the component with the most clusters: (p, dof, fit, component)."""
    comps = components(pairs)
    if not comps:
        return np.nan, 0, None, set()
    comp = max(comps, key=lambda c: (len({cl_of[k] for k in c}), len(c)))
    sub = [p for p in pairs if p["a"] in comp]
    ncl = len({cl_of[k] for k in comp})
    if ncl < 2:
        return np.nan, 0, None, comp
    ref = min(cl_of[k] for k in comp)
    fit = gls(sub, sig, cl_of, t_rest, ref, omega, inst, blr)
    dof = ncl - 1
    dchi = max(fit["chi2_const"] - fit["chi2_free"], 0.0)
    return float(stats.chi2.sf(dchi, dof)), dof, fit, comp


def analyze(o, line, rows, spec, calib, mass, z, keep=None, fwhm=None):
    """The verdict of one object in one line. rows: the object's pair rows (all TARGETIDs); spec: spectra.check();
    fwhm: spectra.object_fwhm(spec) (else the median over the spectra of its measured pairs).
    keep: a dict to receive the internals (pairs, noise, clusters, fit, leave-one-out p values) for a dossier."""
    seen, recs = set(), []
    counts = defaultdict(int)
    for r in rows:
        if r["line"] != line or r["pair"] in seen or not K.independent(r["pair"]):
            continue
        seen.add(r["pair"])
        sa, sb = spec.get((o, r["id_a"], line)), spec.get((o, r["id_b"], line))
        if not sa or not sb or sa["status"] != "good" or sb["status"] != "good":
            counts["pairs with a spectrum not used"] += 1
            continue
        if not SP.measured(r):
            counts["pairs not measured"] += 1
            continue
        kind, sh = K.pair_kind(r), K.shape(r)
        prof = "match" if sh <= K.SHAPE_MAX else "mismatch"
        counts[f"pairs measured, profile {prof}"] += 1
        recs.append(
            dict(
                a=r["id_a"],
                b=r["id_b"],
                d=K.f(r["s_common"]),
                err=K.f(r["err"]),
                err_fwd=K.f(r["err_fwd"]),
                err_rev=K.f(r["err_rev"]),
                floor=calib.floor(kind, line, prof),
                prof=prof,
                shape=sh,
                kind=kind,
                pair=r["pair"],
                mjd_a=K.f(r["mjd_a"]),
                mjd_b=K.f(r["mjd_b"]),
                dt_rest=K.f(r["dt_rest_yr"]),
                mismatch_nsig=K.f(r["dir_mismatch"]) / K.f(r["err"]) if K.f(r["err"]) > 0 else np.nan,
                peaks=(K.f(r["peak_snr_a"]), K.f(r["peak_snr_b"])),
                ids=(r["id_a"], r["id_b"]),
            )
        )
    res = dict(
        object=o,
        line=line,
        **counts,
        n_match=sum(p["prof"] == "match" for p in recs),
        n_mismatch=sum(p["prof"] == "mismatch" for p in recs),
    )
    if not recs:
        res.update(verdict="not measurable", why="no measured pair between good spectra")
        return res, None
    # per-spectrum noise and times
    dir_err = defaultdict(list)
    times = {}
    for p in recs:
        if np.isfinite(p["err_fwd"]):
            dir_err[p["b"]].append(p["err_fwd"])
        if np.isfinite(p["err_rev"]):
            dir_err[p["a"]].append(p["err_rev"])
        times[p["a"]], times[p["b"]] = p["mjd_a"], p["mjd_b"]
    sig = {k: (float(np.median(v)) if v else np.nan) for k, v in dir_err.items()}
    for p in recs:
        for k in (p["a"], p["b"]):
            if not np.isfinite(sig.get(k, np.nan)):
                sig[k] = p["err"] / np.sqrt(2)
    inst = {k: spec[(o, k, line)]["instrument"] for k in sig}
    sig = {k: float(np.hypot(v, calib.tau(inst[k], line))) for k, v in sig.items()}  # + the spectrum's jitter
    omega = calib.omega(line)
    cl_of = clusters_of(times)
    t_rest = {k: t / 365.25 / (1 + z) for k, t in times.items()}
    F = (fwhm or {}).get((o, line))
    if not F:
        fw = [spec[(o, k, line)].get("fwhm", np.nan) for k in sig]
        fw = [x for x in fw if np.isfinite(x) and x > 0]
        F = float(np.median(fw)) if fw else np.nan
    kap, tc = calib.blr(line)
    blr = (kap * F, tc) if np.isfinite(F) and kap > 0 else None
    res.update(
        fwhm=round(F, 0) if np.isfinite(F) else np.nan,
        blr_kms_5yr=round(float(np.sqrt(2.0 * (1 - np.exp(-5.0 / tc)))) * kap * F, 1) if blr else 0.0,
    )

    def pair_sigma(p):
        """One pair's error in the full noise model (display, the count of pairs above 4 sigma, the strict tag)."""
        ia, ib = inst[p["a"]], inst[p["b"]]
        v = p["err"] ** 2 + calib.tau(ia, line) ** 2 + calib.tau(ib, line) ** 2 + p["floor"] ** 2
        v += sum(om**2 for name, om in omega.items() if (ia == name) != (ib == name))
        if blr:
            v += 2.0 * blr[0] ** 2 * (1.0 - np.exp(-abs(t_rest[p["b"]] - t_rest[p["a"]]) / blr[1]))
        return float(np.sqrt(v))

    match0 = [p for p in recs if p["prof"] == "match"]
    eps, n_tri, eps_obj = pair_eps(match0, calib.eps(line))
    match = [dict(p, floor=eps) for p in match0]
    res.update(
        pair_eps=round(eps, 1),
        n_triangles=n_tri,
        pair_eps_object=round(eps_obj, 1) if np.isfinite(eps_obj) else np.nan,
    )
    if keep is not None:
        keep.update(
            recs=recs,
            sig=sig,
            inst=inst,
            omega=omega,
            cl_of=cl_of,
            t_rest=t_rest,
            times=times,
            blr=blr,
            pair_sigma=pair_sigma,
            eps=eps,
        )
    res["n_times_measured"] = len(set(cl_of.values()))
    if not match:
        res.update(
            verdict="profile variable", profile="every", why="every measured pair's profiles do not match"
        )
        return res, None
    p1, dof, fit, comp = motion_p(match, sig, cl_of, t_rest, omega, inst, blr)
    res.update(p_move=p1, dof=dof, n_times=len({cl_of[k] for k in comp}) if comp else 0)
    if fit is None:
        res.update(
            verdict="profile variable" if res["n_mismatch"] else "not measurable",
            profile="some" if res["n_mismatch"] else "",
            why="matching pairs only within one time",
        )
        return res, None
    sub = [p for p in match if p["a"] in comp]
    ncl = len(fit["clusters"])
    if keep is not None:
        keep.update(fit=fit, comp=comp, sub=sub)
    span = (max(times[k] for k in comp) - min(times[k] for k in comp)) / 365.25
    res.update(
        span_yr=round(span, 2),
        closure_p=float(stats.chi2.sf(fit["chi2_free"], len(sub) - (ncl - 1)))
        if len(sub) > ncl - 1
        else np.nan,
    )
    # the width across the times of the curve: the median broad FWHM of the spectra at each time, the largest over the smallest
    # (fwhm_ratio). Above WIDTH_RATIO_MAX a moving line is tagged 'width changes' below: a tag, not a cut.
    wmed = []
    for c in fit["clusters"]:
        w = [spec[(o, k, line)].get("fwhm", np.nan) for k in comp if cl_of[k] == c]
        w = [x for x in w if np.isfinite(x) and x > 0]
        if w:
            wmed.append(float(np.median(w)))
    ratio = max(wmed) / min(wmed) if len(wmed) >= 2 else np.nan
    res["fwhm_ratio"] = round(ratio, 3) if np.isfinite(ratio) else np.nan
    # the strongest single pair across times, for display and the strict tag
    cross = [p for p in sub if cl_of[p["a"]] != cl_of[p["b"]]]
    for p in cross:
        p["sig"] = abs(p["d"]) / pair_sigma(p)
    best = max(cross, key=lambda p: p["sig"]) if cross else None
    res["n_pairs_4sig"] = sum(p["sig"] >= 4 for p in cross)
    if best:
        res.update(
            best_pair=best["pair"],
            best_change=round(best["d"], 1),
            best_sigma=round(best["sig"], 2),
            strict=bool(
                best["shape"] <= 0.3
                and np.isfinite(best["mismatch_nsig"])
                and best["mismatch_nsig"] <= 2
                and min(best["peaks"]) >= 8
            ),
        )
    # verdict
    if p1 < P_MOVE:
        verdict = "moving"
    elif res["n_mismatch"]:
        verdict = "profile variable"
    else:
        verdict = "constant"
    res["verdict"] = verdict
    if res["n_mismatch"]:
        res["profile"] = "some"
    tags = []
    if verdict == "moving":
        tags.append("two times" if ncl == 2 else f"{ncl} times")
        # test 2: steady
        if ncl >= 3 and span > STEADY_SPAN_YR and np.isfinite(fit["chi2_lin"]):
            p2 = float(stats.chi2.sf(max(fit["chi2_lin"] - fit["chi2_free"], 0.0), ncl - 2))
            res.update(p_steady=p2, accel=round(fit["accel"], 1), accel_err=round(fit["accel_err"], 1))
            tags.append("steady" if p2 >= P_STEADY else "not steady")
        else:
            res.update(
                accel=round(fit["accel"], 1) if np.isfinite(fit["accel"]) else np.nan,
                accel_err=round(fit["accel_err"], 1) if np.isfinite(fit["accel_err"]) else np.nan,
            )
        # test 3: the mass, at the nominal mass (the tag, too_fast) and at the generous limit (too_fast_generous, mass_label)
        if mass and np.isfinite(mass.get("VMAX_Q01", np.nan)):
            tc = {c: np.mean([t_rest[k] for k in comp if cl_of[k] == c]) for c in fit["clusters"]}
            fast = mass_test(fit, tc, mass["VMAX_Q01"], mass["PMIN_YR_Q01"])
            mg = generous(mass)
            fast_g = mass_test(fit, tc, mg["VMAX_Q01"], mg["PMIN_YR_Q01"])
            tags.append("faster than its mass allows" if fast else "allowed by its mass")
            if fast:
                res["too_fast"] = fast
            if fast_g:
                res["too_fast_generous"] = fast_g
            res["mass_label"] = mass_label(fast, fast_g)
        else:
            tags.append("no mass")
            res["mass_label"] = "no mass"
        # leave-one-out
        rests = []
        for k in sorted(comp):
            left = [p for p in sub if k not in (p["a"], p["b"])]
            pk, _, _, _ = motion_p(left, sig, cl_of, t_rest, omega, inst, blr)
            if keep is not None:
                keep.setdefault("loo", {})[k] = pk
            if not (np.isfinite(pk) and pk < P_MARG):
                rests.append(k)
        res["rests_on"] = ";".join(rests)
        if ncl == 2:
            tags.append("two spectra or one pair of times")
        elif rests:
            tags.append("disappears when one spectrum is excluded")
        # leave-one-time-out: the same test with every spectrum of one time left out, so that a
        # time of several spectra (an SDSS-RM season, two DESI nights) can be removed whole; names the times whose removal
        # drops the motion below P_MARG (as "year" of the time's first spectrum)
        rests_t = []
        if ncl >= 3:
            for c in sorted(set(cl_of[k] for k in comp)):
                drop = {k for k in comp if cl_of[k] == c}
                left = [p for p in sub if p["a"] not in drop and p["b"] not in drop]
                pk, _, _, _ = motion_p(left, sig, cl_of, t_rest, omega, inst, blr)
                if keep is not None:
                    keep.setdefault("loto", {})[c] = pk
                if not (np.isfinite(pk) and pk < P_MARG):
                    rests_t.append(min(drop, key=lambda k: times[k]))
        res["rests_on_time"] = ";".join(rests_t)
        if ncl >= 3 and rests_t:
            tags.append("disappears when one time is excluded")
        if res.get("n_pairs_4sig", 0) >= 2:
            tags.append("two pairs above 4 sigma")
        if res.get("strict"):
            tags.append("strict")
        if best:
            for k in best["ids"]:
                for tg in spec[(o, k, line)]["tags"]:
                    if tg not in tags and tg in ("weak_peak", "daily", "boss", "class_W", "very_broad_B"):
                        tags.append(tg)
        if res["n_mismatch"]:
            tags.append("profile does not match in some pairs")
        if np.isfinite(res.get("pair_eps_object", np.nan)) and res["pair_eps_object"] > 3 * calib.eps(line):
            tags.append("pairs do not add up")
        if np.isfinite(ratio) and ratio > WIDTH_RATIO_MAX:
            tags.append("width changes")
    res["tags"] = ";".join(tags)
    # the curve also relative to its best-measured time (the time whose velocity differences to all others are the most
    # precise)
    V0 = {
        (i, j): (fit["V"].get((i, j), 0.0) if fit["V"] else 0.0)
        for i in fit["clusters"]
        for j in fit["clusters"]
    }
    cost = {
        r: (
            np.median([V0[(c, c)] + V0[(r, r)] - 2 * V0[(c, r)] for c in fit["clusters"] if c != r])
            if len(fit["clusters"]) > 1
            else 0.0
        )
        for r in fit["clusters"]
    }
    best = min(cost, key=cost.get)
    curve = dict(
        clusters={
            int(c): dict(
                u=round(fit["u"].get(c, 0.0), 1),
                err=round(float(np.sqrt(max(fit["V"].get((c, c), 0.0), 0.0))), 1)
                if c != min(fit["clusters"])
                else 0.0,
                u_best=round(fit["u"].get(c, 0.0) - fit["u"].get(best, 0.0), 1),
                err_best=round(float(np.sqrt(max(V0[(c, c)] + V0[(best, best)] - 2 * V0[(c, best)], 0.0))), 1)
                if c != best
                else 0.0,
                t_obs_yr=round(
                    float(np.mean([1858.87 + times[k] / 365.25 for k in comp if cl_of[k] == c])), 2
                ),
                spectra=sorted(k for k in comp if cl_of[k] == c),
            )
            for c in fit["clusters"]
        },
        ref_cluster=int(min(fit["clusters"])),
        best_ref=int(best),
    )
    return res, curve
