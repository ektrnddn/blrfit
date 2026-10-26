"""Circular-orbit fits to one object's velocity curve (the cluster velocities of verdicts.gls and their covariance), for the
dossier of the robust movers (MULTIEPOCH_PLAN.md). The line-of-sight velocity of the active black hole on a circular orbit
is g(t) = A sin(2 pi t / P) + B cos(2 pi t / P), amplitude K = hypot(A, B). The velocities are relative to the reference
cluster, so the relative model is g(t_c) - g(t_ref) (the centre of mass drops out). The absolute model adds the
reference cluster's measured offset from the narrow lines, v_ref = g(t_ref) + c, with the centre-of-mass velocity c
within +-com_max of the narrow lines (COM_MAX = 300 km/s by default; scan and best_abs take com_max as a parameter since
2026-10-16 so the limit can be tested, e.g. at 500 km/s, without changing the default). The mass condition (q = 0.1,
edge-on, the generous case): P >= P_min and K <= v_max (P / P_min)^(-1/3) (stage6_orbits, bh_masses); the caller chooses
vmax and pmin, so the same functions serve the nominal mass and the generous limit of verdicts.py (P_min x 0.25, v_max x 2:
P_min ~ R^1.5 M^-0.5 and v_max ~ (M/R)^0.5 with the BLR radius halved and the mass doubled). For each period of a grid we
fit A, B (and c) by generalized least squares; where K exceeds the cap we refit on the cap (scanning the phase). Returns
chi2 curves and summaries.
"""

import numpy as np
from scipy import stats

COM_MAX = 300.0
N_PHASE = 720
K_PHYS = 30000.0  # km/s: an unconstrained fit above this is a straight line in disguise (P much longer than the span)


def _gls(X, y, Vi):
    A = X.T @ Vi @ X
    beta = np.linalg.solve(A, X.T @ Vi @ y)
    r = y - X @ beta
    return beta, float(r @ Vi @ r)


def scan(tc, u, V, ref, vmax=np.nan, pmin=np.nan, v_ref=np.nan, s_ref=np.nan, periods=None, com_max=COM_MAX):
    """tc: cluster -> rest-frame time (yr); u: cluster -> velocity relative to ref; V: (a, b) -> covariance of the free
    clusters; com_max: the bound on the centre-of-mass velocity of the absolute fits (km/s). Returns dict with the constant,
    straight-line and sinusoid fits (relative; absolute when v_ref is finite)."""
    free = sorted(c for c in tc if c != ref)
    y = np.array([u[c] for c in free])
    Vm = np.array([[V[(a, b)] for b in free] for a in free])
    Vi = np.linalg.inv(Vm)
    t = np.array([tc[c] for c in free])
    t0 = tc[ref]
    out = dict(n_times=len(free) + 1, span_rest_yr=float(max(tc.values()) - min(tc.values())))
    out["chi2_const"] = float(y @ Vi @ y)
    x = (t - t0)[:, None]
    (a,), chi_lin = _gls(x, y, Vi)
    va = float(np.linalg.inv(x.T @ Vi @ x)[0, 0])
    out.update(accel=float(a), accel_err=float(np.sqrt(va)), chi2_lin=chi_lin, dof_lin=len(free) - 1)
    if periods is None:
        periods = np.logspace(0, 4, 1200)
    absolute = np.isfinite(v_ref) and np.isfinite(s_ref)
    if absolute:
        ya = np.append(y, v_ref)
        Va = np.zeros((len(free) + 1, len(free) + 1))
        Va[:-1, :-1] = Vm
        Va[-1, -1] = s_ref**2
        Vai = np.linalg.inv(Va)
    rel_free, rel_mass, abs_free, abs_mass, Ks = [], [], [], [], []
    phis = np.linspace(0, 2 * np.pi, N_PHASE, endpoint=False)
    for P in periods:
        w = 2 * np.pi / P
        X = np.column_stack([np.sin(w * t) - np.sin(w * t0), np.cos(w * t) - np.cos(w * t0)])
        beta, chi = _gls(X, y, Vi)
        K = float(np.hypot(*beta))
        cap = (
            vmax * (P / pmin) ** (-1.0 / 3.0)
            if (np.isfinite(vmax) and np.isfinite(pmin) and P >= pmin)
            else (np.inf if not np.isfinite(vmax) else 0.0)
        )
        rel_free.append(chi if K <= K_PHYS else np.inf)
        Ks.append(K)
        if K <= cap:
            rel_mass.append(chi)
        elif cap > 0:
            mod = cap * (np.outer(np.cos(phis), X[:, 0]) + np.outer(np.sin(phis), X[:, 1]))
            r = y[None, :] - mod
            rel_mass.append(float(np.min(np.einsum("ij,jk,ik->i", r, Vi, r))))
        else:
            rel_mass.append(np.inf)
        if absolute:
            Xa = np.zeros((len(free) + 1, 3))
            Xa[:-1, :2] = X
            Xa[-1, :] = [np.sin(w * t0), np.cos(w * t0), 1.0]

            def fit_abs(K_fixed=None, Xa=Xa):
                if K_fixed is None:
                    b, c2 = _gls(Xa, ya, Vai)
                    if abs(b[2]) <= com_max:
                        return c2, float(np.hypot(b[0], b[1]))
                    best = np.inf
                    for cfix in (-com_max, com_max):
                        yy = ya - cfix * Xa[:, 2]
                        bb, c3 = _gls(Xa[:, :2], yy, Vai)
                        best = min(best, c3)
                    return best, np.nan
                # on the cap: phase grid, c fitted (bounded) for each phase
                best = np.inf
                for ph in phis:
                    m = K_fixed * (np.cos(ph) * Xa[:, 0] + np.sin(ph) * Xa[:, 1])
                    rr = ya - m
                    e = Xa[:, 2]
                    c_hat = float((e @ Vai @ rr) / (e @ Vai @ e))
                    c_hat = min(max(c_hat, -com_max), com_max)
                    res = rr - c_hat * e
                    best = min(best, float(res @ Vai @ res))
                return best, K_fixed

            chi_a, K_a = fit_abs()
            abs_free.append(chi_a)
            if np.isfinite(K_a) and K_a <= cap:
                abs_mass.append(chi_a)
            elif cap > 0 and np.isfinite(cap):
                abs_mass.append(
                    min(fit_abs(K_fixed=cap)[0], chi_a if (np.isfinite(K_a) and K_a <= cap) else np.inf)
                )
            elif not np.isfinite(cap):
                abs_mass.append(chi_a)
            else:
                abs_mass.append(np.inf)
    rel_free, rel_mass, Ks = np.array(rel_free), np.array(rel_mass), np.array(Ks)

    def summary(chis, n_data, n_par):
        if not np.isfinite(chis).any():
            return dict(chi2=np.inf)
        i = int(np.nanargmin(chis))
        ok = periods[chis <= chis[i] + 2.3]
        dof = n_data - n_par
        return dict(
            chi2=float(chis[i]),
            P=float(periods[i]),
            P_lo=float(ok.min()),
            P_hi=float(ok.max()),
            dof=dof,
            p=float(stats.chi2.sf(chis[i], dof)) if dof > 0 else np.nan,
        )

    out["rel_free"] = summary(rel_free, len(free), 2)
    out["rel_free"]["K"] = float(Ks[int(np.argmin(rel_free))])
    out["rel_mass"] = summary(rel_mass, len(free), 2)
    if absolute:
        out["abs_free"] = summary(np.array(abs_free), len(free) + 1, 3)
        out["abs_mass"] = summary(np.array(abs_mass), len(free) + 1, 3)
    out["curves"] = dict(
        periods=periods,
        rel_free=rel_free,
        rel_mass=rel_mass,
        abs_free=np.array(abs_free) if absolute else None,
        abs_mass=np.array(abs_mass) if absolute else None,
        K=Ks,
    )
    return out


def best_abs(tc, u, V, ref, P, vmax, pmin, v_ref, s_ref, com_max=COM_MAX):
    """The absolute fit of scan() at one period P (the centre of mass c within +-com_max of the narrow lines, the amplitude on
    or below the cap): returns (chi2, A, B, c) with v(t) = A sin(2 pi t / P) + B cos(2 pi t / P) + c on the offset scale."""
    free = sorted(cc for cc in tc if cc != ref)
    y = np.array([u[cc] for cc in free])
    Vm = np.array([[V[(a, b)] for b in free] for a in free])
    t = np.array([tc[cc] for cc in free])
    t0 = tc[ref]
    ya = np.append(y, v_ref)
    Va = np.zeros((len(free) + 1, len(free) + 1))
    Va[:-1, :-1] = Vm
    Va[-1, -1] = s_ref**2
    Vai = np.linalg.inv(Va)
    w = 2 * np.pi / P
    Xa = np.zeros((len(free) + 1, 3))
    Xa[:-1, 0], Xa[:-1, 1] = np.sin(w * t) - np.sin(w * t0), np.cos(w * t) - np.cos(w * t0)
    Xa[-1, :] = [np.sin(w * t0), np.cos(w * t0), 1.0]
    cap = (
        vmax * (P / pmin) ** (-1.0 / 3.0)
        if (np.isfinite(vmax) and np.isfinite(pmin) and P >= pmin)
        else (np.inf if not np.isfinite(vmax) else 0.0)
    )
    best = (np.inf, np.nan, np.nan, np.nan)
    b, c2 = _gls(Xa, ya, Vai)
    if abs(b[2]) <= com_max and np.hypot(b[0], b[1]) <= cap:
        return c2, float(b[0]), float(b[1]), float(b[2])
    if np.isfinite(cap) and cap > 0:
        for cfix in (-com_max, com_max):  # c on its bound, amplitude free (then capped below)
            bb, c3 = _gls(Xa[:, :2], ya - cfix * Xa[:, 2], Vai)
            if np.hypot(*bb) <= cap and c3 < best[0]:
                best = (c3, float(bb[0]), float(bb[1]), cfix)
        e = Xa[:, 2]  # amplitude on the cap, c fitted within its bounds, every phase at once
        ph = np.linspace(0, 2 * np.pi, N_PHASE, endpoint=False)
        A, B = cap * np.cos(ph), cap * np.sin(ph)
        rr = ya[None, :] - np.outer(A, Xa[:, 0]) - np.outer(B, Xa[:, 1])
        c_hat = np.clip((rr @ Vai @ e) / float(e @ Vai @ e), -com_max, com_max)
        res = rr - np.outer(c_hat, e)
        c4 = np.einsum("ij,jk,ik->i", res, Vai, res)
        k = int(np.argmin(c4))
        if c4[k] < best[0]:
            best = (float(c4[k]), float(A[k]), float(B[k]), float(c_hat[k]))
    elif not np.isfinite(cap):
        for cfix in (-com_max, com_max):
            bb, c3 = _gls(Xa[:, :2], ya - cfix * Xa[:, 2], Vai)
            if c3 < best[0]:
                best = (c3, float(bb[0]), float(bb[1]), cfix)
    return best


def best_curve(tc, u, V, ref, P, vmax=np.nan, pmin=np.nan, v_ref=np.nan, s_ref=np.nan, n=400):
    """The absolute (if v_ref given) or relative best sinusoid at period P, as (t grid, v grid) for plotting; respects the
    mass cap the same way as scan()."""
    free = sorted(c for c in tc if c != ref)
    y = np.array([u[c] for c in free])
    Vm = np.array([[V[(a, b)] for b in free] for a in free])
    Vi = np.linalg.inv(Vm)
    t = np.array([tc[c] for c in free])
    t0 = tc[ref]
    w = 2 * np.pi / P
    cap = vmax * (P / pmin) ** (-1.0 / 3.0) if (np.isfinite(vmax) and P >= pmin) else np.inf
    X = np.column_stack([np.sin(w * t) - np.sin(w * t0), np.cos(w * t) - np.cos(w * t0)])
    beta, _ = _gls(X, y, Vi)
    if np.hypot(*beta) > cap:
        phis = np.linspace(0, 2 * np.pi, N_PHASE, endpoint=False)
        best, bb = np.inf, beta
        for ph in phis:
            b = cap * np.array([np.cos(ph), np.sin(ph)])
            r = y - X @ b
            c2 = float(r @ Vi @ r)
            if c2 < best:
                best, bb = c2, b
        beta = bb
    tg = np.linspace(min(tc.values()) - 2, max(tc.values()) + 2, n)
    g = beta[0] * np.sin(w * tg) + beta[1] * np.cos(w * tg)
    g_ref = beta[0] * np.sin(w * t0) + beta[1] * np.cos(w * t0)
    base = (v_ref - g_ref) if np.isfinite(v_ref) else -g_ref
    return tg, g + base
