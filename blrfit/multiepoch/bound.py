"""The orbital bound on a measured change: for a circular binary in which the active black hole of virial mass M_act has a
companion M_act/q, the line-of-sight velocity of the active hole changes between two epochs
Δt apart by at most 2 v(P) |sin(π Δt / P)|, with v(P) = v_max (P / P_min)^(-1/3) for P ≥ P_min,
where P_min and v_max belong to the smallest separation that keeps the broad-line region inside
the active hole's Roche lobe (bh_masses.py: VMAX_Q01 / PMIN_YR_Q01 for q = 0.1, VMAX_Q1 / PMIN_YR_Q1
for q = 1; i = 90°). The permissive bound uses q = 0.1 (the heavier companion makes the active
hole move fastest). A change above the permissive bound at 2σ cannot be orbital motion of the
active hole at that mass; otherwise P_max is the longest period that still allows it.
"""

import numpy as np

P_GRID = np.logspace(0, 5, 2000)  # years, relative to P_min: 1 ... 1e5 P_min


def max_change(vmax, pmin, dt_yr, pgrid=P_GRID):
    """(largest possible |Δv| over dt for P >= P_min, the P at which it occurs)."""
    if not (
        np.isfinite(vmax) and np.isfinite(pmin) and np.isfinite(dt_yr) and vmax > 0 and pmin > 0 and dt_yr > 0
    ):
        return np.nan, np.nan
    P = pmin * pgrid
    dv = 2.0 * vmax * (P / pmin) ** (-1.0 / 3.0) * np.abs(np.sin(np.pi * dt_yr / P))
    k = int(np.argmax(dv))
    return float(dv[k]), float(P[k])


def period_limit(vmax, pmin, dt_yr, s_abs, pgrid=P_GRID):
    """The longest period P_max (yr) at which a change |s| is still possible, or NaN when none is."""
    if not (np.isfinite(s_abs) and s_abs >= 0):
        return np.nan
    P = pmin * pgrid
    dv = 2.0 * vmax * (P / pmin) ** (-1.0 / 3.0) * np.abs(np.sin(np.pi * dt_yr / P))
    ok = dv >= s_abs
    return float(P[ok].max()) if ok.any() else np.nan


def screen(rec, s, err, dt_yr, nsig=2.0):
    """rec: a mass record (VMAX_Q01, PMIN_YR_Q01, VMAX_Q1, PMIN_YR_Q1, LOGMBH).
    Returns dict(orbital_ok, dv_max_q01, dv_max_q1, p_max_q01_yr, p_max_q1_yr, p_min_q1_yr, logmbh)."""
    s_lo = max(abs(s) - nsig * err, 0.0)
    out = dict(logmbh=float(rec.get("LOGMBH", np.nan)))
    for tag in ("Q01", "Q1"):
        vmax, pmin = float(rec.get(f"VMAX_{tag}", np.nan)), float(rec.get(f"PMIN_YR_{tag}", np.nan))
        dvmax, _ = max_change(vmax, pmin, dt_yr)
        out[f"dv_max_{tag.lower()}"] = dvmax
        out[f"p_max_{tag.lower()}_yr"] = (
            period_limit(vmax, pmin, dt_yr, s_lo) if np.isfinite(dvmax) else np.nan
        )
        out[f"p_min_{tag.lower()}_yr"] = pmin
    out["orbital_ok"] = (
        bool(np.isfinite(out["dv_max_q01"]) and s_lo <= out["dv_max_q01"])
        if np.isfinite(out["dv_max_q01"])
        else None
    )
    return out
