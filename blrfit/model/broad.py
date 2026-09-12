"""
The broad line: one to three Gaussians, their bounds, the width-offset hinge,
the coverage rule and the choice of the number of components.

Individual broad components carry no physical meaning; only their sum is
measured (``measure.py``). The number of components is chosen with the
Bayesian information criterion, BIC = chi^2 + k ln N: the model with the fewest
components is kept unless a more complex one improves the BIC by more than
``DBIC``.
"""
from __future__ import annotations

import numpy as np

from ..constants import (S2F, LAM, V_BROAD_MAX, SIG_BROAD_MIN, SIG_BROAD_MAX,
                         BROAD_WIDTH_SLOPE, BROAD_WIDTH_PENALTY_SCALE, BROAD_STARTS_KMS,
                         CORE_COVER_KMS, EDGE_MARGIN_KMS)
from .narrow import amp_guess


def add_broad_block(ps, comps, wave, fsub, lam0, prefix, n_broad, amax, vb_lo, vb_hi,
                    sig_broad_min=SIG_BROAD_MIN):
    """``n_broad`` Gaussians on ``lam0``: amplitude, velocity and width each."""
    for k in range(n_broad):
        a = amp_guess(wave, fsub, lam0) * (0.6 if n_broad > 1 else 0.8) / max(k + 1, 1)
        v0 = 0.0 if k == 0 else (1500.0 if k == 1 else -1500.0)
        v0 = float(np.clip(v0, vb_lo + 1, vb_hi - 1))
        s0 = 1500.0 if k == 0 else 2500.0
        ps.add(f"{prefix}_b{k}_A", a, 0.0, amax)
        ps.add(f"{prefix}_b{k}_v", v0, vb_lo, vb_hi)
        ps.add(f"{prefix}_b{k}_sig", max(s0, 1.2 * sig_broad_min), sig_broad_min, SIG_BROAD_MAX)
        comps.append((f"{prefix}_broad{k}", lam0, f"{prefix}_b{k}_A",
                      f"{prefix}_b{k}_v", f"{prefix}_b{k}_sig", "broad", None))


def add_mgii_doublet_block(ps, comps, wave, fsub, n_broad, amax, vb_lo, vb_hi,
                           sig_broad_min=SIG_BROAD_MIN):
    """Broad Mg II as a 2796/2803 doublet with a fixed 1:1 ratio (optional)."""
    for k in range(n_broad):
        a = amp_guess(wave, fsub, LAM["MgII"], 10.0) * 0.4 / max(k + 1, 1)
        ps.add(f"Mg_b{k}_A", a, 0.0, amax)
        ps.add(f"Mg_b{k}_v", float(np.clip(0.0 if k == 0 else 1500.0 * (-1) ** k, vb_lo + 1, vb_hi - 1)), vb_lo, vb_hi)
        ps.add(f"Mg_b{k}_sig", max(1500.0 if k == 0 else 2500.0, 1.2 * sig_broad_min), sig_broad_min, SIG_BROAD_MAX)
        comps.append((f"Mg_broad{k}_2796", LAM["MgII2796"], f"Mg_b{k}_A", f"Mg_b{k}_v",
                      f"Mg_b{k}_sig", "broad", None))
        comps.append((f"Mg_broad{k}_2803", LAM["MgII2803"], f"Mg_b{k}_A", f"Mg_b{k}_v",
                      f"Mg_b{k}_sig", "broad", (f"Mg_b{k}_A", 1.0)))


def velocity_bounds(v_bounds):
    """Bounds of the broad-component centres: the requested range, never narrower
    than +/-1500 km/s."""
    vb_lo, vb_hi = (-V_BROAD_MAX, V_BROAD_MAX) if v_bounds is None else v_bounds
    return min(vb_lo, -1500.0), max(vb_hi, 1500.0)


def core_covered(v_lo, v_hi):
    """The complex is fitted only if the data reach CORE_COVER_KMS on both sides of the line."""
    return v_lo <= -CORE_COVER_KMS and v_hi >= CORE_COVER_KMS


def covered_velocity_bounds(v_lo, v_hi):
    """Broad centres confined to the covered range less a margin, and to +/- V_BROAD_MAX."""
    return (max(-V_BROAD_MAX, v_lo + EDGE_MARGIN_KMS), min(V_BROAD_MAX, v_hi - EDGE_MARGIN_KMS))


def broad_parameter_pairs(comps):
    """Unique (velocity, width) parameter names of the broad components."""
    pairs = [(vn, sn) for lab, l0, an, vn, sn, kind, rr in comps if kind == "broad"]
    return list(dict.fromkeys(pairs))


WIDTH_SLOPE_SIGMA = BROAD_WIDTH_SLOPE / S2F      # minimum sigma per km/s of |v|


def width_offset_penalty(d, broad_pairs):
    """Hinge: sigma >= (BROAD_WIDTH_SLOPE / S2F) |v| for every broad component."""
    return [max(0.0, WIDTH_SLOPE_SIGMA * abs(d[vn]) - d[sn]) / BROAD_WIDTH_PENALTY_SCALE
            for vn, sn in broad_pairs]


def first_component_starts(vlo_b, vhi_b, n_broad):
    """Starting velocities of the first broad component, clipped to the bounds and
    made unique. An offset line is exactly the case in which a zero-velocity
    start can settle in a local minimum."""
    starts = list(BROAD_STARTS_KMS) if n_broad >= 1 else [0.0]
    starts = [float(np.clip(v, vlo_b + 1, vhi_b - 1)) for v in starts]
    return list(dict.fromkeys(starts))


def select_by_bic(bics, dbic):
    """Index of the simplest model within ``dbic`` of the best BIC."""
    bics = np.asarray(bics, float)
    best = int(np.argmin(bics))
    chosen = best
    for i in range(best):
        if bics[i] - bics[best] < dbic:
            chosen = i; break
    return chosen
