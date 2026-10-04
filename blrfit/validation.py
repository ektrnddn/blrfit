"""Recorded tolerance-region decisions for scientific validation.

The caller supplies an interval constructed with the appropriate independent
units. This module does not turn a fit diagnostic into calibrated uncertainty.
"""
import math


def tolerance_decision(estimate, interval, region, *, confidence=0.95,
                       units, n_observations, n_groups, method):
    """Classify an interval as inside, outside, or overlapping a closed region.

    Exact boundary points are inside. Missing/nonfinite estimates or interval
    endpoints give ``inconclusive``; invalid definitions raise ValueError.
    Counts describe the evidence and must not count repeated observations as
    independent groups. The interval method and its assumptions belong to the
    caller and are retained in the returned record.
    """
    low, high = map(float, region)
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError("tolerance region must have ordered finite endpoints")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    for name, count in (("n_observations", n_observations), ("n_groups", n_groups)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if n_groups > n_observations:
        raise ValueError("group count cannot exceed observation count")
    if not units or not method:
        raise ValueError("units and interval method must be recorded")

    def finite(value):
        return float(value) if value is not None and math.isfinite(float(value)) else None

    value = finite(estimate)
    bounds = [None, None] if interval is None else [finite(x) for x in interval]
    if len(bounds) != 2:
        raise ValueError("interval must have two endpoints")
    result = dict(estimate=value, interval=bounds, confidence=float(confidence),
                  tolerance_region=[low, high], units=units,
                  n_observations=n_observations, n_groups=n_groups, method=method)
    if value is None or None in bounds or not n_observations or not n_groups:
        outcome, reason = "inconclusive", "insufficient finite interval evidence"
    else:
        left, right = bounds
        if left > right:
            raise ValueError("interval endpoints are reversed")
        if left >= low and right <= high:
            outcome, reason = "pass", "whole interval inside tolerance region"
        elif right < low or left > high:
            outcome, reason = "fail", "whole interval outside tolerance region"
        else:
            outcome, reason = "inconclusive", "interval overlaps inside and outside"
    result.update(outcome=outcome, reason=reason)
    return result


def coverage_decision(estimate, interval, *, nominal, tolerance, **kwargs):
    """Apply the same decision to an inclusion fraction, not a compatibility test."""
    if not (0 < nominal < 1 and 0 <= tolerance <= min(nominal, 1 - nominal)):
        raise ValueError("coverage region must lie within [0, 1]")
    out = tolerance_decision(estimate, interval,
                             (nominal - tolerance, nominal + tolerance),
                             units="fraction", **kwargs)
    out.update(nominal=float(nominal), tolerance=float(tolerance))
    return out
