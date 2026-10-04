"""
Parameter bookkeeping for the least-squares fits, and the Gaussian line shape.

Every fit in the package (continuum, host, line complexes) is a bounded
least-squares problem on a named parameter vector. ``ParamSet`` keeps the
names, starting values and bounds, and knows which parameters are fixed, which
are tied to another parameter and which are derived as products of others, so
that the optimiser only sees the free ones. The order in which parameters are
added is the order of the free vector; it is part of the frozen model because
the trust-region solver is not exactly invariant to it.
"""
from __future__ import annotations

import warnings

import numpy as np

from ..constants import C_KMS


class ParamSet:
    """A named parameter vector with bounds, fixed entries, ties and products.

    ``link(name, other)`` forces ``name`` to equal ``other`` at all times;
    ``derive(name, factors)`` defines ``name`` as the product of the listed
    parameters and numbers. Fixed, tied and derived parameters are removed from
    the vector the optimiser sees.
    """

    def __init__(self):
        self.names, self.val, self.lb, self.ub = [], [], [], []
        self.fixed, self.tie = {}, {}
        self.derived = {}          # name -> list of factors (parameter names or floats)

    def derive(self, name, factors):
        self.derived[name] = list(factors)

    def add(self, name, val, lb, ub, fixed=False):
        if name in self.names:
            raise ValueError(f"duplicate parameter {name}")
        self.names.append(name); self.val.append(float(val))
        self.lb.append(float(lb)); self.ub.append(float(ub))
        if fixed:
            self.fixed[name] = float(val)

    def link(self, name, source):
        self.tie[name] = source

    @property
    def free_names(self):
        return [n for n in self.names if n not in self.fixed and n not in self.tie]

    def p0(self):
        """Starting vector of the free parameters, kept strictly inside the bounds.

        A start is clipped as np.clip(x, lo + 1e-9, hi - 1e-9), as before, with a
        margin of a quarter of the interval when that is narrower than 1e-9. A
        NaN start, or an infinite one on the
        side of an infinite bound, is replaced by the middle of the bounds (the
        finite bound when only one is finite, zero when neither is), with a
        RuntimeWarning naming the parameter. Bounds of a free parameter that are
        not an increasing pair of numbers raise ValueError naming it."""
        v = dict(zip(self.names, self.val))
        out = []
        for n in self.free_names:
            i = self.names.index(n)
            lo, hi, x = self.lb[i], self.ub[i], v[n]
            if np.isnan(lo) or np.isnan(hi) or not lo < hi:
                raise ValueError(f"parameter {n}: bounds [{lo}, {hi}] are not an increasing pair of numbers")
            eps = 1e-9 if hi - lo >= 1e-9 else 0.25 * (hi - lo)
            start = np.nan if np.isnan(x) else np.clip(x, lo + eps, hi - eps)
            if not np.isfinite(start):
                mid = (0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi)
                       else (lo if np.isfinite(lo) else (hi if np.isfinite(hi) else 0.0)))
                start = np.clip(mid, lo + eps, hi - eps)
                warnings.warn(f"parameter {n}: start {x} replaced by {float(start):g}, inside its bounds [{lo}, {hi}]",
                              RuntimeWarning, stacklevel=2)
            out.append(start)
        return np.array(out)

    def bounds(self):
        idx = [self.names.index(n) for n in self.free_names]
        return (np.array([self.lb[i] for i in idx]), np.array([self.ub[i] for i in idx]))

    def full(self, pfree):
        """Dictionary of every parameter (free, fixed, tied and derived) for a free vector."""
        d = dict(zip(self.names, self.val))
        d.update(self.fixed)
        d.update(dict(zip(self.free_names, pfree)))
        for _ in range(3):                          # resolve chained ties
            for n, s in self.tie.items():
                d[n] = d[s]
        for n, facs in self.derived.items():
            val = 1.0
            for f in facs:
                val *= d[f] if isinstance(f, str) else f
            d[n] = val
        return d

    def set_values(self, d):
        for i, n in enumerate(self.names):
            if n in d:
                self.val[i] = float(d[n])


def gauss_lam(wave, A, lam0, v_kms, sig_kms):
    """Gaussian in wavelength with peak ``A`` at lam0 (1 + v/c) and width
    sigma_lambda = lambda_c sigma / c. Zero if the amplitude or width is not positive."""
    if A <= 0 or sig_kms <= 0:
        return np.zeros_like(wave)
    lc = lam0 * (1.0 + v_kms / C_KMS)
    sl = lc * sig_kms / C_KMS
    return A * np.exp(-0.5 * ((wave - lc) / sl) ** 2)
