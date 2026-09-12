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
        """Starting vector of the free parameters, kept strictly inside the bounds."""
        v = dict(zip(self.names, self.val))
        return np.array([np.clip(v[n], self.lb[self.names.index(n)] + 1e-9,
                                 self.ub[self.names.index(n)] - 1e-9) for n in self.free_names])

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
