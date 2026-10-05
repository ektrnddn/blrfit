"""
Power-law slopes bluer than a thin accretion disc (f_lambda ~ lambda^-7/3):
the flag ``pl_unphysical`` and the warning of the luminosity functions.
"""

import numpy as np
import pytest

from blrfit.classify import classify
from blrfit.constants import PL_ALPHA_BLUE_LIMIT
from blrfit.physics import lambda_l_lambda
from test_degenerate import j001224_single_start  # noqa: F401  (fixture)


def test_flag_follows_the_limit(j001224_single_start):  # noqa: F811
    m = dict(j001224_single_start["meas"]["Hbeta"])
    for alpha, flagged in ((PL_ALPHA_BLUE_LIMIT - 0.01, True), (PL_ALPHA_BLUE_LIMIT, False), (-1.5, False)):
        m["pl_alpha"] = alpha
        assert ("pl_unphysical" in classify(m)["flags"]) is flagged, alpha


def test_luminosity_warns_for_an_unphysical_slope():
    import warnings

    with pytest.warns(RuntimeWarning, match="pl_unphysical"):
        lum = lambda_l_lambda(dict(pl_norm=50.0, pl_alpha=-3.9), 0.2)
    assert np.isfinite(lum) and lum > 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isfinite(lambda_l_lambda(dict(pl_norm=50.0, pl_alpha=-1.5), 0.2))
