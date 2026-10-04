"""Starting values handed to the solver: always inside the bounds, a clear
error for bounds that cannot hold one, and a clear error for a spectrum without
a usable pixel (the case that produced such bounds)."""
import numpy as np
import pytest
from scipy.optimize import least_squares

from blrfit.model.params import ParamSet


def _ps(**spec):
    ps = ParamSet()
    for name, (val, lb, ub) in spec.items():
        ps.add(name, val, lb, ub)
    return ps


def test_ordinary_start_is_the_value_kept_1e9_inside():
    ps = _ps(a=(0.0, 0.0, 10.0), b=(5.0, 0.0, 10.0), c=(12.0, 0.0, 10.0))
    assert np.array_equal(ps.p0(), np.clip([0.0, 5.0, 12.0], 1e-9, 10.0 - 1e-9))


@pytest.mark.parametrize("val, lb, ub, expect", [
    (np.nan, 2.0, 4.0, 3.0),
    (np.nan, 2.0, np.inf, 2.0 + 1e-9),
    (np.nan, -np.inf, 4.0, 4.0 - 1e-9),
    (np.nan, -np.inf, np.inf, 0.0),
    (np.inf, 2.0, np.inf, 2.0 + 1e-9),
    (-np.inf, -np.inf, 4.0, 4.0 - 1e-9),
])
def test_unusable_start_is_replaced_inside_the_bounds_with_a_warning(val, lb, ub, expect):
    ps = _ps(a=(val, lb, ub))
    with pytest.warns(RuntimeWarning, match="parameter a"):
        x0 = ps.p0()
    assert x0[0] == expect


@pytest.mark.parametrize("val, expect", [(np.inf, 4.0 - 1e-9), (-np.inf, 2.0 + 1e-9)])
def test_infinite_start_with_a_finite_bound_is_clipped_as_before(val, expect):
    import warnings
    ps = _ps(a=(val, 2.0, 4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ps.p0()[0] == expect == np.clip(val, 2.0 + 1e-9, 4.0 - 1e-9)


def test_narrow_interval_keeps_the_start_inside():
    ps = _ps(a=(1.5e-12, 1e-12, 2e-12), b=(0.0, 1e-12, 2e-12))
    x0 = ps.p0()
    lo, hi = ps.bounds()
    assert np.all(x0 > lo) and np.all(x0 < hi)


def test_interval_of_1e9_or_more_keeps_the_old_start():
    for width in (1e-9, 1.5e-9, 3e-9, 1.0):
        ps = _ps(a=(-1.0, 0.0, width), b=(5.0, 0.0, width))
        assert np.array_equal(ps.p0(), np.clip([-1.0, 5.0], 1e-9, width - 1e-9))


@pytest.mark.parametrize("lb, ub", [(np.nan, 1.0), (0.0, np.nan), (1.0, 1.0), (2.0, 1.0)])
def test_invalid_bounds_name_the_parameter(lb, ub):
    ps = _ps(width=(1.0, lb, ub))
    with pytest.raises(ValueError, match="parameter width"):
        ps.p0()


def test_fixed_and_tied_parameters_do_not_need_valid_bounds():
    ps = ParamSet()
    ps.add("a", 1.0, 0.0, 2.0)
    ps.add("b", 1.0, np.nan, np.nan, fixed=True)
    ps.add("c", 1.0, 0.0, 2.0)
    ps.link("c", "a")
    assert np.array_equal(ps.p0(), [1.0])


def test_solver_accepts_a_start_that_was_not_finite():
    ps = _ps(a=(np.nan, -5.0, 5.0), b=(np.nan, 0.0, 10.0))
    with pytest.warns(RuntimeWarning):
        x0 = ps.p0()
    x = np.linspace(0.0, 1.0, 20)
    y = 2.0 + 3.0 * x

    def resid(p):
        d = ps.full(p)
        return d["a"] + d["b"] * x - y

    sol = least_squares(resid, x0, bounds=ps.bounds())
    assert sol.success and np.allclose(sol.x, [2.0, 3.0], atol=1e-6)


@pytest.mark.parametrize("fill", [0.0, np.nan, -1.0])
def test_spectrum_without_a_usable_pixel_is_refused_by_name(fill):
    """Every inverse variance zero, non-finite or negative (a masked fibre): the
    continuum reference flux was the median of an empty set, which made the
    power-law bound NaN and the solver refuse the start. The fit now stops
    before the continuum and says why."""
    from blrfit.model.continuum import fit_continuum
    from blrfit.model.fit import fit_spectrum
    wave = np.linspace(3600.0, 9800.0, 4000)
    flux = np.ones_like(wave)
    ivar = np.full_like(wave, fill)
    with pytest.raises(ValueError, match="no usable pixel"):
        fit_spectrum(wave, flux, ivar, 0.2)
    with pytest.raises(ValueError, match="no usable pixel"):
        fit_continuum(wave / 1.2, flux, ivar)
