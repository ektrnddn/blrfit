"""
The bound state of the Fe II widths (``continuum_info['fe_width_state']``).

A width is at a bound when the solver's active set holds it there or when it
lies within max(1 per cent of the bound's span, 10 km/s) of either bound: 88
km/s for the 1200-10000 km/s range, wider than the solver's own 1e-6
criterion, which misses widths that stopped a few tens of km/s short of a
bound on a flat chi-square. A zero Fe II norm is a separate boolean
(``fe_norm_zero``): the width then multiplies nothing, is unidentified, and is
never counted as a width at a bound. The states fixed, not covered,
unconstrained (zero norm), near-bound and solver-active are kept apart.

The constructed cases evaluate the rule on a parameter set without a solver;
the fitted cases check the same record on real spectra (the optical width of
spec-1704 ends on the 1200 km/s bound with the solver's active set, that of
the DESI example stops at 1259 km/s, inside the tolerance but not active, and
the Fe II norm of SDSS J001224 ends on zero).
"""

import numpy as np
import pytest

import blrfit
from blrfit.constants import FE_FWHM_MAX, FE_FWHM_MIN
from blrfit.io import read_desi, read_sdss
from blrfit.io.desi import read_redrock, redrock_sibling
from blrfit.model.continuum import (
    FE_NORM_ZERO,
    FE_WIDTH_BOUND_TOL_FRAC,
    FE_WIDTH_BOUND_TOL_KMS,
    FE_WIDTH_STATES,
    _add_pl_fe,
    fe_width_states,
    fit_continuum,
)
from blrfit.model.params import ParamSet
from conftest import DATA, DESI_EXAMPLE, DESI_TARGETID, SDSS_EXAMPLE, Z_J001224
from synth import make_spectrum

TOL = max(FE_WIDTH_BOUND_TOL_FRAC * (FE_FWHM_MAX - FE_FWHM_MIN), FE_WIDTH_BOUND_TOL_KMS)  # 88 km/s
SPEC_1704 = DATA + "/spec-1704-53178-0562.fits"
Z_1704 = 0.31839999556541443


def _params(feop_fwhm, feop_norm=0.5, feuv_fixed=False, feuv_fwhm=3000.0, feuv_norm=0.5):
    ps = ParamSet()
    _add_pl_fe(ps, 10.0, True, True, True, pl_start=10.0, feuv_fixed=feuv_fixed)
    d = ps.full(ps.p0())
    d.update(feop_fwhm=feop_fwhm, feop_norm=feop_norm, feuv_fwhm=feuv_fwhm, feuv_norm=feuv_norm)
    return ps, d


def _mask(ps, **active):
    """The solver's active mask over the free parameters (-1 lower, +1 upper, 0 free)."""
    return np.array([active.get(n, 0) for n in ps.free_names], int)


def test_tolerance_is_span_based():
    assert TOL == pytest.approx(88.0)
    assert TOL == FE_WIDTH_BOUND_TOL_FRAC * (FE_FWHM_MAX - FE_FWHM_MIN) > FE_WIDTH_BOUND_TOL_KMS


def test_interior_width_is_not_flagged():
    ps, d = _params(5000.0)
    s = fe_width_states(ps, d, _mask(ps))["feop_fwhm"]
    assert s["state"] == "interior" and not s["at_bound"] and not s["active"] and not s["near"]
    assert s["bound"] == "" and s["value"] == 5000.0 and s["tol_kms"] == pytest.approx(TOL)


@pytest.mark.parametrize("bound, flag", [(FE_FWHM_MIN, -1), (FE_FWHM_MAX, 1)])
def test_width_exactly_on_a_bound_is_flagged(bound, flag):
    ps, d = _params(bound)
    s = fe_width_states(ps, d, _mask(ps, feop_fwhm=flag))["feop_fwhm"]
    assert s["state"] == "solver_active" and s["at_bound"] and s["active"] and s["near"]
    assert s["bound"] == ("lower" if flag < 0 else "upper")
    # the same width without the active set is still at the bound, by the tolerance
    t = fe_width_states(ps, d, _mask(ps))["feop_fwhm"]
    assert t["state"] == "near_bound" and t["at_bound"] and not t["active"] and t["near"]


@pytest.mark.parametrize(
    "width, bound",
    [
        (FE_FWHM_MIN * 1.005, "lower"),
        (FE_FWHM_MIN + 0.5 * TOL, "lower"),
        (FE_FWHM_MAX * 0.995, "upper"),
        (FE_FWHM_MAX - 0.5 * TOL, "upper"),
    ],
)
def test_width_inside_the_tolerance_but_not_active_is_flagged(width, bound):
    ps, d = _params(width)
    s = fe_width_states(ps, d, _mask(ps))["feop_fwhm"]
    assert s["state"] == "near_bound" and s["at_bound"] and s["near"] and s["active"] is False
    assert s["bound"] == bound


@pytest.mark.parametrize("width", [FE_FWHM_MIN + 2 * TOL, FE_FWHM_MAX - 2 * TOL, 3000.0])
def test_width_outside_the_tolerance_is_not_flagged(width):
    ps, d = _params(width)
    s = fe_width_states(ps, d, _mask(ps))["feop_fwhm"]
    assert s["state"] == "interior" and not s["at_bound"]


def test_active_set_alone_flags_a_width_the_tolerance_would_not():
    # the solver may report a parameter active while its value sits away from
    # the bound (the active set is decided in the scaled variables)
    ps, d = _params(FE_FWHM_MIN + 5 * TOL)
    s = fe_width_states(ps, d, _mask(ps, feop_fwhm=-1))["feop_fwhm"]
    assert s["state"] == "solver_active" and s["at_bound"] and s["active"] and not s["near"]


@pytest.mark.parametrize("width, flag", [(FE_FWHM_MIN, -1), (FE_FWHM_MIN + 0.5 * TOL, 0), (5000.0, 0)])
def test_zero_norm_is_unconstrained_and_never_at_bound(width, flag):
    ps, d = _params(width, feop_norm=0.0)
    s = fe_width_states(ps, d, _mask(ps, feop_fwhm=flag, feop_norm=-1))["feop_fwhm"]
    assert s["state"] == "unconstrained" and s["norm_zero"] and not s["at_bound"]
    # the bound criteria are still reported, so the record can be recomputed
    assert s["active"] is bool(flag) and s["near"] is (width - FE_FWHM_MIN <= TOL)
    # a norm at the zero threshold counts as zero, one just above it does not
    assert fe_width_states(ps, dict(d, feop_norm=FE_NORM_ZERO), _mask(ps))["feop_fwhm"]["norm_zero"]
    assert not fe_width_states(ps, dict(d, feop_norm=2 * FE_NORM_ZERO), _mask(ps))["feop_fwhm"]["norm_zero"]


def test_fixed_width_is_fixed_not_at_bound():
    ps, d = _params(3000.0, feuv_fixed=True, feuv_fwhm=FE_FWHM_MIN)
    s = fe_width_states(ps, d, _mask(ps))["feuv_fwhm"]
    assert s["state"] == "fixed" and s["fixed"] and not s["at_bound"] and not s["near"] and not s["active"]
    # a fixed width whose norm is zero is unconstrained first
    t = fe_width_states(ps, dict(d, feuv_norm=0.0), _mask(ps))["feuv_fwhm"]
    assert t["state"] == "unconstrained" and t["fixed"] and t["norm_zero"]


def test_absent_template_is_not_covered():
    ps = ParamSet()
    _add_pl_fe(ps, 10.0, True, True, False, pl_start=10.0)
    d = ps.full(ps.p0())
    states = fe_width_states(ps, d, _mask(ps))
    assert states["feuv_fwhm"]["state"] == "not_covered" and np.isnan(states["feuv_fwhm"]["value"])
    assert not states["feuv_fwhm"]["at_bound"] and not states["feuv_fwhm"]["norm_zero"]
    assert states["feop_fwhm"]["state"] == "interior"
    assert set(states) == {"feop_fwhm", "feuv_fwhm"}
    assert all(s["state"] in FE_WIDTH_STATES for s in states.values())


# ----------------------------------------------------------------------------
# fitted spectra
# ----------------------------------------------------------------------------
def _states(info):
    return info["fe_width_state"]


def test_solver_active_width_of_spec_1704():
    sp = read_sdss(SPEC_1704)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_1704, complexes=("Hbeta",))
    info = res["continuum_info"]
    s = _states(info)["feop_fwhm"]
    assert "feop_fwhm" in info["at_bound"]
    assert (
        s["state"] == "solver_active"
        and s["at_bound"]
        and s["active"]
        and s["near"]
        and s["bound"] == "lower"
    )
    assert s["value"] == res["conti"]["feop_fwhm"] == pytest.approx(FE_FWHM_MIN, abs=1e-6)
    assert "feop_fwhm" in info["at_bound_widths"] and not info["fe_norm_zero"]["feop"]
    # the ultraviolet width is held (239 covered pixels): fixed, not at a bound
    u = _states(info)["feuv_fwhm"]
    assert info["feuv_fwhm_fixed"] and u["state"] == "fixed" and not u["at_bound"]
    assert info["at_bound_widths"] == ["feop_fwhm"]


def test_near_bound_width_of_the_desi_example_is_flagged_without_the_active_set():
    sp = read_desi(DESI_EXAMPLE, DESI_TARGETID, use_desispec=False)
    z = read_redrock(redrock_sibling(DESI_EXAMPLE), DESI_TARGETID)["z"]
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z, ebv=sp["ebv"], complexes=("Hbeta",))
    info = res["continuum_info"]
    s = _states(info)["feop_fwhm"]
    # 1259 km/s: 59 km/s above the lower bound, inside the 88 km/s tolerance,
    # not on the solver's active set and not on its own 1e-6 list
    assert FE_FWHM_MIN < s["value"] < FE_FWHM_MIN + TOL
    assert "feop_fwhm" not in info["at_bound"]
    assert s["state"] == "near_bound" and s["at_bound"] and s["near"] and s["active"] is False
    assert s["bound"] == "lower" and info["at_bound_widths"] == ["feop_fwhm"]
    assert res["host_info"]["applied"] and res["host_info"]["fe_width_state"] == info["fe_width_state"]


def test_zero_norm_of_j001224_is_fe_norm_zero_and_unidentified():
    sp = read_sdss(SDSS_EXAMPLE)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Hbeta",))
    info = res["continuum_info"]
    assert "feop_norm" in info["at_bound"] and res["conti"]["feop_norm"] <= FE_NORM_ZERO
    s = _states(info)["feop_fwhm"]
    assert info["fe_norm_zero"] == {"feop": True, "feuv": False}
    assert s["state"] == "unconstrained" and s["norm_zero"] and not s["at_bound"]
    assert info["at_bound_widths"] == []
    assert _states(info)["feuv_fwhm"]["state"] == "not_covered"


@pytest.mark.parametrize("width, bound", [(12000.0, "upper"), (800.0, "lower")])
def test_width_driven_onto_a_bound_by_the_data(width, bound):
    from blrfit.model.continuum import fe_templates

    z = 0.2
    s = make_spectrum(z=z, snr=20.0, seed=3, narrow=dict(ew_ha=0.0))
    wr = s["wave"] / (1 + z)
    s["flux"] = s["flux"] + fe_templates()[0](wr, 0.6, width, 0.0)
    d, model, info = fit_continuum(wr, s["flux"] * (1 + z), s["ivar"] / (1 + z) ** 2)
    st = _states(info)["feop_fwhm"]
    assert st["at_bound"] and st["bound"] == bound and st["state"] in ("solver_active", "near_bound")
    assert "feop_fwhm" in info["at_bound_widths"] and not st["norm_zero"]


def test_width_recovered_inside_the_range_is_interior():
    from blrfit.model.continuum import fe_templates

    z = 0.2
    s = make_spectrum(z=z, snr=20.0, seed=3, narrow=dict(ew_ha=0.0))
    wr = s["wave"] / (1 + z)
    s["flux"] = s["flux"] + fe_templates()[0](wr, 0.6, 2500.0, 0.0)
    d, model, info = fit_continuum(wr, s["flux"] * (1 + z), s["ivar"] / (1 + z) ** 2)
    st = _states(info)["feop_fwhm"]
    assert st["state"] == "interior" and not st["at_bound"] and info["at_bound_widths"] == []
    assert d["feop_fwhm"] == pytest.approx(2500.0, rel=0.25)
