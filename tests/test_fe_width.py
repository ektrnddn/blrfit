"""
The Fe II widths of the continuum fit: joint recovery next to a broad line,
the at-bound and fixed-width diagnostics, the ultraviolet width policy and
the reproducibility of the fit under a rescaling of the input.

The optical width is recovered from a spectrum of power law + optical Fe II +
a broad Hbeta at +1000 km/s (z = 0.2 on the DESI grid, continuum S/N 20, Fe II
at 0.6 template units, about a third of the continuum at 4600 A). Measured:
true 2500 km/s -> 2508 km/s, true 6000 km/s -> 6579 km/s (a broad template is
partly degenerate with the power law inside the finite windows, so the wide
case is recovered less well); Hbeta c50_sys 999.7 and 1007.7 km/s for a truth
of 1000.0 km/s, class A in both cases. Across seeds 1-5 the widths land within
9 per cent (2500) and within 10 per cent (6000) of the truth and c50_sys
within 23 km/s.

The rescaling test multiplies the flux by (1 + 1e-13) and divides the inverse
variance by the square: a change far below the noise. In 0.1.0 the Fe width
was evaluated on a 50 km/s grid and the finite-difference derivative of the
width vanished; on spec-1592 (Fe II at 1.4 template units, Hbeta only) that
release moved the fitted width from 2219 to 2575 km/s and Hbeta c50_sys from
2490 to 2836 km/s under this change. With the continuous operator the width
moves by 9e-6 km/s and c50_sys by 1e-3 km/s. The example spectrum of SDSS
J001224.01-102226.5 has no Fe II (its norm ends on zero): its width is
unconstrained by construction, and only the velocity is checked for it.
"""
import os

import pytest

import blrfit
from blrfit.constants import FE_FWHM_MAX, FE_FWHM_MIN, FE_UV_FREE_MIN_PIXELS, FE_UV_FWHM_FIXED_KMS
from blrfit.io import read_desi, read_sdss
from blrfit.io.desi import read_redrock, redrock_sibling
from blrfit.model import continuum
from blrfit.model.continuum import _add_pl_fe, fe_templates, fit_continuum, fit_continuum_host
from blrfit.model.params import ParamSet
from conftest import DATA, DESI_EXAMPLE, DESI_TARGETID, SDSS_EXAMPLE, Z_J001224
from synth import make_spectrum

Z_SYNTH = 0.2
FE_NORM = 0.6
EPS = 1e-13
FE_STRONG = os.path.join(DATA, "spec-1592-52990-0139.fits")   # Fe II at 1.4 template units
Z_FE_STRONG = 0.4519999921321869


def _joint_spectrum(width, seed=3):
    s = make_spectrum(z=Z_SYNTH, snr=20.0, seed=seed,
                      broad=[dict(line="Hbeta", v=1000.0, fwhm=4000.0, ew=60.0)])
    s["flux"] = s["flux"] + fe_templates()[0](s["wave"] / (1 + Z_SYNTH), FE_NORM, width, 0.0)
    return s


@pytest.mark.parametrize("width", [2500.0, 6000.0])
def test_joint_fe_width_and_hbeta_offset(width):
    s = _joint_spectrum(width)
    res = blrfit.fit_spectrum(s["wave"], s["flux"], s["ivar"], Z_SYNTH, host=False, complexes=("Hbeta",))
    row = blrfit.summary_row(res)
    assert res["conti"]["feop_fwhm"] == pytest.approx(width, rel=0.25)
    assert abs(row["HB_c50_sys"] - s["truth"]["broad"]["Hbeta"]["c50_sys"]) < 60.0
    assert "feop_fwhm" not in res["continuum_info"]["at_bound"]


def test_at_bound_and_fixed_width_recorded_for_desi_example():
    sp = read_desi(DESI_EXAMPLE, DESI_TARGETID, use_desispec=False)
    z = read_redrock(redrock_sibling(DESI_EXAMPLE), DESI_TARGETID)["z"]
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z, ebv=sp["ebv"], complexes=("Hbeta",))
    info = res["continuum_info"]
    assert isinstance(info["at_bound"], list)
    assert all(isinstance(n, str) for n in info["at_bound"])
    assert isinstance(info["feuv_fwhm_fixed"], bool)
    assert set(info["fe_widths"]) == {"feop_fwhm", "feuv_fwhm"}
    # 214 pixels of the ultraviolet windows at z = 0.22: below the free threshold
    assert info["n_pix_uv"] < FE_UV_FREE_MIN_PIXELS
    assert info["feuv_fwhm_fixed"]
    assert res["conti"]["feuv_fwhm"] == FE_UV_FWHM_FIXED_KMS
    assert info["fe_widths"]["feuv_fwhm"] == FE_UV_FWHM_FIXED_KMS
    assert "feuv_fwhm" not in info["at_bound"]
    assert res["host_info"]["applied"] and res["host_info"]["feuv_fwhm_fixed"]


def test_fe_norm_at_zero_is_listed():
    sp = read_sdss(SDSS_EXAMPLE)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Hbeta",))
    info = res["continuum_info"]
    assert "feop_norm" in info["at_bound"]
    assert res["conti"]["feop_norm"] < 1e-6
    assert not info["fe_uv"] and not info["feuv_fwhm_fixed"] and info["n_pix_uv"] == 0
    assert list(info["fe_widths"]) == ["feop_fwhm"]


@pytest.mark.parametrize("width, bound", [(12000.0, FE_FWHM_MAX), (800.0, FE_FWHM_MIN)])
def test_width_outside_the_range_ends_on_the_bound_and_is_listed(width, bound):
    # a template broader or narrower than the allowed range must end on the
    # bound, in the plain fit and in the joint fit alike
    s = make_spectrum(z=Z_SYNTH, snr=20.0, seed=3, narrow=dict(ew_ha=0.0))
    wr = s["wave"] / (1 + Z_SYNTH)
    s["flux"] = s["flux"] + fe_templates()[0](wr, FE_NORM, width, 0.0)
    fr, ir = s["flux"] * (1 + Z_SYNTH), s["ivar"] / (1 + Z_SYNTH) ** 2
    for fitter in (fit_continuum, fit_continuum_host):
        out = fitter(wr, fr, ir)
        d, info = out[0], out[-1]
        assert d["feop_fwhm"] == pytest.approx(bound, abs=1e-3)
        assert "feop_fwhm" in info["at_bound"]
        assert info["fe_widths"]["feop_fwhm"] == d["feop_fwhm"]


@pytest.mark.parametrize("z, fixed", [(0.2, True), (0.5, False)])
def test_uv_width_fixed_only_where_the_window_is_short(z, fixed):
    s = make_spectrum(z=z, snr=20.0, seed=1, narrow=dict(ew_ha=0.0))
    wr, fr, ir = s["wave"] / (1 + z), s["flux"] * (1 + z), s["ivar"] / (1 + z) ** 2
    for fitter in (fit_continuum, fit_continuum_host):
        out = fitter(wr, fr, ir)
        d, info = out[0], out[-1]
        assert info["fe_uv"] and info["feuv_fwhm_fixed"] is fixed
        assert (info["n_pix_uv"] < FE_UV_FREE_MIN_PIXELS) is fixed
        assert isinstance(info["at_bound"], list)
        if fixed:
            assert d["feuv_fwhm"] == FE_UV_FWHM_FIXED_KMS
            assert "feuv_fwhm" not in info["at_bound"]
        else:
            assert FE_FWHM_MIN <= d["feuv_fwhm"] <= FE_FWHM_MAX
    ps = fit_continuum(wr, fr, ir)[2]["ps"]
    assert ("feuv_fwhm" in ps.fixed) is fixed
    assert ("feuv_fwhm" not in ps.free_names) is fixed


@pytest.mark.parametrize("fixed", [True, False])
def test_fixed_uv_width_leaves_the_free_vector_and_stays_in_the_dictionary(fixed):
    ps = ParamSet()
    _add_pl_fe(ps, 10.0, True, True, True, pl_start=10.0, feuv_fixed=fixed)
    assert ("feuv_fwhm" in ps.free_names) is not fixed
    assert ps.fixed == ({"feuv_fwhm": FE_UV_FWHM_FIXED_KMS} if fixed else {})
    assert ps.full(ps.p0())["feuv_fwhm"] == (FE_UV_FWHM_FIXED_KMS if fixed else 3000.0)
    assert len(ps.p0()) == len(ps.names) - (1 if fixed else 0)


def test_host_attempt_kept_when_the_solver_stops_short(monkeypatch):
    # one function evaluation per attempt: every attempt stops without
    # converging; the first non-negative host is still taken, and its solver
    # record says so
    monkeypatch.setattr(continuum, "MAX_NFEV_CONTI_HOST", 1)
    s = make_spectrum(z=0.1, snr=20.0, seed=2, host_frac=0.5)
    wr, fr, ir = s["wave"] / 1.1, s["flux"] * 1.1, s["ivar"] / 1.1 ** 2
    d, total, host, info = fit_continuum_host(wr, fr, ir)
    assert info["reason"] != "joint fit failed; PL+Fe only"
    assert len(info["solver_attempts"]) == 1
    assert info["solver_attempts"][0]["success"] is False
    assert info["solver"]["success"] is False
    assert isinstance(info["at_bound"], list) and "fe_widths" in info


def test_rescaled_input_reproduces_width_and_velocity():
    sp = read_sdss(FE_STRONG)
    a = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_FE_STRONG, complexes=("Hbeta",))
    b = blrfit.fit_spectrum(sp["wave"], sp["flux"] * (1 + EPS), sp["ivar"] / (1 + EPS) ** 2,
                            Z_FE_STRONG, complexes=("Hbeta",))
    assert a["conti"]["feop_norm"] > 0.5 and "feop_norm" not in a["continuum_info"]["at_bound"]
    assert abs(a["conti"]["feop_fwhm"] - b["conti"]["feop_fwhm"]) < 1.0
    ra, rb = blrfit.summary_row(a), blrfit.summary_row(b)
    assert abs(ra["HB_c50_sys"] - rb["HB_c50_sys"]) < 0.1
    assert ra["HB_class"] == rb["HB_class"]


def test_rescaled_input_reproduces_velocity_without_fe():
    sp = read_sdss(SDSS_EXAMPLE)
    a = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Hbeta",))
    b = blrfit.fit_spectrum(sp["wave"], sp["flux"] * (1 + EPS), sp["ivar"] / (1 + EPS) ** 2,
                            Z_J001224, complexes=("Hbeta",))
    ra, rb = blrfit.summary_row(a), blrfit.summary_row(b)
    assert abs(ra["HB_c50_sys"] - rb["HB_c50_sys"]) < 0.1
    assert ra["HB_class"] == rb["HB_class"]
