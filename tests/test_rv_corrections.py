"""Scientific regressions for the corrected, still uncalibrated RV estimator."""
import numpy as np
import pytest
from scipy.optimize import least_squares

from blrfit import rv


def profile(v, f=None, error=0.02):
    v = np.asarray(v, float)
    return dict(v=v, f=np.exp(-0.5*(v/1500.0)**2) if f is None else np.asarray(f),
                e=np.full(v.size, error), ok=np.ones(v.size, bool), nmod=np.zeros(v.size),
                fwhm=3532.0, c50_sys=0.0, v_sys=0.0)


def test_eiv_matches_explicit_latent_flux_likelihood():
    """Independent optimization of every latent pixel checks the derivation."""
    rng = np.random.default_rng(21)
    v = np.linspace(-2, 2, 51)
    mu = np.exp(-v*v)
    ex = 0.04 + 0.04*(v+2)/4
    ey = 0.03 + 0.05*(2-v)/4
    x = mu + rng.normal(size=v.size)*ex
    y = 2.8*mu + 0.2 + 0.03*v + rng.normal(size=v.size)*ey
    beta, design, variance = rv._profile_eiv(y, x, v*1000, ey**2, ex**2,
                                            np.ones(v.size, bool), "linear")
    def residual(par):
        a, b, c = par[:3]
        latent = par[3:]
        return np.r_[(x-latent)/ex, (y-a*latent-b-c*v)/ey]
    sol = least_squares(residual, np.r_[2.0, 0.0, 0.0, x], max_nfev=2000,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    assert sol.success
    assert beta == pytest.approx(sol.x[:3], rel=2e-6, abs=2e-7)
    assert np.sum((y-design@beta)**2/variance) == pytest.approx(2*sol.cost, rel=1e-9)


@pytest.mark.parametrize("baseline", ["linear", "const", None])
def test_flux_and_error_unit_rescaling_leaves_entire_ccf_invariant(baseline):
    v = np.arange(-6000.0, 6001.0, 30.0)
    rng = np.random.default_rng(8)
    p = profile(v, np.exp(-0.5*((v-240)/1500)**2) + rng.normal(0, .02, len(v)))
    t = profile(v, np.exp(-0.5*(v/1500)**2) + rng.normal(0, .04, len(v)), error=.04)
    p["nmod"] = .4*np.exp(-.5*(v/150)**2)
    t["nmod"] = .2*np.exp(-.5*(v/150)**2)
    p["f"][200] += 5.0  # exercise scale-consistent clipping too
    reference = rv.ccf_shift(p, t, baseline=baseline, mismatch=.01)
    for a, b in ((.1, 1.0), (10.0, 1.0), (1.0, .1), (1e-17, 1e-18)):
        ps = dict(p, **{k:p[k]*a for k in ("f", "e", "nmod")})
        ts = dict(t, **{k:t[k]*b for k in ("f", "e", "nmod")})
        result = rv.ccf_shift(ps, ts, baseline=baseline, mismatch=.01)
        for key in ("dv", "err", "chi2_red", "profile_z", "resid_frac"):
            assert result[key] == pytest.approx(reference[key], rel=1e-7, abs=1e-8)
        assert result["n_clipped"] == reference["n_clipped"] > 0
        assert result["scale"] / (a/b) == pytest.approx(reference["scale"], rel=1e-7)
        np.testing.assert_allclose(result["curve"][1], reference["curve"][1], rtol=1e-10, atol=1e-8)


def log_lattice(v, n=None):
    """Logarithmic (constant wavelength ratio) velocity lattice spanning ``v``."""
    n = len(v) if n is None else n
    return 299792.458 * (np.exp(np.linspace(np.log1p(v[0]/299792.458), np.log1p(v[-1]/299792.458), n)) - 1)


@pytest.mark.parametrize("kind", ["spacing", "origin", "gap", "logarithmic", "template_gap",
                                  "logarithmic_epoch", "logarithmic_template"])
def test_actual_coordinates_prevent_artificial_velocity(kind):
    """A shared lattice (uniform or logarithmic, with or without holes) is
    compared by integer placement and is not regridded; an epoch off the
    template's lattice is interpolated, and a logarithmic template is
    resampled only when the epoch is off its lattice."""
    v = np.arange(-6000.0, 6001.0, 30.0)
    t = profile(v)
    if kind == "spacing":
        p = profile(v[0] + np.arange(len(v))*30.3)
    elif kind == "origin":
        p = profile(v + 0.3)
    elif kind == "gap":
        p = profile(np.delete(v, np.arange(180, 185)))
    elif kind == "template_gap":
        p = profile(v)
        t = profile(np.delete(v, np.arange(180, 185)))
    elif kind == "logarithmic":
        p, t = profile(log_lattice(v)), profile(log_lattice(v))
    elif kind == "logarithmic_epoch":
        p = profile(log_lattice(v))
    else:
        p, t = profile(v), profile(log_lattice(v))
    result = rv.ccf_shift(p, t)
    assert result is not None and abs(result["dv"]) < 1.0
    expected = {"spacing": "epoch", "origin": "epoch", "gap": "none", "template_gap": "none",
                "logarithmic": "none", "logarithmic_epoch": "epoch", "logarithmic_template": "template+epoch"}[kind]
    assert result["resampling"] == expected
    assert result["regridded"] == (expected != "none")
    assert result["interpolation_covariance_ignored"] == result["regridded"]
    assert result["velocity_convention"] == "optical_translation"
    assert result["uncertainty_calibrated"] is False


def test_interpolation_excludes_both_masked_brackets_and_propagates_variance():
    p = profile(np.arange(4.0), [0.0, 999.0, 0.0, 0.0])
    p["ok"][1] = False
    p["e"] = np.array([1., 1., 2., 4.])
    f, e, ok, _, flag = rv._prof_on_template(p, profile(np.array([.5, 1.5, 2.5])))
    assert flag
    assert ok.tolist() == [False, False, True]
    assert np.isnan(f[:2]).all()
    assert f[2] == 0.0
    assert e[2] == pytest.approx(np.sqrt(.25*2**2 + .25*4**2))
    # An exact valid native pixel does not inherit its bad neighbour's mask.
    f, e, ok, _, _ = rv._prof_on_template(p, profile(np.array([1.0, 2.0])))
    assert ok.tolist() == [False, True]
    assert e[1] == 2.0


def test_interpolation_does_not_bridge_deleted_pixels():
    p = profile(np.array([0., 1., 2., 6., 7., 8.]))
    _, _, ok, _, flag = rv._prof_on_template(p, profile(np.arange(.5, 8.0, 1.0)))
    assert flag
    assert ok.tolist() == [True, True, False, False, False, False, True, True]


@pytest.mark.parametrize("bad_v", [[0., 0., 1.], [2., 1., 0.], [0., np.nan, 2.]])
def test_invalid_velocity_coordinates_raise(bad_v):
    with pytest.raises(ValueError, match="strictly increasing"):
        rv.ccf_shift(profile(np.asarray(bad_v)), profile(np.arange(5.0)))


def test_native_profile_arrays_keep_masked_pixel_coordinates():
    x = np.array([6500., 6501., 6502., 6503.])
    r = dict(x=x[[0, 1, 3]], y=np.ones(3), w=np.ones(3), d={}, comps=[],
             native_x=x, native_y=np.array([1., 2., 0., 4.]),
             native_w=np.array([1., 1., 0., 1.]), native_mask=np.array([True, True, False, True]))
    p = rv.broad_profile_data(dict(fits={"Halpha":r}, meas={}))
    assert len(p["v"]) == 4
    assert p["ok"].tolist() == [True, True, False, True]
    assert np.isinf(p["e"][2])


def test_bootstrap_reports_the_method_actually_used():
    v = np.arange(-6000.0, 6001.0, 30.0)
    p, t = profile(v), profile(v)
    result = rv.ccf_shift(p, t, n_mc=1)
    assert result["err_method"] == "delta_chi2"
    assert result["n_mc_requested"] == 1
    assert 0 <= result["n_mc_success"] <= 1
    assert result["bootstrap_fallback_reason"] == "insufficient_successful_draws"
    assert result["err"] == result["err_dchi2"]
    result = rv.ccf_shift(p, t, n_mc=10)
    assert result["err_method"] == "bootstrap"
    assert result["n_mc_success"] == 10
    assert result["bootstrap_fallback_reason"] is None


def test_bidirectional_invalid_error_is_symmetric(monkeypatch):
    def fake(prof, template, **kwargs):
        return dict(dv=prof["shift"] - template["shift"], err=prof["err"], at_bound=False,
                    dv_pix=30., chi2_red=1., npix=100, regridded=False, resid_frac=.01,
                    err_method="delta_chi2")
    monkeypatch.setattr(rv, "ccf_shift", fake)
    a, b = dict(shift=20., err=np.nan), dict(shift=0., err=10.)
    ab, ba = rv.shift_bidirectional(a, b), rv.shift_bidirectional(b, a)
    assert np.isnan(ab["err"]) and np.isnan(ba["err"])
    assert ab["dv"] == -ba["dv"]
    assert ab["consistent"] is ba["consistent"] is False
    assert ab["statistically_valid"] is ba["statistically_valid"] is False


@pytest.mark.parametrize("key,value", [("dv", np.nan), ("dv", np.inf), ("err", np.nan), ("err", 0.), ("err", -1.)])
def test_invalid_measurements_never_reliable_or_two_line_consistent(key, value):
    valid = dict(name="Halpha", dv=100., err=10., at_bound=False, profile_z=0., dir_mismatch=0., frame_ok=True)
    invalid = dict(valid, **{key:value})
    assert rv.is_reliable(valid) is True
    assert rv.is_reliable(dict(valid, frame_ok=False)) is False
    assert rv.is_reliable(invalid) is False
    assert rv.two_line_consistent(valid, invalid) is None


def test_two_line_difference_can_propagate_shared_covariance():
    a, b = dict(dv=10., err=4.), dict(dv=4., err=5.)
    result = rv.two_line_consistent(a, b, covariance=8.)
    assert result["sigma"] == pytest.approx(6./5.)
    with pytest.raises(ValueError, match="covariance"):
        rv.two_line_consistent(a, b, covariance=21.)


# ---------------------------------------------------------------------------
# lattices: integer placement on a shared lattice, resampling only off it
# ---------------------------------------------------------------------------
def test_shared_log_lattice_equals_integer_placement():
    """Two spectra on one logarithmic lattice are compared exactly as on an
    index lattice with the same pixel size: same curve, same shift in pixels,
    same error, chi-square and scale, and nothing is resampled."""
    v = np.arange(-6000.0, 6001.0, 30.0)
    lv = log_lattice(v)
    dpix = np.median(np.diff(lv))
    rng = np.random.default_rng(3)
    t = profile(lv, np.exp(-0.5*(lv/1500.0)**2) + rng.normal(0, .03, lv.size), error=.03)
    p = profile(lv, 1.3*np.exp(-0.5*((lv - 5*dpix)/1500.0)**2) + rng.normal(0, .03, lv.size), error=.03)
    iv = np.arange(lv.size) * dpix
    on_log = rv.ccf_shift(p, t, baseline="const", window=(-1e9, 1e9))
    on_index = rv.ccf_shift(dict(p, v=iv), dict(t, v=iv), baseline="const", window=(-1e9, 1e9))
    assert on_log["resampling"] == on_index["resampling"] == "none"
    assert on_log["regridded"] is False and on_log["interpolation_covariance_ignored"] is False
    np.testing.assert_allclose(on_log["curve"][1], on_index["curve"][1], rtol=1e-12, atol=1e-9)
    assert on_log["dv"] / on_log["dv_pix"] == pytest.approx(on_index["dv"] / on_index["dv_pix"], abs=1e-9)
    assert on_log["dv"] / on_log["dv_pix"] == pytest.approx(5.0, abs=0.3)
    for key in ("err", "chi2_red", "npix", "scale", "profile_z", "err_method"):
        assert on_log[key] == pytest.approx(on_index[key], rel=1e-9) if isinstance(on_log[key], float) else on_log[key] == on_index[key]


def test_off_lattice_epoch_is_resampled_and_shared_holes_are_not():
    v = np.arange(-6000.0, 6001.0, 30.0)
    lv = log_lattice(v)
    t = profile(v)
    # the same lattice with a hole is placed, not interpolated across the hole
    p = profile(np.delete(v, np.arange(190, 193)))
    s = rv.ccf_shift(p, t)
    assert s["resampling"] == "none" and s["regridded"] is False
    # an epoch on a logarithmic lattice against a uniform template: the epoch is interpolated
    s = rv.ccf_shift(profile(lv), t)
    assert s["resampling"] == "epoch" and s["regridded"] is True and abs(s["dv"]) < 1.0
    # a uniform epoch against a logarithmic template: the template is resampled and the epoch interpolated
    s = rv.ccf_shift(t, profile(lv))
    assert s["resampling"] == "template+epoch" and s["regridded"] is True and abs(s["dv"]) < 1.0
    # a sub-lattice (every second pixel) is on the lattice; a half-pixel offset is not
    assert rv._on_lattice(v[::2], v) is True
    assert rv._on_lattice(v + 15.0, v) is False
    assert rv._on_lattice(v + 20000.0, v) is False


# ---------------------------------------------------------------------------
# fit-result stand-ins for pair_analysis
# ---------------------------------------------------------------------------
LAM0 = {"Halpha": 6564.61, "Hbeta": 4862.68}
C = 299792.458


def fit_result(name="Halpha", shift=0.0, scale=1.0, seed=0, noise=0.03, hole=None, fwhm=3500.0, oiii=None):
    """A stand-in for one complex of a ``fit_spectrum`` result: a Gaussian
    broad line (no fitted components, so the whole signal counts as data) on a
    uniform rest-wavelength lattice, with ``hole`` (a slice of pixel indices)
    masked in the native arrays and, for Hbeta, a narrow [O III] 5007 line at
    ``oiii`` km/s (sigma 150 km/s) when requested."""
    lo, hi = (6400.0, 6800.0) if name == "Halpha" else (4700.0, 5100.0)
    x = np.arange(lo, hi, 0.65)
    v = (x / LAM0[name] - 1.0) * C
    rng = np.random.default_rng(seed)
    y = scale * np.exp(-0.5 * ((v - shift) / (fwhm / 2.3548)) ** 2) + rng.normal(0.0, noise, x.size)
    if oiii is not None:
        vo = (x / 5008.24 - 1.0) * C
        y = y + 3.0 * scale * np.exp(-0.5 * ((vo - oiii) / 150.0) ** 2)
    w = np.full(x.size, 1.0 / noise)
    mask = np.ones(x.size, bool)
    if hole is not None:
        mask[hole] = False
        w[hole] = 0.0
    r = dict(x=x, y=y, w=w, d={}, comps=[], native_x=x, native_y=y, native_w=w, native_mask=mask)
    return dict(fits={name: r}, meas={name: dict(v_sys=0.0, fwhm=fwhm, c50_sys=shift)})


def zeropoint(dv, err=8.0, line="OIII", at_bound=False, err_method="delta_chi2"):
    return dict(dv=dv, err=err, line=line, source=line, at_bound=at_bound, err_method=err_method)


def test_pair_records_both_directions_and_masked_pixels(monkeypatch):
    """The fitted scale of the template is recorded in both directions (their
    product is one for one profile at two flux levels), with the chi-square,
    profile_z, error, its method and the pixel count of each direction, and
    the masked native pixels of each epoch inside the window."""
    monkeypatch.setattr(rv, "narrow_zeropoint", lambda a, b: zeropoint(10.0))
    a = fit_result(scale=2.0, seed=1, hole=slice(300, 303))
    b = fit_result(scale=1.0, seed=2)
    p = rv.pair_analysis(a, b, name="Halpha", details=True)
    s_ab, s_ba = p["details"]["s_ab"], p["details"]["s_ba"]
    assert p["scale_ab"] == s_ab["scale"] and p["scale_ba"] == s_ba["scale"]
    assert p["scale_ab"] == pytest.approx(2.0, rel=0.05) and p["scale_ab"] * p["scale_ba"] == pytest.approx(1.0, abs=0.05)
    for key in ("chi2_red", "profile_z", "err", "npix"):
        assert p[key + "_ab"] == s_ab[key] and p[key + "_ba"] == s_ba[key]
    assert p["err_method_ab"] == s_ab["err_method"] == "delta_chi2" and p["err_method_ba"] == "delta_chi2"
    assert p["chi2_red"] == max(p["chi2_red_ab"], p["chi2_red_ba"])
    assert p["profile_z"] == max(p["profile_z_ab"], p["profile_z_ba"])
    assert p["err"] == max(p["err_ab"], p["err_ba"]) and p["npix"] == min(p["npix_ab"], p["npix_ba"])
    assert p["n_masked_a"] == s_ab["n_masked_prof"] == s_ba["n_masked_template"] == 3
    assert p["n_masked_b"] == s_ba["n_masked_prof"] == s_ab["n_masked_template"] == 0
    assert p["regridded"] is False and s_ab["resampling"] == "none"
    assert abs(p["dv"]) < 3 * p["err"]
    # masked pixels outside the window are not counted
    far = rv.pair_analysis(fit_result(seed=1, hole=slice(0, 20)), b, name="Halpha")
    assert far["n_masked_a"] == 0


@pytest.mark.parametrize("zp_dv,ok,reason", [
    (100.0, True, ""), (-199.0, True, ""), (600.0, False, "zero point +600 km/s exceeds veto"),
    (-250.0, False, "zero point -250 km/s exceeds veto")])
def test_frame_veto_and_one_zero_point_rule(monkeypatch, zp_dv, ok, reason):
    """A narrow-line zero point beyond FRAME_VETO_KMS vetoes the pair for both
    lines; within it the frame is good, Hbeta is corrected by the zero point
    with its error added in quadrature and Halpha keeps its measured shift."""
    monkeypatch.setattr(rv, "narrow_zeropoint", lambda a, b: zeropoint(zp_dv, err=8.0))
    for name in ("Halpha", "Hbeta"):
        p = rv.pair_analysis(fit_result(name, shift=300.0, seed=1), fit_result(name, seed=2), name=name)
        assert p["frame_ok"] is ok and p["frame_reason"] == reason
        assert p["zp_dv"] == zp_dv and p["zp_err"] == 8.0 and p["zp_source"] == p["zp_line"] == "OIII"
        assert p["zp_at_bound"] is False and "zp_ok" not in p
        assert p["zp_applied"] is (ok and name == "Hbeta")
        if p["zp_applied"]:
            assert p["dv_corrected"] == pytest.approx(p["dv"] - zp_dv)
            assert p["err_corrected"] == pytest.approx(np.hypot(p["err"], 8.0))
        else:
            assert p["dv_corrected"] == p["dv"] and p["err_corrected"] == p["err"]
        assert p["profile_z"] < 5 and p["consistent"] and not p["at_bound"]
        assert rv.is_reliable(p) is ok


def test_frame_not_ok_without_a_measured_zero_point(monkeypatch):
    a, b = fit_result("Hbeta", seed=1), fit_result("Hbeta", seed=2)
    monkeypatch.setattr(rv, "narrow_zeropoint", lambda x, y: None)
    p = rv.pair_analysis(a, b, name="Hbeta")
    assert p["frame_ok"] is False and p["frame_reason"] == "no narrow zero point"
    assert np.isnan(p["zp_dv"]) and np.isnan(p["zp_err"]) and p["zp_source"] is None and p["zp_line"] == ""
    assert p["zp_applied"] is False and p["dv_corrected"] == p["dv"] and rv.is_reliable(p) is False
    monkeypatch.setattr(rv, "narrow_zeropoint", lambda x, y: zeropoint(790.0, at_bound=True))
    p = rv.pair_analysis(a, b, name="Hbeta")
    assert p["frame_ok"] is False and p["frame_reason"] == "zero point at search bound" and p["zp_at_bound"] is True
    assert p["zp_applied"] is False and rv.is_reliable(p) is False
    monkeypatch.setattr(rv, "narrow_zeropoint", lambda x, y: zeropoint(np.nan))
    p = rv.pair_analysis(a, b, name="Hbeta")
    assert p["frame_ok"] is False and p["frame_reason"] == "no narrow zero point"
    # a zero point that is not requested is not measured
    p = rv.pair_analysis(a, b, name="Hbeta", zp=False)
    assert p["frame_ok"] is False and p["frame_reason"] == "no narrow zero point" and rv.is_reliable(p) is False
    # the zero-point rule as a function of its own
    f = rv.frame_check(zeropoint(50.0, err=np.nan, err_method="unavailable"), "Hbeta", -300.0, 20.0)
    assert f["frame_ok"] and f["zp_applied"] and f["dv_corrected"] == -350.0 and np.isnan(f["err_corrected"])
    assert f["zp_err_method"] == "unavailable"


def test_zero_point_record_carries_bound_and_source():
    """narrow_zeropoint returns the cross-correlation record of the [O III]
    region with line, source and at_bound; a shift beyond the +/-800 km/s
    search ends at the bound; nothing measurable gives None."""
    b = fit_result("Hbeta", seed=2, oiii=0.0)
    z = rv.narrow_zeropoint(fit_result("Hbeta", seed=1, oiii=40.0), b)
    assert z["line"] == z["source"] == "OIII" and z["at_bound"] is False
    assert abs(z["dv"] - 40.0) < 15.0 and np.isfinite(z["err"]) and z["err"] > 0
    assert z["err_method"] == "delta_chi2" and z["bracket"] == "both"
    far = rv.narrow_zeropoint(fit_result("Hbeta", seed=1, oiii=1000.0), b)
    assert far is not None and far["at_bound"] is True and np.isnan(far["err"])
    assert rv.frame_check(far, "Hbeta", -300.0, 20.0)["frame_reason"] == "zero point at search bound"
    assert rv.narrow_zeropoint(dict(fits={}, meas={}), b) is None
    assert rv.narrow_zeropoint(fit_result("Halpha", seed=1), fit_result("Halpha", seed=2))["source"] == "SII"


# ---------------------------------------------------------------------------
# graded statistical error
# ---------------------------------------------------------------------------
def test_error_grades_on_constructed_curves():
    """The Delta chi-square interval brackets a parabola on both sides
    ('delta_chi2', the analytic half-width); a curve that never rises by 6.63
    on one side gives the bracketed half-width ('delta_chi2_one_sided'); a
    curve capped below 6.63 on both sides gives the curvature error; a flat
    curve has no error, and none of these is reported as 'delta_chi2'."""
    ns = np.arange(-60, 61); n = ns.astype(float); k0 = 60; dpix = 30.0; a = 0.5
    r = rv._shift_error(ns, a * n**2, k0, dpix)
    half = np.sqrt(rv.DCHI2_99 / a)
    assert r["err_method"] == "delta_chi2" and r["bracket"] == "both"
    assert r["dv"] == pytest.approx(0.0, abs=1e-6)
    assert r["err"] == pytest.approx(half * dpix / rv.SIG_FROM_99, rel=1e-3) and r["err"] == r["err_dchi2"]
    assert r["err_lo"] == pytest.approx(-half * dpix, rel=1e-3) and r["err_hi"] == pytest.approx(half * dpix, rel=1e-3)
    assert r["err_curve"] == pytest.approx(dpix / np.sqrt(a), rel=1e-6)
    one = np.where(n <= 0, a * n**2, np.minimum(a * n**2, 4.0))
    r = rv._shift_error(ns, one, k0, dpix)
    assert r["err_method"] == "delta_chi2_one_sided" and r["bracket"] == "lo"
    assert np.isfinite(r["err_lo"]) and np.isnan(r["err_hi"]) and np.isnan(r["err_dchi2"])
    assert r["err"] == pytest.approx((r["dv"] - r["err_lo"]) / rv.SIG_FROM_99) and r["err"] > 0
    r = rv._shift_error(ns, one[::-1], k0, dpix)
    assert r["err_method"] == "delta_chi2_one_sided" and r["bracket"] == "hi"
    assert r["err"] == pytest.approx((r["err_hi"] - r["dv"]) / rv.SIG_FROM_99)
    capped = np.minimum(a * n**2, 4.0)
    r = rv._shift_error(ns, capped, k0, dpix)
    assert r["err_method"] == "curvature" and r["bracket"] == "none"
    assert np.isnan(r["err_lo"]) and np.isnan(r["err_hi"]) and np.isnan(r["err_dchi2"])
    assert np.isfinite(r["err"]) and r["err"] > 0 and r["err"] == r["err_curve"]
    r = rv._shift_error(ns, np.zeros(ns.size), k0, dpix)
    assert r["err_method"] == "unavailable" and np.isnan(r["err"]) and np.isnan(r["err_curve"])
    # a plateau that the raw curve reaches only far away is still not a bracket:
    # the polynomial edge is never taken for a crossing
    assert np.isnan(rv._raw_cross(n, capped, k0, 6.63, +1)) and np.isnan(rv._raw_cross(n, capped, k0, 6.63, -1))
    assert rv._raw_cross(n, a * n**2, k0, 6.63, +1) == pytest.approx(half, rel=0.05)


def test_bidirectional_reports_mixed_error_methods(monkeypatch):
    def fake(prof, template, **kwargs):
        return dict(dv=prof["shift"] - template["shift"], err=prof["err"], at_bound=False, dv_pix=30.,
                    chi2_red=prof["chi2"], npix=100, regridded=False, resid_frac=.01, err_method=prof["method"],
                    scale=prof["scale"], profile_z=0., n_masked_prof=prof.get("masked", 0))
    monkeypatch.setattr(rv, "ccf_shift", fake)
    a = dict(shift=20., err=10., method="delta_chi2", chi2=1.5, scale=2., masked=2)
    b = dict(shift=0., err=12., method="curvature", chi2=1.1, scale=.5)
    s = rv.shift_bidirectional(a, b)
    assert s["statistically_valid"] and s["err_method"] == "mixed"
    assert s["err_method_ab"] == "delta_chi2" and s["err_method_ba"] == "curvature"
    assert s["err"] == 12. and s["err_ab"] == 10. and s["err_ba"] == 12.
    assert s["scale_ab"] == 2. and s["scale_ba"] == .5 and s["chi2_red_ab"] == 1.5 and s["chi2_red_ba"] == 1.1
    assert s["n_masked_a"] == 2 and s["n_masked_b"] == 0 and s["npix_ab"] == s["npix_ba"] == 100
    b["method"] = "delta_chi2"
    assert rv.shift_bidirectional(a, b)["err_method"] == "delta_chi2"
    b["method"], b["err"] = "unavailable", np.nan
    s = rv.shift_bidirectional(a, b)
    assert s["err_method"] == "unavailable" and np.isnan(s["err"]) and s["err_method_ba"] == "unavailable"


# ---------------------------------------------------------------------------
# the bundled pairs
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def j001224():
    """The two SDSS epochs of J001224 (2001, 2013), fitted at the redshift of Liu et al. (2014)."""
    from blrfit import fit_spectrum, read_spectrum
    from conftest import SDSS_EXAMPLE, SDSS_EXAMPLE_2, Z_J001224
    out = []
    for path in (SDSS_EXAMPLE, SDSS_EXAMPLE_2):
        sp = read_spectrum(path)
        out.append(fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Halpha", "Hbeta")))
    return out


def test_bundled_sdss_pair_is_placed_not_resampled(j001224):
    """The two SDSS epochs of J001224 share the SDSS logarithmic lattice
    exactly (every rest-frame pixel coincides), so the pair is compared by
    integer placement with nothing resampled and the numbers equal those of
    the same data on an index lattice. Measured on this tree: Halpha
    -217.5 +/- 15.4 km/s (profile changed, z_prof 44), Hbeta -292.6 +/- 28.7
    with z_prof 0.8 (stable), [O III] zero point -14 +/- 12 km/s, Hbeta
    corrected to -278 +/- 31 and reliable. The resampled comparison of the
    earlier development tree gave -189 +/- 12 and -311 +/- 26 (z_prof 5.7,
    'mild'); the frozen 0.1.0 gave -294 +/- 39 for Hbeta with z_prof 10."""
    r2001, r2013 = j001224
    for name, expected in (("Halpha", -217.5), ("Hbeta", -292.6)):
        p = rv.pair_analysis(r2013, r2001, name=name, details=True)
        assert p["regridded"] is False and p["interpolation_covariance_ignored"] is False
        assert p["details"]["s_ab"]["resampling"] == p["details"]["s_ba"]["resampling"] == "none"
        assert abs(p["dv"] - expected) < 50.0 and 10.0 < p["err"] < 60.0 and p["err_method"] == "delta_chi2"
        assert p["consistent"] and not p["at_bound"] and p["n_masked_a"] == p["n_masked_b"] == 0
        assert p["zp_source"] == "OIII" and abs(p["zp_dv"]) < 40.0 and p["frame_ok"] and p["frame_reason"] == ""
        if name == "Hbeta":
            assert p["zp_applied"] and p["dv_corrected"] == pytest.approx(p["dv"] - p["zp_dv"])
            assert p["err_corrected"] == pytest.approx(np.hypot(p["err"], p["zp_err"]))
            assert p["profile_z"] < 5.0 and p["profile_grade"] == "stable" and rv.is_reliable(p)
        else:
            assert not p["zp_applied"] and p["dv_corrected"] == p["dv"] and p["err_corrected"] == p["err"]
            assert p["profile_z"] > 5.0 and not rv.is_reliable(p)
        # numerically the integer placement of the native arrays on an index lattice
        pa, pb = rv.broad_profile_data(r2013, name), rv.broad_profile_data(r2001, name)
        assert pa["v"].size == pb["v"].size and np.array_equal(pa["v"], pb["v"]) and rv._on_lattice(pa["v"], pb["v"])
        s = rv.ccf_shift(pa, pb, baseline="const")
        inwin = (pb["v"] >= s["window"][0]) & (pb["v"] <= s["window"][1])
        iv = np.arange(pb["v"].size) * s["dv_pix"]
        t = rv.ccf_shift(dict(pa, v=iv), dict(pb, v=iv), baseline="const", window=(iv[inwin].min(), iv[inwin].max()))
        np.testing.assert_allclose(s["curve"][1], t["curve"][1], rtol=1e-10, atol=1e-8)
        assert s["dv"] / s["dv_pix"] == pytest.approx(t["dv"] / t["dv_pix"], abs=1e-9)
        for key in ("err", "chi2_red", "scale", "profile_z"):
            assert s[key] == pytest.approx(t[key], rel=1e-9)
        assert s["npix"] == t["npix"] and s["resampling"] == t["resampling"] == "none"


@pytest.mark.slow
def test_bundled_sdss_desi_pair_is_regridded():
    """J001247: the DESI coadd (2021) against the 2001 SDSS spectrum. The
    lattices differ, so the SDSS epoch is interpolated onto the DESI template
    in one direction and the SDSS template resampled to a uniform velocity
    grid with the DESI epoch interpolated in the other. Measured on this
    tree: Halpha +71.9 +/- 9.6 km/s (z_prof 9.8, 'mild'; 4 masked DESI pixels
    inside the window), Hbeta +132.7 +/- 24.7 (z_prof 0.75, corrected to
    +118.6 +/- 25.3), zero point +14 +/- 5 km/s."""
    import os
    from blrfit import fit_spectrum, read_spectrum
    from conftest import DESI_EXAMPLE, DESI_TARGETID, EXAMPLES
    sp_s = read_spectrum(os.path.join(EXAMPLES, "spec-0652-52138-0326.fits"))
    sp_d = read_spectrum(DESI_EXAMPLE, targetid=DESI_TARGETID)
    r_s = fit_spectrum(sp_s["wave"], sp_s["flux"], sp_s["ivar"], 0.2203, complexes=("Halpha", "Hbeta"))
    r_d = fit_spectrum(sp_d["wave"], sp_d["flux"], sp_d["ivar"], 0.2203, ebv=float(sp_d["ebv"]), complexes=("Halpha", "Hbeta"))
    for name, expected in (("Halpha", 71.9), ("Hbeta", 132.7)):
        p = rv.pair_analysis(r_d, r_s, name=name, details=True)
        assert p["regridded"] is True and p["interpolation_covariance_ignored"] is True
        assert p["details"]["s_ab"]["resampling"] == "template+epoch" and p["details"]["s_ba"]["resampling"] == "epoch"
        assert abs(p["dv"] - expected) < 50.0 and p["consistent"] and not p["at_bound"]
        assert p["frame_ok"] and p["zp_source"] == "OIII" and abs(p["zp_dv"]) < 40.0
        assert p["zp_applied"] is (name == "Hbeta")
    assert rv.pair_analysis(r_d, r_s, name="Halpha")["n_masked_a"] > 0


def test_second_minimum_of_a_two_well_curve():
    dv = np.arange(-60, 61) * 30.0
    G = np.minimum((dv / 300.0) ** 2, ((dv - 1500.0) / 300.0) ** 2 + 3.0)
    alt, dG = rv._second_minimum(dv, G, int(np.argmin(G)), 500.0)
    assert alt == pytest.approx(1500.0) and dG == pytest.approx(3.0)
    one = (dv / 300.0) ** 2
    alt, dG = rv._second_minimum(dv, one, int(np.argmin(one)), 500.0)
    assert np.isnan(alt) and np.isnan(dG)


@pytest.mark.parametrize("sab,sba,ok", [(1.0, 1.0, True), (2.0, 0.5, True), (3.9, 0.26, True),
                                         (4.5, 0.22, False), (0.01, 50.0, False), (2.0, 2.0, False),
                                         (np.nan, 1.0, False)])
def test_flux_factor_plausibility(sab, sba, ok):
    assert rv._scales_ok(sab, sba) is ok


def test_implausible_or_ambiguous_pairs_are_not_reliable():
    good = dict(name="Halpha", dv=100., err=10., at_bound=False, profile_z=0., dir_mismatch=0., frame_ok=True,
                scale_ok=True, ambiguous=False)
    assert rv.is_reliable(good)
    assert not rv.is_reliable(dict(good, scale_ok=False))
    assert not rv.is_reliable(dict(good, ambiguous=True))


def _gauss_profile(v, centre, fwhm, amp=1.0, err=0.01, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    f = amp * np.exp(-0.5 * ((v - centre) / (fwhm / 2.3548)) ** 2)
    if extra is not None:
        f = f + extra(v)
    f = f + rng.normal(0.0, err, v.size)
    return dict(v=v, f=f, e=np.full(v.size, err), ok=np.ones(v.size, bool), nmod=np.zeros(v.size),
                fwhm=fwhm, c50_sys=centre, v_sys=0.0)


def test_large_true_shift_is_recovered_by_the_two_stage_search():
    """A 3900 km/s translation (the size of the largest change in the catalogue)
    of a FWHM 4300 km/s profile with data well beyond the window is measured in
    two stages, not flagged at the bound, and the searched range covers it."""
    v = np.arange(-15000.0, 15000.01, 30.0)
    t = _gauss_profile(v, 0.0, 4300.0, seed=1)
    p = _gauss_profile(v, 3900.0, 4300.0, seed=2)
    r = rv.ccf_shift(p, t, vmax=5000.0)
    assert r["two_stage"] and not r["at_bound"] and abs(r["dv"] - 3900.0) < 30.0
    lo, hi = r["search_range"]
    assert lo <= -3900.0 and hi >= 3900.0


def test_common_pixel_set_keeps_the_pixel_count_constant():
    """Data only a little beyond the window and an epoch changed by a red
    shoulder (reduced chi-square about 5): every shift of the stage-1 search
    uses one pixel set, so the statistic cannot drift toward small overlaps;
    the range is reduced below vmax to keep half the window, and the shift
    found is small."""
    v = np.arange(-7500.0, 7500.01, 30.0)
    t = _gauss_profile(v, 0.0, 4000.0, seed=3)
    p = _gauss_profile(v, 0.0, 4000.0, seed=4, extra=lambda x: 0.4 * np.exp(-0.5 * ((x - 2500.0) / 800.0) ** 2))
    r = rv.ccf_shift(p, t, vmax=5000.0)
    lo_n, hi_n = r["npix_search"]
    assert r["two_stage"] and lo_n == hi_n
    assert r["common_frac"] >= rv.CCF_COMMON_MIN_FRAC
    lo, hi = r["search_range"]
    assert -5000.0 < lo < 0.0 < hi < 5000.0
    assert abs(r["dv"]) < 1000.0 and not r["at_bound"]


def test_single_stage_search_is_kept_for_the_zero_point():
    v = np.arange(-1500.0, 1500.01, 40.0)
    t = _gauss_profile(v, 0.0, 400.0, seed=5)
    p = _gauss_profile(v, 60.0, 400.0, seed=6)
    r = rv.ccf_shift(p, t, vmax=800.0, window=(-1500.0, 1500.0), baseline="const", min_pix=10, two_stage=False)
    assert not r["two_stage"] and abs(r["dv"] - 60.0) < 15.0


def test_single_pixel_spike_is_masked_before_the_search():
    """A 250-sigma spike in one pixel of the epoch is found against its own
    neighbours and masked before the search, so it cannot choose the shift at
    which the outliers are identified; noise around a broad and a narrow line
    raises no such flag."""
    v = np.arange(-6000.0, 6001.0, 30.0)
    rng = np.random.default_rng(8)
    f = np.exp(-0.5 * ((v - 240.0) / 1500.0) ** 2) + rng.normal(0.0, 0.02, v.size)
    f[200] += 5.0
    p = dict(v=v, f=f, e=np.full(v.size, 0.02), ok=np.ones(v.size, bool), nmod=np.zeros(v.size))
    assert np.flatnonzero(rv._spikes(p, 5.0)).tolist() == [200]
    t = _gauss_profile(v, 0.0, 3532.0, err=0.04, seed=9)
    r = rv.ccf_shift(dict(p, fwhm=3532.0, c50_sys=0.0, v_sys=0.0), t)
    assert r["n_spikes"] == 1 and not r["at_bound"] and abs(r["dv"] - 240.0) < 40.0
    quiet = dict(p, f=np.exp(-0.5 * (v / 1500.0) ** 2) + 0.3 * np.exp(-0.5 * (v / 150.0) ** 2)
                 + rng.normal(0.0, 0.02, v.size))
    assert rv._spikes(quiet, 5.0).sum() <= 1



def test_direction_cut_can_be_replaced_per_call():
    """A recalibrated direction cut is applied by the caller through dir_cut
    (a number for the pair's line or a per-line mapping); without it the
    package constant decides, and a line missing from a mapping is never
    reliable."""
    pair = dict(name="Halpha", dv=100., err=10., at_bound=False, profile_z=0., dir_mismatch=300., frame_ok=True,
                scale_ok=True, ambiguous=False)
    assert rv.CCF_DIR_CUT_KMS["Halpha"] > 300.0 and rv.is_reliable(pair)
    assert not rv.is_reliable(pair, dir_cut=250.0)
    assert rv.is_reliable(pair, dir_cut=350.0)
    assert not rv.is_reliable(pair, dir_cut={"Halpha": 290.0, "Hbeta": 900.0})
    assert rv.is_reliable(pair, dir_cut={"Halpha": 310.0})
    assert not rv.is_reliable(pair, dir_cut={"Hbeta": 900.0})
    assert not rv.is_reliable(dict(pair, dir_mismatch=np.nan), dir_cut=1e9)
    hb = dict(pair, name="Hbeta")
    assert not rv.is_reliable(hb) and rv.is_reliable(hb, dir_cut={"Hbeta": 400.0})


def test_zero_point_is_antisymmetric_and_checks_its_directions():
    """The zero point is measured both ways: swapping the spectra flips its
    sign exactly, and a zero point whose two directions disagree beyond their
    errors vetoes the frame."""
    v = np.arange(-1500.0, 1500.01, 69.0)
    w = np.arange(-1500.0, 1500.01, 40.0)
    rng = np.random.default_rng(3)

    def narrow(grid, centre, err, seed):
        r = np.random.default_rng(seed)
        f = np.exp(-0.5 * ((grid - centre) / 150.0) ** 2) + r.normal(0.0, err, grid.size)
        return dict(v=grid, f=f, e=np.full(grid.size, err), ok=np.ones(grid.size, bool), nmod=np.zeros(grid.size))

    a, b = narrow(v, 45.0, 0.03, 1), narrow(w, 0.0, 0.05, 2)
    opts = dict(vmax=800.0, window=(-1500.0, 1500.0), baseline="const", min_pix=10, two_stage=False)
    ab = rv.ccf_shift(a, b, **opts); ba = rv.ccf_shift(b, a, **opts)
    assert abs(ab["dv"] + ba["dv"]) > 0.0            # one direction alone is not antisymmetric
    zp = dict(dv=0.5 * (ab["dv"] - ba["dv"]), err=max(ab["err"], ba["err"]), line="OIII", source="OIII",
              at_bound=False, err_method="delta_chi2", consistent=True)
    assert abs(zp["dv"] - 45.0) < 25.0
    assert rv.frame_check(zp, "Hbeta", 100.0, 20.0)["frame_ok"]
    bad = dict(zp, consistent=False)
    f = rv.frame_check(bad, "Hbeta", 100.0, 20.0)
    assert not f["frame_ok"] and "inconsistent" in f["frame_reason"] and not f["zp_applied"]

