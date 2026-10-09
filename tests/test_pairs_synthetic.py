"""
The template cross-correlation on synthetic profiles with known shifts:
recovery without noise, the linear terms, the bounds, the narrow-line
variants and the frame check, and, with noise, the bias and the coverage of
the errors. The stochastic tests use fixed seeds and enough realisations that
the stated region is at least three standard errors wide.
"""

import numpy as np
import pytest

from blrfit import pairs
from blrfit.constants import PAIR_STEP_KMS
from synth_pairs import make_epoch, refit_broad

LINE = "Halpha"
SHIFTS = (-2500.0, -1000.0, -300.0, -150.0, 0.0, 150.0, 300.0, 1000.0, 2500.0)
PROFILES = {
    "1g_fwhm2000": [(10.0, 0.0, 2000.0 / 2.3548)],
    "2g_asym_fwhm5000": [(10.0, -400.0, 1800.0), (5.0, 1500.0, 3200.0)],
    "3g_fwhm10000": [(6.0, -1500.0, 2500.0), (8.0, 300.0, 4500.0), (4.0, 2500.0, 3000.0)],
}
COVERAGE_68 = (0.683 - 0.08, 0.683 + 0.08)
COVERAGE_95 = (0.954 - 0.04, 0.954 + 0.04)


def broad_flags(rec):
    """Flags other than the narrow-frame ones (broad-only synthetic epochs have no narrow lines)."""
    return [f for f in rec["flags"] if not f.startswith("narrow_")]


def pair(shift, profile="1g_fwhm2000", grid="desi", line=LINE):
    a = make_epoch(line, grid=grid, broad=PROFILES[profile])
    b = make_epoch(line, grid=grid, broad=PROFILES[profile], shift=shift)
    return a, b


def coverage(pulls):
    pulls = np.abs(np.asarray(pulls))
    return float(np.mean(pulls <= 1.0)), float(np.mean(pulls <= 1.96))


# ----------------------------------------------------------------------------
# noise-free recovery


@pytest.mark.parametrize("grid", ["desi", "sdss"])
@pytest.mark.parametrize("profile", list(PROFILES))
def test_noise_free_recovery(grid, profile):
    for shift in SHIFTS:
        a, b = pair(shift, profile, grid)
        rec = pairs.measure_pair(a, b, LINE)
        assert not broad_flags(rec), (shift, rec["flags"])
        assert abs(rec["s_fwd"] - shift) < 1.0, (profile, grid, shift, rec["s_fwd"])
        assert abs(rec["s_rev"] + shift) < 1.0, (profile, grid, shift, rec["s_rev"])
        assert abs(rec["s_common"] - shift) < 1.0
        assert rec["dir_mismatch"] < 1.0
        assert abs(rec["scale_fwd"] - 1.0) < 1e-3 and abs(rec["scale_rev"] - 1.0) < 1e-3
        assert abs(rec["dchi2_shape_fwd"]) < 0.05 and abs(rec["dchi2_shape_rev"]) < 0.05
        assert rec["retained"] and rec["errors_a"] == "record"


def test_flux_scale_and_baseline():
    """A flux scale and a linear baseline in the data epoch: with the slope term
    the shift and the scale are exact; without it the baseline biases the shift."""
    for scale in (0.5, 2.0):
        for shift in (-1000.0, 300.0):
            a = make_epoch(LINE, broad=PROFILES["2g_asym_fwhm5000"])
            b = make_epoch(
                LINE, broad=PROFILES["2g_asym_fwhm5000"], shift=shift, flux_scale=scale, baseline=(2.0, 0.5)
            )
            rec = pairs.measure_pair(a, b, LINE, slope=True)
            assert abs(rec["s_fwd"] - shift) < 1.0, (scale, shift, rec["s_fwd"])
            assert abs(rec["scale_fwd"] - scale) < 1e-3
            rec0 = pairs.measure_pair(a, b, LINE, slope=False)
            assert abs(rec0["s_fwd"] - shift) > abs(rec["s_fwd"] - shift)


def test_at_bound_is_reported_not_measured():
    a, b = pair(4500.0)
    rec = pairs.measure_pair(a, b, LINE)
    assert "fwd:at_bound" in rec["flags"] and "rev:at_bound" in rec["flags"]
    assert np.isnan(rec["s_common"]) and np.isnan(rec["err"]) and not rec["retained"]
    assert "not_retained" in rec["flags"]


def test_resolution_difference_kernel():
    """A template broadened by 75 km/s (the largest SDSS-DESI resolution
    difference) on an asymmetric profile moves the shift by well under the
    smallest tolerance of the validation (30 km/s)."""
    for shift in (0.0, 300.0, 1000.0):
        a, b = pair(shift, "2g_asym_fwhm5000")
        rec = pairs.measure_pair(a, b, LINE, extra_sigma_kms=75.0)
        assert abs(rec["s_common"] - shift) < 3.0


def test_second_minimum_detection():
    grid = pairs.velocity_grid(2000.0, 10.0)
    chi2 = (grid / 100.0) ** 2
    chi2 += -60.0 * np.exp(-0.5 * ((grid - 1200.0) / 80.0) ** 2) + 120.0 * (np.abs(grid - 1200.0) < 400)
    k0 = int(np.argmin(chi2))
    s_alt, d_alt = pairs.second_minimum(grid, chi2, k0, min_sep_kms=500.0)
    assert abs(s_alt - 1200.0) < 40.0 and d_alt > 0
    s_none, d_none = pairs.second_minimum(grid, (grid / 100.0) ** 2, k0, min_sep_kms=500.0)
    assert np.isnan(s_none) and np.isnan(d_none)


def test_velocity_grid_is_centred():
    g = pairs.velocity_grid()
    assert g[0] == -4000.0 and g[-1] == 4000.0 and g[g.size // 2] == 0.0
    assert np.allclose(np.diff(g), PAIR_STEP_KMS)


# ----------------------------------------------------------------------------
# narrow lines and the frame


@pytest.mark.parametrize("narrow_mode", ["none", "weight", "solve", "prior"])
def test_narrow_residual_variants(narrow_mode):
    """B's narrow model is 7 per cent too strong in the record (the data keep the
    truth): the residual does not move with the broad line. The 'solve' and
    'prior' variants remove it; the others are reported."""
    narrow = dict(amp=12.0, v=0.0, sig=140.0)
    for shift in (300.0, 1000.0):
        a = make_epoch(LINE, broad=PROFILES["1g_fwhm2000"], narrow=narrow)
        b = make_epoch(LINE, broad=PROFILES["1g_fwhm2000"], narrow=narrow, shift=shift, fit_narrow_error=0.07)
        rec = pairs.measure_pair(a, b, LINE, narrow=narrow_mode)
        if narrow_mode == "solve":
            assert abs(rec["s_fwd"] - shift) < 3.0, (shift, rec["s_fwd"])
        if narrow_mode == "prior":
            assert abs(rec["s_fwd"] - shift) < 5.0, (shift, rec["s_fwd"])
        # without noise the nominal errors are tiny, so the residual's bias of tens of km/s is
        # "inconsistent" between the directions; the direction values stay within the tolerance
        assert abs(rec["s_fwd"] - shift) < 100.0 and abs(rec["s_rev"] + shift) < 100.0, (narrow_mode, shift)


def test_narrow_shift_and_frame_check():
    """B carries a +120 km/s frame offset (narrow and broad lines) on top of a
    +180 km/s broad-line shift: in the common frame the broad shift is 300, the
    narrow shift 120, the corrected shift 180. A 350 km/s frame offset is vetoed."""
    narrow = dict(amp=8.0, v=0.0, sig=150.0)
    a = make_epoch("Hbeta", broad=[(10.0, 0.0, 1500.0)], narrow=narrow)
    b = make_epoch("Hbeta", broad=[(10.0, 0.0, 1500.0)], narrow=narrow, shift=300.0, narrow_shift=120.0)
    rec = pairs.measure_pair(a, b, "Hbeta")
    assert abs(rec["s_common"] - 300.0) < 1.0
    assert abs(rec["dv_narrow"] - 120.0) < 1.0, rec["dv_narrow"]
    assert abs(rec["s_corrected"] - 180.0) < 1.5
    assert abs(rec["v_sys_diff"] - 120.0) < 1e-6
    assert "frame_offset_large" not in rec["flags"]
    assert rec["narrow_cores"] == ["Hbeta_n", "OIII5007c", "OIII4959c"]
    b2 = make_epoch("Hbeta", broad=[(10.0, 0.0, 1500.0)], narrow=narrow, shift=0.0, narrow_shift=350.0)
    rec2 = pairs.measure_pair(a, b2, "Hbeta")
    assert "frame_offset_large" in rec2["flags"] and abs(rec2["dv_narrow"] - 350.0) < 1.0
    rec3 = pairs.measure_pair(a, b, "Hbeta", frame=False)
    assert np.isnan(rec3["dv_narrow"]) and abs(rec3["s_common"] - 300.0) < 1.0


# ----------------------------------------------------------------------------
# noise: bias and coverage


def single_direction_pulls(snr, shift, n, seed):
    rng = np.random.default_rng(seed)
    a = make_epoch(LINE, broad=PROFILES["1g_fwhm2000"])
    ep_a = pairs.epoch_profile(a, LINE)
    devs, errs = [], []
    for _ in range(n):
        b = make_epoch(LINE, broad=PROFILES["1g_fwhm2000"], shift=shift, snr_peak=snr, rng=rng)
        r = pairs.measure_direction(pairs.epoch_profile(b, LINE), ep_a)
        assert not r["flags"], r["flags"]
        devs.append(r["s"] - shift)
        errs.append(r["err"])
    devs, errs = np.array(devs), np.array(errs)
    return devs, errs, devs / errs


@pytest.mark.parametrize("snr", [8.0, 30.0])
def test_noisy_single_direction_coverage(snr):
    """Template exact, data noisy at peak S/N ``snr``: the curvature error covers
    the truth at the nominal rates (300 realisations per shift)."""
    for shift in (0.0, 1000.0):
        devs, errs, pulls = single_direction_pulls(snr, shift, 300, 20261007)
        se = devs.std(ddof=1) / np.sqrt(devs.size)
        f68, f95 = coverage(pulls)
        assert abs(devs.mean()) < max(3.0 * se, 2.0)
        assert COVERAGE_68[0] <= f68 <= COVERAGE_68[1], f68
        assert COVERAGE_95[0] <= f95 <= COVERAGE_95[1], f95


def symmetric_pulls(n, shift, snr, seed):
    rng = np.random.default_rng(seed)
    sym, fwd, devs, mism = [], [], [], []
    for _ in range(n):
        a = refit_broad(make_epoch(LINE, broad=PROFILES["1g_fwhm2000"], snr_peak=snr, rng=rng), LINE)
        b = refit_broad(
            make_epoch(LINE, broad=PROFILES["1g_fwhm2000"], shift=shift, snr_peak=snr, rng=rng), LINE
        )
        rec = pairs.measure_pair(a, b, LINE)
        assert not broad_flags(rec), rec["flags"]
        devs.append(rec["s_common"] - shift)
        sym.append((rec["s_common"] - shift) / rec["err"])
        fwd.append((rec["s_fwd"] - shift) / rec["err_fwd"])
        mism.append(rec["dir_mismatch"] / rec["err"])
    return np.array(devs), np.array(sym), np.array(fwd), np.array(mism)


def test_noisy_symmetric_estimate_coverage():
    """Both epochs noisy (peak S/N 15), each record refitted to its own data so
    the template carries its noise: the symmetric estimate with the error
    hypot(err_fwd, err_rev) covers the truth at the nominal rates, and a single
    direction's curvature error alone does not (it ignores the template's noise)."""
    devs, sym, fwd, mism = symmetric_pulls(200, 300.0, 15.0, 7)
    f68, f95 = coverage(sym)
    g68, _g95 = coverage(fwd)
    se = devs.std(ddof=1) / np.sqrt(devs.size)
    assert abs(devs.mean()) < max(3.0 * se, 2.0)
    assert COVERAGE_68[0] <= f68 <= COVERAGE_68[1], f68
    assert COVERAGE_95[0] <= f95 <= COVERAGE_95[1], f95
    assert g68 < COVERAGE_68[0], g68
    assert np.median(mism) < 1.0  # the two directions agree under noise alone


# ----------------------------------------------------------------------------
# the pair table


def test_pair_row_columns_and_terms():
    a, b = pair(300.0, "2g_asym_fwhm5000")
    meta_a = dict(id="night-20210101", mjd=59215.0, kind="desi", targetid="1")
    meta_b = dict(id="night-20230101", mjd=59945.0, kind="desi", targetid="1")
    row = pairs.pair_record(a, b, LINE, meta_a, meta_b, role="reference")
    assert tuple(row) == pairs.PAIR_COLUMNS
    assert (
        row["pair"] == "night-20210101__night-20230101"
        and row["kind"] == "desi-desi"
        and row["role"] == "reference"
    )
    assert row["dt_days"] == 730.0 and abs(row["dt_rest_yr"] - 730.0 / 365.25 / 1.1) < 1e-9
    assert row["sigma_sys"] == 45.0 and abs(row["err_total"] - np.hypot(row["err"], 45.0)) < 1e-9
    assert row["retained"] and row["stable_shape"] and row["shape_max"] < 0.01
    assert isinstance(row["flags"], str) and "not_retained" not in row["flags"]
    # a line without a calibrated term keeps the statistical error and says so
    rec = pairs.measure_pair(a, b, LINE)
    rec["line"] = "MgII"
    row2 = pairs.pair_row(rec, a, meta_a, meta_b)
    assert (
        np.isnan(row2["sigma_sys"])
        and row2["err_total"] == row2["err"]
        and "uncalibrated_line" in row2["flags"]
    )


def test_independent_and_enumerate_pairs():
    assert not pairs.independent(dict(id="spec-0651-52141-0072"), dict(id="spec-0651-52200-0072"))
    assert pairs.independent(dict(id="spec-0651-52141-0072"), dict(id="spec-7169-56628-0344"))
    assert not pairs.independent(dict(id="x"), dict(id="x"))
    assert not pairs.independent(dict(id="a", plate=651, fiber=72), dict(id="b", plate=651, fiber=72))
    e1 = dict(id="e1", mjd=100.0, res=make_epoch(LINE, snr_peak=10.0, rng=np.random.default_rng(1)))
    e2 = dict(id="e2", mjd=200.0, res=make_epoch(LINE, snr_peak=30.0, rng=np.random.default_rng(2)))
    e3 = dict(id="e3", mjd=300.0, res=make_epoch(LINE, snr_peak=20.0, rng=np.random.default_rng(3)))
    prs, ref = pairs.enumerate_pairs([e3, e1, e2], lines=(LINE,))
    assert ref is e2
    got = {(p["a"]["id"], p["b"]["id"]): p["role"] for p in prs}
    assert got == {("e1", "e2"): "reference+consecutive", ("e2", "e3"): "reference+consecutive"}
    e4 = dict(id="e4", mjd=400.0, res=make_epoch(LINE, snr_peak=5.0, rng=np.random.default_rng(4)))
    prs, ref = pairs.enumerate_pairs([e1, e2, e3, e4], lines=(LINE,))
    got = {(p["a"]["id"], p["b"]["id"]): p["role"] for p in prs}
    assert got[("e2", "e4")] == "reference" and got[("e3", "e4")] == "consecutive"
