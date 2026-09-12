"""
Unit tests of the non-parametric profile measures, the classification rules
and the empirical error model on analytic profiles and hand-built measure
dictionaries. No fitting is involved; every test runs in milliseconds.

The measures are defined in ``blrfit.measure.profile_measures`` (Marziani et
al. 1996; Eracleous et al. 2012): for a Gaussian the bisector centres at every
level coincide with the centre, W(1/2) is the FWHM, the kurtosis index
W(3/4) / W(1/4) is sqrt(ln(4/3) / ln 4) = 0.4555 and the asymmetry index is
zero. The classification thresholds are those of ``blrfit.constants.DEFAULT_THRESH``
and the flag rules those of ``blrfit.classify``.
"""
import numpy as np
import pytest

from blrfit.measure import profile_measures
from blrfit.classify import classify, is_measurable, is_strong_offset
from blrfit.errors import empirical_error
from blrfit.constants import S2F, DEFAULT_THRESH

GRID = np.arange(-25000.0, 25000.01, 5.0)      # the 5 km/s grid of measure_complex
KI_GAUSS = np.sqrt(np.log(4.0 / 3.0) / np.log(4.0))   # 0.4555


def gaussian(v0, fwhm, amp=1.0, v=GRID):
    """Gaussian profile of peak ``amp`` at v0 with the given FWHM on the grid."""
    return amp * np.exp(-0.5 * ((v - v0) / (fwhm / S2F)) ** 2)


# ---------------------------------------------------------------------------
# profile_measures on analytic profiles
# ---------------------------------------------------------------------------
def test_single_gaussian_measures_are_its_centre_and_width():
    m = profile_measures(GRID, gaussian(1200.0, 3500.0))
    # the centre lies on the grid, so the peak is exact; the bisectors and the
    # centroid are interpolated on a 5 km/s grid: 1 km/s covers the interpolation
    assert m["v_peak"] == 1200.0
    assert abs(m["c50"] - 1200.0) < 1.0
    assert abs(m["c25"] - 1200.0) < 1.0
    assert abs(m["c75"] - 1200.0) < 1.0
    assert abs(m["centroid"] - 1200.0) < 1.0
    # W(1/2) is the FWHM to within one grid step
    assert abs(m["fwhm"] - 3500.0) < 5.0
    # W(1/4) = FWHM sqrt(ln 4 / ln 2), W(3/4) = FWHM sqrt(ln(4/3) / ln 2)
    assert abs(m["W25"] - 3500.0 * np.sqrt(np.log(4.0) / np.log(2.0))) < 5.0
    assert abs(m["W75"] - 3500.0 * np.sqrt(np.log(4.0 / 3.0) / np.log(2.0))) < 5.0
    # K.I. of a Gaussian is 0.4555; 0.005 covers the grid discretisation of both widths
    assert abs(m["KI"] - 0.456) < 0.005
    assert abs(m["KI"] - KI_GAUSS) < 0.005
    # A.I. of a symmetric profile is zero; 0.01 is 50 km/s of asymmetry on W(1/4) = 4950
    assert abs(m["AI"]) < 0.01
    assert abs(m["skew"]) < 0.01
    assert m["n_peaks"] == 1
    assert not np.isfinite(m["peak_sep"]) and not np.isfinite(m["dip_frac"])
    # second moment of a Gaussian is sigma = FWHM / 2.3548 = 1486 km/s
    assert abs(m["sigma_line"] - 3500.0 / S2F) < 2.0
    # the half-maximum crossings are symmetric about the centre
    assert abs(m["vB50"] - (1200.0 - 1750.0)) < 5.0
    assert abs(m["vR50"] - (1200.0 + 1750.0)) < 5.0


def test_measures_are_independent_of_the_amplitude():
    m1 = profile_measures(GRID, gaussian(-800.0, 4200.0, amp=1.0))
    m2 = profile_measures(GRID, gaussian(-800.0, 4200.0, amp=37.5))
    for k in ("v_peak", "c50", "c25", "fwhm", "W25", "AI", "KI", "centroid", "n_peaks"):
        assert m1[k] == pytest.approx(m2[k], abs=1e-9), k


def test_two_equal_gaussians_are_two_peaks_with_a_deep_dip():
    P = gaussian(-2000.0, 2500.0) + gaussian(2000.0, 2500.0)
    m = profile_measures(GRID, P)
    assert m["n_peaks"] == 2
    # the two maxima of the sum sit a few km/s inside the component centres
    # (the overlap pulls them together); measured separation 3990 km/s
    assert abs(m["peak_sep"] - 4000.0) < 50.0
    # the dip of two Gaussians 4000 km/s apart with FWHM 2500 is 66 per cent,
    # far above the 8 per cent double-peak threshold
    assert m["dip_frac"] > 0.08
    assert m["dip_frac"] > 0.5
    # the profile is symmetric about zero: its bisector centre is at zero
    assert abs(m["c50"]) < 5.0
    assert abs(m["centroid"]) < 1.0


def test_asymmetric_two_component_profile_has_a_tilted_bisector():
    # a Gaussian plus a broader, weaker component displaced by 2000 km/s: the
    # red shoulder moves the low-level bisector but not the peak
    P = gaussian(0.0, 3500.0) + 0.5 * gaussian(2000.0, 6000.0)
    m = profile_measures(GRID, P)
    tilt = abs(m["c25"] - m["c75"]) / m["fwhm"]
    # measured on this profile: A.I. = 0.195, tilt = 0.136 FWHM
    assert abs(m["AI"]) > 0.12 or tilt > 0.10
    assert m["AI"] > 0.15
    assert tilt > 0.12
    assert m["n_peaks"] == 1
    assert m["c25"] > m["c50"] > m["v_peak"]       # the shoulder is red: bisector drifts redward downwards
    assert m["centroid"] > m["v_peak"]


def test_flat_or_empty_profiles_give_nan_measures():
    for P in (np.zeros_like(GRID), np.full_like(GRID, np.nan), None):
        m = profile_measures(GRID, P)
        assert all(not np.isfinite(m[k]) for k in ("v_peak", "c50", "fwhm", "AI", "KI", "n_peaks"))


def test_measures_handle_non_finite_pixels():
    P = gaussian(500.0, 3000.0)
    P[::97] = np.nan
    m = profile_measures(GRID, P)
    assert abs(m["c50"] - 500.0) < 10.0
    assert abs(m["fwhm"] - 3000.0) < 20.0


# ---------------------------------------------------------------------------
# classify on hand-built measure dictionaries
# ---------------------------------------------------------------------------
def measures(offset=0.0, **over):
    """A symmetric Gaussian-like measure dictionary of a strong broad line at
    c(1/2) - v_sys = ``offset``, with every key ``classify`` reads; ``over``
    replaces individual entries."""
    m = dict(fwhm=4000.0, broad_flux_snr=50.0, broad_peak_snr=20.0, sys_snr=30.0,
             narrow_peak_snr=30.0, v_sys=0.0, c50_sys=offset, c25=offset, c50=offset,
             c75=offset, v_peak=offset, centroid=offset, v_peak_sys=offset,
             peak_top_sys=offset, AI=0.0, KI=0.456, n_peaks=1, peak_sep=np.nan,
             dip_frac=np.nan, chi2_red=1.0, host_frac=0.0, pl_alpha=-1.5,
             v_cover_lo=-9000.0, v_cover_hi=9000.0, v_sii=np.nan, flux_SII6716=0.0,
             flux_SII6731=0.0, systemic_source="own narrow group", v_o3=np.nan,
             o3_core_snr=np.nan)
    m.update(over)
    return m


def test_class_a_is_a_significant_symmetric_offset():
    c = classify(measures(1200.0))
    assert c["label"] == "A"
    assert c["flags"] == []
    assert c["features"]["offset"] == 1200.0
    assert c["features"]["tilt"] == 0.0
    # negative offsets count equally
    assert classify(measures(-1200.0))["label"] == "A"
    # just above the 300 km/s threshold is A, just below is F
    assert classify(measures(301.0))["label"] == "A"
    assert classify(measures(299.0))["label"] == "F"


def test_class_a_requires_three_sigma_when_an_error_is_given():
    m = measures(1200.0)
    assert classify(m, err={"c50_sys": 100.0})["label"] == "A"     # 12 sigma
    assert classify(m, err={"c50_sys": 399.0})["label"] == "A"     # 3.01 sigma
    assert classify(m, err={"c50_sys": 500.0})["label"] == "F"     # 2.4 sigma
    assert classify(m, err={"c50_sys": np.nan})["label"] == "A"    # no usable error: threshold only
    assert classify(m, err={})["label"] == "A"


def test_class_b_two_resolved_peaks():
    m = measures(0.0, fwhm=5000.0, n_peaks=2, peak_sep=4000.0, dip_frac=0.5)
    assert classify(m)["label"] == "B"
    # separation must exceed max(0.4 FWHM, 1500): 0.4 x 5000 = 2000
    assert classify(measures(0.0, fwhm=5000.0, n_peaks=2, peak_sep=1900.0, dip_frac=0.5))["label"] == "F"
    assert classify(measures(0.0, fwhm=3200.0, n_peaks=2, peak_sep=1400.0, dip_frac=0.5))["label"] == "F"
    assert classify(measures(0.0, fwhm=3200.0, n_peaks=2, peak_sep=1600.0, dip_frac=0.5))["label"] == "B"
    # the dip must exceed 8 per cent
    assert classify(measures(0.0, fwhm=5000.0, n_peaks=2, peak_sep=4000.0, dip_frac=0.07))["label"] == "F"
    # the two-peak rule applies only at FWHM >= 3000
    assert classify(measures(0.0, fwhm=2900.0, n_peaks=2, peak_sep=4000.0, dip_frac=0.5))["label"] == "F"
    # a double-peaked profile with a net offset is still B, not A
    assert classify(measures(1200.0, fwhm=5000.0, n_peaks=2, peak_sep=4000.0, dip_frac=0.5))["label"] == "B"


def test_class_b_very_broad_and_asymmetric_or_flat():
    assert classify(measures(0.0, fwhm=7500.0, AI=0.25))["label"] == "B"
    assert classify(measures(0.0, fwhm=7500.0, AI=-0.25))["label"] == "B"
    assert classify(measures(0.0, fwhm=7500.0, KI=0.55))["label"] == "B"
    # below 7000 km/s the same shape is C (asymmetric) or F (flat but symmetric)
    assert classify(measures(0.0, fwhm=6500.0, AI=0.25))["label"] == "C"
    assert classify(measures(0.0, fwhm=6500.0, KI=0.55))["label"] == "F"
    # very broad but symmetric and Gaussian-like: F with the very_broad flag
    c = classify(measures(0.0, fwhm=8500.0))
    assert c["label"] == "F" and "very_broad" in c["flags"]
    assert "very_broad" not in classify(measures(0.0, fwhm=7999.0))["flags"]


def test_class_c_asymmetric_by_each_criterion():
    # |A.I.| >= 0.12
    assert classify(measures(0.0, AI=0.15))["label"] == "C"
    assert classify(measures(0.0, AI=-0.15))["label"] == "C"
    assert classify(measures(0.0, AI=0.11))["label"] == "F"
    # bisector tilt |c25 - c75| / FWHM >= 0.10: 600 km/s on FWHM 4000
    assert classify(measures(0.0, c25=300.0, c75=-300.0))["label"] == "C"
    assert classify(measures(0.0, c25=150.0, c75=-150.0))["label"] == "F"
    # |v_peak - centroid| / FWHM >= 0.20: 1000 km/s on FWHM 4000
    assert classify(measures(0.0, centroid=1000.0))["label"] == "C"
    assert classify(measures(0.0, centroid=700.0))["label"] == "F"
    # an asymmetric profile with a large offset is C, not A
    c = classify(measures(1500.0, AI=0.3))
    assert c["label"] == "C" and "offset +1500" in c["reasons"][0]


def test_class_f_normal():
    c = classify(measures(0.0))
    assert c["label"] == "F"
    assert c["flags"] == []
    assert classify(measures(-250.0))["label"] == "F"


def test_class_e_no_broad_line():
    assert classify(measures(1200.0, fwhm=1100.0))["label"] == "E"        # FWHM < 1200
    assert classify(measures(1200.0, fwhm=np.nan))["label"] == "E"
    assert classify(measures(1200.0, broad_flux_snr=4.9))["label"] == "E"  # integrated S/N < 5
    assert classify(measures(1200.0, broad_peak_snr=1.4))["label"] == "E"  # peak S/N < 1.5
    assert classify(measures(1200.0, broad_flux_snr=np.nan))["label"] == "E"
    # E precedes every other test: no systemic reference is needed to say E
    assert classify(measures(1200.0, fwhm=1100.0, v_sys=np.nan))["label"] == "E"
    assert classify({})["label"] == "E"


def test_class_x_no_systemic_reference():
    assert classify(measures(1200.0, sys_snr=2.9))["label"] == "X"
    assert classify(measures(1200.0, v_sys=np.nan))["label"] == "X"
    # sys_snr falls back to narrow_peak_snr when absent
    m = measures(1200.0); del m["sys_snr"]
    assert classify(dict(m, narrow_peak_snr=2.0))["label"] == "X"
    assert classify(dict(m, narrow_peak_snr=3.0))["label"] == "A"


def test_class_w_too_narrow_or_too_weak():
    assert classify(measures(1200.0, fwhm=1900.0))["label"] == "W"        # 1200 <= FWHM < 2000
    assert classify(measures(1200.0, broad_flux_snr=7.9))["label"] == "W"  # 5 <= S/N < 8
    assert classify(measures(1200.0, fwhm=2000.0, broad_flux_snr=8.0))["label"] == "A"


def test_thresholds_can_be_overridden():
    m = measures(1200.0)
    assert classify(m, t={"offset_kms": 2000.0})["label"] == "F"
    assert classify(m, t={"class_fwhm": 5000.0})["label"] == "W"
    assert classify(measures(0.0, AI=0.15), t={"ai_max": 0.20})["label"] == "F"
    assert DEFAULT_THRESH["offset_kms"] == 300.0     # the default is untouched


@pytest.mark.parametrize("flag, over", [
    ("poor_fit", dict(chi2_red=2.5)),
    ("low_snr", dict(broad_flux_snr=9.9)),
    ("low_peak_snr", dict(broad_peak_snr=4.9)),
    ("host_dominated", dict(host_frac=0.80)),
    ("pl_at_bound", dict(pl_alpha=-4.95)),
    ("pl_at_bound", dict(pl_alpha=2.95)),
    ("narrow_at_bound", dict(v_sys=1425.0)),
    ("narrow_at_bound", dict(v_sys=-1425.0)),
    ("edge", dict(v_cover_hi=5999.0)),
    ("edge", dict(v_cover_lo=-5999.0)),
    ("sii_disagree", dict(v_sii=151.0, flux_SII6716=1.0)),
    ("sii_disagree", dict(v_sii=-200.0, flux_SII6731=1.0)),
    ("sys_disagree", dict(v_o3=-401.0, o3_core_snr=5.0)),
    ("sys_disagree", dict(v_o3=np.nan, v_o3_pre=500.0, o3_pre_snr=5.0)),
    ("peak_disagree", dict(peak_top_sys=0.26 * 4000.0)),
    ("extreme_offset", dict(c50_sys=4001.0, c25=4001.0, c75=4001.0, v_peak=4001.0,
                            centroid=4001.0, v_peak_sys=4001.0, peak_top_sys=4001.0)),
])
def test_flag_is_raised_at_its_threshold(flag, over):
    c = classify(measures(0.0, **over))
    assert flag in c["flags"], (flag, c)
    assert c["label"] in ("A", "B", "C", "F")      # flags do not change the class


@pytest.mark.parametrize("flag, over", [
    ("poor_fit", dict(chi2_red=2.4)),
    ("low_snr", dict(broad_flux_snr=10.0)),
    ("low_peak_snr", dict(broad_peak_snr=5.0)),
    ("host_dominated", dict(host_frac=0.79)),
    ("pl_at_bound", dict(pl_alpha=-4.8)),
    ("pl_at_bound", dict(pl_alpha=2.8)),
    ("narrow_at_bound", dict(v_sys=1400.0)),
    ("edge", dict(v_cover_lo=-6000.0, v_cover_hi=6000.0)),
    ("sii_disagree", dict(v_sii=149.0, flux_SII6716=1.0)),
    ("sii_disagree", dict(v_sii=300.0)),                       # no [S II] flux: no flag
    ("sys_disagree", dict(v_o3=-399.0, o3_core_snr=5.0)),
    ("sys_disagree", dict(v_o3=-800.0, o3_core_snr=4.9)),      # [O III] too weak to judge
    ("sys_disagree", dict(v_o3=-800.0, o3_core_snr=50.0, systemic_source="[OIII] core")),
    ("peak_disagree", dict(peak_top_sys=0.24 * 4000.0)),
    ("peak_disagree", dict(peak_top_sys=np.nan)),
    ("extreme_offset", dict(c50_sys=3999.0, c25=3999.0, c75=3999.0, v_peak=3999.0,
                            centroid=3999.0, v_peak_sys=3999.0, peak_top_sys=3999.0)),
])
def test_flag_is_not_raised_below_its_threshold(flag, over):
    assert flag not in classify(measures(0.0, **over))["flags"]


def test_sii_disagree_is_measured_against_the_systemic():
    # [S II] 100 km/s from a systemic at +1000: no flag; 200 km/s away: flag
    assert "sii_disagree" not in classify(measures(0.0, v_sys=1000.0, v_sii=1100.0, flux_SII6716=1.0))["flags"]
    assert "sii_disagree" in classify(measures(0.0, v_sys=1000.0, v_sii=800.0, flux_SII6716=1.0))["flags"]


def test_flags_are_not_computed_for_classes_e_x_w():
    # the quality flags describe a classified broad profile; E, X and W carry none
    for over in (dict(fwhm=1000.0), dict(sys_snr=1.0), dict(fwhm=1900.0)):
        c = classify(measures(1200.0, chi2_red=5.0, host_frac=0.9, v_cover_hi=4000.0, **over))
        assert c["flags"] == []
        assert c["features"] == {}


def test_classify_output_structure():
    c = classify(measures(1200.0))
    assert set(c) == {"label", "reasons", "features", "flags"}
    assert isinstance(c["reasons"], list) and len(c["reasons"]) == 1
    for k in ("offset", "e_offset", "tilt", "peak_minus_cen", "AI", "KI", "n_peaks", "peak_sep", "dip", "fwhm"):
        assert k in c["features"]


def test_classify_on_measured_gaussian_profiles():
    """The measures of an analytic Gaussian, completed with the fit-level keys,
    classify as A at +1200 km/s and F at zero offset."""
    extra = dict(broad_flux_snr=50.0, broad_peak_snr=20.0, sys_snr=30.0, v_sys=0.0,
                 chi2_red=1.0, host_frac=0.0, pl_alpha=-1.5, v_cover_lo=-9000.0, v_cover_hi=9000.0)
    for v0, label in ((1200.0, "A"), (0.0, "F"), (-450.0, "A")):
        m = profile_measures(GRID, gaussian(v0, 3500.0))
        m.update(extra, c50_sys=m["c50"], v_peak_sys=m["v_peak"], peak_top_sys=m["v_peak"])
        assert classify(m)["label"] == label, v0
    # two Gaussians 5000 km/s apart, each FWHM 3000: class B from the measured shape
    m = profile_measures(GRID, gaussian(-2500.0, 3000.0) + gaussian(2500.0, 3000.0))
    m.update(extra, c50_sys=m["c50"], v_peak_sys=m["v_peak"], peak_top_sys=m["v_peak"])
    assert classify(m)["label"] == "B"


# ---------------------------------------------------------------------------
# catalogue selections and the empirical error model
# ---------------------------------------------------------------------------
def test_is_measurable_definition():
    assert is_measurable("A", [], 10.0, 3000.0)
    assert is_measurable("F", ["low_snr"], 8.0, 2000.0)      # boundaries are inclusive
    for label in ("B", "C"):
        assert is_measurable(label, [], 20.0, 4000.0)
    for label in ("E", "X", "W", ""):
        assert not is_measurable(label, [], 20.0, 4000.0)
    assert not is_measurable("A", [], 7.9, 3000.0)
    assert not is_measurable("A", [], 10.0, 1999.0)
    assert not is_measurable("A", ["edge"], 10.0, 3000.0)
    assert not is_measurable("A", ["very_broad", "edge"], 10.0, 3000.0)
    assert not is_measurable("A", [], np.nan, 3000.0)
    assert not is_measurable("A", [], 10.0, np.nan)
    assert is_measurable("A", None, 10.0, 3000.0)


def test_is_strong_offset_definition():
    assert is_strong_offset("A", [], 10.0, 3000.0, 1500.0)
    assert is_strong_offset("A", [], 10.0, 3000.0, -1500.0)
    assert is_strong_offset("A", [], 10.0, 3000.0, 1000.0)
    assert is_strong_offset("A", [], 10.0, 3000.0, 4000.0)
    assert not is_strong_offset("A", [], 10.0, 3000.0, 999.0)
    assert not is_strong_offset("A", [], 10.0, 3000.0, 4001.0)
    assert not is_strong_offset("A", [], 10.0, 3000.0, np.nan)
    assert not is_strong_offset("E", [], 10.0, 3000.0, 1500.0)
    assert not is_strong_offset("A", ["edge"], 10.0, 3000.0, 1500.0)


def test_empirical_error_model():
    # 650 / sqrt(S/N): 65 km/s at S/N 100, 205.5 at S/N 10
    assert empirical_error(100.0) == pytest.approx(65.0)
    assert empirical_error(10.0) == pytest.approx(205.548, abs=0.01)
    # the 45 km/s floor at very high S/N (650 / sqrt(S/N) < 45 above S/N 208.6)
    assert empirical_error(1e6) == 45.0
    assert empirical_error(1e4) == 45.0
    assert empirical_error(208.0) == pytest.approx(650.0 / np.sqrt(208.0))
    assert empirical_error(209.0) == 45.0
    # factor 1.5 for |Delta v| >= 1000 km/s, in either direction; not below
    assert empirical_error(100.0, 1000.0) == pytest.approx(97.5)
    assert empirical_error(100.0, -1200.0) == pytest.approx(97.5)
    assert empirical_error(100.0, 999.0) == pytest.approx(65.0)
    assert empirical_error(1e6, 2000.0) == pytest.approx(67.5)     # the floor is inflated too
    # undefined S/N gives no error
    assert np.isnan(empirical_error(np.nan))
    assert np.isnan(empirical_error(0.0))
    assert np.isnan(empirical_error(-5.0))
    # array input, element-wise
    e = empirical_error(np.array([100.0, 10.0, np.nan]), np.array([0.0, 2000.0, 0.0]))
    assert e.shape == (3,)
    assert e[0] == pytest.approx(65.0) and e[1] == pytest.approx(1.5 * 205.548, abs=0.02) and np.isnan(e[2])
