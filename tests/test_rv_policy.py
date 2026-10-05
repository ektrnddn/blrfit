import numpy as np
import pytest

from blrfit.rv_policy import measurement_policy, two_line_policy


def pair(**changes):
    # the fields of rv.pair_analysis after its frame check (rv.frame_check):
    # a 20 +/- 40 km/s zero point inside the veto, applied to Hbeta
    p = dict(
        name="Hbeta",
        dv=120.0,
        err=30.0,
        zp_dv=20.0,
        zp_err=40.0,
        frame_ok=True,
        frame_reason="",
        zp_source="OIII",
        zp_applied=True,
        dv_corrected=100.0,
        err_corrected=50.0,
        profile_grade="stable",
        snr_proxy=12.0,
        profile_z=1.0,
        dir_mismatch=1.0,
        at_bound=False,
    )
    p.update(changes)
    return p


@pytest.mark.parametrize("kinds", [("sdss", "sdss"), ("table", "desi"), ("table", "table")])
def test_unsupported_pair_gets_no_cross_survey_floor(kinds):
    r = measurement_policy(pair(), "Hbeta", *kinds)
    assert np.isnan(r["legacy_error_floor"])
    assert r["legacy_error_floor_kind"] == "unsupported_instrument_pair"
    assert np.isnan(r["error_floor"])
    assert not r["calibration_supported"] and not r["reliable"]


def test_zero_point_error_and_shift_use_one_policy():
    r = measurement_policy(pair(), "Hbeta", "sdss", "desi")
    assert r["dv_corrected"] == 100.0
    assert r["err_total"] == 50.0
    assert r["significance"] == 2.0
    assert r["legacy_error_floor"] > 0
    assert np.isnan(r["error_floor"])
    assert "covariance" in r["uncertainty_policy"]


def test_two_line_uses_corrected_values_and_reported_uncertainty():
    a = dict(measured=True, dv=900.0, err=1.0, dv_corrected=100.0, err_total=50.0)
    b = dict(measured=True, dv=-900.0, err=1.0, dv_corrected=110.0, err_total=60.0)
    r = two_line_policy(a, b)
    assert r["consistent"]
    assert r["difference"] == -10.0
    assert r["sigma"] == pytest.approx(10 / np.hypot(50, 60))
    assert not r["calibration_supported"]


def test_invalid_measurement_is_never_statistically_valid():
    r = measurement_policy(pair(dv=np.nan, dv_corrected=np.nan), "Hbeta", "desi", "desi")
    assert not r["statistical_valid"] and not r["reliable"]
    assert np.isnan(r["significance"])


def test_vetoed_frame_is_reported_and_never_passes_the_quality_check():
    r = measurement_policy(
        pair(
            zp_dv=600.0,
            frame_ok=False,
            frame_reason="zero point +600 km/s exceeds veto",
            zp_applied=False,
            dv_corrected=120.0,
            err_corrected=30.0,
        ),
        "Hbeta",
        "sdss",
        "desi",
    )
    assert not r["frame_ok"] and "exceeds veto" in r["frame_reason"]
    assert not r["zp_applied"] and r["dv_corrected"] == 120.0
    assert not r["diagnostic_quality_pass"] and not r["reliable"]


def test_per_direction_diagnostics_are_carried():
    r = measurement_policy(
        pair(scale_ab=2.0, scale_ba=0.5, err_method="delta_chi2", n_masked_a=3), "Hbeta", "desi", "desi"
    )
    assert r["scale_ab"] == 2.0 and r["scale_ba"] == 0.5 and r["n_masked_a"] == 3
    assert r["err_method"] == "delta_chi2"
