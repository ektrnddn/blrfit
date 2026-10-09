"""The candidate tiers on constructed pair rows: every tier, the conditions that
move a change between them, the orbital bound and the independence rule."""

import numpy as np
import pytest

from blrfit import tiers
from blrfit.physics import orbital_limits, blr_radius_ltd


def row(line="Halpha", s=900.0, err=100.0, pair="a__b", id_a="a", id_b="b", **kw):
    r = dict(
        targetid="t1",
        pair=pair,
        line=line,
        role="reference",
        id_a=id_a,
        id_b=id_b,
        retained=True,
        s_common=s,
        err=err,
        err_total=err,
        shape_max=0.1,
        dchi2_shape_fwd=np.nan,
        dchi2_shape_rev=np.nan,
        npix_fwd=100,
        npix_rev=100,
        scale_fwd=1.0,
        scale_rev=1.0,
        snr_a=20.0,
        snr_b=20.0,
        peak_snr_a=10.0,
        peak_snr_b=10.0,
        cls_a="A",
        cls_b="A",
        dt_rest_yr=5.0,
        v_sys_diff=0.0,
        dir_mismatch=50.0,
        flags="",
    )
    r.update(kw)
    return r


def verdict(rows, **kw):
    return tiers.classify_target(tiers.pair_verdicts(rows), **kw)


def test_platinum_and_binary():
    tier, reasons, best = verdict([row("Halpha"), row("Hbeta", s=850.0, err=150.0)])
    assert tier == "platinum" and best["line"] == "Halpha" and best["agree"] is True
    assert "the other line agrees" in reasons[0]
    tier, reasons, best = verdict([row("Halpha")])
    assert tier == "binary" and best["agree"] is None and "not measured" in reasons[0]
    # the platinum conditions: a direction mismatch above 2 sigma, or a peak S/N below 8, keep it at binary
    both = dict(s=850.0, err=150.0)
    assert (
        verdict([row("Halpha", dir_mismatch=250.0), row("Hbeta", dir_mismatch=400.0, **both)])[0] == "binary"
    )
    assert verdict([row("Halpha", peak_snr_b=6.0), row("Hbeta", peak_snr_b=6.0, **both)])[0] == "binary"
    assert (
        verdict([row("Halpha", shape_max=0.4), row("Hbeta", s=850.0, err=150.0, shape_max=0.4)])[0]
        == "binary"
    )
    # the platinum candidate may be the other line: the best pair is the platinum one of highest significance
    tier, reasons, best = verdict([row("Halpha", shape_max=0.4), row("Hbeta", s=850.0, err=150.0)])
    assert tier == "platinum" and best["line"] == "Hbeta"


def test_almost_profile_stable_none():
    assert verdict([row(shape_max=0.8)])[0] == "almost"  # one failed condition
    assert verdict([row(shape_max=0.8), row("Hbeta", s=-900.0, err=100.0, shape_max=0.8)])[0] == "profile"
    assert verdict([row(s=350.0)])[0] == "almost"  # marginal, clean
    tier, reasons, best = verdict([row(s=100.0)])
    assert tier == "stable" and best is None and "largest change 1.0 sigma" in reasons[0]
    assert verdict([row(snr_a=5.0)])[0] == "none"
    assert verdict([row(retained=False)])[0] == "none"
    assert verdict([row(flags="frame_offset_large")])[0] == "almost"
    assert verdict([row(v_sys_diff=400.0)])[0] == "almost"
    assert verdict([row(cls_b="B")])[0] == "almost"
    assert verdict([row(scale_fwd=3.0, scale_rev=3.0)])[0] == "none"  # scales not reciprocal


def test_two_line_disagreement_and_shape_use_total_errors():
    rows = [row("Halpha", s=900.0, err=100.0), row("Hbeta", s=100.0, err=100.0)]
    tier, reasons, best = verdict(rows)
    assert tier == "almost" and "two-line" in reasons[0]
    rows = [row("Halpha", s=900.0, err=100.0), row("Hbeta", s=600.0, err=200.0)]
    assert verdict(rows)[0] == "platinum"


def test_disk_from_reference_classes_and_from_epochs():
    tier, reasons, best = verdict([row()], reference_classes={"Halpha": "B", "Hbeta": "A"})
    assert tier == "disk" and "largest change 9.0 sigma" in reasons[0]
    tier, reasons, best = verdict([row(s=100.0, cls_a="B")])
    assert tier == "disk" and best is None


def test_orbital_bound():
    lim = orbital_limits(1e8, blr_radius_ltd(1e44), 0.1)
    mass = dict(
        logmbh=8.0, vmax_q01=lim["vmax_kms"], pmin_q01_yr=lim["pmin_yr"], vmax_q1=np.nan, pmin_q1_yr=np.nan
    )
    tier, reasons, best = verdict([row(s=900.0, err=100.0, dt_rest_yr=5.0)], mass=mass)
    assert tier == "binary" and best["orbit"]["orbital_ok"] is True and "P <=" in reasons[0]
    slow = dict(mass, vmax_q01=200.0)  # no orbit at this mass reaches 700 km/s (2 sigma below 900)
    tier, reasons, best = verdict([row(s=900.0, err=100.0, dt_rest_yr=5.0)], mass=slow)
    assert tier == "almost" and "exceeds any orbit" in reasons[0] and best["orbit"]["orbital_ok"] is False
    # upper-case catalogue names are accepted; without a bound the screen is undecided
    orb = tiers.orbital_screen(
        dict(LOGMBH=8.0, VMAX_Q01=lim["vmax_kms"], PMIN_YR_Q01=lim["pmin_yr"]), 900.0, 100.0, 5.0
    )
    assert orb["orbital_ok"] is True and orb["logmbh"] == 8.0
    assert tiers.orbital_screen(dict(logmbh=8.0), 900.0, 100.0, 5.0)["orbital_ok"] is None


def test_reversal_of_sign():
    rows = [
        row(pair="a__r", id_a="a", id_b="r", dt_rest_yr=1.0, s=900.0),
        row(pair="r__b", id_a="r", id_b="b", dt_rest_yr=2.0, s=-900.0),
        row(pair="r__c", id_a="r", id_b="c", dt_rest_yr=3.0, s=900.0),
    ]
    tier, reasons, best = verdict(rows)
    assert tier == "almost" and "reverse sign 2 times" in reasons[0]


def test_dependent_pairs_are_left_out():
    rows = [
        row(
            pair="spec-0651-52141-0072__spec-0651-52200-0072",
            id_a="spec-0651-52141-0072",
            id_b="spec-0651-52200-0072",
        )
    ]
    assert verdict(rows)[0] == "none"
    rows = [row(flags="dependent")]
    assert verdict(rows)[0] == "none"


def test_classify_table_rows_and_order():
    rows = [row(), dict(row(s=100.0), targetid="t2"), dict(row(snr_a=1.0), targetid="t3")]
    out = tiers.classify_table(rows, masses={}, classes={"t2": {"Halpha": "A"}})
    assert [r["targetid"] for r in out] == ["t1", "t2", "t3"]
    assert [r["tier"] for r in out] == ["binary", "stable", "none"]
    assert list(out[0]) == list(tiers.TIER_COLUMNS) and out[0]["n_pairs"] == 1
    assert out[0]["sigma"] == pytest.approx(9.0) and out[0]["two_line_agree"] == ""
    assert np.isnan(out[1]["s_kms"]) and out[1]["line"] == ""
    # strings as a CSV reader gives them
    text = {k: ("" if v is None else str(v)) for k, v in row().items()}
    assert tiers.classify_table([text])[0]["tier"] == "binary"
