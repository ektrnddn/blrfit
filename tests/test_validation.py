"""The preregistered tolerance examples and boundary/missing-evidence rules."""
import math

import pytest

from blrfit.validation import coverage_decision, tolerance_decision


def decide(interval, estimate=10, **kwargs):
    return tolerance_decision(estimate, interval, (-30, 30), units="km/s",
                              n_observations=40, n_groups=20,
                              method="declared group interval", **kwargs)


@pytest.mark.parametrize("interval, expected", [
    ([5, 15], "pass"), ([-5, 45], "inconclusive"),
    ([-30, 30], "pass"), ([30, 30], "pass"),
    ([30, 45], "inconclusive"), ([-45, -30], "inconclusive"),
    ([31, 45], "fail"), ([-45, -31], "fail"),
    ([-40, 40], "inconclusive"), ([None, None], "inconclusive"),
    ([math.nan, 15], "inconclusive"), (None, "inconclusive"),
])
def test_interval_relation(interval, expected):
    assert decide(interval)["outcome"] == expected


def test_zero_compatibility_is_not_required_for_practical_accuracy():
    assert decide([5, 15])["outcome"] == "pass"
    assert decide([-5, 45])["outcome"] == "inconclusive"


def test_coverage_has_its_own_tolerance():
    args = dict(nominal=.683, tolerance=.08, n_observations=200,
                n_groups=100, method="group bootstrap")
    assert coverage_decision(.69, [.65, .72], **args)["outcome"] == "pass"
    assert coverage_decision(.65, [.5, .8], **args)["outcome"] == "inconclusive"
    assert coverage_decision(.5, [.45, .55], **args)["outcome"] == "fail"


def test_missing_estimate_or_counts_cannot_pass():
    assert decide([5, 15], estimate=math.nan)["outcome"] == "inconclusive"
    out = tolerance_decision(10, [5, 15], [-30, 30], units="km/s",
                             n_observations=0, n_groups=0, method="no observations")
    assert out["outcome"] == "inconclusive"


def test_metadata_is_retained():
    out = decide([5, 15])
    assert out["confidence"] == .95 and out["units"] == "km/s"
    assert out["n_observations"] == 40 and out["n_groups"] == 20
    assert out["tolerance_region"] == [-30, 30]


@pytest.mark.parametrize("region", [[31, 30], [math.nan, 30], [-30, math.inf]])
def test_invalid_region_refused(region):
    with pytest.raises(ValueError):
        tolerance_decision(0, [-1, 1], region, units="km/s",
                           n_observations=2, n_groups=1, method="test")


def test_reversed_interval_refused():
    with pytest.raises(ValueError, match="reversed"):
        decide([15, 5])
