"""Check the declared experiment, not a fitter accuracy claim."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from blrfit.model.continuum import HOST_WINDOW, host_window_statistics, pca_templates

spec = importlib.util.spec_from_file_location(
    "host_rich_weak", Path(__file__).parents[1] / "tools/host_rich_weak.py"
)
HW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(HW)


class ZeroNoise:
    def standard_normal(self, n):
        return np.zeros(n)


@pytest.mark.parametrize("host,snr", [(0.3, 1), (0.6, 5)])
def test_sum_snr_is_the_guard_statistic_on_noiseless_expectation(host, snr):
    wave, flux, ivar, truth = HW.make_spectrum(0.1, host, snr, "sum", 600, 4000, 150, ZeroNoise())
    wr = wave / 1.1
    templates = pca_templates()
    inhost = (wr > templates["gw"].min() + 2) & (wr < templates["gw"].max() - 2)
    record = host_window_statistics(wr, flux, ivar, np.ones(len(wave), bool), inhost)
    assert record["snr"] == pytest.approx(snr, rel=1e-12)
    assert truth["expected_sum_snr"] == pytest.approx(snr, rel=1e-12)
    assert truth["n_band_pixels"] == np.count_nonzero((wr > HOST_WINDOW[0]) & (wr < HOST_WINDOW[1]))


def test_simulated_noise_is_repeatable_and_has_recorded_input_identity():
    args = (0.1, 0.45, 3, "sum", 600, 4000, 150)
    a = HW.make_spectrum(*args, np.random.default_rng(7))
    b = HW.make_spectrum(*args, np.random.default_rng(7))
    c = HW.make_spectrum(*args, np.random.default_rng(8))
    assert HW.array_identity(*a[:3]) == HW.array_identity(*b[:3])
    assert HW.array_identity(*a[:3]) != HW.array_identity(*c[:3])


def test_blue_weak_family_keeps_declared_blue_sum_and_usable_red_noise():
    wave, flux, ivar, truth = HW.make_spectrum(
        0.1, 0.6, 2, "sum", 600, 4000, 150, ZeroNoise(), noise_profile="blue_weak"
    )
    wr = wave / 1.1
    blue = (wr > 4200) & (wr < 5000)
    red = (wr > 6800) & (wr < 6900)
    assert np.sum(ivar[blue] * flux[blue]) / np.sqrt(np.sum(ivar[blue])) == pytest.approx(2)
    assert np.median(ivar[red]) > np.median(ivar[blue]) * 100
    assert truth["red_pixel_snr"] == 15 and truth["noise_profile"] == "blue_weak"


def rows(errors, measurable=True):
    return [
        dict(
            truth={"c50_sys": 600},
            fits={
                "plfe": dict(
                    status="returned",
                    host_undetermined=False,
                    lines={
                        "Halpha": dict(
                            c50_sys=600 + v, label="A" if measurable else "W", measurable=measurable
                        )
                    },
                )
            },
        )
        for v in errors
    ]


def test_no_measurable_lines_cannot_establish_accuracy():
    result = HW.summarize_cell(rows(np.zeros(20), False), "Halpha", "plfe", 1)
    assert result["n_finite"] == 20 and result["n_measurable"] == 0
    assert result["bias_decision"]["outcome"] == "inconclusive"
    assert result["bias_decision"]["estimate"] is None


def test_insufficient_measurable_noise_realizations_cannot_establish_accuracy():
    result = HW.summarize_cell(rows(np.zeros(19)), "Halpha", "plfe", 1)
    assert result["bias_decision"]["outcome"] == "inconclusive"


def test_bias_failure_and_pass_are_separate_from_conditional_population_scope():
    for error, expected in [(0, "pass"), (100, "fail")]:
        result = HW.summarize_cell(rows(np.full(20, error)), "Halpha", "plfe", 1)
        assert result["bias_decision"]["outcome"] == expected
        assert result["bias_decision"]["n_groups"] == 1
        assert not result["physical_population_validation"]


def test_existing_output_refused_before_any_fit(tmp_path):
    with pytest.raises(FileExistsError):
        HW.main(["--out", str(tmp_path), "--n", "1"])
