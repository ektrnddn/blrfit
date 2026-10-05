"""The broad search compares the same profile pixels at every trial shift."""

import numpy as np
import pytest

from blrfit import rv


def profile(v, shift=0.0, holes=()):
    ok = np.ones(len(v), bool)
    for start, stop in holes:
        ok[start:stop] = False
    return dict(
        v=v,
        f=np.exp(-0.5 * ((v - shift) / 1600.0) ** 2),
        e=np.full(len(v), 0.05),
        ok=ok,
        nmod=np.zeros(len(v)),
        fwhm=3767.0,
        c50_sys=shift,
        v_sys=0.0,
    )


@pytest.mark.parametrize("holes", [[(280, 285)], [(140, 145), (315, 317)], [(285, 300)]])
def test_internal_template_gaps_keep_identical_profile_positions(monkeypatch, holes):
    v = np.arange(-9000.0, 9001.0, 30.0)
    positions = []
    original = rv._profile_eiv

    def traced(y, x, gv, var_y, var_x, keep, baseline):
        positions.append(gv.copy())
        return original(y, x, gv, var_y, var_x, keep, baseline)

    monkeypatch.setattr(rv, "_profile_eiv", traced)
    result = rv.ccf_shift(
        profile(v), profile(v, holes=holes), vmax=1000.0, despike=False, clip=np.inf, n_clip=0
    )
    assert result is not None and result["two_stage"]
    assert result["npix_search"][0] == result["npix_search"][1]
    ntrial = int(round(np.ptp(result["search_range"]) / result["dv_pix"])) + 1
    assert ntrial > 5
    # Equal counts alone can hide changing pixel identities around a gap.
    assert all(np.array_equal(positions[0], p) for p in positions[1:ntrial])
    assert abs(result["dv"]) < 1.0


def test_disjoint_support_refuses_variable_pixel_fallback():
    v = np.arange(-9000.0, 9001.0, 30.0)
    a, b = profile(v), profile(v)
    a["ok"][:400] = False
    b["ok"][200:] = False
    assert rv.ccf_shift(a, b, vmax=7000.0, min_pix=30) is None


@pytest.mark.parametrize("shift", [-600.0, 0.0, 600.0])
def test_ungapped_shift_and_estimator_identity(shift):
    v = np.arange(-9000.0, 9001.0, 30.0)
    result = rv.shift_bidirectional(profile(v, shift), profile(v), vmax=1000.0)
    assert result is not None
    assert result["dv"] == pytest.approx(shift, abs=1.0)
    assert result["algorithm_version"] == "profile-eiv-v3"
    for direction in (result["s_ab"], result["s_ba"]):
        assert direction["algorithm_version"] == "profile-eiv-v3"
        assert direction["npix_search"][0] == direction["npix_search"][1]
