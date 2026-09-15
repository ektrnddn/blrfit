"""
Reproducibility of the end point of ``fit_spectrum``.

Two properties are held. (a) A rescaling of the flux by 1 + 1e-13, with the
inverse variance divided by its square, changes the input in its last bit and
nothing else; the fit must come back with the same classes, flags, component
counts and systemic sources, and with every velocity and width of the summary
row of both lines within REPRO_KMS = 0.1 km/s (one three-thousandth of the
300 km/s class threshold). 0.1.0 rounded the Fe II broadening width to 50 km/s
inside the continuum solver, which then saw no width derivative and ended
where the last bit of the data took it: under this rescaling the frozen 0.1.0
release moves c50_sys of Hbeta of the 2013 epoch of J001224
(spec-7169-56628-0344) by 30 km/s, its FWHM by 72 km/s and the Fe II width
from 2846 to 1519 km/s (reference stack: numpy 1.26.4, scipy 1.13.1, macOS
arm64), while on the 2001 epoch, whose Fe II norm ends at zero, nothing moves.
0.2.0 evaluates the width directly; ``tests/test_fe_width.py`` measures the
recovery, and its ``test_rescaled_input_reproduces_width_and_velocity`` holds
the Fe II width and c50_sys of the Fe-strong spec-1592 (Hbeta only) under the
same rescaling. This test covers every velocity key of both lines on the two
SDSS epochs of J001224 (spec-0651-52141-0072 and spec-7169-56628-0344, host
and Fe II on, the defaults).

Two kinds of entry are excepted, and the exceptions are pinned per epoch so
that they cannot widen unnoticed. A parameter that multiplies nothing at the
end point is unconstrained by the data: the Fe II width where the Fe II norm
ended on its zero bound (``continuum_info['at_bound']`` lists ``feop_norm``;
the 2001 epoch) and the narrow-line-region wing width where the wing fraction
ended on its zero bound (the active bounds of the selected solver attempt;
the 2001 epoch, whose Hbeta takes the Halpha wing as a fixed prior). Both are
compared only when defined. The second-moment width ``sigma_line`` weights
the far wings with the velocity squared, where the flattest directions of a
three-component decomposition lie, and is held to REPRO_MOMENT_KMS = 0.5
km/s: on the 2001 epoch it moves by 0.11 km/s while every offset moves by less
than 0.02 km/s (between platforms it is the least stable statistic of all,
see ``tests/test_pins.py``).

The 0.1 km/s holds on the numerical stack of the catalogue run (numpy 1.26,
scipy 1.13; macOS and the Linux runner) and under BLRFIT_STRICT_PINS. With
current releases (numpy 2, scipy 1.16, Linux) the same rescaling moves the
Halpha centroid of the 2001 epoch by 0.29 km/s and its W25 by 0.32 km/s: there
the host-decomposed end point is reproducible to about 1 km/s (CHANGELOG,
known open items), and the tolerance is REPRO_KMS_OTHER = 1.5 km/s
(REPRO_MOMENT_KMS_OTHER = 3 km/s for sigma_line), twenty times below the
30 km/s that the 0.1.0 width rounding produced. Classes, flags, component
counts and systemic sources must agree exactly on every stack.

(b) Two identical calls give identical results, every number of the summary
row, every fitted parameter and the continuum bit for bit: the fit has no
hidden state.
"""
import numpy as np
import pytest
import scipy

import blrfit
from blrfit.constants import C_KMS
from blrfit.io import read_sdss
from blrfit.model.fit import PREFIX
from conftest import SDSS_EXAMPLE, SDSS_EXAMPLE_2, Z_J001224
from test_pins import KMS_STATS, STRICT

EPS = 1e-13
# the numerical stack of the catalogue run, on which the end point is held to 0.1 km/s
REFERENCE_STACK = np.__version__.startswith("1.26.") and scipy.__version__.startswith("1.13.")
REPRO_KMS_OTHER, REPRO_MOMENT_KMS_OTHER = 1.5, 3.0      # other stacks, see the module docstring
REPRO_KMS = 0.1 if (REFERENCE_STACK or STRICT) else REPRO_KMS_OTHER                # every velocity and width
REPRO_MOMENT_KMS = 0.5 if (REFERENCE_STACK or STRICT) else REPRO_MOMENT_KMS_OTHER  # the second-moment width
EPOCHS = {"2001": SDSS_EXAMPLE, "2013": SDSS_EXAMPLE_2}
# the entries undefined at the end point, per epoch (see the module docstring)
UNDEFINED = {"2001": {"conti_feop_fwhm", "HA_nw_sig", "HB_nw_sig"}, "2013": set()}


def _fit(path, scale=1.0):
    sp = read_sdss(path)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"] * scale, sp["ivar"] / scale ** 2, Z_J001224)
    return res, blrfit.summary_row(res)


@pytest.fixture(scope="module")
def base_fits():
    return {epoch: _fit(path) for epoch, path in EPOCHS.items()}


def _same(a, b):
    return a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))


def _stat(key):
    return key.split("_", 1)[1] if key[:3] in ("HA_", "HB_", "MG_") else key


def _undefined(res):
    """Summary keys of parameters that multiply nothing at the end point."""
    out = set()
    if "feop_norm" in res["continuum_info"]["at_bound"]:
        out.add("conti_feop_fwhm")
    ha = res["fits"].get("Halpha")
    if ha is not None:
        selected = ha["solver"]["attempts"][ha["solver"]["selected_attempt"]]
        if selected["active_bounds"].get("nw_f") == -1:
            out |= {f"{PREFIX[name]}_nw_sig" for name in res["fits"]}
    return out


def _velocity_departures(ra, rb, skip):
    out = []
    for k, va in ra.items():
        stat = _stat(k)
        if k in skip or not (stat in KMS_STATS or stat == "z_sys"):
            continue
        vb = rb[k]
        if np.isnan(va) and np.isnan(vb):
            continue
        d = abs(va - vb) * (C_KMS if stat == "z_sys" else 1.0)
        tol = REPRO_MOMENT_KMS if stat == "sigma_line" else REPRO_KMS
        if not d <= tol:
            out.append(f"{k} {va!r} vs {vb!r}: {d:.4f} km/s (tolerance {tol})")
    return out


@pytest.mark.parametrize("epoch", sorted(EPOCHS), ids=lambda e: f"J001224-{e}")
def test_last_bit_rescaling_leaves_every_velocity(epoch, base_fits):
    res_a, ra = base_fits[epoch]
    res_b, rb = _fit(EPOCHS[epoch], 1.0 + EPS)
    assert set(res_a["fits"]) == {"Halpha", "Hbeta"} == set(res_b["fits"])
    assert set(ra) == set(rb)
    skip = _undefined(res_a) | _undefined(res_b)
    assert skip == UNDEFINED[epoch], skip
    departures = _velocity_departures(ra, rb, skip)
    for name in res_a["fits"]:
        p = PREFIX[name]
        for k in ("class", "flags", "n_broad", "systemic_source"):
            if ra[f"{p}_{k}"] != rb[f"{p}_{k}"]:
                departures.append(f"{p}_{k} {ra[f'{p}_{k}']!r} vs {rb[f'{p}_{k}']!r}")
    assert not departures, f"J001224 {epoch}:\n  " + "\n  ".join(departures)


def test_identical_calls_give_identical_results(base_fits):
    res_a, ra = base_fits["2001"]
    res_b, rb = _fit(EPOCHS["2001"])
    assert set(ra) == set(rb)
    differing = [k for k in ra if not _same(ra[k], rb[k])]
    assert not differing, differing
    assert set(res_a["fits"]) == set(res_b["fits"])
    for name in res_a["fits"]:
        assert res_a["fits"][name]["d"] == res_b["fits"][name]["d"], name
        assert res_a["fits"][name]["chi2"] == res_b["fits"][name]["chi2"], name
        assert list(res_a["fits"][name]["all_bic"]) == list(res_b["fits"][name]["all_bic"]), name
    assert res_a["conti"] == res_b["conti"]
    for k in ("wave_rest", "flux_rest", "ivar_rest", "conti_model", "host_model"):
        assert np.array_equal(res_a[k], res_b[k]), k
