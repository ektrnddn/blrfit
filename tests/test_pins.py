"""
Regression pins: four SDSS spectra with the exact output of the fitter that
produced the DESI catalogue (summary row, fitted parameters, chi-square).

The pins were written by the frozen production code; the package must
reproduce them. On the numerical stack that wrote them (numpy 1.26.4, scipy
1.13.1, macOS arm64) the agreement is bit for bit (set BLRFIT_STRICT_PINS=1 to
require it). On other stacks and platforms the bounded trust-region solver
converges to slightly different points, and the same numpy and scipy versions
on Linux differ as well (another BLAS, whose thread count also enters):
measured against numpy 2.5 / scipy 1.18 on macOS and on the Linux runners of
the test workflow, the pinned bisector velocities move by up to 17 km/s, the
centroids by up to 67 km/s (the wing-sensitive first moment of a two-component
Halpha), the widths by up to 25 km/s, the second-moment width by up to 84 km/s,
chi-square by up to 4 per cent, and the peak of the two-humped class-B pin by
60 km/s, the peak of a flat-topped or two-humped profile being the unstable
statistic that makes c(1/2) the primary offset; no class or flag changes
anywhere. The default tolerances sit above that spread (30 km/s on the bisector
velocities, 15 km/s on the narrow-line systemic velocity and width, 100 km/s on
the centroids and the peak velocities, 60 km/s on the widths, 150 km/s on the
second-moment width, 8 per cent on chi-square and 5 per cent on the
signal-to-noise ratios) plus equality of classes, flags, component counts and
systemic sources; the pinned spectra were chosen away from the class
thresholds so that a few tens of km/s cannot flip a class. Every departure of
a spectrum is reported in one message. Bit-for-bit reproduction is checked on
the reference stack before a release; the tolerant comparison guards the
physics everywhere else.
"""
import json
import os

import numpy as np
import pytest

import blrfit
from blrfit.io import read_sdss
from conftest import DATA, EXAMPLES

STRICT = os.environ.get("BLRFIT_STRICT_PINS", "") not in ("", "0")
VEL_KEYS = ("v_sys", "sig_sys", "v_o3", "v_peak", "centroid", "c25", "c50", "c75", "fwhm", "W25", "W75",
            "v_peak_sys", "centroid_sys", "c50_sys", "peak_top_sys", "sigma_line")


def tolerance_kms(key):
    """Allowed departure from the pin for one velocity or width statistic, km/s."""
    if key == "sigma_line":
        return 150.0
    if key.startswith(("v_peak", "peak_top", "centroid")):
        return 100.0
    if key in ("fwhm", "W25", "W75"):
        return 60.0
    if key in ("v_sys", "sig_sys", "v_o3"):
        return 15.0
    return 30.0                                   # the bisector velocities c25, c50, c75, c50_sys


def _pins():
    with open(os.path.join(DATA, "pins.json")) as fh:
        return json.load(fh)["pins"]


def _spectrum_path(fn):
    for d in (EXAMPLES, DATA):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


@pytest.mark.parametrize("pin", _pins(), ids=lambda p: p["file"])
def test_pin_reproduced(pin):
    sp = read_sdss(_spectrum_path(pin["file"]))
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], pin["z"], ebv=pin["ebv"],
                              complexes=tuple(pin["complexes"]))
    row = blrfit.summary_row(res)
    ref = pin["summary"]
    assert set(res["fits"]) == set(pin["params"])
    departures = []
    for name, pref in pin["params"].items():
        d = res["fits"][name]["d"]
        assert set(d) == set(pref)
        p = {"Halpha": "HA", "Hbeta": "HB", "MgII": "MG"}[name]
        for k in ("class", "flags", "n_broad", "systemic_source"):
            assert row[f"{p}_{k}"] == ref[f"{p}_{k}"], (name, k, row[f"{p}_{k}"], ref[f"{p}_{k}"])
        if STRICT:
            for k, v in pref.items():
                assert d[k] == v, (name, k)
            assert res["fits"][name]["chi2"] == pin["chi2"][name]
            continue
        chi2, chi2_ref = res["fits"][name]["chi2"], pin["chi2"][name]
        if chi2 != pytest.approx(chi2_ref, rel=0.08):
            departures.append(f"{name} chi2 {chi2:.2f} vs pinned {chi2_ref:.2f}")
        for k in VEL_KEYS:
            r = ref.get(f"{p}_{k}")
            got = row[f"{p}_{k}"]
            if r is None:
                assert not np.isfinite(got), (name, k, got)
            elif not abs(got - r) < tolerance_kms(k):
                departures.append(f"{name} {k} {got:.1f} vs pinned {r:.1f} km/s (tolerance {tolerance_kms(k):.0f})")
        for k in ("AI", "KI"):
            if row[f"{p}_{k}"] != pytest.approx(ref[f"{p}_{k}"], abs=0.03):
                departures.append(f"{name} {k} {row[f'{p}_{k}']:.3f} vs pinned {ref[f'{p}_{k}']:.3f}")
        for k in ("broad_flux_snr", "broad_peak_snr", "sys_snr"):
            if row[f"{p}_{k}"] != pytest.approx(ref[f"{p}_{k}"], rel=0.05):
                departures.append(f"{name} {k} {row[f'{p}_{k}']:.2f} vs pinned {ref[f'{p}_{k}']:.2f}")
    assert not departures, pin["file"] + ":\n  " + "\n  ".join(departures)
    if STRICT:
        for k, v in ref.items():
            got = row[k]
            if v is None:
                assert isinstance(got, float) and np.isnan(got), k
            else:
                assert got == v, k


def test_pin_values_are_the_catalogue_values():
    """The four pins span the classes: C (the worked example, both lines), A, B, F."""
    pins = {p["file"]: p for p in _pins()}
    assert pins["spec-0651-52141-0072.fits"]["summary"]["HB_class"] == "C"
    assert pins["spec-0651-52141-0072.fits"]["summary"]["HA_class"] == "C"
    assert abs(pins["spec-0651-52141-0072.fits"]["summary"]["HB_c50_sys"] - (-1243.5)) < 1
    assert pins["spec-1237-52762-0298.fits"]["summary"]["HB_class"] == "A"
    assert abs(pins["spec-1237-52762-0298.fits"]["summary"]["HB_c50_sys"] - (-2130.5)) < 1
    assert pins["spec-1592-52990-0139.fits"]["summary"]["HB_class"] == "B"
    assert pins["spec-1704-53178-0562.fits"]["summary"]["HB_class"] == "F"
    assert abs(pins["spec-1704-53178-0562.fits"]["summary"]["HB_c50_sys"]) < 120


def test_tolerances_cover_the_measured_spread():
    assert tolerance_kms("c50") == 30.0 and tolerance_kms("c50_sys") == 30.0
    assert tolerance_kms("centroid") == 100.0 and tolerance_kms("centroid_sys") == 100.0
    assert tolerance_kms("v_peak") == 100.0 and tolerance_kms("peak_top_sys") == 100.0
    assert tolerance_kms("fwhm") == 60.0 and tolerance_kms("sigma_line") == 150.0
    assert tolerance_kms("v_sys") == 15.0
