"""
Regression pins: four SDSS spectra with the exact output of the fitter that
produced the DESI catalogue (summary row, fitted parameters, chi-square).

The pins were written by the frozen production code; the package must
reproduce them. On the numerical stack that wrote them (numpy 1.26.4, scipy
1.13.1) the agreement is bit for bit (set BLRFIT_STRICT_PINS=1 to require it).
Across versions the bounded trust-region solver converges to slightly different
points: between that stack and numpy 2.5 / scipy 1.18 the pinned bisector
and peak velocities moved by up to 10 km/s, the centroid by up to 22 km/s, the
widths by up to 25 km/s (the second-moment width by 32 km/s on macOS and 84 km/s
on Linux) and chi-square by
up to 4 per cent, without changing any class or flag. The default tolerances are set
to that spread (15 km/s on the bisector and peak velocities, 60 km/s on the
wing-sensitive centroid and on the widths, 150 km/s on the second-moment width,
8 per cent on chi-square and 5 per cent on the
signal-to-noise ratios) plus equality of classes, flags and
component counts; the pinned spectra were chosen away from the class
thresholds so that a few km/s cannot flip a class.
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
    for name, pref in pin["params"].items():
        d = res["fits"][name]["d"]
        assert set(d) == set(pref)
        p = {"Halpha": "HA", "Hbeta": "HB", "MgII": "MG"}[name]
        assert row[f"{p}_class"] == ref[f"{p}_class"], name
        assert row[f"{p}_flags"] == ref[f"{p}_flags"], name
        assert row[f"{p}_n_broad"] == ref[f"{p}_n_broad"], name
        assert row[f"{p}_systemic_source"] == ref[f"{p}_systemic_source"]
        if STRICT:
            for k, v in pref.items():
                assert d[k] == v, (name, k)
            assert res["fits"][name]["chi2"] == pin["chi2"][name]
        else:
            assert res["fits"][name]["chi2"] == pytest.approx(pin["chi2"][name], rel=0.08)
            for k in VEL_KEYS:
                r = ref.get(f"{p}_{k}")
                if r is None:
                    assert not np.isfinite(row[f"{p}_{k}"])
                else:
                    # widths and the wing-sensitive first moment move more than the bisectors
                    # the second moment (sigma_line) of one pin moved by 84 km/s on Linux with current numpy/scipy
                    tol = 150.0 if k == "sigma_line" else (60.0 if k in ("fwhm", "W25", "W75") or k.startswith("centroid") else 15.0)
                    assert abs(row[f"{p}_{k}"] - r) < tol, (name, k, row[f"{p}_{k}"], r)
            for k in ("AI", "KI"):
                assert row[f"{p}_{k}"] == pytest.approx(ref[f"{p}_{k}"], abs=0.03)
            for k in ("broad_flux_snr", "broad_peak_snr", "sys_snr"):
                assert row[f"{p}_{k}"] == pytest.approx(ref[f"{p}_{k}"], rel=0.05)
    if STRICT:
        for k, v in ref.items():
            got = row[k]
            if v is None:
                assert isinstance(got, float) and np.isnan(got), k
            elif isinstance(v, float):
                assert got == v, k
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
