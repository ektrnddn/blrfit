"""
Galactic E(B-V) of the command line: the SFD98 value for every survey. DESI
coadds carry it in their FIBERMAP; SDSS and generic inputs get it from the SFD98
map at their coordinates when dustmaps is installed, and otherwise 0 with a
warning and the flag ebv_assumed_zero. (conftest.py hides dustmaps from every
test; the stand-in below provides a map.)
"""

import json
import sys
import types

import numpy as np
import pytest

from blrfit.cli import main
from blrfit.io import dust
from conftest import SDSS_EXAMPLE, Z_J001224

ARGS = [
    "fit",
    SDSS_EXAMPLE,
    "--survey",
    "sdss",
    "--z",
    str(Z_J001224),
    "--lines",
    "Hbeta",
    "--no-figure",
    "--quiet",
]


def _doc(tmp_path):
    with open(tmp_path / "spec-0651-52141-0072_fit.json") as fh:
        return json.load(fh)


@pytest.fixture
def stand_in_map(monkeypatch):
    class SFDQuery:
        def __call__(self, coord):
            return np.float64(0.031)

    package, module = types.ModuleType("dustmaps"), types.ModuleType("dustmaps.sfd")
    module.SFDQuery = SFDQuery
    package.sfd = module
    monkeypatch.setattr(dust, "_QUERY", None)
    monkeypatch.setitem(sys.modules, "dustmaps", package)
    monkeypatch.setitem(sys.modules, "dustmaps.sfd", module)


def test_sdss_without_a_map_is_flagged(tmp_path):
    assert main(ARGS + ["--out", str(tmp_path)]) == 0
    doc = _doc(tmp_path)
    assert doc["input"]["ebv"] == 0.0 and doc["input"]["ebv_assumed_zero"] is True
    assert doc["input"]["ebv_source"] == "assumed 0: dustmaps not installed"
    assert "ebv_assumed_zero" in doc["lines"]["Hbeta"]["flags"]


def test_sdss_with_the_sfd_map(tmp_path, stand_in_map):
    assert main(ARGS + ["--out", str(tmp_path)]) == 0
    doc = _doc(tmp_path)
    assert doc["input"]["ebv"] == pytest.approx(0.031) and doc["input"]["ebv_source"] == "SFD map"
    assert doc["input"]["ebv_assumed_zero"] is False
    assert "ebv_assumed_zero" not in doc["lines"]["Hbeta"]["flags"]


def test_an_explicit_value_wins(tmp_path):
    assert main(ARGS + ["--ebv", "0.02", "--out", str(tmp_path)]) == 0
    doc = _doc(tmp_path)
    assert doc["input"]["ebv"] == 0.02 and doc["input"]["ebv_source"] == "argument"
    assert doc["input"]["ebv_assumed_zero"] is False


def test_explicit_sfd_without_a_map_stops(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(ARGS + ["--ebv", "sfd", "--out", str(tmp_path)])
    assert "dustmaps not installed" in str(exc.value)


def test_sfd_ebv_needs_coordinates():
    assert dust.sfd_ebv(None, None) == (None, "no coordinates")
    assert dust.sfd_ebv(np.nan, 10.0) == (None, "no coordinates")
