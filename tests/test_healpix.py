"""The pure-numpy nested HEALPix index against reference values from healpy."""
import json
import os

import numpy as np
import pytest

from blrfit.io.healpix import ang2pix_nest
from conftest import DATA


def test_reference_values():
    ref = json.load(open(os.path.join(DATA, "healpix_reference.json")))
    assert np.array_equal(ang2pix_nest(ref["nside"], np.array(ref["ra"]), np.array(ref["dec"])), np.array(ref["pix"]))
    for ra, dec, pix in ref["special"]:
        assert ang2pix_nest(ref["nside"], ra, dec) == pix


def test_example_positions():
    assert ang2pix_nest(64, 3.1997083, -8.7834722) == 17260     # the DESI example object
    assert ang2pix_nest(64, 3.1000417, -10.3740278) == 17251


def test_scalar_and_array_and_bad_nside():
    assert isinstance(ang2pix_nest(64, 10.0, 10.0), int)
    assert ang2pix_nest(64, [10.0, 20.0], [10.0, -20.0]).shape == (2,)
    with pytest.raises(ValueError):
        ang2pix_nest(63, 0.0, 0.0)


def test_against_healpy():
    hp = pytest.importorskip("healpy")
    rng = np.random.default_rng(11)
    for nside in (1, 8, 64, 512):
        ra = rng.uniform(0, 360, 50000); dec = np.degrees(np.arcsin(rng.uniform(-1, 1, 50000)))
        assert np.array_equal(ang2pix_nest(nside, ra, dec), hp.ang2pix(nside, ra, dec, lonlat=True, nest=True))
    ra = np.repeat(np.arange(0, 360, 1.0), 181); dec = np.tile(np.arange(-90, 91, 1.0), 360)
    assert np.array_equal(ang2pix_nest(64, ra, dec), hp.ang2pix(64, ra, dec, lonlat=True, nest=True))
