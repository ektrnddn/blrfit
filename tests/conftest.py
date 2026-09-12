import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
EXAMPLES = os.path.join(ROOT, "examples", "data")
sys.path.insert(0, HERE)      # tests.synth

SDSS_EXAMPLE = os.path.join(EXAMPLES, "spec-0651-52141-0072.fits")       # SDSS J001224.01-102226.5, 2001
SDSS_EXAMPLE_2 = os.path.join(EXAMPLES, "spec-7169-56628-0344.fits")     # the same object, 2013
DESI_EXAMPLE = os.path.join(EXAMPLES, "coadd-main-dark-17260-39627574082538900.fits")  # SDSS J001247.93-084700.5
DESI_TARGETID = 39627574082538900
CSV_EXAMPLE = os.path.join(EXAMPLES, "J001224_rest_air_nm.csv")
Z_J001224 = 0.2288


@pytest.fixture(scope="session")
def data_dir():
    return DATA


@pytest.fixture(scope="session")
def examples_dir():
    return EXAMPLES
