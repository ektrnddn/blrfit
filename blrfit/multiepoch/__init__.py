"""Velocity curves and verdicts from every pair of dated spectra of an object (``blrfit multiepoch``).

The chain, in order, each stage a function that reads tables and writes a directory:

1. ``calibrate.run_calibration``: the noise model, calibrated on objects whose broad lines should not move (a per-spectrum
   jitter by instrument, an offset of one object's SDSS or BOSS spectra against its DESI ones, an independent per-pair
   term from closed loops, and the wandering of the broad line as a damped random walk in units of its width).
2. ``run.run_verdicts``: the spectrum check (``spectra``), then for every object and line a generalized-least-squares
   velocity curve on all its measured pairs with a stable profile and the verdict (``verdicts``): moving (the motion test
   at p < 2.7e-3), constant, profile variable or not measurable, with the tags (steady, allowed by its mass, disappears
   when one spectrum or one time is excluded, ...).
3. ``orbits.run_orbits``: the circular orbits the virial mass allows, fitted to every curve with three or more times.
4. ``population.run_population``: strong offsets against matched objects without one.
5. ``fraction.run_fraction``: the fraction of strong offsets that can be binaries, against the period.

``groups`` sorts the robust movers by behavior. ``common.Inputs`` names the tables every stage reads.

    from blrfit.multiepoch import Inputs, run_calibration, run_verdicts, run_orbits
    inp = Inputs(pairs="pairs.csv", catalog="catalog.fits", masses="masses.fits", facts="facts.csv")
    run_calibration(inp, "out")
    run_verdicts(inp, "out/CALIBRATION.json", "out")
    run_orbits("out", inp)
"""

from .calibrate import run_calibration
from .common import Inputs
from .fraction import run_fraction
from .groups import GROUPS, ORBIT_GROUPS, group_of, orbit_group_of
from .orbits import run_orbits
from .population import run_population
from .run import run_verdicts
from .spectra import check as check_spectra
from .verdicts import Calibration, analyze

__all__ = [
    "Inputs",
    "run_calibration",
    "run_verdicts",
    "run_orbits",
    "run_population",
    "run_fraction",
    "check_spectra",
    "analyze",
    "Calibration",
    "GROUPS",
    "ORBIT_GROUPS",
    "group_of",
    "orbit_group_of",
]
