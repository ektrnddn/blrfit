"""
blrfit: broad AGN emission lines fitted against the narrow-line systemic
velocity, classified and flagged; velocity changes between the epochs of one
object (``blrfit.pairs``) and candidate tiers from them (``blrfit.tiers``).

    from blrfit import read_spectrum, fit_spectrum, summary_row, plot_fit
    sp  = read_spectrum("spec-0651-52141-0072.fits")
    res = fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z=0.2288, complexes=("Halpha", "Hbeta"))
    res["meas"]["Hbeta"]["c50_sys"], res["cls"]["Hbeta"]["label"]

    from blrfit import measure_pair
    rec = measure_pair(res_2001, res_2013, "Hbeta")
    rec["s_common"], rec["err"], rec["retained"], rec["flags"]

The command-line tool ``blrfit fit`` does the same and writes a JSON summary
and a diagnostic figure; ``blrfit pair`` fits the epochs of one object and
measures their changes; ``blrfit tiers`` classifies objects from pair tables.
The earlier ``blrfit.rv`` remains available from Python. See the README for the
model and its validation.
"""

from .model.fit import fit_spectrum, remeasure, summary_row
from .measure import measure_complex, profile_measures
from .classify import classify, is_measurable, is_strong_offset, DEFAULT_THRESH, LABEL_TEXT, FLAG_TEXT
from .errors import monte_carlo, empirical_error
from .plot import (
    plot_fit,
    plot_epochs_overlay,
    plot_ccf,
    broad_residual_profile,
    rv_curve,
    plot_rv_curve,
    plot_pair,
)
from .io import read_spectrum, read_sdss, read_desi, read_table
from .physics import lambda_l_lambda, continuum_luminosity, target_mass
from .pairs import measure_pair, pair_record, enumerate_pairs
from .tiers import classify_target, classify_table
from . import constants, rv, physics, pairs, tiers

__version__ = "0.4.0.dev0"

__all__ = [
    "fit_spectrum",
    "remeasure",
    "summary_row",
    "measure_complex",
    "profile_measures",
    "classify",
    "is_measurable",
    "is_strong_offset",
    "DEFAULT_THRESH",
    "LABEL_TEXT",
    "FLAG_TEXT",
    "monte_carlo",
    "empirical_error",
    "plot_fit",
    "plot_epochs_overlay",
    "plot_ccf",
    "broad_residual_profile",
    "rv_curve",
    "plot_rv_curve",
    "read_spectrum",
    "read_sdss",
    "read_desi",
    "read_table",
    "plot_pair",
    "lambda_l_lambda",
    "continuum_luminosity",
    "target_mass",
    "measure_pair",
    "pair_record",
    "enumerate_pairs",
    "classify_target",
    "classify_table",
    "constants",
    "rv",
    "physics",
    "pairs",
    "tiers",
    "__version__",
]
