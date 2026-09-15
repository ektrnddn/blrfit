"""
Regression pins: four SDSS spectra with the exact output of the fitter that
produced the DESI catalogue (``tests/data/pins.json``: summary row, fitted
parameters, chi-square and BIC of every line complex, continuum parameters,
host information and the [O III] pre-fit).

These are frozen 0.1.0 fixtures. The historical parameter-reconstruction test
explicitly restores that release's rounded Fe-width operator in a scoped
monkeypatch. The fresh-fit test uses the current, continuous operator. Thus
historical reproduction does not require reintroducing quantization into the
production model. Current Fe-width recovery is tested in test_corrections.py.

The check has two parts, because the pins hold two kinds of number.

The first part is checked on every platform to tolerances that only a change
of the model, a penalty, a measure or a class rule can exceed. Everything
downstream of the optimiser is a pure function of the data and the parameters:
the continuum model, the line-complex model, the chi-square with its penalty
terms, the profile measures and the classification. Starting from the pinned
parameters, ``test_pin_evaluated_from_the_parameters`` recomputes them with
the package's own functions (no solver is run) and requires the pinned
chi-square to a relative 1e-9, every velocity and width of the summary row to
1e-3 km/s (the profile grid is 5 km/s), the fixed, tied and derived parameter
entries rebuilt from the free ones to a relative 1e-12, every other number to
a relative 1e-6, and equal classes, reasons, flags, component counts and
systemic sources (the systemic redshift, which derives from v_sys, to the same
1e-3 km/s). The preprocessing (bad pixels, de-reddening, error floor, rest
frame) is a copy of the first step of ``fit_spectrum`` (``rest_frame`` below);
the second part checks that the rest-frame arrays of the fresh fit equal it,
so a change of that step, its default error floor included, is caught. On the
reference stack the agreement is bit for bit; on numpy 2.5 / scipy 1.18 on
macOS the largest departure is 5e-12 km/s and chi-square shows none. A
departure here means that the model, a penalty, a measure or a class rule
changed, not the solver. The pinned BIC list of every complex holds the end
points of the optimiser for the one-, two- and three-component trials, not
pure functions of the pinned parameters: the default mode checks
``select_by_bic`` on the pinned list and the BIC of the selected model; the
BIC of the non-selected trial models is checked in strict mode only.

Two rules of the fit are checked for consistency with the pinned decisions,
because the first part takes those decisions as given: the host is applied
exactly when the eigenspectrum count is positive and the host fraction reaches
MIN_HOST_FRAC, and the Halpha narrow kinematics are passed to Hbeta exactly
when the Halpha narrow peak reaches NARROW_PRIOR_MIN_SNR.

Coverage limits of the four pins: all have E(B-V) = 0, so the extinction law
is not exercised here (``tests/test_extinction.py`` pins it); every hinge is
zero at the pinned parameters of all four pins (the width hinge of the far
broad components, the [O III] core-wing width hinge and amplitude ordering,
and the width hinge of the narrow-line-region wing), so the pins constrain
only the [S II] tie, the systemic prior and, for spec-1704, the velocity and
fraction priors of the narrow-line-region wing; the five penalty functions
(the far-broad width hinge, the [O III] ordering, the narrow-line-region wing
hinge and priors, the [S II] tie and the systemic prior) are pinned at active
parameter values by ``test_penalty_terms_pinned``; the host is applied for
one pin (spec-1704) and rejected below MIN_HOST_FRAC for the other three.

The second part is the optimiser's end point: a fresh fit of each spectrum
(``test_pin_reproduced``). The bounded trust-region solver ends where the
platform's floating-point details take it, and for the degenerate
decompositions of a broad profile into two or three Gaussians the end point
differs between platforms and between runs on the same platform (the BLAS
kernels differ with the CPU). Measured on the Linux runners of the test
workflow and on numpy 2.5 / scipy 1.18 on macOS against the reference stack:
the bisector velocities move by up to about 30 km/s, the peak of the
two-humped pin and one centroid by up to about 70 km/s, the widths by up to
about 200 km/s, the second-moment width by up to about 500 km/s, and
chi-square drops by up to about 20 per cent when another local minimum is
found; classes, flags, component counts and systemic sources were unchanged in
every run. The continuum fit is degenerate as well, between the host and the
power law: the host fraction of spec-1704 is 0.142 on the reference stack and
0.098 on a Linux runner, below the 0.10 threshold, so the host is rejected
there, with the same classes and offsets. The fresh fit is therefore held only
to what the science uses: equal classes, flags, component counts and systemic
sources, the primary offset c50_sys within 100 km/s (one third of the class
threshold), and a chi-square at most 10 per cent above the pin (lower is
allowed); its rest-frame arrays must equal those of the first part. Every
departure of a spectrum is reported in one message.

The third part is the pin of the corrected fitter itself,
``tests/data/pins_0.2.0.json``, written by ``tools/make_pins.py fit`` from this
tree on the same four spectra and on the DESI example coadd (at its redrock
redshift, with the E(B-V) of its fibermap), with the content of ``pins.json``
and a ``produced_by`` block (blrfit version, git description of the tree,
library versions, platform, date). ``test_current_pin_reproduced`` holds a
fresh fit to the tolerances of the second part (the same constants, the same
comparison) and, under BLRFIT_STRICT_PINS, bit for bit: the fitted parameters,
chi-square and the BIC list of every complex, the continuum parameters, the
host decision (eigenspectrum count, host fraction, reason string), the [O III]
pre-fit and every entry of the summary row, the solver bookkeeping columns
included. That is the release check of 0.2.0 on the reference stack (numpy
1.26.4, scipy 1.13.1, macOS arm64).

BLRFIT_STRICT_PINS=1 makes the first part bit-exact as well. The fresh fit of
the legacy pins is held to the loose tolerances in both modes: the 0.2.0 model
differs from 0.1.0 by design (the continuous Fe II operator; the ultraviolet
Fe II width fixed where its window is short), and what that changes on the
pinned spectra is recorded in ``docs/deltas_0.1.0_to_0.2.0.csv``.
"""
import json
import os

import numpy as np
import pytest

import blrfit
from blrfit.classify import classify
from blrfit.constants import (BROAD_WIDTH_SLOPE, C_KMS, COMPLEX_LINE, COMPLEX_WINDOW, DBIC,
                              ERR_FLOOR, HOST_CONTINUUM_ONLY, LAM, MIN_HOST_FRAC,
                              NARROW_PRIOR_MIN_SNR, O3_START_MIN_SNR, SYS_PRIOR_KMS,
                              SYS_PRIOR_MIN_SNR)
from blrfit.io import read_desi, read_sdss
from blrfit.io.desi import read_redrock, redrock_sibling
from blrfit.measure import measure_complex
from blrfit.model.broad import (broad_parameter_pairs, covered_velocity_bounds, select_by_bic,
                                width_offset_penalty)
from blrfit.model.continuum import conti_model, fe_templates, host_continuum_only, pca_templates
from blrfit.model.continuum import FeTemplate
from blrfit.constants import S2F
from scipy.ndimage import gaussian_filter1d
from blrfit.model.extinction import deredden
from blrfit.model.fit import PREFIX, _lumdist_cm
from blrfit.model.lines import build_complex, eval_components
from blrfit.model.narrow import (has_oiii_ordering, nlr_wing_penalty, oiii_order_penalty,
                                 sii_soft_tie_penalty, systemic_prior_penalty)
from conftest import DATA, EXAMPLES

STRICT = os.environ.get("BLRFIT_STRICT_PINS", "") not in ("", "0")

# the end point of a fresh fit (second part)
END_POINT_KMS = 100.0          # c50_sys: one third of the 300 km/s class threshold
CHI2_WORSE = 0.10              # chi-square may exceed the pin by this fraction; lower is allowed

# the evaluation at the pinned parameters (first part)
CHI2_REL = 1e-9
KMS_ABS = 1e-3                 # velocities and widths, km/s (the profile grid is 5 km/s)
Z_ABS = KMS_ABS / C_KMS        # z_sys: the same 1e-3 km/s
OTHER_REL = 1e-6               # fluxes, ratios, indices, signal-to-noise
KMS_STATS = frozenset(("v_sys", "sig_sys", "v_o3", "v_sii", "sig_sii", "v_peak", "centroid", "c25", "c50",
                       "c75", "c90", "fwhm", "W25", "W75", "sigma_line", "peak_sep", "v_peak_sys",
                       "centroid_sys", "c50_sys", "c25_sys", "c75_sys", "peak_top", "peak_top_sys",
                       "centroid25_sys", "centroid50_sys", "v_o3_peak", "v_o3_pre", "nw_v", "nw_sig",
                       "v_cover_lo", "v_cover_hi", "v_single_gauss", "data_v_peak", "data_c50",
                       "data_centroid_win",
                       "conti_feop_fwhm"))     # the Fe II broadening, a FWHM in km/s


def _pins():
    with open(os.path.join(DATA, "pins.json")) as fh:
        return json.load(fh)["pins"]


def _spectrum_path(fn):
    for d in (EXAMPLES, DATA):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def _nan(v):
    return np.nan if v is None else v


def _close(got, ref, rel, abs_):
    if STRICT:
        return got == ref
    return abs(got - ref) <= max(rel * abs(ref), abs_)


def _row_departures(row, ref, kms_abs=KMS_ABS, other_rel=OTHER_REL):
    """Every summary-row entry that differs from the pin: strings, booleans and
    integers exactly, velocities and widths to ``kms_abs`` km/s, other numbers
    to a relative ``other_rel``."""
    out = []
    assert set(row) == set(ref)
    for k, v in ref.items():
        got = row[k]
        stat = k.split("_", 1)[1] if k[:3] in ("HA_", "HB_", "MG_") else k
        if v is None:
            if not (isinstance(got, float) and np.isnan(got)):
                out.append(f"{k} {got!r} vs pinned NaN")
        elif isinstance(v, (bool, str, int)):
            if got != v:
                out.append(f"{k} {got!r} vs pinned {v!r}")
        elif stat in KMS_STATS:
            if not _close(got, v, 0.0, kms_abs):
                out.append(f"{k} {got:.6f} vs pinned {v:.6f} km/s")
        elif stat == "z_sys":
            if not _close(got, v, 0.0, Z_ABS):
                out.append(f"{k} {got!r} vs pinned {v!r}")
        elif not _close(got, v, other_rel, 1e-9):
            out.append(f"{k} {got!r} vs pinned {v!r}")
    return out


# ----------------------------------------------------------------------------
# first part: the model, chi-square, measures and classes at the pinned parameters
# ----------------------------------------------------------------------------
def rest_frame(sp, z, ebv, err_floor=ERR_FLOOR):
    """The first step of ``fit_spectrum`` replicated statement by statement (bad
    pixels zeroed, de-reddening, the error floor in quadrature, the shift to
    the rest frame). The second part checks that the ``wave_rest``,
    ``flux_rest`` and ``ivar_rest`` arrays of a fresh fit equal these, so a
    change of that step, its default error floor included, is caught."""
    wave_obs = np.asarray(sp["wave"], float); flux = np.asarray(sp["flux"], float)
    ivar = np.asarray(sp["ivar"], float)
    bad = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
    flux = np.where(bad, 0.0, flux); ivar = np.where(bad, 0.0, ivar)
    flux, ivar = deredden(wave_obs, flux, ivar, ebv)
    if err_floor and err_floor > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(ivar > 0, 1.0 / ivar, 0.0) + (err_floor * np.abs(flux)) ** 2
            ivar = np.where(ivar > 0, 1.0 / var, 0.0)
    return wave_obs / (1 + z), flux * (1 + z), ivar / (1 + z) ** 2


def pinned_continuum(pin, wr):
    """Power law + Fe II and the host model from the pinned continuum parameters,
    as ``fit_spectrum`` stores them."""
    cd = pin["conti"]; hi = pin["host_info"]
    cmodel = conti_model(wr, cd, *fe_templates())
    host = np.zeros_like(wr)
    if hi["applied"]:
        P = pca_templates()
        inhost = (wr > P["gw"].min() + 2) & (wr < P["gw"].max() - 2)
        for i in range(hi["n_gal"]):
            host += cd[f"gal{i}"] * np.where(inhost, np.interp(wr, P["gw"], P["gp"][i], left=0, right=0), 0.0)
        host = np.clip(host, 0, None)
        if HOST_CONTINUUM_ONLY:
            host = host_continuum_only(wr, host)
        cmodel = (cmodel + host) - host        # conti_model is stored as (total) - host
    return cmodel, host


def pinned_complex(pin, name, wr, fsub, ir):
    """The fit dictionary of one complex at the pinned parameters: the window,
    the components of ``build_complex`` and the chi-square of ``fit_complex``
    (weighted residuals followed by the penalty terms in their frozen order),
    evaluated at the pinned parameter dictionary."""
    pref = pin["params"][name]; p = PREFIX[name]
    n_broad = pin["summary"][f"{p}_n_broad"]
    lo, hi = COMPLEX_WINDOW[name]
    m = (wr > lo) & (wr < hi) & np.isfinite(fsub) & (ir > 0)
    x, y, w = wr[m], fsub[m], np.sqrt(ir[m])
    vcov = (x / LAM[COMPLEX_LINE[name]] - 1.0) * C_KMS
    v_cover = (float(vcov.min()), float(vcov.max()))
    kw = dict(v_bounds=covered_velocity_bounds(*v_cover))
    if pin["summary"][f"{p}_systemic_source"] == "Halpha prior":
        # the Halpha narrow kinematics are fixed entries of the pinned Hbeta dictionary
        kw.update(v_sys_prior=pref["n_v"], sig_sys_prior=pref["n_sig"])
        if "nw_f" in pref:
            kw["nw_prior"] = (pref["nw_f"], pref["nw_v"], pref["nw_sig"])
    ps, comps = build_complex(name, x, y, n_broad=n_broad, **kw)
    assert set(ps.names) | set(ps.derived) == set(pref), name
    d = ps.full(np.array([pref[k] for k in ps.free_names]))     # fixed, tied and derived entries rebuilt

    r = (y - eval_components(x, d, comps)) * w
    pairs = broad_parameter_pairs(comps)
    if BROAD_WIDTH_SLOPE > 0 and pairs:
        r = np.concatenate([r, width_offset_penalty(d, pairs)])
    pre = pin["o3_prefit"]
    if (name == "Halpha" and pre["v_o3"] is not None and pre["snr"] >= max(O3_START_MIN_SNR, SYS_PRIOR_MIN_SNR)
            and "n_v" in ps.names and "n_v" not in ps.fixed and "n_v" not in ps.tie):
        r = np.concatenate([r, systemic_prior_penalty(d, (pre["v_o3"], SYS_PRIOR_KMS))])
    if name == "Halpha" and "nw_sig" in ps.names and "nw_sig" not in ps.fixed:
        r = np.concatenate([r, nlr_wing_penalty(d)])
    if name == "Halpha" and "s2_sig" in ps.names:
        r = np.concatenate([r, sii_soft_tie_penalty(d)])
    if has_oiii_ordering(name, ps):
        r = np.concatenate([r, oiii_order_penalty(d)])
    chi2 = float(np.sum(r ** 2))
    n = int(m.sum()); k = len(ps.free_names)
    return dict(name=name, ps=ps, comps=comps, d=d, chi2=chi2, npix=n, nfree=k, bic=chi2 + k * np.log(n),
                n_broad=n_broad, window=(lo, hi), x=x, y=y, w=w, v_cover=v_cover,
                all_bic=list(pin["all_bic"][name]))


def pinned_result(pin, sp):
    """A result dictionary evaluated at the pinned parameters, with the measures
    and classes assembled as in ``fit_spectrum``."""
    wr, fr, ir = rest_frame(sp, pin["z"], pin["ebv"])
    cmodel, host = pinned_continuum(pin, wr)
    fsub = fr - host - cmodel
    dl = _lumdist_cm(pin["z"])
    hi = dict(pin["host_info"]); hi["host_frac_4200_5000"] = _nan(hi["host_frac_4200_5000"])
    # the host is applied exactly when eigenspectra were kept and the host fraction reaches MIN_HOST_FRAC
    frac = hi["host_frac_4200_5000"]
    assert hi["applied"] == (hi["n_gal"] > 0 and np.isfinite(frac) and frac >= MIN_HOST_FRAC), hi
    pre = pin["o3_prefit"]
    res = dict(z=pin["z"], wave_rest=wr, flux_rest=fr, ivar_rest=ir, host_model=host, host_info=hi,
               conti=dict(pin["conti"]), conti_model=cmodel, flux_sub=fsub,
               o3_prefit=dict(v_o3=_nan(pre["v_o3"]), snr=pre["snr"]), fits={}, meas={}, cls={}, err={})
    prior_snr = None
    for name in [c for c in ("Halpha", "Hbeta", "MgII") if c in pin["params"]]:
        r = pinned_complex(pin, name, wr, fsub, ir)
        m = measure_complex(r, cmodel + host, wr, pin["z"], dl_cm=dl, host_model=host)
        m["host_frac"] = float(hi["host_frac_4200_5000"]) if hi["applied"] else 0.0
        m["pl_alpha"] = float(pin["conti"]["pl_alpha"])
        src = pin["summary"][f"{PREFIX[name]}_systemic_source"]
        m["systemic_source"] = src
        if name == "Hbeta":
            # the Halpha narrow kinematics are passed on exactly when the Halpha narrow lines are detected
            assert (src == "Halpha prior") == (prior_snr is not None), (src, prior_snr)
            if prior_snr is not None:
                m["sys_snr"] = prior_snr
        if name == "Halpha":
            m["v_o3_pre"] = res["o3_prefit"]["v_o3"]; m["o3_pre_snr"] = res["o3_prefit"]["snr"]
        res["fits"][name] = r; res["meas"][name] = m
        if name == "Halpha" and m["narrow_peak_snr"] >= NARROW_PRIOR_MIN_SNR and np.isfinite(m["v_sys"]):
            prior_snr = m["narrow_peak_snr"]
    for name, m in res["meas"].items():
        res["cls"][name] = classify(m)
    return res


@pytest.mark.parametrize("pin", _pins(), ids=lambda p: p["file"])
def test_legacy_pin_evaluated_from_the_parameters(pin, monkeypatch):
    """Reconstruct frozen 0.1.0 values with its historical Fe operator."""
    def historical_broadening(self, width):
        f = int(round(max(width, self.intrinsic + 10.) / 50.)) * 50.
        sigma = np.sqrt(max(f**2 - self.intrinsic**2, 100.)) / S2F / self.pix_kms
        return gaussian_filter1d(self.flux, sigma, mode='nearest')

    monkeypatch.setattr(FeTemplate, 'broadened', historical_broadening)
    sp = read_sdss(_spectrum_path(pin["file"]))
    res = pinned_result(pin, sp)
    departures = []
    for name, pref in pin["params"].items():
        r = res["fits"][name]
        for k, v in pref.items():                     # the derived entries are rebuilt from the free ones
            if not _close(r["d"][k], v, 1e-12, 0.0):
                departures.append(f"{name} parameter {k} {r['d'][k]!r} vs pinned {v!r}")
        chi2_ref = pin["chi2"][name]
        if not _close(r["chi2"], chi2_ref, CHI2_REL, 0.0):
            departures.append(f"{name} chi2 {r['chi2']!r} vs pinned {chi2_ref!r}")
        bics = pin["all_bic"][name]
        if not _close(r["bic"], bics[r["n_broad"] - 1], CHI2_REL, 0.0):
            departures.append(f"{name} BIC {r['bic']!r} vs pinned {bics[r['n_broad'] - 1]!r}")
        if select_by_bic(bics, DBIC) != r["n_broad"] - 1:
            departures.append(f"{name} BIC selection {select_by_bic(bics, DBIC) + 1} vs pinned n_broad {r['n_broad']}")
    departures += _row_departures(blrfit.summary_row(res), pin["summary"])
    assert not departures, pin["file"] + ":\n  " + "\n  ".join(departures)


# ----------------------------------------------------------------------------
# second part: the end point of a fresh fit
# ----------------------------------------------------------------------------
def _same(a, b):
    """Equal, with NaN equal to NaN."""
    return a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))


def _end_point_departures(res, row, pin):
    """What the science would notice in a fresh fit against a pin: a class, a
    flag, a component count or a systemic source that differs, c50_sys further
    than END_POINT_KMS, a chi-square more than CHI2_WORSE above the pin. The
    host decision is not held here: it is platform dependent near the
    threshold (strict mode compares it)."""
    ref = pin["summary"]
    departures = []
    if set(res["fits"]) != set(pin["params"]):
        departures.append(f"complexes fitted {sorted(res['fits'])} vs pinned {sorted(pin['params'])}")
    for name in pin["params"]:
        if name not in res["fits"]:
            continue
        p = PREFIX[name]
        for k in ("class", "flags", "n_broad", "systemic_source"):
            if row[f"{p}_{k}"] != ref[f"{p}_{k}"]:
                departures.append(f"{name} {k} {row[f'{p}_{k}']!r} vs pinned {ref[f'{p}_{k}']!r}")
        got, r = row[f"{p}_c50_sys"], _nan(ref[f"{p}_c50_sys"])
        if np.isfinite(r):
            if not abs(got - r) < END_POINT_KMS:
                departures.append(f"{name} c50_sys {got:.1f} vs pinned {r:.1f} km/s (tolerance {END_POINT_KMS:.0f})")
        elif np.isfinite(got):
            departures.append(f"{name} c50_sys {got:.1f} vs pinned NaN")
        chi2, chi2_ref = res["fits"][name]["chi2"], pin["chi2"][name]
        if not chi2 <= (1 + CHI2_WORSE) * chi2_ref:
            departures.append(f"{name} chi2 {chi2:.2f} vs pinned {chi2_ref:.2f} (more than {100 * CHI2_WORSE:.0f} per cent worse)")
    return departures


def _rest_frame_departures(res, sp, pin):
    """The fresh fit started from the rest-frame arrays of the first part."""
    wr, fr, ir = rest_frame(sp, pin["z"], pin["ebv"])
    return [f"{k} of the fresh fit differs from the preprocessing of the first part"
            for k, a in (("wave_rest", wr), ("flux_rest", fr), ("ivar_rest", ir)) if not np.array_equal(res[k], a)]


@pytest.mark.parametrize("pin", _pins(), ids=lambda p: p["file"])
def test_pin_reproduced(pin):
    """A fresh fit reaches the pinned classes, flags, component counts and
    systemic sources, the pinned c50_sys within END_POINT_KMS and a chi-square
    not more than CHI2_WORSE above the pin, from the rest-frame arrays of the
    first part. Not bit for bit in strict mode: the model changed since 0.1.0
    (the third part holds the corrected fitter to its own pins)."""
    sp = read_sdss(_spectrum_path(pin["file"]))
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], pin["z"], ebv=pin["ebv"],
                              complexes=tuple(pin["complexes"]))
    departures = _end_point_departures(res, blrfit.summary_row(res), pin) + _rest_frame_departures(res, sp, pin)
    assert not departures, pin["file"] + ":\n  " + "\n  ".join(departures)


# ----------------------------------------------------------------------------
# third part: the pins of the corrected fitter (tests/data/pins_0.2.0.json)
# ----------------------------------------------------------------------------
CURRENT_PINS = "pins_0.2.0.json"
DESI_TARGETID = 39627574082538900


def _current_pins():
    with open(os.path.join(DATA, CURRENT_PINS)) as fh:
        return json.load(fh)


def _read_pinned_spectrum(pin):
    """The spectrum of a pin of either survey. A DESI pin is read at its target
    identifier; its pinned redshift and E(B-V) must be those of the redrock
    file next to the coadd and of the fibermap."""
    path = _spectrum_path(pin["file"])
    if pin.get("kind") == "desi":
        sp = read_desi(path, pin["targetid"], use_desispec=False)
        assert read_redrock(redrock_sibling(path), pin["targetid"])["z"] == pin["z"]
        assert sp["ebv"] == pin["ebv"]
        return sp
    return read_sdss(path)


def _bit_exact_departures(res, row, pin):
    """Every pinned number a fresh fit does not reproduce exactly: the
    parameters, chi-square and BIC list of every complex, the continuum
    parameters, the host decision, the [O III] pre-fit and the summary row."""
    departures = []
    for name, pref in pin["params"].items():
        d = res["fits"][name]["d"]
        if set(d) != set(pref):
            departures.append(f"{name} parameter names differ from the pin")
        departures += [f"{name} parameter {k} {d[k]!r} vs pinned {v!r}" for k, v in pref.items()
                       if k in d and not _same(float(d[k]), v)]
        if res["fits"][name]["chi2"] != pin["chi2"][name]:
            departures.append(f"{name} chi2 {res['fits'][name]['chi2']!r} vs pinned {pin['chi2'][name]!r}")
        if [float(b) for b in res["fits"][name]["all_bic"]] != pin["all_bic"][name]:
            departures.append(f"{name} BIC list {[float(b) for b in res['fits'][name]['all_bic']]!r} vs pinned {pin['all_bic'][name]!r}")
    conti = {k: float(v) for k, v in res["conti"].items()}
    if set(conti) != set(pin["conti"]) or any(not _same(conti[k], v) for k, v in pin["conti"].items() if k in conti):
        departures.append(f"continuum {conti!r} vs pinned {pin['conti']!r}")
    hi, href = res["host_info"], pin["host_info"]
    got = (bool(hi["applied"]), int(hi["n_gal"]), float(hi["host_frac_4200_5000"]), hi["reason"])
    ref = (href["applied"], href["n_gal"], _nan(href["host_frac_4200_5000"]), href["reason"])
    if not all(_same(a, b) for a, b in zip(got, ref)):
        departures.append(f"host decision {got!r} vs pinned {ref!r}")
    got = (float(res["o3_prefit"]["v_o3"]), float(res["o3_prefit"]["snr"]))
    ref = (_nan(pin["o3_prefit"]["v_o3"]), _nan(pin["o3_prefit"]["snr"]))
    if not all(_same(a, b) for a, b in zip(got, ref)):
        departures.append(f"[O III] pre-fit {got!r} vs pinned {ref!r}")
    departures += _row_departures(row, pin["summary"])
    return departures


@pytest.mark.parametrize("pin", _current_pins()["pins"], ids=lambda p: p["file"])
def test_current_pin_reproduced(pin):
    """A fresh fit of the corrected fitter reproduces its own pin: the classes,
    flags, component counts and systemic sources, c50_sys within END_POINT_KMS
    and chi-square within CHI2_WORSE (the tolerances of the second part), from
    the rest-frame arrays of the first part; bit for bit under
    BLRFIT_STRICT_PINS, the summary row included."""
    sp = _read_pinned_spectrum(pin)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], pin["z"], ebv=pin["ebv"],
                              complexes=tuple(pin["complexes"]))
    row = blrfit.summary_row(res)
    departures = _end_point_departures(res, row, pin) + _rest_frame_departures(res, sp, pin)
    if STRICT:
        departures += _bit_exact_departures(res, row, pin)
    assert not departures, pin["file"] + ":\n  " + "\n  ".join(departures)


def test_current_pins_provenance():
    """The 0.2.0 pin file names its producer, holds the four SDSS spectra of
    the legacy pins at their redshifts, E(B-V) and complexes with the legacy
    classes and component counts (no class changed between the versions on
    these spectra), and the DESI example (F for Halpha, C for Hbeta); every
    pinned line was fitted from a converged continuum by a converged attempt,
    so the flag-and-keep rule of the solver is not exercised by the pins."""
    cur = _current_pins()
    made = cur["produced_by"]
    for k in ("blrfit", "git", "numpy", "scipy", "astropy", "python", "platform", "date"):
        assert made[k], k
    assert made["blrfit"].startswith("0.2.0")
    legacy = {p["file"]: p for p in _pins()}
    pins = {p["file"]: p for p in cur["pins"]}
    assert set(legacy) < set(pins)
    for fn, old in legacy.items():
        new = pins[fn]
        assert (new["z"], new["ebv"], new["complexes"]) == (old["z"], old["ebv"], old["complexes"])
        assert set(new["params"]) == set(old["params"])
        for name in old["params"]:
            p = PREFIX[name]
            assert new["summary"][f"{p}_class"] == old["summary"][f"{p}_class"], (fn, name)
            assert new["summary"][f"{p}_n_broad"] == old["summary"][f"{p}_n_broad"], (fn, name)
    desi = [p for p in cur["pins"] if p.get("kind") == "desi"]
    assert len(desi) == 1 and desi[0]["targetid"] == DESI_TARGETID
    assert desi[0]["summary"]["HA_class"] == "F" and desi[0]["summary"]["HB_class"] == "C"
    for pin in cur["pins"]:
        assert pin["summary"]["continuum_status"] == "success", pin["file"]
        assert isinstance(pin["summary"]["conti_at_bound"], str) and isinstance(pin["summary"]["conti_feuv_fwhm_fixed"], bool)
        for name in pin["params"]:
            p = PREFIX[name]
            assert pin["summary"][f"{p}_fit_status"] == "success" and pin["summary"][f"{p}_converged"] is True, (pin["file"], name)
            assert pin["summary"][f"{p}_bic_margin"] is not None, (pin["file"], name)

def test_penalty_terms_pinned():
    """The five penalty terms of the chi-square at hand-chosen parameter values
    where each hinge or prior is active, and at values where it is inactive
    (zero). Every hinge (the far-broad width hinge, the [O III] width hinge and
    amplitude ordering, the narrow-line-region wing width hinge) is zero at the
    pinned parameters of all four pins, so the pins do not constrain them.
    The literal values were computed on the reference stack (numpy 1.26.4,
    scipy 1.13.1) from the frozen constants: BROAD_WIDTH_SLOPE / S2F = 0.4 /
    2.3548 of |v| as the minimum sigma at BROAD_WIDTH_PENALTY_SCALE = 10 km/s
    per unit residual; the [O III] width hinge at 10 km/s per unit and the
    amplitude ordering at O3_AMP_ORDER_SCALE = 0.05 of core + wing; the
    narrow-line-region wing hinge at 3 km/s per unit with NW_V_PRIOR_KMS = 150
    and NW_F_PRIOR = 0.25; the [S II] tie at SII_PRIOR_SIG_FRAC = 0.2 of the
    narrow sigma and SII_PRIOR_V_KMS = 60; the systemic prior at
    SYS_PRIOR_KMS = 150."""
    rel = 1e-9
    pairs = [("b1_v", "b1_sig"), ("b2_v", "b2_sig")]
    # far-broad width hinge: sigma >= 0.16987 |v|; the first component is 209.6 km/s too narrow
    assert width_offset_penalty(dict(b1_v=-3000.0, b1_sig=300.0, b2_v=1000.0, b2_sig=1000.0), pairs) \
        == pytest.approx([20.95930801728115, 0.0], rel=rel)
    assert width_offset_penalty(dict(b1_v=-3000.0, b1_sig=600.0, b2_v=1000.0, b2_sig=1000.0), pairs) == [0.0, 0.0]
    # [O III] ordering: the core 100 km/s wider than the wing, the wing twice as tall as the core
    assert oiii_order_penalty(dict(o3_sig=400.0, w_sig=300.0, OIII5007c_A=1.0, OIII5007w_A=2.0)) \
        == pytest.approx([10.0, 6.6666666222222215], rel=rel)
    assert oiii_order_penalty(dict(o3_sig=200.0, w_sig=600.0, OIII5007c_A=3.0, OIII5007w_A=1.0)) == [0.0, 0.0]
    # Halpha wing: 50 km/s narrower than the core, 150 km/s bluewards of it, fraction 0.1
    assert nlr_wing_penalty(dict(n_sig=200.0, nw_sig=150.0, nw_v=-100.0, n_v=50.0, nw_f=0.1)) \
        == pytest.approx([16.666666666666668, -1.0, 0.4], rel=rel)
    assert nlr_wing_penalty(dict(n_sig=150.0, nw_sig=400.0, nw_v=50.0, n_v=50.0, nw_f=0.0)) == [0.0, 0.0, 0.0]
    # [S II] tie: 50 km/s wider and 60 km/s redward of narrow Halpha + [N II]
    assert sii_soft_tie_penalty(dict(s2_sig=250.0, n_sig=200.0, s2_v=80.0, n_v=20.0)) \
        == pytest.approx([1.25, 1.0], rel=rel)
    assert sii_soft_tie_penalty(dict(s2_sig=200.0, n_sig=200.0, s2_v=20.0, n_v=20.0)) == [0.0, 0.0]
    # systemic prior: the narrow group 150 km/s from the [O III] pre-fit velocity
    assert systemic_prior_penalty(dict(n_v=100.0), (-50.0, SYS_PRIOR_KMS)) == pytest.approx([1.0], rel=rel)
    assert systemic_prior_penalty(dict(n_v=-50.0), (-50.0, SYS_PRIOR_KMS)) == [0.0]


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
