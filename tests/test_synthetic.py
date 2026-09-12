"""
Single-epoch validation on synthetic DESI-like spectra with known truth.

Every spectrum is built by ``tests/synth.make_spectrum`` (power law, optional
host, narrow lines, broad Gaussians, Gaussian noise on the DESI grid) and fitted
end to end with ``blrfit.fit_spectrum`` at the redshift at which the narrow
lines are sought. The configurations are those of the regression suite
described with the fitter:

* bulk shifts of +/-1200 km/s at continuum S/N 6-25, recovered to better than
  60 km/s in Halpha and 80 km/s in Hbeta;
* double-peaked, asymmetric and symmetric profiles, classes B, C and F;
* pure narrow-line galaxies (Gaussian and pedestal narrow lines), class E;
* host-dominated systems (85 per cent host light) with broad Halpha at +800;
* [S II] kinematics discrepant from the Balmer group;
* non-Gaussian narrow lines with 25-40 per cent of their flux in a pedestal
  three times wider than the core, with a weak broad Halpha at +800;
* a narrow-line system displaced by 880 km/s from the input redshift;
* [O III] displaced from the low-ionisation lines;
* Halpha at the red edge of the spectrum;
* the structure of the result and the frozen settings.

Continuum S/N is per pixel at the power-law level near Halpha. Broad-line
equivalent widths are against the AGN continuum at the line. The broad-line
peak S/N quoted in the comments is the peak of the injected broad model above
everything else divided by the noise per pixel (``broad_peak_snr`` below);
the fitter's own ``broad_peak_snr`` is lower because it adds a 2 per cent
flux-calibration floor to the errors, which dominates in host-dominated spectra.

Each configuration is fitted once per session (the fits are cached by
configuration) so that several tests can inspect the same result. A fit of
both complexes takes 3-8 s on a laptop.
"""
import copy
import json

import numpy as np
import pytest

import blrfit
from blrfit.classify import is_measurable
from blrfit.constants import S2F, V_BROAD_MAX, V_NARROW_MAX, DBIC, NLR_WING, MAX_BROAD, LAM, C_KMS
from synth import make_spectrum, DESI_WAVE

_FITS = {}


def _key(kw):
    return json.dumps(kw, sort_keys=True)


def fit(**kw):
    """The synthetic spectrum ``make_spectrum(**kw)`` and its fit of the Halpha
    and Hbeta complexes at the spectrum's own redshift, cached by configuration."""
    key = _key(kw)
    if key not in _FITS:
        sp = make_spectrum(**kw)
        res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], kw.get("z", 0.25),
                                  complexes=("Halpha", "Hbeta"))
        _FITS[key] = (sp, res)
    return _FITS[key]


def broad_peak_snr(kw, line="Halpha"):
    """Peak of the injected broad model of ``line`` above the rest of the model,
    divided by the noise per pixel: the difference between the noiseless models
    with and without the broad components."""
    sp = make_spectrum(**kw)
    kw0 = dict(kw); kw0["broad"] = None
    sp0 = make_spectrum(**kw0)
    diff = sp["truth"]["model"] - sp0["truth"]["model"]
    wr = sp["wave"] / (1.0 + kw.get("z", 0.25))
    near = np.abs(wr - LAM[line]) < 250.0
    return float(diff[near].max() / sp["truth"]["noise"])


def config(z=0.25, snr=15.0, seed=0, broad=None, narrow=None, **more):
    """Keyword dictionary for ``make_spectrum`` with the defaults of this suite."""
    kw = dict(z=z, snr=float(snr), seed=seed, broad=broad, narrow=dict(ew_ha=40.0))
    if narrow:
        kw["narrow"].update(narrow)
    kw.update(more)
    return kw


def measures(res, name):
    return res["meas"][name], res["cls"][name]


# ---------------------------------------------------------------------------
# (1) bulk shifts
# ---------------------------------------------------------------------------
def shift_config(v, snr):
    """Broad Halpha (EW 300 A, FWHM 3800) and Hbeta (EW 80 A) at velocity v with
    the default narrow lines (narrow Halpha EW 40 A)."""
    return config(z=0.25, snr=snr, seed=11,
                  broad=[dict(line="Halpha", v=v, fwhm=3800.0, ew=300.0),
                         dict(line="Hbeta", v=v, fwhm=3800.0, ew=80.0)])


SHIFT_CASES = [pytest.param(v, snr, id=f"v{v:+.0f}_snr{snr}",
                            marks=([] if snr in (6, 25) else [pytest.mark.slow]))
               for v in (1200.0, -1200.0) for snr in (6, 10, 15, 25)]


@pytest.mark.parametrize("v, snr", SHIFT_CASES)
def test_bulk_shift_recovered_and_classified_a(v, snr):
    """Bulk shifts of +/-1200 km/s at continuum S/N 6-25 come back within 60 km/s
    (Halpha) and 80 km/s (Hbeta), class A, with no quality flag.

    Broad-line peak S/N over the eight cases: Halpha 20-85, Hbeta 11-48.
    Measured |recovered - true| over the eight cases: Halpha 1-29 km/s, Hbeta
    2-17 km/s; the largest errors are at S/N 6.
    """
    kw = shift_config(v, snr)
    sp, res = fit(**kw)
    truth = sp["truth"]
    for name, tol in (("Halpha", 60.0), ("Hbeta", 80.0)):
        m, c = measures(res, name)
        tb = truth["broad"][name]
        assert c["label"] == "A", (name, c)
        assert abs(m["c50_sys"] - tb["c50_sys"]) < tol, (name, m["c50_sys"], tb["c50_sys"])
        assert abs(m["v_peak_sys"] - tb["v_peak_sys"]) < tol, (name, m["v_peak_sys"])
        # the narrow-line systemic is at the input redshift: measured |v_sys| <= 9 km/s
        assert abs(m["v_sys"] - truth["v_sys"]) < 60.0
        # width within 10 per cent: measured Halpha within 1 per cent, Hbeta within 5 per cent (S/N 6)
        assert abs(m["fwhm"] - tb["fwhm"]) < 0.10 * tb["fwhm"], (name, m["fwhm"], tb["fwhm"])
        # a single symmetric Gaussian: no shape flag, no bound, no edge, good fit
        assert c["flags"] == [], (name, c["flags"])
        assert m["n_peaks"] == 1
        assert is_measurable(c["label"], c["flags"], m["broad_flux_snr"], m["fwhm"])


# ---------------------------------------------------------------------------
# (2) profile shapes
# ---------------------------------------------------------------------------
def test_double_peaked_profile_classified_b():
    """Two broad Gaussians at -2500 and +2500 km/s (FWHM 3000 each, equal flux)
    in both lines are class B with two resolved peaks about 5000 km/s apart.

    Measured: Halpha peak separation 5010 km/s, dip 0.71; Hbeta 4990 km/s, dip
    0.70; c(1/2) - v_sys = -4 and -11 km/s for a true value of +3.
    """
    kw = config(z=0.25, snr=15, seed=3,
                broad=[dict(line="Halpha", v=-2500.0, fwhm=3000.0, ew=150.0),
                       dict(line="Halpha", v=2500.0, fwhm=3000.0, ew=150.0),
                       dict(line="Hbeta", v=-2500.0, fwhm=3000.0, ew=40.0),
                       dict(line="Hbeta", v=2500.0, fwhm=3000.0, ew=40.0)])
    sp, res = fit(**kw)
    for name in ("Halpha", "Hbeta"):
        m, c = measures(res, name)
        tb = sp["truth"]["broad"][name]
        assert tb["n_peaks"] == 2
        assert c["label"] == "B", (name, c)
        assert m["n_peaks"] == 2
        # the two maxima of the summed profile are the component centres to
        # within the overlap: 5 per cent of the separation (measured 10 km/s)
        assert abs(m["peak_sep"] - 5000.0) < 250.0, (name, m["peak_sep"])
        assert m["dip_frac"] > 0.08
        # the symmetric pair has no net displacement: measured within 15 km/s
        assert abs(m["c50_sys"] - tb["c50_sys"]) < 100.0
        assert m["n_broad"] >= 2


def test_asymmetric_profile_classified_c():
    """A broad Gaussian at the systemic velocity plus a broader component
    (FWHM 6000) displaced by +2000 km/s is class C in both lines.

    Truth of the summed profile: c(1/2) = +312, A.I. = 0.15, tilt 0.10 FWHM.
    Measured: Halpha c(1/2) - v_sys = +322, A.I. 0.146; Hbeta +292, A.I. 0.130.
    """
    kw = config(z=0.25, snr=15, seed=4,
                broad=[dict(line="Halpha", v=0.0, fwhm=3500.0, ew=200.0),
                       dict(line="Halpha", v=2000.0, fwhm=6000.0, ew=120.0),
                       dict(line="Hbeta", v=0.0, fwhm=3500.0, ew=50.0),
                       dict(line="Hbeta", v=2000.0, fwhm=6000.0, ew=30.0)])
    sp, res = fit(**kw)
    for name in ("Halpha", "Hbeta"):
        m, c = measures(res, name)
        tb = sp["truth"]["broad"][name]
        assert c["label"] == "C", (name, c)
        tilt = abs(m["c25"] - m["c75"]) / m["fwhm"]
        assert abs(m["AI"]) >= 0.12 or tilt >= 0.10 or abs(m["v_peak"] - m["centroid"]) / m["fwhm"] >= 0.20
        assert np.sign(m["AI"]) == np.sign(tb["AI"]) and tb["AI"] > 0      # the red shoulder
        # the offset of the summed profile is recovered to within 100 km/s (measured 10 and 20)
        assert abs(m["c50_sys"] - tb["c50_sys"]) < 100.0, (name, m["c50_sys"], tb["c50_sys"])
        assert m["n_peaks"] == 1


def test_symmetric_unshifted_profile_classified_f():
    """One broad Gaussian at the systemic velocity in both lines is class F with
    |c(1/2) - v_sys| < 100 km/s (measured -5 and -6 km/s) and no flag."""
    kw = config(z=0.25, snr=15, seed=5,
                broad=[dict(line="Halpha", v=0.0, fwhm=3500.0, ew=300.0),
                       dict(line="Hbeta", v=0.0, fwhm=3500.0, ew=80.0)])
    sp, res = fit(**kw)
    for name in ("Halpha", "Hbeta"):
        m, c = measures(res, name)
        assert c["label"] == "F", (name, c)
        assert abs(m["c50_sys"]) < 100.0, (name, m["c50_sys"])
        assert abs(m["v_peak_sys"]) < 100.0
        assert c["flags"] == []
        assert abs(m["fwhm"] - 3500.0) < 350.0        # measured 3516 in both lines
        assert abs(m["AI"]) < 0.12 and abs(m["KI"] - 0.456) < 0.05


# ---------------------------------------------------------------------------
# (3) pure narrow-line galaxies
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ped_frac, snr", [(0.0, 25), (0.3, 10), (0.3, 25)],
                         ids=["gaussian_snr25", "pedestal30_snr10", "pedestal30_snr25"])
def test_pure_narrow_line_galaxy_has_no_broad_line(ped_frac, snr):
    """Strong narrow lines (narrow Halpha EW 150 A) and no broad component,
    with Gaussian narrow lines or with 30 per cent of the flux in a pedestal
    three times wider than the core: class E in both lines (the best "broad"
    component sits at the minimum width with integrated S/N below 8).

    Measured: class E in every case; the best "broad" component sits at the
    1200 km/s minimum width with integrated S/N 0.8-6.3 and peak S/N 0.1-1.2.
    """
    kw = config(z=0.1, snr=snr, seed=7, broad=None,
                narrow=dict(ew_ha=150.0, ped_frac=ped_frac, ped_width=3.0))
    sp, res = fit(**kw)
    assert sp["truth"]["broad"] == {}
    for name in ("Halpha", "Hbeta"):
        m, c = measures(res, name)
        assert c["label"] == "E", (name, c)
        assert not is_measurable(c["label"], c["flags"], m["broad_flux_snr"], m["fwhm"])
        # the narrow lines themselves are detected at high S/N (measured >= 124)
        assert m["narrow_peak_snr"] > 50.0
    # the systemic velocity is that of the narrow lines (measured |v_sys| < 0.3 km/s)
    assert abs(res["meas"]["Halpha"]["v_sys"]) < 60.0


# ---------------------------------------------------------------------------
# (4) host-dominated systems
# ---------------------------------------------------------------------------
def host_config(ew, snr, seed=21):
    """85 per cent host light at 4700 A, broad Halpha at +800 km/s (FWHM 3800)
    with the given equivalent width against the AGN continuum, no broad Hbeta."""
    return config(z=0.1, snr=snr, seed=seed, host_frac=0.85,
                  broad=[dict(line="Halpha", v=800.0, fwhm=3800.0, ew=float(ew))])


@pytest.mark.parametrize("ew, snr", [(60, 8), (60, 15), (120, 8), (120, 15), (200, 8), (200, 15)],
                         ids=lambda x: str(x))
def test_host_dominated_broad_halpha_offset_recovered(ew, snr):
    """A broad Halpha at +800 km/s under 85 per cent host light is recovered
    within 120 km/s, class A, with the host_dominated flag set, over broad
    equivalent widths of 60-200 A and continuum S/N 8-15.

    Broad-line peak S/N (injected peak over the noise): 5.4, 10.1, 10.8, 20.3,
    18.0, 33.8; the fitter's own peak S/N, with its 2 per cent calibration
    floor on a spectrum 6.7 times brighter than the AGN continuum, is 2.8-10.
    Measured c(1/2) - v_sys: +888, +871, +840, +832, +822, +817 km/s. The
    same grid on two further noise seeds gives +736 to +833 km/s; the sign of
    the bias depends on the noise realisation and its size on the line
    strength. No broad Hbeta is injected and none is found (class E).
    """
    kw = host_config(ew, snr)
    sp, res = fit(**kw)
    m, c = measures(res, "Halpha")
    assert sp["truth"]["host_frac"] == 0.85
    assert c["label"] == "A", c
    # measured +817 to +888 km/s (see the docstring); 120 leaves 30 km/s for the noise stream
    assert abs(m["c50_sys"] - 800.0) < 120.0, m["c50_sys"]
    assert abs(m["v_peak_sys"] - 800.0) < 120.0, m["v_peak_sys"]
    assert "host_dominated" in c["flags"], c["flags"]
    # the host fraction of the 4200-5000 A light is recovered: measured 0.83-0.84 for 0.85 at 4700 A
    assert 0.80 <= m["host_frac"] <= 0.90, m["host_frac"]
    assert res["host_info"]["applied"]
    # the narrow lines still define the systemic (measured v_sys +1 to +4 km/s)
    assert abs(m["v_sys"]) < 60.0
    assert abs(m["fwhm"] - 3800.0) < 380.0          # measured within 60 km/s
    mb, cb = measures(res, "Hbeta")
    assert cb["label"] in ("E", "W"), cb


@pytest.mark.parametrize("snr", [8, 15])
def test_host_dominated_weak_line_is_rejected_or_flagged(snr):
    """A broad Halpha of only 30 A against the AGN continuum under 85 per cent
    host light (injected peak S/N 2.7 and 5.1; fitter peak S/N 1.4 and 1.5 with
    the calibration floor) is not a trustworthy measurement: it is either class
    E or carries the low_peak_snr flag. Measured: class E at S/N 8; class A
    with low_peak_snr at S/N 15, where c(1/2) - v_sys = +951 km/s.
    """
    kw = host_config(30, snr)
    sp, res = fit(**kw)
    m, c = measures(res, "Halpha")
    assert c["label"] == "E" or "low_peak_snr" in c["flags"], c
    if c["label"] != "E":
        assert "host_dominated" in c["flags"]
        assert m["broad_peak_snr"] < 5.0


# ---------------------------------------------------------------------------
# (5) [S II] kinematics
# ---------------------------------------------------------------------------
def sii_config(sii_dv):
    """Strong [S II] ([S II] 6716 equal to narrow Halpha, narrow Halpha EW 80 A)
    displaced by sii_dv from the Balmer group; broad lines at the systemic."""
    return config(z=0.1, snr=15, seed=31,
                  broad=[dict(line="Halpha", v=0.0, fwhm=3500.0, ew=150.0),
                         dict(line="Hbeta", v=0.0, fwhm=3500.0, ew=40.0)],
                  narrow=dict(ew_ha=80.0, sii=1.0, sii_dv=sii_dv))


@pytest.mark.parametrize("sii_dv", [pytest.param(0.0, id="dv0"),
                                    pytest.param(200.0, id="dv200", marks=pytest.mark.slow),
                                    pytest.param(250.0, id="dv250"),
                                    pytest.param(300.0, id="dv300", marks=pytest.mark.slow)])
def test_sii_kinematics_discrepant_from_the_balmer_group(sii_dv):
    """With [S II] displaced by 200-300 km/s the systemic velocity stays with
    narrow Halpha + [N II] (within 60 km/s of the truth; measured within 2
    km/s), [S II] is measured at its own velocity (measured within 1 km/s) and
    the sii_disagree flag is raised; with no displacement the flag is absent.
    """
    kw = sii_config(sii_dv)
    sp, res = fit(**kw)
    m, c = measures(res, "Halpha")
    assert abs(m["v_sys"] - sp["truth"]["v_sys"]) < 60.0, m["v_sys"]
    assert abs(m["v_sii"] - (sp["truth"]["v_sys"] + sii_dv)) < 60.0, m["v_sii"]
    assert m["flux_SII6716"] > 0
    if sii_dv > 150.0:
        assert "sii_disagree" in c["flags"], c["flags"]
    else:
        assert "sii_disagree" not in c["flags"], c["flags"]
    # the broad line at the systemic is unaffected: class F, measured |c50_sys| <= 8 km/s
    assert c["label"] == "F", c
    assert abs(m["c50_sys"]) < 100.0
    mb, cb = measures(res, "Hbeta")
    assert cb["label"] == "F" and "sii_disagree" not in cb["flags"]


# ---------------------------------------------------------------------------
# (6) non-Gaussian narrow lines with a weak broad Halpha
# ---------------------------------------------------------------------------
def pedestal_config(ped_frac):
    """Narrow lines (narrow Halpha EW 80 A) with ``ped_frac`` of their flux in a
    pedestal three times wider than the core, and a weak broad Halpha of 40 A at
    +800 km/s (FWHM 3500): injected peak S/N 7.3."""
    return config(z=0.1, snr=15, seed=41,
                  broad=[dict(line="Halpha", v=800.0, fwhm=3500.0, ew=40.0)],
                  narrow=dict(ew_ha=80.0, ped_frac=ped_frac, ped_width=3.0))


def test_pedestal_narrow_lines_do_not_bias_a_weak_broad_halpha():
    """With 25 and 40 per cent of the narrow flux in a pedestal three times
    wider than the core, a weak broad Halpha at +800 km/s (peak S/N 7) is
    recovered within 100 km/s, and the pedestal changes the recovered velocity
    by less than 100 km/s relative to the same spectrum with Gaussian narrow
    lines; the narrow-line-region wing of the model absorbs the pedestal.

    Measured c(1/2) - v_sys: +776 (no pedestal), +734 (25 per cent), +736 (40
    per cent), i.e. a pedestal-induced shift of -42 km/s, against the -421
    km/s (+379 recovered) of a model without the wing. Wing fraction and width
    recovered: 0.11 / 432 km/s for a true amplitude ratio 0.111 and sigma 450;
    0.23 / 439 km/s for 0.222 and 450. Over three noise seeds and narrow EW
    40-80 A the pedestal cases span +724 to +777 km/s.
    """
    kw0 = pedestal_config(0.0)
    assert 5.0 <= broad_peak_snr(kw0) <= 10.0            # the weak-line regime: measured 7.3
    sp0, res0 = fit(**kw0)
    m0, c0 = measures(res0, "Halpha")
    assert c0["label"] == "A"
    assert abs(m0["c50_sys"] - 800.0) < 100.0, m0["c50_sys"]
    # Gaussian narrow lines: the wing takes little (measured fraction 0.06)
    assert m0["nw_f"] <= 0.15
    for ped_frac in (0.25, 0.40):
        kw = pedestal_config(ped_frac)
        sp, res = fit(**kw)
        m, c = measures(res, "Halpha")
        assert c["label"] == "A", (ped_frac, c)
        assert abs(m["c50_sys"] - 800.0) < 100.0, (ped_frac, m["c50_sys"])
        assert abs(m["v_peak_sys"] - 800.0) < 100.0, (ped_frac, m["v_peak_sys"])
        assert abs(m["c50_sys"] - m0["c50_sys"]) < 100.0, (ped_frac, m["c50_sys"], m0["c50_sys"])
        # the wing recovers the pedestal: amplitude ratio ped / (1 - ped) / width ratio,
        # width 3 x 150 km/s; tolerances 0.05 and 100 km/s (measured within 0.01 and 20 km/s)
        amp_ratio = ped_frac / (1.0 - ped_frac) / 3.0
        assert abs(m["nw_f"] - amp_ratio) < 0.05, (ped_frac, m["nw_f"], amp_ratio)
        assert abs(m["nw_sig"] - 3.0 * 150.0) < 100.0, (ped_frac, m["nw_sig"])
        assert abs(m["v_sys"]) < 60.0
        assert m["fwhm"] > 2000.0 and m["n_broad"] == 1


# ---------------------------------------------------------------------------
# (7) narrow system displaced from the input redshift
# ---------------------------------------------------------------------------
def test_narrow_system_displaced_880_kms_from_the_input_redshift():
    """The whole narrow-line system sits at +880 km/s from the input redshift
    and the broad lines at +800 km/s relative to it (+1680 in the input frame).
    The systemic is found at 880 km/s (within 50; measured 880.8) without
    reaching the +/-1500 km/s bound, and the offset relative to it is +800
    within 80 km/s (measured Halpha +787, Hbeta +775).
    """
    kw = config(z=0.25, snr=10, seed=51, v_sys=880.0,
                broad=[dict(line="Halpha", v=1680.0, fwhm=3800.0, ew=300.0),
                       dict(line="Hbeta", v=1680.0, fwhm=3800.0, ew=80.0)])
    sp, res = fit(**kw)
    assert sp["truth"]["broad"]["Halpha"]["c50_sys"] == pytest.approx(800.0, abs=1.0)
    for name in ("Halpha", "Hbeta"):
        m, c = measures(res, name)
        assert abs(m["v_sys"] - 880.0) < 50.0, (name, m["v_sys"])
        assert abs(m["c50_sys"] - 800.0) < 80.0, (name, m["c50_sys"])
        assert c["label"] == "A", (name, c)
        assert "narrow_at_bound" not in c["flags"]
        assert c["flags"] == []
    m = res["meas"]["Halpha"]
    assert abs(m["v_sii"] - 880.0) < 60.0                   # [S II] follows (measured 879.5)
    assert abs(res["meas"]["Hbeta"]["v_o3"] - 880.0) < 60.0  # so does the [O III] core (measured 879.9)
    # the systemic redshift is the input redshift corrected by v_sys
    assert m["z_sys"] == pytest.approx((1 + 0.25) * (1 + m["v_sys"] / C_KMS) - 1, rel=1e-9)


# ---------------------------------------------------------------------------
# (8) [O III] displaced from the low-ionisation lines
# ---------------------------------------------------------------------------
def test_oiii_displaced_from_the_low_ionisation_lines():
    """With [O III] at -500 km/s from Halpha, [N II] and [S II] (narrow Halpha
    EW 60 A) the Halpha systemic follows its own narrow group (within 60 km/s;
    measured +0.1) despite the [O III]-informed prior, the [O III] core is
    measured at -500 (measured -499.6), and sys_disagree is raised on Halpha.
    """
    kw = config(z=0.1, snr=15, seed=61,
                broad=[dict(line="Halpha", v=0.0, fwhm=3500.0, ew=150.0),
                       dict(line="Hbeta", v=0.0, fwhm=3500.0, ew=40.0)],
                narrow=dict(ew_ha=60.0, o3_dv=-500.0))
    sp, res = fit(**kw)
    m, c = measures(res, "Halpha")
    assert abs(m["v_sys"]) < 60.0, m["v_sys"]
    assert "sys_disagree" in c["flags"], c["flags"]
    # the preliminary [O III] fit that informs the Halpha start saw the displacement
    assert abs(m["v_o3_pre"] - (-500.0)) < 60.0, m["v_o3_pre"]
    assert m["o3_pre_snr"] >= 5.0
    assert c["label"] == "F" and abs(m["c50_sys"]) < 100.0
    mb, cb = measures(res, "Hbeta")
    assert abs(mb["v_o3"] - (-500.0)) < 60.0, mb["v_o3"]
    # narrow Hbeta follows the Halpha systemic, not [O III]
    assert mb["systemic_source"] == "Halpha prior"
    assert abs(mb["v_sys"]) < 60.0
    assert "sys_disagree" in cb["flags"]


# ---------------------------------------------------------------------------
# (9) a line at the edge of the spectrum
# ---------------------------------------------------------------------------
def edge_config(z):
    return config(z=z, snr=15, seed=71,
                  broad=[dict(line="Halpha", v=0.0, fwhm=3500.0, ew=300.0),
                         dict(line="Hbeta", v=0.0, fwhm=3500.0, ew=80.0)])


def red_coverage_kms(z):
    """Velocity of the last DESI pixel relative to Halpha at redshift z."""
    return (DESI_WAVE[-1] / (LAM["Halpha"] * (1.0 + z)) - 1.0) * C_KMS


@pytest.mark.parametrize("z", [0.475, 0.49], ids=["z0.475_flagged", "z0.49_not_fitted"])
def test_halpha_at_the_red_edge_of_the_spectrum(z):
    """Halpha within 4400 km/s (z = 0.475) or 1300 km/s (z = 0.49) of the red
    end of the DESI grid. In the first case the complex is fitted, carries the
    edge flag, is not measurable and shows no offset (measured c(1/2) - v_sys =
    +1.5 km/s, class F); in the second the core is not covered on both sides
    and the complex is not fitted at all. Hbeta is fitted normally in both.
    """
    cover = red_coverage_kms(z)
    kw = edge_config(z)
    sp, res = fit(**kw)
    if cover >= 3500.0:
        assert cover < 6000.0
        assert "Halpha" in res["fits"]
        m, c = measures(res, "Halpha")
        assert "edge" in c["flags"], c["flags"]
        assert c["label"] != "A"
        assert c["label"] == "F"
        assert abs(m["c50_sys"]) < 100.0, m["c50_sys"]
        assert m["v_cover_hi"] < 6000.0 and m["v_cover_hi"] == pytest.approx(cover, abs=5.0)
        assert not is_measurable(c["label"], c["flags"], m["broad_flux_snr"], m["fwhm"])
        # broad-component centres are confined to the covered range less 1000 km/s
        for k, val in res["fits"]["Halpha"]["d"].items():
            if k.startswith("Ha_b") and k.endswith("_v"):
                assert val <= cover - 1000.0 + 1e-6
    else:
        assert "Halpha" not in res["fits"]
        assert "Halpha" not in res["meas"] and "Halpha" not in res["cls"]
        assert "HA_class" not in blrfit.summary_row(res)
    mb, cb = measures(res, "Hbeta")
    assert cb["label"] == "F", cb
    assert abs(mb["c50_sys"]) < 100.0, mb["c50_sys"]     # measured -7 and +6 km/s
    assert "edge" not in cb["flags"]
    assert mb["v_cover_lo"] < -6000.0 and mb["v_cover_hi"] > 6000.0


# ---------------------------------------------------------------------------
# (10) result structure and frozen settings
# ---------------------------------------------------------------------------
def test_result_structure_and_frozen_settings():
    """The result of a fit carries the documented keys, the summary row the
    per-line columns, and ``settings`` the frozen constants of the production
    fitter: broad centres within +/-8000 km/s, narrow group within +/-1500,
    Delta BIC 10, the narrow-line-region wing, up to three broad Gaussians."""
    sp, res = fit(**shift_config(1200.0, 25))
    for k in ("z", "wave_rest", "flux_rest", "ivar_rest", "host_model", "host_info", "conti",
              "conti_model", "flux_sub", "o3_prefit", "settings", "fits", "meas", "cls", "mc", "err"):
        assert k in res, k
    assert res["z"] == 0.25
    assert set(res["fits"]) == {"Halpha", "Hbeta"} == set(res["meas"]) == set(res["cls"])
    assert res["mc"] == {} and res["err"] == {}              # nmc = 0
    s = res["settings"]
    assert s["v_broad_max"] == 8000.0 == V_BROAD_MAX
    assert s["v_narrow_max"] == 1500.0 == V_NARROW_MAX
    assert s["dbic"] == 10.0 == DBIC
    assert s["nlr_wing"] is True and NLR_WING is True
    assert s["max_broad"] == 3 == MAX_BROAD
    assert s["sig_broad_min"] == pytest.approx(1200.0 / S2F)
    assert s["err_floor"] == 0.02
    assert s["host"] is True and s["fe"] is True and s["oiii_wing"] is True and s["heii"] is True
    assert s["sii_mode"] == "soft" and s["broad_width_slope"] == 0.4
    for name in ("Halpha", "Hbeta"):
        r = res["fits"][name]
        for k in ("d", "comps", "chi2", "npix", "nfree", "bic", "n_broad", "x", "y", "w", "v_cover", "all_bic"):
            assert k in r, (name, k)
        assert 1 <= r["n_broad"] <= 3 and len(r["all_bic"]) == 3
        assert r["bic"] == pytest.approx(r["chi2"] + r["nfree"] * np.log(r["npix"]))
        m = res["meas"][name]
        for k in ("v_sys", "c50_sys", "v_peak_sys", "fwhm", "AI", "KI", "n_peaks", "broad_flux_snr",
                  "broad_peak_snr", "sys_snr", "host_frac", "pl_alpha", "v_cover_lo", "v_cover_hi",
                  "chi2_red", "systemic_source", "broad_flux", "broad_ew", "z_sys"):
            assert k in m, (name, k)
        assert m["z_sys"] == pytest.approx((1 + res["z"]) * (1 + m["v_sys"] / C_KMS) - 1)
    assert res["meas"]["Halpha"]["systemic_source"] == "own narrow group"
    assert res["meas"]["Hbeta"]["systemic_source"] == "Halpha prior"
    # summary row
    row = blrfit.summary_row(res, prefix_meta=dict(targetid=1))
    assert row["targetid"] == 1 and row["z_in"] == 0.25
    for k in ("HA_class", "HB_class", "HA_c50_sys", "HB_c50_sys", "HA_v_sys", "HB_v_sys", "HA_fwhm",
              "HA_flags", "HB_flags", "HA_reason", "HA_systemic_source", "HA_broad_flux_snr",
              "host_frac", "host_applied", "conti_pl_alpha", "HA_n_broad", "HB_v_o3"):
        assert k in row, k
    assert row["HA_class"] == res["cls"]["Halpha"]["label"] == "A"
    assert row["HB_class"] == res["cls"]["Hbeta"]["label"] == "A"
    assert row["HA_c50_sys"] == res["meas"]["Halpha"]["c50_sys"]
    assert row["HB_c50_sys"] == res["meas"]["Hbeta"]["c50_sys"]
    assert row["HA_flags"] == "" and isinstance(row["HA_reason"], str) and row["HA_reason"]
    assert "MG_class" not in row
    # the broad equivalent width against the total continuum is the injected one
    # to within 10 per cent (measured 300 A for 300 A): the continuum is a pure power law here
    assert abs(res["meas"]["Halpha"]["broad_ew"] - 300.0) < 30.0, res["meas"]["Halpha"]["broad_ew"]
    # remeasuring the stored fits with the current measure and classify code
    # reproduces the classes and offsets
    before = {n: (res["cls"][n]["label"], res["meas"][n]["c50_sys"]) for n in res["meas"]}
    res2 = blrfit.remeasure(copy.deepcopy(res))
    after = {n: (res2["cls"][n]["label"], res2["meas"][n]["c50_sys"]) for n in res2["meas"]}
    assert after == before


# ---------------------------------------------------------------------------
# (11) an [O III] blue wing, and pedestals outside the wing model
# ---------------------------------------------------------------------------
def test_oiii_blue_wing_does_not_move_the_systemic():
    """A blueshifted [O III] wing (velocity -300 or -500 km/s, sigma 600 or 800
    km/s, 30 or 50 per cent of the core flux) is absorbed by the wing component:
    the systemic velocity stays at the narrow lines (measured within 1 km/s),
    no sys_disagree flag is raised, and the fitted wing velocity and width are
    the injected ones (measured -308 / 598 and -510 / 797)."""
    for wing in ((-300.0, 600.0, 0.3), (-500.0, 800.0, 0.5)):
        kw = config(z=0.1, snr=15, seed=61,
                    broad=[dict(line="Halpha", v=0.0, fwhm=3500.0, ew=150.0),
                           dict(line="Hbeta", v=0.0, fwhm=3500.0, ew=40.0)],
                    narrow=dict(ew_ha=60.0, o3=10.0, o3_wing=wing))
        sp, res = fit(**kw)
        for name in ("Halpha", "Hbeta"):
            m, c = measures(res, name)
            assert abs(m["v_sys"]) < 60.0, (wing, name, m["v_sys"])
            assert "sys_disagree" not in c["flags"], (wing, name, c["flags"])
            assert c["label"] == "F" and abs(m["c50_sys"]) < 100.0
        d = res["fits"]["Hbeta"]["d"]
        assert abs(d["w_v"] - wing[0]) < 100.0, (wing, d["w_v"])
        assert abs(d["w_sig"] - wing[1]) < 150.0, (wing, d["w_sig"])
        assert abs(res["meas"]["Hbeta"]["v_o3"]) < 60.0


@pytest.mark.slow
@pytest.mark.parametrize("ped_width, ped_v", [(4.0, 100.0), (5.0, 0.0), (5.0, 100.0), (5.0, -100.0)])
def test_pedestal_wider_than_the_wing_model(ped_width, ped_v):
    """The narrow-line-region wing is bounded at sigma 510 km/s (FWHM 1200, the
    narrow/broad boundary). A pedestal of 4 times the core width (sigma 600)
    displaced by +100 km/s still leaves a weak broad Halpha at +800 km/s
    within 100 km/s (measured +704, class A). A pedestal of 5 times the core
    width (sigma 750, FWHM 1766: itself beyond the boundary) is only partly
    absorbed by the wing, which saturates at its bound; the rest goes into
    the broad components, which then form a two-peaked profile (class B) with
    c(1/2) biased low by 130-230 km/s (measured +621, +669, +568 for
    displacements 0, +100, -100). The systemic velocity is unaffected
    (measured within 1 km/s). This documents a limit of the frozen model
    rather than a validated regime."""
    kw = config(z=0.1, snr=15, seed=41,
                broad=[dict(line="Halpha", v=800.0, fwhm=3500.0, ew=40.0)],
                narrow=dict(ew_ha=80.0, ped_frac=0.3, ped_width=ped_width, ped_v=ped_v))
    sp, res = fit(**kw)
    m, c = measures(res, "Halpha")
    assert abs(m["v_sys"]) < 60.0, m["v_sys"]
    assert m["nw_sig"] > 480.0                       # the wing saturates at its 510 km/s bound
    if ped_width <= 4.0:
        assert c["label"] == "A" and abs(m["c50_sys"] - 800.0) < 120.0, (c["label"], m["c50_sys"])
    else:
        assert c["label"] in ("A", "B", "C") and 500.0 < m["c50_sys"] < 900.0, (c["label"], m["c50_sys"])
