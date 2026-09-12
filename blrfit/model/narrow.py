"""
The narrow-line system of each complex and the penalties that constrain it.

Every complex is a list of Gaussian components ``(label, lam0, amp_name,
v_name, sig_name, kind, ratio_to)`` sharing parameters through their names:
components with the same ``v_name``/``sig_name`` move together, and
``ratio_to = (source_amp_name, factor)`` fixes a doublet ratio. ``kind`` is
'narrow' (a narrow core), 'nwing' (the narrow-line-region wing under a narrow
core), 'wing' (the blueshifted [O III] wing) or 'broad'. Only 'broad'
components enter the profile measurements.

Halpha complex: narrow Halpha and the [N II] doublet share one velocity and
width (``n_v``, ``n_sig``), which define the systemic velocity; [S II] has its
own kinematics tied softly to them; one narrow-line-region wing with a common
amplitude fraction sits under every narrow line.

Hbeta complex: the [O III] doublet is a core plus a blueshifted wing, ordered in
width and amplitude; narrow Hbeta and He II 4686 either follow the Halpha
systemic (velocity, width and wing fixed from the Halpha fit, the default when
the Halpha narrow lines are detected) or are tied to the [O III] core.

Mg II complex: one narrow Mg II line inside a +/-700 km/s window.

The reasons for each constraint are given in ``constants.py``.
"""
from __future__ import annotations

import numpy as np

from ..constants import (C_KMS, LAM, R_NII, R_OIII, V_NARROW_MAX, V_NARROW_MAX_MGII,
                         SIG_NARROW_MIN, SIG_NARROW_MAX, SIG_O3_CORE_MAX, SIG_WING_MIN,
                         SIG_WING_MAX, V_WING_MIN, V_WING_MAX, NLR_WING, NW_F_MAX,
                         NW_V_PRIOR_KMS, NW_F_PRIOR, SII_PRIOR_SIG_FRAC, SII_PRIOR_V_KMS,
                         O3_ORDER_WIDTH, O3_ORDER_AMPLITUDE, O3_AMP_ORDER_SCALE)


def amp_guess(wave, f, lam0, halfwin=15.0):
    """Starting amplitude: the maximum of the continuum-subtracted data within
    +/- halfwin Angstrom of the line, at least 1e-3."""
    m = (wave > lam0 - halfwin) & (wave < lam0 + halfwin)
    return float(max(np.nanmax(f[m]) if m.sum() else 0.0, 1e-3))


def add_halpha_narrow(ps, comps, wave, fsub, amax, v_sys_prior=None, sig_sys_prior=None):
    """Narrow Halpha + [N II] (tied), [S II] (soft-tied) and the narrow-line-region wing."""
    ps.add("n_v", 0.0 if v_sys_prior is None else v_sys_prior, -V_NARROW_MAX, V_NARROW_MAX,
           fixed=v_sys_prior is not None)
    ps.add("n_sig", 120.0 if sig_sys_prior is None else sig_sys_prior, SIG_NARROW_MIN,
           SIG_NARROW_MAX, fixed=sig_sys_prior is not None)
    for lab, lam0 in (("Halpha_n", LAM["Halpha"]), ("NII6584", LAM["NII6584"])):
        ps.add(f"{lab}_A", 0.5 * amp_guess(wave, fsub, lam0, 6.0), 0.0, amax)
        comps.append((lab, lam0, f"{lab}_A", "n_v", "n_sig", "narrow", None))
    # [S II] with its own kinematics; the soft tie to (n_v, n_sig) is a penalty in the fit
    ps.add("s2_v", 0.0 if v_sys_prior is None else v_sys_prior, -V_NARROW_MAX, V_NARROW_MAX)
    ps.add("s2_sig", 120.0 if sig_sys_prior is None else sig_sys_prior, SIG_NARROW_MIN,
           SIG_NARROW_MAX)
    for lab, lam0 in (("SII6716", LAM["SII6716"]), ("SII6731", LAM["SII6731"])):
        ps.add(f"{lab}_A", 0.5 * amp_guess(wave, fsub, lam0, 6.0), 0.0, amax)
        comps.append((lab, lam0, f"{lab}_A", "s2_v", "s2_sig", "narrow", None))
    comps.append(("NII6548", LAM["NII6548"], "NII6584_A", "n_v", "n_sig", "narrow",
                  ("NII6584_A", 1.0 / R_NII)))
    if NLR_WING:
        n_v0 = 0.0 if v_sys_prior is None else float(v_sys_prior)
        ps.add("nw_f", 0.10, 0.0, NW_F_MAX)
        ps.add("nw_v", n_v0, -V_NARROW_MAX, V_NARROW_MAX)
        ps.add("nw_sig", 300.0, SIG_NARROW_MIN, SIG_NARROW_MAX)
        ps.derive("Halpha_nw_A", ["Halpha_n_A", "nw_f"])
        ps.derive("NII6584_nw_A", ["NII6584_A", "nw_f"])
        ps.derive("NII6548_nw_A", ["NII6584_A", 1.0 / R_NII, "nw_f"])
        ps.derive("SII6716_nw_A", ["SII6716_A", "nw_f"])
        ps.derive("SII6731_nw_A", ["SII6731_A", "nw_f"])
        for lab, lam0 in (("Halpha_nw", LAM["Halpha"]), ("NII6584_nw", LAM["NII6584"]),
                          ("NII6548_nw", LAM["NII6548"]), ("SII6716_nw", LAM["SII6716"]),
                          ("SII6731_nw", LAM["SII6731"])):
            comps.append((lab, lam0, f"{lab}_A", "nw_v", "nw_sig", "nwing", None))


def oiii_start_velocity(wave, fsub, v0):
    """Starting velocity of the [O III] core from the data: the smoothed peak near
    5007 A inside the narrow window, if it stands 5 MAD above the median.

    An optimiser started at zero velocity (or at the Halpha prior) sees no
    gradient towards an [O III] line several hundred km/s away, and the wing
    then takes the line."""
    vg = (wave / LAM["OIII5007"] - 1.0) * C_KMS
    sel = np.abs(vg) < V_NARROW_MAX
    if sel.sum() > 10:
        ys = np.convolve(fsub[sel], np.ones(3) / 3.0, mode="same")
        j = int(np.argmax(ys))
        mad = 1.4826 * np.median(np.abs(ys - np.median(ys))) + 1e-12
        if (ys[j] - np.median(ys)) / mad > 5.0:
            return float(np.clip(vg[sel][j], -V_NARROW_MAX + 1, V_NARROW_MAX - 1))
    return v0


def add_hbeta_narrow(ps, comps, wave, fsub, amax, v_sys_prior=None, sig_sys_prior=None,
                     oiii_wing=True, heii=True, nw_prior=None):
    """[O III] core (+ wing), narrow Hbeta (+ carried-over wing) and He II 4686."""
    o3_v0 = oiii_start_velocity(wave, fsub, 0.0 if v_sys_prior is None else float(v_sys_prior))
    ps.add("o3_v", o3_v0, -V_NARROW_MAX, V_NARROW_MAX)
    ps.add("o3_sig", 120.0, SIG_NARROW_MIN, SIG_O3_CORE_MAX)
    ps.add("OIII5007c_A", amp_guess(wave, fsub, LAM["OIII5007"], 6.0), 0.0, amax)
    comps.append(("OIII5007c", LAM["OIII5007"], "OIII5007c_A", "o3_v", "o3_sig", "narrow", None))
    comps.append(("OIII4959c", LAM["OIII4959"], "OIII5007c_A", "o3_v", "o3_sig", "narrow",
                  ("OIII5007c_A", 1.0 / R_OIII)))
    if v_sys_prior is not None:
        # narrow Hbeta and He II follow the Halpha systemic
        ps.add("n_v", v_sys_prior, -V_NARROW_MAX, V_NARROW_MAX, fixed=True)
        ps.add("n_sig", sig_sys_prior if sig_sys_prior else 120.0, SIG_NARROW_MIN,
               SIG_NARROW_MAX, fixed=sig_sys_prior is not None)
    else:
        # ... or the [O III] core (Shen et al. 2011)
        ps.add("n_v", 0.0, -V_NARROW_MAX, V_NARROW_MAX); ps.link("n_v", "o3_v")
        ps.add("n_sig", 120.0, SIG_NARROW_MIN, SIG_NARROW_MAX); ps.link("n_sig", "o3_sig")
    ps.add("Hbeta_n_A", 0.3 * amp_guess(wave, fsub, LAM["Hbeta"], 6.0), 0.0, amax)
    comps.append(("Hbeta_n", LAM["Hbeta"], "Hbeta_n_A", "n_v", "n_sig", "narrow", None))
    if NLR_WING and nw_prior is not None:      # (fraction, velocity, width) fixed from the Halpha fit
        ps.add("nw_f", nw_prior[0], 0.0, 1.0, fixed=True)
        ps.add("nw_v", nw_prior[1], -V_NARROW_MAX, V_NARROW_MAX, fixed=True)
        ps.add("nw_sig", nw_prior[2], SIG_NARROW_MIN, SIG_NARROW_MAX, fixed=True)
        ps.derive("Hbeta_nw_A", ["Hbeta_n_A", "nw_f"])
        comps.append(("Hbeta_nw", LAM["Hbeta"], "Hbeta_nw_A", "nw_v", "nw_sig", "nwing", None))
    if heii:
        ps.add("HeII_n_A", 0.05 * amp_guess(wave, fsub, LAM["HeII4686"], 6.0), 0.0, amax)
        comps.append(("HeII4686_n", LAM["HeII4686"], "HeII_n_A", "n_v", "n_sig", "narrow", None))
    if oiii_wing:
        ps.add("w_v", float(np.clip(o3_v0 - 300.0, V_WING_MIN + 1, V_WING_MAX - 1)), V_WING_MIN, V_WING_MAX)
        ps.add("w_sig", 600.0, SIG_WING_MIN, SIG_WING_MAX)
        ps.add("OIII5007w_A", 0.15 * amp_guess(wave, fsub, LAM["OIII5007"], 6.0), 0.0, amax)
        comps.append(("OIII5007w", LAM["OIII5007"], "OIII5007w_A", "w_v", "w_sig", "wing", None))
        comps.append(("OIII4959w", LAM["OIII4959"], "OIII5007w_A", "w_v", "w_sig", "wing",
                      ("OIII5007w_A", 1.0 / R_OIII)))


def add_mgii_narrow(ps, comps, wave, fsub, amax):
    ps.add("n_v", 0.0, -V_NARROW_MAX_MGII, V_NARROW_MAX_MGII)
    ps.add("n_sig", 200.0, SIG_NARROW_MIN, SIG_NARROW_MAX)
    ps.add("MgII_n_A", 0.2 * amp_guess(wave, fsub, LAM["MgII"], 8.0), 0.0, amax)
    comps.append(("MgII_n", LAM["MgII"], "MgII_n_A", "n_v", "n_sig", "narrow", None))


# ----------------------------------------------------------------------------
# Penalty terms appended to the weighted residual vector (Gaussian priors and
# hinges). Each function returns a list of floats.
# ----------------------------------------------------------------------------
def systemic_prior_penalty(d, n_v_prior):
    """Gaussian prior pulling the Halpha narrow group towards the [O III] pre-fit velocity."""
    return [(d["n_v"] - n_v_prior[0]) / n_v_prior[1]]


def nlr_wing_penalty(d):
    """Wing at least as wide as the core (hinge), velocity near the core (prior),
    fraction weakly pulled to zero (prior)."""
    return [max(0.0, d["n_sig"] - d["nw_sig"]) / 3.0,
            (d["nw_v"] - d["n_v"]) / NW_V_PRIOR_KMS,
            d["nw_f"] / NW_F_PRIOR]


def sii_soft_tie_penalty(d):
    """Gaussian priors tying the [S II] width and velocity to narrow Halpha + [N II]."""
    return [(d["s2_sig"] - d["n_sig"]) / (SII_PRIOR_SIG_FRAC * d["n_sig"]),
            (d["s2_v"] - d["n_v"]) / SII_PRIOR_V_KMS]


def oiii_order_penalty(d):
    """[O III] wing at least as wide as the core, core at least as tall as the wing."""
    pen = [max(0.0, d["o3_sig"] - d["w_sig"]) / 10.0]
    if O3_ORDER_AMPLITUDE:
        ac, aw = d["OIII5007c_A"], d["OIII5007w_A"]
        pen.append(max(0.0, aw - ac) / (O3_AMP_ORDER_SCALE * (ac + aw) + 1e-9))
    return pen


def has_oiii_ordering(name, ps):
    return name == "Hbeta" and O3_ORDER_WIDTH and "w_sig" in ps.names
