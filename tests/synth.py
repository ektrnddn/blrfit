"""
Synthetic DESI-like spectra with known line configurations, for the validation tests.

A spectrum is a power-law continuum, an optional host galaxy (the first Yip et
al. 2004 eigenspectrum, which carries its own stellar features and weak
emission lines), narrow lines with tied kinematics and optional non-Gaussian
pedestals, an optional [O III] blue wing, the broad Balmer lines as one or more
Gaussians at chosen velocities and widths, and Gaussian noise. Everything is
built on the DESI wavelength grid (3600-9824 A, 0.8 A) in the observed frame at
the requested redshift; the input redshift given to the fitter is the redshift
at which the narrow lines sit unless ``v_sys`` displaces them.

Velocities are km/s relative to the input redshift; the truth dictionary
records the broad-profile measures the fitter should recover.
"""
from __future__ import annotations

import numpy as np

from blrfit.constants import C_KMS, LAM, R_OIII, R_NII, S2F
from blrfit.model.continuum import pca_templates
from blrfit.measure import profile_measures

DESI_WAVE = np.arange(3600.0, 9824.01, 0.8)
SDSS_LOGLAM = 10.0 ** np.arange(np.log10(3800.0), np.log10(9200.0), 1e-4)


def gauss_v(wave_rest, lam0, v, sigma, flux):
    """Gaussian of integrated flux ``flux`` at lam0 (1 + v/c) with width sigma (km/s)."""
    lc = lam0 * (1.0 + v / C_KMS); sl = lc * sigma / C_KMS
    return flux / (sl * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wave_rest - lc) / sl) ** 2)


def make_spectrum(z=0.25, snr=15.0, pl_norm=10.0, pl_alpha=-1.5, host_frac=0.0,
                  broad=None, narrow=None, v_sys=0.0, seed=0, wave=None, err_floor=0.0):
    """Build one synthetic spectrum.

    snr        continuum signal-to-noise ratio per pixel at the power-law level near Halpha
    pl_norm    power-law flux at 3000 A rest (1e-17 erg/s/cm^2/A)
    host_frac  fraction of the total continuum at 4700 A rest contributed by the host
    broad      list of dict(line='Halpha'|'Hbeta', v=km/s, fwhm=km/s, ew=A) components; the
               equivalent width is against the AGN continuum at the line
    narrow     dict(sigma=km/s, ew_ha=A, nii=[N II]6584/Halpha_n, sii=[S II]6716/Halpha_n,
               sii_ratio=6716/6731, balmer=Halpha_n/Hbeta_n, o3=[O III]5007/Hbeta_n,
               ped_frac=fraction of each narrow line's flux in a pedestal, ped_width=pedestal
               width / core width, ped_v=pedestal velocity offset, o3_wing=(v, sigma, flux
               fraction) of an [O III] blue wing, sii_dv=[S II] velocity offset, o3_dv=[O III]
               velocity offset)
    v_sys      velocity of the whole narrow-line system relative to the input redshift
    wave       observed-frame grid (default the DESI grid)
    Returns dict(wave, flux, ivar, truth) with the arrays in the observed frame.
    """
    rng = np.random.default_rng(seed)
    wave = DESI_WAVE if wave is None else np.asarray(wave, float)
    wr = wave / (1 + z)
    pl = pl_norm * (wr / 3000.0) ** pl_alpha
    cont_agn = pl.copy()
    host = np.zeros_like(wr)
    if host_frac > 0:
        P = pca_templates()
        e0 = np.interp(wr, P["gw"], P["gp"][0], left=0.0, right=0.0)
        e0 = np.clip(e0, 0, None)
        ref = float(np.interp(4700.0, P["gw"], P["gp"][0]))
        agn_ref = float(pl_norm * (4700.0 / 3000.0) ** pl_alpha)
        host = e0 / ref * agn_ref * host_frac / (1.0 - host_frac)
    model = cont_agn + host
    truth = dict(z=z, v_sys=v_sys, host_frac=host_frac, broad={}, narrow=dict(narrow or {}))

    nar = dict(sigma=150.0, ew_ha=40.0, nii=1.0, sii=0.4, sii_ratio=1.3, balmer=3.0, o3=8.0,
               ped_frac=0.0, ped_width=3.0, ped_v=0.0, o3_wing=None, sii_dv=0.0, o3_dv=0.0)
    nar.update(narrow or {})
    if nar["ew_ha"] > 0:
        c_ha = float(np.interp(LAM["Halpha"], wr, cont_agn))
        f_ha = nar["ew_ha"] * c_ha
        lines = [(LAM["Halpha"], f_ha, v_sys), (LAM["NII6584"], nar["nii"] * f_ha, v_sys),
                 (LAM["NII6548"], nar["nii"] * f_ha / R_NII, v_sys),
                 (LAM["SII6716"], nar["sii"] * f_ha, v_sys + nar["sii_dv"]),
                 (LAM["SII6731"], nar["sii"] * f_ha / nar["sii_ratio"], v_sys + nar["sii_dv"]),
                 (LAM["Hbeta"], f_ha / nar["balmer"], v_sys),
                 (LAM["OIII5007"], nar["o3"] * f_ha / nar["balmer"], v_sys + nar["o3_dv"]),
                 (LAM["OIII4959"], nar["o3"] * f_ha / nar["balmer"] / R_OIII, v_sys + nar["o3_dv"])]
        for lam0, f, v in lines:
            core = f * (1.0 - nar["ped_frac"])
            model = model + gauss_v(wr, lam0, v, nar["sigma"], core)
            if nar["ped_frac"] > 0:
                model = model + gauss_v(wr, lam0, v + nar["ped_v"], nar["sigma"] * nar["ped_width"], f * nar["ped_frac"])
        if nar["o3_wing"] is not None:
            wv, wsig, wfrac = nar["o3_wing"]
            f_o3 = nar["o3"] * f_ha / nar["balmer"]
            model = model + gauss_v(wr, LAM["OIII5007"], v_sys + nar["o3_dv"] + wv, wsig, wfrac * f_o3)
            model = model + gauss_v(wr, LAM["OIII4959"], v_sys + nar["o3_dv"] + wv, wsig, wfrac * f_o3 / R_OIII)
        truth["narrow"].update(sigma=nar["sigma"], flux_ha=f_ha)

    vgrid = np.arange(-25000.0, 25000.01, 5.0)
    for comp in (broad or []):
        lam0 = LAM[comp["line"]]
        c_line = float(np.interp(lam0, wr, cont_agn))
        flux = comp["ew"] * c_line
        sig = comp["fwhm"] / S2F
        model = model + gauss_v(wr, lam0, comp["v"], sig, flux)
        P = truth["broad"].setdefault(comp["line"], dict(profile=np.zeros_like(vgrid), flux=0.0))
        lam_g = lam0 * (1.0 + vgrid / C_KMS)
        P["profile"] = P["profile"] + gauss_v(lam_g, lam0, comp["v"], sig, flux)
        P["flux"] += flux
    for line, P in truth["broad"].items():
        m = profile_measures(vgrid, P["profile"])
        P.update({k: m[k] for k in ("v_peak", "centroid", "c25", "c50", "c75", "fwhm", "AI", "KI", "n_peaks")})
        for k in ("v_peak", "centroid", "c50"):
            P[f"{k}_sys"] = P[k] - v_sys
        P.pop("profile")

    noise_level = float(np.interp(LAM["Halpha"], wr, cont_agn)) / snr
    sigma = np.full_like(model, noise_level)
    if err_floor > 0:
        sigma = np.sqrt(sigma**2 + (err_floor * np.abs(model)) ** 2)
    flux = model + rng.standard_normal(model.size) * sigma
    ivar = 1.0 / sigma**2
    truth["noise"] = noise_level; truth["model"] = model
    return dict(wave=wave, flux=flux, ivar=ivar, truth=truth)


__all__ = ["make_spectrum", "gauss_v", "DESI_WAVE", "SDSS_LOGLAM"]
