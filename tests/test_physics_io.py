"""Gzipped SDSS and DESI files, and the continuum luminosity of the fitted power law."""
import gzip
import os
import shutil

import numpy as np
import pytest

import blrfit
from blrfit.constants import C_KMS, LAM, PL_PIVOT, S2F
from blrfit.io import read_spectrum, read_sdss, read_desi, is_sdss_spec, is_desi_coadd, is_desi_spectra
from blrfit.io.desi import redrock_sibling
from blrfit.physics import lambda_l_lambda, continuum_luminosity, lumdist_cm
from conftest import SDSS_EXAMPLE, DESI_EXAMPLE, DESI_TARGETID, EXAMPLES

REDROCK_EXAMPLE = os.path.join(EXAMPLES, "redrock-main-dark-17260-39627574082538900.fits")


def _gzip_copy(src, dst):
    with open(src, "rb") as f, gzip.open(dst, "wb") as g:
        shutil.copyfileobj(f, g)
    return str(dst)


# ----------------------------------------------------------------------------
# gzipped files
# ----------------------------------------------------------------------------
def test_sniffers_accept_gzipped_names():
    assert is_sdss_spec("spec-0651-52141-0072.fits") and is_sdss_spec("spec-0651-52141-0072.fits.gz")
    assert is_sdss_spec("/a/b/SPEC-0651-52141-0072.FITS.GZ")
    assert not is_sdss_spec("spec-0651-52141-0072.fits.bz2") and not is_sdss_spec("spec-0651.txt")
    assert is_desi_coadd("coadd-main-dark-17260.fits") and is_desi_coadd("coadd-main-dark-17260.fits.gz")
    assert not is_desi_coadd("coadd-main-dark-17260.fits.gz.part")
    assert is_desi_spectra("spectra-main-dark-17260.fits.gz") and not is_desi_coadd("spectra-main-dark-17260.fits.gz")


def test_gzipped_sdss_reads_identically(tmp_path):
    gz = _gzip_copy(SDSS_EXAMPLE, tmp_path / "spec-0651-52141-0072.fits.gz")
    a = read_spectrum(SDSS_EXAMPLE)
    b = read_spectrum(gz)
    assert b["kind"] == "sdss" and b["path"] == gz
    for k in ("wave", "flux", "ivar"):
        assert np.array_equal(a[k], b[k]), k
    for k in ("z", "zerr", "mjd", "plate", "fiber", "ra", "dec", "date", "cls"):
        assert a[k] == b[k], k
    assert read_sdss(gz)["z"] == a["z"]


def test_gzipped_desi_reads_identically(tmp_path):
    gz = _gzip_copy(DESI_EXAMPLE, tmp_path / (os.path.basename(DESI_EXAMPLE) + ".gz"))
    _gzip_copy(REDROCK_EXAMPLE, tmp_path / (os.path.basename(REDROCK_EXAMPLE) + ".gz"))
    a = read_spectrum(DESI_EXAMPLE, targetid=DESI_TARGETID)
    b = read_spectrum(gz, targetid=DESI_TARGETID)
    assert b["kind"] == "desi" and b["targetid"] == DESI_TARGETID
    for k in ("wave", "flux", "ivar", "mask"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    for k in ("z", "zerr", "zwarn", "spectype", "ebv", "ra", "dec", "mjd", "mjd_min", "mjd_max", "nexp",
              "survey", "program", "healpix"):
        assert a[k] == b[k], k
    assert read_desi(gz, DESI_TARGETID, use_desispec=False)["z"] == a["z"]
    with pytest.raises(ValueError):
        read_spectrum(gz)


def test_redrock_sibling_across_compression(tmp_path):
    """A gzipped coadd finds a plain redrock file and a plain coadd a gzipped one;
    the sibling with the coadd's own suffix is preferred when both exist."""
    plain = str(tmp_path / os.path.basename(DESI_EXAMPLE))
    shutil.copy(DESI_EXAMPLE, plain)
    gz = _gzip_copy(DESI_EXAMPLE, plain + ".gz")
    assert redrock_sibling(plain) is None and redrock_sibling(gz) is None
    rr_plain = str(tmp_path / os.path.basename(REDROCK_EXAMPLE))
    shutil.copy(REDROCK_EXAMPLE, rr_plain)
    assert redrock_sibling(plain) == rr_plain and redrock_sibling(gz) == rr_plain
    assert read_spectrum(gz, targetid=DESI_TARGETID)["z"] == pytest.approx(0.2203154, abs=1e-6)
    rr_gz = _gzip_copy(REDROCK_EXAMPLE, rr_plain + ".gz")
    assert redrock_sibling(gz) == rr_gz and redrock_sibling(plain) == rr_plain
    os.remove(rr_plain)
    assert redrock_sibling(plain) == rr_gz
    assert read_spectrum(plain, targetid=DESI_TARGETID)["z"] == pytest.approx(0.2203154, abs=1e-6)
    assert redrock_sibling("spec-0651-52141-0072.fits") is None


# ----------------------------------------------------------------------------
# continuum luminosity
# ----------------------------------------------------------------------------
def _planck18_dl_cm(z):
    from astropy.cosmology import Planck18
    import astropy.units as u
    return float(Planck18.luminosity_distance(z).to(u.cm).value)


def test_lambda_l_lambda_helper():
    """The formula on a synthetic conti dictionary: 4 pi D_L^2 lam f_rest(lam) 1e-17
    with f_rest the fitted power law at lam, and no (1+z) beyond the one the fit applied."""
    z = 0.25; conti = dict(pl_norm=10.0, pl_alpha=-1.5)
    dl = _planck18_dl_cm(z)
    assert lumdist_cm(z) == dl
    for lam in (5100.0, 3000.0, 4400.0):
        expect = 4 * np.pi * dl**2 * lam * 10.0 * (lam / PL_PIVOT) ** -1.5 * 1e-17
        assert lambda_l_lambda(conti, z, lam_rest=lam) == pytest.approx(expect, rel=1e-12)
    assert lambda_l_lambda(conti, z) == lambda_l_lambda(conti, z, lam_rest=5100.0)
    # the pivot: lambda L_lambda(3000) is 4 pi D_L^2 * 3000 * pl_norm * 1e-17 whatever the slope
    assert lambda_l_lambda(dict(pl_norm=10.0, pl_alpha=0.7), z, lam_rest=PL_PIVOT) == pytest.approx(
        4 * np.pi * dl**2 * PL_PIVOT * 10.0 * 1e-17, rel=1e-12)
    # summary-row spelling of the keys, and arrays
    row = dict(conti_pl_norm=np.array([10.0, 0.0, np.nan]), conti_pl_alpha=np.array([-1.5, -1.5, -1.5]))
    out = lambda_l_lambda(row, np.array([z, z, z]))
    assert out[0] == pytest.approx(lambda_l_lambda(conti, z), rel=1e-12)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert np.array_equal(lumdist_cm(np.array([z, 0.5])), [_planck18_dl_cm(z), _planck18_dl_cm(0.5)])
    assert np.isnan(lambda_l_lambda(dict(pl_norm=-1.0, pl_alpha=-1.5), z))
    assert np.isnan(lambda_l_lambda(dict(pl_alpha=-1.5), z))
    assert np.isnan(continuum_luminosity(dict(conti={}, z=z)))


def test_continuum_luminosity_round_trip_through_the_fitter():
    """An observed-frame spectrum built from a rest-frame lambda L_lambda(5100) of
    1e44 erg/s (Hogg 1999 eqs 22-23: f_obs(lam_obs) = L_lambda(lam_rest) / (4 pi D_L^2 (1+z))
    per observed Angstrom) is fitted and the luminosity recovered within 0.01 dex."""
    z = 0.25; alpha = -1.5; l5100 = 1e44
    dl = _planck18_dl_cm(z)
    wave = np.arange(3600.0, 9824.01, 0.8)                    # DESI grid, observed frame
    wr = wave / (1 + z)
    l_lambda = l5100 / 5100.0 * (wr / 5100.0) ** alpha         # erg/s/A per rest-frame Angstrom
    f_obs = l_lambda / (4 * np.pi * dl**2 * (1 + z)) * 1e17    # per observed Angstrom, 1e-17 units
    cont_rest = f_obs * (1 + z)                                # what the fit works on
    # a modest broad Hbeta and the narrow lines so that the Hbeta complex fits normally; the line
    # flux is defined in the rest frame and carried to the observed frame with the same Jacobian
    def gauss(lam0, v, sigma_kms, flux_rest):
        lc = lam0 * (1 + v / C_KMS); sl = lc * sigma_kms / C_KMS
        return flux_rest / (sl * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wr - lc) / sl) ** 2) / (1 + z)
    c_hb = float(np.interp(LAM["Hbeta"], wr, cont_rest))
    lines = (gauss(LAM["Hbeta"], 0.0, 3000.0 / S2F, 60.0 * c_hb)
             + gauss(LAM["Hbeta"], 0.0, 150.0, 5.0 * c_hb)
             + gauss(LAM["OIII5007"], 0.0, 150.0, 40.0 * c_hb)
             + gauss(LAM["OIII4959"], 0.0, 150.0, 40.0 * c_hb / 2.98))
    model = f_obs + lines
    rng = np.random.default_rng(3)
    sigma = np.full_like(model, c_hb / (1 + z) / 30.0)          # S/N 30 per pixel at Hbeta
    flux = model + rng.standard_normal(model.size) * sigma
    ivar = 1.0 / sigma**2
    res = blrfit.fit_spectrum(wave, flux, ivar, z, host=False, fe=False, complexes=("Hbeta",))
    assert "Hbeta" in res["meas"]
    assert res["conti"]["pl_alpha"] == pytest.approx(alpha, abs=0.02)
    lum = continuum_luminosity(res)
    assert abs(np.log10(lum / l5100)) < 0.01, np.log10(lum / l5100)
    assert lum == lambda_l_lambda(res["conti"], res["z"])
    assert lum == lambda_l_lambda(blrfit.summary_row(res), res["z"])
    # the same luminosity in erg/s per unit as broad_lum: both use D_L(z) of Planck 2018
    assert res["meas"]["Hbeta"]["broad_lum"] == pytest.approx(
        4 * np.pi * dl**2 * res["meas"]["Hbeta"]["broad_flux"] * 1e-17, rel=1e-9)
