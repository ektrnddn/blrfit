"""Offline pieces not covered elsewhere: the fetch URL and path construction, the
redrock reader, the rejection of per-exposure files, the diagnostic figure, and
a Mg II fit on a synthetic spectrum."""
import os

import numpy as np
import pytest

import blrfit
from blrfit.constants import C_KMS, LAM
from blrfit.io import read_redrock, is_desi_spectra, read_spectrum
from blrfit.io.fetch import sdss_spec_url, desi_coadd_url, desi_healpix, DESI_SURVEY_PROGRAMS
from blrfit.io.sdss import mjd_to_date
from conftest import EXAMPLES, DESI_EXAMPLE, DESI_TARGETID, SDSS_EXAMPLE, Z_J001224


def test_sdss_spec_urls():
    url, fn = sdss_spec_url(651, 52141, 72, "26")
    assert fn == "spec-0651-52141-0072.fits"
    assert url == "https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/0651/spec-0651-52141-0072.fits"
    url, fn = sdss_spec_url(7169, 56628, 344, "v5_13_0")
    assert url.endswith("eboss/spectro/redux/v5_13_0/spectra/lite/7169/spec-7169-56628-0344.fits")


def test_desi_coadd_urls_and_healpix():
    assert desi_healpix(3.1997083, -8.7834722) == 17260
    url = desi_coadd_url("dr1", "main", "dark", 17260)
    assert url == ("https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix/main/dark/172/17260/"
                   "coadd-main-dark-17260.fits")
    assert desi_coadd_url("edr", "sv3", "bright", 10159).startswith(
        "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/healpix/sv3/bright/101/10159/")
    assert ("main", "dark") in DESI_SURVEY_PROGRAMS["dr1"] and ("main", "dark") not in DESI_SURVEY_PROGRAMS["edr"]
    with pytest.raises(KeyError):
        desi_coadd_url("dr3", "main", "dark", 1)


def test_redrock_reader_and_spectra_rejection():
    rr = os.path.join(EXAMPLES, "redrock-main-dark-17260-39627574082538900.fits")
    d = read_redrock(rr, DESI_TARGETID)
    assert d["z"] == pytest.approx(0.2203154, abs=1e-6) and d["zwarn"] == 0 and d["spectype"] == "QSO"
    assert not np.isfinite(read_redrock(rr, 1)["z"])
    assert is_desi_spectra("spectra-main-dark-17260.fits") and not is_desi_spectra(DESI_EXAMPLE)
    with pytest.raises(ValueError):
        read_spectrum("spectra-main-dark-17260.fits", targetid=1)


def test_mjd_to_date():
    assert mjd_to_date(52141) == "2001-08-20"
    assert mjd_to_date(59524.2) == "2021-11-06"
    assert mjd_to_date(float("nan")) == ""


def test_plot_fit_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    sp = read_spectrum(SDSS_EXAMPLE)
    res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], Z_J001224, complexes=("Hbeta",))
    fig = blrfit.plot_fit(res, title="test")
    assert len(fig.axes) == 3                     # continuum panel, one line panel, its residual panel
    labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert "c(1/2)" in labels and "total" in labels
    top = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "power law + Fe II" in top             # the host is rejected for this spectrum
    fig.savefig(tmp_path / "f.png", dpi=50)
    assert (tmp_path / "f.png").stat().st_size > 1000


def test_mgii_synthetic_offset():
    """A broad Mg II at +900 km/s (FWHM 2500) with a narrow Mg II at zero, at
    z = 1 on the DESI grid with white noise: class A with c(1/2) - v_sys within
    150 km/s of +900 (measured +893) for the single-line and the doublet model."""
    rng = np.random.default_rng(5)
    w = np.arange(3600.0, 9824.01, 0.8); z = 1.0; wr = w / (1 + z)
    cont = 5.0 * (wr / 3000.0) ** -1.5
    lam0 = LAM["MgII"]
    line = (60.0 * np.exp(-0.5 * ((wr - lam0 * (1 + 900 / C_KMS)) / (lam0 * 2500 / 2.3548 / C_KMS)) ** 2)
            + 10.0 * np.exp(-0.5 * ((wr - lam0) / (lam0 * 200 / C_KMS)) ** 2))
    sig = cont / 8.0
    flux = cont + line + rng.standard_normal(w.size) * sig
    for kw in (dict(), dict(mgii_doublet=True)):
        res = blrfit.fit_spectrum(w, flux, 1 / sig**2, z, complexes=("MgII",), **kw)
        m, c = res["meas"]["MgII"], res["cls"]["MgII"]
        assert c["label"] == "A", (kw, c)
        assert abs(m["c50_sys"] - 900.0) < 150.0, (kw, m["c50_sys"])
        assert abs(m["v_sys"]) < 100.0
    row = blrfit.summary_row(res)
    assert row["MG_class"] == "A" and "HA_class" not in row
