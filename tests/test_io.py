"""Readers: SDSS spec files, DESI coadds (astropy and desispec paths), generic tables."""
import os

import numpy as np
import pytest

from blrfit.io import (read_sdss, read_desi, read_table, read_spectrum, write_table, air_to_vacuum,
                       vacuum_to_air, is_sdss_spec, is_desi_coadd, coadd_cameras)
from blrfit.io.desi import write_single_target
from conftest import SDSS_EXAMPLE, SDSS_EXAMPLE_2, DESI_EXAMPLE, DESI_TARGETID, CSV_EXAMPLE, Z_J001224


def test_read_sdss_dr7_file():
    d = read_sdss(SDSS_EXAMPLE)
    assert d["kind"] == "sdss"
    assert d["plate"] == 651 and d["fiber"] == 72 and d["mjd"] == 52141
    assert d["date"] == "2001-08-20"
    assert d["z"] == pytest.approx(0.22438775, abs=1e-6)
    assert len(d["wave"]) == 3841 and np.all(np.diff(d["wave"]) > 0)
    assert d["wave"][0] == pytest.approx(3805.397, abs=0.01)
    assert (d["ivar"] > 0).sum() > 3500
    assert abs(d["ra"] - 3.1001) < 1e-3 and abs(d["dec"] + 10.374) < 1e-3


def test_read_sdss_boss_file_uppercase_columns():
    d = read_sdss(SDSS_EXAMPLE_2)
    assert d["plate"] == 7169 and d["fiber"] == 344 and d["mjd"] == 56628
    assert d["z"] == pytest.approx(0.2274392, abs=1e-6)
    assert len(d["wave"]) == 4626


def test_read_desi_fixture_astropy():
    d = read_desi(DESI_EXAMPLE, DESI_TARGETID, use_desispec=False)
    assert d["kind"] == "desi" and d["targetid"] == DESI_TARGETID
    assert len(d["wave"]) == 7781
    assert d["wave"][0] == 3600.0 and d["wave"][-1] == pytest.approx(9824.0, abs=1e-6)
    assert np.allclose(np.diff(d["wave"]), 0.8)
    assert d["z"] == pytest.approx(0.2203154, abs=1e-6) and d["zwarn"] == 0 and d["spectype"] == "QSO"
    assert d["ebv"] == pytest.approx(0.03225, abs=1e-4)
    assert d["survey"] == "main" and d["program"] == "dark" and d["healpix"] == 17260
    assert d["mjd"] == pytest.approx(59524.20, abs=0.01)
    assert abs(d["ra"] - 3.19971) < 1e-4 and abs(d["dec"] + 8.78349) < 1e-4
    assert (d["ivar"] > 0).sum() > 7700 and d["mask"].dtype.kind == "i"


def test_read_desi_desispec_path_identical():
    pytest.importorskip("desispec")
    a = read_desi(DESI_EXAMPLE, DESI_TARGETID, use_desispec=False)
    b = read_desi(DESI_EXAMPLE, DESI_TARGETID, use_desispec=True)
    for k in ("wave", "flux", "ivar", "mask"):
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_read_desi_missing_target():
    with pytest.raises(ValueError):
        read_desi(DESI_EXAMPLE, 12345, use_desispec=False)


def test_coadd_cameras_overlap_rule():
    """Non-overlap pixels are copied; overlap pixels are the ivar-weighted mean; masks OR."""
    w1 = np.arange(0.0, 10.0, 1.0); w2 = np.arange(8.0, 18.0, 1.0); w3 = np.arange(16.0, 24.0, 1.0)
    f1 = np.full(10, 1.0); f2 = np.full(10, 3.0); f3 = np.full(8, 5.0)
    i1 = np.full(10, 1.0); i2 = np.full(10, 3.0); i3 = np.full(8, 1.0)
    m1 = np.zeros(10, int); m2 = np.zeros(10, int); m3 = np.zeros(8, int)
    m1[8] = 4; m2[0] = 0; i2[1] = 0.0; f2[1] = 99.0   # a masked pixel and a dead pixel in the overlap
    wave, flux, ivar, mask = coadd_cameras([w1, w2, w3], [f1, f2, f3], [i1, i2, i3], [m1, m2, m3])
    assert np.array_equal(wave, np.arange(0.0, 24.0, 1.0))
    assert flux[0] == 1.0 and flux[12] == 3.0 and flux[20] == 5.0
    assert flux[8] == pytest.approx((1.0 * 1.0 + 3.0 * 3.0) / 4.0) and ivar[8] == 4.0
    assert flux[9] == pytest.approx(1.0) and ivar[9] == 1.0          # dead pixel in camera 2 ignored
    assert flux[16] == pytest.approx((3.0 * 3.0 + 5.0 * 1.0) / 4.0)
    assert mask[8] == 0                                              # ivar > 0 clears the mask
    assert np.all(ivar >= 0)


def test_write_single_target_roundtrip(tmp_path):
    out = tmp_path / "coadd-copy.fits"
    write_single_target(DESI_EXAMPLE, DESI_TARGETID, str(out))
    a = read_desi(DESI_EXAMPLE, DESI_TARGETID, use_desispec=False)
    b = read_desi(str(out), DESI_TARGETID, use_desispec=False)
    for k in ("wave", "flux", "ivar", "mask"):
        assert np.array_equal(a[k], b[k])


def test_dispatch_by_name():
    assert is_sdss_spec(SDSS_EXAMPLE) and not is_desi_coadd(SDSS_EXAMPLE)
    assert is_desi_coadd(DESI_EXAMPLE) and not is_sdss_spec(DESI_EXAMPLE)
    assert read_spectrum(SDSS_EXAMPLE)["kind"] == "sdss"
    assert read_spectrum(DESI_EXAMPLE, targetid=DESI_TARGETID)["kind"] == "desi"
    with pytest.raises(ValueError):
        read_spectrum(DESI_EXAMPLE)


def test_air_vacuum_conversion():
    assert air_to_vacuum(6562.79) == pytest.approx(6564.61, abs=0.02)
    assert air_to_vacuum(4861.33) == pytest.approx(4862.68, abs=0.02)
    w = np.linspace(3000.0, 10000.0, 500)
    assert np.allclose(vacuum_to_air(air_to_vacuum(w)), w, atol=1e-6)


def test_read_table_conversions(tmp_path):
    """A rest-frame, air, nm table with 1-sigma errors comes back as the observed vacuum Angstrom spectrum.
    The air-to-vacuum conversion applies to the wavelengths as given (here rest frame), before the frame shift."""
    sp = read_sdss(SDSS_EXAMPLE); z = Z_J001224
    good = sp["ivar"] > 0
    err = np.where(good, 1.0 / np.sqrt(np.where(good, sp["ivar"], 1.0)), np.nan)
    p = tmp_path / "s.csv"
    write_table(str(p), vacuum_to_air(sp["wave"] / (1 + z)) / 10.0, sp["flux"] * 0.1, err=err * 0.1, names=("lam", "f", "e"))
    d = read_table(str(p), wave="lam", flux="f", err="e", wave_unit="nm", frame="rest", air=True, z=z, flux_scale=10.0)
    assert d["kind"] == "table" and d["z"] == z
    assert np.allclose(d["wave"], sp["wave"], rtol=1e-8)
    assert np.allclose(d["flux"], sp["flux"], rtol=1e-6)
    assert np.allclose(d["ivar"][good], sp["ivar"][good], rtol=1e-6)
    assert np.all(d["ivar"][~good] == 0)


def test_read_table_needs_one_error_column(tmp_path):
    p = tmp_path / "s.csv"; p.write_text("wave,flux,err\n5000,1,0.1\n5001,1,0.1\n")
    with pytest.raises(ValueError):
        read_table(str(p), wave="wave", flux="flux")
    with pytest.raises(ValueError):
        read_table(str(p), wave="wave", flux="flux", err="err", ivar="err")
    with pytest.raises(ValueError):
        read_table(str(p), wave="wave", flux="flux", err="err", frame="rest")
    with pytest.raises(KeyError):
        read_table(str(p), wave="lambda", flux="flux", err="err")


def test_read_table_fits_and_ecsv(tmp_path):
    from astropy.table import Table
    t = Table([np.linspace(4000, 9000, 50), np.ones(50), np.full(50, 0.05)], names=("WAVE", "FLUX", "IVAR"))
    t.write(tmp_path / "s.fits", overwrite=True); t.write(tmp_path / "s.ecsv", format="ascii.ecsv", overwrite=True)
    for fn in ("s.fits", "s.ecsv"):
        d = read_table(str(tmp_path / fn), wave="wave", flux="flux", ivar="ivar")
        assert len(d["wave"]) == 50 and np.all(d["ivar"] == 0.05)


def test_csv_example_exists():
    assert os.path.exists(CSV_EXAMPLE)
    d = read_table(CSV_EXAMPLE, wave="lambda_nm", flux="f_lambda", err="sigma", wave_unit="nm", frame="rest", air=True,
                   z=Z_J001224, flux_scale=10.0)
    sp = read_sdss(SDSS_EXAMPLE)
    assert np.allclose(d["wave"], sp["wave"], rtol=1e-8)
