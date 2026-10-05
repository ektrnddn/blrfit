"""Identity, radius, input mapping and uncertainty boundaries; no live network."""

import json
import sys
from argparse import Namespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from blrfit.cli import build_parser, _load
from blrfit.io import read_spectrum, read_table, read_sdss
from blrfit.io import public, fetch
from blrfit import input_workflow


def test_missing_fetch_extra_is_actionable_and_local_io_still_works(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", None)
    for query in (lambda: public.query_desi(targetid=1), lambda: fetch.query_sdss(10.0, 5.0)):
        with pytest.raises(ImportError, match=r"\[fetch\]"):
            query()
    path = tmp_path / "local.fits"
    Table(dict(wave=[5000.0, 5001.0], flux=[1.0, 2.0], ivar=[4.0, 9.0])).write(path)
    sp = read_table(path, ivar="ivar", z=0.2)
    assert np.array_equal(sp["flux"], [1.0, 2.0])


def test_default_and_explicit_surveys():
    p = build_parser()
    a = p.parse_args(["fit", "--targetid", "39627574082538901"])
    assert a.targetid == 39627574082538901 and a.survey == "desi"
    assert a.nmc == 0 and a.radius == 1.5 and a.lines == "Halpha,Hbeta"
    assert p.parse_args(["fit", "renamed.fits", "--survey", "sdss"]).survey == "sdss"
    for bad in ("39627574082538901.0", "1;drop", "-1", str(2**63)):
        with pytest.raises(ValueError):
            public.exact_targetid(bad)


def tap(monkeypatch, body):
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs["params"]["QUERY"])
        return Namespace(text=body, raise_for_status=lambda: None)

    monkeypatch.setattr("requests.get", get)
    return calls


HEADER = "targetid,target_ra,target_dec,survey,program,healpix,z,zwarn\n"


def test_exact_id_and_duplicate_products(monkeypatch):
    row = "39627574082538901,10,5,main,dark,12,0.2,0\n"
    calls = tap(monkeypatch, HEADER + row + row + "123,10,5,main,dark,12,0.2,0\n")
    rows = public.query_desi(targetid="39627574082538901")
    assert len(rows) == 1 and rows[0]["targetid"] == 39627574082538901
    assert "s.targetid=39627574082538901" in calls[0]
    assert "photometry" in calls[0]


def test_cone_wrap_and_radius_checked_locally(monkeypatch):
    tap(monkeypatch, HEADER + "1,359.9999,0,main,dark,12,0.2,0\n2,359.999,0,main,dark,12,0.2,0\n")
    rows = public.query_desi(ra=0.0, dec=0.0)
    assert [r["targetid"] for r in rows] == [1]
    assert rows[0]["sep_arcsec"] == pytest.approx(0.36)


@pytest.mark.parametrize("body", ["<VOTABLE>ERROR</VOTABLE>", "unexpected\nvalue\n"])
def test_service_error_is_not_no_match(monkeypatch, body):
    tap(monkeypatch, body)
    with pytest.raises(RuntimeError, match="service"):
        public.query_desi(targetid=1)


def test_empty_catalogue_and_overflow(monkeypatch):
    tap(monkeypatch, HEADER)
    assert public.query_desi(targetid=1) == []
    tap(monkeypatch, HEADER + "1,10,5,main,dark,12,0.2,0\n" * 501)
    with pytest.raises(ValueError, match="too many"):
        public.query_desi(targetid=1)


def test_ambiguous_desi_saved_before_any_download(tmp_path, monkeypatch):
    rows = [dict(targetid=t, ra=10.0, dec=5.0) for t in (1, 2)]
    monkeypatch.setattr(input_workflow, "query_desi", lambda **kw: rows)
    monkeypatch.setattr(
        input_workflow, "fetch_desi_products", lambda *a, **kw: pytest.fail("downloaded ambiguous match")
    )
    a = build_parser().parse_args(["fetch", "--ra", "10", "--dec", "5"])
    with pytest.raises(ValueError, match="multiple DESI"):
        input_workflow.retrieve(a, tmp_path)
    doc = json.loads((tmp_path / "fetch_manifest.json").read_text())
    assert doc["desi_candidates"] == rows and not doc["complete"]


def test_sdss_repeats_and_radius(monkeypatch):
    tap(
        monkeypatch,
        "plate,mjd,fiberid,run2d,ra,dec,z,class\n"
        "1,50000,1,v5_13_2,0,0,.2,QSO\n"
        "1,50001,1,v5_13_2,.0003,0,.2,QSO\n"
        "1,50002,1,v5_13_2,.0005,0,.2,QSO\n"
        "1,50000,1,v5_13_2,0,0,.2,QSO\n",
    )
    found = fetch.query_sdss(0.0, 0.0)
    assert len(found) == 2 and [r["mjd"] for r in found] == [50000, 50001]
    assert all(r["sep_arcsec"] <= 1.5 and "/dr17/" in r["url"] for r in found)
    assert "/v5_13_2/" in found[0]["url"]
    with pytest.raises(ValueError):
        fetch.query_sdss(0.0, 0.0, radius_arcsec=2.0)


def test_vector_rows_not_flattened_and_z_column(tmp_path):
    path = tmp_path / "multi.fits"
    Table(
        dict(
            wave=[[5000.0, 5001.0, 5002.0]] * 2,
            flux=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            err=[[0.1, 0.1, 0.1]] * 2,
            z=[0.2, 0.3],
        )
    ).write(path)
    with pytest.raises(ValueError, match="multiple spectra"):
        read_table(path, err="err")
    sp = read_table(path, err="err", row=1, z_column="z")
    assert np.array_equal(sp["flux"], [4.0, 5.0, 6.0]) and sp["z"] == 0.3
    assert np.allclose(sp["ivar"], 100.0)


def test_image_hdus_units_noise_and_redshift(tmp_path):
    p = tmp_path / "image.fits"
    head = fits.PrimaryHDU()
    head.header["Z"] = 0.2
    fits.HDUList(
        [
            head,
            fits.ImageHDU([500.0, 501.0], name="WAVELENGTH"),
            fits.ImageHDU([1.0, 2.0], name="FLUX"),
            fits.ImageHDU([0.1, 0.2], name="ERROR"),
        ]
    ).writeto(p)
    sp = read_table(
        p,
        wave_hdu="WAVELENGTH",
        flux_hdu="FLUX",
        err_hdu="ERROR",
        z_key="Z",
        wave_unit="nm",
        frame="rest",
        flux_frame="rest",
        flux_scale=10.0,
    )
    assert np.allclose(sp["wave"], [6000.0, 6012.0])
    assert np.allclose(sp["flux"], np.array([10.0, 20.0]) / 1.2)
    assert np.allclose(sp["ivar"], [1.44, 0.36])
    assert sp["z"] == 0.2


def test_generic_requires_noise_and_no_conflicting_z(tmp_path):
    p = tmp_path / "simple.fits"
    Table(dict(wave=[5000.0, 5001.0], flux=[1.0, 2.0], ivar=[4.0, 9.0], mask=[0, 1], z=[0.2, 0.2])).write(p)
    with pytest.raises(ValueError, match="exactly one"):
        read_table(p, z=0.2)
    with pytest.raises(ValueError, match="choose only one"):
        read_table(p, ivar="ivar", z=0.2, z_column="z")
    sp = read_table(p, ivar="ivar", mask="mask", z_column="z")
    assert np.array_equal(sp["ivar"], [4.0, 0.0])


def test_sdss_renamed_file_and_conservative_masks(tmp_path):
    path = tmp_path / "renamed.fits"
    spectrum = Table(
        dict(
            loglam=np.log10([5000.0, 5001.0, 5002.0]),
            flux=[1.0, 2.0, 3.0],
            ivar=[4.0, 5.0, 6.0],
            and_mask=[0, 1, 0],
        )
    )
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(spectrum), fits.BinTableHDU(Table(dict(Z=[0.2])))]
    ).writeto(path)
    sp = read_spectrum(path, survey="sdss", mask_policy="conservative")
    assert np.array_equal(sp["ivar"], [4.0, 0.0, 6.0]) and sp["z"] == 0.2
    assert np.array_equal(read_sdss(path)["ivar"], [4.0, 5.0, 6.0])
    a = build_parser().parse_args(["fit", str(path), "--survey", "sdss"])
    assert np.array_equal(_load(str(path), a)[0]["ivar"], sp["ivar"])
    with pytest.raises(ValueError, match="not a DESI"):
        read_spectrum(path, survey="desi")


def test_public_fit_keeps_every_product_and_failure(tmp_path, monkeypatch):
    records = [dict(kind="sdss", path=f"spec-{i}.fits", url=f"https://example.org/{i}") for i in range(3)]
    monkeypatch.setattr(input_workflow, "retrieve", lambda *a: dict(desi=[], sdss=records, complete=True))
    a = build_parser().parse_args(["fit", "--ra", "10", "--dec", "5", "--out", str(tmp_path)])
    calls = []

    def fit_one(one):
        calls.append(one)
        if one.spectrum == "spec-1.fits":
            raise SystemExit("no usable continuum")

    assert input_workflow.fit_public(a, fit_one) == 1
    assert len(calls) == 3 and len({c.stem for c in calls}) == 3
    doc = json.loads((tmp_path / "fit_manifest.json").read_text())
    assert [r["status"] for r in doc["results"]] == ["returned", "failed", "returned"]
    assert not doc["uncertainty_calibrated"]


def test_all_matches_in_same_coadd_do_not_overwrite(tmp_path, monkeypatch):
    records = [
        dict(kind="desi", path=f"coadd-{t}.fits", coadd_url="https://example.org/coadd.fits", targetid=t)
        for t in (123, 456)
    ]
    monkeypatch.setattr(input_workflow, "retrieve", lambda *a: dict(desi=records, sdss=[], complete=True))
    a = build_parser().parse_args(
        ["fit", "--ra", "10", "--dec", "5", "--all-matches", "--out", str(tmp_path)]
    )
    calls = []
    assert input_workflow.fit_public(a, lambda one: calls.append(one)) == 0
    assert len({c.stem for c in calls}) == 2


def test_public_fits_are_named_after_their_products(tmp_path, monkeypatch):
    desi = dict(
        kind="desi",
        release="dr1",
        survey="main",
        program="dark",
        healpix=17260,
        targetid=39627574082538900,
        path="inputs/desi/coadd.fits",
    )
    sdss = dict(kind="sdss", plate=651, mjd=52141, fiberid=72, path="inputs/sdss/spec.fits")
    # the same identity twice (never expected from the archives) still gives two outputs
    monkeypatch.setattr(
        input_workflow, "retrieve", lambda *a: dict(desi=[desi], sdss=[sdss, dict(sdss)], complete=True)
    )
    a = build_parser().parse_args(["fit", "--ra", "3.2", "--dec", "-8.8", "--out", str(tmp_path)])
    calls = []
    assert input_workflow.fit_public(a, lambda one: calls.append(one)) == 0
    assert [c.stem for c in calls] == [
        "desi-dr1-main-dark-17260-39627574082538900",
        "spec-0651-52141-0072",
        "spec-0651-52141-0072-2",
    ]


def test_single_row_inferred_but_duplicate_desi_rows_rejected(tmp_path):
    from test_fetch_identity import _coadd, TARGETID

    path = tmp_path / "coadd-test.fits"
    _coadd(path, 1.0)
    with pytest.raises(ValueError, match="multi-row"):
        read_spectrum(path, survey="desi")
    with fits.open(path, mode="update") as h:
        h["FIBERMAP"].data["TARGETID"][:] = TARGETID
    with pytest.raises(ValueError, match="multiple spectral rows"):
        read_spectrum(path, survey="desi", targetid=TARGETID)


def test_sdss_download_identity_verified(tmp_path, monkeypatch):
    from conftest import SDSS_EXAMPLE

    row = dict(plate=1, mjd=50000, fiberid=1, filename="wrong.fits", url="https://example.org/wrong", z=0.2)
    monkeypatch.setattr(fetch, "query_sdss", lambda *a, **kw: [row])
    monkeypatch.setattr(fetch, "download", lambda *a, **kw: SDSS_EXAMPLE)
    with pytest.raises(ValueError, match="identity"):
        fetch.fetch_sdss(0.0, 0.0, tmp_path, verbose=False)
