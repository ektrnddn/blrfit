"""Offline regression checks for immutable public DESI retrievals."""
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from blrfit.io import fetch
from blrfit.io.desi import read_desi


TARGETID = 123456789


def _coadd(path, flux, *, exposures=True):
    primary = fits.PrimaryHDU()
    primary.header.update(SURVEY="sv3", PROGRAM="dark", HPXPIXEL=12)
    hdus = [primary, fits.BinTableHDU(Table({
        "TARGETID": [TARGETID, 987], "TARGET_RA": [10.0, 20.0],
        "TARGET_DEC": [5.0, 5.0], "EBV": [0.02, 0.01],
    }), name="FIBERMAP")]
    if exposures:
        hdus.append(fits.BinTableHDU(Table({
            "TARGETID": [TARGETID, TARGETID, 987], "EXPID": [101, 102, 888],
            "NIGHT": [20210101, 20210102, 20210101], "TILEID": [10, 10, 11],
            "MJD": [59215.0, 59216.0, 59215.0],
        }), name="EXP_FIBERMAP"))
    hdus.extend([
        fits.ImageHDU(np.arange(4000., 4100.), name="B_WAVELENGTH"),
        fits.ImageHDU(np.full((2, 100), float(flux)), name="B_FLUX"),
        fits.ImageHDU(np.ones((2, 100)), name="B_IVAR"),
    ])
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _redrock(path, z):
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table({
        "TARGETID": [TARGETID, 987], "Z": [z, 0.1], "ZERR": [0.001, 0.001],
        "ZWARN": [0, 0], "SPECTYPE": ["QSO", "STAR"],
    }), name="REDSHIFTS")]).writeto(path, overwrite=True)


@pytest.fixture
def public_server(tmp_path, monkeypatch):
    server = tmp_path / "server"
    server.mkdir()
    for rel, value in (("dr1", 1), ("edr", 2)):
        _coadd(server / f"{rel}-coadd.fits", value)
        _redrock(server / f"{rel}-redrock.fits", value / 10.)
    monkeypatch.setattr(fetch, "DESI_SURVEY_PROGRAMS", {rel: [("sv3", "dark")] for rel in ("dr1", "edr")})
    monkeypatch.setattr(fetch, "desi_healpix", lambda ra, dec: 12)
    monkeypatch.setattr(fetch, "_url_exists", lambda url: True)

    def source(url, kind):
        rel = "dr1" if "/dr1/" in url else "edr"
        return server / f"{rel}-{kind}.fits"

    monkeypatch.setattr(fetch, "_open_remote", lambda url: fits.open(source(url, "coadd"), memmap=False))

    def download(url, dest, **kwargs):
        src = source(url, "redrock")
        if not src.exists():
            return None
        shutil.copyfile(src, dest)
        return str(dest)

    monkeypatch.setattr(fetch, "download", download)
    return server, tmp_path / "output"


def _fetch(output, releases=("dr1", "edr")):
    return fetch.fetch_desi(10., 5., output, targetid=TARGETID, releases=releases, verbose=False)


def _checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_releases_are_distinct_but_shared_exposures_are_identified(public_server):
    _, output = public_server
    rows = _fetch(output)
    assert len({row["path"] for row in rows}) == 2
    assert rows[0]["exposure_fingerprint"] == rows[1]["exposure_fingerprint"]
    assert rows[0]["exposure_fingerprint"] is not None
    for row, expected_flux, expected_z in zip(rows, (1, 2), (.1, .2)):
        assert f"/{row['release']}/{row['specprod']}/" in row["path"]
        spectrum = read_desi(row["path"], TARGETID, use_desispec=False)
        np.testing.assert_array_equal(spectrum["flux"], expected_flux)
        assert spectrum["z"] == expected_z
        assert row["exposure_ids"] == [101, 102]
        assert len(row["exposures"]) == 2
        manifest = json.loads(Path(row["provenance_path"]).read_text())
        assert manifest["targetid"] == TARGETID
        assert f"/{row['release']}/" in manifest["coadd_url"]
        for name, sha in manifest["files"].items():
            assert _checksum(Path(row["path"]).parent / name) == sha

    # Re-fetching identical server content returns the same immutable files.
    before = {row["path"]: Path(row["path"]).stat().st_mtime_ns for row in rows}
    again = _fetch(output)
    assert [row["path"] for row in again] == [row["path"] for row in rows]
    assert {path: Path(path).stat().st_mtime_ns for path in before} == before
    assert not list(output.rglob(".staging-*"))
    assert not list(output.rglob(".redrock-*"))


def test_new_content_without_redrock_never_reuses_stale_sibling(public_server):
    server, output = public_server
    original = _fetch(output, ("dr1",))[0]
    original_sha = _checksum(original["path"])
    rr_sha = _checksum(original["redrock"])
    _coadd(server / "dr1-coadd.fits", 7)
    (server / "dr1-redrock.fits").unlink()
    current = _fetch(output, ("dr1",))[0]
    assert current["path"] != original["path"]
    assert current["redrock"] is None
    assert not current["redrock_available"]
    spectrum = read_desi(current["path"], TARGETID, use_desispec=False)
    np.testing.assert_array_equal(spectrum["flux"], 7)
    assert np.isnan(spectrum["z"])
    assert not list(Path(current["path"]).parent.glob("redrock-*"))
    assert _checksum(original["path"]) == original_sha
    assert _checksum(original["redrock"]) == rr_sha


def test_failed_extraction_preserves_previous_bundle_and_publishes_nothing(public_server, monkeypatch):
    server, output = public_server
    original = _fetch(output, ("dr1",))[0]
    before = {p: _checksum(p) for p in output.rglob("*") if p.is_file()}
    _coadd(server / "dr1-coadd.fits", 8)

    def fail_after_coadd(h, tid, out_path, **kwargs):
        Path(out_path).write_bytes(b"incomplete fits")
        raise OSError("simulated failed redrock extraction")

    monkeypatch.setattr(fetch, "write_single_target", fail_after_coadd)
    with pytest.raises(OSError, match="simulated"):
        _fetch(output, ("dr1",))
    assert {p: _checksum(p) for p in output.rglob("*") if p.is_file()} == before
    assert read_desi(original["path"], TARGETID, use_desispec=False)["z"] == .1
    assert not list(output.rglob(".staging-*"))
    assert not list(output.rglob(".redrock-*"))


def test_corrupted_saved_bundle_is_rejected(public_server):
    _, output = public_server
    original = _fetch(output, ("dr1",))[0]
    Path(original["redrock"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="inconsistent"):
        _fetch(output, ("dr1",))
    # No automatic overwrite of a manifest's supposedly immutable input.
    assert Path(original["redrock"]).read_bytes() == b"corrupt"


def test_redrock_without_matching_target_is_not_published(public_server):
    server, output = public_server
    with fits.open(server / "dr1-redrock.fits", mode="update") as rr:
        rr["REDSHIFTS"].data["TARGETID"][:] = [111, 222]
    with pytest.raises(ValueError, match="no unique redshift row"):
        _fetch(output, ("dr1",))
    assert not [p for p in output.rglob("*") if p.is_file()]
    assert not list(output.rglob(".staging-*"))


def test_failed_publish_preserves_existing_bundles(public_server, monkeypatch):
    server, output = public_server
    _fetch(output, ("dr1",))
    before = {p: _checksum(p) for p in output.rglob("*") if p.is_file()}
    _coadd(server / "dr1-coadd.fits", 9)

    def fail_publish(source, dest):
        raise PermissionError("simulated failed publish")

    monkeypatch.setattr(fetch.os, "rename", fail_publish)
    with pytest.raises(PermissionError, match="failed publish"):
        _fetch(output, ("dr1",))
    assert {p: _checksum(p) for p in output.rglob("*") if p.is_file()} == before
    assert not list(output.rglob(".staging-*"))


def test_missing_exposure_information_does_not_claim_unique_epoch(public_server):
    server, output = public_server
    _coadd(server / "dr1-coadd.fits", 1, exposures=False)
    row = _fetch(output, ("dr1",))[0]
    assert row["exposure_ids"] == []
    assert row["exposure_fingerprint"] is None


@pytest.mark.parametrize("body,fail", [(b"", False), (b"partial", True)])
def test_download_failure_preserves_destination_and_cleans_temporary_files(tmp_path, monkeypatch, body, fail):
    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_content(self, size):
            yield body
            if fail:
                raise OSError("connection interrupted")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=lambda *args, **kwargs: Response()))
    dest = tmp_path / "spectrum.fits"
    dest.write_bytes(b"previous complete data")
    assert fetch.download("https://example.invalid/spectrum", dest, clobber=True) is None
    assert dest.read_bytes() == b"previous complete data"
    assert list(tmp_path.iterdir()) == [dest]


# ----------------------------------------------------------------------------
# tile epochs of a public release
# ----------------------------------------------------------------------------
# Two tiles of the target in the dark healpix coadd (tile 10 over two nights,
# tile 12 absent from the tiles table) and one in the bright coadd (tile 11);
# another target shares tile 10 and sits alone on tile 99.
TILES = {
    10: dict(lastnight=20210102, petal=3, program="dark", expids=[101, 102], nights=[20210101, 20210102],
             mjds=[59215.1, 59216.1], flux=3.0, z=0.31),
    11: dict(lastnight=20210301, petal=7, program="bright", expids=[201], nights=[20210301],
             mjds=[59274.2], flux=5.0, z=0.32),
    12: dict(lastnight=None, petal=0, program="dark", expids=[301], nights=[20210401],
             mjds=[59305.3], flux=7.0, z=0.33),
}
OTHER = 987


def _healpix_coadd(path, program):
    primary = fits.PrimaryHDU()
    primary.header.update(SURVEY="main", PROGRAM=program, HPXPIXEL=12)
    tiles = [t for t, v in TILES.items() if v["program"] == program]
    exp = dict(TARGETID=[], EXPID=[], NIGHT=[], TILEID=[], MJD=[], PETAL_LOC=[], FIBER=[])
    for t in tiles:
        v = TILES[t]
        for e, n, m in zip(v["expids"], v["nights"], v["mjds"]):
            exp["TARGETID"].append(TARGETID); exp["EXPID"].append(e); exp["NIGHT"].append(n)
            exp["TILEID"].append(t); exp["MJD"].append(m); exp["PETAL_LOC"].append(v["petal"])
            exp["FIBER"].append(500 * v["petal"] + 17)
    for t, e, n in ((10, 101, 20210101), (99, 901, 20210505)):
        exp["TARGETID"].append(OTHER); exp["EXPID"].append(e); exp["NIGHT"].append(n)
        exp["TILEID"].append(t); exp["MJD"].append(59300.); exp["PETAL_LOC"].append(1); exp["FIBER"].append(501)
    expids_all = [e for t in tiles for e in TILES[t]["expids"]]
    hdus = [primary,
            fits.BinTableHDU(Table({"TARGETID": [TARGETID, OTHER], "TARGET_RA": [10.0, 20.0],
                                    "TARGET_DEC": [5.0, 5.0], "EBV": [0.02, 0.01]}), name="FIBERMAP"),
            fits.BinTableHDU(Table(exp), name="EXP_FIBERMAP"),
            fits.ImageHDU(np.arange(4000., 4100.), name="B_WAVELENGTH"),
            fits.ImageHDU(np.full((2, 100), float(len(expids_all))), name="B_FLUX"),
            fits.ImageHDU(np.ones((2, 100)), name="B_IVAR")]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _tile_coadd(path, tileid, with_target=True):
    v = TILES[tileid]
    night = v["lastnight"] or max(v["nights"])
    primary = fits.PrimaryHDU()
    primary.header.update(SPGRP="cumulative", SPGRPVAL=night, NIGHT=night, TILEID=tileid, PETAL=v["petal"])
    tids = [OTHER, TARGETID] if with_target else [OTHER, OTHER + 1]
    exp = dict(TARGETID=[OTHER], EXPID=[v["expids"][0]], NIGHT=[v["nights"][0]], TILEID=[tileid],
               MJD=[v["mjds"][0]], PETAL_LOC=[v["petal"]])
    if with_target:
        for e, n, m in zip(v["expids"], v["nights"], v["mjds"]):
            exp["TARGETID"].append(TARGETID); exp["EXPID"].append(e); exp["NIGHT"].append(n)
            exp["TILEID"].append(tileid); exp["MJD"].append(m); exp["PETAL_LOC"].append(v["petal"])
    hdus = [primary,
            fits.BinTableHDU(Table({"TARGETID": tids, "PETAL_LOC": [v["petal"]] * 2, "TILEID": [tileid] * 2,
                                    "TARGET_RA": [20.0, 10.0], "TARGET_DEC": [5.0, 5.0], "EBV": [0.01, 0.02],
                                    "COADD_NUMEXP": [1, len(v["expids"])]}), name="FIBERMAP"),
            fits.BinTableHDU(Table(exp), name="EXP_FIBERMAP"),
            fits.ImageHDU(np.arange(4000., 4100.), name="B_WAVELENGTH"),
            fits.ImageHDU(np.vstack([np.full(100, -1.0), np.full(100, v["flux"])]), name="B_FLUX"),
            fits.ImageHDU(np.ones((2, 100)), name="B_IVAR"),
            fits.ImageHDU(np.zeros((2, 100), dtype=np.int32), name="B_MASK")]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _tile_redrock(path, tileid):
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(Table({
        "TARGETID": [OTHER, TARGETID], "Z": [0.5, TILES[tileid]["z"]], "ZERR": [0.001, 0.001],
        "ZWARN": [0, 0], "SPECTYPE": ["GALAXY", "QSO"],
    }), name="REDSHIFTS")]).writeto(path, overwrite=True)


def _tiles_csv(path):
    lines = ["TILEID,SURVEY,PROGRAM,FAPRGRM,NEXP,TILERA,TILEDEC,LASTNIGHT"]
    for t in (10, 11, 99):
        v = TILES.get(t, dict(program="bright", lastnight=20210505, expids=[0]))
        lines.append(f"{t},main,{v['program']},{v['program']},{len(v['expids'])},10.0,5.0,{v['lastnight']}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def epoch_server(tmp_path, monkeypatch):
    server = tmp_path / "server"
    server.mkdir()
    files = {}
    for program in ("dark", "bright"):
        _healpix_coadd(server / f"hp-{program}.fits", program)
        files[fetch.desi_coadd_url("dr1", "main", program, 12)] = server / f"hp-{program}.fits"
    for t in (10, 11):
        _tile_coadd(server / f"tile-{t}.fits", t)
        _tile_redrock(server / f"rr-{t}.fits", t)
        v = TILES[t]
        files[fetch.desi_tile_coadd_url("dr1", t, v["lastnight"], v["petal"])] = server / f"tile-{t}.fits"
        files[fetch.desi_tile_redrock_url("dr1", t, v["lastnight"], v["petal"])] = server / f"rr-{t}.fits"
    _tiles_csv(server / "tiles-iron.csv")
    files[fetch.desi_tiles_csv_url("dr1")] = server / "tiles-iron.csv"
    monkeypatch.setattr(fetch, "DESI_SURVEY_PROGRAMS", {"dr1": [("main", "dark"), ("main", "bright")]})
    monkeypatch.setattr(fetch, "desi_healpix", lambda ra, dec: 12)
    monkeypatch.setattr(fetch, "_url_exists", lambda url: url in files)
    monkeypatch.setattr(fetch, "_open_remote", lambda url, block_size=None: fits.open(files[url], memmap=False))

    def download(url, dest, **kwargs):
        if url not in files:
            return None
        shutil.copyfile(files[url], dest)
        return str(dest)

    monkeypatch.setattr(fetch, "download", download)
    return files, server / "tiles-iron.csv", tmp_path / "output"


def test_tile_urls():
    base = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron"
    assert fetch.desi_tile_coadd_url("dr1", 8915, 20220228, 7) == \
        f"{base}/tiles/cumulative/8915/20220228/coadd-7-8915-thru20220228.fits"
    assert fetch.desi_tile_redrock_url("dr1", "8915", "20220228", "7") == \
        f"{base}/tiles/cumulative/8915/20220228/redrock-7-8915-thru20220228.fits"
    assert fetch.desi_tiles_csv_url("dr1") == f"{base}/tiles-iron.csv"
    assert fetch.desi_tiles_csv_url("edr") == "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/tiles-fuji.csv"
    assert fetch._redrock_url(fetch.desi_coadd_url("dr1", "main", "dark", 17260)).endswith(
        "/healpix/main/dark/172/17260/redrock-main-dark-17260.fits")


def test_tiles_table_reader(epoch_server):
    _, csv_path, _ = epoch_server
    tiles = fetch.read_desi_tiles_table(csv_path)
    assert sorted(tiles) == [10, 11, 99]
    assert tiles[10] == dict(tileid=10, survey="main", program="dark", lastnight=20210102,
                             tilera=10.0, tiledec=5.0, nexp=2)


def test_list_desi_epochs_groups_exposures_by_tile(epoch_server):
    _, csv_path, output = epoch_server
    epochs = fetch.list_desi_epochs(10., 5., TARGETID, tiles_csv=csv_path, verbose=False)
    assert [e["tileid"] for e in epochs] == [10, 11, 12]          # by first exposure, the other target's tile 99 absent
    first = epochs[0]
    assert first["nights"] == [20210101, 20210102] and first["expids"] == [101, 102]
    assert first["petal"] == 3 and first["lastnight"] == 20210102 and first["program"] == "dark"
    assert first["epoch_key"] == "10-20210102-3" and first["in_tiles_table"]
    assert first["coadd_url"] == fetch.desi_tile_coadd_url("dr1", 10, 20210102, 3)
    assert first["redrock_url"] == fetch.desi_tile_redrock_url("dr1", 10, 20210102, 3)
    assert first["mjds"] == [59215.1, 59216.1] and first["healpix"] == 12 and first["survey"] == "main"
    assert epochs[1]["program"] == "bright" and epochs[1]["petal"] == 7 and epochs[1]["expids"] == [201]
    missing = epochs[2]
    assert not missing["in_tiles_table"]
    assert missing["lastnight"] is None and missing["epoch_key"] is None and missing["coadd_url"] is None
    assert missing["nights"] == [20210401] and missing["expids"] == [301]
    # the tiles table is downloaded once into out_dir when not given
    again = fetch.list_desi_epochs(10., 5., TARGETID, out_dir=output, verbose=False)
    assert again == epochs
    assert (output / "tiles-iron.csv").read_text() == csv_path.read_text()
    with pytest.raises(ValueError, match="tiles_csv"):
        fetch.list_desi_epochs(10., 5., TARGETID, verbose=False)
    assert fetch.list_desi_epochs(10., 5., 4242, tiles_csv=csv_path, verbose=False) == []


def test_fetch_desi_epochs_publishes_one_bundle_per_tile(epoch_server):
    _, csv_path, output = epoch_server
    rows = fetch.fetch_desi_epochs(10., 5., TARGETID, output, tiles_csv=csv_path, verbose=False)
    assert [r["status"] for r in rows] == ["ok", "ok", "no_lastnight"]
    assert rows[2]["path"] is None
    ok = rows[:2]
    assert len({r["path"] for r in ok}) == 2
    assert len({r["exposure_fingerprint"] for r in ok}) == 2 and None not in {r["exposure_fingerprint"] for r in ok}
    for r in ok:
        v = TILES[r["tileid"]]
        stem = f"{v['petal']}-{r['tileid']}-thru{v['lastnight']}-{TARGETID}"
        assert Path(r["path"]) == output / "desi" / "dr1" / "iron" / "tiles" / stem / r["bundle_id"] / f"coadd-{stem}.fits"
        assert Path(r["redrock"]).name == f"redrock-{stem}.fits"
        assert r["exposure_ids"] == v["expids"] and r["nights"] == v["nights"] and r["expids_match"] is True
        assert r["epoch_key"] == f"{r['tileid']}-{v['lastnight']}-{v['petal']}"
        spectrum = read_desi(r["path"], TARGETID, use_desispec=False)
        np.testing.assert_array_equal(spectrum["flux"], v["flux"])
        assert spectrum["z"] == v["z"] and spectrum["tileid"] == r["tileid"]
        assert spectrum["nexp"] == len(v["expids"]) and abs(spectrum["mjd"] - np.mean(v["mjds"])) < 1e-6
        manifest = json.loads(Path(r["provenance_path"]).read_text())
        for key in ("tileid", "lastnight", "petal", "nights", "exposure_ids", "survey", "program", "healpix"):
            assert manifest[key] == r[key]
        assert manifest["group"] == "cumulative" and manifest["targetid"] == TARGETID
        assert manifest["coadd_url"] == fetch.desi_tile_coadd_url("dr1", r["tileid"], v["lastnight"], v["petal"])
        assert manifest["redrock_available"]
        assert set(manifest["files"]) == {f"coadd-{stem}.fits", f"redrock-{stem}.fits"}
        for name, sha in manifest["files"].items():
            assert _checksum(Path(r["path"]).parent / name) == sha
        assert r["bundle_id"] == hashlib.sha256(
            Path(r["provenance_path"]).read_text().rstrip("\n").encode()).hexdigest()
    # The healpix coadds share the identity scheme. The dark one stacks tiles 10
    # and 12, so its exposure set differs from every tile epoch; the bright one
    # holds tile 11 alone and its fingerprint says so: it is the same epoch.
    healpix = fetch.fetch_desi(10., 5., output, targetid=TARGETID, releases=("dr1",), verbose=False)
    by_program = {r["program"]: r for r in healpix}
    assert by_program["dark"]["exposure_ids"] == [101, 102, 301]
    assert by_program["dark"]["exposure_fingerprint"] not in {r["exposure_fingerprint"] for r in ok}
    assert by_program["bright"]["exposure_fingerprint"] == ok[1]["exposure_fingerprint"]
    assert set(json.loads(Path(by_program["dark"]["provenance_path"]).read_text())) < set(
        json.loads(Path(ok[0]["provenance_path"]).read_text()))
    # repeated retrieval of identical content reuses the immutable bundles
    again = fetch.fetch_desi_epochs(10., 5., TARGETID, output, tiles_csv=csv_path, verbose=False)
    assert [r["path"] for r in again] == [r["path"] for r in rows]
    assert not list(output.rglob(".staging-*")) and not list(output.rglob(".redrock-*"))


def test_fetch_desi_epochs_reports_missing_and_foreign_tile_files(epoch_server, monkeypatch):
    files, csv_path, output = epoch_server
    epochs = fetch.list_desi_epochs(10., 5., TARGETID, tiles_csv=csv_path, verbose=False)
    del files[fetch.desi_tile_coadd_url("dr1", 11, 20210301, 7)]
    _tile_coadd(files[fetch.desi_tile_coadd_url("dr1", 10, 20210102, 3)], 10, with_target=False)
    monkeypatch.setattr(fetch, "list_desi_epochs", lambda *args, **kwargs: pytest.fail("epochs were given"))
    rows = fetch.fetch_desi_epochs(10., 5., TARGETID, output, epochs=epochs, verbose=False)
    assert [r["status"] for r in rows] == ["not_in_coadd", "missing", "no_lastnight"]
    assert all(r["path"] is None for r in rows)
    assert [r["tileid"] for r in rows] == [10, 11, 12]
    assert not [p for p in output.rglob("*") if p.is_file()]
