"""
Public spectra of a sky position: SDSS through astroquery and the science
archive server, DESI public releases through the DESI file server
(data.desi.lbl.gov/public).

Nothing here runs unless asked for (the ``blrfit fetch`` subcommand or a
direct call); the fitting code never touches the network.

SDSS: every spectroscopic visit within the search radius is listed with
``astroquery.sdss.SDSS.query_region`` (data release 16) and the
``spec-PLATE-MJD-FIBER.fits`` files are downloaded from the DR16 science archive
server. Repeat visits are kept: they are epochs.

DESI: the position gives the nside = 64 nested healpix; for each public release
(DR1 = ``iron``, EDR = ``fuji``) and each survey/program combination, the
healpix coadd is looked up on the file server. Files that exist are opened
remotely with range requests (``astropy`` + ``fsspec``), so only the FIBERMAP
and the rows of the matched target are transferred, not the whole 100-800 MB
file. The matched target is written as a single-target coadd, with the
redrock redshift file reduced alongside, in the layout ``read_desi`` expects.
A TARGETID alone does not determine the healpix, so RA and Dec are always
required; a TARGETID, when given, selects the row exactly instead of by
position.

DESI tile epochs: a target observed on several tiles has one spectrum per
tile, and the release keeps each as a tile-cumulative coadd (all exposures of
the tile through its last night, one file per petal of about 500 targets).
``list_desi_epochs`` reads the exposure table of the healpix coadds of the
position, groups the exposures of the target by tile and looks the last night
of each tile up in the release's tiles table; ``fetch_desi_epochs`` then
extracts the target from each tile coadd by range requests into bundles with
the same layout, checksums and provenance as the healpix bundles. The healpix
coadd of a survey/program stacks all of those tiles, so it is not an epoch in
itself. DR1 (``iron``) holds the observations up to 2022 June 13.
"""
from __future__ import annotations

import os
import hashlib
import json
import time
import tempfile
from pathlib import Path

import numpy as np

from .healpix import ang2pix_nest
from .desi import write_single_target

SDSS_SAS = "https://data.sdss.org/sas/dr16"
DESI_PUBLIC = "https://data.desi.lbl.gov/public"
DESI_RELEASES = {"dr1": "iron", "edr": "fuji"}
DESI_SURVEY_PROGRAMS = {
    "dr1": [("main", "dark"), ("main", "bright"), ("main", "backup"),
            ("sv3", "dark"), ("sv3", "bright"), ("sv3", "backup"),
            ("sv1", "dark"), ("sv1", "bright"), ("sv1", "backup"), ("sv1", "other"),
            ("sv2", "dark"), ("sv2", "bright"), ("sv2", "backup"),
            ("special", "dark"), ("special", "bright"), ("special", "backup"), ("cmx", "other")],
    "edr": [("sv3", "dark"), ("sv3", "bright"), ("sv3", "backup"),
            ("sv1", "dark"), ("sv1", "bright"), ("sv1", "backup"), ("sv1", "other"),
            ("sv2", "dark"), ("sv2", "bright"), ("sv2", "backup"),
            ("special", "dark"), ("special", "bright"), ("special", "backup"), ("cmx", "other")],
}
DESI_NSIDE = 64
DESI_TILE_GROUP = "cumulative"    # the tile coadds that stack every exposure of a tile through its last night
# Block size of the range requests. The rows of one target are 11 KB in a
# flux extension and 121 KB in a resolution extension, so 64 KB blocks
# transfer 3 MB per coadd against 29 MB with 1 MB blocks, at the same wall
# time (3.5 s, measured on a 223 MB DR1 tile coadd); the tables are read whole
# in one request either way, and the extracted files are identical.
DESI_REMOTE_BLOCK_SIZE = 1 << 16


# ----------------------------------------------------------------------------
# SDSS
# ----------------------------------------------------------------------------
def sdss_spec_url(plate, mjd, fiber, run2d):
    """URL and file name of a spec file on the DR16 science archive server."""
    p4, m5, f4 = f"{int(plate):04d}", f"{int(mjd):05d}", f"{int(fiber):04d}"
    fn = f"spec-{p4}-{m5}-{f4}.fits"
    r = str(run2d).strip()
    if r in {"26", "103", "104"}:
        rel = f"sdss/spectro/redux/{r}/spectra/{p4}/{fn}"
    else:
        rel = f"eboss/spectro/redux/v5_13_0/spectra/lite/{p4}/{fn}"
    return f"{SDSS_SAS}/{rel}", fn


def query_sdss(ra, dec, radius_arcsec=2.0, data_release=16):
    """List of dicts (plate, mjd, fiberid, run2d, ra, dec, z, cls, url, filename)
    for every SDSS spectrum within the radius."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.sdss import SDSS
    res = SDSS.query_region(SkyCoord(float(ra) * u.deg, float(dec) * u.deg),
                            radius=radius_arcsec * u.arcsec, spectro=True, data_release=data_release)
    rows = []
    if res is None or len(res) == 0:
        return rows
    low = {c.lower(): c for c in res.colnames}

    def g(sp, name, default=np.nan):
        c = low.get(name.lower())
        if c is None:
            return default
        v = sp[c]
        try:
            return default if np.ma.is_masked(v) else v
        except TypeError:
            return v

    seen = set()
    for sp in res:
        try:
            key = (int(g(sp, "plate")), int(g(sp, "mjd")), int(g(sp, "fiberID")))
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        run2d = str(g(sp, "run2d", "v5_13_0"))
        url, fn = sdss_spec_url(*key, run2d=run2d)
        rows.append(dict(plate=key[0], mjd=key[1], fiberid=key[2], run2d=run2d,
                         ra=float(g(sp, "ra", np.nan)), dec=float(g(sp, "dec", np.nan)),
                         z=float(g(sp, "z", np.nan)), cls=str(g(sp, "class", "")), url=url, filename=fn))
    return rows


def download(url, dest, clobber=False, timeout=120):
    """Download ``url`` to ``dest`` through a temporary file; a partial file never
    gets the final name. Returns the path, or None on failure."""
    import requests
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not clobber:
        return str(dest)
    tmp = None
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None
            # A unique temporary file prevents simultaneous downloads from
            # truncating or publishing each other's partial response.
            with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f".{dest.name}.",
                                             suffix=".part", delete=False) as fh:
                tmp = Path(fh.name)
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        if tmp.stat().st_size == 0:
            return None
        os.replace(tmp, dest)
        return str(dest)
    except Exception:
        return None
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def fetch_sdss(ra, dec, out_dir, radius_arcsec=2.0, data_release=16, verbose=True):
    """Download every SDSS spectrum within the radius into ``out_dir``.
    Returns the list of ``query_sdss`` rows with a ``path`` entry for each file obtained."""
    rows = query_sdss(ra, dec, radius_arcsec=radius_arcsec, data_release=data_release)
    if verbose:
        print(f"SDSS: {len(rows)} spectrum(s) within {radius_arcsec}\" of ({ra:.5f}, {dec:+.5f})")
    for r in rows:
        p = download(r["url"], Path(out_dir) / r["filename"])
        r["path"] = p
        if verbose:
            print(f"  {r['filename']}  MJD {r['mjd']}  z = {r['z']:.4f}  {'ok' if p else 'FAILED'}")
    return rows


# ----------------------------------------------------------------------------
# DESI
# ----------------------------------------------------------------------------
def desi_healpix(ra, dec):
    return int(ang2pix_nest(DESI_NSIDE, ra, dec))


def desi_coadd_url(release, survey, program, healpix):
    spec = DESI_RELEASES[release]
    hp = int(healpix)
    return (f"{DESI_PUBLIC}/{release}/spectro/redux/{spec}/healpix/{survey}/{program}/{hp // 100}/{hp}/"
            f"coadd-{survey}-{program}-{hp}.fits")


def _redrock_url(coadd_url):
    """URL of the redrock file next to a coadd (same directory, ``redrock-`` prefix)."""
    head, name = coadd_url.rsplit("/", 1)
    return head + "/" + name.replace("coadd-", "redrock-", 1)


def desi_tile_coadd_url(release, tileid, lastnight, petal):
    """URL of the tile-cumulative coadd of one petal of a tile: every exposure
    of the tile through its last night, ``coadd-<petal>-<tileid>-thru<lastnight>.fits``."""
    spec = DESI_RELEASES[release]
    t, n, pt = int(tileid), int(lastnight), int(petal)
    return (f"{DESI_PUBLIC}/{release}/spectro/redux/{spec}/tiles/{DESI_TILE_GROUP}/{t}/{n}/"
            f"coadd-{pt}-{t}-thru{n}.fits")


def desi_tile_redrock_url(release, tileid, lastnight, petal):
    """URL of the redrock file of a tile-cumulative coadd."""
    return _redrock_url(desi_tile_coadd_url(release, tileid, lastnight, petal))


def desi_tiles_csv_url(release):
    """URL of the tiles table of a release (one row per tile: SURVEY, PROGRAM,
    TILERA, TILEDEC, LASTNIGHT, ...)."""
    spec = DESI_RELEASES[release]
    return f"{DESI_PUBLIC}/{release}/spectro/redux/{spec}/tiles-{spec}.csv"


def fetch_desi_tiles_table(release, out_dir, clobber=False):
    """Download the tiles table of a release into ``out_dir`` (once: an existing
    file is kept). Returns the path."""
    url = desi_tiles_csv_url(release)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = download(url, Path(out_dir) / url.rsplit("/", 1)[1], clobber=clobber)
    if path is None:
        raise OSError(f"could not download {url}")
    return path


def read_desi_tiles_table(path):
    """dict TILEID -> dict(tileid, survey, program, lastnight, tilera, tiledec,
    nexp) from a tiles table (``tiles-<specprod>.csv``)."""
    import csv

    def num(row, key, cast, default):
        v = row.get(key)
        try:
            return cast(v) if v not in (None, "") else default
        except ValueError:
            return default

    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            tileid = int(row["TILEID"])
            out[tileid] = dict(tileid=tileid, survey=str(row.get("SURVEY", "")).strip(),
                               program=str(row.get("PROGRAM", "")).strip(),
                               lastnight=num(row, "LASTNIGHT", int, None),
                               tilera=num(row, "TILERA", float, np.nan), tiledec=num(row, "TILEDEC", float, np.nan),
                               nexp=num(row, "NEXP", int, -1))
    return out


def _url_exists(url, timeout=20):
    import requests
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def _open_remote(url, block_size=DESI_REMOTE_BLOCK_SIZE):
    from astropy.io import fits
    return fits.open(url, use_fsspec=True, fsspec_kwargs={"block_size": int(block_size)}, lazy_load_hdus=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _exposure_provenance(h, targetid):
    """Retain the exposure membership rather than treating releases as epochs.

    A matching fingerprint identifies the same set of EXPIDs for this target;
    different fingerprints can still overlap. Missing EXPIDs never establish
    independence. Records preserve dates and tile identifiers where supplied.
    """
    records, expids = [], []
    if "EXP_FIBERMAP" in h:
        table = h["EXP_FIBERMAP"].data
        names = set(table.columns.names)
        if "TARGETID" in names:
            rows = np.flatnonzero(np.asarray(table["TARGETID"]).astype(np.int64) == int(targetid))
            columns = [name for name in ("EXPID", "NIGHT", "TILEID", "MJD") if name in names]
            for row in rows:
                record = {}
                for name in columns:
                    value = table[name][row]
                    value = value.item() if isinstance(value, np.generic) else value
                    if isinstance(value, bytes):
                        value = value.decode("ascii", errors="replace").strip()
                    if np.ma.is_masked(value) or (isinstance(value, float) and not np.isfinite(value)):
                        value = None
                    record[name.lower()] = value
                records.append(record)
            if "EXPID" in names:
                expids = sorted({int(r["expid"]) for r in records
                                 if r["expid"] is not None and int(r["expid"]) >= 0})
    records.sort(key=_canonical_json)
    fingerprint = None
    if records and expids and all(r.get("expid") is not None and int(r["expid"]) >= 0 for r in records):
        fingerprint = hashlib.sha256(_canonical_json({"targetid": int(targetid),
                                                      "exposure_ids": expids}).encode()).hexdigest()
    return {"exposures": records, "exposure_ids": expids,
            "exposure_fingerprint": fingerprint}


def _extract_desi_bundle(h, targetid, out_dir, release, survey, program, healpix, url):
    """Publish the healpix coadd of a survey/program reduced to one target
    (``out_dir/desi/<release>/<specprod>/<survey>-<program>-<healpix>-<targetid>/<bundle_id>``)."""
    specprod = DESI_RELEASES[release]
    stem = f"{survey}-{program}-{healpix}-{targetid}"
    identity = dict(schema_version=1, release=release, specprod=specprod, survey=survey, program=program,
                    healpix=int(healpix), targetid=int(targetid))
    return _publish_desi_bundle(h, targetid, out_dir / "desi" / release / specprod / stem, stem, url, identity)


def _extract_desi_tile_bundle(h, targetid, out_dir, release, epoch, url):
    """Publish one tile epoch reduced to one target
    (``out_dir/desi/<release>/<specprod>/tiles/<petal>-<tileid>-thru<lastnight>-<targetid>/<bundle_id>``).
    The identity carries the tile, its last night and the petal; the nights and
    exposures come from the tile coadd's own exposure table."""
    specprod = DESI_RELEASES[release]
    tileid, lastnight, petal = int(epoch["tileid"]), int(epoch["lastnight"]), int(epoch["petal"])
    stem = f"{petal}-{tileid}-thru{lastnight}-{targetid}"
    identity = dict(schema_version=1, release=release, specprod=specprod, group=DESI_TILE_GROUP,
                    survey=epoch.get("survey"), program=epoch.get("program"),
                    healpix=int(epoch["healpix"]) if epoch.get("healpix") is not None else None,
                    tileid=tileid, lastnight=lastnight, petal=petal, targetid=int(targetid))
    return _publish_desi_bundle(h, targetid, out_dir / "desi" / release / specprod / "tiles" / stem, stem, url,
                                identity, with_nights=True)


def _publish_desi_bundle(h, targetid, parent, stem, url, identity, with_nights=False):
    """Validate and atomically publish an immutable coadd/redrock/metadata set.

    The directory is renamed only after all files are complete. Including the
    release, reduction and file hashes in its identity preserves previous
    retrievals and ensures a missing redrock can never find a stale sibling.
    ``identity`` holds the fields that name the source (release, survey,
    program, healpix or tile, target); the URLs, file checksums and exposure
    membership are added here, and the observing nights when ``with_nights``.
    """
    from astropy.io import fits
    from .desi import read_desi

    parent.mkdir(parents=True, exist_ok=True)
    rr_url = _redrock_url(url)
    with tempfile.TemporaryDirectory(dir=parent, prefix=".staging-") as stage:
        stage = Path(stage)
        coadd_name, rr_name = f"coadd-{stem}.fits", f"redrock-{stem}.fits"
        # Download the large parent redrock outside the bundle: only the
        # extracted target is part of the immutable artifact.
        with tempfile.TemporaryDirectory(dir=parent, prefix=".redrock-") as rr_stage:
            rr_tmp = download(rr_url, Path(rr_stage) / "redrock.fits")
            write_single_target(h, targetid, str(stage / coadd_name), redrock_in=rr_tmp,
                                redrock_out=str(stage / rr_name) if rr_tmp else None)
        data = read_desi(str(stage / coadd_name), targetid, use_desispec=False)
        files = {coadd_name: _sha256(stage / coadd_name)}
        if rr_tmp:
            with fits.open(stage / rr_name, memmap=False) as rr:
                matches = np.asarray(rr["REDSHIFTS"].data["TARGETID"]).astype(np.int64) == int(targetid)
                if np.count_nonzero(matches) != 1:
                    raise ValueError(f"Redrock has no unique redshift row for TARGETID {targetid}: {rr_url}")
            files[rr_name] = _sha256(stage / rr_name)
        provenance = dict(identity, coadd_url=url, redrock_url=rr_url, redrock_available=bool(rr_tmp),
                          files=files, **_exposure_provenance(h, targetid))
        if with_nights:
            provenance["nights"] = sorted({int(r["night"]) for r in provenance["exposures"]
                                           if r.get("night") is not None})
        encoded = _canonical_json(provenance)
        bundle_id = hashlib.sha256(encoded.encode()).hexdigest()
        (stage / "provenance.json").write_text(encoded + "\n", encoding="utf-8")
        destination = parent / bundle_id
        try:
            os.rename(stage, destination)
        except OSError:
            # A concurrent or earlier identical retrieval may already exist.
            # Never silently reuse damaged content, or overwrite a saved result.
            if not destination.is_dir():
                raise
            same = (destination / "provenance.json").read_text(encoding="utf-8") == encoded + "\n"
            same = same and {p.name for p in destination.iterdir()} == set(files) | {"provenance.json"}
            same = same and all(_sha256(destination / name) == digest for name, digest in files.items())
            if not same:
                raise ValueError(f"Existing DESI bundle is inconsistent: {destination}")
        paths = dict(path=str(destination / coadd_name),
                     redrock=str(destination / rr_name) if rr_tmp else None,
                     provenance_path=str(destination / "provenance.json"), bundle_id=bundle_id)
        return data, dict(provenance, **paths)


def fetch_desi(ra, dec, out_dir, targetid=None, radius_arcsec=1.0, releases=("dr1", "edr"),
               verbose=True):
    """Find and extract the DESI public spectra of a position.

    Returns a list of dicts (release, specprod, survey, program, healpix,
    targetid, sep_arcsec, path, redrock, z, zwarn, spectype, provenance_path,
    bundle_id, exposure_ids, exposure_fingerprint) for every matched coadd.
    Single-target files and their provenance are published together in
    immutable subdirectories of ``out_dir/desi/<release>/<specprod>``.
    Different releases may reuse exposures; they are not independent epochs.
    """
    hp = desi_healpix(ra, dec)
    out = []
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cosd = np.cos(np.radians(dec))
    for rel in releases:
        for survey, program in DESI_SURVEY_PROGRAMS[rel]:
            url = desi_coadd_url(rel, survey, program, hp)
            if not _url_exists(url):
                continue
            t0 = time.time()
            with _open_remote(url) as h:
                fm = h["FIBERMAP"].data
                tids = np.asarray(fm["TARGETID"]).astype(np.int64)
                if targetid is not None:
                    idx = np.flatnonzero(tids == int(targetid))
                    if idx.size == 0:
                        if verbose:
                            print(f"  {rel} {survey}/{program} healpix {hp}: TARGETID {int(targetid)} not in this coadd")
                        continue
                    i = int(idx[0])
                else:
                    sep = np.hypot((np.asarray(fm["TARGET_RA"], float) - ra) * cosd,
                                   np.asarray(fm["TARGET_DEC"], float) - dec) * 3600.0
                    i = int(np.argmin(sep))
                    if sep[i] > radius_arcsec:
                        if verbose:
                            print(f"  {rel} {survey}/{program} healpix {hp}: nearest target {sep[i]:.1f}\" away, no match")
                        continue
                tid = int(tids[i])
                sep_i = float(np.hypot((float(fm["TARGET_RA"][i]) - ra) * cosd, float(fm["TARGET_DEC"][i]) - dec) * 3600.0)
                d, provenance = _extract_desi_bundle(h, tid, out_dir, rel, survey, program, hp, url)
            rec = dict(provenance, sep_arcsec=sep_i,
                       z=d["z"], zwarn=d["zwarn"], spectype=d["spectype"], mjd=d["mjd"], ebv=d["ebv"])
            out.append(rec)
            if verbose:
                print(f"  {rel} {survey}/{program} healpix {hp}: TARGETID {tid} ({sep_i:.2f}\"), "
                      f"z = {d['z']:.4f} {d['spectype']} ZWARN {d['zwarn']}, MJD {d['mjd']:.1f} -> {rec['path']} "
                      f"[{time.time() - t0:.1f} s]")
    if verbose and not out:
        print(f"DESI: no public spectrum within {radius_arcsec}\" of ({ra:.5f}, {dec:+.5f}) in {', '.join(releases)} (healpix {hp})")
    return out


def list_desi_epochs(ra, dec, targetid, release="dr1", tiles_csv=None, out_dir=None, verbose=True):
    """Tile epochs of a target in a public release, from the exposure tables
    of the healpix coadds of its position.

    Every survey/program healpix coadd of the position is opened remotely and
    the exposures of ``targetid`` in its EXP_FIBERMAP are grouped by tile: a
    tile is one epoch, observed on one petal over one or more nights. The last
    night of the tile, which names its cumulative coadd, comes from the tiles
    table of the release (``tiles_csv``; downloaded into ``out_dir`` when not
    given). Returns a list of dicts (release, specprod, survey, program,
    healpix, targetid, tileid, lastnight, petal, nights, expids, mjds,
    in_tiles_table, epoch_key, coadd_url, redrock_url) ordered by first
    exposure; ``epoch_key`` is ``<tileid>-<lastnight>-<petal>``. A tile absent
    from the tiles table has ``lastnight``, ``epoch_key`` and the URLs None.
    An empty list means that no healpix coadd of the release lists the target.
    """
    hp = desi_healpix(ra, dec)
    tid = int(targetid)
    if tiles_csv is None:
        if out_dir is None:
            raise ValueError("list_desi_epochs needs tiles_csv= or out_dir= (where the tiles table is downloaded)")
        tiles_csv = fetch_desi_tiles_table(release, out_dir)
    tiles = read_desi_tiles_table(tiles_csv)
    spec = DESI_RELEASES[release]
    epochs = []
    for survey, program in DESI_SURVEY_PROGRAMS[release]:
        url = desi_coadd_url(release, survey, program, hp)
        if not _url_exists(url):
            continue
        with _open_remote(url) as h:
            if "EXP_FIBERMAP" not in h:
                if verbose:
                    print(f"  {release} {survey}/{program} healpix {hp}: no exposure table, epochs unknown")
                continue
            table = h["EXP_FIBERMAP"].data
            names = set(table.columns.names)
            rows = np.flatnonzero(np.asarray(table["TARGETID"]).astype(np.int64) == tid)
            if rows.size == 0 or "TILEID" not in names:
                if verbose:
                    print(f"  {release} {survey}/{program} healpix {hp}: TARGETID {tid} not in the exposure table")
                continue
            tileids = np.asarray(table["TILEID"][rows]).astype(np.int64)
            for tileid in sorted(set(tileids.tolist())):
                sel = rows[tileids == tileid]
                if "PETAL_LOC" in names:
                    petals = sorted({int(v) for v in table["PETAL_LOC"][sel]})
                elif "FIBER" in names:
                    petals = sorted({int(v) // 500 for v in table["FIBER"][sel]})
                else:
                    petals = []
                if len(petals) != 1:
                    # a target has one fiber per tile; anything else is not a tile coadd of this target
                    if verbose:
                        print(f"  {release} {survey}/{program} tile {tileid}: petal of TARGETID {tid} undetermined {petals}")
                    continue
                tile = tiles.get(tileid)
                lastnight = tile["lastnight"] if tile else None
                ep = dict(release=release, specprod=spec, survey=survey, program=program, healpix=hp, targetid=tid,
                          tileid=tileid, lastnight=lastnight, petal=petals[0],
                          nights=sorted({int(v) for v in table["NIGHT"][sel]}) if "NIGHT" in names else [],
                          expids=sorted({int(v) for v in table["EXPID"][sel]}) if "EXPID" in names else [],
                          mjds=sorted(float(v) for v in table["MJD"][sel]) if "MJD" in names else [],
                          in_tiles_table=tile is not None, epoch_key=None, coadd_url=None, redrock_url=None)
                if lastnight is not None:
                    ep["epoch_key"] = f"{tileid}-{lastnight}-{petals[0]}"
                    ep["coadd_url"] = desi_tile_coadd_url(release, tileid, lastnight, petals[0])
                    ep["redrock_url"] = _redrock_url(ep["coadd_url"])
                epochs.append(ep)
    epochs.sort(key=lambda e: (min(e["mjds"]) if e["mjds"] else float(min(e["nights"])) if e["nights"] else np.inf,
                               e["tileid"]))
    if verbose:
        print(f"DESI {release}: {len(epochs)} tile epoch(s) of TARGETID {tid} (healpix {hp}): "
              + ", ".join(f"{e['tileid']}/{e['program']} {'-'.join(str(n) for n in e['nights'])}"
                          + ("" if e["in_tiles_table"] else " (not in the tiles table)") for e in epochs))
    return epochs


def fetch_desi_epochs(ra, dec, targetid, out_dir, release="dr1", tiles_csv=None, epochs=None, verbose=True):
    """Extract every tile epoch of a target from the tile-cumulative coadds of a
    public release.

    ``epochs`` defaults to ``list_desi_epochs`` of the position (the tiles
    table is then downloaded into ``out_dir`` unless ``tiles_csv`` is given).
    Each epoch is published, with the layout, checksums and provenance of the
    healpix bundles of ``fetch_desi``, under
    ``out_dir/desi/<release>/<specprod>/tiles/<petal>-<tileid>-thru<lastnight>-<targetid>/<bundle_id>``;
    its provenance records tileid, lastnight, petal, nights and the exposures
    of the tile coadd itself, whose fingerprint identifies the epoch.
    Returns one dict per listed epoch, in the order of the listing: the
    listing fields plus ``status`` and, for 'ok', path, redrock,
    provenance_path, bundle_id, exposure_ids, exposure_fingerprint, nights,
    z, zwarn, spectype, mjd, ebv and ``expids_match`` (the tile coadd holds
    exactly the exposures the healpix table listed for the tile). The other
    statuses are 'no_lastnight' (tile absent from the tiles table), 'missing'
    (no coadd file on the server) and 'not_in_coadd' (the target is not in
    that petal's file); such epochs have ``path`` None.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tid = int(targetid)
    if epochs is None:
        epochs = list_desi_epochs(ra, dec, tid, release=release, tiles_csv=tiles_csv, out_dir=out_dir,
                                  verbose=verbose)
    out = []
    for ep in epochs:
        rec = dict(ep, status=None, path=None, redrock=None)
        t0 = time.time()
        if ep.get("lastnight") is None:
            rec["status"] = "no_lastnight"
        else:
            url = ep.get("coadd_url") or desi_tile_coadd_url(release, ep["tileid"], ep["lastnight"], ep["petal"])
            if not _url_exists(url):
                rec["status"] = "missing"
            else:
                with _open_remote(url) as h:
                    fm = h["FIBERMAP"].data
                    if np.count_nonzero(np.asarray(fm["TARGETID"]).astype(np.int64) == tid) == 0:
                        rec["status"] = "not_in_coadd"
                    else:
                        d, provenance = _extract_desi_tile_bundle(h, tid, out_dir, release, ep, url)
                        listed = sorted(int(v) for v in ep.get("expids") or [])
                        rec.update(provenance, status="ok", z=d["z"], zwarn=d["zwarn"], spectype=d["spectype"],
                                   mjd=d["mjd"], ebv=d["ebv"],
                                   expids_match=(listed == provenance["exposure_ids"]) if listed else None)
        out.append(rec)
        if verbose:
            key = ep.get("epoch_key") or f"{ep['tileid']}-?-{ep['petal']}"
            if rec["status"] == "ok":
                print(f"  {release} tile epoch {key}: z = {rec['z']:.4f} {rec['spectype']} ZWARN {rec['zwarn']}, "
                      f"MJD {rec['mjd']:.1f}, {len(rec['exposure_ids'])} exposure(s) -> {rec['path']} "
                      f"[{time.time() - t0:.1f} s]")
            else:
                print(f"  {release} tile epoch {key}: {rec['status']}")
    return out
