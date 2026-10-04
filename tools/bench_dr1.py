"""
Build the local DESI DR1 bench: our objects with two or more DESI tile epochs
and SDSS spectra, extracted from the public release by range requests, so that
the whole chain (fit, epochs, cross-correlation, null scatter) runs on real
masked pixels and real flux ratios without a NERSC round trip.

    PYTHONPATH=. python tools/bench_dr1.py [--out DIR] [--catalog FITS] [--targetids FILE|ID,ID,...]
                                          [--no-sdss] [--list-only]

For each TARGETID the position and the catalogue redshift come from the
catalogue (``phase5/strong_offsets.fits``: TARGETID, RA, DEC, z_in). The DR1
healpix coadds of the position (one per survey/program, each a stack of every
tile) are extracted with ``fetch_desi``, the tile epochs with
``fetch_desi_epochs`` and the SDSS spectra with ``fetch_sdss``, all into
``<out>/<TARGETID>/``; the tiles table ``tiles-iron.csv`` is downloaded once
into ``<out>``. ``<out>/manifest.json`` records per object the files with
their checksums, the epochs with nights and exposure ids, the redshifts and a
status; it is rewritten after every object, so an interrupted run resumes
(bundles and downloads already on disk are reused). Objects absent from the
release are reported, not fatal. A table (TARGETID, DESI epochs, SDSS spectra,
status) and the totals are printed at the end.

The default target list is the 29 strong-offset objects of the catalogue that
the long-baseline table gives two or more DESI epochs before the DR1 cutoff
(2022 June 13) and at least one SDSS spectrum.
"""
import argparse
import datetime
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np

from blrfit.io import fetch

DEFAULT_TARGETIDS = (
    39627788503746647, 39633208601479081, 39627963536245812, 39628384577260677, 39628009438709144,
    39628306017944469, 39627829402407487, 39627745864453021, 39628443943437858, 39633352294139091,
    39627957097992903, 39627855193184206, 39633251853141627, 39633127722715686, 39632951360620188,
    39627925569406016, 39633339577008658, 39628453397398213, 39627983744402379, 39633328944449959,
    39633413510008024, 39633342349446421, 39627779033008899, 39628526432814213, 2782072460541953,
    2842497248133120, 39627811392061482, 39633263869822755, 39628454588584657,
)
PROJECT = "/Users/ktrnddn/Documents/MBHB_project"
DEFAULT_CATALOG = os.path.join(PROJECT, "phase5", "strong_offsets.fits")
DEFAULT_OUT = os.path.join(PROJECT, "bench_dr1")
RELEASE = "dr1"


def jsonable(v):
    """Python scalars for json; NaN and infinities become null."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [jsonable(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, bytes):
        return v.decode("ascii", errors="replace")
    return v


def parse_targetids(arg):
    if arg is None:
        return list(DEFAULT_TARGETIDS)
    if os.path.exists(arg):
        if arg.endswith(".npy"):
            return [int(t) for t in np.load(arg)]
        with open(arg) as fh:
            return [int(tok) for line in fh for tok in line.replace(",", " ").split() if not tok.startswith("#")]
    return [int(tok) for tok in arg.split(",") if tok.strip()]


def read_catalog(path, targetids):
    """dict TARGETID -> dict(ra, dec, z, survey, program, healpix) for the requested targets."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        t = h[1].data
        names = t.columns.names
    tids = np.asarray(t["TARGETID"]).astype(np.int64)
    ra_col = "RA" if "RA" in names else "TARGET_RA"
    dec_col = "DEC" if "DEC" in names else "TARGET_DEC"
    z_col = next((c for c in ("z_in", "Z", "z") if c in names), None)
    out = {}
    for tid in targetids:
        i = np.flatnonzero(tids == int(tid))
        if i.size == 0:
            continue
        i = int(i[0])
        out[int(tid)] = dict(ra=float(t[ra_col][i]), dec=float(t[dec_col][i]),
                             z=float(t[z_col][i]) if z_col else np.nan,
                             survey=str(t["SURVEY"][i]).strip() if "SURVEY" in names else "",
                             program=str(t["PROGRAM"][i]).strip() if "PROGRAM" in names else "",
                             healpix=int(t["HEALPIX"][i]) if "HEALPIX" in names else -1)
    return out


def write_manifest(path, manifest):
    fd, tmp = tempfile.mkstemp(prefix=".manifest_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(jsonable(manifest), fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def bundle_entry(rec):
    """The manifest entry of a fetched bundle (healpix coadd or tile epoch)."""
    keys = ("survey", "program", "healpix", "tileid", "lastnight", "petal", "epoch_key", "status", "nights",
            "expids", "mjds", "in_tiles_table", "path", "redrock", "provenance_path", "bundle_id", "files",
            "coadd_url", "redrock_url", "redrock_available", "exposure_ids", "exposure_fingerprint", "expids_match",
            "z", "zerr", "zwarn", "spectype", "mjd", "ebv", "sep_arcsec")
    return {k: rec[k] for k in keys if k in rec}


def status_of(entry):
    if entry["errors"] and not entry["healpix_coadds"] and not entry["epochs"]:
        return "error"
    if not entry["healpix_coadds"] and not entry["epochs"]:
        return "absent"
    n = entry["n_desi_epochs"]
    if n == 0:
        return "no_epochs"
    if n == 1:
        return "single_epoch"
    return "ok" if entry["n_sdss"] > 0 else "desi_only"


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def build_one(tid, cat, out_dir, tiles_csv, with_sdss, list_only, verbose=True):
    t0 = time.time()
    obj_dir = os.path.join(out_dir, str(tid))
    entry = dict(targetid=int(tid), ra=cat["ra"], dec=cat["dec"], z_catalog=cat["z"],
                 catalog_survey=cat["survey"], catalog_program=cat["program"],
                 healpix=fetch.desi_healpix(cat["ra"], cat["dec"]), directory=obj_dir,
                 healpix_coadds=[], epochs=[], sdss=[], errors=[], n_desi_epochs=0, n_sdss=0)
    if verbose:
        print(f"\n=== TARGETID {tid}  ({cat['ra']:.5f}, {cat['dec']:+.5f})  z = {cat['z']:.4f}  "
              f"{cat['survey']}/{cat['program']}  healpix {entry['healpix']}", flush=True)
    try:
        epochs = fetch.list_desi_epochs(cat["ra"], cat["dec"], tid, release=RELEASE, tiles_csv=tiles_csv,
                                        verbose=verbose)
        entry["epochs"] = [bundle_entry(dict(e, status="listed")) for e in epochs]
    except Exception as e:
        entry["errors"].append(f"list_desi_epochs: {type(e).__name__}: {e}")
        epochs = []
    if list_only:
        entry["n_desi_epochs"] = sum(1 for e in epochs if e["lastnight"] is not None)
        entry["status"] = status_of(entry) if entry["errors"] or epochs else "absent"
        entry["seconds"] = round(time.time() - t0, 1)
        return entry
    os.makedirs(obj_dir, exist_ok=True)
    try:
        rows = fetch.fetch_desi(cat["ra"], cat["dec"], obj_dir, targetid=tid, releases=(RELEASE,), verbose=verbose)
        entry["healpix_coadds"] = [bundle_entry(dict(r, status="ok")) for r in rows]
    except Exception as e:
        entry["errors"].append(f"fetch_desi: {type(e).__name__}: {e}")
    if epochs:
        try:
            rows = fetch.fetch_desi_epochs(cat["ra"], cat["dec"], tid, obj_dir, release=RELEASE, epochs=epochs,
                                           verbose=verbose)
            entry["epochs"] = [bundle_entry(r) for r in rows]
        except Exception as e:
            entry["errors"].append(f"fetch_desi_epochs: {type(e).__name__}: {e}")
    entry["n_desi_epochs"] = sum(1 for e in entry["epochs"] if e.get("status") == "ok")
    if with_sdss:
        try:
            rows = fetch.fetch_sdss(cat["ra"], cat["dec"], os.path.join(obj_dir, "sdss"), verbose=verbose)
            for r in rows:
                r["sha256"] = fetch._sha256(r["path"]) if r.get("path") else None
            entry["sdss"] = [{k: r.get(k) for k in ("plate", "mjd", "fiberid", "run2d", "filename", "path",
                                                     "sha256", "z", "cls", "url", "ra", "dec")} for r in rows]
        except Exception as e:
            entry["errors"].append(f"fetch_sdss: {type(e).__name__}: {e}")
    entry["n_sdss"] = sum(1 for s in entry["sdss"] if s.get("path"))
    entry["status"] = status_of(entry)
    entry["seconds"] = round(time.time() - t0, 1)
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"bench directory (default {DEFAULT_OUT})")
    ap.add_argument("--catalog", default=DEFAULT_CATALOG,
                    help="FITS table with TARGETID, RA, DEC and z_in (default the strong-offset catalogue)")
    ap.add_argument("--targetids", default=None,
                    help="comma-separated TARGETIDs, or a text/.npy file of them (default: the bench list)")
    ap.add_argument("--no-sdss", action="store_true", help="skip the SDSS query and downloads")
    ap.add_argument("--list-only", action="store_true", help="list the epochs, extract nothing")
    a = ap.parse_args(argv)

    targetids = parse_targetids(a.targetids)
    cat = read_catalog(a.catalog, targetids)
    os.makedirs(a.out, exist_ok=True)
    manifest_path = os.path.join(a.out, "manifest.json")
    manifest = dict(release=RELEASE, specprod=fetch.DESI_RELEASES[RELEASE], catalog=os.path.abspath(a.catalog),
                    created=datetime.datetime.now().isoformat(timespec="seconds"), objects={})
    if os.path.exists(manifest_path) and not a.list_only:
        try:
            with open(manifest_path) as fh:
                manifest["objects"] = json.load(fh).get("objects", {})
        except (OSError, ValueError):
            pass
    tiles_csv = fetch.fetch_desi_tiles_table(RELEASE, a.out)
    print(f"{len(targetids)} target(s), {len(cat)} in {a.catalog}; tiles table {tiles_csv}")
    t_start = time.time()
    for tid in targetids:
        if tid not in cat:
            manifest["objects"][str(tid)] = dict(targetid=int(tid), status="not_in_catalog", errors=[],
                                                 healpix_coadds=[], epochs=[], sdss=[], n_desi_epochs=0, n_sdss=0)
            print(f"\n=== TARGETID {tid}: not in the catalogue, skipped")
            continue
        try:
            entry = build_one(tid, cat[tid], a.out, tiles_csv, not a.no_sdss, a.list_only)
        except Exception as e:
            traceback.print_exc()
            entry = dict(targetid=int(tid), status="error", errors=[f"{type(e).__name__}: {e}"],
                         healpix_coadds=[], epochs=[], sdss=[], n_desi_epochs=0, n_sdss=0)
        manifest["objects"][str(tid)] = entry
        if not a.list_only:
            write_manifest(manifest_path, manifest)
    elapsed = time.time() - t_start

    print(f"\n{'TARGETID':>18s} {'DESI ep.':>8s} {'listed':>6s} {'SDSS':>4s}  status")
    counts = dict(two_epochs=0, with_sdss=0, both=0)
    for tid in targetids:
        e = manifest["objects"][str(tid)]
        n_listed = sum(1 for ep in e.get("epochs", []) if ep.get("lastnight") is not None)
        print(f"{tid:>18d} {e.get('n_desi_epochs', 0):8d} {n_listed:6d} {e.get('n_sdss', 0):4d}  {e.get('status', '')}"
              + (f"  [{'; '.join(e['errors'])}]" if e.get("errors") else ""))
        if e.get("n_desi_epochs", 0) >= 2:
            counts["two_epochs"] += 1
            counts["both"] += e.get("n_sdss", 0) > 0
        counts["with_sdss"] += e.get("n_sdss", 0) > 0
    statuses = {}
    for tid in targetids:
        s = manifest["objects"][str(tid)].get("status", "")
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\n{len(targetids)} objects: {counts['two_epochs']} with >= 2 DESI epochs, {counts['with_sdss']} with SDSS, "
          f"{counts['both']} with both; statuses {statuses}")
    if not a.list_only:
        print(f"{dir_size(a.out) / 1e6:.1f} MB in {a.out}; {elapsed / 60:.1f} min; manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
