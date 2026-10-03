"""Public discovery followed by independent single-spectrum fits."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .io import fetch
from .io.public import query_desi, fetch_desi_products, position, separation


def _manifest(path, value):
    # The CLI JSON cleaner also turns non-finite metadata into JSON null.
    from .cli import _clean
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".manifest-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(_clean(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def retrieve(a, directory):
    """Save all candidates and require an explicit choice for multiple DESI IDs.

    SDSS neighbours are labelled positional candidates, never asserted to be
    the same physical source. Repeated visits are retained by product identity.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "fetch_manifest.json"
    if manifest.exists():
        raise ValueError(f"output already contains a search: {manifest}; use a new --out directory")
    if a.survey not in ("desi", "sdss"):
        raise ValueError("public discovery supports --survey desi (default) or sdss")
    if a.survey == "sdss" and a.targetid is not None:
        raise ValueError("--targetid is a DESI identifier; for SDSS alone give RA and Dec")
    if (a.ra is None) != (a.dec is None):
        raise ValueError("give both --ra and --dec")
    if a.ra is not None:
        position(a.ra, a.dec)
    if not 0 < a.radius <= 1.5:
        raise ValueError("SDSS radius must be positive and at most 1.5 arcsec")
    report = dict(schema="blrfit-public-inputs-1", survey=a.survey, targetid=a.targetid,
        ra=a.ra, dec=a.dec, sdss_radius_arcsec=a.radius, desi=[], sdss=[], errors=[],
        complete=False, product_scope="public coadds; not a complete nightly observing census",
        association="SDSS matches are positional candidates; inspect identity and redshift")
    _manifest(manifest, report)
    try:
        ra, dec = a.ra, a.dec
        if a.survey == "desi":
            rows = query_desi(targetid=a.targetid, ra=ra, dec=dec, radius_arcsec=a.desi_radius,
                              releases=tuple(x.strip() for x in a.releases.split(",") if x.strip()))
            report["desi_candidates"] = rows
            _manifest(manifest, report)
            ids = sorted({r["targetid"] for r in rows})
            if len(ids) > 1 and not a.all_matches:
                raise ValueError(f"multiple DESI TARGETIDs {ids}; select --targetid or use --all-matches; candidates saved in {manifest}")
            if ra is None and rows:
                ra, dec = rows[0]["ra"], rows[0]["dec"]
                if any(separation(ra, dec, r["ra"], r["dec"]) > 0.01 for r in rows):
                    raise ValueError("TARGETID has inconsistent catalogue coordinates; supply a reviewed position")
                report["resolved_ra"], report["resolved_dec"] = ra, dec
            report["desi"] = fetch_desi_products(rows, directory, verbose=not a.quiet)
        if a.survey == "sdss" or a.include_sdss:
            if ra is None:
                raise ValueError("SDSS matching needs RA/Dec or a resolved public DESI TARGETID")
            report["sdss"] = fetch.fetch_sdss(ra, dec, directory, radius_arcsec=a.radius,
                                               data_release=17, verbose=not a.quiet)
            for r in report["sdss"]:
                r["kind"] = "sdss"
                if not r["path"]:
                    report["errors"].append(f"download failed: {r['url']}")
        report["complete"] = not report["errors"]
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        _manifest(manifest, report)
        raise
    _manifest(manifest, report)
    return report


def fit_public(a, fit_one):
    if a.stem:
        raise ValueError("public inputs receive separate product-based names; --stem is only for a local file")
    directory = Path(a.out)
    report = retrieve(a, directory / "inputs")
    results = []
    for r in report["desi"] + report["sdss"]:
        if not r.get("path"):
            continue
        one = copy.copy(a)
        one.spectrum = r["path"]
        one.survey = r["kind"]
        one.targetid = r.get("targetid")
        one.include_sdss = False
        one.public_source = r
        one.redrock = r.get("redrock")
        one.out = str(directory / "fits")
        identity = json.dumps([r.get("coadd_url", r.get("url")), r.get("targetid")], separators=(",", ":"))
        token = hashlib.sha256(identity.encode()).hexdigest()[:12]
        one.stem = f"{one.survey}_{token}"
        try:
            fit_one(one)
            results.append(dict(input=r, status="returned", output=str(Path(one.out)/(one.stem+"_fit.json"))))
        except (Exception, SystemExit) as exc:
            # Retain failed products; one bad/no-continuum spectrum must not
            # hide the results or identities of the other observations.
            results.append(dict(input=r, status="failed", error=str(exc)))
        _manifest(directory / "fit_manifest.json", dict(discovery=report, results=results,
                  uncertainty_calibrated=False))
    if not results:
        _manifest(directory / "fit_manifest.json", dict(discovery=report, results=[],
                  status="no_spectra", uncertainty_calibrated=False))
    if not a.quiet:
        print(f"{sum(r['status']=='returned' for r in results)}/{len(results)} spectra returned; "
              f"see {directory/'fit_manifest.json'}")
    return 0 if results and report["complete"] and all(r["status"] == "returned" for r in results) else 1


def fetch_public(a):
    report = retrieve(a, a.out)
    count = len(report["desi"]) + sum(bool(r.get("path")) for r in report["sdss"])
    if not a.quiet:
        print(f"{count} public spectra; manifest {Path(a.out)/'fetch_manifest.json'}")
    return 0 if report["complete"] and count else 1
