"""Indexed public DESI discovery. Network failures are not empty catalogues.

The Data Lab zpix index supplies exact product locations, including objects on
HEALPix boundaries. The spectra and redshifts are read from the DESI archive.
This interface returns full-depth coadds, not independent observing nights.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np

from . import fetch as F

TAP_URL = "https://datalab.noirlab.edu/tap/sync"
MAX_PRODUCTS = 500


def exact_targetid(value):
    text = str(value)
    if not re.fullmatch(r"[0-9]+", text) or not 0 <= int(text) < 2**63:
        raise ValueError("TARGETID must be an exact nonnegative 64-bit integer")
    return int(text)


def position(ra, dec):
    ra, dec = float(ra), float(dec)
    if not np.isfinite([ra, dec]).all() or not 0 <= ra < 360 or not -90 <= dec <= 90:
        raise ValueError("RA must be in [0,360) and Dec in [-90,90], in degrees")
    return ra, dec


def separation(ra, dec, other_ra, other_dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return SkyCoord(ra*u.deg, dec*u.deg).separation(
        SkyCoord(np.asarray(other_ra)*u.deg, np.asarray(other_dec)*u.deg)).arcsec


def query_desi(*, targetid=None, ra=None, dec=None, radius_arcsec=1.5, releases=("dr1",)):
    """All matching zpix rows; IDs remain integers, including above 2**53.

    Coordinates, if supplied with an ID, constrain that ID as well. The service
    response is checked locally against the requested ID and angular radius.
    """
    requests = F.require_requests()
    if targetid is None and ra is None:
        raise ValueError("give --targetid or both --ra and --dec")
    if (ra is None) != (dec is None):
        raise ValueError("give both --ra and --dec")
    if not np.isfinite(radius_arcsec) or not 0 < radius_arcsec <= 60:
        raise ValueError("search radius must be positive and at most 60 arcsec")
    tid = exact_targetid(targetid) if targetid is not None else None
    if ra is not None:
        ra, dec = position(ra, dec)
    clauses = []
    if tid is not None:
        clauses.append(f"s.targetid={tid}")
    if ra is not None:
        clauses.append(f"'t'=q3c_radial_query(p.ra,p.dec,{ra:.15g},{dec:.15g},{radius_arcsec/3600:.15g})")
    rows, seen = [], set()
    for release in dict.fromkeys(releases):
        if release not in F.DESI_RELEASES:
            raise ValueError(f"unsupported DESI release {release!r}; choose dr1 or edr")
        query = (f"SELECT TOP {MAX_PRODUCTS+1} s.targetid,p.ra AS target_ra,p.dec AS target_dec,"
                 "s.survey,s.program,s.healpix,s.z,s.zwarn "
                 f"FROM desi_{release}.zpix AS s JOIN desi_{release}.photometry AS p "
                 "ON s.targetid=p.targetid WHERE " + " AND ".join(clauses))
        response = requests.get(TAP_URL, params=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv",
                                                     QUERY=query), timeout=60)
        response.raise_for_status()
        reader = csv.DictReader(io.StringIO(response.text))
        required = {"targetid", "target_ra", "target_dec", "survey", "program", "healpix", "z", "zwarn"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("DESI catalogue service returned an error or an unexpected schema; retry later")
        found = list(reader)
        if len(found) >= MAX_PRODUCTS+1:
            raise ValueError("too many public products; narrow the query")
        for r in found:
            t = exact_targetid(r["targetid"])
            rra, rdec = position(r["target_ra"], r["target_dec"])
            sep = float(separation(ra, dec, rra, rdec)) if ra is not None else None
            if (tid is not None and t != tid) or (sep is not None and sep > radius_arcsec):
                continue
            survey, program = r["survey"].strip(), r["program"].strip()
            if (survey, program) not in F.DESI_SURVEY_PROGRAMS[release]:
                raise ValueError("unknown survey/program in DESI catalogue response")
            hp = int(r["healpix"])
            if not 0 <= hp < 12*64**2:
                raise ValueError("invalid HEALPix in DESI catalogue response")
            key = (release, survey, program, hp, t)
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(release=release, survey=survey, program=program, healpix=hp, targetid=t,
                             ra=rra, dec=rdec, sep_arcsec=sep, catalogue_z=float(r["z"]),
                             catalogue_zwarn=int(r["zwarn"]), product="healpix_coadd"))
    return rows


def fetch_desi_products(rows, out_dir, *, verbose=True):
    """Read exactly the indexed target rows and retain their source provenance."""
    result = []
    for row in rows:
        url = F.desi_coadd_url(row["release"], row["survey"], row["program"], row["healpix"])
        if verbose:
            print(f"DESI {row['release']} TARGETID {row['targetid']}: reading public coadd", flush=True)
        with F._open_remote(url) as h:
            fm = h["FIBERMAP"].data
            idx = np.flatnonzero(np.asarray(fm["TARGETID"], dtype=np.int64) == row["targetid"])
            if len(idx) != 1:
                raise ValueError("public coadd has no unique row for the indexed TARGETID")
            i = idx[0]
            if separation(row["ra"], row["dec"], float(fm["TARGET_RA"][i]), float(fm["TARGET_DEC"][i])) > 0.01:
                raise ValueError("public coadd position differs from its catalogue record")
            data, provenance = F._extract_desi_bundle(h, row["targetid"], Path(out_dir),
                row["release"], row["survey"], row["program"], row["healpix"], url)
        result.append(dict(row, **{k: v for k, v in provenance.items() if k not in row},
                           z=data["z"], kind="desi"))
    return result
