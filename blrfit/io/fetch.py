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
"""
from __future__ import annotations

import os
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
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
        os.replace(tmp, dest)
        return str(dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


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


def _url_exists(url, timeout=20):
    import requests
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def _open_remote(url):
    from astropy.io import fits
    return fits.open(url, use_fsspec=True, fsspec_kwargs={"block_size": 1 << 20}, lazy_load_hdus=True)


def fetch_desi(ra, dec, out_dir, targetid=None, radius_arcsec=1.0, releases=("dr1", "edr"),
               verbose=True):
    """Find and extract the DESI public spectra of a position.

    Returns a list of dicts (release, survey, program, healpix, targetid, sep_arcsec,
    path, redrock, z, zwarn, spectype) for every coadd in which the target was
    found; the single-target files are written into ``out_dir``.
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
                stem = f"{survey}-{program}-{hp}-{tid}"
                coadd_out = out_dir / f"coadd-{stem}.fits"
                rr_url = url.replace("coadd-", "redrock-")
                rr_tmp = None
                with tempfile.TemporaryDirectory() as td:
                    rr_tmp = download(rr_url, Path(td) / "redrock.fits")
                    write_single_target(h, tid, str(coadd_out),
                                        redrock_in=rr_tmp, redrock_out=str(out_dir / f"redrock-{stem}.fits") if rr_tmp else None)
            from .desi import read_desi
            d = read_desi(str(coadd_out), tid, use_desispec=False)
            rec = dict(release=rel, survey=survey, program=program, healpix=hp, targetid=tid, sep_arcsec=sep_i,
                       path=str(coadd_out), redrock=str(out_dir / f"redrock-{stem}.fits") if rr_tmp else None,
                       z=d["z"], zwarn=d["zwarn"], spectype=d["spectype"], mjd=d["mjd"], ebv=d["ebv"])
            out.append(rec)
            if verbose:
                print(f"  {rel} {survey}/{program} healpix {hp}: TARGETID {tid} ({sep_i:.2f}\"), "
                      f"z = {d['z']:.4f} {d['spectype']} ZWARN {d['zwarn']}, MJD {d['mjd']:.1f} -> {coadd_out.name} "
                      f"[{time.time() - t0:.1f} s]")
    if verbose and not out:
        print(f"DESI: no public spectrum within {radius_arcsec}\" of ({ra:.5f}, {dec:+.5f}) in {', '.join(releases)} (healpix {hp})")
    return out
