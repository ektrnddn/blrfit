"""
DESI spectra: healpix and tile coadds read with astropy alone.

A DESI coadd file has one row per target in ``FIBERMAP`` and, per camera B, R
and Z, the extensions ``*_WAVELENGTH``, ``*_FLUX``, ``*_IVAR``, ``*_MASK`` and
``*_RESOLUTION`` (flux in 1e-17 erg/s/cm^2/A on a 0.8 A grid in vacuum). The
cameras overlap (B/R around 5760-5800 A, R/Z around 7520-7620 A) on identical
wavelengths. The camera coadd implemented here reproduces
``desispec.coaddition.coadd_cameras``: outside the overlaps the pixels are
copied; inside, the flux is the inverse-variance-weighted mean, the inverse
variances add, and the masks are combined with OR. The mask is returned but
not applied: the DESI catalogue fits used the inverse variance alone, which is
zero for unusable pixels.

The redshift is not in the coadd. It is taken from the sibling
``redrock-*.fits`` file (extension ``REDSHIFTS``) when that file sits next to
the coadd, and must otherwise be given. The Galactic E(B-V) of the target
(``FIBERMAP['EBV']``) and the exposure dates (``EXP_FIBERMAP['MJD']``) are
returned as well.

Coadd and redrock files compressed with gzip (``*.fits.gz``) are read in
place; the redrock sibling of a coadd may be compressed or not independently
of the coadd.
"""
from __future__ import annotations

import os
import re

import numpy as np

CAMERAS = ("B", "R", "Z")
WAVE_TOLERANCE = 1e-4


FITS_SUFFIXES = (".fits", ".fits.gz")


def is_desi_coadd(path):
    """True for a DESI healpix or tile coadd (``coadd-*.fits``, plain or gzipped)."""
    name = os.path.basename(str(path)).lower()
    return name.startswith("coadd-") and name.endswith(FITS_SUFFIXES)


def is_desi_spectra(path):
    """True for a per-exposure DESI ``spectra-*.fits`` file (plain or gzipped),
    which holds one row per exposure of a target and is not what the fitter
    should read."""
    name = os.path.basename(str(path)).lower()
    return name.startswith("spectra-") and name.endswith(FITS_SUFFIXES)


def redrock_sibling(path):
    """Path of the redrock file next to a coadd, or None if it does not exist.
    The sibling is sought with the coadd's own suffix first, then with the other
    one (``.fits`` for a gzipped coadd, ``.fits.gz`` for a plain one)."""
    d, name = os.path.split(str(path))
    if not name.lower().startswith("coadd-"):
        return None
    stem = name[6:]
    for suf in FITS_SUFFIXES:
        if stem.lower().endswith(suf):
            stem = stem[:-len(suf)]
            break
    else:
        rr = os.path.join(d, "redrock-" + stem)
        return rr if os.path.exists(rr) else None
    own = name.lower().endswith(".fits.gz")
    for suf in ((".fits.gz", ".fits") if own else (".fits", ".fits.gz")):
        rr = os.path.join(d, "redrock-" + stem + suf)
        if os.path.exists(rr):
            return rr
    return None


def coadd_cameras(waves, fluxes, ivars, masks=None):
    """Combine the per-camera arrays of one spectrum onto the common grid.

    ``waves``, ``fluxes``, ``ivars`` (and ``masks``) are lists over cameras in
    order of increasing mean wavelength. Returns (wave, flux, ivar, mask).
    """
    wave = None
    for w in waves:
        w = np.asarray(w, float)
        wave = w if wave is None else np.append(wave, w[w > wave[-1] + WAVE_TOLERANCE])
    n = wave.size
    flux = np.zeros(n); ivar = np.zeros(n); mask = np.zeros(n, dtype=np.int32)
    overlap_any = np.zeros(n, bool)
    slices, overlaps = [], []
    for i, w in enumerate(waves):
        w = np.asarray(w, float)
        start = int(np.searchsorted(wave, w[0])); sl = slice(start, start + w.size)
        if np.any(np.abs(w - wave[sl]) > WAVE_TOLERANCE):
            raise ValueError("camera wavelength grids are not aligned")
        ov = np.zeros(w.size, bool)
        if i > 0:
            ov |= np.isin(np.round(w, 4), np.round(np.asarray(waves[i - 1], float), 4))
        if i < len(waves) - 1:
            ov |= np.isin(np.round(w, 4), np.round(np.asarray(waves[i + 1], float), 4))
        slices.append(sl); overlaps.append(ov)
    for i, (sl, ov) in enumerate(zip(slices, overlaps)):
        f = np.asarray(fluxes[i], float); iv = np.asarray(ivars[i], float)
        m = np.asarray(masks[i]) if masks is not None else np.zeros(f.size, dtype=np.int32)
        idx = np.arange(sl.start, sl.stop)
        flux[idx[~ov]] = f[~ov]; ivar[idx[~ov]] = iv[~ov]; mask[idx[~ov]] = m[~ov]
        flux[idx[ov]] += iv[ov] * f[ov]; ivar[idx[ov]] += iv[ov]; mask[idx[ov]] |= m[ov].astype(np.int32)
        overlap_any[idx[ov]] = True
    flux[overlap_any] /= (ivar[overlap_any] + (ivar[overlap_any] == 0))
    mask[ivar > 0] = 0
    return wave, flux, ivar, mask


def _fibermap_row(fm, targetid):
    idx = np.flatnonzero(np.asarray(fm["TARGETID"]).astype(np.int64) == int(targetid))
    if idx.size == 0:
        raise ValueError(f"TARGETID {int(targetid)} not in FIBERMAP")
    return int(idx[0])


def read_desi(path, targetid, redrock=None, use_desispec=None):
    """Read one target from a DESI coadd. Returns dict(wave, flux, ivar, mask, z,
    zerr, zwarn, spectype, ebv, ra, dec, mjd, mjd_min, mjd_max, nexp, targetid,
    survey, program, healpix, path, kind='desi'); ``z`` is NaN when no redrock
    file is found.

    ``use_desispec``: None uses desispec when importable, True requires it,
    False uses the astropy reader. Both give identical arrays.
    """
    tid = int(targetid)
    if is_desi_spectra(path):
        raise ValueError(f"{path} is a per-exposure spectra file (one row per exposure); "
                         f"use the coadd-*.fits file of the same healpix or tile")
    if use_desispec is None:
        import importlib.util
        try:
            use_desispec = importlib.util.find_spec("desispec") is not None
        except (ImportError, ValueError):
            use_desispec = False
    if use_desispec:
        wave, flux, ivar, mask = _read_with_desispec(path, tid)
        meta = _read_meta(path, tid)
    else:
        wave, flux, ivar, mask, meta = _read_with_astropy(path, tid)
    out = dict(wave=wave, flux=flux, ivar=ivar, mask=mask, targetid=tid, path=str(path), kind="desi")
    out.update(meta)
    rr = redrock if redrock is not None else redrock_sibling(path)
    out.update(read_redrock(rr, tid) if rr else dict(z=np.nan, zerr=np.nan, zwarn=-1, spectype=""))
    return out


def _read_with_desispec(path, tid):
    from desispec.io import read_spectra
    from desispec.coaddition import coadd_cameras as _cc
    sp = read_spectra(str(path), targetids=[tid])
    sp = _cc(sp)
    idx = np.flatnonzero(np.asarray(sp.fibermap["TARGETID"]) == tid)
    if idx.size == 0:
        raise ValueError(f"TARGETID {tid} not in {path}")
    i = int(idx[0])
    band = "brz" if "brz" in sp.wave else list(sp.wave)[0]
    wave = np.asarray(sp.wave[band], float)
    flux = np.asarray(sp.flux[band][i], float)
    ivar = np.asarray(sp.ivar[band][i], float)
    mask = np.asarray(sp.mask[band][i]) if sp.mask is not None else np.zeros_like(flux, bool)
    return wave, flux, ivar, mask


def _read_meta(path, tid):
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        fm = h["FIBERMAP"].data
        i = _fibermap_row(fm, tid)
        meta = _meta_from_fibermap(fm, i, h)
    return meta


def _meta_from_fibermap(fm, i, h):
    names = fm.columns.names
    meta = dict(ebv=float(fm["EBV"][i]) if "EBV" in names else 0.0,
                ra=float(fm["TARGET_RA"][i]) if "TARGET_RA" in names else np.nan,
                dec=float(fm["TARGET_DEC"][i]) if "TARGET_DEC" in names else np.nan,
                nexp=int(fm["COADD_NUMEXP"][i]) if "COADD_NUMEXP" in names else -1,
                mjd=np.nan, mjd_min=np.nan, mjd_max=np.nan)
    hdr = h[0].header
    meta["survey"] = str(hdr.get("SURVEY", "")).strip()
    meta["program"] = str(hdr.get("PROGRAM", "")).strip()
    meta["healpix"] = int(hdr["HPXPIXEL"]) if "HPXPIXEL" in hdr else -1
    meta["tileid"] = int(hdr["TILEID"]) if "TILEID" in hdr else -1
    if "EXP_FIBERMAP" in h:
        ef = h["EXP_FIBERMAP"].data
        rows = np.flatnonzero(np.asarray(ef["TARGETID"]).astype(np.int64) == int(fm["TARGETID"][i]))
        if rows.size and "MJD" in ef.columns.names:
            mj = np.asarray(ef["MJD"][rows], float)
            meta.update(mjd=float(np.mean(mj)), mjd_min=float(np.min(mj)), mjd_max=float(np.max(mj)))
            if meta["nexp"] < 0:
                meta["nexp"] = int(rows.size)
    return meta


def _read_with_astropy(path, tid):
    # memmap=False: the mask extensions of tile coadds are unsigned integers stored
    # with BZERO, which astropy cannot slice through a memory map; ``section`` still
    # reads only the rows it is asked for
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        fm = h["FIBERMAP"].data
        i = _fibermap_row(fm, tid)
        cams = [c for c in CAMERAS if f"{c}_WAVELENGTH" in h]
        if not cams:
            raise ValueError(f"no camera extensions in {path}")
        waves, fluxes, ivars, masks = [], [], [], []
        for c in cams:
            waves.append(np.asarray(h[f"{c}_WAVELENGTH"].data, float))
            fluxes.append(np.asarray(h[f"{c}_FLUX"].section[i, :], float))
            ivars.append(np.asarray(h[f"{c}_IVAR"].section[i, :], float))
            masks.append(np.asarray(h[f"{c}_MASK"].section[i, :]) if f"{c}_MASK" in h
                         else np.zeros(waves[-1].size, dtype=np.int32))
        order = np.argsort([np.mean(w) for w in waves])
        waves = [waves[k] for k in order]; fluxes = [fluxes[k] for k in order]
        ivars = [ivars[k] for k in order]; masks = [masks[k] for k in order]
        meta = _meta_from_fibermap(fm, i, h)
    if len(cams) == 1:
        return waves[0], fluxes[0], ivars[0], masks[0], meta
    wave, flux, ivar, mask = coadd_cameras(waves, fluxes, ivars, masks)
    return wave, flux, ivar, mask, meta


def read_redrock(path, targetid):
    """Redshift of one target from a redrock file (extension REDSHIFTS)."""
    from astropy.io import fits
    tid = int(targetid)
    with fits.open(path, memmap=False) as h:
        t = h["REDSHIFTS"].data
        idx = np.flatnonzero(np.asarray(t["TARGETID"]).astype(np.int64) == tid)
        if idx.size == 0:
            return dict(z=np.nan, zerr=np.nan, zwarn=-1, spectype="")
        i = int(idx[0])
        return dict(z=float(t["Z"][i]), zerr=float(t["ZERR"][i]) if "ZERR" in t.columns.names else np.nan,
                    zwarn=int(t["ZWARN"][i]) if "ZWARN" in t.columns.names else -1,
                    spectype=str(t["SPECTYPE"][i]).strip() if "SPECTYPE" in t.columns.names else "")


def write_single_target(hdul_or_path, targetid, out_path, redrock_in=None, redrock_out=None):
    """Write a coadd file (and optionally its redrock file) reduced to one
    target, keeping the DESI extension layout so that both readers accept it.
    ``hdul_or_path`` may be an open HDUList (also a lazily loaded remote one)."""
    from astropy.io import fits
    tid = int(targetid)
    own = isinstance(hdul_or_path, (str, os.PathLike))
    h = fits.open(hdul_or_path, memmap=False) if own else hdul_or_path
    try:
        fm = h["FIBERMAP"].data
        i = _fibermap_row(fm, tid)
        out = [fits.PrimaryHDU(header=h[0].header)]
        for hdu in h[1:]:
            name = hdu.name
            if name in ("FIBERMAP", "SCORES", "EXTRA_CATALOG"):
                out.append(fits.BinTableHDU(data=hdu.data[i:i + 1], header=hdu.header, name=name))
            elif name == "EXP_FIBERMAP":
                rows = np.flatnonzero(np.asarray(hdu.data["TARGETID"]).astype(np.int64) == tid)
                out.append(fits.BinTableHDU(data=hdu.data[rows], header=hdu.header, name=name))
            elif re.match(r"^[BRZ]_WAVELENGTH$", name):
                out.append(fits.ImageHDU(data=np.asarray(hdu.data), header=hdu.header, name=name))
            elif re.match(r"^[BRZ]_(FLUX|IVAR|MASK|RESOLUTION|MODEL)$", name):
                out.append(fits.ImageHDU(data=np.asarray(hdu.section[i:i + 1]), header=hdu.header, name=name))
        fits.HDUList(out).writeto(out_path, overwrite=True)
    finally:
        if own:
            h.close()
    if redrock_in is not None and redrock_out is not None:
        rr = fits.open(redrock_in, memmap=False)
        try:
            outr = [fits.PrimaryHDU(header=rr[0].header)]
            for hdu in rr[1:]:
                if getattr(hdu, "columns", None) is not None and "TARGETID" in hdu.columns.names:
                    rows = np.flatnonzero(np.asarray(hdu.data["TARGETID"]).astype(np.int64) == tid)
                    outr.append(fits.BinTableHDU(data=hdu.data[rows], header=hdu.header, name=hdu.name))
            fits.HDUList(outr).writeto(redrock_out, overwrite=True)
        finally:
            rr.close()
    return out_path
