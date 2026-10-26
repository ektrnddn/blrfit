"""Shared pieces of the multi epoch analysis: the inputs (a pair table, a catalog, masses, spectrum facts), the
instrument and kind of every spectrum and pair, the profile-shape and signal-to-noise thresholds, and the populations
that calibrate the noise model.

Inputs
------
``Inputs`` names the tables the chain reads; every stage takes one.

* ``pairs``: the pair table of ``blrfit pair`` (``pairs.PAIR_COLUMNS``, one row per pair, line and direction pair),
  with the per-spectrum fields the chain uses: ``cls_a/b`` (profile class), ``snr_a/b`` (integrated broad S/N),
  ``peak_snr_a/b``, ``fwhm_a/b``, ``mjd_a/b``, ``id_a/b`` (spectrum identifiers), ``dt_days``, ``dt_rest_yr``,
  and, where the pairs were measured on a catalog of several identifiers per object, ``object``, ``tid_a``,
  ``tid_b``; optional ``bpk_a/b`` (the branch of the class B rule: ``two`` resolved peaks or ``broad``),
  ``fflags_a/b`` (the fit's quality flags, ``;``-joined) and ``fstatus_a/b`` (``success|True`` for a converged fit).
  Rows with ``variant`` other than ``none`` or ``design`` other than ``real`` (injection tests) are skipped.
* ``catalog``: one row per identifier (``TARGETID``, ``RA``, ``DEC``, ``z_in``, ``HA_class``, ``HA_c50_sys``,
  ``HA_fwhm``, ``HA_broad_flux_snr``, ``HB_class``, ``CLEAN``) with the coadd fit of every object: it defines the
  objects (identifiers within 1.5 arcsec are one object), the populations and the matched comparison.
* ``masses``: one row per identifier with the virial mass and the orbital limits of ``blrfit.physics.target_mass``
  (``LOGMBH``, ``R_BLR_LTD``, ``VMAX_Q01``, ``PMIN_YR_Q01``).
* ``facts`` (optional): one row per dated spectrum and line with its own fit (``id``, ``line``, ``chosen``,
  ``c50_sys``, ``class``, ``class_reason``, ``instrument``, ``broad_flux_snr``): the offset scale of the orbit fits
  and the two-peak rule of the spectra of objects without a coadd fit.
* ``all_fits`` (optional): the coadd fits with the two-peak measurements (``HA_n_peaks``, ``HA_peak_sep``,
  ``HA_dip_frac``, ``HA_fwhm`` and the same for Hbeta): the double-peaked objects, set aside from the verdicts.

Instruments: DESI (a night), SDSS (the SDSS-I/II spectrographs, MJD < 55000) and BOSS (the rebuilt BOSS/eBOSS
spectrographs, MJD >= 55000).
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOSS_MJD = 55000.0  # the last SDSS-I/II spectrum of our catalog is MJD 54941, the first BOSS one 55240
SHORT_DAYS = 30.0  # DESI nights at most this far apart: the error-calibration set, no change expected
SHAPE_MAX = (
    0.5  # the shape statistic per pixel above which two profiles do not match (``pairs.shape_per_pixel``)
)
SNR_MIN = 8.0  # integrated broad-line S/N of both spectra of a pair
VSYS_MAX = (
    200.0  # the two fits' own narrow velocities must agree to this (km/s): else the frame does not hold
)
GOOD_CLASSES = ("A", "C", "F")
LINES = ("Halpha", "Hbeta")
MERGE_ARCSEC = 1.5  # identifiers within this of one another are one object


def f(v):
    """A float from a table cell: numbers, numeric strings, empty and masked cells."""
    if v is None or isinstance(v, np.ma.core.MaskedConstant):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


_SDSS_SITE = re.compile(r"site_sdss_(\d+)_(\d+)_(\d+)")
_SDSS_SPEC = re.compile(r"(?<!\d)(\d{3,5})-(\d{5})-(\d{3,4})(?!\d)")


def sdss_pmf(spec_id):
    """(plate, MJD, fiber) of an SDSS spectrum identifier (``site_sdss_PLATE_MJD_FIBER`` or ``spec-PLATE-MJD-FIBER``),
    else None."""
    m = _SDSS_SITE.search(spec_id) or _SDSS_SPEC.search(spec_id)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def independent(pair_label):
    """False for two SDSS spectra of the same plate and fiber (a re-observation whose later coadd includes the
    earlier exposures) and for a spectrum paired with itself; the label is ``id_a__id_b``."""
    a, b = pair_label.split("__")
    pa, pb = sdss_pmf(a), sdss_pmf(b)
    if pa and pb and pa[0] == pb[0] and pa[2] == pb[2]:
        return False
    return a != b


def instrument(spec_id, mjd):
    if spec_id.startswith("desi"):
        return "DESI"
    return "BOSS" if f(mjd) >= BOSS_MJD else "SDSS"


def reduction(spec_id):
    return (
        "daily" if spec_id.startswith("desi_daily") else ("loa" if spec_id.startswith("desi_loa") else "sdss")
    )


def pair_kind(r):
    """desi-desi short (<= 30 d) | desi-desi | sdss-desi | boss-desi | sdss-sdss | sdss-boss | boss-boss."""
    ia, ib = instrument(r["id_a"], r["mjd_a"]), instrument(r["id_b"], r["mjd_b"])
    order = {"SDSS": 0, "BOSS": 1, "DESI": 2}
    a, b = sorted((ia, ib), key=order.get)
    k = f"{a.lower()}-{b.lower()}"
    if k == "desi-desi" and f(r["dt_days"]) <= SHORT_DAYS:
        k += " short"
    return k


def shape(r):
    """The shape statistic of a pair per pixel, the larger of the two directions."""
    return max(
        f(r["dchi2_shape_fwd"]) / max(f(r["npix_fwd"]), 1),
        f(r["dchi2_shape_rev"]) / max(f(r["npix_rev"]), 1),
    )


def nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


# ----------------------------------------------------------------------------
# the inputs
# ----------------------------------------------------------------------------
def read_table(path):
    """A table of any format astropy reads (FITS, ECSV, CSV)."""
    from astropy.table import Table

    path = str(path)
    if path.endswith(".csv"):
        return Table.read(path, format="ascii.csv")
    return Table.read(path)


def load_pairs(path):
    """The rows of a pair table (CSV, or gzipped CSV), the injection tests left out."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as fh:
        return [
            r
            for r in csv.DictReader(fh)
            if r.get("variant", "none") == "none" and r.get("design", "real") == "real"
        ]


def load_masses(path):
    M = read_table(path)

    def cell(v):
        if isinstance(v, np.ma.core.MaskedConstant):
            return np.nan
        return float(v) if np.issubdtype(np.asarray(v).dtype, np.number) else v

    return {str(int(r["TARGETID"])): {c: cell(r[c]) for c in M.colnames} for r in M}


def duplicate_groups(C, radius_arcsec=MERGE_ARCSEC):
    """TARGETID -> object key: catalog rows within ``radius_arcsec`` of one another are one object (two DESI
    identifiers for one quasar)."""
    ra = np.radians(np.asarray(C["RA"], float))
    dec = np.radians(np.asarray(C["DEC"], float))
    tids = [str(int(t)) for t in C["TARGETID"]]
    order = np.argsort(dec)
    key = {t: t for t in tids}
    r = np.radians(radius_arcsec / 3600.0)
    for i in range(len(order)):
        a = order[i]
        for j in range(i + 1, len(order)):
            b = order[j]
            if dec[b] - dec[a] > r:
                break
            dra = (ra[a] - ra[b]) * np.cos(0.5 * (dec[a] + dec[b]))
            if np.hypot(dra, dec[a] - dec[b]) <= r:
                root = min(key[tids[a]], key[tids[b]])
                for t in (tids[a], tids[b]):
                    key[t] = root
    for t in tids:  # resolve chains
        while key[key[t]] != key[t]:
            key[t] = key[key[t]]
    return key


def object_key(catalog):
    """TARGETID -> object key (identifiers within 1.5 arcsec are one object)."""
    g = duplicate_groups(catalog)
    return lambda tid: g.get(tid, tid)


@dataclass
class Inputs:
    """The tables of one analysis (module docstring). Paths; the loaders cache nothing, every stage reads what it
    needs."""

    pairs: str
    catalog: str
    masses: str | None = None
    facts: str | None = None
    all_fits: str | None = None

    def load_pairs(self):
        return load_pairs(self.pairs)

    def load_catalog(self):
        return read_table(self.catalog)

    def load_masses(self):
        return load_masses(self.masses) if self.masses else {}

    def check(self):
        for name in ("pairs", "catalog", "masses", "facts", "all_fits"):
            p = getattr(self, name)
            if p and not Path(p).exists():
                raise FileNotFoundError(f"{name}: {p}")
        return self


def populations(catalog, okey=None):
    """TARGETID sets: 'null' = coadd |dv(Halpha)| < 300 km/s, class A or F, clean (ordinary lines that should not
    move); 'broad' = coadd |dv(Halpha)| < 1000 km/s (everything but the strong offsets). With okey, the sets hold
    object keys and an object belongs when its key identifier does."""

    def col(n):
        c = catalog[n]
        return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)

    dv = col("HA_c50_sys")
    tids = [str(int(t)) for t in catalog["TARGETID"]]
    cls = [str(c).strip() for c in catalog["HA_class"]]
    clean = [_b(c) for c in catalog["CLEAN"]]
    null = {
        t
        for t, d, c, cl in zip(tids, dv, cls, clean)
        if np.isfinite(d) and abs(d) < 300 and c in ("A", "F") and cl
    }
    broad = {t for t, d in zip(tids, dv) if np.isfinite(d) and abs(d) < 1000}
    if okey is not None:
        null = {t for t in null if okey(t) == t}
        broad = {t for t in broad if okey(t) == t}
    return {"null": null, "broad": broad}


def _b(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "t", "yes")


def spectrum_fields(rows):
    """(TARGETID, spectrum id, line) -> dict(cls, snr, peak_snr, mjd, fwhm, instrument, reduction, ...), from the pair
    rows (rows of a line not fitted carry empty S/N: the filled values win)."""
    out = {}
    for r in rows:
        for e in "ab":
            k = (r["targetid"], r[f"id_{e}"], r["line"])
            new = dict(
                cls=r[f"cls_{e}"],
                bpk=r.get(f"bpk_{e}", ""),
                fflags=r.get(f"fflags_{e}", ""),
                fstatus=r.get(f"fstatus_{e}", ""),
                snr=f(r[f"snr_{e}"]),
                peak_snr=f(r[f"peak_snr_{e}"]),
                mjd=f(r[f"mjd_{e}"]),
                fwhm=f(r[f"fwhm_{e}"]),
                instrument=instrument(r[f"id_{e}"], r[f"mjd_{e}"]),
                reduction=reduction(r[f"id_{e}"]),
            )
            old = out.get(k)
            if old is None or (not np.isfinite(old["snr"]) and np.isfinite(new["snr"])):
                out[k] = new
    return out


def two_peak_rule(n_peaks, peak_sep, dip, fwhm):
    """The two-peak branch of the class B rule (``blrfit.classify``): two resolved peaks more than
    max(0.4 FWHM, 1500 km/s) apart, a dip deeper than 8 per cent, FWHM >= 3000 km/s."""
    try:
        n, sep, d, fw = (f(x) for x in (n_peaks, peak_sep, dip, fwhm))
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(fw)
        and fw >= 3000
        and n >= 2
        and np.isfinite(sep)
        and sep > max(0.4 * fw, 1500.0)
        and np.isfinite(d)
        and d > 0.08
    )


def double_peaked_objects(catalog, okey, facts_path=None, all_fits=None):
    """Object keys whose broad line has two resolved peaks in the coadd (the two-peak rule), in Halpha where the
    coadd measures it, else in Hbeta. Objects without a coadd fit in ``all_fits``: a class B coadd whose best DESI
    spectrum (largest broad S/N) meets the two-peak rule in that line (the facts table). Empty without either table."""
    out = set()
    have = set()
    if all_fits and Path(all_fits).exists():
        A = read_table(all_fits)
        for r in A:
            t = str(int(r["TARGETID"]))
            if okey(t) != t:
                continue
            have.add(t)
            ha_ok = np.isfinite(f(r["HA_fwhm"])) and str(r["HA_class"]).strip() not in ("", "E", "X")
            if ha_ok:
                if two_peak_rule(r["HA_n_peaks"], r["HA_peak_sep"], r["HA_dip_frac"], r["HA_fwhm"]):
                    out.add(t)
            elif two_peak_rule(r["HB_n_peaks"], r["HB_peak_sep"], r["HB_dip_frac"], r["HB_fwhm"]):
                out.add(t)
    if facts_path:
        best = {}
        with open(facts_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("line") and r["chosen"] == "True" and r["instrument"] == "DESI":
                    k = (r["object"], r["line"])
                    if k not in best or f(r["broad_flux_snr"]) > f(best[k]["broad_flux_snr"]):
                        best[k] = r
        for r in catalog:
            t = str(int(r["TARGETID"]))
            if okey(t) != t or t in have:
                continue
            ha, hb = str(r["HA_class"]).strip(), str(r["HB_class"]).strip()
            line = "Halpha" if ha not in ("", "E", "X") else "Hbeta"
            if (ha if line == "Halpha" else hb) == "B":
                b = best.get((t, line))
                if b and b["class"] == "B" and "two peaks" in b.get("class_reason", ""):
                    out.add(t)
    return out


def by_object(rows, okey):
    d = defaultdict(list)
    for r in rows:
        d[okey(r["targetid"])].append(r)
    return d


def members_of(rows, okey):
    """object key -> the identifiers behind it, from the pair rows."""
    members = defaultdict(set)
    for r in rows:
        members[okey(r["targetid"])].add(r["targetid"])
        for t in (r.get("tid_a"), r.get("tid_b")):
            if t:
                members[okey(r["targetid"])].add(t)
    return members


def redshifts(catalog):
    return {str(int(t)): float(z) for t, z in zip(catalog["TARGETID"], np.asarray(catalog["z_in"], float))}


def load_facts(path):
    """(spectrum id, line) -> the spectrum's own offset c50_sys (km/s), from the facts table."""
    facts = {}
    if not path:
        return facts
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("line") and r["chosen"] == "True" and r["c50_sys"] not in ("", "nan"):
                facts[(r["id"], r["line"])] = float(r["c50_sys"])
    return facts


def mass_of(o, masses, members):
    """The mass record of an object: its key identifier's, else the first of its other identifiers'."""
    return masses.get(o) or next((masses.get(t) for t in sorted(members.get(o, ())) if masses.get(t)), None)


def redshift_of(o, zof, members, rows=()):
    """The redshift of an object: its key identifier's, else another identifier's, else the one its fits used."""
    z = zof.get(o, np.nan)
    if not np.isfinite(z):
        z = next((zof[t] for t in members.get(o, ()) if np.isfinite(zof.get(t, np.nan))), np.nan)
    if not np.isfinite(z):
        zs = [f(r.get(f"z_{e}")) for e in "ab" for r in rows if r.get(f"tid_{e}") == o] + [
            f(r.get("z_a")) for r in rows
        ]
        z = next((x for x in zs if np.isfinite(x)), 0.0)
    return z
