"""
The Liu et al. (2014) and Eracleous et al. (2012) same-spectrum anchor.

Liu et al. published, for each of their 399 offset quasars, the plate, fibre
and date of the SDSS spectrum they measured and the peak velocity offset of
broad Hbeta. Fitting the same spectra (388 available in DR16) at their
redshifts with the frozen 0.1.0 model gave, for the 370 with a measurable
broad Hbeta, a Pearson r of 0.91, a median difference of +6 km/s, an NMAD of
104 km/s and 96 per cent sign agreement between our peak offsets and theirs.

The spectra (300 MB) are not part of the repository. Point BLRFIT_ANCHOR_DIR
at a directory holding

* ``lit_targets.fits``: one row per object of Table 1 of Liu et al. (2014)
  and of Eracleous et al. (2012) with columns LID (an integer identifier of
  the object), Z (the published redshift), PLATE, MJD, FIBER (the SDSS
  spectrum Liu et al. measured), VOFFP (their peak velocity offset, km/s) and
  SAMPLE_LIT ('Liu+2014' or 'Eracleous+2012');
* ``SDSS_fits/<LID>/spec-PPPP-MMMMM-FFFF.fits``: every SDSS spectrum of those
  objects (824 in the project's copy: the exact Liu spectra, the other epochs
  of the same objects and the Eracleous objects), as served by the SDSS
  science archive (``blrfit fetch`` or ``blrfit.io.fetch.download``);
* optionally ``compare.fits``, the stored per-spectrum results of the DESI
  catalogue run with 0.1.0 (one row per spectrum, keyed by LID, PLATE and
  MJD_SPEC; columns EXACT, OUR_HB_CLASS, OUR_HB_C50, OUR_HB_PEAK,
  OUR_HA_CLASS, OUR_HA_C50).

Every spectrum is fitted at the published redshift with the default settings
(host and Fe II on, E(B-V) = 0, Halpha and Hbeta), as in the catalogue run.
Two things are asserted.

The population statistics against the literature, on the exact Liu spectra
with a measurable broad Hbeta (classes A, B, C, F), as in 0.1.0: n >= 365,
Pearson r >= 0.90, |median difference| <= 20 km/s, NMAD <= 115 km/s and sign
agreement >= 95 per cent between our peak offset (``HB_v_peak_sys``) and
theirs.

The per-spectrum comparison with the stored 0.1.0 results, when
``compare.fits`` is present: the fraction of spectra whose Hbeta class changed
must be below CLASS_CHANGE_MAX = 0.10 and the NMAD of the change of c50_sys
of Hbeta below DELTA_NMAD_KMS = 10 km/s. The gates sit just above the values
measured with 0.2.0 on 2026-09-14 (8.6 per cent and 6.4 km/s); the comment at
the constants attributes them to the Fe II corrections. The 0.1.0 gate
(98 per cent of the spectra reproduce the stored class and peak within 100
km/s) is not applied: the corrections change the end point by design.

With BLRFIT_WRITE_DELTAS=1 the per-spectrum deltas (both classes, both
c50_sys, the difference, the BIC margin and the continuum state of the
corrected fit, for Hbeta and Halpha) are written to
``docs/deltas_anchor_0.1.0_to_0.2.0.csv``, sorted by LID, plate and MJD, one
row per fitted spectrum with a stored result; the file is written before the
gates are asserted.

    BLRFIT_ANCHOR_DIR=/path/to/lit BLRFIT_NPROC=8 BLRFIT_WRITE_DELTAS=1 \\
        pytest -m slow -s tests/test_anchor_liu.py
"""
import csv
import os
from collections import Counter

import numpy as np
import pytest

from conftest import ROOT

ANCHOR = os.environ.get("BLRFIT_ANCHOR_DIR", "")
NPROC = int(os.environ.get("BLRFIT_NPROC", "4"))
WRITE_DELTAS = os.environ.get("BLRFIT_WRITE_DELTAS", "") not in ("", "0")
DELTAS_CSV = os.path.join(ROOT, "docs", "deltas_anchor_0.1.0_to_0.2.0.csv")

# the literature: the 0.1.0 tolerances (measured then: r 0.91, median +6, NMAD 104 km/s, sign 96 per cent)
LIT_N_MIN = 365
LIT_R_MIN = 0.90
LIT_MEDIAN_KMS = 20.0
LIT_NMAD_KMS = 115.0
LIT_SIGN_MIN = 0.95
# the stored 0.1.0 run. Measured 2026-09-14 (numpy 1.26.4, scipy 1.13.1, macOS arm64):
# 71 of 823 Hbeta classes change (8.6 per cent), NMAD of the Hbeta c50_sys change
# 6.4 km/s; Halpha 8 of 476 classes (1.7 per cent), NMAD 0.6 km/s. Refitted with the
# historical 50 km/s Fe II operator only 2.1 per cent of the Hbeta classes change,
# and with the ultraviolet Fe II width left free 10.0 per cent: about three quarters
# of the Hbeta changes come from the continuous Fe II operator (the Fe II blends
# under Hbeta move the decomposition of objects near a class boundary), and fixing
# the ultraviolet width removes changes rather than adding them (docs/DELTAS.md).
# The gates sit just above the measured values so that a later change shows up.
CLASS_CHANGE_MAX = 0.10
DELTA_NMAD_KMS = 10.0
BROAD_CLASSES = ("A", "B", "C", "F")

DELTA_COLUMNS = ["lid", "spectrum", "sample", "exact", "z",
                 "hb_class_0.1.0", "hb_class_0.2.0", "hb_c50_sys_0.1.0", "hb_c50_sys_0.2.0", "hb_delta_c50",
                 "hb_bic_margin_0.2.0", "hb_n_broad_0.2.0", "hb_fit_status_0.2.0",
                 "ha_class_0.1.0", "ha_class_0.2.0", "ha_c50_sys_0.1.0", "ha_c50_sys_0.2.0", "ha_delta_c50",
                 "ha_bic_margin_0.2.0",
                 "conti_feop_fwhm_0.2.0", "conti_at_bound_0.2.0", "feuv_fwhm_fixed_0.2.0", "continuum_status_0.2.0"]

pytestmark = pytest.mark.slow


def _fit_one(job):
    lid, path, z = job
    import blrfit
    from blrfit.io import read_sdss
    sp = read_sdss(path)
    try:
        res = blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z, complexes=("Halpha", "Hbeta"))
    except Exception:
        return lid, os.path.basename(path), None
    return lid, os.path.basename(path), blrfit.summary_row(res)


def _jobs(T):
    """Every spectrum under SDSS_fits/<LID>/ of an object of the literature
    table, at its published redshift."""
    z_of = {int(r["LID"]): float(r["Z"]) for r in T}
    base = os.path.join(ANCHOR, "SDSS_fits")
    jobs = []
    for lid in sorted(os.listdir(base)):
        d = os.path.join(base, lid)
        if not (lid.isdigit() and os.path.isdir(d) and int(lid) in z_of):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.startswith("spec-") and fn.endswith(".fits"):
                jobs.append((int(lid), os.path.join(d, fn), z_of[int(lid)]))
    return jobs


def _plate_mjd(fn):
    parts = fn[:-5].split("-")
    return int(parts[1]), int(parts[2])


def _nmad(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return 1.4826 * float(np.median(np.abs(d - np.median(d)))) if d.size else np.nan


def _num(v):
    return "" if v is None or not np.isfinite(v) else f"{v:.3f}"


def _delta_row(lid, fn, row, s, sample, exact):
    hb_old, ha_old = str(s["OUR_HB_CLASS"]).strip(), str(s["OUR_HA_CLASS"]).strip()
    hb_c50_old, ha_c50_old = float(s["OUR_HB_C50"]), float(s["OUR_HA_C50"])
    hb_c50_new, ha_c50_new = float(row.get("HB_c50_sys", np.nan)), float(row.get("HA_c50_sys", np.nan))
    return {"lid": lid, "spectrum": fn, "sample": sample, "exact": exact, "z": row["z_in"],
            "hb_class_0.1.0": hb_old, "hb_class_0.2.0": row.get("HB_class", ""),
            "hb_c50_sys_0.1.0": hb_c50_old, "hb_c50_sys_0.2.0": hb_c50_new, "hb_delta_c50": hb_c50_new - hb_c50_old,
            "hb_bic_margin_0.2.0": row.get("HB_bic_margin", np.nan), "hb_n_broad_0.2.0": row.get("HB_n_broad", ""),
            "hb_fit_status_0.2.0": row.get("HB_fit_status", ""),
            "ha_class_0.1.0": ha_old, "ha_class_0.2.0": row.get("HA_class", ""),
            "ha_c50_sys_0.1.0": ha_c50_old, "ha_c50_sys_0.2.0": ha_c50_new, "ha_delta_c50": ha_c50_new - ha_c50_old,
            "ha_bic_margin_0.2.0": row.get("HA_bic_margin", np.nan),
            "conti_feop_fwhm_0.2.0": row.get("conti_feop_fwhm", np.nan), "conti_at_bound_0.2.0": row.get("conti_at_bound", ""),
            "feuv_fwhm_fixed_0.2.0": row.get("conti_feuv_fwhm_fixed", ""), "continuum_status_0.2.0": row.get("continuum_status", "")}


def _write_deltas(deltas, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DELTA_COLUMNS)
        w.writeheader()
        for d in deltas:
            w.writerow({k: (_num(v) if isinstance(v, float) else v) for k, v in d.items()})


@pytest.mark.skipif(not ANCHOR, reason="set BLRFIT_ANCHOR_DIR to the literature spectra")
def test_anchor_population_and_deltas():
    from astropy.table import Table
    T = Table.read(os.path.join(ANCHOR, "lit_targets.fits"))
    cpath = os.path.join(ANCHOR, "compare.fits")
    C = Table.read(cpath) if os.path.exists(cpath) else None
    sample = {int(r["LID"]): str(r["SAMPLE_LIT"]).strip() for r in T}
    liu = T[np.char.startswith(np.asarray(T["SAMPLE_LIT"]).astype(str), "Liu")]
    exact = {(int(r["LID"]), f"spec-{int(r['PLATE']):04d}-{int(r['MJD']):05d}-{int(r['FIBER']):04d}.fits")
             for r in liu if np.isfinite(r["PLATE"])}
    lit = {int(r["LID"]): float(r["VOFFP"]) for r in liu}
    jobs = _jobs(T)
    n_exact = sum((lid, os.path.basename(p)) in exact for lid, p, _ in jobs)
    assert n_exact >= 380, f"only {n_exact} of the exact Liu spectra found"
    import multiprocessing as mp
    with mp.Pool(NPROC) as pool:
        rows = list(pool.imap_unordered(_fit_one, jobs, chunksize=2))
    rows.sort(key=lambda t: (t[0], _plate_mjd(t[1])))
    n_failed = sum(row is None for _, _, row in rows)

    # the literature, on the exact Liu spectra with a measurable broad Hbeta
    x, y = [], []
    for lid, fn, row in rows:
        if row is None or (lid, fn) not in exact:
            continue
        if row.get("HB_class", "") in BROAD_CLASSES and np.isfinite(row["HB_v_peak_sys"]) and np.isfinite(lit[lid]):
            x.append(lit[lid]); y.append(row["HB_v_peak_sys"])
    x = np.array(x); y = np.array(y); d = y - x
    n = len(x); r = np.corrcoef(x, y)[0, 1]; med = float(np.median(d)); nmad = _nmad(d)
    sign = float(np.mean(np.sign(x) == np.sign(y)))
    print(f"\n{len(rows)} spectra fitted, {n_failed} failed; Liu et al. (2014) exact spectra: n = {n}, r = {r:.3f}, "
          f"median {med:+.1f}, NMAD {nmad:.1f} km/s, sign agreement {100 * sign:.1f} %")

    # the stored 0.1.0 run, every fitted spectrum with a stored row
    deltas = []
    if C is not None:
        stored = {(int(c["LID"]), int(c["PLATE"]), int(round(float(c["MJD_SPEC"])))): c for c in C}
        for lid, fn, row in rows:
            key = (lid,) + _plate_mjd(fn)
            if row is not None and key in stored:
                deltas.append(_delta_row(lid, fn, row, stored[key], sample.get(lid, ""), (lid, fn) in exact))
    if deltas:
        classes = [(dl["hb_class_0.1.0"], dl["hb_class_0.2.0"]) for dl in deltas if dl["hb_class_0.1.0"]]
        changed = Counter((a, b) for a, b in classes if a != b)
        frac_changed = sum(changed.values()) / len(classes)
        dc = np.array([dl["hb_delta_c50"] for dl in deltas], float)
        dc = dc[np.isfinite(dc)]
        nmad_dc = _nmad(dc)
        print(f"stored 0.1.0 run: {len(deltas)} spectra compared, {len(classes)} with a stored Hbeta class, "
              f"{sum(changed.values())} changed ({100 * frac_changed:.2f} %): {dict(changed)}; "
              f"delta c50_sys of Hbeta: n = {dc.size}, median {np.median(dc):+.2f}, NMAD {nmad_dc:.2f} km/s, "
              f"16-84 {np.percentile(dc, 16):+.1f} .. {np.percentile(dc, 84):+.1f}, "
              f"|delta| > 100 km/s: {int(np.sum(np.abs(dc) > 100))}")
        if WRITE_DELTAS:
            _write_deltas(deltas, DELTAS_CSV)
            print(f"wrote {DELTAS_CSV}: {len(deltas)} rows")

    assert n >= LIT_N_MIN
    assert r >= LIT_R_MIN
    assert abs(med) <= LIT_MEDIAN_KMS
    assert nmad <= LIT_NMAD_KMS
    assert sign >= LIT_SIGN_MIN
    if deltas:
        assert frac_changed < CLASS_CHANGE_MAX, f"{100 * frac_changed:.2f} per cent of the Hbeta classes changed: {dict(changed)}"
        assert nmad_dc < DELTA_NMAD_KMS, f"NMAD of delta c50_sys {nmad_dc:.1f} km/s"
