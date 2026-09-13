"""
The Liu et al. (2014) same-spectrum anchor.

Liu et al. published, for each of their 399 offset quasars, the plate, fibre
and date of the SDSS spectrum they measured and the peak velocity offset of
broad Hbeta. Fitting the same spectra (388 available in DR16) at their
redshifts with the frozen model gives, for the 370 with a measurable broad
Hbeta, a Pearson r of 0.91, a median difference of +6 km/s, an NMAD of 104
km/s and 96 per cent sign agreement between our peak offsets and theirs.

The spectra (300 MB) are not part of the repository. Point BLRFIT_ANCHOR_DIR
at a directory holding

* ``lit_targets.fits``: one row per object of Table 1 of Liu et al. (2014)
  with columns LID (any integer identifier of the object), Z (their
  redshift), PLATE, MJD, FIBER (the SDSS spectrum they measured), VOFFP (their
  peak velocity offset, km/s) and SAMPLE_LIT (the string 'Liu+2014');
* ``SDSS_fits/<LID>/spec-PPPP-MMMMM-FFFF.fits``: those spectra, as served by
  the SDSS science archive (``blrfit fetch`` or ``blrfit.io.fetch.download``);
* optionally ``compare.fits``, the stored per-spectrum results of the DESI
  catalogue run (columns LID, PLATE, EXACT, OUR_HB_CLASS, OUR_HB_PEAK); when
  present, at least 98 per cent of the spectra must reproduce their stored class and
  peak velocity (within 100 km/s, the end-point tolerance of ``tests/test_pins.py`` applied
  here to the peak, which moved by up to about 70 km/s between platforms: the solver's end
  point depends on the platform; on the reference numerical stack the agreement is exact).

    BLRFIT_ANCHOR_DIR=/path/to/lit BLRFIT_NPROC=8 pytest -m slow tests/test_anchor_liu.py
"""
import os

import numpy as np
import pytest

ANCHOR = os.environ.get("BLRFIT_ANCHOR_DIR", "")
END_POINT_KMS = 100.0      # the end-point tolerance of tests/test_pins.py: the solver's end point depends on the platform
NPROC = int(os.environ.get("BLRFIT_NPROC", "4"))

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


@pytest.mark.skipif(not ANCHOR, reason="set BLRFIT_ANCHOR_DIR to the Liu et al. (2014) spectra")
def test_liu2014_exact_spectra():
    from astropy.table import Table
    T = Table.read(os.path.join(ANCHOR, "lit_targets.fits"))
    cpath = os.path.join(ANCHOR, "compare.fits")
    C = Table.read(cpath) if os.path.exists(cpath) else None
    liu = T[np.char.startswith(np.asarray(T["SAMPLE_LIT"]).astype(str), "Liu")]
    jobs = []
    for r in liu:
        if not np.isfinite(r["PLATE"]):
            continue
        fn = f"spec-{int(r['PLATE']):04d}-{int(r['MJD']):05d}-{int(r['FIBER']):04d}.fits"
        p = os.path.join(ANCHOR, "SDSS_fits", str(int(r["LID"])), fn)
        if os.path.exists(p):
            jobs.append((int(r["LID"]), p, float(r["Z"])))
    assert len(jobs) >= 380, f"only {len(jobs)} of the exact Liu spectra found"
    import multiprocessing as mp
    with mp.Pool(NPROC) as pool:
        rows = list(pool.imap_unordered(_fit_one, jobs, chunksize=2))
    lit = {int(r["LID"]): float(r["VOFFP"]) for r in liu}
    stored = {(int(r["LID"]), int(r["PLATE"])): (str(r["OUR_HB_CLASS"]).strip(), float(r["OUR_HB_PEAK"]))
              for r in C if bool(r["EXACT"])} if C is not None else {}
    x, y, agree = [], [], []
    for lid, fn, row in rows:
        if row is None:
            continue
        cls = row.get("HB_class", "")
        plate = int(fn.split("-")[1])
        if (lid, plate) in stored:
            scls, speak = stored[(lid, plate)]
            agree.append(scls == cls and (not np.isfinite(speak) or abs(row["HB_v_peak_sys"] - speak) < END_POINT_KMS))
        if cls in ("A", "B", "C", "F") and np.isfinite(row["HB_v_peak_sys"]) and np.isfinite(lit[lid]):
            x.append(lit[lid]); y.append(row["HB_v_peak_sys"])
    x = np.array(x); y = np.array(y); d = y - x
    n = len(x); r = np.corrcoef(x, y)[0, 1]; med = np.median(d); nmad = 1.4826 * np.median(np.abs(d - med))
    sign = np.mean(np.sign(x) == np.sign(y))
    print(f"\nLiu et al. (2014) anchor: n = {n}, r = {r:.3f}, median {med:+.1f}, NMAD {nmad:.1f} km/s, sign agreement {100 * sign:.1f} %"
          + (f"; per-spectrum agreement with the stored catalogue run {100 * np.mean(agree):.1f} % of {len(agree)}" if agree else ""))
    assert n >= 365
    assert r >= 0.90
    assert abs(med) <= 20
    assert nmad <= 115
    assert sign >= 0.95
    if agree:
        assert np.mean(agree) >= 0.98
