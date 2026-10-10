"""
Several spectra in one call: the list of inputs, the catalogue table of their
summary rows, and worker processes for spectra or Monte Carlo draws.

Every spectrum of a batch is fitted exactly as it would be alone, with the same
seed, and results are returned in the order of the inputs. Worker processes
receive the same inputs as a serial run, so a table or a Monte Carlo sample
does not depend on the number of processes.
"""

from __future__ import annotations

import collections
import contextlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------
TABLE_SUFFIXES = (".fits", ".fit", ".fits.gz", ".ecsv", ".csv")


def read_list(path):
    """The spectra listed in a file, as dicts with ``path``, ``targetid`` (or
    None) and ``z`` (or None: from the file).

    A table (.fits, .ecsv or .csv) needs a column ``path`` and may have
    ``targetid`` and ``z``; any other file holds one spectrum per line,
    optionally followed by its TARGETID, with '#' starting a comment. Relative
    paths are taken from the current directory, as on the command line. A
    TARGETID must be an exact integer; a negative or empty one means none."""
    from .io.public import exact_targetid

    out = []
    if str(path).lower().endswith(TABLE_SUFFIXES):
        from astropy.table import Table

        t = Table.read(path)
        cols = {c.lower(): c for c in t.colnames}
        if "path" not in cols:
            raise ValueError(f"{path}: a list table needs a column 'path'")
        for row in t:
            entry = dict(path=str(row[cols["path"]]).strip(), targetid=None, z=None)
            tid = row[cols["targetid"]] if "targetid" in cols else None
            if tid is not None and not np.ma.is_masked(tid) and not (isinstance(tid, np.integer) and tid < 0):
                entry["targetid"] = exact_targetid(tid)
            z = row[cols["z"]] if "z" in cols else None
            if z is not None and not np.ma.is_masked(z) and np.isfinite(z):
                entry["z"] = float(z)
            out.append(entry)
    else:
        with open(path) as fh:
            for number, line in enumerate(fh, 1):
                words = line.split("#", 1)[0].split()
                if not words:
                    continue
                if len(words) > 2:
                    raise ValueError(f"{path}, line {number}: expected a path and optionally a TARGETID")
                tid = exact_targetid(words[1]) if len(words) == 2 else None
                out.append(dict(path=words[0], targetid=tid, z=None))
    if not out:
        raise ValueError(f"{path} lists no spectrum")
    return out


# ----------------------------------------------------------------------------
# the catalogue table
# ----------------------------------------------------------------------------
FLUX = "1e-17 erg / (s cm2)"
FLUX_DENSITY = "1e-17 erg / (s cm2 Angstrom)"
# Units of the summary-row columns, by the name that follows the line prefix
# (HA_, HB_, MG_) or conti_; a Monte Carlo error (*_e_<name>) has the unit of
# <name>. Fluxes are those of the input spectrum, read as 1e-17 erg/s/cm^2/A
# (the unit of SDSS and DESI spectra; the column flux_unit says what each
# input supplied).
UNITS = {
    **dict.fromkeys(
        (
            "v_sys",
            "sig_sys",
            "v_o3",
            "v_sii",
            "sig_sii",
            "v_peak",
            "centroid",
            "c25",
            "c50",
            "c75",
            "c90",
            "fwhm",
            "W25",
            "W75",
            "sigma_line",
            "peak_sep",
            "v_peak_sys",
            "centroid_sys",
            "c50_sys",
            "c25_sys",
            "c75_sys",
            "peak_top",
            "peak_top_sys",
            "centroid25_sys",
            "centroid50_sys",
            "v_o3_peak",
            "v_o3_pre",
            "nw_v",
            "nw_sig",
            "v_cover_lo",
            "v_cover_hi",
            "v_single_gauss",
            "data_v_peak",
            "data_c50",
            "data_centroid_win",
            "dv_spread",
            "fwhm_spread",
            "mc_sigma",
            "mc_gap_kms",
            "feop_fwhm",
        ),
        "km / s",
    ),
    "broad_flux": FLUX,
    "conti_at_line": FLUX_DENSITY,
    "pl_norm": FLUX_DENSITY,
    "broad_ew": "Angstrom",
    "broad_ew_agn": "Angstrom",
    "broad_lum": "erg / s",
    "ebv": "mag",
}
IDENTITY = (
    "spectrum",
    "targetid",
    "stem",
    "status",
    "error",
    "z_source",
    "ebv",
    "ebv_source",
    "ebv_assumed_zero",
    "flux_unit",
)


def column_unit(name):
    """The unit of a summary-row column (a string), or None for a dimensionless or non-numeric one."""
    for prefix in ("HA_", "HB_", "MG_", "conti_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.startswith("e_"):
        name = name[2:]
    return UNITS.get(name)


def scalar(value):
    """A table cell: numpy scalars as Python values, NaN kept, None kept (masked)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


def summary_table(records, meta=None):
    """An astropy Table of catalogue records (dicts of scalars), one row per
    record in the given order: the identity columns first, then every other
    key in the order it first appears. Values a record lacks are masked."""
    from astropy.table import MaskedColumn, Table

    names = dict.fromkeys(k for k in IDENTITY if any(k in r for r in records))
    for r in records:
        names.update(dict.fromkeys(r))
    columns = []
    for name in names:
        values = [r.get(name) for r in records]
        present = [v for v in values if v is not None]
        missing = [v is None for v in values]
        if present and all(isinstance(v, (bool, np.bool_)) for v in present):
            data, dtype = [bool(v) if v is not None else False for v in values], bool
        elif present and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in present):
            data, dtype = [int(v) if v is not None else 0 for v in values], np.int64
        elif present and all(isinstance(v, (int, float, np.integer, np.floating)) for v in present):
            data, dtype = [float(v) if v is not None else np.nan for v in values], float
        else:
            data, dtype = ["" if v is None else str(v) for v in values], str
        unit = column_unit(name) if dtype is float else None
        columns.append(MaskedColumn(data, name=name, dtype=dtype, mask=missing, unit=unit))
    return Table(columns, meta=dict(meta or {}))


def table_format(path):
    """The astropy format of a table path: FITS (.fits, .fit, .fits.gz), ECSV
    (.ecsv) or CSV (.csv; without units and types, for other programs)."""
    lower = str(path).lower()
    if lower.endswith((".fits", ".fit", ".fits.gz")):
        return "fits"
    if lower.endswith(".ecsv"):
        return "ascii.ecsv"
    if lower.endswith(".csv"):
        return "ascii.csv"
    raise ValueError(f"{path}: the table is written as FITS (.fits), ECSV (.ecsv) or CSV (.csv)")


def write_table(table, path):
    """Write a table as FITS, ECSV or CSV (see ``table_format``), creating its directory."""
    fmt = table_format(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    table.write(path, format=fmt, overwrite=True)


def _cell(value):
    """A Python scalar from a table cell: masked to None, numpy scalars to
    Python, bytes to text."""
    if np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_rows(path):
    """The rows of a table written by this package (FITS, ECSV or CSV), as
    dictionaries of Python scalars with the column names as keys."""
    from astropy.table import Table

    t = Table.read(path, format=table_format(path))
    return [{c: _cell(row[c]) for c in t.colnames} for row in t]


# ----------------------------------------------------------------------------
# worker processes
# ----------------------------------------------------------------------------
_BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


@contextlib.contextmanager
def _one_blas_thread():
    """Start worker processes with one BLAS thread each (unless the caller set
    the thread counts), so that N workers use N cores; the fitted numbers do
    not depend on the thread count."""
    saved = {k: os.environ.get(k) for k in _BLAS_THREADS}
    for k in _BLAS_THREADS:
        os.environ.setdefault(k, "1")
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def ordered_map(func, items, jobs=1, initializer=None, initargs=()):
    """Yield ``func(item)`` for every item, in the order of ``items``.

    With ``jobs`` > 1 the calls run in that many worker processes, started
    with 'spawn' (so this must be called under ``if __name__ == "__main__":``
    in a script), with at most four items per worker in flight; ``func`` and
    ``initializer`` must be module-level functions. ``items`` is consumed in
    order in either case."""
    if jobs <= 1:
        if initializer is not None:
            initializer(*initargs)
        for item in items:
            yield func(item)
        return
    context = multiprocessing.get_context("spawn")
    with (
        _one_blas_thread(),
        ProcessPoolExecutor(
            int(jobs), mp_context=context, initializer=initializer, initargs=initargs
        ) as pool,
    ):
        pending = collections.deque()
        for item in items:
            pending.append(pool.submit(func, item))
            if len(pending) >= 4 * jobs:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
