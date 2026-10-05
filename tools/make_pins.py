"""
Write the pins of the fitter as it is, and the delta table against older pins.

    PYTHONPATH=. python tools/make_pins.py fit --out tests/data/pins_0.3.0.json
    PYTHONPATH=. python tools/make_pins.py deltas --legacy tests/data/pins_0.2.0.json \\
        --current tests/data/pins_0.3.0.json --out tests/data/deltas_0.2.0_to_0.3.0.csv

``fit`` fits the SDSS spectra of a legacy pin file (``tests/data/pins.json`` by
default) at their pinned redshift, E(B-V) and complexes, and the DESI example
coadd at its redrock redshift with the E(B-V) of its fibermap, with whichever
blrfit is imported (the working tree under PYTHONPATH, or an installed
version), and stores per spectrum what ``pins.json`` stores: the summary row,
the fitted parameter dictionary, chi-square and the BIC list of every complex,
the continuum parameters, the host decision and the [O III] pre-fit. The
``produced_by`` block records the blrfit version, the git description of its
source tree, the numpy, scipy, astropy and python versions, the platform and
the date. NaN is stored as null, as in ``pins.json``. The DESI entry carries
``kind`` and ``targetid``; the SDSS entries have the layout of ``pins.json``.

``deltas`` writes one row per spectrum and line of the current pin file with
the class, c50_sys, FWHM, component count and chi-square of both versions, the
BIC margin of the current fit and its continuum state (Fe II width, parameters
at a bound, whether the ultraviolet Fe II width was fixed). Several ``--legacy``
files may be given; the first that holds the spectrum supplies its old numbers
and is named in the last column. A spectrum without a legacy entry gets empty
old columns.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import re
import subprocess

import numpy as np

import blrfit
from blrfit.io import read_desi, read_sdss
from blrfit.io.desi import read_redrock, redrock_sibling
from blrfit.model.fit import PREFIX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
EXAMPLES = os.path.join(ROOT, "examples", "data")
DESI_EXAMPLE = os.path.join(EXAMPLES, "coadd-main-dark-17260-39627574082538900.fits")
DESI_TARGETID = 39627574082538900
DESI_COMPLEXES = ("Halpha", "Hbeta")
HOST_KEYS = ("applied", "n_gal", "host_frac_4200_5000", "reason")
O3_KEYS = ("v_o3", "snr")


def jsonable(v):
    """Python scalars for json; NaN and infinities become null as in pins.json."""
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
    return v


def spectrum_path(fn):
    for d in (EXAMPLES, DATA):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def git_describe(path):
    try:
        out = subprocess.run(
            ["git", "-C", path, "describe", "--tags", "--dirty", "--always"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def produced_by():
    import astropy
    import scipy

    source = os.path.dirname(os.path.dirname(os.path.abspath(blrfit.__file__)))
    return dict(
        code=f"blrfit {blrfit.__version__} (tools/make_pins.py)",
        blrfit=blrfit.__version__,
        git=git_describe(source),
        numpy=np.__version__,
        scipy=scipy.__version__,
        astropy=astropy.__version__,
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        date=datetime.date.today().isoformat(),
        note="summary_row, fitted parameters, chi-square and BIC list of every complex, continuum "
        "parameters, host decision and [O III] pre-fit of fit_spectrum at the pinned redshift, "
        "E(B-V) and complexes; NaN stored as null",
    )


def pin_record(fn, res, z, ebv, complexes, **extra):
    rec = dict(file=fn, kind=extra.pop("kind", "sdss"))
    rec.update(extra)
    rec.update(
        z=float(z),
        complexes=list(complexes),
        ebv=float(ebv),
        summary=jsonable(blrfit.summary_row(res)),
        params={n: jsonable(f["d"]) for n, f in res["fits"].items()},
        chi2={n: float(f["chi2"]) for n, f in res["fits"].items()},
        all_bic={n: [float(b) for b in f["all_bic"]] for n, f in res["fits"].items()},
        conti=jsonable(res["conti"]),
        host_info={k: jsonable(res["host_info"].get(k)) for k in HOST_KEYS},
        o3_prefit={k: jsonable(res["o3_prefit"].get(k)) for k in O3_KEYS},
    )
    return rec


def _brief(rec):
    s = rec["summary"]
    return {n: (s[f"{PREFIX[n]}_class"], s[f"{PREFIX[n]}_c50_sys"]) for n in rec["params"]}


def cmd_fit(a):
    with open(a.legacy) as fh:
        legacy = json.load(fh)["pins"]
    pins = []
    for pin in legacy:
        sp = read_sdss(spectrum_path(pin["file"]))
        res = blrfit.fit_spectrum(
            sp["wave"], sp["flux"], sp["ivar"], pin["z"], ebv=pin["ebv"], complexes=tuple(pin["complexes"])
        )
        pins.append(pin_record(pin["file"], res, pin["z"], pin["ebv"], pin["complexes"]))
        print(pin["file"], _brief(pins[-1]))
    if not a.no_desi:
        tid = int(a.targetid)
        sp = read_desi(a.desi, tid, use_desispec=False)
        z = read_redrock(redrock_sibling(a.desi), tid)["z"]
        res = blrfit.fit_spectrum(
            sp["wave"], sp["flux"], sp["ivar"], z, ebv=sp["ebv"], complexes=DESI_COMPLEXES
        )
        pins.append(
            pin_record(os.path.basename(a.desi), res, z, sp["ebv"], DESI_COMPLEXES, kind="desi", targetid=tid)
        )
        print(os.path.basename(a.desi), _brief(pins[-1]))
    with open(a.out, "w") as fh:
        json.dump(dict(produced_by=produced_by(), pins=pins), fh, indent=1)
        fh.write("\n")
    print(f"wrote {a.out}: {len(pins)} pins")


def _version_label(pin_file, fallback):
    """The blrfit version that wrote a pin file, without a development or
    release-candidate suffix (the column label of the delta table)."""
    made = pin_file.get("produced_by") or {}
    v = str(made.get("blrfit") or fallback)
    return re.sub(r"(\.dev\d+|rc\d+)$", "", v)


def delta_columns(old, new):
    return [
        "spectrum",
        "line",
        f"class_{old}",
        f"class_{new}",
        f"c50_sys_{old}",
        f"c50_sys_{new}",
        "delta_c50",
        f"fwhm_{old}",
        f"fwhm_{new}",
        f"n_broad_{old}",
        f"n_broad_{new}",
        f"chi2_{old}",
        f"chi2_{new}",
        f"bic_margin_{new}",
        f"conti_feop_fwhm_{old}",
        f"conti_feop_fwhm_{new}",
        "conti_at_bound",
        "feuv_fwhm_fixed",
        f"source_{old}",
    ]


def _num(v, fmt="{:.3f}"):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def cmd_deltas(a):
    legacy, old_label = {}, a.old_label
    for path in a.legacy:
        with open(path) as fh:
            content = json.load(fh)
        if old_label is None:
            old_label = _version_label(content, "old")
        for pin in content["pins"]:
            legacy.setdefault(pin["file"], (pin, os.path.basename(path)))
    with open(a.current) as fh:
        content = json.load(fh)
    current = content["pins"]
    new_label = a.new_label or _version_label(content, "new")
    old, new = old_label, new_label
    rows = []
    for pin in current:
        prev, src = legacy.get(pin["file"], (None, ""))
        s = pin["summary"]
        for name in [n for n in ("Halpha", "Hbeta", "MgII") if n in pin["params"]]:
            p = PREFIX[name]
            o = prev["summary"] if prev is not None and name in prev["params"] else {}
            c50_old, c50_new = o.get(f"{p}_c50_sys"), s.get(f"{p}_c50_sys")
            delta = c50_new - c50_old if c50_old is not None and c50_new is not None else None
            rows.append(
                {
                    "spectrum": pin["file"],
                    "line": name,
                    f"class_{old}": o.get(f"{p}_class", ""),
                    f"class_{new}": s.get(f"{p}_class", ""),
                    f"c50_sys_{old}": _num(c50_old),
                    f"c50_sys_{new}": _num(c50_new),
                    "delta_c50": _num(delta),
                    f"fwhm_{old}": _num(o.get(f"{p}_fwhm")),
                    f"fwhm_{new}": _num(s.get(f"{p}_fwhm")),
                    f"n_broad_{old}": "" if o.get(f"{p}_n_broad") is None else int(o[f"{p}_n_broad"]),
                    f"n_broad_{new}": "" if s.get(f"{p}_n_broad") is None else int(s[f"{p}_n_broad"]),
                    f"chi2_{old}": _num((prev or {}).get("chi2", {}).get(name)),
                    f"chi2_{new}": _num(pin["chi2"].get(name)),
                    f"bic_margin_{new}": _num(s.get(f"{p}_bic_margin")),
                    f"conti_feop_fwhm_{old}": _num(o.get("conti_feop_fwhm"), "{:.2f}"),
                    f"conti_feop_fwhm_{new}": _num(s.get("conti_feop_fwhm"), "{:.2f}"),
                    "conti_at_bound": s.get("conti_at_bound", ""),
                    "feuv_fwhm_fixed": s.get("conti_feuv_fwhm_fixed", ""),
                    f"source_{old}": src if prev is not None else "",
                }
            )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=delta_columns(old, new))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}: {len(rows)} rows")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the pinned spectra and write a pin file")
    f.add_argument("--out", required=True)
    f.add_argument(
        "--legacy",
        default=os.path.join(DATA, "pins.json"),
        help="pin file whose SDSS spectra, redshifts, E(B-V) and complexes are refitted",
    )
    f.add_argument("--desi", default=DESI_EXAMPLE, help="DESI coadd with its redrock file next to it")
    f.add_argument("--targetid", default=DESI_TARGETID, type=int)
    f.add_argument("--no-desi", action="store_true")
    f.set_defaults(func=cmd_fit)
    d = sub.add_parser("deltas", help="write the per-spectrum, per-line delta table")
    d.add_argument("--legacy", action="append", required=True, help="old pin file(s), first match wins")
    d.add_argument("--current", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--old-label", help="column label of the old pins (default: their blrfit version)")
    d.add_argument("--new-label", help="column label of the current pins (default: their blrfit version)")
    d.set_defaults(func=cmd_deltas)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
