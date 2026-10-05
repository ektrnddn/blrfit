"""
Attribute the change of a fit between the 0.1.0 configuration and the current
tree to the corrections, by controlled refits.

    PYTHONPATH=. python tools/attribute_deltas.py --pins tests/data/pins_0.3.0.json \\
        --legacy tests/data/pins.json --out docs/attribution_pins.csv

Baseline and toggles. The baseline is the 0.1.0 configuration reproduced
inside the current tree: every correction that has a switch is turned off.
Each correction is a toggle with an off value (the 0.1.0 behaviour) and an on
value (the current default of ``fit_spectrum``). A toggle is applied through a
keyword of ``fit_spectrum`` when the tree has it; two toggles have a scoped
patch of the module that implements the correction for trees without the
keyword:

  fe_operator         the continuous Fe II broadening (on) against the
                      historical operator that rounded the width to 50 km/s
                      inside the solver (off; the operator restored by
                      tests/test_pins.py). Patched on FeTemplate.broadened
                      when fit_spectrum has no ``fe_operator`` keyword.
  fe_uv_width_policy  the ultraviolet Fe II width policy: the current default
                      (on) against policy B, the width always free (off; the
                      0.1.0 parametrisation, in which the width stayed at its
                      3000 km/s start under the historical operator). Patched
                      through FE_UV_FREE_MIN_PIXELS = 0 when fit_spectrum has
                      no ``fe_uv_width_policy`` keyword.
  host_guard          the host-fraction guard (on) against the 0.1.0 host
                      decision (off). Keyword only; it changes degenerate
                      spectra only.
  conti_multistart    the continuum started from several points (on, since
                      0.3.0) against the single start of 0.1.0 and 0.2.0
                      (off). Keyword only.

A toggle whose keyword the tree does not have and that has no patch is
reported as unavailable and left out of the search. The corrections without a
switch are part of both configurations: the input and solver guards of 0.2.0,
which leave every spectrum that fitted before unchanged, and the at-bound
bookkeeping of the Fe II widths, which records the solver's criterion and the
span-based one side by side in every fit and changes no fitted value. On the
reference stack the baseline reproduces every entry of the summary row of the
0.1.0 pins bit for bit; ``--legacy`` checks a run against a legacy pin file.

Search. Per spectrum: the current fit (every toggle on) and the baseline; then
each single toggle from the baseline; then, for a line that no single toggle
reproduces, every pair; then the complement path (every toggle on, one off),
which checks that the attribution does not depend on the order in which the
toggles are applied. "Reproduces" means the same class, flags and component
count and |delta c50_sys| <= TOL_KMS (1.5 km/s, the rule of the confidence
intervals) against the current fit. Outcomes, per spectrum and line:

  unchanged          the baseline reproduces the current fit
  attributed-single  a single toggle reproduces it
  attributed-joint   a pair reproduces it and no single toggle does
  numerical          no toggle set reproduces it, but a perturbation of the
                     solver start (the broad-component starting velocities
                     moved by PERTURB_KMS, or the flux rescaled by 1 + 1e-13)
                     carries the baseline to the current end point or the
                     current configuration to the baseline's, or
                     |bic_margin| < EDGE_BIC under every toggle set
  unresolved         none of the above

``edge`` (|bic_margin| < EDGE_BIC under every toggle set) is reported with
every outcome as a sensitivity modifier, never as a cause. The effect size of
each toggle is the change of c50_sys it produces alone from the baseline
(``dc50_<toggle>``), with the class, component-count and flag changes it
produces (``change_<toggle>``). ``complement_necessary`` lists the toggles
whose removal from the current configuration breaks the reproduction;
``order_independent`` says whether that set equals the attributed one.

Output: one row per spectrum and fitted line (``--out``), with the outcome,
the attributed toggles, the effect sizes, class, flags, component count,
c50_sys and bic_margin of both configurations, the legacy check and the
number of fits; ``--json`` writes every fitted configuration of every
spectrum. The spectra come from a pin file (``--pins``; SDSS entries and DESI
entries with a target identifier, as tools/make_pins.py writes them) or a CSV
list (``--list``: columns path, z, and optionally ebv, complexes as
"Halpha+Hbeta", targetid).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import itertools
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d

import blrfit
from blrfit.constants import S2F
from blrfit.io import read_desi, read_sdss
from blrfit.io.desi import read_redrock, redrock_sibling
from blrfit.model import broad as broad_module
from blrfit.model import continuum as continuum_module
from blrfit.model.continuum import FeTemplate
from blrfit.model.fit import PREFIX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
EXAMPLES = os.path.join(ROOT, "examples", "data")

TOL_KMS = 1.5  # |delta c50_sys| within which two fits agree (the confidence-interval rule)
EDGE_BIC = 5.0  # |bic_margin| below which a decomposition is on the edge
PERTURB_KMS = 1.0  # start perturbation: every broad starting velocity moved by this
LASTBIT_EPS = 1e-13  # last-bit perturbation: flux rescaled by 1 + this (tests/test_reproducibility.py)
LINES = ("Halpha", "Hbeta", "MgII")
OUTCOMES = ("unchanged", "attributed-single", "attributed-joint", "numerical", "unresolved")


# ----------------------------------------------------------------------------
# Toggles
# ----------------------------------------------------------------------------
def _historical_broadening(self, width):
    """The Fe II operator of 0.1.0: the width rounded to 50 km/s inside the solver."""
    f = int(round(max(width, self.intrinsic + 10.0) / 50.0)) * 50.0
    sigma = np.sqrt(max(f**2 - self.intrinsic**2, 100.0)) / S2F / self.pix_kms
    return gaussian_filter1d(self.flux, sigma, mode="nearest")


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _patch_historical_operator():
    return _patched(FeTemplate, "broadened", _historical_broadening)


def _patch_uv_width_free():
    return _patched(continuum_module, "FE_UV_FREE_MIN_PIXELS", 0)


class Toggle:
    """One correction: its keyword in fit_spectrum, the off value (0.1.0) and,
    where the tree has no keyword, a patch that turns the correction off."""

    def __init__(self, name, off, patch=None, doc=""):
        self.name, self.off, self.patch, self.doc = name, off, patch, doc

    def keyword(self):
        return self.name in inspect.signature(blrfit.fit_spectrum).parameters

    def on(self):
        return (
            inspect.signature(blrfit.fit_spectrum).parameters[self.name].default
            if self.keyword()
            else "default"
        )

    def available(self):
        return self.keyword() or self.patch is not None


TOGGLES = [
    Toggle(
        "fe_operator",
        off="historical",
        patch=_patch_historical_operator,
        doc="continuous Fe II broadening against the historical 50 km/s operator",
    ),
    Toggle(
        "fe_uv_width_policy",
        off="B",
        patch=_patch_uv_width_free,
        doc="ultraviolet Fe II width policy against policy B (always free)",
    ),
    Toggle("host_guard", off=False, doc="host-fraction guard against the 0.1.0 host decision"),
    Toggle("conti_multistart", off=False, doc="continuum started from several points against one start"),
]


def available_toggles(names=None):
    """The toggles the tree can apply, in the order of TOGGLES; ``names``
    restricts them. Returns (available, unavailable names)."""
    wanted = [t for t in TOGGLES if names is None or t.name in names]
    return [t for t in wanted if t.available()], [t.name for t in wanted if not t.available()]


@contextlib.contextmanager
def configured(toggles, on, perturb=None):
    """Keyword arguments of fit_spectrum for the toggles in ``on`` (the others
    off), with the patches of the toggles that have no keyword applied while
    the context is open; ``perturb`` 'start' moves every broad starting
    velocity by PERTURB_KMS."""
    kw = {}
    with contextlib.ExitStack() as stack:
        for t in toggles:
            if t.keyword():
                if t.name not in on:
                    kw[t.name] = t.off
            elif t.name not in on:
                stack.enter_context(t.patch())
        if perturb == "start":
            starts = tuple(v + PERTURB_KMS for v in broad_module.BROAD_STARTS_KMS)
            stack.enter_context(_patched(broad_module, "BROAD_STARTS_KMS", starts))
        yield kw


# ----------------------------------------------------------------------------
# One fit, reduced to what the attribution compares
# ----------------------------------------------------------------------------
def line_record(row, name):
    """Class, flags, component count, c50_sys and bic_margin of one line of a
    summary row (None when the line was not fitted)."""
    p = PREFIX[name]
    if f"{p}_class" not in row:
        return None
    flags = row.get(f"{p}_flags", "")
    return dict(
        cls=row[f"{p}_class"],
        flags=tuple(f for f in flags.split(",") if f),
        n_broad=int(row[f"{p}_n_broad"]) if row.get(f"{p}_n_broad") is not None else -1,
        c50=float(_nan(row.get(f"{p}_c50_sys"))),
        bic_margin=float(_nan(row.get(f"{p}_bic_margin"))),
    )


def _nan(v):
    return np.nan if v is None else v


def record_of(res):
    """The per-line records and the continuum state of a fit result."""
    row = blrfit.summary_row(res)
    lines = {n: line_record(row, n) for n in LINES}
    conti = dict(
        feop_fwhm=float(res["conti"].get("feop_fwhm", np.nan)),
        feuv_fwhm=float(res["conti"].get("feuv_fwhm", np.nan)),
        at_bound=row.get("conti_at_bound", ""),
        feuv_fwhm_fixed=row.get("conti_feuv_fwhm_fixed", ""),
        host_applied=row.get("host_applied", ""),
        continuum_status=row.get("continuum_status", ""),
    )
    settings = {
        k: v
        for k, v in res.get("settings", {}).items()
        if isinstance(v, (str, bool, int, float)) and k not in ("nmc", "seed")
    }
    return dict(lines={n: r for n, r in lines.items() if r is not None}, conti=conti, settings=settings)


def fit_record(sp, z, ebv, complexes, toggles, on, perturb=None):
    """Fit one spectrum under a toggle configuration and reduce the result."""
    flux, ivar = np.asarray(sp["flux"], float), np.asarray(sp["ivar"], float)
    if perturb == "lastbit":
        flux, ivar = flux * (1.0 + LASTBIT_EPS), ivar / (1.0 + LASTBIT_EPS) ** 2
    with configured(toggles, on, perturb) as kw:
        res = blrfit.fit_spectrum(sp["wave"], flux, ivar, z, ebv=ebv, complexes=tuple(complexes), **kw)
    return record_of(res)


# ----------------------------------------------------------------------------
# The search
# ----------------------------------------------------------------------------
def same(a, b, tol=TOL_KMS):
    """Two line records agree: the same class, flags and component count and
    c50_sys within ``tol`` km/s (both NaN counts as agreement); a line fitted
    in one configuration only does not."""
    if a is None or b is None:
        return a is None and b is None
    if (a["cls"], a["flags"], a["n_broad"]) != (b["cls"], b["flags"], b["n_broad"]):
        return False
    if np.isnan(a["c50"]) and np.isnan(b["c50"]):
        return True
    return bool(abs(a["c50"] - b["c50"]) <= tol)


def change_string(a, b):
    """What differs between two line records, compactly: 'A>C', 'n1>2',
    '+poor_fit', '-low_snr'."""
    if a is None or b is None:
        return "" if a is b else ("fitted" if a is None else "unfitted")
    parts = []
    if a["cls"] != b["cls"]:
        parts.append(f"{a['cls']}>{b['cls']}")
    if a["n_broad"] != b["n_broad"]:
        parts.append(f"n{a['n_broad']}>{b['n_broad']}")
    parts += [f"+{f}" for f in b["flags"] if f not in a["flags"]]
    parts += [f"-{f}" for f in a["flags"] if f not in b["flags"]]
    return ";".join(parts)


def config_key(on, perturb=None):
    key = "+".join(sorted(on)) or "baseline"
    return f"{key}|{perturb}" if perturb else key


class Search:
    """The refits of one spectrum, cached by configuration. ``fit(on, perturb)``
    returns the reduced record of the fit with the toggles in ``on`` switched
    on and the others off."""

    def __init__(self, fit, names):
        self._fit, self.names = fit, list(names)
        self.records = {}

    def fit(self, on, perturb=None):
        key = config_key(on, perturb)
        if key not in self.records:
            self.records[key] = self._fit(frozenset(on), perturb)
        return self.records[key]

    @property
    def n_fits(self):
        return len(self.records)

    def line(self, on, name, perturb=None):
        return self.fit(on, perturb)["lines"].get(name)


def attribute_line(search, name, tol=TOL_KMS, edge=EDGE_BIC, perturb=True):
    """Attribute the change of one line between the baseline and the current
    configuration; see the module docstring for the outcomes. Returns None
    when the current fit does not fit the line."""
    names = search.names
    every = frozenset(names)
    cur, base = search.line(every, name), search.line(frozenset(), name)
    if cur is None:
        return None
    out = dict(
        outcome="unresolved",
        toggles=[],
        reproducing_singles=[],
        reproducing_pairs=[],
        complement_necessary=[],
        order_independent="",
        perturbation="",
        edge=False,
        effect={},
        change={},
        base=base,
        cur=cur,
    )
    for t in names:
        rec = search.line(frozenset([t]), name)
        out["effect"][t] = (rec["c50"] - base["c50"]) if (rec is not None and base is not None) else np.nan
        out["change"][t] = change_string(base, rec)
        if same(rec, cur, tol):
            out["reproducing_singles"].append(t)
    if same(base, cur, tol):
        out["outcome"] = "unchanged"
    elif out["reproducing_singles"]:
        out["outcome"] = "attributed-single"
        out["toggles"] = list(out["reproducing_singles"])
    else:
        for a, b in itertools.combinations(names, 2):
            if same(search.line(frozenset([a, b]), name), cur, tol):
                out["reproducing_pairs"].append((a, b))
        if out["reproducing_pairs"]:
            out["outcome"] = "attributed-joint"
            out["toggles"] = list(out["reproducing_pairs"][0])
    # the complement path: every toggle on, one off
    for t in names:
        if not same(search.line(every - {t}, name), cur, tol):
            out["complement_necessary"].append(t)
    if out["outcome"] in ("attributed-single", "attributed-joint"):
        out["order_independent"] = set(out["complement_necessary"]) == set(out["toggles"])
    # the edge: |bic_margin| below EDGE_BIC under every toggle set fitted so far
    margins = [
        r["lines"][name]["bic_margin"]
        for k, r in search.records.items()
        if "|" not in k and name in r["lines"]
    ]
    out["edge"] = bool(margins) and all(np.isfinite(m) and abs(m) < edge for m in margins)
    if out["outcome"] == "unresolved":
        if out["edge"]:
            out["outcome"] = "numerical"
            out["perturbation"] = "edge"
        elif perturb:
            for kind in ("start", "lastbit"):
                if same(search.line(frozenset(), name, kind), cur, tol):
                    out["outcome"] = "numerical"
                    out["perturbation"] = f"baseline+{kind}"
                    break
                if same(search.line(every, name, kind), base, tol):
                    out["outcome"] = "numerical"
                    out["perturbation"] = f"current+{kind}"
                    break
    return out


# ----------------------------------------------------------------------------
# Spectra and the legacy check
# ----------------------------------------------------------------------------
def spectrum_path(fn):
    if os.path.exists(fn):
        return fn
    for d in (EXAMPLES, DATA):
        p = os.path.join(d, os.path.basename(fn))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def read_pinned(entry):
    """The spectrum of one pin-file entry (SDSS, or DESI at its target identifier)."""
    path = spectrum_path(entry["file"])
    if entry.get("kind") == "desi":
        return read_desi(path, int(entry["targetid"]), use_desispec=False)
    return read_sdss(path)


def jobs_from_pins(path):
    with open(path) as fh:
        pins = json.load(fh)["pins"]
    return [
        dict(
            file=p["file"],
            kind=p.get("kind", "sdss"),
            targetid=p.get("targetid"),
            z=float(p["z"]),
            ebv=float(p.get("ebv", 0.0)),
            complexes=list(p["complexes"]),
        )
        for p in pins
    ]


def jobs_from_list(path):
    """A CSV list: path, z, and optionally ebv, complexes ('Halpha+Hbeta'), targetid."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            tid = r.get("targetid")
            comp = r.get("complexes") or "Halpha+Hbeta"
            z = r.get("z", "")
            job = dict(
                file=r["path"],
                kind="desi" if tid else "sdss",
                targetid=int(tid) if tid else None,
                z=float(z) if z else np.nan,
                ebv=float(r.get("ebv") or 0.0),
                complexes=[c for c in comp.replace(",", "+").split("+") if c],
            )
            out.append(job)
    return out


# Diagnostic flags introduced in 0.3.0 without a switch: a legacy pin cannot hold them, so the
# check of the baseline against a legacy summary leaves them out (the attribution itself keeps them)
FLAGS_SINCE_0_3 = frozenset({"degenerate"})


def _without_flags(rec, flags):
    return None if rec is None else dict(rec, flags=tuple(f for f in rec["flags"] if f not in flags))


def legacy_summaries(path):
    with open(path) as fh:
        return {p["file"]: p["summary"] for p in json.load(fh)["pins"]}


def legacy_check(base, summary, tol=TOL_KMS):
    """Whether the baseline reproduces a legacy summary row: the same classes,
    flags (those of FLAGS_SINCE_0_3 left out) and component counts and c50_sys
    within ``tol``. Returns (verdict, largest |delta c50_sys|)."""
    departures, dmax = [], 0.0
    for name in LINES:
        ref = line_record(summary, name)
        got = _without_flags(base["lines"].get(name), FLAGS_SINCE_0_3)
        if ref is None and got is None:
            continue
        if ref is not None and got is not None and np.isfinite(ref["c50"]) and np.isfinite(got["c50"]):
            dmax = max(dmax, abs(got["c50"] - ref["c50"]))
        if not same(got, ref, tol):
            departures.append(f"{name} {change_string(ref, got) or 'c50'}")
    return ("reproduced" if not departures else "differs: " + "; ".join(departures)), dmax


# ----------------------------------------------------------------------------
# One spectrum end to end
# ----------------------------------------------------------------------------
def attribute_spectrum(job, toggle_names=None, tol=TOL_KMS, edge=EDGE_BIC, perturb=True, legacy=None):
    """Fit one spectrum under every configuration the search needs and return
    (rows, configurations): one row per fitted line and the reduced record of
    every configuration fitted."""
    toggles, unavailable = available_toggles(toggle_names)
    names = [t.name for t in toggles]
    sp = read_pinned(job)
    z = job["z"] if np.isfinite(job.get("z", np.nan)) else float(sp["z"])
    if job.get("kind") == "desi" and not np.isfinite(job.get("z", np.nan)):
        rr = redrock_sibling(spectrum_path(job["file"]))
        if rr is not None:
            z = float(read_redrock(rr, job["targetid"])["z"])
    ebv = job.get("ebv", 0.0)
    if job.get("kind") == "desi" and "ebv" in sp and not job.get("ebv"):
        ebv = float(sp["ebv"])
    search = Search(lambda on, pert: fit_record(sp, z, ebv, job["complexes"], toggles, on, pert), names)
    results = {name: attribute_line(search, name, tol=tol, edge=edge, perturb=perturb) for name in LINES}
    base = search.fit(frozenset())
    verdict, dmax = ("", np.nan)
    if legacy is not None and os.path.basename(job["file"]) in legacy:
        verdict, dmax = legacy_check(base, legacy[os.path.basename(job["file"])], tol)
    rows = []
    for name, r in results.items():
        if r is None:
            continue
        b, c = r["base"], r["cur"]
        row = dict(
            spectrum=os.path.basename(job["file"]),
            line=name,
            z=z,
            outcome=r["outcome"],
            toggles="+".join(r["toggles"]),
            reproducing_singles="|".join(r["reproducing_singles"]),
            reproducing_pairs="|".join("+".join(p) for p in r["reproducing_pairs"]),
            complement_necessary="+".join(r["complement_necessary"]),
            order_independent=r["order_independent"],
            edge=r["edge"],
            perturbation=r["perturbation"],
            class_base=b["cls"] if b else "",
            class_cur=c["cls"],
            flags_base=",".join(b["flags"]) if b else "",
            flags_cur=",".join(c["flags"]),
            n_broad_base=b["n_broad"] if b else "",
            n_broad_cur=c["n_broad"],
            c50_base=b["c50"] if b else np.nan,
            c50_cur=c["c50"],
            delta_c50=(c["c50"] - b["c50"]) if b else np.nan,
            bic_margin_base=b["bic_margin"] if b else np.nan,
            bic_margin_cur=c["bic_margin"],
        )
        for t in names:
            row[f"dc50_{t}"] = r["effect"][t]
            row[f"change_{t}"] = r["change"][t]
        row.update(
            baseline_vs_legacy=verdict,
            legacy_dc50=dmax,
            n_fits=search.n_fits,
            toggles_available="+".join(names),
            toggles_unavailable="+".join(unavailable),
        )
        rows.append(row)
    return rows, search.records


def columns(names):
    fixed = [
        "spectrum",
        "line",
        "z",
        "outcome",
        "toggles",
        "reproducing_singles",
        "reproducing_pairs",
        "complement_necessary",
        "order_independent",
        "edge",
        "perturbation",
        "class_base",
        "class_cur",
        "flags_base",
        "flags_cur",
        "n_broad_base",
        "n_broad_cur",
        "c50_base",
        "c50_cur",
        "delta_c50",
        "bic_margin_base",
        "bic_margin_cur",
    ]
    per_toggle = [c for t in names for c in (f"dc50_{t}", f"change_{t}")]
    return (
        fixed
        + per_toggle
        + ["baseline_vs_legacy", "legacy_dc50", "n_fits", "toggles_available", "toggles_unavailable"]
    )


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else f"{v:.3f}"
    return "" if v is None else str(v)


def write_rows(rows, names, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns(names))
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in columns(names)})


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def _run_job(args):
    job, opts = args
    return job, attribute_spectrum(job, **opts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pins", help="pin file whose spectra are attributed (tools/make_pins.py layout)")
    src.add_argument("--list", help="CSV list of spectra: path, z[, ebv, complexes, targetid]")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these file names")
    ap.add_argument("--legacy", help="legacy pin file (0.1.0) against which the baseline is checked")
    ap.add_argument("--out", required=True, help="CSV, one row per spectrum and line")
    ap.add_argument("--json", help="every fitted configuration of every spectrum")
    ap.add_argument("--toggles", help="comma-separated subset of the toggles")
    ap.add_argument("--tol", type=float, default=TOL_KMS, help="|delta c50_sys| of a reproduction, km/s")
    ap.add_argument("--edge", type=float, default=EDGE_BIC, help="|bic_margin| of an edge")
    ap.add_argument("--no-perturb", action="store_true", help="skip the start perturbations")
    ap.add_argument("--nproc", type=int, default=1)
    a = ap.parse_args(argv)

    jobs = jobs_from_pins(a.pins) if a.pins else jobs_from_list(a.list)
    if a.only:
        keep = {os.path.basename(x) for x in a.only}
        jobs = [j for j in jobs if os.path.basename(j["file"]) in keep]
    names_wanted = [s for s in a.toggles.split(",") if s] if a.toggles else None
    toggles, unavailable = available_toggles(names_wanted)
    names = [t.name for t in toggles]
    print(
        f"toggles: {', '.join(names) or 'none'}"
        + (f"; unavailable: {', '.join(unavailable)}" if unavailable else "")
    )
    for t in toggles:
        print(f"  {t.name}: off={t.off!r} on={t.on()!r} via {'keyword' if t.keyword() else 'patch'}; {t.doc}")
    legacy = legacy_summaries(a.legacy) if a.legacy else None
    opts = dict(toggle_names=names_wanted, tol=a.tol, edge=a.edge, perturb=not a.no_perturb, legacy=legacy)
    rows, dump = [], []
    if a.nproc > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(a.nproc) as pool:
            done = pool.imap(_run_job, [(j, opts) for j in jobs])
            results = list(done)
    else:
        results = [_run_job((j, opts)) for j in jobs]
    for job, (jrows, records) in results:
        rows += jrows
        dump.append(
            dict(
                file=os.path.basename(job["file"]),
                z=jrows[0]["z"] if jrows else job["z"],
                rows=jrows,
                configurations=records,
            )
        )
        for r in jrows:
            print(
                f"{r['spectrum']} {r['line']}: {r['outcome']} {r['toggles']} "
                f"dc50={_fmt(r['delta_c50'])} edge={r['edge']} fits={r['n_fits']} {r['baseline_vs_legacy']}"
            )
    write_rows(rows, names, a.out)
    counts = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
    print(f"wrote {a.out}: {len(rows)} rows; " + ", ".join(f"{o} {n}" for o, n in counts.items() if n))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(
                _jsonable(
                    dict(toggles=names, unavailable=unavailable, tol_kms=a.tol, edge_bic=a.edge, spectra=dump)
                ),
                fh,
                indent=1,
            )
            fh.write("\n")
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
