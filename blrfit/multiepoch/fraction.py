"""Upper limit on the fraction of strong offsets that are single-lined binaries, against the orbital period, in the
framework of the multi epoch verdicts (the logic is that of Shen et al. 2013 and Wang et al. 2017).

Sensitivity. For every testable strong offset with a virial mass, a binary of rest-frame period P is drawn many times:
mass ratio q_a = M_active / M_companion log-uniform in 0.1-1, isotropic inclination, the broad line region inside the
active hole's Roche lobe (a f_R(q_a) >= R_BLR), and the coadd offset as the orbital velocity at the DESI epoch
(K >= |dv|, the phase set by K sin phi = dv, either of the two solutions). The orbit gives the change d0 of every pair
the object's Halpha motion test uses. With the full noise model the test's chi2 difference is then a noncentral chi2
with noncentrality d0^T Q d0 (``verdicts.motion_matrix``), so the probability that the object would be called moving follows
exactly for Gaussian noise, without simulating the noise. S(P) is the summed probability over a set of objects: the
number of movers a population made entirely of such binaries would give.

Limit. In each of the matchings of ``population`` (strong offsets against objects with |dv| < 1000 km/s alike in FWHM,
S/N, number of times and span), keeping the pairs whose candidate has a mass: the one-sided 95% upper bound of the excess
of movers, (b - c)/n + 1.645 sqrt(b + c - (b - c)^2/n) / n for b candidate-only and c control-only movers among n pairs,
times n, over the summed sensitivity of those candidates: f95(P) = excess95 / S(P), capped at 1. Reported: the median
over the matchings. The BLR radius enters only through the Roche condition; the runs with R_BLR halved and doubled
bracket that systematic.

    blrfit multiepoch fraction --run DIR --pairs PAIRS.csv --catalog CATALOG --masses MASSES
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, ncx2

from . import common as K
from . import population as PO
from . import spectra as SP
from . import verdicts as V

G = 6.67430e-11
MSUN = 1.98892e30
LTD = 2.99792458e8 * 86400.0
YR = 3.15576e7
PERIODS = np.unique(np.concatenate([np.geomspace(5, 10000, 34), [10, 20, 50, 100, 200, 500, 1000]])).round(
    1
)  # rest-frame years
KEY_M = "matched strong offsets, excess over the reference set, offset orbital"
KEY_ALL = "strong offsets (|dv| >= 1000), offset orbital"


def roche_frac(q):
    q = np.asarray(q, float)
    return np.where(q > 1.0 / 1.88, 0.38 + 0.2 * np.log10(q), 0.46224 * (q / (1 + q)) ** (1.0 / 3.0))


def object_model(keep):
    """From the internals of ``verdicts.analyze``: the spectra of the motion test, their rest-frame times, the DESI epoch,
    W = M^T Q M (noncentrality = v^T W v for spectrum velocities v) and the degrees of freedom."""
    sub = keep["sub"]
    Q, dof = V.motion_matrix(
        sub, keep["sig"], keep["cl_of"], keep["t_rest"], keep["omega"], keep["inst"], keep["blr"]
    )
    if Q is None:
        return None
    spec = sorted({p["a"] for p in sub} | {p["b"] for p in sub})
    si = {k: i for i, k in enumerate(spec)}
    M = np.zeros((len(sub), len(spec)))
    for j, p in enumerate(sub):
        M[j, si[p["b"]]] += 1.0
        M[j, si[p["a"]]] -= 1.0
    t = np.array([keep["t_rest"][k] for k in spec])
    desi = [keep["t_rest"][k] for k in spec if keep["inst"][k] == "DESI"]
    return dict(t=t, t_desi=float(np.mean(desi)) if desi else float(t.max()), W=M.T @ Q @ M, dof=dof)


def detection(model, m_act, r_blr_ltd, dv0, periods, draws):
    """The probability, per period, that the object would be called moving if it were a binary of that period."""
    q, sini, branch = draws
    thr = chi2.isf(V.P_MOVE, model["dof"])
    m_tot = m_act * (1 + 1 / q) * MSUN
    dt = model["t"] - model["t_desi"]
    out = np.zeros(periods.size)
    for k, P in enumerate(periods):
        a = (G * m_tot * (P * YR) ** 2 / (4 * np.pi**2)) ** (1 / 3)
        K_ = (2 * np.pi * G * m_tot / (P * YR)) ** (1 / 3) * sini / (1 + q) / 1e3
        ok = (a * roche_frac(q) >= r_blr_ltd * LTD) & (K_ >= abs(dv0))
        if not ok.any():
            continue
        s0 = np.clip(dv0 / np.where(K_ > 0, K_, 1.0), -1, 1)
        ph = np.where(branch, np.arcsin(s0), np.pi - np.arcsin(s0))
        v = K_[:, None] * np.sin(ph[:, None] + 2 * np.pi * dt[None, :] / P)
        lam = np.einsum("ij,jk,ik->i", v, model["W"], v)
        p = np.where(
            lam > 1e-9, ncx2.sf(thr, model["dof"], np.maximum(lam, 1e-9)), chi2.sf(thr, model["dof"])
        )
        out[k] = float(np.mean(np.where(ok, p, 0.0)))
    return out


def run_fraction(run, inputs, ndraw=2000, seeds=200, seed=20261015, verbose=True):
    """The binary-fraction limit from the verdicts of ``run`` (verdicts.csv, objects.csv, CALIBRATION.json), written to
    BINARY_FRACTION.json / .md (and the R_BLR x0.5 / x2 variants) in ``run``. Returns the results by R_BLR factor."""
    a = _Args(run=str(run), ndraw=ndraw, seeds=seeds, seed=seed)
    run = Path(a.run)
    rows = inputs.load_pairs()
    C = inputs.load_catalog()
    okey = K.object_key(C)
    spec = SP.check(rows, okey)
    fwhm = SP.object_fwhm(spec)
    calib = V.Calibration(run / "CALIBRATION.json")
    masses = inputs.load_masses()
    cat, st = PO.load(run, C, okey)
    byobj = K.by_object(rows, okey)
    members = K.members_of(rows, okey)
    cand = [o for o in st if PO.STRONG(cat[o])]
    pool = [o for o in st if PO.BELOW_1000(cat[o]) and o not in cand]
    # sensitivity of every strong offset with a mass
    sens = {}
    for o in cand:
        mass = K.mass_of(o, masses, members)
        if (
            not mass
            or not np.isfinite(mass.get("LOGMBH", np.nan))
            or not np.isfinite(mass.get("R_BLR_LTD", np.nan))
        ):
            continue
        keep = {}
        V.analyze(o, "Halpha", byobj[o], spec, calib, mass, cat[o]["z"], keep=keep, fwhm=fwhm)
        if "sub" not in keep:
            continue
        model = object_model(keep)
        if model is None:
            continue
        rng = np.random.default_rng([a.seed, int(o) % (2**62)])
        q = 10 ** rng.uniform(-1, 0, a.ndraw)
        sini = np.sqrt(1 - rng.uniform(0, 1, a.ndraw) ** 2)
        draws = (q, sini, rng.random(a.ndraw) < 0.5)
        sens[o] = {
            x: detection(model, 10 ** mass["LOGMBH"], mass["R_BLR_LTD"] * x, cat[o]["dv"], PERIODS, draws)
            for x in (1.0, 0.5, 2.0)
        }
    if verbose:
        print(f"{len(cand)} testable strong offsets, {len(sens)} with a mass and a motion test", flush=True)
    key = PO.stratum(cat, st)
    res = {}
    for x, tag in ((1.0, ""), (0.5, "_rblr_x0.5"), (2.0, "_rblr_x2")):
        f95s, Ss, ex, pcs, prs, ns = [], [], [], [], [], []
        for pairs in PO.matchings(cand, pool, key, a.seeds):
            pairs = [(c, r) for c, r in pairs if c in sens]
            n = len(pairs)
            b = sum(st[c]["move"] and not st[r]["move"] for c, r in pairs)
            c_ = sum(st[r]["move"] and not st[c]["move"] for c, r in pairs)
            d = (b - c_) / n
            up = d + 1.645 * np.sqrt(max(b + c_ - (b - c_) ** 2 / n, 0.0)) / n
            S = np.sum([sens[c][x] for c, _ in pairs], axis=0)
            e95 = max(up, 0.0) * n
            f95s.append(np.minimum(1.0, e95 / np.where(S > 0, S, np.nan)))
            Ss.append(S)
            ex.append(e95)
            ns.append(n)
            pcs.append(np.mean([st[c]["move"] for c, _ in pairs]))
            prs.append(np.mean([st[r]["move"] for _, r in pairs]))
        f95 = np.nanmedian(np.array(f95s), axis=0)
        S_all = np.sum([v[x] for v in sens.values()], axis=0)
        out = {
            KEY_M: dict(
                n_objects=int(np.median(ns)),
                periods=PERIODS.tolist(),
                S=np.median(np.array(Ss), axis=0).tolist(),
                f95=f95.tolist(),
                f95_16_84=[
                    np.nanpercentile(np.array(f95s), 16, axis=0).tolist(),
                    np.nanpercentile(np.array(f95s), 84, axis=0).tolist(),
                ],
                p_cand=float(np.median(pcs)),
                p_ref=float(np.median(prs)),
                excess95=float(np.median(ex)),
                matchings=a.seeds,
            ),
            KEY_ALL: dict(n_objects=len(sens), periods=PERIODS.tolist(), S=S_all.tolist()),
        }
        (run / f"BINARY_FRACTION{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
        res[x] = out
        if verbose:
            print(
                f"R_BLR x{x}: matched {out[KEY_M]['n_objects']} (median), movers {out[KEY_M]['p_cand']:.3f} vs {out[KEY_M]['p_ref']:.3f}, excess95 {out[KEY_M]['excess95']:.1f}; "
                "f95 at 20/50/100/200/500 yr: "
                + ", ".join(f"{v:.2f}" for v in np.interp([20, 50, 100, 200, 500], PERIODS, f95))
                + "; S/n at 100 yr: "
                + f"{np.interp(100, PERIODS, out[KEY_M]['S']) / out[KEY_M]['n_objects']:.3f}",
                flush=True,
            )
    m = res[1.0][KEY_M]
    show = [20, 50, 70, 100, 150, 200, 300, 500]
    L = [
        "# The binary fraction the multi epoch verdicts allow",
        "",
        __doc__.split("\n\n")[1].replace("\n", " "),
        "",
        __doc__.split("\n\n")[2].replace("\n", " "),
        "",
        f"Matched candidates with a mass (median over {a.seeds} matchings): {m['n_objects']}; moving {100 * m['p_cand']:.1f}% against {100 * m['p_ref']:.1f}% "
        f"of their matches; one-sided 95% bound of the excess {m['excess95']:.1f} objects. All strong offsets with a mass and a motion test: {len(sens)}.",
        "",
        "| P (yr) | " + " | ".join(f"{P:.0f}" for P in show) + " |",
        "|---|" + "---|" * len(show),
    ]

    def at(arr):
        return np.interp(show, PERIODS, np.asarray(arr, float))

    for x, lab in ((1.0, "nominal"), (0.5, "R_BLR halved"), (2.0, "R_BLR doubled")):
        r = res[x]
        L.append(
            f"| S/n, matched ({lab}) | "
            + " | ".join(f"{v / r[KEY_M]['n_objects']:.3f}" for v in at(r[KEY_M]["S"]))
            + " |"
        )
        L.append(
            f"| f95 ({lab}) | "
            + " | ".join(("%.2f" % v) if np.isfinite(v) else "—" for v in at(r[KEY_M]["f95"]))
            + " |"
        )
        k = int(np.nanargmin(r[KEY_M]["f95"])) if np.any(np.isfinite(r[KEY_M]["f95"])) else 0
        L.append(
            f"| lowest f95 ({lab}) | {r[KEY_M]['f95'][k]:.2f} at {PERIODS[k]:.0f} yr; below 1 for {PERIODS[np.array(r[KEY_M]['f95']) < 1].min() if np.any(np.array(r[KEY_M]['f95']) < 1) else float('nan'):.0f}-{PERIODS[np.array(r[KEY_M]['f95']) < 1].max() if np.any(np.array(r[KEY_M]['f95']) < 1) else float('nan'):.0f} yr |"
            + " |" * (len(show) - 1)
        )
    L.append(
        "| S/n, all strong offsets | "
        + " | ".join(f"{v / res[1.0][KEY_ALL]['n_objects']:.3f}" for v in at(res[1.0][KEY_ALL]["S"]))
        + " |"
    )
    (run / "BINARY_FRACTION.md").write_text("\n".join(L) + "\n")
    if verbose:
        print("->", run / "BINARY_FRACTION.md")
    return res


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)
