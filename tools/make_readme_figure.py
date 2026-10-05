"""
The figures of the documentation, made with blrfit from public SDSS spectra:

* docs/figures/classes.png (README): the broad Hbeta profile of one quasar of
  each class, A, B, C and F, after subtraction of the fitted continuum and
  narrow lines;
* docs/figures/fit_example.png (docs/outputs.md): the diagnostic figure of
  SDSS J001224.01-102226.5, observed in 2001.

    PYTHONPATH=. python tools/make_readme_figure.py [--spectra DIR] [--out docs/figures]

Three of the spectra are in this repository (tests/data, examples/data). The
fourth, spec-2588-54174-0521, is read from --spectra when it is there and is
otherwise downloaded from SDSS DR17 (this needs the fetch extra and network
access). As in the comparison with Liu et al. (2014) in docs/validation.md, the
spectra are read with the inverse-variance mask (``mask_policy="ivar"``) and
fitted at the redshifts of Liu et al., without Galactic extinction.
"""

import argparse
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# class, description, SDSS name, spectrum (in the repository or by name), redshift, RA, Dec (deg)
EXAMPLES = (
    ("A", "bulk shift", "J101719.02+151620.8", "spec-2588-54174-0521.fits", 0.2463, 154.32925, 15.272444),
    ("B", "double-peaked", "J091833.82+315621.1", "tests/data/spec-1592-52990-0139.fits", 0.452, None, None),
    ("C", "asymmetric", "J001224.01−102226.5", "examples/data/spec-7169-56628-0344.fits", 0.2288, None, None),
    ("F", "normal", "J140827.51+142233.1", "tests/data/spec-1704-53178-0562.fits", 0.3184, None, None),
)
FIT_EXAMPLE = ("J001224.01−102226.5", "examples/data/spec-0651-52141-0072.fits", 0.2288)


def spectrum_path(name, ra, dec, spectra):
    """A spectrum of the repository, one found under ``spectra``, or one downloaded from SDSS DR17."""
    if os.path.exists(os.path.join(ROOT, name)):
        return os.path.join(ROOT, name)
    for directory, _, files in os.walk(spectra or ""):
        if name in files:
            return os.path.join(directory, name)
    from blrfit.io import fetch

    for r in fetch.fetch_sdss(ra, dec, tempfile.mkdtemp(), radius_arcsec=1.5, data_release=17):
        if r.get("filename") == name and r.get("path"):
            return r["path"]
    sys.exit(f"{name}: not under --spectra and not downloaded")


def fit(path, z):
    import blrfit

    sp = blrfit.read_spectrum(path, survey="sdss", mask_policy="ivar")
    return blrfit.fit_spectrum(sp["wave"], sp["flux"], sp["ivar"], z, ebv=0.0, complexes=("Halpha", "Hbeta"))


def profile_panel(ax, res, label, description, name, z):
    from blrfit.plot import broad_residual_profile

    v, data, model = broad_residual_profile(res, "Hbeta", vwin=9000.0, smooth_kms=150.0)
    m, cls = res["meas"]["Hbeta"], res["cls"]["Hbeta"]
    if cls["label"] != label:
        sys.exit(f"{name}: class {cls['label']}, expected {label}")
    peak = float(np.max(model))
    ax.plot(v, data / peak, color="0.62", lw=0.8)
    ax.plot(v, model / peak, color="k", lw=1.5)
    ax.axvline(0.0, color="0.35", ls=":", lw=1.0)
    blue, red = m["vB50"] - m["v_sys"], m["vR50"] - m["v_sys"]
    ax.plot([blue, red], [0.5, 0.5], color="tab:orange", lw=2.0, solid_capstyle="butt")
    ax.plot([m["c50_sys"]], [0.5], "o", color="tab:orange", ms=5.5, zorder=5)
    ax.set_title(f"{label}   {description}", loc="left", fontsize=11, fontweight="bold")
    ax.text(
        0.98,
        0.95,
        f"SDSS {name}\nz = {z:.3f}\nΔv = {m['c50_sys']:+.0f} km/s\nFWHM = {m['fwhm']:.0f} km/s".replace(
            "-", "−"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        linespacing=1.4,
    )
    ax.set_xlim(-9000, 9000)
    ax.set_xticks([-8000, -4000, 0, 4000, 8000])
    ax.set_ylim(-0.25, 1.45)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spectra", help="directory searched for spectra that are not in the repository")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import blrfit

    os.makedirs(a.out, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharex=True, sharey=True)
    for ax, (label, description, name, spectrum, z, ra, dec) in zip(axes.flat, EXAMPLES):
        res = fit(spectrum_path(spectrum, ra, dec, a.spectra), z)
        profile_panel(ax, res, label, description, name, z)
    for ax in axes[1]:
        ax.set_xlabel("velocity relative to the narrow lines (km/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("flux / broad peak")
    handles = [
        plt.Line2D([], [], color="0.62", lw=0.8),
        plt.Line2D([], [], color="k", lw=1.5),
        plt.Line2D([], [], color="tab:orange", lw=2.0, marker="o", ms=5.5),
    ]
    fig.legend(
        handles,
        ["data − continuum − narrow lines", "broad model", "half-maximum chord and c(1/2)"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(a.out, "classes.png")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"-> {path}")

    name, spectrum, z = FIT_EXAMPLE
    res = fit(os.path.join(ROOT, spectrum), z)
    fig = blrfit.plot_fit(res, title=f"SDSS {name} (2001)  z = {z:.4f}")
    path = os.path.join(a.out, "fit_example.png")
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
