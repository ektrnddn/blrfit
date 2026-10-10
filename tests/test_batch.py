"""Several spectra per call: input lists, the catalogue table, and worker processes
that change nothing but the time taken."""

import json

import numpy as np
import pytest
from astropy.table import Table

import blrfit
from blrfit.batch import column_unit, read_list, summary_table, write_table
from blrfit.cli import main
from conftest import SDSS_EXAMPLE, SDSS_EXAMPLE_2, Z_J001224
from synth import make_spectrum

# Float columns of a summary row that carry no unit: ratios, indices, S/N, chi-square and BIC values.
DIMENSIONLESS = {
    "z_in",
    "host_frac",
    "pl_alpha",
    "feop_norm",
    "feuv_norm",
    "z_sys",
    "skew",
    "AI",
    "KI",
    "dip_frac",
    "broad_peak_snr",
    "broad_flux_snr",
    "narrow_peak_snr",
    "sys_snr",
    "o3_core_snr",
    "o3_pre_snr",
    "nw_f",
    "chi2_red",
    "bic_margin",
    "bic_gap",
    "mc_alias_fraction",
    "n_peaks",
}


def _synthetic_fit(**kw):
    z = 0.12
    s = make_spectrum(
        z=z,
        snr=30.0,
        seed=5,
        broad=[
            dict(line="Halpha", v=300.0, fwhm=4000.0, ew=120.0),
            dict(line="Hbeta", v=300.0, fwhm=4000.0, ew=40.0),
        ],
        narrow=dict(ew_ha=20.0),
    )
    return blrfit.fit_spectrum(s["wave"], s["flux"], s["ivar"], z, complexes=("Halpha", "Hbeta"), **kw)


def test_list_files(tmp_path):
    text = tmp_path / "spectra.txt"
    text.write_text("# epoch 1\nspec-a.fits\n\ncoadd-b.fits 39627574082538900  # with its TARGETID\n")
    assert read_list(text) == [
        dict(path="spec-a.fits", targetid=None, z=None),
        dict(path="coadd-b.fits", targetid=39627574082538900, z=None),
    ]
    table = tmp_path / "spectra.csv"
    Table(dict(path=["a.fits", "b.fits"], targetid=[-1, 39627574082538900], z=[np.nan, 0.5])).write(table)
    assert read_list(table) == [
        dict(path="a.fits", targetid=None, z=None),
        dict(path="b.fits", targetid=39627574082538900, z=0.5),
    ]
    (tmp_path / "bad.txt").write_text("a.fits 12 extra\n")
    with pytest.raises(ValueError, match="line 1"):
        read_list(tmp_path / "bad.txt")
    (tmp_path / "empty.txt").write_text("# nothing\n")
    with pytest.raises(ValueError, match="lists no spectrum"):
        read_list(tmp_path / "empty.txt")
    Table(dict(file=["a.fits"])).write(tmp_path / "nopath.ecsv")
    with pytest.raises(ValueError, match="column 'path'"):
        read_list(tmp_path / "nopath.ecsv")


@pytest.mark.parametrize("suffix", [".fits", ".ecsv"])
def test_summary_table_keeps_types_units_and_missing_values(tmp_path, suffix):
    records = [
        dict(
            spectrum="a",
            targetid=39627574082538900,
            status="fitted",
            HA_c50_sys=-1060.5,
            HA_n_broad=3,
            HA_class="C",
        ),
        dict(spectrum="b", targetid=None, status="failed", error="file not found"),
        dict(
            spectrum="c",
            targetid=7,
            status="fitted",
            HA_c50_sys=np.nan,
            HA_n_broad=1,
            HA_class="E",
            HB_converged=True,
        ),
    ]
    t = summary_table(records, meta=dict(NMC=0))
    assert t.colnames[:4] == ["spectrum", "targetid", "status", "error"]
    path = tmp_path / ("table" + suffix)
    write_table(t, path)
    back = Table.read(path)
    assert back["targetid"][0] == 39627574082538900 and back["targetid"].mask[1]
    assert str(back["HA_c50_sys"].unit) == "km / s" and back["HA_c50_sys"][0] == -1060.5
    assert back["HA_c50_sys"].mask[1] or np.isnan(back["HA_c50_sys"][1])
    assert back["HA_n_broad"].dtype.kind == "i" and back["HA_n_broad"][2] == 1 and back["HA_n_broad"].mask[1]
    assert list(back["HA_class"].filled("")) == ["C", "", "E"] and back["error"][1] == "file not found"
    assert back.meta["NMC"] == 0
    with pytest.raises(ValueError, match="FITS"):
        write_table(t, tmp_path / "table.txt")


def test_every_float_column_has_a_unit_or_is_dimensionless():
    row = blrfit.summary_row(_synthetic_fit(nmc=2))
    assert any(k.startswith("HA_e_") for k in row)  # the Monte Carlo columns are covered too
    for key, value in row.items():
        if isinstance(value, (float, np.floating)):
            name = key.split("_", 1)[1] if key[:3] in ("HA_", "HB_", "MG_") else key
            name = name.removeprefix("conti_").removeprefix("e_")
            assert column_unit(key) is not None or name in DIMENSIONLESS, key
    assert column_unit("HB_e_c50_sys") == "km / s" and column_unit("HA_broad_lum") == "erg / s"


def _same(a, b):
    """Equal through dicts, lists and arrays, with NaN equal to NaN."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b, equal_nan=np.asarray(a).dtype.kind == "f")
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a):
        return bool(np.isnan(b))
    return a == b


def test_monte_carlo_draws_do_not_depend_on_the_number_of_processes():
    one, two = _synthetic_fit(nmc=3, seed=4), _synthetic_fit(nmc=3, seed=4, jobs=2)
    assert _same(one["mc_info"], two["mc_info"])
    assert _same(one["mc"], two["mc"]) and _same(one["err"], two["err"])
    # the draws themselves (too few for percentiles) differ with the seed: the comparison can fail
    other = _synthetic_fit(nmc=3, seed=5)
    assert not _same(one["mc_info"]["lines"], other["mc_info"]["lines"])


def _batch(out, *extra):
    return main(
        [
            "fit",
            SDSS_EXAMPLE,
            SDSS_EXAMPLE_2,
            "--survey",
            "sdss",
            "--z",
            str(Z_J001224),
            "--lines",
            "Hbeta",
            "--no-figure",
            "--quiet",
            "--out",
            str(out),
            *extra,
        ]
    )


def test_batch_products_do_not_depend_on_the_number_of_processes(tmp_path):
    assert _batch(tmp_path / "serial") == 0
    assert _batch(tmp_path / "parallel", "--jobs", "2") == 0
    for name in ("spec-0651-52141-0072_fit.json", "spec-7169-56628-0344_fit.json", "blrfit_summary.fits"):
        assert (tmp_path / "serial" / name).read_bytes() == (tmp_path / "parallel" / name).read_bytes()
    t = Table.read(tmp_path / "serial" / "blrfit_summary.fits")
    assert list(t["stem"]) == ["spec-0651-52141-0072", "spec-7169-56628-0344"]
    assert list(t["status"]) == ["fitted", "fitted"] and list(t["HB_class"]) == ["C", "C"]
    doc = json.loads((tmp_path / "serial" / "spec-0651-52141-0072_fit.json").read_text())
    assert t["HB_c50_sys"][0] == pytest.approx(doc["lines"]["Hbeta"]["dv"], abs=1e-9)


def test_batch_records_failures_and_names(tmp_path, capsys):
    listing = tmp_path / "spectra.txt"
    listing.write_text(f"{tmp_path / 'one' / 'missing.fits'}\n{tmp_path / 'two' / 'missing.fits'}\n")
    rc = main(
        ["fit", "--list", str(listing), "--out", str(tmp_path / "out"), "--table", str(tmp_path / "t.ecsv")]
    )
    assert rc == 1
    t = Table.read(tmp_path / "t.ecsv")
    assert list(t["stem"]) == ["missing", "missing-2"] and list(t["status"]) == ["failed", "failed"]
    assert "file not found" in t["error"][0]
    assert "0 fitted, 2 failed" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--stem"):
        main(["fit", SDSS_EXAMPLE, SDSS_EXAMPLE_2, "--stem", "x", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="--jobs"):
        main(["fit", SDSS_EXAMPLE, "--jobs", "0", "--out", str(tmp_path)])
    with pytest.raises(
        SystemExit, match="ECSV"
    ):  # an unsupported table format is refused before anything is fitted
        main(
            ["fit", SDSS_EXAMPLE, SDSS_EXAMPLE_2, "--table", str(tmp_path / "t.txt"), "--out", str(tmp_path)]
        )
