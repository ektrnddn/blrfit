# Object-level deltas, blrfit 0.1.0 to 0.2.0

`deltas_0.1.0_to_0.2.0.csv` holds one row per spectrum and line for the four pinned SDSS
spectra of `tests/data/pins.json` and the DESI example coadd of `examples/data`, fitted at the
pinned redshift, E(B-V) and complexes (the DESI example at its redrock redshift with the E(B-V)
of its fibermap, Halpha and Hbeta). It is written by

    PYTHONPATH=. python tools/make_pins.py fit --out tests/data/pins_0.2.0.json
    PYTHONPATH=. python tools/make_pins.py deltas --legacy tests/data/pins.json \
        --legacy <pins of the frozen 0.1.0 release> --current tests/data/pins_0.2.0.json \
        --out docs/deltas_0.1.0_to_0.2.0.csv

## Columns

| column | content |
|---|---|
| `spectrum`, `line` | file name; `Halpha` or `Hbeta` |
| `class_0.1.0`, `class_0.2.0` | the class of the line (A, B, C, F, W, E, X) in each version |
| `c50_sys_0.1.0`, `c50_sys_0.2.0`, `delta_c50` | the primary offset, the velocity that halves the broad flux, relative to the systemic velocity, km/s; `delta_c50` = 0.2.0 minus 0.1.0 |
| `fwhm_0.1.0`, `fwhm_0.2.0` | FWHM of the broad profile, km/s |
| `n_broad_0.1.0`, `n_broad_0.2.0` | number of broad Gaussian components chosen by BIC |
| `chi2_0.1.0`, `chi2_0.2.0` | chi-square of the line complex (weighted residuals plus the penalty terms) |
| `bic_margin_0.2.0` | distance in BIC between the chosen component count and the nearest other count (0.2.0 only; a small margin marks a decomposition on the edge) |
| `conti_feop_fwhm_0.1.0`, `conti_feop_fwhm_0.2.0` | FWHM of the optical Fe II template, km/s; undefined when the Fe II norm is zero |
| `conti_at_bound` | continuum parameters that ended on a bound in 0.2.0 (comma separated; empty when none) |
| `feuv_fwhm_fixed` | whether 0.2.0 fixed the ultraviolet Fe II width at 3000 km/s because fewer than 300 continuum pixels fall in 2200-3090 A |
| `source_0.1.0` | where the 0.1.0 numbers come from (see below) |

## Sources and platform

The 0.1.0 numbers of the four SDSS spectra are `tests/data/pins.json`, the output of the
production fitter of the DESI catalogue (2026-09-11). The DESI example has no 0.1.0 pin; its
0.1.0 row was fitted with the frozen 0.1.0 release (tag `v0.1.0`, commit a76bfc4) by the same
script, which reproduces every number of `pins.json` bit for bit on this platform (checked on
2026-09-14). The 0.2.0 numbers are `tests/data/pins_0.2.0.json`, written from the corrected
tree (`0.2.0.dev0`, git `v0.1.0-1-ga76bfc4-dirty`) on 2026-09-14. Both were computed on the
same Mac: macOS 15 arm64 (Darwin 24.6.0), Python 3.12.2, numpy 1.26.4, scipy 1.13.1, astropy
6.1.3. On another numerical stack the end points differ at the level described in
`tests/test_pins.py` (offsets by tens of km/s for degenerate decompositions), which is larger
than most of the deltas below; the table is a same-platform comparison.

## What drives each delta

No class, flag, component count or systemic source changed on any row; the deltas are in the
continuous quantities. The 0.2.0 columns record that every line was fitted from a converged
continuum by a converged solver attempt, so the flag-and-keep rule of the solver and the
continuum-status bookkeeping are not exercised here, and no Monte Carlo was run (the alias
diagnostic does not apply).

* **spec-0651 (J001224, 2001; classes C, C).** The optical Fe II norm ends at zero in both
  versions (`feop_norm` is listed at its bound), so the Fe II width multiplies nothing and its
  value (8006 then, 9956 now) is undefined; there is no ultraviolet Fe II at this redshift (no
  pixel of the 2200-3090 A windows). The deltas of 0.07 and 0.04 km/s are the solver's end
  point along a different iteration path, nothing physical.
* **spec-1237 (class A, z = 0.479).** The continuous Fe II operator: 0.1.0 rounded the Fe II
  width to 50 km/s inside the solver, which therefore saw no width derivative unless a
  finite-difference step happened to cross a rounding boundary; here the ultraviolet width
  never left its 3000 km/s start. In 0.2.0 both widths move (optical 2475 to 2421 km/s,
  ultraviolet to the 10000 km/s bound, listed) with 390 ultraviolet pixels available, the
  Fe II norms rise (optical 0.52 to 0.59, ultraviolet 0.45 to 0.57 template units), and the
  single broad component shifts by -22.8 km/s with a FWHM 49 km/s larger. The BIC margin of the
  one-component choice is 6.5, the smallest of the set.
* **spec-1592 (class B, z = 0.452, Fe II at 1.4 template units).** The continuous Fe II operator
  again (optical width 2219 to 3121 km/s, ultraviolet 3000 to 8972 km/s with 333 pixels, norms up
  by 20 per cent); c50_sys moves by -4.4 km/s, the FWHM by +50 km/s, chi-square improves.
* **spec-1704 (classes F, F; host applied, z = 0.318).** Two corrections act. The continuous
  optical Fe II operator drives the optical width from 2877 km/s to the 1200 km/s lower bound
  (listed) with the norm up from 0.24 to 0.33; and the ultraviolet width, 4999 km/s in 0.1.0, is
  fixed at 3000 km/s because only 239 of the ultraviolet pixels are covered (below 300). Both
  lines move by -2.5 km/s; the host fraction is 0.155 (0.142 in 0.1.0). Measured when the
  fixed-width rule was introduced on top of the continuous operator, that rule alone moved
  Hbeta by +0.09 km/s; the rest is the operator.
* **DESI J001247 (coadd 39627574082538900; classes F, C with `poor_fit`; host applied,
  z = 0.220).** As for spec-1704: the optical Fe II width goes from 4225 km/s to 1259 km/s (near
  the lower bound, not on it) with the norm doubled, and the ultraviolet width, 6013 km/s in
  0.1.0, is fixed at 3000 km/s (214 pixels). Hbeta moves by -0.75 km/s (the fixed width alone
  accounts for +0.22 km/s of it, measured as above), Halpha by -0.08 km/s.

In short: every delta larger than 0.1 km/s is the continuous Fe II operator (a fix: the frozen
operator had no width derivative and its end point depended on the last bit of the data,
`tests/test_reproducibility.py`), with a contribution of a few tenths of a km/s from the fixed
ultraviolet width (a policy change) on the two host-applied spectra where it applies; the
solver and continuum policies are no-ops on these five spectra.

## The 824-spectrum anchor

The anchor set (Liu et al. 2014 and Eracleous et al. 2012 objects, every SDSS epoch, 824
spectra of which 823 are fitted by both versions; the frozen fitter fails on
spec-11352-58456-0931 and so does 0.2.0) is compared per spectrum with the stored 0.1.0
catalogue run by `tests/test_anchor_liu.py`. With `BLRFIT_ANCHOR_DIR` set and
`BLRFIT_WRITE_DELTAS=1` it writes `deltas_anchor_0.1.0_to_0.2.0.csv` (both classes, both
c50_sys and their difference for Hbeta and Halpha, the BIC margin and the continuum state of the
corrected fit). Run on 2026-09-14 on the platform above:

| | Hbeta | Halpha |
|---|---|---|
| spectra with a stored class | 823 | 476 |
| class changes, 0.2.0 as released | 71 (8.6 per cent) | 8 (1.7 per cent) |
| NMAD of the c50_sys change | 6.4 km/s | 0.6 km/s |
| spectra with abs(delta c50_sys) > 100 km/s | 38 | 4 |
| class changes, historical 50 km/s Fe II operator | 17 (2.1 per cent) | 5 (1.1 per cent) |
| class changes, ultraviolet Fe II width always free | 82 (10.0 per cent) | 11 (2.3 per cent) |

The literature statistics are unchanged within their tolerance: on the 372 exact Liu spectra
with a measurable broad Hbeta, r = 0.909, median difference +5.2 km/s, NMAD 106.7 km/s, sign
agreement 96.0 per cent (0.1.0: 370, 0.909, +6.2, 104.1, 96.2 per cent).

Attribution. The two extra refits change one correction at a time. With the historical operator
put back the Hbeta class changes fall from 8.6 to 2.1 per cent and the offsets agree to 0.0 km/s
in NMAD, so about three quarters of the changes come from the continuous Fe II operator; the rest
is the fixed ultraviolet width, which replaces the random-walk end point of 0.1.0 where the
window is short. Leaving the ultraviolet width free raises the changes to 10.0 per cent, so fixing
it removes changes rather than adding them. The transitions are A to C (17), C to A (21), F to C
(10) and C to F (8), with a few W to A and B to C: objects whose one- or two-component choice or
whose significance test sits near its threshold, where the Fe II blends under Hbeta move the
decomposition. Halpha, the selection line of the catalogue, is barely affected. In the corrected
fits the optical Fe II width is not constrained in many spectra: it sits on the 1200 km/s lower
bound in 18 per cent of them, and the ultraviolet width reaches its 10000 km/s bound where it is
free; `conti_at_bound` lists these cases.
