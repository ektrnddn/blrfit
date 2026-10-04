# Historical RC1 validation record

This preserves the pre-0.2.0 validation document. Its future-tense statements and
development status are historical; see [current status](VALIDATION_STATUS.md).

# Current validation status

> Interface development note (2026-10-03): `0.2.0rc2.dev0` changes public/local
> input handling and output labels while retaining the RC1 numerical core.
> The scientific outcomes below are unchanged. Input software tests and live
> retrieval examples do not complete uncertainty calibration. See
> [the input-interface note](INPUT_INTERFACE.md) and
> [the uncertainty guide](UNCERTAINTIES.md). A fixed 2,640-spectrum, 200-draw
> synthetic study is being prepared to evaluate center, width and flux errors
> on two sampling grids. It has no confirmation result yet.


Updated September 28, 2026. Version: **0.2.0rc1, scoped release candidate**.

The immediate milestone is Halpha/Hbeta single-spectrum fitting. The numerical
model remains the current development model; no additional component or relaxed
selection threshold was introduced to make weak spectra pass. The default UV
Fe II policy is A: fix its width to 3000 km/s when its continuum window has fewer
than 300 usable pixels, otherwise fit the width. The author approved policy A for this scoped candidate and the first primary
point-fit pass; alternatives B/C are not declared superior.

## Real-data engineering preflight: all planned inputs accounted for

Jobs **58972199** and **58975562** together account for the fixed 577-spectrum
preflight: 465 coadd fits and 105 difficult-input fits returned, while seven
spectra were explicitly rejected for having zero usable pixels. There are no
remaining input or fit exceptions in that combined accounting. The original job
remains FAILED because its SDSS lookup used the wrong base directory; the
24-second continuation corrected those three lookups without changing the
numerical fitter or repeating the 570 saved fits.

The frozen installed candidate passed 571 software tests (26 deselected, three
expected failures). Spectrum-aware checkpoint resume, no-op resume and refusal
of a truncated copy were exercised. All 18 protected input/v1/source hashes match
before and after; the 2,542 indexed original files also remain unchanged.

Returned fits are not all scientifically usable line measurements. Fourteen
line windows are explicitly unavailable, and one Halpha result is marked
`success_unconverged` / `HA_converged=false`. These statuses remain in the table
and must not be discarded when interpreting classifications. Seven class labels
changed across six spectra relative to the older development run. The bounded
nine-spectrum comparison is now complete: all historical records match their
old table across 139 fields each. On identical input arrays and the same local
runtime, historical and current source produce exactly identical physical
parameters, measurements, classes and component counts for all 18 line fits.
Only the documented selection-margin bookkeeping differs. No parameter,
threshold or model component was changed to obtain agreement.

The two saved NERSC runs also have identical native wavelength grids, masks and
weights in all 18 lines. Their numerical outcomes still differ across runtime
stacks, including several component choices near a selection boundary, a weak
systemic-reference ambiguity, and the explicitly unconverged Halpha fit. The
largest Hbeta offset difference (1418 km/s) belongs to an E/no-broad-line result;
the largest Halpha difference (3307 km/s) is X/no-measurable-systemic in the new
run. One retained class-A case still changes its Halpha offset by 431 km/s.
A matching class or a converged solver alone therefore does not establish a
unique physical solution. Historical raw inputs/full continuum arrays and the
old numerical stack were not replayed, so no individual library is blamed.
These diagnostics support numerical/decomposition sensitivity, not universal
platform invariance or proof that either historical answer is true.

All cases and the original outcomes are retained; this fixed comparison has
ended without selecting extra tests or tuning the model. Project audit:
`comparison_review_20260927T234306Z`.

This closes the preflight's missing-input execution work, not the scientific
gates. The author has approved the scoped candidate and first primary point-fit
run. Exact release deployment and all-row result verification remain execution
checks; no incomplete scientific criterion is reclassified by that approval. The conditional uncertainty
results below are unchanged. The proposed first primary pass uses `nmc=0`: its
classifications are descriptive point-fit outputs, not calibrated significance
claims or a completed uncertainty catalogue.

Project audit: `incoming_sdss_completion_58975562_20260927T230317Z`.

## Latest corrected-candidate confirmation: partial support, overall inconclusive

NERSC job **58960516** completed successfully in 53:06, retaining all 1,980
fresh-noise spectra without computation errors. The exact returned report reproduces
locally from the saved measurements. The corrected candidate wheel SHA256 is
`373692d12588f33256fb280aa30a31318908619d68c5534264f87eb017402295`.
All records use the supplied statistical noise for MC perturbations and retain
the 2% floor in the fitting weights. Cells, conditional eligibility, MC30 and
acceptance criteria were unchanged; this study uses new seeds, not pooled old fits.

All eight formal pooled offset/width accuracy checks and all four one-error c50
coverage checks pass for injected peak S/N 15 and 30. The two-error check passes
for Halpha/SNR30; the other three are **inconclusive**, so the overall criterion
remains **inconclusive**, not pass. There are 13 formal passes, three inconclusive
checks and no formal failures under this fixed experiment.

| Line / injected S/N | Eligible / planned | One-error coverage (95% cell-bootstrap interval) | Two-error coverage (95% cell-bootstrap interval) |
|---|---:|---|---|
| Halpha / 15 | 581/660 | 68.67% (65.16–72.29%): pass | 93.12% (91.01–95.09%): inconclusive |
| Halpha / 30 | 660/660 | 70.30% (66.52–73.79%): pass | 94.24% (92.58–95.76%): pass |
| Hbeta / 15 | 660/660 | 67.27% (64.24–70.30%): pass | 93.03% (91.06–95.00%): inconclusive |
| Hbeta / 30 | 660/660 | 68.18% (64.09–71.82%): pass | 91.67% (89.70–93.64%): inconclusive |

The whole interval must lie inside the fixed regions 60.3–76.3% and 91.4–99.4%
respectively. The three unresolved two-error intervals cross the lower boundary;
they are neither verified coverage nor established failures. All flags and
unavailable cases remain recorded. The 79 unavailable Halpha/SNR15 cases occur in
six weak-narrow/combined-nuisance cells and have class X. S/N8 is descriptive
(541/660 Halpha and 650/660 Hbeta available). No threshold, exclusion, multiplier
or component was changed after this result; no automatic extension until pass.

These passes support only the stated conditional, pooled synthetic-grid checks,
not each individual configuration, real-survey errors or the independent host
systemic frame. Halpha uses its own narrow group; formal Hbeta/SNR30 uses the
Halpha prior throughout. Hbeta/SNR15 has 79 independent-[OIII] outcomes, whose
separate diagnostic does not establish a general independent-frame claim.

That confirmation run passed 99 software tests with one optional desispec-reader comparison
skipped because desispec was not importable in the isolated environment. The
Astropy DESI and CLI fixture paths passed; the subsequent real-data desispec,
difficult-input/resume/v1 preflight is reported above. The subsequent author approval permits the scoped software candidate and first
point-fit pass only. The unresolved two-error claims remain blocked, and the
result is not a validated uncertainty catalogue.

Return SHA256: `16f8b0dcac12c2647cb1a35f8651e45ec80d22d51ebf77e5e9066fbc7657e998`.
Project audit: `incoming_fitter_noise_58960516_20260927T183802Z`.
The earlier failed confirmation and original V17/RV evidence below are unchanged.

## Earlier conditional confirmation: uncertainty criterion failed

NERSC job 58930754 retained 1,980 new synthetic spectra: all 33 development
configurations at injected peak S/N 8, 15 and 30, with 20 fresh noise realizations
per configuration/SNR and MC30. The unchanged candidate wheel has SHA256
`20f2b7f84159751a817f3caeef1048ca085412d1b28c07992c132cad0ab3c390`.
Its report was recovered after a packaging-only symlink error and reproduced
exactly from the saved measurements. There were no computation errors; the
scientific outcome is **fail** under the prospectively approved conditional rule.

Pooled offset and width accuracy pass for both lines at S/N 15 and 30, conditional
on the fixed measurability rule. Halpha at S/N 15 retains 585/660 cases; the other
three formal pools retain 660/660. The offsets refer to the reported narrow frame,
not an independently validated host-galaxy systemic velocity.

| Line / injected S/N | Truth within one reported MC error (95% cell-bootstrap interval) | Decision |
|---|---|---|
| Halpha / 15 | 72.65% (68.11–77.40%) | Inconclusive |
| Halpha / 30 | 76.21% (73.18–79.09%) | Inconclusive |
| Hbeta / 15 | 70.91% (68.03–73.94%) | Pass |
| Hbeta / 30 | 80.15% (77.12–83.18%) | Fail: overcoverage |

The one-error acceptance band is 60.3–76.3%. All four formal two-error intervals
pass their separate 91.4–99.4% band. These are approximate cell-bootstrap results
on a controlled synthetic grid, with conditional eligibility and all other flags
retained. They do not establish calibration for each individual configuration or
for real survey data. S/N 8 and width-error coverage remain descriptive.

The generator injects supplied independent pixel noise. The tested wheel added
its 2% variance floor for fitting and used that enlarged variance for MC noise.
A subsequent six-case saved-fit diagnosis completed 360 MC draw refits with
identical ordinary fits, weights, hosts, counts and seeds. Changing only the
perturbation variance to supplied noise reduced Hbeta errors by 15–28% in those
cases. One host-rich Hbeta comparison retained a 5% local/NERSC difference; the
other eleven line errors agreed to within 0.1%. This establishes an effect in
those saved states, not the entire cause of full-grid coverage or a new pass.

The current development candidate separates `ivar_stat_rest` from fit weights.
Its `input` MC policy draws supplied pixel noise; the same 2% floor remains in
all fitting weights. The historical `effective` policy remains available and
explicitly labelled. No component, fitting threshold or tuned error multiplier
was added. Ordinary fitted profiles are unchanged, but MC errors and classifications
that use those errors can change. **The fresh corrected-candidate confirmation is reported above; this failed
study tested the previous wheel.** Fixed-host,
component-count, weak-narrow and independent-frame limitations still apply.

The confirmation manifest SHA256 is
`652990bdf7b7b8055e72d753ca7415a99ee46344b3c84068d20a619db6d68239`;
the recovered return SHA256 is
`142a79d8f1ad7e9187de14401d086c2b1a89ab1de0b664283054aa784b9b07aa`.
The project audit is `incoming_fitter_confirmation_58930754_20260927T161320Z`.
Original V17 results below remain unchanged and were not pooled with this study.
The remote software checks were 65 passed and one skipped because desispec was
unavailable; real DESI-reader integration remains part of the later preflight.

## Earlier evidence and its scope

| Evaluation | Result | Scope |
|---|---|---|
| Current versus legacy real-spectrum classifications | 823 fits from 824 inputs; one unusable input. All 79 changed line classifications have documented single/joint-toggle explanations. | Attribution explains changes, not which astrophysical fit is true. Eight changes have small selection margins and one has multiple sufficient explanations. |
| Known-truth offsets and widths | In the corrected 198-spectrum primary, pooled Hbeta accuracy passes at peak S/N 15 and 30, and Halpha at 30. | 33 specified synthetic configurations, Gaussian pixel noise, declared narrow reference and tolerance regions. Not general survey completeness or independent host-galaxy systemic accuracy. |
| Weak Halpha support | At S/N 15, seven primary outcomes are unmeasurable across five cells. Of the prescribed formal follow-up checks, nine pass and two remain inconclusive. | All 310 selected follow-ups are recorded separately; they do not enlarge the primary coverage sample. Unsupported cases remain visible. |
| MC uncertainty coverage | All formal primary c50 coverage intervals remain inconclusive. | Quoted MC errors have not been shown to have calibrated 68%/95% coverage. No formal primary coverage failure was established by those intervals either. |
| Host-rich, weak spectra | A separate 360-spectrum family gave measurable Halpha: PL+Fe-only passes 11 of 12 conditional cells, with one inconclusive. Hbeta is unmeasurable. | Limited synthetic/noise conditions; no general host-subtraction accuracy or new fallback policy is established. |
| Between-epoch velocity response | A high-S/N SDSS–DESI Hbeta stratum fails its injected differential-response criterion. | Mean response to +150 km/s is about +54 km/s; to +600 is about +379 km/s, across 19 pairs in seven groups. The end-to-end path includes fitting, subtraction and velocity estimation. |

The primary known-truth manifest is
`07cca4d965f60ea6a54052af839890a1dad7669d9dfa80565ec2b2ee224b2100`.
The completed follow-up report is
`REPORT_20260919T024346.180966Z.json` in the project's V17 audit. It records 508
spectra in total and preserves the original primary decisions. The earlier
63-record experiment with a generator truth-definition defect is excluded.
The velocity-response result is from NERSC job 58873301, not a comparison that
assumes historical astrophysical fits are truth.

## Interpretation of outputs

Monte Carlo errors condition on the fitted host and component count. They do
not encompass all host, model-selection, systemic-reference or instrumental
uncertainty. A multimodality flag is a diagnostic with limited sensitivity, not
a validated coverage statement. Record flags, failures and unavailable quantities
alongside values. Numerical completion and a plausible-looking fit do not certify
accuracy, calibrated errors, orbital motion or a binary.

The tests above concern the stated synthetic inputs and populations. They do not
certify MgII, every weak/host-dominated spectrum, the physical host-galaxy systemic
frame, velocity-change significance, population rates or survey completeness.
The existing measurement eligibility rule is not a guarantee of accuracy; it
must itself be included in any stated validation population.

## Release status

The broad RV common-pixel correction is implemented, and its regression checks
pass within their recorded scope. Experimental joint velocity/uncertainty methods
remain separate and have not replaced the production algorithm. Repairing ten
numerical interval failures did not produce a scientific pass.

A final single-spectrum release still needs its supported-range and uncertainty
decision and exact-artifact verification. The author authorized release completion
on October 3; that authorization does not change any scientific outcome. The
numerical candidate has completed the NERSC engineering preflight above.
No published validated catalogue or full-data rerun is implied by this document.
The historical 0.1.0 examples/results remain historical evidence. Public release,
production refitting and subsequent scientific interpretation have separate
acceptance steps.
