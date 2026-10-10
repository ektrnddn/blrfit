# Single-spectrum uncertainty study, repeated with blrfit 0.4.0

The study of [`../uncertainty_20261003`](../uncertainty_20261003/README.md) was run with version
0.2.0rc2.dev0. Version 0.3.0 changed the continuum fit (several starts, masks grown by two pixels,
new flags) and 0.4.0 keeps that fitter; this directory repeats the study with the 0.4.0 wheel, on the
same roster, seeds, criteria and stopping rule (`PROTOCOL.json`, which differs from the 2026-10-03
protocol only in `created`, `repeat_of` and `point_fitter`). `study.py` and `generators.py` are the
frozen sources of the first study, unchanged; `study.measure` was checked to run against 0.4.0 on
two roster spectra of each grid before the bundle was built.

**Status: prepared, not run.** REPORT.json, DECISIONS.json and STUDY_IDENTITY.json appear here when
the job returns.

## Run it

1. From a clean checkout of the release commit (the bundle records the commit and the wheel):
   ```bash
   python validation/uncertainty_20261010/prepare.py --out ~/blrfit_uncertainty_20261010
   ```
   This builds the wheel from `git archive HEAD`, unpacks it into `installed/`, copies the study sources,
   writes `REQUEST.json` (the package hashes, the protocol hash, the CFS backup directory) and
   `MANIFEST.json`, and records the hashes in `BUNDLE_IDENTITY.json` here.
2. Copy the bundle to Perlmutter and submit (the commands are printed by `prepare.py`):
   ```bash
   scp -r ~/blrfit_uncertainty_20261010 perlmutter:/pscratch/sd/e/eka/_2BINARIES/blrfit_uncertainty_20261010
   ssh perlmutter 'mkdir -p /global/cfs/cdirs/desi/users/eka/software/blrfit_validation && cd /pscratch/sd/e/eka/_2BINARIES/blrfit_uncertainty_20261010 && sbatch run.sbatch'
   ```
   `run.sbatch` asks for 64 CPUs, 96 GiB and 12 hours in the shared queue, loads the DESI environment,
   puts `installed/` first on the path with one numerical thread per worker, verifies the bundle
   (`study.py verify`) and runs the 2,640 spectra with 32 workers (5.3 hours in October). The job
   output ends with `RETURN_FILE=` and `SHA256=` of the compact return archive
   (`$PSCRATCH/SINGLE_UNCERTAINTY_<job>.tar.gz`, records and reports without the fit pickles, which
   are backed up to the CFS directory).
3. Copy the archive back and finish the record:
   ```bash
   python validation/uncertainty_20261010/finish.py ~/Downloads/SINGLE_UNCERTAINTY_<job>.tar.gz
   ```
   `finish.py` verifies every member against the archive's index and the run against
   `BUNDLE_IDENTITY.json`, reduces the saved records again with the frozen `study.py`, compares the
   result with the report the job wrote, and writes `REPORT.json`, `DECISIONS.json` and
   `STUDY_IDENTITY.json`; it prints the table for `docs/validation.md`.

No selection, tolerance, component count or error multiplier may be adjusted after seeing the
result; the protocol's stopping rule is one fixed roster.
