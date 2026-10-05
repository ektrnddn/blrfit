# Fixed single-spectrum uncertainty study, October 2026

**Result: 67 pass, five inconclusive, zero fail. Overall: inconclusive.**
This is conditional pooled evidence, not universal error calibration. Read the
[scientific scope and five unresolved checks](../../docs/validation.md).

The frozen candidate was commit `323f2a4dd40b9f993c5f429142850ca5a1f3f799`, version
`0.2.0rc2.dev0`. Release 0.2.0 changes documentation/version metadata, not its
numerical model or error formula. `STUDY_IDENTITY.json` records the exact wheel,
source, protocol and return identities and the NERSC numerical-library versions.

## Reproduce the report without fitting

Install blrfit 0.2.0 and obtain
[`single_uncertainty_59295726_records.tar.gz`](https://github.com/ektrnddn/blrfit/releases/download/v0.2.0/single_uncertainty_59295726_records.tar.gz)
from the GitHub release. Its SHA256 is
`20462ae64ab563287c6161ede73c91b1faa75a760719dce6cd039fbffc0f368d`.
From a source checkout:

```bash
python validation/uncertainty_20261003/reproduce.py /path/to/single_uncertainty_59295726_records.tar.gz
```

This verifies the archive, all member hashes and the frozen experiment, then
regenerates the report from the saved records. It runs no optimizer and reads no
pickles. Allow several GB of free memory/disk for expanded JSON records. The
original local review reproduced every scientific value exactly; the standalone
checker permits only 1e-12 floating-point roundoff and reports its occurrence.
Decisions, denominators and nonnumeric metadata must match exactly.

- `PROTOCOL.json`: size, seeds, criteria, MC200 and stopping rule frozen before execution.
- `study.py`: exact executing study/reduction source. Its `measure` wrapper requests
  200 draws and never invokes the generator module's inherited diagnostic fits.
- `generators.py`: exact generator source, including unused historical helpers.
  Its inherited module description mentions MC30/diagnostic refits; those do not
  define this study. `PROTOCOL.json` and the actual wrapper define the run.
- `REPORT.json`: complete original report, including availability, flags and all pools.
- `DECISIONS.json`: flattened list of all 72 original decisions.
- `STUDY_IDENTITY.json`: provenance and numerical environment.

The release asset contains all 2,640 JSON measurement/MC-diagnostic records,
the original run/report, the file index and a durable-copy receipt. It excludes
the ordinary fit pickle bodies, which remain preserved on NERSC. The remote job
verified those bodies and the full 1,852,846,080-byte backup; the compact download
and local review verified the returned certificates, not an independent download
of every fit body or backup byte.

The original study command expects its frozen installed-wheel deployment and
manifest, so this directory is not an invitation to launch another fit campaign.
The published reduction is sufficient to inspect the result without new fitting.
No selection, tolerance, component count or error multiplier was adjusted after
seeing this result. Previous MC30 and velocity studies remain separate.
