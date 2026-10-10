"""Take the return archive of the repeated study and finish the record: verify the archive against the
bundle that prepare.py built, reduce the saved records again with the frozen study.py and compare with the
report the job wrote, then write REPORT.json, DECISIONS.json and STUDY_IDENTITY.json here. Nothing is
fitted and no pickle is read.

    python validation/uncertainty_20261010/finish.py ~/Downloads/SINGLE_UNCERTAINTY_<job>.tar.gz
"""

import argparse
import hashlib
import json
import math
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study  # noqa: E402  (the frozen reduction)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def flatten(report):
    """One record per decision, as DECISIONS.json of the 2026-10-03 study lists them."""
    out = []
    for pool in sorted(report["pools"]):
        metrics = report["pools"][pool]["metrics"]
        for metric in sorted(metrics):
            acc = metrics[metric]["accuracy"]
            out.append(
                dict(
                    check="accuracy",
                    decision=acc["decision"],
                    estimate=acc["estimate"],
                    interval=acc["interval"],
                    metric=metric,
                    n=acc["n"],
                    pool=pool,
                )
            )
            for check in ("one_error", "two_error"):
                c = metrics[metric]["coverage"][check]
                out.append(
                    dict(
                        check=check,
                        decision=c["decision"],
                        estimate=c["estimate"],
                        interval=c["interval"],
                        metric=metric,
                        n=c["n"],
                        pool=pool,
                    )
                )
    return out


def compare(a, b, path=""):
    """Equal up to floating-point roundoff; returns the paths that differ by roundoff only."""
    diffs = []
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError("different report keys at " + path)
        for k in a:
            diffs += compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            raise ValueError("different report lengths at " + path)
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            diffs += compare(x, y, f"{path}/{i}")
    elif a != b:
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) or not math.isclose(
            a, b, abs_tol=1e-12, rel_tol=1e-12
        ):
            raise ValueError("report differs at " + path)
        diffs.append(path)
    return diffs


def finish(archive):
    identity = json.loads((HERE / "BUNDLE_IDENTITY.json").read_text())
    for name, expected in identity["source_hashes"].items():
        if sha(HERE / name) != expected:
            raise ValueError("study source changed since the bundle was built: " + name)
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    with tempfile.TemporaryDirectory(prefix="blrfit-return-") as directory:
        root = Path(directory)
        seen, total, top = set(), 0, None
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                p = PurePosixPath(member.name)
                if not member.isfile() or p.is_absolute() or ".." in p.parts or len(p.parts) < 2:
                    raise ValueError("unexpected archive member " + member.name)
                top = top or p.parts[0]
                if p.parts[0] != top or not top.startswith("SINGLE_UNCERTAINTY_"):
                    raise ValueError("unexpected archive layout " + member.name)
                rel = PurePosixPath(*p.parts[1:]).as_posix()
                total += member.size
                if rel in seen or len(seen) >= 6000 or member.size > 512 * 1024**2 or total > 8 * 1024**3:
                    raise ValueError("duplicate or oversized return")
                seen.add(rel)
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, path.open("xb") as dest:
                    for block in iter(lambda: src.read(1024**2), b""):
                        dest.write(block)
        index = json.loads((root / "FILES.json").read_text())
        if seen != set(index) | {"FILES.json"}:
            raise ValueError("return membership differs from its index")
        for name, expected in index.items():
            if sha(root / name) != expected:
                raise ValueError("return member changed: " + name)
        run = json.loads((root / "RUN.json").read_text())
        for key in ("request_sha256", "protocol_sha256", "manifest_sha256"):
            if str(run[key]) != str(identity[key]):
                raise ValueError("the return is not from this bundle: " + key)
        if run["protocol"] != protocol:
            raise ValueError("the run's protocol differs from PROTOCOL.json")
        rows = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
        report = study.reduce(rows, protocol)
        saved = json.loads((root / "REPORT.json").read_text())
        roundoff = compare(report, {k: saved[k] for k in report})
        decisions = flatten(saved)
        (HERE / "REPORT.json").write_text(json.dumps(saved, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (HERE / "DECISIONS.json").write_text(
            json.dumps(decisions, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        record = dict(
            candidate_commit=identity["candidate_commit"],
            candidate_tags=identity["candidate_tags"],
            candidate_version=identity["candidate_version"],
            candidate_wheel_sha256=identity["wheel_sha256"],
            environment=run["environment"],
            job_id=run["job_id"],
            manifest_sha256=run["manifest_sha256"],
            protocol_sha256=run["protocol_sha256"],
            request_sha256=run["request_sha256"],
            return_archive=Path(archive).name,
            return_archive_sha256=sha(archive),
            records=len(rows),
            roundoff_only_differences=len(roundoff),
            repeat_of="uncertainty_20261003",
            source_hashes={
                name: sha(HERE / name)
                for name in ("DECISIONS.json", "PROTOCOL.json", "REPORT.json", "generators.py", "study.py")
            },
        )
        (HERE / "STUDY_IDENTITY.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    counts = {}
    for d in decisions:
        counts.setdefault((d["metric"], d["check"]), {}).setdefault(d["decision"], 0)
        counts[(d["metric"], d["check"])][d["decision"]] += 1
    print(
        f"records {len(rows)}; outcome {saved['conditional_outcome']}; decisions {saved['decisions']}; "
        f"elapsed {saved['elapsed_seconds'] / 3600:.1f} h; roundoff-only differences {len(roundoff)}"
    )
    print("| Quantity | Accuracy | Coverage |\n|---|---|---|")
    for metric, label in (("c50_sys", "Δv (`c50_sys`)"), ("fwhm", "FWHM"), ("broad_flux", "Broad flux")):
        acc = counts.get((metric, "accuracy"), {})
        cov = {}
        for check in ("one_error", "two_error"):
            for k, v in counts.get((metric, check), {}).items():
                cov[k] = cov.get(k, 0) + v
        fmt = lambda c: ", ".join(f"{v} {k}" for k, v in sorted(c.items()))  # noqa: E731
        print(f"| {label} | {fmt(acc)} | {fmt(cov)} |")
    print("\nnot passing:")
    for d in decisions:
        if d["decision"] != "pass":
            lo, hi = d["interval"]
            print(
                f"  {d['pool']} {d['metric']} {d['check']}: {d['decision']}, {d['estimate']:.3f} ({lo:.3f}-{hi:.3f}), n {d['n']}"
            )
    print("\nwritten: REPORT.json, DECISIONS.json, STUDY_IDENTITY.json in", HERE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path, help="SINGLE_UNCERTAINTY_<job>.tar.gz returned by the job")
    finish(ap.parse_args().archive)
