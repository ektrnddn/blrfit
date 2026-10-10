"""Build the frozen bundle of the uncertainty study for one commit of blrfit: the wheel of that commit
unpacked into installed/, the study sources, the protocol, the batch script, and the request and manifest
files that study.py verifies before and during the run. Nothing is fitted here.

    python validation/uncertainty_20261010/prepare.py --out ~/blrfit_uncertainty_20261010

Then copy the bundle to Perlmutter and submit run.sbatch from inside it (the commands are printed).
The bundle is built from `git archive HEAD` of this repository, so the working tree must be clean
(--allow-dirty to override, for a rehearsal only). BUNDLE_IDENTITY.json is written next to this file
with the hashes that finish.py compares with the return.
"""

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAYLOAD = ("study.py", "generators.py", "PROTOCOL.json", "run.sbatch")
SCHEMA = "blrfit-single-spectrum-uncertainty-request-2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the bundle directory to create (must not exist)")
    ap.add_argument(
        "--durable-parent",
        default="/global/cfs/cdirs/desi/users/eka/software/blrfit_validation",
        help="CFS directory on Perlmutter where study.py backs up the ordinary fits (must exist there)",
    )
    ap.add_argument(
        "--allow-dirty", action="store_true", help="build from HEAD although the working tree has changes"
    )
    ap.add_argument("--python", default=sys.executable, help="the python that builds the wheel")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser().resolve()
    if out.exists():
        sys.exit(f"{out} exists; choose a new directory")
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not a.allow_dirty:
        sys.exit(
            "the working tree has uncommitted changes; commit them (the bundle records the commit) or pass --allow-dirty"
        )
    tags = git("tag", "--points-at", "HEAD").split()
    with tempfile.TemporaryDirectory(prefix="blrfit-bundle-") as tmp:
        tmp = Path(tmp)
        src = tmp / "src"
        src.mkdir()
        archive = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"], cwd=REPO, check=True, capture_output=True
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as t:
            t.extractall(src)
        subprocess.run(
            [a.python, "-m", "pip", "wheel", "--no-deps", "-w", str(tmp / "wheel"), str(src)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        wheels = sorted((tmp / "wheel").glob("blrfit-*.whl"))
        if len(wheels) != 1:
            sys.exit(f"expected one wheel, found {wheels}")
        wheel = wheels[0]
        out.mkdir(parents=True)
        installed = out / "installed"
        with zipfile.ZipFile(wheel) as z:
            z.extractall(installed)
        shutil.copy2(wheel, out / wheel.name)
        wheel_sha = sha(wheel)
    package = installed / "blrfit"
    version = re.search(r'^__version__ = "([^"]+)"', (package / "__init__.py").read_text(), re.M).group(1)
    package_hashes = {
        str(p.relative_to(package)): sha(p)
        for p in sorted(package.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    for name in PAYLOAD:
        shutil.copy2(HERE / name, out / name)
    request = dict(
        schema=SCHEMA,
        created=str(date.today()),
        study=HERE.name,
        repeat_of="uncertainty_20261003",
        candidate_commit=commit,
        candidate_tags=tags,
        candidate_version=version,
        wheel=wheel.name,
        wheel_sha256=wheel_sha,
        package_hashes=package_hashes,
        protocol_sha256=sha(HERE / "PROTOCOL.json"),
        durable_parent=a.durable_parent,
        environment="source /global/common/software/desi/desi_environment.sh main; PYTHONNOUSERSITE=1; "
        "PYTHONPATH=<bundle>/installed; one numerical thread per worker (run.sbatch)",
        outputs="RUN.json, records/*.json, REPORT.json and BACKUP.json under $PSCRATCH/SINGLE_UNCERTAINTY_<job>; "
        "the compact return archive next to it; the ordinary fits backed up to durable_parent",
    )
    (out / "REQUEST.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    manifest = {name: sha(out / name) for name in (*PAYLOAD, "REQUEST.json")}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    identity = dict(
        bundle=str(out),
        candidate_commit=commit,
        candidate_tags=tags,
        candidate_version=version,
        wheel=wheel.name,
        wheel_sha256=wheel_sha,
        request_sha256=sha(out / "REQUEST.json"),
        manifest_sha256=sha(out / "MANIFEST.json"),
        protocol_sha256=request["protocol_sha256"],
        source_hashes={name: sha(HERE / name) for name in PAYLOAD},
        prepared=str(date.today()),
        working_tree_clean=not dirty,
    )
    (HERE / "BUNDLE_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: identity[k]
                for k in ("candidate_commit", "candidate_version", "wheel_sha256", "request_sha256")
            },
            indent=1,
        )
    )
    print(f"\nbundle: {out}  ({len(package_hashes)} package files)")
    print("next:")
    print(f"  scp -r {out} perlmutter.nersc.gov:/pscratch/sd/e/eka/_2BINARIES/{out.name}")
    print(
        f"  ssh perlmutter.nersc.gov 'mkdir -p {a.durable_parent} && cd /pscratch/sd/e/eka/_2BINARIES/{out.name} && sbatch run.sbatch'"
    )
    print(
        "  (about 5.3 h on 32 workers in October; then copy back the return archive and run finish.py on it)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
