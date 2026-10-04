"""Verify and reduce the saved synthetic records; never fit or unpickle spectra."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def reproduce(archive):
    identity = json.loads((HERE/'STUDY_IDENTITY.json').read_text())
    if sha(archive) != identity['return_archive_sha256']:
        raise ValueError('Archive does not match the frozen study')
    for name, expected in identity['source_hashes'].items():
        if sha(HERE/name) != expected:
            raise ValueError('Study source/report changed: '+name)
    with tempfile.TemporaryDirectory(prefix='blrfit-record-review-') as directory:
        root = Path(directory); seen = set(); total = 0
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                p = PurePosixPath(member.name)
                if (not member.isfile() or p.is_absolute() or '..' in p.parts
                        or len(p.parts) < 2 or p.parts[0] != 'SINGLE_UNCERTAINTY_59295726'):
                    raise ValueError('Unexpected archive member')
                rel = PurePosixPath(*p.parts[1:]).as_posix()
                total += member.size
                if rel in seen or len(seen) >= 6000 or member.size > 512*1024**2 or total > 8*1024**3:
                    raise ValueError('Duplicate or oversized return')
                seen.add(rel)
                path = root/rel; path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, path.open('xb') as dest:
                    for block in iter(lambda: src.read(1024**2), b''):
                        dest.write(block)
        index = json.loads((root/'FILES.json').read_text())
        if seen != set(index)|{'FILES.json'}:
            raise ValueError('Return membership differs')
        for name, expected in index.items():
            if sha(root/name) != expected:
                raise ValueError('Return member changed: '+name)
        run = json.loads((root/'RUN.json').read_text())
        for key in ('job_id', 'request_sha256', 'protocol_sha256', 'manifest_sha256'):
            if str(run[key]) != str(identity[key]):
                raise ValueError('Experiment identity differs: '+key)
        protocol = json.loads((HERE/'PROTOCOL.json').read_text())
        if run['protocol'] != protocol:
            raise ValueError('Protocol differs')
        rows = [json.loads(p.read_text()) for p in sorted((root/'records').glob('*.json'))]
        import study  # Frozen reduction code, using the installed numerical dependencies.
        result = study.reduce(rows, protocol)
        saved = json.loads((root/'REPORT.json').read_text())
        if saved != json.loads((HERE/'REPORT.json').read_text()):
            raise ValueError('Published report differs from the return')
        differences = []
        def compare(a, b, path=''):
            if isinstance(a, dict):
                if set(a) != set(b):
                    raise ValueError('Different report keys: '+path)
                for key in a:
                    compare(a[key], b[key], path+'/'+str(key))
            elif isinstance(a, list):
                if len(a) != len(b):
                    raise ValueError('Different report lengths: '+path)
                for i, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, path+'/'+str(i))
            elif a != b:
                if (not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                        or not math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)):
                    raise ValueError('Report differs: '+path)
                differences.append(path)
        compare(result, {key: saved[key] for key in result})
        print(json.dumps(dict(records=len(rows), decisions=result['decisions'],
              conditional_outcome=result['conditional_outcome'], report_matches=True,
              roundoff_only_differences=len(differences), new_fits=0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    reproduce(parser.parse_args().archive)
