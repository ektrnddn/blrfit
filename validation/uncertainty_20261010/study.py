"""One frozen conditional uncertainty experiment; never fit catalogue data."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import signal
import sys
import tarfile
import time
import traceback

import numpy as np
import generators as G

HERE = Path(__file__).resolve().parent
LINES = ('Halpha', 'Hbeta')


class CaseBudgetExceeded(BaseException):
    """Escape lower-level Exception handlers; a timeout is not a failed MC draw."""


def budget_expired(*args):
    raise CaseBudgetExceeded('case wall-clock budget exceeded')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(G.clean_json(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def roster(protocol):
    rows = []
    for gi, grid in enumerate(protocol['grids']):
        for block in range(protocol['blocks']):
            seed = protocol['seed_base'] + gi*100 + block
            gen = G.make_protocol(seed)
            for cell in gen['cells']:
                for snr in protocol['snrs']:
                    for realization in range(protocol['realizations_per_block']):
                        task = G.make_task(cell['cell_id'], snr, realization, gen)
                        rows.append(dict(id=f'{grid}_b{block:02d}_{task["task_id"]}',
                                         grid=grid, seed=seed, task=task))
    if len(rows) != protocol['expected_spectra'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('roster count/identity differs')
    if len({tuple(r['task']['noise_seed_sequence']) for r in rows}) != len(rows):
        raise ValueError('noise seed collision')
    return rows


def verify():
    import blrfit
    manifest = json.loads((HERE/'MANIFEST.json').read_text())
    for rel, expected in manifest.items():
        p = HERE/rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or p.is_symlink() or sha(p) != expected:
            raise ValueError('payload changed: '+rel)
    req = json.loads((HERE/'REQUEST.json').read_text())
    package = Path(blrfit.__file__).resolve().parent
    if package != HERE/'installed/blrfit':
        raise ValueError('wrong installed package: '+str(package))
    files = {str(p.relative_to(package)): sha(p) for p in package.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    if files != req['package_hashes']:
        raise ValueError('installed wheel source differs')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError('require one numerical thread per worker: '+key)
    if sha(HERE/'PROTOCOL.json') != req['protocol_sha256']:
        raise ValueError('protocol identity differs')
    p = json.loads((HERE/'PROTOCOL.json').read_text())
    roster(p)
    return p


def measure(case, p, out, fitter=None, deadline=None):
    import blrfit
    fitter = blrfit.fit_spectrum if fitter is None else fitter
    start = time.monotonic()
    result = dict(case=case, status='error', lines={})
    if deadline is not None and start >= deadline:
        return dict(case=case, status='not_run_deadline', lines={}, elapsed_seconds=0.)
    previous = signal.signal(signal.SIGALRM, budget_expired)
    signal.alarm(p['resources']['case_seconds'])
    try:
        sp = G.generate_spectrum(case['task'], G.make_protocol(case['seed']), grid=case['grid'])
        res = fitter(sp['wave'], sp['flux'], sp['ivar'], sp['truth']['z'],
                     complexes=LINES, nmc=p['nmc'], seed=case['task']['mc_seed'],
                     mc_noise_policy='input')
        if fitter is blrfit.fit_spectrum:
            info = res.get('mc_info', {})
            if (info.get('n_requested') != p['nmc'] or info.get('noise_policy') != 'input'
                    or len(info.get('draws', [])) != p['nmc']):
                raise ValueError('MC request/draw accounting differs')
        fit_path = Path(out)/'fits'/(case['id']+'.pkl.gz')
        with fit_path.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as f:
                pickle.dump(res, f, protocol=4)
        result.update(status='returned', input_sha256=G.array_sha256(sp['wave'], sp['flux'], sp['ivar']),
                      truth=sp['truth'], fit_sha256=sha(fit_path), fit_bytes=fit_path.stat().st_size,
                      fit_path='fits/'+fit_path.name, settings=res['settings'],
                      mc_info=res.get('mc_info', {}), host_info=res.get('host_info', {}))
        for line in LINES:
            m = res.get('meas', {}).get(line, {})
            cls = res.get('cls', {}).get(line, {})
            mi = res.get('mc_info', {}).get('lines', {}).get(line, {})
            result['lines'][line] = dict(
                measured=m, error=res.get('err', {}).get(line, {}),
                percentiles=res.get('mc', {}).get(line, {}), classification=cls,
                status=res.get('fit_status', {}).get(line, {}),
                measurable=bool(G.is_measurable(cls.get('label',''), cls.get('flags',[]),
                                     m.get('broad_flux_snr', np.nan), m.get('fwhm', np.nan))),
                mc_n_finite=mi.get('n_finite', {}), mc_flags=mi.get('flags', []))
    except (Exception, CaseBudgetExceeded) as exc:
        result.update(status='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    result['elapsed_seconds'] = time.monotonic()-start
    result['protocol_sha256'] = sha(HERE/'PROTOCOL.json')
    return G.clean_json(result)


def finite(x):
    return x is not None and np.isfinite(x)


def pooled(groups, seed, n_boot):
    # Same all-planned-cell ratio bootstrap as the previous confirmation.
    sums = np.array([sum(v) for v in groups.values()], float)
    counts = np.array([len(v) for v in groups.values()], int)
    stat = dict(n=int(counts.sum()), cells=int((counts>0).sum()), planned_cells=len(groups),
                estimate=None, interval=None, cell_counts={k:len(v) for k,v in groups.items()},
                method='95% percentile cell-block ratio bootstrap; all planned cells retained')
    if stat['n']:
        stat['estimate'] = float(sums.sum()/counts.sum())
    if stat['cells'] < 2:
        return stat
    idx = np.random.default_rng(seed).integers(0, len(groups), (n_boot, len(groups)))
    denom = counts[idx].sum(axis=1)
    stat['zero_denominator_draws'] = int((denom == 0).sum())
    if not np.all(denom):
        return stat
    cell_means = sums[counts>0]/counts[counts>0]
    if np.all(cell_means == cell_means[0]):
        stat['guard'] = 'degenerate observed cell statistics'
        return stat
    stat['interval'] = np.percentile(sums[idx].sum(axis=1)/denom, [2.5,97.5]).tolist()
    return stat


def decision(stat, limits):
    ci = stat['interval']
    if ci is None:
        return 'inconclusive'
    if limits[0] <= ci[0] and ci[1] <= limits[1]:
        return 'pass'
    if ci[1] < limits[0] or ci[0] > limits[1]:
        return 'fail'
    return 'inconclusive'


def reduce(rows, p):
    expected = {r['id']:r for r in roster(p)}
    ids = [r['case']['id'] for r in rows]
    if len(ids) != len(set(ids)) or any(r['case'] != expected.get(r['case']['id']) for r in rows):
        raise ValueError('duplicate/unexpected case')
    missing = sorted(set(expected)-set(ids))
    errors = [r['case']['id'] for r in rows if r['status'] != 'returned']
    cells = [r['cell_id'] for r in G.make_protocol()['cells']]
    report = dict(expected=len(expected), recorded=len(rows), missing=missing, errors=errors,
                  execution_complete=not missing and not errors, pools={},
                  scientific_acceptance=False, uncertainty_calibrated=False,
                  release_approved=False, scope=p['claim'])
    tests = []
    for gi, grid in enumerate(p['grids']):
        for si, snr in enumerate(p['snrs']):
            for li, line in enumerate(LINES):
                subset = [r for r in rows if r['case']['grid']==grid and r['case']['task']['snr']==snr]
                result = dict(planned=660, returned=sum(r['status']=='returned' for r in subset),
                              measurable=sum(r.get('lines',{}).get(line,{}).get('measurable',False) for r in subset),
                              class_counts=dict(Counter(r.get('lines',{}).get(line,{}).get('classification',{}).get('label','missing') for r in subset)),
                              flag_counts=dict(Counter(f for r in subset for f in r.get('lines',{}).get(line,{}).get('classification',{}).get('flags',[]))),
                              metrics={})
                for mi, metric in enumerate(p['metrics']):
                    groups = {c:[] for c in cells}
                    hits = {level:{c:[] for c in cells} for level in ('one_error','two_error')}
                    missing_errors = 0
                    for r in subset:
                        d = r.get('lines',{}).get(line,{})
                        if not d.get('measurable'):
                            continue
                        truth = r['truth']['lines'][line][metric]
                        value = d['measured'].get(metric)
                        error = d['error'].get(metric)
                        if not finite(value):
                            continue
                        cell = r['case']['task']['cell_id']
                        normalization = max(30., .05*abs(truth)) if metric=='c50_sys' else truth
                        groups[cell].append((value-truth)/normalization)
                        if (finite(error) and error>0 and d['mc_n_finite'].get(metric,0)>=p['min_finite_mc']
                                and 'mc_too_few' not in d['mc_flags']):
                            for level, multiplier in (('one_error',1),('two_error',2)):
                                hits[level][cell].append(int(abs(value-truth)<=multiplier*error))
                        else:
                            missing_errors += 1
                    seed = p['report_seed'] + gi*10000+si*1000+li*100+mi*10
                    stats = pooled(groups, seed, p['n_boot'])
                    stats['decision'] = decision(stats, p['accuracy_regions'][metric])
                    tests.append(stats['decision'])
                    item = dict(accuracy=stats, coverage={}, missing_errors=missing_errors)
                    for i, level in enumerate(hits):
                        stat = pooled(hits[level], seed+i+1, p['n_boot'])
                        stat['decision'] = decision(stat,p['coverage_bands'][level])
                        item['coverage'][level]=stat
                        tests.append(stat['decision'])
                    result['metrics'][metric]=item
                report['pools'][f'{grid}_{line}_snr{snr}']=result
    report['decisions']=dict(Counter(tests))
    report['conditional_outcome']=('inconclusive' if not report['execution_complete'] else
                                   'fail' if 'fail' in tests else 'inconclusive' if 'inconclusive' in tests else 'pass')
    return report


def package(out):
    out=Path(out)
    indexed={}
    for p in sorted(out.rglob('*')):
        if p.is_symlink():
            raise ValueError('unexpected output symlink')
        if p.is_file() and 'fits' not in p.relative_to(out).parts and p.name!='FILES.json':
            indexed[str(p.relative_to(out))]=sha(p)
    write(out/'FILES.json',indexed)
    archive=out.with_suffix('.tar.gz')
    with archive.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz:
            with tarfile.open(fileobj=gz,mode='w') as t:
                for rel in sorted([*indexed,'FILES.json']):
                    t.add(out/rel,arcname=out.name+'/'+rel,recursive=False)
    print('RETURN_FILE='+str(archive),flush=True)
    print('SHA256='+sha(archive),flush=True)


def preserve(out, destination):
    """Retain ordinary fits and reports on CFS; verify every archived byte."""
    out, destination = Path(out), Path(destination)
    if destination.is_symlink() or destination.exists() or destination.parent.is_symlink():
        raise ValueError('backup destination already exists or is a symlink')
    files = {}
    for path in sorted(out.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink in retained results')
        if path.is_file():
            files[str(path.relative_to(out))] = sha(path)
    # Pickles are already compressed; avoid recompressing them or losing source files.
    with destination.open('xb') as raw:
        with tarfile.open(fileobj=raw, mode='w') as t:
            for rel in files:
                t.add(out/rel, arcname=rel, recursive=False)
    seen = set()
    with tarfile.open(destination, 'r:') as t:
        for member in t:
            if not member.isfile() or member.name in seen or member.name not in files:
                raise ValueError('backup membership differs')
            seen.add(member.name)
            h = hashlib.sha256()
            with t.extractfile(member) as f:
                for block in iter(lambda: f.read(1024**2), b''):
                    h.update(block)
            if h.hexdigest() != files[member.name]:
                raise ValueError('backup bytes differ')
    if seen != set(files):
        raise ValueError('backup incomplete')
    return dict(path=str(destination), sha256=sha(destination), files=len(files),
                bytes=destination.stat().st_size, every_member_hash_verified=True)


def run(out):
    import scipy, astropy, blrfit
    p=verify()
    out=Path(out).absolute();out.mkdir(exist_ok=False)
    (out/'fits').mkdir();(out/'records').mkdir()
    write(out/'RUN.json',dict(protocol=p,protocol_sha256=sha(HERE/'PROTOCOL.json'),
        request_sha256=sha(HERE/'REQUEST.json'),manifest_sha256=sha(HERE/'MANIFEST.json'),
        request=json.loads((HERE/'REQUEST.json').read_text()),job_id=os.getenv('SLURM_JOB_ID'),
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__,blrfit=blrfit.__version__)))
    start=time.monotonic();rows=[]
    try:
        with ProcessPoolExecutor(max_workers=p['resources']['workers'], initializer=verify) as pool:
            deadline=start+p['resources']['dispatch_hours']*3600
            future={pool.submit(measure,c,p,str(out),deadline=deadline):c for c in roster(p)}
            for f in as_completed(future):
                c=future[f]
                try:
                    row=f.result()
                except Exception as exc:
                    row=dict(case=c,status='worker_error',error=repr(exc),lines={})
                write(out/'records'/(c['id']+'.json'),row);rows.append(row)
                if len(rows)%20==0:
                    print(f'RECORDED {len(rows)}/{p["expected_spectra"]}',flush=True)
        report=reduce(rows,p)
        report['elapsed_seconds']=time.monotonic()-start
        report['ordinary_fit_bytes']=sum(r.get('fit_bytes',0) for r in rows)
        for row in rows:
            if row['status']=='returned' and sha(out/row['fit_path'])!=row['fit_sha256']:
                raise ValueError('saved fit changed')
        report['ordinary_fit_hashes_verified']=sum(r['status']=='returned' for r in rows)
        write(out/'REPORT.json',report)
        req=json.loads((HERE/'REQUEST.json').read_text())
        backup=Path(req['durable_parent'])/(out.name+'.tar')
        write(out/'BACKUP.json',preserve(out,backup))
        verify()
    except Exception:
        write(out/'RUN_ERROR.json',dict(traceback=traceback.format_exc(),recorded=len(rows)))
        raise
    finally:
        package(out)
    print(json.dumps({k:v for k,v in report.items() if k!='pools'}),flush=True)
    return 0 if report['execution_complete'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['verify','run'])
    parser.add_argument('--out')
    a=parser.parse_args()
    if a.command=='verify':
        p=verify();print('FROZEN_UNCERTAINTY_REQUEST_VERIFIED; no fitting',p['expected_spectra'])
    else:
        raise SystemExit(run(a.out))
