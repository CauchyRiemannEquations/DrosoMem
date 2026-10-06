"""Final immutable-artifact, recorded-source, chronology and scope audit."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from tdc_support import read,write,sha,seal,git


def inventory(revision):
    result = {}
    for line in git('ls-tree','-r',revision).splitlines():
        meta,name = line.split('\t',1)
        result[name] = meta.split()[2]
    return result


def audit(out):
    started = time.perf_counter()
    root = Path('results/tdc_v2')
    out.mkdir(parents=True,exist_ok=False)
    baseline = read(root/'v1_audit/audit.json')['baseline_commit']
    before,current = inventory(baseline),inventory('HEAD')
    historical = {n:h for n,h in before.items() if n.startswith('results/')}
    assert len(historical) == 39608
    assert all(current.get(n)==h for n,h in historical.items())
    # Research numerical source/data/config modules also retain v1 bytes.
    protected = {n:h for n,h in before.items() if n.startswith(('src/','data/','configs/','scripts/'))}
    assert all(current.get(n)==h for n,h in protected.items())
    changed = git('diff','--name-only',baseline,'HEAD','--','results/').splitlines()
    assert all(n.startswith('results/tdc_v2/') for n in changed)
    checks = []
    incomplete = []
    digest_cache = {}
    def cached_sha(p):
        key = p.resolve()
        if key not in digest_cache:
            digest_cache[key] = sha(p)
        return digest_cache[key]
    for path in sorted(root.rglob('manifest.json')):
        if out in path.parents:
            continue
        m = read(path)
        for name,digest in m['artifacts'].items():
            p = path.parent/name
            assert cached_sha(p)==digest,(path,name)
        complete = m.get('complete',True)
        checks.append(dict(manifest=path.as_posix(),manifest_sha256=cached_sha(path),artifacts=len(m['artifacts']),complete=complete))
        if not complete:
            incomplete.append(path.as_posix())
    assert incomplete == ['results/tdc_v2/final_audit_attempt1/manifest.json','results/tdc_v2/p2_graphs/manifest.json']
    # Verify archived tracked source bytes at their recorded Git revision.
    # Current files may legitimately differ after an earlier console-only repair.
    source_checks = []
    trees = {}
    blob_hashes = {}
    for path in sorted(root.glob('*/source.json')):
        s = read(path)
        revision = s['source_commit']
        if revision not in trees:
            trees[revision] = inventory(revision)
        tree = trees[revision]
        for name,digest in s['hashes'].items():
            if name in tree:
                blob = tree[name]
                if blob not in blob_hashes:
                    data = subprocess.check_output(['git','cat-file','blob',blob])
                    blob_hashes[blob] = hashlib.sha256(data).hexdigest()
                assert blob_hashes[blob] == digest,(path,name,revision)
                mode = 'recorded Git source bytes'
            else:
                assert cached_sha(Path(name)) == digest,(path,name)
                mode = 'pinned external/cache input bytes'
            source_checks.append(dict(record=path.as_posix(),source_commit=revision,path=name,mode=mode,sha256=digest))
    p1 = read(root/'p1_main/summary.json')
    p1v = read(root/'p1_main_validation/checks.json')
    p1exact = read(root/'p1_decision_validation/checks.json')
    p2 = read(root/'p2_main/summary.json')
    p2v = read(root/'p2_main_validation/checks.json')
    graphs = read(root/'p2_graph_validation_retry/checks.json')
    assert p1v['all_checks_pass'] and p1exact['all_checks_pass'] and p2v['all_checks_pass'] and graphs['all_checks_pass']
    assert p1['primary_pass'] is p1v['primary_pass'] is p1exact['primary_pass'] is True
    assert p2['primary_pass'] is p2v['endpoints']['legacy5']['registered_rule_pass']
    for study,validation in [('p1_main','p1_main_validation'),('p2_main','p2_main_validation')]:
        assert read(root/validation/'checks.json')['result_manifest_sha256'] == cached_sha(root/study/'manifest.json')
    assert (p1v['cases'],p1v['replayed_streams'],p1v['independently_refit_lag_heads']) == (12,24,252)
    assert (p2v['cases'],p2v['replayed_streams'],p2v['independently_refit_lag_heads']) == (159,318,3339)
    assert graphs['independently_regenerated_nulls'] == 30
    assert read(root/'p2_retry_identity_audit/checks.json')['identical_completed_graphs'] == 24
    raw = pd.read_csv(root/'p2_main/raw-lags.csv')
    assert len(raw)==3339 and len(raw[['level','graph_id','seed']].drop_duplicates())==159
    for level,null_count,blocks in [('legacy5',20,6),('brain5',10,3)]:
        part = raw[raw.level==level]
        assert part.graph_id.nunique()==null_count+1 and part.seed.nunique()==blocks
        assert set(part.lag)==set(range(21)) and set(part.samples)=={2000}
        for n in ['correct','frequency_correct','current_correct']:
            assert part[n].between(0,2000).all()
    p1source,p2source = read(root/'p1_main/source.json'),read(root/'p2_main/source.json')
    for protocol,revision in [('981441f',p1source['source_commit']),('68eb469',p2source['source_commit'])]:
        assert subprocess.run(['git','merge-base','--is-ancestor',protocol,revision]).returncode==0
    arrays = archives = 0
    for path in sorted(root.rglob('*.npz')):
        with np.load(path,allow_pickle=False) as a:
            archives += 1
            for name in a.files:
                x = a[name]
                arrays += 1
                if x.dtype.kind in 'fci':
                    assert np.isfinite(x).all(),(path,name)
    resources = {}
    for study in ['p1_main','p1_main_validation','p2_graphs_retry','p2_graph_validation_retry','p2_main','p2_main_validation']:
        parent = read(root/study/'resources.json')
        assert parent['seconds']<=7200
        assert parent['peak_sampled_rss_bytes']<=4294967296
        tree_path = root/study/'process-tree-resources.json'
        tree_usage = read(tree_path) if tree_path.exists() else None
        if tree_usage:
            assert tree_usage['seconds']<=7200 and tree_usage['peak_sampled_process_tree_rss_bytes']<=4294967296
        resources[study] = dict(parent=parent,process_tree=tree_usage)
    documents = ['docs/drosomem-v1-closeout.md','docs/temporal-memory-curve-protocol.md','docs/temporal-memory-curve-results.md',
                 'docs/temporal-null-ensemble-protocol.md','docs/temporal-null-ensemble-results.md','docs/temporal-memory-synthesis.md',
                 'README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md']
    assert all(Path(n).is_file() for n in documents)
    import posixpath,re
    for name in documents:
        for target in re.findall(r'\]\(([^)]+)\)',Path(name).read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            dest = posixpath.normpath(posixpath.join(str(Path(name).parent).replace('\\','/'),target.split('#')[0]))
            planned_self_output = dest == (out/'audit.json').as_posix()
            assert dest in current or Path(dest).exists() or planned_self_output,(name,target)
    write(out/'manifest-checks.json',checks)
    write(out/'recorded-source-checks.json',source_checks)
    write(out/'resources.json',resources)
    write(out/'document-sha256.json',{n:sha(n) for n in documents})
    write(out/'audit.json',dict(all_checks_pass=True,baseline_commit=baseline,audit_source_commit=git('rev-parse','HEAD'),
                               verifier_sha256=sha(__file__),historical_result_blobs_unchanged=len(historical),
                               original_numerical_source_data_config_blobs_unchanged=len(protected),
                               manifests_checked=len(checks),checksum_entries=sum(x['artifacts'] for x in checks),
                               retained_incomplete_attempts=incomplete,recorded_source_entries_verified=len(source_checks),
                               archives_inspected=archives,arrays_inspected=arrays,nonfinite_arrays=0,
                               p1_primary_pass=p1['primary_pass'],p2_primary_pass=p2['primary_pass'],
                               p2_secondary_pass=p2['levels']['brain5']['registered_rule_pass'],
                               p1_independently_replayed_streams=24,p2_independently_replayed_streams=318,
                               p2_exact_full_state_trace_hashes=p2v['exact_full_state_trace_hashes'],
                               protocol_commits_precede_result_sources=True,seconds=time.perf_counter()-started,
                               scope='Preserves all baseline Git blobs; new checksum/source/array validation is separate from the bounded v1 materialized audit. No biological population or formal capacity inference.'))
    seal(out,dict(complete=True,source_commit=git('rev-parse','HEAD')))
    print(read(out/'audit.json'),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    audit(a.out)
