"""Close the mechanism study without altering old artifacts or source hashes."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from mechanism_support import read,write,sha,attempt,seal,git,history_preserved


def tree(revision):
    result={}
    for line in git('ls-tree','-r',revision).splitlines():
        meta,name=line.split('\t',1);result[name]=meta.split()[2]
    return result


def audit(out):
    root=Path('results/tdc_mechanism_v1');c=read(root/'main/config.json')
    with attempt(out,c,'mechanism-closeout-audit'):
        preservation=history_preserved(c)
        digests={}
        def cached(p):
            p=Path(p).resolve()
            if p not in digests:digests[p]=sha(p)
            return digests[p]
        manifests=[];incomplete=[]
        for p in sorted(root.rglob('manifest.json')):
            if out in p.parents:continue
            m=read(p)
            for name,value in m['artifacts'].items():assert cached(p.parent/name)==value,(p,name)
            manifests.append(dict(path=p.as_posix(),sha256=cached(p),artifacts=len(m['artifacts']),complete=m.get('complete',True)))
            if not m.get('complete',True):incomplete.append(p.as_posix())
        repair=read(root/'validation_memory_repair/repair.json')
        expected=['results/tdc_mechanism_v1/test_attempt1/manifest.json']+repair['incomplete_manifest_paths']
        assert incomplete==sorted(expected)
        failed=read(root/'main_validation/failure.json')
        assert failed['message']=='Registered aggregate process-tree budget exceeded'
        assert repair['scientific_parameters_changed'] is False
        assert repair['unchanged_main_manifest_sha256']==cached(root/'main/manifest.json')
        assert repair['original_verifier_sha256']==cached(root/'validation_memory_repair/original-verifier.py.txt')
        assert repair['repaired_verifier_sha256']==cached('scripts/verify_temporal_mechanism.py')
        recorded=[];trees={};blobs={}
        for p in root.glob('*/source.json'):
            s=read(p);revision=s['source_commit']
            if revision not in trees:trees[revision]=tree(revision)
            inventory=trees[revision]
            for name,value in s['hashes'].items():
                if name in inventory:
                    blob=inventory[name]
                    if blob not in blobs:blobs[blob]=hashlib.sha256(subprocess.check_output(['git','cat-file','blob',blob])).hexdigest()
                    assert blobs[blob]==value,(p,name,revision)
                else:assert cached(name)==value,(p,name)
                recorded.append(dict(record=p.as_posix(),source_commit=revision,path=name,sha256=value))
        smoke=read(root/'smoke_validation/checks.json');main=read(root/'main_validation_retry1/checks.json');summary=read(root/'main/summary.json')
        assert smoke['all_checks_pass'] and main['all_checks_pass']
        assert (smoke['cases'],smoke['replayed_streams'],smoke['independently_refit_lag_heads'])==(8,16,168)
        assert (main['cases'],main['replayed_streams'],main['independently_refit_lag_heads'])==(36,72,756)
        assert main['instantaneous_certificates']==9 and main['zero_state_input_access_certificates']==36
        assert main['result_manifest_sha256']==cached(root/'main/manifest.json')
        assert smoke['result_manifest_sha256']==cached(root/'smoke/manifest.json')
        assert main['outcome']==summary['outcome']
        assert main['endpoints']['legacy5']['synaptic_increment']['registered_rule_pass'] is summary['primary_pass']
        source=read(root/'main/source.json')
        assert subprocess.run(['git','merge-base','--is-ancestor','6c6c296',source['source_commit']]).returncode==0
        f=pd.read_csv(root/'main/raw-lags.csv')
        assert len(f)==756 and len(f[['level','arm','seed']].drop_duplicates())==36
        for level,n in [('legacy5',6),('brain5',3)]:
            part=f[f.level==level]
            assert part.seed.nunique()==n and set(part.arm)==set(c['arms']) and set(part.lag)==set(range(21))
            assert set(part.samples)=={2000}
        arrays=archives=0
        for p in root.rglob('*.npz'):
            with np.load(p,allow_pickle=False) as a:
                archives+=1
                for key in a.files:
                    x=a[key];arrays+=1
                    if x.dtype.kind in 'fci':assert np.isfinite(x).all(),(p,key)
        usage={}
        for n in ['smoke','smoke_validation','smoke_validation_retry1','main','main_validation_retry1']:
            parent=read(root/n/'resources.json');pool=read(root/n/'process-tree-resources.json')
            assert parent['seconds']<=c['max_seconds'] and pool['seconds']<=c['max_seconds']
            assert pool['peak_sampled_process_tree_rss_bytes']<=c['max_rss_bytes']
            usage[n]=dict(parent=parent,process_tree=pool)
        documents=['docs/temporal-mechanism-protocol.md','docs/temporal-mechanism-results.md','README.md','docs/research-status.md','docs/research-roadmap.md','docs/next-work.md']
        import re,posixpath
        inventory=tree('HEAD')
        for name in documents:
            assert Path(name).exists()
            for target in re.findall(r'\]\(([^)]+)\)',Path(name).read_text(encoding='utf-8')):
                if '://' in target or target.startswith('#'):continue
                dest=posixpath.normpath(posixpath.join(Path(name).parent.as_posix(),target.split('#')[0]))
                assert dest in inventory or Path(dest).exists() or dest==(out/'audit.json').as_posix(),(name,target)
        write(out/'manifest-checks.json',manifests);write(out/'recorded-source-checks.json',recorded)
        write(out/'stage-resources.json',usage);write(out/'document-sha256.json',{name:sha(name) for name in documents})
        write(out/'audit.json',dict(all_checks_pass=True,**preservation,source_commit=git('rev-parse','HEAD'),verifier_sha256=sha(__file__),
                                  manifests_checked=len(manifests),checksum_entries=sum(m['artifacts'] for m in manifests),recorded_source_entries=len(recorded),
                                  retained_incomplete_attempts=incomplete,archives_inspected=archives,arrays_inspected=arrays,nonfinite_arrays=0,
                                  independent_main_streams=72,independent_main_heads=756,instantaneous_certificates=9,
                                  exact_main_state_hashes=main['exact_full_state_trace_hashes'],outcome=summary['outcome'],primary_pass=summary['primary_pass'],
                                  protocol_precedes_source=True,biological_plasticity_performed=False))
    seal(out,dict(complete=True,source_commit=git('rev-parse','HEAD')))
    print(read(out/'audit.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.out)
