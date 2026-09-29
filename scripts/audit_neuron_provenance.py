"""Preserve recorded execution bytes when Git normalized only line endings."""
import ast
from pathlib import Path
import subprocess
import hashlib
import types
import numpy as np
from threadpoolctl import threadpool_limits
from alphabet_memory import read,check
from flying.training import whole_brain_memory as core


def sha(data):return hashlib.sha256(data).hexdigest()


@threadpool_limits.wrap(limits=1)
def main():
    out=Path('results/neuron_source_provenance');out.mkdir(exist_ok=False)
    rows=[];copied={};manifests={}
    for root in [Path('results/neuron_input_control'),Path('results/neuron_population_panel')]:
        m=check(root);manifests[root.as_posix()]=core.sha256(root/'manifest.json')
        revision=m['context']['git_commit']
        for path,expected in m['context']['source_sha256'].items():
            actual=Path(path).read_bytes();assert sha(actual)==expected,path
            canonical=subprocess.check_output(['git','show',revision+':'+path])
            equal=sha(canonical)==expected
            row=dict(run=root.as_posix(),execution_revision=revision,path=path,recorded_sha256=expected,
                git_sha256=sha(canonical),git_bytes_match=equal)
            if not equal:
                assert actual.replace(b'\r\n',b'\n')==canonical,path
                assert path.endswith('.py')
                assert ast.dump(ast.parse(actual))==ast.dump(ast.parse(canonical)),path
                dest=out/'execution-source'/path
                if path not in copied:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with dest.open('xb') as f:f.write(actual)
                    copied[path]=dest
                assert dest.read_bytes()==actual
                row.update(difference='CRLF execution bytes versus LF Git bytes only',
                    python_AST_identical=True,execution_bytes_path=dest.as_posix())
            rows.append(row)
    # Re-execute representative input-only and full-lesion cases using the
    # committed LF runner, not the execution-byte snapshot. Outcomes unchanged.
    code=subprocess.check_output(['git','show','5e13acd:scripts/neuron_panel.py'])
    module=types.ModuleType('canonical_neuron_panel');module.__file__='scripts/neuron_panel.py'
    exec(compile(code,module.__file__,'exec'),module.__dict__)
    replays=[]
    for path in [Path('results/neuron_input_control/main/gamma_c701_s71142'),
                 Path('results/neuron_population_panel/confirmation/DAN_c701_s91142')]:
        m=read(path/'manifest.json');i=m['identity']
        a,g,metrics,n,_,_=module.execute(m['config'],i['circuit_seed'],i,i['arm'],
            Path(m['full_lesion_path']) if m['full_lesion_path'] else None)
        with np.load(path/'checkpoint.npz') as saved:
            assert set(saved.files)==set(a)
            for key in a:np.testing.assert_array_equal(saved[key],a[key],err_msg=key)
        assert g==read(path/'graph.json') and metrics==read(path/'metrics.json') and n==read(path/'neural.json')
        replays.append(dict(path=path.as_posix(),canonical_LF_full_replay=True,exact_all_arrays_and_metrics=True))
    audit=dict(source_entries_checked=len(rows),unique_line_ending_differences=len(copied),
        semantic_differences=0,source_manifests=manifests,rows=rows,canonical_replays=replays,
        historical_manifests_modified=False,
        reproduction='For exact historical source hashes, copy execution-source files byte-for-byte over the same relative paths in the recorded revision. Canonical LF code has the same Python AST; two representative full cases also replay exactly.')
    core.write_json(out/'audit.json',audit)
    core.write_json(out/'manifest.json',dict(audit_script_sha256=core.sha256(__file__),
        artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))
    print(dict(entries=len(rows),preserved_execution_files=len(copied),canonical_replays=len(replays),semantic_differences=0))


if __name__=='__main__':main()
