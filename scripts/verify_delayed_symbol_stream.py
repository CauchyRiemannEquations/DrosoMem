"""Consume finalized runs while the cohort continues; never fit partial artifacts."""
import argparse
from pathlib import Path
import time
from flying.training import whole_brain_memory as core
from run_delayed_symbol import read,check,validate,supervise


def verify(source,out,cache):
    c=read(source/'config.json');validate(c);out.mkdir(parents=True,exist_ok=False)
    context=core.source_context(c);script=core.sha256(__file__);started=time.monotonic();count=0
    core.write_json(out/'config.json',c)
    core.write_json(out/'started.json',dict(context=context,script_sha256=script))
    def wait(path):
        while not path.exists():
            if time.monotonic()-started>14400:raise TimeoutError('Incomplete cohort; partial verification preserved')
            time.sleep(2)
    for i,b in enumerate(c['blocks']):
        for circuit in c['circuit_seeds']:
            for level in c['conditions']:
                stem=f'{level}_c{circuit}_s{b["seed"]}';wait(source/stem/'manifest.json')
                args=['--config',str(out/'config.json'),'--cache',str(cache),'--out',str(out/stem),'--block',str(i),
                    '--circuit',str(circuit),'--level',level,'--source',str(source/stem)]
                usage=supervise(args,out/(stem+'.log'),c);core.write_json(out/stem/'resources.json',usage)
                count+=1;print(f'{count} independently regenerated and refitted: {stem}',flush=True)
    wait(source/'manifest.json');check(source)
    assert core.source_context(c)==context and core.sha256(__file__)==script
    core.write_json(out/'manifest.json',dict(config=c,context=context,script_sha256=script,complete=True,runs=count,
        source=str(source),source_manifest_sha256=core.sha256(source/'manifest.json'),exact_replays=count,exact_refits=count,
        artifacts={p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'));a=p.parse_args();verify(a.source,a.out,a.cache)
