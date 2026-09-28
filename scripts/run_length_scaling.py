"""Orchestrate registered lengths using the unchanged ACT II numerical runner."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from flying.data.sequences import SequenceDataset
from flying.training import sequence_memory as memory
from flying.training import whole_brain_memory as core
from verify_sequence_memory_stream import main as stream_verify


def length_config(spec, n):
    c = copy.deepcopy(spec['base'])
    c['dataset']['length'] = n
    c['eval_length'] = n - c['prompt_length']
    c['prefix_window'] = min(32, c['eval_length'])
    memory.validate(c)
    return c


def validate_spec(spec):
    lengths = spec['lengths']
    if not lengths or lengths != sorted(set(lengths)):
        raise ValueError('Lengths must be nonempty, unique and ascending')
    if spec['base']['families'] != ['random']:
        raise ValueError('Only registered random family')
    for n in lengths:
        if type(n) is not int or n <= spec['base']['prompt_length']:
            raise ValueError('Invalid sequence length')
        length_config(spec, n)
    for block in spec['base']['blocks']:
        longest = SequenceDataset('random',block['dataset_seed'],**length_config(spec,max(lengths))['dataset']).symbols()
        for n in lengths:
            shorter = SequenceDataset('random',block['dataset_seed'],**length_config(spec,n)['dataset']).symbols()
            np.testing.assert_array_equal(shorter,longest[:n])


def script_hashes():
    return {str(p.as_posix()):core.sha256(p) for p in [Path(__file__),Path(__file__).with_name('verify_sequence_memory_stream.py')]}


def all_files(out):
    return {p.relative_to(out).as_posix():core.sha256(p) for p in out.rglob('*') if p.is_file() and p != out/'manifest.json'}


def run(config, cache, out):
    spec=json.loads(config.read_text());validate_spec(spec)
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'config.json',spec)
    scripts=script_hashes();context=core.source_context(spec['base'])
    core.write_json(out/'started.json',dict(context=context,scripts=scripts,config_sha256=core.fingerprint(spec)))
    configs=out/'configs';configs.mkdir()
    for n in spec['lengths']:
        c=length_config(spec,n);path=configs/f'n{n}.json';core.write_json(path,c)
        memory.run(path,cache,out/f'n{n}')
    assert script_hashes()==scripts and core.source_context(spec['base'])==context
    core.write_json(out/'manifest.json',dict(config=spec,config_sha256=core.fingerprint(spec),context=context,
                    scripts=scripts,complete=True,artifacts=all_files(out)))


def verify(source, cache, out):
    spec=json.loads((source/'config.json').read_text());validate_spec(spec)
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();scripts=script_hashes()
    for n in spec['lengths']:
        while not (source/f'n{n}'/'config.json').exists():
            if time.monotonic()-started>14400:raise TimeoutError('Incomplete length cohort')
            time.sleep(2)
        stream_verify(source/f'n{n}',out/f'n{n}',cache)
    while not (source/'manifest.json').exists():
        if time.monotonic()-started>14400:raise TimeoutError('Missing complete source manifest')
        time.sleep(2)
    m=json.loads((source/'manifest.json').read_text());assert m['config']==spec and m['complete']
    for name,digest in m['artifacts'].items():assert core.sha256(source/name)==digest,name
    records=[json.loads((out/f'n{n}'/'manifest.json').read_text()) for n in spec['lengths']]
    assert script_hashes()==scripts
    core.write_json(out/'manifest.json',dict(source=str(source),source_manifest_sha256=core.sha256(source/'manifest.json'),
        scripts=scripts,complete=True,exact_replays=sum(r['exact_replays'] for r in records),
        exact_refits=sum(r['exact_refits'] for r in records),artifacts=all_files(out)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','verify'])
    p.add_argument('--config',type=Path,default=Path('configs/length_scaling.json'))
    p.add_argument('--cache',type=Path,default=Path('outputs/act1-graphs'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    a=p.parse_args()
    if a.command=='run':run(a.config,a.cache,a.out)
    else:verify(a.source,a.cache,a.out)
