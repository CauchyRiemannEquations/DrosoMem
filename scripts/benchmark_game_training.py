"""One prespecified fresh fit, not a search for a fast or high-scoring seed."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from flying.game.opponent import file_hash
from flying.training.phase5_prefix_confirmation import grid, segment_digits
from flying.training.phase5_retention import fit_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/game_training_benchmark.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    source = Path('results/phase5_retention')
    manifest = json.loads((source/'manifest.json').read_text())
    cfg = manifest['context']['config']
    # Use the first configured condition and initialization, before timing/score.
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        key, reservoir, mbon = next(grid(cfg))
        digits = segment_digits(key['offset'], cfg['pi_length'])
        states = reservoir.states(digits[:-1])
        prepared = time.perf_counter()
        heads, _ = fit_head(states, digits[1:], mbon, key, 0, 'fixed', cfg)
        trained = time.perf_counter()
        runtime = [{k:v for k,v in lib.items() if k != 'filepath'} for lib in threadpool_info()]
    name = 'checkpoints/condition_000.npz'
    hashes = {k.replace('\\','/'):v for k,v in manifest['file_sha256'].items()}
    assert file_hash(source/name) == hashes[name]
    with np.load(source/name, allow_pickle=False) as archive:
        payload = json.loads(str(archive['payload'].item()))
    assert payload['key'] == key
    expected = {r['epoch']:r['readout_sha256'] for r in payload['rows']
                if r['treatment'] == 'fixed' and r['initialization'] == 0}
    assert {epoch:head.digest() for epoch,head in heads.items()} == expected
    result = dict(scope='one_fresh_fixed_readout_fit_not_a_performance_guarantee',
                  condition=key, initialization=0, updates=6000, trained_digits=200,
                  trained_parameters=heads[6000].parameter_count,
                  topology='fixed', blas_threads=1, python=platform.python_version(),
                  platform=platform.platform(), processor=platform.processor(), numpy=np.__version__,
                  numerical_libraries=runtime, preparation_seconds=prepared-start,
                  training_seconds=trained-prepared, preparation_and_training_seconds=trained-start,
                  excludes='Python imports, UI startup, checkpoint verification and file writes',
                  source_manifest_sha256=file_hash(source/'manifest.json'),
                  exact_archived_stage_head_matches=len(heads), head_sha256=expected,
                  benchmark_script_sha256=file_hash(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
