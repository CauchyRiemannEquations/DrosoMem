"""Inference only: no pi generator, target digits, cached rollout or training."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy import sparse

from flying.brain.timed_reservoir import TimedReservoir
from flying.models.nonlinear_readout import NonlinearReadout

ARRAYS = {'weights_data','weights_indices','weights_indptr','roles','patterns',
          'indices','mean','scale','w1','b1','w2','b2','leak','schedule'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenEncoder:
    def __init__(self, patterns):
        self.patterns = patterns.copy()
        self.patterns.flags.writeable = False

    def __call__(self, digit):
        return self.patterns[digit]


class Opponent:
    """Independent mutable reservoir state, immutable learned parameters."""
    def __init__(self, identity, reservoir, head, prompt='314', horizon=197):
        self.identity = identity
        self.prompt = prompt
        self.horizon = horizon
        self._reservoir = reservoir
        self._head = head
        self.reset()

    def reset(self):
        self._reservoir.reset()
        for digit in self.prompt:
            self._reservoir.step(int(digit))
        self.generated = 0

    def next_digit(self):
        if self.generated >= self.horizon:
            raise StopIteration('Opponent reached its declared recall horizon')
        digit = int(self._head.predict(self._reservoir.state)[0])
        self._reservoir.step(digit)
        self.generated += 1
        return digit


class OpponentCatalog:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.digest = file_hash(self.path)
        value = json.loads(self.path.read_text(encoding='utf-8'))
        if value.get('format_version') != 1 or value.get('mode') != 'pretrained':
            raise ValueError('Unsupported opponent catalog')
        if value.get('prompt') != '314' or value.get('horizon') != 197:
            raise ValueError('This catalog must describe the validated game opening')
        self.entries = value['opponents']
        ids = [entry['id'] for entry in self.entries]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Empty or duplicate opponent identities')
        for entry in self.entries:
            name = entry['artifact']
            if Path(name).name != name or '/' in name or '\\' in name or not name.endswith('.npz'):
                raise ValueError('Opponent artifact must be a local NPZ filename')

    def select(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('Match seed must be a nonnegative integer')
        # No scores enter selection. Catalog order is fixed at export time.
        entry = self.entries[random.Random(seed).randrange(len(self.entries))]
        return self.load(entry['id'])

    def load(self, identity):
        entry = next((item for item in self.entries if item['id'] == identity),None)
        if entry is None:
            raise ValueError('Unknown opponent')
        path = self.path.parent/entry['artifact']
        if not path.is_file() or file_hash(path) != entry['sha256']:
            raise ValueError('Opponent file missing or checksum mismatch')
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files) != ARRAYS:
                raise ValueError('Unexpected inference artifact fields')
            a = {name:archive[name].copy() for name in archive.files}
        n = len(a['roles']); indices = a['indices']; h = len(a['b1'])
        if (a['roles'].ndim != 1 or a['patterns'].shape != (10,n)
                or indices.ndim != 1 or indices.dtype.kind not in 'iu'
                or not np.array_equal(indices,np.flatnonzero(a['roles']=='MBON'))
                or len(indices) != 48 or h != 8):
            raise ValueError('Invalid opponent dimensions or observation mask')
        expected = {'mean':(48,), 'scale':(48,), 'w1':(48,8), 'b1':(8,), 'w2':(8,10), 'b2':(10,)}
        if any(a[k].shape != shape for k,shape in expected.items()):
            raise ValueError('Invalid readout dimensions')
        for name in ['weights_data','patterns',*expected]:
            if not np.all(np.isfinite(a[name])):
                raise ValueError('Nonfinite opponent parameter')
        if np.any(a['scale'] <= 0):
            raise ValueError('Invalid normalization scale')
        if (a['weights_indices'].dtype.kind not in 'iu' or a['weights_indptr'].dtype.kind not in 'iu'
                or a['weights_data'].ndim != 1 or a['weights_indices'].shape != a['weights_data'].shape
                or a['weights_indptr'].shape != (n+1,) or a['weights_indptr'][0] != 0
                or a['weights_indptr'][-1] != len(a['weights_data'])
                or np.any(np.diff(a['weights_indptr']) < 0)
                or np.any(a['weights_indices'] < 0) or np.any(a['weights_indices'] >= n)):
            raise ValueError('Invalid sparse connectivity')
        weights = sparse.csr_matrix((a['weights_data'],a['weights_indices'],a['weights_indptr']),shape=(n,n))
        reservoir = TimedReservoir(weights,FrozenEncoder(a['patterns']),a['roles'],
                                   float(a['leak'].item()),str(a['schedule'].item()))
        head = NonlinearReadout(indices,hidden=8)
        head.mean = a['mean']; head.scale = a['scale']
        head.parameters = {name:a[name] for name in ['w1','b1','w2','b2']}
        if head.digest() != entry['readout_sha256']:
            raise ValueError('Readout provenance mismatch')
        for array in [head.indices,head.mean,head.scale,*head.parameters.values()]:
            array.flags.writeable = False
        return Opponent(identity,reservoir,head)
