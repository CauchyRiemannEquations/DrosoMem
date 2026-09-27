import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flying.data.whole_brain import validate_batch, input_partition
from flying.training.phase6 import assert_exact


def test_input_accounting_separates_cancellation_weak_external_autapse():
    row = sparse.csr_matrix([[10., -12., 3., -7., 20.]])
    part = input_partition(row, dict(post=4, mask=np.array([True, True, True, False, True])))
    assert part == dict(autapse=20, weak=3, outside=7, retained=22,
                       total=52, excitatory=33, inhibitory=19)


def test_source_ids_remain_exact_and_indices_signs_verified():
    ids = np.array([720575940596125868, 720575940597856265], dtype=np.int64)
    d = pd.DataFrame(dict(Presynaptic_ID=ids, Postsynaptic_ID=ids[::-1],
                         Presynaptic_Index=[0, 1], Postsynaptic_Index=[1, 0],
                         Connectivity=[1, 7], Excitatory=[1, -1]))
    d['Excitatory x Connectivity'] = [1, -7]
    validate_batch(d, ids)
    for column, value in [('Presynaptic_Index', 2), ('Presynaptic_ID', ids[1]),
                          ('Excitatory', 0), ('Connectivity', 0),
                          ('Excitatory x Connectivity', 5)]:
        bad = d.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError):
            validate_batch(bad, ids)
    d['Presynaptic_ID'] = d.Presynaptic_ID.astype(float)
    with pytest.raises(ValueError):
        validate_batch(d, ids)


def test_replay_requires_every_array_and_exact_values():
    assert_exact({'a': np.array([1, 2])}, {'a': np.array([1, 2])})
    with pytest.raises(ValueError):
        assert_exact({'a': np.array([1])}, {})
    with pytest.raises(AssertionError):
        assert_exact({'a': np.array([1.])}, {'a': np.array([1.0000000001])})


def test_resource_sampling_includes_venv_redirector_children():
    pytest.importorskip('psutil')
    from types import SimpleNamespace
    from flying.training.phase6 import tree_rss
    interpreter = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=800000000))
    redirector = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=4000000),
                                 children=lambda recursive: [interpreter])
    assert tree_rss(redirector) == 804000000
