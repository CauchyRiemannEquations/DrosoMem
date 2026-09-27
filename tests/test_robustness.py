import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from flying.brain.timed_reservoir import TimedReservoir
from flying.brain.plasticity import weight_hash
from flying.evaluation.perturbation import KINDS, drop_edges, recall
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training import phase2_robustness as study


def config():
    return json.loads(Path('configs/phase2_robustness.json').read_text())


def tiny():
    roles = np.array(['KC','MBON'])
    model = TimedReservoir(sparse.csr_matrix((2,2)),lambda digit: np.zeros(2),roles,1.,'mbon_after_kc')
    head = NonlinearReadout([1],hidden=2,seed=3)
    head.initialize(np.array([[-.1,.2],[.1,-.2]]))
    return model,head,roles


def test_null_perturbations_reproduce_real_research_checkpoint():
    saved = next(study.models(config()))
    reservoir, head, roles = saved['reservoir'], saved['head'], saved['roles']
    expected = saved['expected']
    before = reservoir.state.copy()
    for kind in ('clean',*KINDS):
        got = recall(reservoir,head,roles,'314',197,kind,0,[23,42,0])
        assert got['prediction'] == expected
        assert got['removed_edges'] == got['clipped_coordinates'] == 0
        assert got['perturbed_weight_sha256'] == weight_hash(reservoir.weights)
        np.testing.assert_array_equal(reservoir.state,before)


def test_injection_timing_matches_manual_gaussian_sequence_and_clipping():
    model,_,roles = tiny()
    class Head:
        def __init__(self): self.seen = []
        def predict(self,state):
            self.seen.append(state.copy())
            return np.array([int(state[1] > 0)])
    seed, strength, horizon = [19,3,0],10.,6
    pulse, ongoing = Head(), Head()
    a = recall(model,pulse,roles,'314',horizon,'pulse',strength,seed)
    b = recall(model,ongoing,roles,'314',horizon,'ongoing',strength,seed)
    rng = np.random.default_rng(np.random.SeedSequence([*seed,1]))
    raw = np.array([strength*rng.standard_normal(2) for _ in range(horizon)])
    expected = np.clip(raw,-1,1)
    np.testing.assert_array_equal(ongoing.seen,expected)
    np.testing.assert_array_equal(pulse.seen[0],expected[0])
    np.testing.assert_array_equal(pulse.seen[1:],np.zeros((horizon-1,2)))
    assert b['prediction'] == ''.join(str(int(x > 0)) for x in expected[:,1])
    assert a['clipped_coordinates'] == int((np.abs(raw[0]) > 1).sum())
    assert b['clipped_coordinates'] == int((np.abs(raw) > 1).sum())


def test_dropout_is_exact_nested_and_preserves_surviving_signed_weights():
    original = sparse.csr_matrix(np.array([[1.,-2.,3.],[-4.,5.,-6.],[7.,-8.,9.]]))
    original.data.flags.writeable = False
    before = weight_hash(original)
    small, ns = drop_edges(original,.25,np.random.default_rng(37))
    large, nl = drop_edges(original,.5,np.random.default_rng(37))
    assert (ns,nl) == (2,4)
    assert small.nnz == 7 and large.nnz == 5
    assert np.all((large.toarray() != 0) <= (small.toarray() != 0))
    for result in [small,large]:
        kept = result.toarray() != 0
        np.testing.assert_array_equal(result.toarray()[kept],original.toarray()[kept])
    assert weight_hash(original) == before


@pytest.mark.parametrize('kind',KINDS)
def test_seed_replay_and_no_source_mutation(kind):
    model,head,roles = tiny()
    before = head.digest(),weight_hash(model.weights),model.state.copy()
    a = recall(model,head,roles,'314',12,kind,.01,[33,42,0])
    b = recall(model,head,roles,'314',12,kind,.01,[33,42,0])
    assert a == b
    assert head.digest() == before[0] and weight_hash(model.weights) == before[1]
    np.testing.assert_array_equal(model.state,before[2])


@pytest.mark.parametrize('kind,strength',[('bad',0),('clean',1),('pulse',-1),('ongoing',float('nan')),('edge_dropout',2)])
def test_invalid_perturbation(kind,strength):
    model,head,roles = tiny()
    with pytest.raises(ValueError):
        recall(model,head,roles,'314',3,kind,strength,[1,2,3])


def passing_frame():
    rows = []
    cfg = config()
    for cohort in cfg['cohorts']:
        for circuit in ['a','b']:
            for seed in range(3):
                for initialization in range(3):
                    identity = f'{cohort["name"]}_{circuit}_{seed}_{initialization}'
                    for kind in ['clean',*KINDS]:
                        rows.append(dict(cohort=cohort['name'],model='fly',circuit=circuit,seed=seed,
                                         model_id=identity,kind=kind,strength=0 if kind == 'clean' else cfg['primary'][kind],
                                         pi_memory_score=40,censored=False,clipped_coordinates=0))
    return pd.DataFrame(rows)


def test_confirmation_cannot_be_hidden_by_initial_cohort_or_pooled_mean():
    cfg,frame = config(),passing_frame()
    assert study.summary(frame,cfg)['both_cohorts_all_primary_passed']
    mask = (frame.cohort == 'confirmation') & (frame.kind == 'ongoing') & (frame.seed == 0)
    frame.loc[mask,'pi_memory_score'] = 24
    # Overall recall ratio is still >80%, but only 4/6 blocks pass.
    got = study.summary(frame,cfg)
    decision = got['primary_by_cohort']['confirmation']['ongoing']
    assert decision['mean_retention'] and not decision['condition_retention']
    assert not decision['prefix_32_retention']
    assert not got['both_cohorts_all_primary_passed']
    assert got['primary_by_cohort']['initial']['ongoing']['passed']


def test_absent_32_completion_is_not_vacuous_success():
    cfg,frame = config(),passing_frame()
    frame.pi_memory_score = 4
    got = study.summary(frame,cfg)
    assert got['primary_by_cohort']['initial']['pulse']['mean_retention']
    assert not got['primary_by_cohort']['initial']['pulse']['prefix_32_retention']
    assert not got['both_cohorts_all_primary_passed']


def test_exact_planned_population_and_trial_count():
    cfg = config(); study.validate(cfg)
    assert sum(36*len(list(study.trials(cfg,c))) for c in cfg['cohorts']) == 16128
    bad = copy.deepcopy(cfg)
    bad['cohorts'][1]['noise_seeds'] = bad['cohorts'][0]['noise_seeds']
    with pytest.raises(ValueError): study.validate(bad)
    bad = copy.deepcopy(cfg); bad['primary']['pulse'] = 2.
    with pytest.raises(ValueError): study.validate(bad)


def test_resume_full_replay_and_corruption(tmp_path,monkeypatch):
    cfg = config()
    cfg['expected_models'] = 2
    cfg['strengths'] = {kind:[0,cfg['primary'][kind]] for kind in KINDS}
    for ix,cohort in enumerate(cfg['cohorts']): cohort['noise_seeds'] = [9000+ix*100,9001+ix*100]
    def fake_models(cfg):
        for cohort in cfg['cohorts']:
            reservoir,head,roles = tiny()
            yield dict(key=dict(cohort=cohort['name'],model_id=cohort['name'],circuit='tiny',seed=cohort['model_seeds'][0],
                                model='fly',initialization=0,weight_sha256=weight_hash(reservoir.weights),readout_sha256=head.digest()),
                       reservoir=reservoir,head=head,roles=roles,target='1'*197,expected='0'*197,circuit_index=0,state_std={})
    monkeypatch.setattr(study,'models',fake_models)
    path = tmp_path/'config.json'; path.write_text(json.dumps(cfg))
    direct,resumed = tmp_path/'direct',tmp_path/'resumed'
    study.run(path,direct)
    assert not study.run(path,resumed,max_models=1)['complete']
    assert study.run(path,resumed,resume=True) == dict(complete=True,created=1,reused=1)
    assert (direct/'evaluations.csv').read_bytes() == (resumed/'evaluations.csv').read_bytes()
    got = study.verify(resumed)
    assert got['full_recalls_replayed'] == 20 and got['null_perturbations_equal_clean'] == 6
    altered = copy.deepcopy(cfg); altered['cohorts'][0]['noise_seeds'] = [9002,9003]
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='context changed'): study.run(path,resumed,resume=True)
    path.write_text(json.dumps(cfg))
    (resumed/'checkpoints/model_000.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'): study.run(path,resumed,resume=True)
    with pytest.raises(ValueError,match='checksum'): study.verify(resumed)


def test_source_manifest_pin_is_enforced():
    cohort = config()['cohorts'][0]
    cohort['manifest_sha256'] = 'wrong'
    with pytest.raises(ValueError,match='manifest changed'):
        study.source_manifest(cohort)
