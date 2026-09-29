from pathlib import Path
import sys,copy
import pytest
sys.path.insert(0,str(Path('scripts').resolve()))
from kc_confirmation import validate,aggregate
from alphabet_memory import read


def test_confirmation_rejects_old_seeds_or_tuned_settings():
    c=read('configs/kc_confirmation.json');validate(c)
    for key,value in [('gain',.8),('alpha',.1),('material_effect',.01),('bootstrap_seed',64399)]:
        changed=copy.deepcopy(c);changed[key]=value
        with pytest.raises(ValueError):validate(changed)
    changed=copy.deepcopy(c);changed['blocks'][0]['seed']=61142
    with pytest.raises(ValueError):validate(changed)


def test_discovery_and_many_controls_cannot_rescue_failed_fresh_block():
    c=dict(primary_lags=[1,2],bootstrap_seed=74399,bootstrap_draws=100,access_margin=.05,retention_tolerance=.05,material_effect=.05)
    rows=[]
    for seed,gain in [(71142,.6),(71143,.6),(71144,-.01)]:
        for ci in [701,702]:
            for lag in [1,2]:
                for arm in ['gamma','matched0','matched1','matched2']:
                    delta=gain if arm=='gamma' else .4
                    rows.append(dict(cohort='main',family='full',arm=arm,group='gamma' if arm=='gamma' else 'matched',seed=seed,circuit_seed=ci,lag=lag,
                        test_accuracy=.8-delta,target_refit_accuracy=.8,refit_minus_frozen=delta,intact_minus_frozen=delta,
                        frequency_excess=.55-delta,null_excess=.55-delta,r2_vs_frequency=.2,
                        refit_frequency_excess=.55,refit_null_excess=.55,refit_r2_vs_frequency=.5))
    _,blocks,_,summary=aggregate(rows,c,True,True)
    assert len(blocks)==6 and summary['cells']['full/gamma']['refit_minus_frozen']['n']==3
    assert summary['gates']['full/matched']['refit_benefit']
    assert summary['discovery_H1'] and not summary['primary_H1'] and not summary['confirmation_H1']
    assert not summary['cohorts_pooled']
