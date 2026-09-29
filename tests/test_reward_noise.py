import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reward_noise import summarize


def test_equal_noise_benefit_is_not_learning_specific_and_small_n_is_preserved():
    rows=[];c=dict(bootstrap_seed=17,bootstrap_draws=1000)
    for cohort in ['discovery','confirmation']:
        for seed in [1,2,3]:
            for ci in [701,702]:
                for arm in ['contingent','frozen','yoked']:
                    for mode in ['clean','noisy']:
                        accuracy=(.4 if arm=='contingent' else .25)+(.1 if mode=='noisy' else 0)
                        rows.append(dict(cohort=cohort,seed=seed,circuit_seed=ci,arm=arm,mode=mode,accuracy=accuracy,frequency_accuracy=.25,replicas=32))
    _,_,s=summarize(rows,c)
    assert s['confirmed']['noisy_coding'] and not s['confirmed']['noise_interaction']
    assert not s['noise_removal_explanation_supported']
    assert s['discovery']['contrasts']['interaction_vs_frozen']['n']==3
