from pathlib import Path
import sys
import copy
import pandas as pd
import pytest
sys.path.insert(0,str(Path('scripts').resolve()))
from fresh_alignment import validate
from moment_alignment import decision
from alphabet_memory import read


def test_fresh_runner_rejects_silent_budget_and_dynamics_changes():
    c=read('configs/fresh_alignment.json');validate(c)
    for key,value in [('gain',.95),('train_samples',4000),('input_fraction',.2),('alpha',.1)]:
        changed=copy.deepcopy(c);changed[key]=value
        with pytest.raises(ValueError):validate(changed)


def test_fresh_confirmation_keeps_all_three_block_rule_and_separate_gates():
    c=read('configs/fresh_alignment.json')
    b=pd.DataFrame(dict(frequency_excess=[.2]*3,null_excess=[.2]*3,r2_vs_frequency=[-.1]*3,
        transfer_minus_target_refit=[-.3]*3,aligned_minus_unaligned=[.2]*3))
    g=decision(b,'confirmation',c)
    assert g['alignment_improvement'] and not g['past_access'] and not g['retention_tolerance']
    b['aligned_minus_unaligned']=[.3,.3,0.]
    assert not decision(b,'confirmation',c)['alignment_improvement']
    b['r2_vs_frequency']=.1;b['transfer_minus_target_refit']=[-.01,-.01,-.06]
    assert decision(b,'confirmation',c)['past_access'] and not decision(b,'confirmation',c)['retention_tolerance']
