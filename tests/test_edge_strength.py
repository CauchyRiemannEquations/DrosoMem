from pathlib import Path
import sys,copy
import numpy as np
import pytest
sys.path.insert(0,str(Path('scripts').resolve()))
from alphabet_memory import read
from edge_strength import validate,strength_masks,anatomy
from verify_edge_strength import independent_mask
from flying.training import whole_brain_memory as core


def test_effective_weight_matching_and_independent_masks():
    c=read('configs/edge_strength.json');block=c['cohorts']['discovery']['blocks'][0]
    raw,ids,roles=anatomy(701);masks=strength_masks(raw,roles,block,c)
    w=core.normalize_condition(raw,'incoming_l1',.9);w.sort_indices();target=masks['DAN']
    for arm,mask in masks.items():
        reference=independent_mask(raw,ids,roles,None,dict(arm=arm,**block),c)
        np.testing.assert_array_equal(mask,reference)
        if arm.startswith('dan_control'):
            assert mask.sum()==target.sum() and not np.array_equal(mask,target)
            assert abs(abs(w.data[mask]).sum()-abs(w.data[target]).sum())<target.sum()*.001+1e-9


def test_cannot_relax_strength_bins_or_reuse_prior_cohort():
    c=read('configs/edge_strength.json');validate(c)
    for key,value in [('normalized_bin_width',.01),('material_effect',.01)]:
        d=copy.deepcopy(c);d[key]=value
        with pytest.raises(ValueError):validate(d)
    d=copy.deepcopy(c);d['cohorts']['discovery']['blocks'][0]['seed']=121142
    with pytest.raises(ValueError):validate(d)
