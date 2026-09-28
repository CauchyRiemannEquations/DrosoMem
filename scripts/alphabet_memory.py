"""K-symbol adapters; historical numerical modules remain byte-identical."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from flying.models.nonlinear_readout import NonlinearReadout
from flying.training import whole_brain_memory as core
from flying.data.sequences import SequenceDataset


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def check(path):
    m=read(path/'manifest.json')
    for name,digest in m['artifacts'].items():assert core.sha256(path/name)==digest,(path,name)
    return m


def context(c):
    r=core.source_context(c)
    r['adapter_sha256']={f'scripts/{n}.py':core.sha256(Path(__file__).parent/(n+'.py')) for n in ['alphabet_memory','run_alphabet_memory']}
    return r


class AlphabetExperimentConfig:
    def __init__(self,values):self.values=values
    def validate(self):
        c=self.values
        assert c['conditions']==['legacy5','brain1'] and c['alphabet_sizes']==[2,4,10,16]
        assert c['dataset']==dict(length=128,offset=0,prompt=[0,1,0])
        assert c['prompt_length']==3 and c['eval_length']==125
        assert c['hidden_units']==8 and c['epochs'] in [20,2000]
        assert (c['learning_rate'],c['l2'],c['prefix_window'],c['prefix_weight'])==(.03,1e-5,32,4)
        assert (c['gain'],c['leak'],c['input_fraction'],c['input_amplitude'])==(.9,.6,.1,.5)
        assert c['schedule']=='mbon_after_kc' and c['normalization']=='incoming_l1'
        assert len(set(c['circuit_seeds']))==len(c['circuit_seeds'])
        for key in ['model_seed','dataset_seed']:assert len({b[key] for b in c['blocks']})==len(c['blocks'])


def symbol_bank(roles,seed,fraction=.1,amplitude=.5,k=16):
    if type(k) is not int or k<2 or not 0<fraction<=1 or amplitude<=0:raise ValueError('Invalid symbol encoding')
    indices=np.flatnonzero(np.asarray(roles)=='KC')
    if len(indices)<10:raise ValueError('Insufficient KC population')
    rng=np.random.default_rng(seed);count=max(1,int(len(indices)*fraction));patterns=np.zeros((k,len(roles)))
    for symbol in range(k):patterns[symbol,indices[rng.choice(len(indices),count,replace=False)]]=amplitude
    patterns.flags.writeable=False
    return patterns


class SymbolEncoder(core.MappedEncoder):
    def __call__(self,symbol):
        if not isinstance(symbol,(int,np.integer)) or not 0<=symbol<len(self.patterns):raise ValueError('Symbol outside alphabet')
        return super().__call__(symbol)


class SymbolReadout(NonlinearReadout):
    def __init__(self,indices,hidden=8,seed=0,alphabet_size=10):
        super().__init__(indices,hidden,seed)
        if type(alphabet_size) is not int or alphabet_size<2:raise ValueError('Invalid alphabet')
        self.alphabet_size=alphabet_size
    def initialize(self,states):
        x=np.atleast_2d(states)[:,self.indices]
        self.mean=x.mean(axis=0);self.scale=np.maximum(x.std(axis=0),1e-5)
        rng=np.random.default_rng(self.seed);k=self.alphabet_size
        self.parameters=dict(w1=rng.normal(size=(len(self.indices),self.hidden))/np.sqrt(len(self.indices)),
            b1=np.zeros(self.hidden),w2=rng.normal(size=(self.hidden,k))/np.sqrt(self.hidden),b2=np.zeros(k))
    def objective(self,states,labels,l2=1e-5,sample_weight=None):
        labels=np.asarray(labels)
        if labels.dtype.kind not in 'iu' or np.any(labels<0) or np.any(labels>=self.alphabet_size):raise ValueError('Label outside alphabet')
        return super().objective(states,labels,l2,sample_weight)


def build_bank(cache,condition,c):
    model,observed,base=core.build_model(cache,condition,c)
    directory=Path(f'data/flywire_783_mb_left_kc512_s{condition.circuit_seed}')
    _,original_ids,_=core.load_connectome(directory);roles,_=core.load_roles(directory,original_ids)
    original_ids=np.asarray(original_ids,dtype=np.int64)
    local=symbol_bank(roles,condition.seed,c['input_fraction'],c['input_amplitude'])
    if condition.level=='legacy5':ids=original_ids
    else:
        with np.load(Path(cache)/'nodes.npz',allow_pickle=False) as nodes:ids=nodes['ids']
    ix=pd.Index(ids).get_indexer(original_ids);assert (ix>=0).all()
    bank=np.zeros((16,len(ids)));bank[:,ix]=local;bank.flags.writeable=False
    np.testing.assert_array_equal(bank[:10],model.encoder.patterns)
    meta=dict(original_ids=original_ids,local=local,eligible=original_ids[np.asarray(roles)=='KC'].astype(str).tolist())
    return model,observed,base,bank,meta


def configure(model,base,bank,meta,k):
    model.encoder=SymbolEncoder(bank[:k]);model.reset()
    patterns=meta['local'][:k]
    g=dict(base,alphabet_size=k,input_root_ids=[meta['original_ids'][p!=0].astype(str).tolist() for p in patterns],
        input_mapping_sha256=hashlib.sha256(meta['original_ids'].tobytes()+patterns.tobytes()).hexdigest(),
        eligible_input_root_ids=meta['eligible'],stimulated_union_count=int(np.any(patterns!=0,axis=0).sum()),
        stimulated_per_symbol=np.count_nonzero(patterns,axis=1).tolist())
    return g


def dataset(c,k,seed):return SequenceDataset('random',seed,alphabet_size=k,**c['dataset'])


def fit(features,symbols,condition,c,k):
    head=SymbolReadout(np.arange(48),c['hidden_units'],core.nonlinear_seed(condition.seed,c['initialization']),k)
    weights=core.sample_weights(len(symbols)-1,c['prompt_length'],c['prefix_window'],c['prefix_weight'])
    _,history=head.fit(features,symbols[1:],epochs=c['epochs'],learning_rate=c['learning_rate'],l2=c['l2'],checkpoints=(c['epochs'],),sample_weight=weights)
    assert head.parameter_count==392+9*k
    return head,history,weights
