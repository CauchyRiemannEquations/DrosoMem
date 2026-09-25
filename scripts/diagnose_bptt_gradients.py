"""Post hoc initial-gradient diagnosis; no new training or selection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flying.brain.constrained_bptt import ConstrainedBPTT
from flying.brain.diagnostics import normalize_condition
from flying.brain.mushroom_body import KCEncoder,CircuitReservoir,SelectedReadout,role_shuffled
from flying.data.connectome import load_connectome
from flying.data.mushroom_body import load_roles
from flying.data.pi_digits import pi_digits


def diagnose(directory):
    root=Path(directory);cfg=json.loads((root/'config.json').read_text());digits=pi_digits(cfg['pi_length']);rows=[]
    with threadpool_limits(limits=1):
        for directory in cfg['circuits']:
            a,ids,_=load_connectome(directory);roles,_=load_roles(directory,ids)
            for seed in cfg['seeds']:
                encoder=KCEncoder(roles,seed,cfg['input_fraction'],cfg['input_amplitude']);shuffled,_=role_shuffled(a,roles,seed)
                for norm in cfg['normalizations']:
                    for model,w in [('fly',a),('role_shuffled',shuffled)]:
                        initial=normalize_condition(w,norm,cfg['gain']);initial.sort_indices()
                        r=CircuitReservoir(initial,encoder,cfg['leak'],1);p=SelectedReadout(np.flatnonzero(roles=='MBON'))
                        p.fit(r.states(digits[:-1]),digits[1:],epochs=cfg['readout_epochs'],learning_rate=cfg['readout_learning_rate'],l2=cfg['readout_l2'])
                        t=ConstrainedBPTT(initial,roles,encoder,p,cfg['leak'],cfg['floor'])
                        loss,full,_=t.objective(digits[:-1],digits[1:]);_,local,_=t.objective(digits[:-1],digits[1:],temporal=False)
                        direction=-local/max(np.linalg.norm(local),1e-300);eps=1e-6
                        observed=(t.objective(digits[:-1],digits[1:],eps*direction,gradient=False)[0]-t.objective(digits[:-1],digits[1:],-eps*direction,gradient=False)[0])/(2*eps)
                        predicted=float(full@direction)
                        assert np.isclose(observed,predicted,rtol=1e-3,atol=1e-6)
                        rows.append(dict(circuit=Path(directory).name,seed=seed,normalization=norm,model=model,
                            full_gradient_norm=float(np.linalg.norm(full)),local_gradient_norm=float(np.linalg.norm(local)),
                            cosine=float(full@local/(np.linalg.norm(full)*np.linalg.norm(local))),
                            loss_direction_negative_local_predicted=predicted,loss_direction_negative_local_numeric=observed))
    frame=pd.DataFrame(rows);frame.to_csv(root/'gradient_diagnostics.csv',index=False)
    print(frame.groupby(['normalization','model'])[['cosine','full_gradient_norm','local_gradient_norm']].agg(['mean','min','max']).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();diagnose(a.directory)
