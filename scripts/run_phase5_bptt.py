import argparse
from flying.training.phase5_bptt import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase5_bptt.json');p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)
