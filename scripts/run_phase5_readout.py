import argparse
from flying.training.phase5_readout import run
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase5_readout.json');p.add_argument('--output',default='results/phase5_readout')
a=p.parse_args();run(a.config,a.output)
