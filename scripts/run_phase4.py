import argparse
from flying.training.phase4 import run
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase4.json');p.add_argument('--output',default='results/phase4')
a=p.parse_args();run(a.config,a.output)
