import argparse
from flying.training.phase5 import run
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase5.json');p.add_argument('--output',default='results/phase5');a=p.parse_args();run(a.config,a.output)
