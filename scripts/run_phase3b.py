import argparse
from flying.training.phase3b import run
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase3b.json');p.add_argument('--output',default='results/phase3b')
a=p.parse_args();run(a.config,a.output)
