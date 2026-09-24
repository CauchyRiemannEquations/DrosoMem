import argparse
from flying.training.phase3 import run
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/phase3.json'); p.add_argument('--output',default='results/phase3')
a=p.parse_args(); run(a.config,a.output)
