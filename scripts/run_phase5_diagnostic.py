import argparse
from flying.training.phase5_diagnostic import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/phase5_diagnostic.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['discovery', 'confirmation'], default='discovery')
    parser.add_argument('--selection')
    args = parser.parse_args()
    run(args.config, args.output, args.stage, args.selection)
