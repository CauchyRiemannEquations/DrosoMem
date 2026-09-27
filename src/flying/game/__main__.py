"""Local console prototype; the graphical study screen is a later layer."""
import argparse
import json
from pathlib import Path
import secrets
import time

from flying.game.match import Match
from flying.game.opponent import OpponentCatalog


def main():
    parser = argparse.ArgumentParser(description='Human versus a pretrained FlyWire model')
    parser.add_argument('--catalog',default='assets/opponents/fixed32/catalog.json')
    parser.add_argument('--seed',type=int,help='Reproducible opponent selection seed')
    parser.add_argument('--study-seconds',type=float,default=600)
    parser.add_argument('--horizon',type=int,default=197)
    parser.add_argument('--record',type=Path,help='Save a completed match to a NEW JSON file')
    args = parser.parse_args()
    if args.record is not None and args.record.exists():
        parser.error('Record file already exists')
    seed = secrets.randbits(64) if args.seed is None else args.seed
    catalog = OpponentCatalog(args.catalog)
    match = Match(catalog.select(seed),args.study_seconds,args.horizon)
    print(f'Flying - pretrained opponent / match seed {seed}')
    print(f'Opponent: {match.opponent.identity}')
    print('The model is already trained. This timer is your study time, not live model training.')
    print('Prompt: 314. Score excludes these three digits. First error ends recall for that player.')
    print('Local console is honor-based: terminal scrollback cannot be securely hidden.')
    if match.study_material is not None:
        print('\nStudy digits: '+match.study_material+'\n')
    try:
        while match.phase == 'studying':
            print(f'\rStudy time remaining: {match.remaining_seconds:6.1f}s',end='',flush=True)
            time.sleep(min(1.,match.remaining_seconds))
        print('\nRecall starts. Study material is now unavailable through the match API.')
        match.begin_recall()
        while match.phase == 'recalling':
            if match.human.score.finished:
                result = match.submit()
            else:
                value = input(f'Digit {match.human.attempts+1} after 314: ').strip()
                try:
                    result = match.submit(value)
                except ValueError as error:
                    print(error); continue
            print(f"You: {result['human_digit']} [{result['human']['correct']}] | "
                  f"Model: {result['model_digit']} [{result['model']['correct']}]")
    except (KeyboardInterrupt,EOFError):
        print('\nMatch cancelled; no completed result recorded.')
        return
    record = dict(**match.record(),match_seed=seed,catalog_sha256=catalog.digest)
    print(f"Result: {match.winner}; human {match.human.correct}, model {match.model.correct}")
    if args.record is not None:
        args.record.parent.mkdir(parents=True,exist_ok=True)
        with args.record.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
