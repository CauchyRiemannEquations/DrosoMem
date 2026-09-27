"""Read-only inventory and archived artifact checksum audit (no context rewriting)."""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(p) for p in subprocess.check_output(['git', 'ls-files'], text=True).splitlines()]
    with (out/'inventory.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['path', 'bytes', 'sha256'])
        for p in files:
            w.writerow([p.as_posix(), p.stat().st_size, digest(p)])
    checks = []
    for p in sorted(Path('results').rglob('*manifest*.json')):
        obj = json.loads(p.read_text(encoding='utf-8'))
        entry = dict(manifest=p.as_posix(), keys=list(obj), checked=0, missing=[], mismatches=[])
        for key in ['file_sha256', 'files', 'output_sha256']:
            mapping = obj.get(key, {})
            if not isinstance(mapping, dict):
                continue
            for name, expected in mapping.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    continue
                path = p.parent/name.replace('\\', '/')
                if not path.exists():
                    entry['missing'].append(str(path)); continue
                entry['checked'] += 1
                if digest(path) != expected:
                    entry['mismatches'].append(str(path))
        checks.append(entry)
    (out/'artifact-checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (out/'recent-commits.txt').write_text(subprocess.check_output(['git','log','-30','--format=%H %ad %s','--date=iso'], text=True), encoding='utf-8')
    summary = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   tracked_files=len(files), manifests=len(checks),
                   checked=sum(x['checked'] for x in checks),
                   missing=sum(len(x['missing']) for x in checks), mismatches=sum(len(x['mismatches']) for x in checks),
                   checkpoint_npz_files=sum(p.suffix=='.npz' and 'results' in p.parts for p in files))
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
