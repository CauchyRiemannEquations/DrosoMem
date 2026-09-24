"""Download pinned author-distributed v783 table (~101 MB), then subset."""
import argparse
from pathlib import Path
import shutil
import urllib.request
from flying.data.connectome import SOURCE_URL, SOURCE_SHA256, sha256, extract_subset

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='data/raw/Connectivity_783.parquet')
    p.add_argument('--output', default='data/flywire_783_subset')
    p.add_argument('--neurons', type=int, default=300)
    p.add_argument('--min-synapses', type=int, default=5)
    args = p.parse_args()
    dest = Path(args.raw); dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        part = dest.with_suffix('.part')
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=120) as r, part.open('wb') as f:
                shutil.copyfileobj(r, f)
            if sha256(part) != SOURCE_SHA256:
                raise ValueError('Downloaded source checksum mismatch')
            part.replace(dest)
        finally:
            part.unlink(missing_ok=True)
    print(extract_subset(dest, args.output, args.neurons, args.min_synapses))

if __name__ == '__main__':
    main()
