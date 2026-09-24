import argparse
from pathlib import Path
from flying.data.subsets import build_subsets
from flying.data.connectome import extract_subset

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='data/raw/Connectivity_783.parquet')
    p.add_argument('--output',default='data')
    a=p.parse_args()
    for result in build_subsets(a.raw,a.output):
        print(result,flush=True)
    print(extract_subset(a.raw,Path(a.output)/'flywire_783_hubs_1000',neurons=1000))
