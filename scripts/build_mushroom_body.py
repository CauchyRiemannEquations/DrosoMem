import argparse
from flying.data.mushroom_body import download_annotations,build_circuits
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',default='data/raw/Connectivity_783.parquet')
    p.add_argument('--annotations',default='data/raw/annotations_783.tsv')
    p.add_argument('--output',default='data')
    args=p.parse_args();download_annotations(args.annotations)
    for m in build_circuits(args.raw,args.annotations,args.output):print(m,flush=True)
