"""Download exactly the GENCO inputs GRID MIRAGE needs (never the full 3.7 GB IEEE 118 dataset).

checkpoints: gridfm/genco-pf-datakit  case{14,118}_ieee/{tiny,small,base}/seed0 + case118_ieee/{tiny,base}/seed1
IEEE 14:  gridfm/pf_small_case14_ieee  @ pinned revision, bus/gen/branch tables (~0.3 GB)
IEEE 118: gridfm/pf_small_case118_ieee @ pinned revision, all bus_data partitions (~0.8 GB,
          needed to replicate the official split) + only the gen/branch partitions holding the
          recorded scenarios (200 scenarios per partition, partition = scenario // 200).
"""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = {'case118': '3f5a6e266f759be15e3777bf4d70133ff40e6fa7',
            'case14': '5669b8cec913ce1a1ef174d0283c6f6949135455'}
TABLES = ['README.md', 'args.log', 'n_scenarios.txt']


def recorded_case118_scenarios():
    root = Path(__file__).resolve().parents[1] / 'results'
    ids = set()
    for meta in root.glob('genco*/metadata.json'):
        m = json.loads(meta.read_text())
        if 'case118' in Path(m['config']).name:
            ids |= set(m['scenarios'])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='external-genco')
    ap.add_argument('--cases', nargs='+', default=['case14', 'case118'])
    ap.add_argument('--sizes', nargs='+', default=['tiny', 'small', 'base'])
    ap.add_argument('--scenarios', type=int, nargs='+', help='IEEE 118 scenario ids (default: all recorded in results/)')
    a = ap.parse_args()
    snapshot_download('gridfm/genco-pf-datakit', local_dir=f'{a.dest}/genco-pf-datakit',
                      allow_patterns=([f'{c}_ieee/{s}/seed0/*' for c in a.cases for s in a.sizes]\n                      + ([f'case118_ieee/{s}/seed1/*' for s in ('tiny', 'base') if s in a.sizes]\n                         if 'case118' in a.cases else []) + ['README.md']))
    if 'case14' in a.cases:
        snapshot_download('gridfm/pf_small_case14_ieee', repo_type='dataset', revision=REVISION['case14'],
                          allow_patterns=TABLES + [f'{t}_data.parquet/*' for t in ('bus', 'gen', 'branch')],
                          local_dir=f'{a.dest}/data/case14_ieee/raw', max_workers=16)
    if 'case118' in a.cases:
        ids = a.scenarios or recorded_case118_scenarios()
        parts = sorted({s // 200 for s in ids})
        pats = TABLES + ['bus_data.parquet/*']
        pats += [f'{t}_data.parquet/scenario_partition={p}/*' for t in ('gen', 'branch') for p in parts]
        snapshot_download('gridfm/pf_small_case118_ieee', repo_type='dataset', revision=REVISION['case118'],
                          allow_patterns=pats, local_dir=f'{a.dest}/data/case118_ieee/raw', max_workers=16)
        print('IEEE 118 partitions', parts)


if __name__ == '__main__':
    main()
