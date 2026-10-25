"""Download exactly the GENCO inputs GRID MIRAGE needs (no full 3.7 GB dataset).

checkpoint: gridfm/genco-pf-datakit case118_ieee/base/seed0 (~80 MB)
dataset:    gridfm/pf_small_case118_ieee @ pinned revision: all bus_data partitions
            (~0.8 GB, needed to replicate the official split) plus only the
            gen/branch partitions holding the requested scenarios (200 scenarios
            per partition, partition = scenario // 200, verified on this revision).
"""
import argparse
from huggingface_hub import snapshot_download

DATASET_REVISION = '3f5a6e266f759be15e3777bf4d70133ff40e6fa7'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='external-genco')
    ap.add_argument('--scenarios', type=int, nargs='+',
                    help='default: the scenarios recorded in results/genco_n50 and results/genco_pilot8')
    a = ap.parse_args()
    if not a.scenarios:
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1] / 'results'
        a.scenarios = sorted({s for run in ('genco_n50', 'genco_pilot8')
                              for s in json.loads((root / run / 'metadata.json').read_text())['scenarios']})
    snapshot_download('gridfm/genco-pf-datakit', allow_patterns=['case118_ieee/base/seed0/*', 'README.md'],
                      local_dir=f'{a.dest}/genco-pf-datakit')
    parts = sorted({s // 200 for s in a.scenarios})
    pats = ['README.md', 'args.log', 'n_scenarios.txt', 'bus_data.parquet/*']
    pats += [f'{t}_data.parquet/scenario_partition={p}/*' for t in ('gen', 'branch') for p in parts]
    snapshot_download('gridfm/pf_small_case118_ieee', repo_type='dataset', revision=DATASET_REVISION,
                      allow_patterns=pats, local_dir=f'{a.dest}/data/case118_ieee/raw', max_workers=16)
    print('partitions', parts)


if __name__ == '__main__':
    main()
