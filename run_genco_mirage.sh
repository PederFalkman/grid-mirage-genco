#!/usr/bin/env bash
# GRID MIRAGE on real GENCO predictions, small validated subset (no 250k processing).
# Inputs: either `python src/fetch_genco_inputs.py --dest external-genco` (checkpoint + bus_data +
# only the needed gen/branch partitions), or the layout created by run_official_genco.sh:
#   external-genco/gridfm-graphkit                     (branch genco-paper-repro, pip install -e)
#   external-genco/gridfm-graphkit/genco-pf-datakit/case118_ieee/base/seed0/{yaml,pt}
#   external-genco/gridfm-graphkit/data/case118_ieee/raw/{bus,gen,branch}_data.parquet
# Override with GRAPHKIT=..., CKPT_DIR=..., RAW_DIR=..., NUM_SAMPLES=..., VENV=...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
[[ -f "$VENV/bin/activate" ]] && source "$VENV/bin/activate"
GRAPHKIT="${GRAPHKIT:-$ROOT/external-genco/gridfm-graphkit}"
BASE="$ROOT/external-genco"; [[ -d "$BASE/genco-pf-datakit" ]] || BASE="$GRAPHKIT"
CKPT_DIR="${CKPT_DIR:-$BASE/genco-pf-datakit/case118_ieee/base/seed0}"
RAW_DIR="${RAW_DIR:-$BASE/data/case118_ieee/raw}"
OUT="${OUT:-$ROOT/results/genco_n50}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

python - "$CKPT_DIR" "$RAW_DIR" <<'PY'
import sys
from pathlib import Path
ckpt, raw = map(Path, sys.argv[1:])
for f in ['case118_ieee_base_seed0.yaml', 'best_model_state_dict.pt', 'normalizer_stats.pt']:
    assert (ckpt / f).is_file(), f'missing {ckpt / f}'
for f in ['bus_data.parquet', 'gen_data.parquet', 'branch_data.parquet']:
    assert (raw / f).exists(), f'missing {raw / f}'
import gridfm_graphkit, torch, torch_geometric, torch_scatter  # noqa: F401
print('inputs and dependencies present; torch', torch.__version__)
PY

python -m pytest -q "$ROOT/tests"
mkdir -p "$OUT"
python "$ROOT/src/genco_mirage.py" \
  --config "$CKPT_DIR/case118_ieee_base_seed0.yaml" \
  --model-path "$CKPT_DIR/best_model_state_dict.pt" \
  --normalizer-stats "$CKPT_DIR/normalizer_stats.pt" \
  --raw-dir "$RAW_DIR" \
  --num-samples "${NUM_SAMPLES:-50}" --permutations 5 --seed 2026 \
  --out "$OUT" 2>&1 | tee "$OUT/run.log"
python "$ROOT/src/summarize_genco.py" "$OUT"
python "$ROOT/src/plot_genco.py" "$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
echo "Done. Review $OUT before committing; *.npz is git-ignored, CSV/PNG/JSON are not."
