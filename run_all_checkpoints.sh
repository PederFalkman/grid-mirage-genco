#!/usr/bin/env bash
# GRID MIRAGE across released GENCO PF checkpoints (IEEE 14 / 118 x tiny / small / base, seed 0).
# Inputs: python src/fetch_genco_inputs.py --dest external-genco   (GraphKit installed separately)
# Each run: 50 official test scenarios x 7 load levels x (original + 5 relabelings), float32 + float64.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-$ROOT/external-genco}"
N="${NUM_SAMPLES:-50}"
cd "$ROOT"
python -m pytest -q tests
for case in case14 case118; do
  for size in tiny small base; do
    C="$BASE/genco-pf-datakit/${case}_ieee/$size/seed0"
    RAW="$BASE/data/${case}_ieee/raw"
    OUT="results/genco_${case}_${size}_n${N}"
    [[ "$case" == case118 && "$size" == base ]] && OUT="results/genco_n${N}"   # main run, historical name
    mkdir -p "$OUT"
    python src/genco_mirage.py --config "$C/${case}_ieee_${size}_seed0.yaml" --model-path "$C/best_model_state_dict.pt" \
      --normalizer-stats "$C/normalizer_stats.pt" --raw-dir "$RAW" --num-samples "$N" --permutations 5 --seed 2026 \
      --out "$OUT" 2>&1 | tee "$OUT/run.log"
    python src/summarize_genco.py "$OUT"
    python src/hybrid_mirage.py "$OUT" --raw-dir "$RAW"
    python src/plot_genco.py "$OUT"
  done
done
python src/compare_checkpoints.py
python src/plot_schematic.py results/genco_n50/schematic.png
