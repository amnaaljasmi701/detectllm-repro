#!/usr/bin/env bash
# One-command reproduction. Usage: bash run_all.sh [smoke|quick|full] [output_dir]
set -euo pipefail
PRESET=${1:-full}
OUT=${2:-results/$PRESET}
echo "== unit tests"
python -m pytest -q tests
echo "== experiments ($PRESET) -> $OUT"
python scripts/run.py --preset "$PRESET" --out "$OUT"
echo "== tables, figures, error analysis"
python scripts/aggregate.py --results "$OUT"
echo "Done. See $OUT/summary/"
