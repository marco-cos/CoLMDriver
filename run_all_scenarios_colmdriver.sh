#!/bin/bash
SCENARIO_DIR="opencdascenarios"

for zipfile in "$SCENARIO_DIR"/*.zip; do
  name=$(basename "$zipfile" .zip)
  echo "Running scenario $name with CoLMDriver"

  python tools/run_custom_eval.py \
    --zip "$zipfile" \
    --scenario-name "$name" \
    --results-tag "${name}_colmdriver" \
    --port 2000 \
    --overwrite
done
