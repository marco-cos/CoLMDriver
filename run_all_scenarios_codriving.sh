#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCENARIO_DIR="${REPO_ROOT}/opencdascenarios"

# ===== CODRIVING baseline settings =====
AGENT_CODRIVING="simulation/leaderboard/team_code/pnp_agent_e2e_v2v.py"
AGENT_CONFIG_CODRIVING="simulation/leaderboard/team_code/agent_config/pnp_config_codriving_5_10.yaml"
PORT_CODRIVING=2600
# ======================================

if [ -f "/data/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/data/miniconda3/etc/profile.d/conda.sh"
fi
conda activate colmdrivermarco2

cd "$REPO_ROOT"

rm -rf simulation/leaderboard/data/CustomRoutes/*

for zipfile in "${SCENARIO_DIR}"/*.zip; do
    name=$(basename "$zipfile" .zip)
    echo "===================================================="
    echo "Running CODRIVING baseline on scenario: $name"
    echo "Zip: $zipfile"
    echo "===================================================="

    python tools/run_custom_eval.py \
        --zip "$zipfile" \
        --scenario-name "$name" \
        --results-tag "${name}_codriving" \
        --agent "$AGENT_CODRIVING" \
        --agent-config "$AGENT_CONFIG_CODRIVING" \
        --port $PORT_CODRIVING \
        --overwrite
done
