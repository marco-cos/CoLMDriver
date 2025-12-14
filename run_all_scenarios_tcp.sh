#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SCENARIO_DIR="${REPO_ROOT}/opencdascenarios"

# ===== TCP baseline settings =====
AGENT_TCP="simulation/leaderboard/team_code/tcp_agent.py"
AGENT_CONFIG_TCP="simulation/leaderboard/team_code/agent_config/tcp_5_10_config.yaml"
PORT_TCP=2600
# =================================

# Activate env from /data
if [ -f "/data/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/data/miniconda3/etc/profile.d/conda.sh"
fi
conda activate colmdrivermarco2

cd "$REPO_ROOT"

# Optional: clear any previously unpacked routes
rm -rf simulation/leaderboard/data/CustomRoutes/*

for zipfile in "${SCENARIO_DIR}"/*.zip; do
    name=$(basename "$zipfile" .zip)
    echo "===================================================="
    echo "Running TCP baseline on scenario: $name"
    echo "Zip: $zipfile"
    echo "===================================================="

    python tools/run_custom_eval.py \
        --zip "$zipfile" \
        --scenario-name "$name" \
        --results-tag "${name}_tcp" \
        --agent "$AGENT_TCP" \
        --agent-config "$AGENT_CONFIG_TCP" \
        --port $PORT_TCP \
        --overwrite
done
