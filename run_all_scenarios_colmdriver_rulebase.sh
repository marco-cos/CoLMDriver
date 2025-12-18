#!/bin/bash

SCENARIO_DIR="opencdascenarios"

echo "Running 12 scenarios for COLMDRIVER_RULEBASE..."
echo "========================================"

# Loop through all zip files in the scenario directory
for zipfile in "$SCENARIO_DIR"/*.zip; do
    name=$(basename "$zipfile" .zip)
    
    # Safety check: exit if no zip files are found
    if [ ! -e "$zipfile" ]; then
        echo "Error: No scenario zip files found in $SCENARIO_DIR. Please upload them."
        exit 1
    fi

    echo "Starting scenario: $name"

    # --overwrite is included to prevent errors when re-running evaluation on the same tag.
    python tools/run_custom_eval.py \
      --zip "$zipfile" \
      --scenario-name "$name" \
      --results-tag "${name}_rulebase_final" \
      --agent simulation/leaderboard/team_code/colmdriver_agent.py \
      --agent-config simulation/leaderboard/team_code/agent_config/colmdriver_rulebase_config.yaml \
      --port 2002 \
      --overwrite
    
    echo "Finished $name"
    echo "----------------------------------------"
done
