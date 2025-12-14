#!/bin/bash

  SCENARIO_DIR="opencdascenarios"

  for zipfile in "$SCENARIO_DIR"/*.zip; do
      name=$(basename "$zipfile" .zip)
      echo "Running scenario $name"
    
  python tools/run_custom_eval.py \
    --zip "$zipfile" \
    --scenario-name "$name" \
    --results-tag "${name}_lmdrive" \
    --agent simulation/leaderboard/team_code/lmdriver_agent.py \
    --agent-config simulation/leaderboard/team_code/agent_config/lmdriver_config_8_10.py \
    --port 4833 \
    --overwrite
done