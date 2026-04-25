#!/usr/bin/env bash
# Submit Qwen3.6-27B GRPO as a Ray Job.
#
# Usage:
#   ./ray_submit.sh                         # uses RAY_ADDRESS or http://127.0.0.1:8265
#   RAY_ADDRESS=http://HEAD:8265 ./ray_submit.sh [extra args...]
#
# The dispatched job uses a `ray.train.torch.TorchTrainer` with one worker
# per GPU (controlled via --num_workers; -1 = all GPUs Ray reports).  Each
# worker gets 1 GPU + an even share of the cluster CPUs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"

STAMP="$(date +%Y%m%d-%H%M%S)"
SUB_ID="${SUB_ID:-qwen3-27b-grpo-$STAMP}"

# Build the runtime env: we only need env vars; code & venv live on shared disk.
RUNTIME_ENV=$(python3 -c '
import json, os, sys
here = sys.argv[1]
print(json.dumps({
    "env_vars": {
        "HF_HOME": f"{here}/hf_cache",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
}))
' "$HERE")

mkdir -p "$HERE/hf_cache" "$HERE/logs" "$HERE/ray_results"
LOG_FILE="$HERE/logs/submit-$STAMP.log"
echo "$SUB_ID" > "$HERE/logs/latest_submission_id.txt"

echo "[ray_submit] RAY_ADDRESS=$RAY_ADDRESS"
echo "[ray_submit] submission id: $SUB_ID"
echo "[ray_submit] log file: $LOG_FILE"

exec ray job submit \
    --address "$RAY_ADDRESS" \
    --submission-id "$SUB_ID" \
    --no-wait \
    --runtime-env-json "$RUNTIME_ENV" \
    -- python "$HERE/ray_worker.py" "$@" 2>&1 | tee "$LOG_FILE"
