#!/usr/bin/env bash
# Submit the Qwen3.6-27B GRPO run to the KubeRay cluster.
#
# Usage:
#   ./submit.sh                 # default: all 32 GPUs, 3000 max steps
#   ./submit.sh --max_steps 100 # forward any extra flag to the training CLI
#
# You can override the Ray endpoint:
#   RAY_ADDRESS=http://<head>:8265 ./submit.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export RAY_ADDRESS="${RAY_ADDRESS:-http://172.17.193.214:8265}"
echo "[submit] RAY_ADDRESS=$RAY_ADDRESS"

NOWAIT_FLAG=""
if [[ "${NO_WAIT:-0}" == "1" ]]; then
    NOWAIT_FLAG="--no-wait"
fi

# Ray uploads this directory (~4 files, <10 KB) as the working_dir and
# applies runtime_env.yaml to every worker.  The actual model/dataset are
# read from the shared /storage volume at runtime -- no upload.
exec ray job submit \
    --working-dir "$HERE" \
    --runtime-env "$HERE/runtime_env.yaml" \
    $NOWAIT_FLAG \
    -- python ray_driver.py "$@"
