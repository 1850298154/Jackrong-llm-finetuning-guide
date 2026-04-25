#!/usr/bin/env bash
# Tail the most recent Ray submission's status + logs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB=${1:-$(cat "$HERE/logs/latest_submission_id.txt")}
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
echo "[ray_tail] submission: $SUB"
echo "[ray_tail] status:"
ray job status --address "$RAY_ADDRESS" "$SUB" 2>&1 | tail -3
echo "[ray_tail] logs (tail -F):"
exec ray job logs --address "$RAY_ADDRESS" --follow "$SUB"
