#!/usr/bin/env bash
# 实时查看 GRPO 训练 loss / reward / 步速
# 用法: bash watch.sh [刷新间隔秒, 默认 30]
set -euo pipefail
export RAY_ADDRESS=${RAY_ADDRESS:-http://172.17.193.214:8265}
SUB=${SUB:-qwen36-27b-grpo-20260425-230652}
INTERVAL=${1:-30}

extract() {
  ray job logs $SUB 2>&1 | python3 -c "
import sys, re
pat = re.compile(r\"'step': (\d+), 'epoch': ([\d.]+), 'loss': (-?[\d.e+-]+), .*?'reward': (-?[\d.e+-]+), 'reward_std': (-?[\d.e+-]+)\")
seen = {}
for line in sys.stdin:
    for m in pat.finditer(line):
        step = int(m.group(1))
        seen[step] = (step, float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)))
for step in sorted(seen)[-12:]:
    s, ep, loss, rew, std = seen[step]
    print(f'step {s:>4}  epoch {ep:5.2f}  loss {loss:+9.4f}  reward {rew:6.3f} \u00b1{std:5.3f}')
"
}

while true; do
  clear
  echo "========== $(date '+%F %T')  job=$SUB =========="
  ray job status $SUB 2>&1 | grep -E "Status|running|SUCC|FAIL"
  echo ""
  echo "--- 进度 (step / 3000) ---"
  ray job logs $SUB 2>&1 | grep -oE "[0-9]+/3000 \[[^]]+\]" | tail -3
  echo ""
  echo "--- 最近 12 步 loss / reward ---"
  extract
  echo ""
  echo "(Ctrl-C 退出; 每 ${INTERVAL}s 刷新一次)"
  sleep "$INTERVAL"
done
