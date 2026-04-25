#!/usr/bin/env bash
# Create the python venv we use for GRPO training.
# transformers==5.6.2 gives us native Qwen3.6-27B (`qwen3_5`) support.
# trl==1.2.0 is the matching GRPO implementation.  We keep torch from the
# system site-packages (pre-built with flash_attn + vllm bindings).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/venv"
UV=${UV:-/home/z/.local/bin/uv}

"$UV" venv --system-site-packages --python 3.12 "$VENV"
"$UV" pip install --python "$VENV/bin/python" --no-deps \
    "transformers==5.6.2" "trl==1.2.0"
"$UV" pip install --python "$VENV/bin/python" \
    "huggingface_hub>=1.5,<2" "tokenizers>=0.22,<0.23" "safetensors>=0.7"

"$VENV/bin/python" - <<'PY'
import transformers, trl, torch, peft, accelerate, datasets
print("transformers:", transformers.__version__)
print("trl         :", trl.__version__)
print("torch       :", torch.__version__)
print("peft        :", peft.__version__)
print("accelerate  :", accelerate.__version__)
print("datasets    :", datasets.__version__)
PY
