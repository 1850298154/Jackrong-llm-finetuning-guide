"""Ray Job driver for the Qwen3.6-27B GRPO training.

Submitted via ``ray job submit``, this script runs on the Ray cluster
(not on the submitter) and calls straight into the training module's
``_run_with_ray_train`` entry point, which creates a
``ray.train.torch.TorchTrainer`` with one worker per GPU.

All forwarded CLI args are the same as the ones accepted by
``Qwen3.6-27B-GRPO.py`` -- e.g.::

    python ray_worker.py --num_workers 8 --lora_rank 64
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

TRAIN_SCRIPT = Path(
    "/storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/"
    "train_code/Qwen3.6-27B-GRPO.py"
)

# Make sure we run inside the MCP-Atlas venv if it exists, so that
# `transformers>=5.6` + `trl>=1.2` are importable inside the Ray worker.
_VENV_PY = Path(
    "/storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/"
    "train/03_mcp/MCP-Atlas/venv/bin/python"
)
if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
    os.execv(str(_VENV_PY), [str(_VENV_PY), __file__, *sys.argv[1:]])


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location(
        "qwen3_grpo_trainer", str(TRAIN_SCRIPT)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mod = _load_trainer_module()
    args = mod._parse_args()
    mod._run_with_ray_train(args)


if __name__ == "__main__":
    main()
