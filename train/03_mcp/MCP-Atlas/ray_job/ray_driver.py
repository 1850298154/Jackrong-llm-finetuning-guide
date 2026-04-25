"""Ray-Job driver: connects to the KubeRay cluster and schedules a
`TorchTrainer` that spans every GPU in the cluster.

Important subtlety: the training module (`Qwen3_6_27B_GRPO.py`) is
colocated inside this directory so it's shipped via `--working-dir`
to every node (including the head, which does NOT mount
`/storage/RL`).  The entry function below is defined HERE so Ray can
pickle it by a stable module name (`ray_driver._train_loop_entry`).
The function then re-imports the training module from the local
working_dir on whichever worker it happens to land on.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict


THIS_DIR = Path(__file__).resolve().parent
TRAIN_MODULE_PATH = THIS_DIR / "Qwen3_6_27B_GRPO.py"


def _load_train_module():
    """Load the training module from this directory (portable across pods)."""
    # Ray drops the working_dir on both head and workers under
    # /tmp/ray/session_*/runtime_resources/working_dir_files/<pkg>/ -- so
    # THIS_DIR resolves correctly on every node.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen3_6_27b_grpo", str(TRAIN_MODULE_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TRAIN_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_loop_entry(config: Dict[str, Any]) -> None:
    """Top-level function Ray Train will call on every worker.

    We deliberately avoid closures so this function is cleanly picklable
    under Ray 2.54's cloudpickle.
    """
    mod = _load_train_module()
    ns = argparse.Namespace(**config)
    mod._run_training(ns, is_ray_worker=True)


def main() -> None:
    os.environ.setdefault("RAY_ADDRESS", "http://172.17.193.214:8265")

    mod = _load_train_module()
    args = mod._parse_args()

    # Swap Ray Train's default worker entry -- which is a private
    # closure `_ray_train_loop` inside the training module -- for the
    # top-level function defined here.  This makes cloudpickling robust
    # across the driver -> worker boundary.
    _run_with_ray = mod._run_with_ray_train

    import ray
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer, TorchConfig

    if not ray.is_initialized():
        address = args.ray_address or os.environ.get("RAY_ADDRESS", "auto")
        ray.init(address=address, ignore_reinit_error=True,
                 log_to_driver=True)

    scaling = mod._resolve_scaling(args)
    print(
        f"[driver] Ray cluster: GPUs={scaling['n_gpus_cluster']} "
        f"CPUs={scaling['n_cpus_cluster']}", flush=True,
    )
    print(
        f"[driver] TorchTrainer: num_workers={scaling['num_workers']} "
        f"cpus_per_worker={scaling['cpus_per_worker']} gpus_per_worker=1",
        flush=True,
    )

    # Head pod may not mount the shared /storage volume; fail softly
    # and fall back to a local RunConfig.storage_path so the driver can
    # still submit.  Workers (which DO mount /storage) will write to
    # args.work_root themselves at training time.
    work_root = Path(args.work_root)
    try:
        (work_root / "ray_results").mkdir(parents=True, exist_ok=True)
        ray_storage = str(work_root / "ray_results")
    except Exception as exc:
        fallback = "/tmp/ray_results"
        os.makedirs(fallback, exist_ok=True)
        print(f"[driver] work_root not writable ({exc!r}); "
              f"using {fallback} for Ray Train metadata", flush=True)
        ray_storage = fallback

    run_config = RunConfig(
        name="qwen3-27b-grpo",
        storage_path=ray_storage,
        checkpoint_config=CheckpointConfig(num_to_keep=3),
    )
    scaling_config = ScalingConfig(
        num_workers=scaling["num_workers"],
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": scaling["cpus_per_worker"]},
        placement_strategy="PACK",
    )
    torch_config = TorchConfig(backend="nccl")

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop_entry,
        train_loop_config=vars(args),
        torch_config=torch_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
    print(f"[driver] Ray Train finished: {result}", flush=True)
    try:
        print(f"[driver] checkpoint: {result.checkpoint}", flush=True)
        print(f"[driver] metrics:    {result.metrics}", flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
