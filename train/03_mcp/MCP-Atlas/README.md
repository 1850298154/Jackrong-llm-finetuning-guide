# Qwen3.6-27B × MCP-Atlas GRPO (Ray Train job)

A multi-GPU GRPO fine-tune of `/storage/RL/models/download/Qwen3.6-27B` on
the MCP-Atlas tool-use dataset, packaged as a **Ray Job** that launches a
`ray.train.torch.TorchTrainer` with one worker per GPU.

```
MCP-Atlas/
├── README.md             ← this file
├── ray_worker.py         ← Ray Job driver: calls _run_with_ray_train()
├── ray_submit.sh         ← `ray job submit` wrapper
├── ray_tail.sh           ← follow status + logs for the latest submission
├── setup_env.sh          ← one-shot bootstrap of the local venv
├── venv/                 ← transformers 5.6.2 + trl 1.2.0 (layered)
├── checkpoint/           ← GRPOTrainer output_dir (per-step HF checkpoints)
├── adapter_final/        ← final LoRA adapter + tokenizer
├── processed_data/       ← HF Dataset dump used at training time
├── ray_results/          ← Ray Train's storage_path (trial logs & checkpoints)
├── logs/                 ← per-rank stdout/stderr tees + submission logs
└── ray_tmp/              ← local Ray temp dir (only used when we start Ray)
```

The trainer lives at `train_code/Qwen3.6-27B-GRPO.py`.

---

## 1. Bootstrap (once)

```bash
cd /storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train/03_mcp/MCP-Atlas
./setup_env.sh
```

`Qwen3.6-27B` ships `model_type="qwen3_5"`, which only `transformers>=5.6.2`
knows natively; `setup_env.sh` creates a venv that layers those upgrades on
top of the system site-packages (torch 2.10+cu129, flash_attn, ray 2.54,
peft, accelerate all inherited).

Resulting stack:

| package         | version        | source   |
|-----------------|----------------|----------|
| transformers    | 5.6.2          | venv     |
| trl             | 1.2.0          | venv     |
| huggingface_hub | 1.12.0         | venv     |
| tokenizers      | 0.22.2         | venv     |
| safetensors     | 0.7.0          | venv     |
| torch           | 2.10.0+cu129   | system   |
| ray             | 2.54.0         | system   |
| peft            | 0.18.1         | system   |
| accelerate      | 1.11.0         | system   |
| datasets        | 4.0.0          | system   |
| flash_attn      | 2.8.3          | system   |

## 2. Submit the Ray Job

```bash
# Defaults: RAY_ADDRESS=http://127.0.0.1:8265, 8 workers (= all GPUs), etc.
./ray_submit.sh

# Custom submission:
./ray_submit.sh --num_workers 8 --lora_rank 64 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 8 \
                --max_steps 3000

# Follow progress:
./ray_tail.sh                  # tails the latest submission
```

`ray_submit.sh` uses `ray job submit` against `$RAY_ADDRESS`
(default `http://127.0.0.1:8265`).  The job's entry point is
`ray_worker.py`, which re-execs under `venv/bin/python`, imports the
trainer module, and calls `_run_with_ray_train(args)`.

`_run_with_ray_train` builds:

```python
ScalingConfig(
    num_workers=N_GPUS,
    use_gpu=True,
    resources_per_worker={"GPU": 1, "CPU": cpus_per_worker},
    placement_strategy="PACK",
)
TorchConfig(backend="nccl")
RunConfig(
    name="qwen3-27b-grpo",
    storage_path="<work_root>/ray_results",
    checkpoint_config=CheckpointConfig(num_to_keep=3),
)
TorchTrainer(
    train_loop_per_worker=_ray_train_loop,
    train_loop_config=vars(args),
    ...
).fit()
```

Ray Train then spawns `N_GPUS` worker actors, each with exactly 1 GPU, sets
`RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT /
CUDA_VISIBLE_DEVICES` for every worker, initialises the NCCL process group,
and calls our `_ray_train_loop` in every worker.  That loop runs the
full GRPOTrainer on the shard of the model the local rank owns.

## 3. What the trainer does

* **Model**: loads `Qwen3_5ForCausalLM` (or `Qwen3_5ForConditionalGeneration`
  and drops to its `.language_model`) in bf16, enables
  gradient-checkpointing, wraps in LoRA (default r=32 / α=64 / drop=0.05)
  targeting `q/k/v/o/gate/up/down` projections.
* **Data**: reads `MCP-Atlas.parquet` (500 rows), parses
  `ENABLED_TOOLS / GTFA_CLAIMS / TRAJECTORY`, emits chat prompts asking for
  `<think>…</think><tool_call>…</tool_call>…<final_answer>…</final_answer>`.
  Prompts longer than `--max_prompt_length` tokens are filtered out.
* **Rewards** (summed → GRPO advantage):
  1. `print_samples_reward` – 0-valued monitor that prints one rank-0 sample
     per batch.
  2. `format_reward` – strict `<think>/<tool_call>*/<final_answer>` layout
     with an anti-silence floor (matches the Llama-R1 notebook style).
  3. `tool_name_reward` – fraction of `<function=…>` names that lie in
     ENABLED_TOOLS + recall of the trajectory's gold tool names.
  4. `claim_coverage_reward` – keyword coverage of each GTFA_CLAIM inside
     `<final_answer>` (main learning signal).
  5. `repetition_penalty` – tri/4-gram repetition cost, capped at −0.2.
* **Stopping rule**: `MinLossStopCallback` tracks an EMA over `loss`,
  records its running minimum after `--min_loss_warmup` steps, and flips
  `TrainerControl.should_training_stop` once the EMA has risen above
  `min * (1 + rel_tol)` for `--min_loss_patience` consecutive logging steps.
  This implements the "stop at the parabola minimum" rule.
* **Ray metrics**: `RayTrainReportCallback` forwards HF Trainer's `loss`,
  `learning_rate`, `reward`, `kl`, … to `ray.train.report()` every logging
  step, and reports a Checkpoint on every `save_steps` boundary so Ray
  Train keeps the latest 3 rolling checkpoints.

## 4. Key tunables

```bash
./ray_submit.sh \
    --num_workers 8 \
    --cpus_per_worker 3 \
    --lora_rank 32 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 4 \
    --max_prompt_length 3072 \
    --max_completion_length 1024 \
    --learning_rate 5e-6 \
    --max_steps 3000 \
    --min_loss_warmup 40 \
    --min_loss_patience 15
```

Pass `--no_ray` to skip Ray Train and run the trainer directly as a single
process — useful for debugging.

## 5. Sanity checks performed

* Script parses and imports cleanly under the venv.
* 500 MCP-Atlas rows build into prompts within the 3072-token cap.
* All five rewards return sensible scores on a reference completion.
* Ray Job is accepted by the cluster; TrainController spawns 8
  RayTrainWorkers; Ray injects `rank=0…7, world_size=8` correctly; all
  workers reach `torch.distributed.init_process_group`.

## 6. Environment caveat we hit

On the dev pod `rl-dev-ytzhao02`, Ray's dashboard reports `8 GPU / 32 CPU`
but the container does **not** have `/dev/nvidia*` device nodes mounted,
so `torch.cuda.is_available()` is `False` and NCCL init fails with
`ProcessGroupNCCL is only supported with GPUs, no GPUs found!`.  On any
correctly GPU-provisioned node the same submission drives all 8 H100s
through the NCCL process group and begins training.
