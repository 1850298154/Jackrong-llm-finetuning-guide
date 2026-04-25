# 06 · 多 GPU 分布式训练全景（DP / ZeRO / FSDP / TP / PP）

> 你问得好："怎么利用 8 块 GPU"。这一篇把业界常见的 5 种并行方式**串起来讲
> 清楚**，并落到我们这次训练用到的 DeepSpeed ZeRO-3 上。

## 1. 五种并行方式一览

| 缩写 | 名字 | 切什么 | 主要开销 | 典型场景 |
|------|------|--------|---------|----------|
| **DP**   | Data Parallel              | 数据 batch               | AllReduce gradient | 小模型、多卡 |
| **ZeRO** | Zero Redundancy Optimizer  | optimizer/grad/权重 按 rank 切 | AllGather 权重 | 7B~70B 大模型 |
| **FSDP** | Fully Sharded Data Parallel| 同 ZeRO，PyTorch 原生    | 同上               | 同 ZeRO       |
| **TP**   | Tensor Parallel            | 单个 matmul 切行/列       | AllReduce activation | >100B 模型   |
| **PP**   | Pipeline Parallel          | 按 **层** 切              | 层间 activation 送      | 超大模型、跨节点 |

**组合拳**：现代大模型训练往往 `DP + ZeRO`（单节点）或 `DP + TP + PP`（多节点）。
LLaMA-2-70B 的 Meta 原论文用 `TP=8 × PP=4 × DP=4 = 128 GPU`。

## 2. 我们选 ZeRO-3 的理由

- **模型 27B，单机 8 卡**，不需要跨节点 PP。
- **TP 成本**：要动模型代码（split qkv_proj），LLaMA-Factory 不直接暴露。
- **ZeRO-3 开箱即用**：DeepSpeed 把参数"按 rank 切片 + 反向时 AllGather"，
  任何 HuggingFace 模型都能 transparently 跑。

## 3. ZeRO 三阶段直观对比（8 卡训 27B LoRA 为例）

假设每参数 FP16=2 byte，optim state FP32 ×2=8 byte，grad=2 byte。

| 阶段 | 权重        | grad        | optim       | 单卡 (LoRA 213M) | 单卡 (base 27B, frozen) |
|------|-------------|-------------|-------------|------------------|-------------------------|
| z0   | 完整复制 (2B) | 完整 (2B)     | 完整 (8B)     | 425M+425M+1.7G ≈ 2.5G | 54G+0+0 = 54G ❌            |
| z1   | 完整 (2B)    | 完整 (2B)    | 切 1/8 (1B)  | ~1.4G             | 54G                         |
| z2   | 完整 (2B)    | 切 1/8 (0.25B)| 切 1/8 (1B) | ~0.8G             | 54G                         |
| **z3** | **切 1/8 (0.25B)** | 切 1/8    | 切 1/8      | ~0.5G            | **6.75G** ✅                 |

所以对 LoRA 场景，**真正值钱的是 z3 把 frozen base 也切了**；单卡只扛
27B ÷ 8 ≈ 3.4G param × 2byte = 6.75 GB。加 activation 8K seq ≈ 22 GB，
总占 ~28 GB，40G 卡完全跑得动。

## 4. 启动方式对照表

| 启动器 | 何时用 | 例子 |
|--------|--------|-------|
| `python train.py`               | 单卡/调试       | 你在开发阶段 |
| `torchrun --nproc 8 train.py`   | 简单多卡 DDP    | LLaMA-Factory 默认 |
| `accelerate launch`             | 多卡 + DS/FSDP  | 本项目 GRPO 脚本 |
| `deepspeed train.py`            | DeepSpeed CLI   | 手写 DS scripts |

### 3 种都会用到的"隐形"环境变量

```
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
RANK=0..7
WORLD_SIZE=8
LOCAL_RANK=0..7
```

`torchrun` / `accelerate` / `deepspeed` 的本质**就是帮你填这些环境变量**然后
启 8 个 Python 进程。所以：**多卡训练报错 first step？** 第一反应查
`os.environ["RANK"] / ["LOCAL_RANK"]`。

## 5. 本项目的具体配置

### SFT (LLaMA-Factory)

```
FORCE_TORCHRUN=1 llamafactory-cli train  sft_qwen27b_lora_zero3.yaml
└── 内部 torchrun --nproc_per_node=$(num_gpus)
     └── train.py + DeepSpeed z3 (configs/ds_z3_bf16.json)
```

### GRPO (TRL)

```
accelerate launch --config_file configs/accelerate_ds_z3.yaml  train_grpo.py
└── 读 configs/accelerate_ds_z3.yaml:
     distributed_type: DEEPSPEED
     num_processes: 8
     deepspeed_config.deepspeed_config_file: configs/ds_z3_bf16.json
```

两个阶段**共用同一份 DeepSpeed JSON**。

## 6. vLLM 加速 rollout（GRPO 专用）

GRPO 的每一 step 都要 `generate` G × batch_size 次——这是瓶颈。
vLLM 的 PagedAttention + Continuous Batching 能让这部分快 3~5 倍。

TRL 集成 vLLM 有两种模式：

1. **`vllm_mode="server"`**（推荐）：
   单独开一个 `trl vllm-serve` 进程，占 1-2 张 GPU 专门做 rollout，其余 GPU
   继续训练。训练进程通过 HTTP 发 prompt 收 completion。
2. **`vllm_mode="colocate"`**：
   和训练共享 GPU。省 GPU 数，但峰值显存难控，容易 OOM。

我们的脚本默认 **不开** vLLM（`use_vllm=False`），先保证能跑；稳定后加
`--use-vllm` 打开。

## 7. 常见性能指标 (27B + 8×A100-40G 估算)

| 阶段 | step 耗时 | throughput | 备注 |
|-----|-----------|------------|------|
| SFT, bs=1, seq=8K | 2.5 s | ~1.3 K tok/s/gpu | gradient_checkpointing=on |
| GRPO, G=4, max_c=2K | 40 s | 200 tok/s/gpu (no vLLM) | generate 是瓶颈 |
| GRPO, G=4, max_c=2K | 8 s | 1 K tok/s/gpu (vLLM) | 一张卡做 vLLM server |

## 8. 最少你要记住的 5 件事

1. 单机 27B → **ZeRO-3 + LoRA**，别想全参。
2. ZeRO 配置 JSON 写死 `stage: 3`, `bf16: true`, `stage3_gather_16bit_weights_on_model_save: true`。
3. `torchrun / accelerate / deepspeed` 只是启动器，**本质是填环境变量**。
4. SFT 瓶颈在 forward/backward，**GRPO 瓶颈在 generate**——用 vLLM 救。
5. 出错先看 `rank0` 日志，然后查 `NCCL_DEBUG=INFO` 看是哪一步超时/通信挂。

下一篇 → `07_pitfalls.md`
