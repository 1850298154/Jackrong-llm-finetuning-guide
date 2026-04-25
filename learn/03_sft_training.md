# 03 · SFT 训练：LLaMA-Factory + LoRA + DeepSpeed ZeRO-3（8×GPU）

> 目标：把 Qwen3.6-27B 用 181 条 MCP-Atlas 轨迹做 **监督微调**，
> 产出 LoRA adapter 放在
> `/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/checkpoint/sft/`。

## 1. 为什么是 LoRA（不是全参）

27B × 2 bytes（bf16 权重）≈ **54 GB**，每个参数还需要：
- optimizer state（AdamW 2 × fp32 = 8 byte/param）
- gradient（bf16 = 2 byte/param）
- activation（取决于 seq_len 和 checkpointing）

全参训练 27B ≈ **540 GB+** 峰值显存。即使 ZeRO-3 八卡均摊，每卡仍要 ~70 GB +
activation，H100 80G 很极限，A100 40G 直接不行。

**LoRA 方案**：冻结原始 27B 权重（只前向，不反传），在每个 linear 上挂一对
`A ∈ R^{d×r}, B ∈ R^{r×d}`（r=64），反向图只经过这些小矩阵：
- 可训练参数 ~ 200M（27B 的 0.7%）
- optimizer state ~ 1.6 GB
- 前向时原权重可以 8-bit / 4-bit 量化，**单卡 40G 就够**

实测在 8 × A100-40G 上，27B + LoRA64 + bf16 + ZeRO-3 + seq 8192，峰值 ≈ 28 GB/卡。

## 2. 为什么是 DeepSpeed ZeRO-3

DeepSpeed ZeRO 把"本来每卡都完整持有一份"的三样东西切到 N 张卡：

| 阶段 | 切分对象                   | 显存省多少      |
|-----|----------------------------|-----------------|
| z1  | optimizer state            | 4× → 1/8 × 4×   |
| z2  | z1 + gradient              | +gradient / N  |
| z3  | z2 + **fp16/bf16 权重**    | 全切，最省      |

**LoRA 场景下 z2 就够**，但我们有 27B 的 frozen base：base 权重也能放到
ZeRO-3 里被切 → 每卡只持 1/8 的 base = 6.75 GB。**所以 LoRA + ZeRO-3 依然有意义**。

FSDP 是等价选项。选 DeepSpeed 因为 LLaMA-Factory 默认集成、配置成熟。

## 3. 超参设计（抓"loss 抛物线最低点"）

```
global_batch_size = per_dev_bs × grad_accum × n_gpus = 1 × 4 × 8 = 32
steps_per_epoch   = 181 / 32 ≈ 6
num_epochs        = 8              → 总共 ~48 optimizer steps
lr                = 1e-4           LoRA 标配（比全参大 10×）
warmup_steps      = 20             前 ~3 epoch warmup
scheduler         = cosine         平滑降到 0
weight_decay      = 0.01
```

**为什么 epochs=8 看上去很多？**
- 数据只有 181 条，1 epoch 约 6 optimizer step，远远没收敛。
- 多跑几轮让 eval loss 走完"下降 → 最低点 → 过拟合上升"的抛物线。
- 配合 `load_best_model_at_end=true` + `metric=eval_loss`，训练完自动
  把权重回滚到最低点 ckpt。这样你**不用盯曲线**。

## 4. 关键文件

```
configs/
├── ds_z3_bf16.json                 # DeepSpeed ZeRO-3 配置
└── sft_qwen27b_lora_zero3.yaml     # LLaMA-Factory 主配置
code/run_sft.sh                     # 一键启动脚本
```

`ds_z3_bf16.json` 的关键字段：
```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "stage3_gather_16bit_weights_on_model_save": true,   // 保存时合并权重
    "overlap_comm": true,                                 // 通信和计算重叠
    "contiguous_gradients": true
  }
}
```

`sft_qwen27b_lora_zero3.yaml` 的关键字段：

```yaml
# Method
finetuning_type: lora
lora_target: all          # 命中 q/k/v/o/gate/up/down 全部
lora_rank: 64
lora_alpha: 128

# Data
template: qwen            # 自动用 Qwen 的 chat_template
cutoff_len: 8192
packing: false            # tools 数据 packing 会破坏 tool_response 对位关系

# Train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
num_train_epochs: 8
lr_scheduler_type: cosine
warmup_steps: 20
bf16: true
gradient_checkpointing: true
flash_attn: auto

# Eval（抓最低点）
do_eval: true
eval_strategy: epoch
save_strategy: epoch          # 必须和 eval_strategy 相同
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false

# DS
deepspeed: .../ds_z3_bf16.json
```

## 5. 启动

```bash
cd /storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas
bash code/run_sft.sh
```

`run_sft.sh` 内部：
```bash
export FORCE_TORCHRUN=1   # 让 llamafactory-cli 启 torchrun
llamafactory-cli train configs/sft_qwen27b_lora_zero3.yaml
```

LLaMA-Factory 检测到 `FORCE_TORCHRUN=1` 会自动：
```
torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) \
         --nnodes=1 train.py <yaml>
```
所以**不用手写 torchrun**。

## 6. 预期日志

```
[INFO] num_gpus = 8
[INFO] Trainable params: 213M / 27.0B (0.79%)
[INFO] ZeRO stage 3, bf16, gradient_checkpointing=True
***** Running training *****
  Num examples = 181
  Num Epochs = 8
  Total optimization steps = 48
  Logging every 5 steps

Step  5/48  loss 1.82
Step 10/48  loss 1.34
Step 15/48  loss 1.01
Step 20/48  loss 0.83
Step 24/48  [eval] eval_loss 0.77  (epoch 4 end)
Step 25/48  loss 0.68
...
Step 48/48  [eval] eval_loss 0.71  (epoch 8 end)
[INFO] Loading best checkpoint ... (epoch 6, eval_loss 0.65)
```

## 7. 产物

```
checkpoint/sft/
├── adapter_config.json
├── adapter_model.safetensors   # LoRA 权重 ≈ 400 MB
├── tokenizer.json
├── trainer_state.json          # 含每一步 loss / eval_loss
├── all_results.json
├── training_loss.png           # plot_loss=true 自动生成
└── checkpoint-12/, checkpoint-24/, ... (最多 3 个)
```

## 8. 推理 / 继续用这个 adapter

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "/storage/RL/models/download/Qwen3.6-27B",
    torch_dtype="bfloat16", device_map="auto",
)
model = PeftModel.from_pretrained(
    base, ".../checkpoint/sft"
)
# 或 merge 后直接当完整模型用：
# model = model.merge_and_unload()
```

## 9. 踩坑 & 学习心得

1. **`save_strategy` 必须和 `eval_strategy` 相同**，否则
   `load_best_model_at_end` 报错（transformers 5.2 把校验抓得很严）。
2. **`warmup_ratio` 已废弃**，换成 `warmup_steps`。
3. **`packing=True` 对工具调用数据是灾难**：packing 会把多个样本首尾相接，
   assistant 的 `<tool_call>` 和后面的 `<tool_response>` 可能被错位，损失
   对齐完全乱。记住：有 role=tool 的数据集，**关 packing**。
4. **`lora_target=all` 优于只打 q,v**。推理时 LoRA 合并开销一样，训练的
   representational capacity 大很多。
5. **27B 模型保存时合并权重**：务必开
   `stage3_gather_16bit_weights_on_model_save=true`；否则 ZeRO-3 保存的
   是每卡的 shard，loader 读不回来。
6. **日志里 `DeepSpeed` 警告 "CPU operator detected"** 大多无害，是
   bitsandbytes/deepspeed 在无 GPU 环境初始化时的友好降级。

下一篇 → `04_mcp_env.md`（GRPO 要用的模拟 MCP 环境）
