# 05 · GRPO 训练：TRL + 自定义 reward + SFT 热启动

> 目标：以 SFT 得到的 LoRA adapter 为初始 policy，用 GRPO 在 MCP 模拟环境
> 上继续优化 Qwen3.6-27B，输出放在
> `/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/checkpoint/grpo/`。

## 1. GRPO 是什么（30 秒版）

**Group Relative Policy Optimization**，DeepSeek-R1 同款算法。核心：

1. 对每个 prompt 生成 **G 个 completion**（group）
2. 算每个 completion 的 reward `r_i`
3. 组内**归一化**：`A_i = (r_i - mean(r)) / std(r)` （advantage）
4. 对 policy 梯度用 PPO 式 clipped-surrogate 更新，KL 拉回 reference 模型

**为什么优于 PPO**：无需单独的 value model（省一半显存）；advantage 直接从同
prompt 的组内对比来，reward 尺度鲁棒。

## 2. 我们的 reward（回顾）

见 `04_mcp_env.md`。总 reward =
```
format_reward ∈ [-0.5, 0.8]        # <tool_call>/<think>/收尾文本
+ tool_legality ∈ [-∞, +1]         # 工具名合法性
+ claim_coverage ∈ [0, 2]          # gold claims 命中比例
```

TRL `GRPOTrainer` 支持把多个 reward function 作为 list 传入，并通过
`reward_weights=[1,1,1]` 直接加权求和。

## 3. 关键超参选择

```python
GRPOConfig(
    per_device_train_batch_size = 1,        # 27B + 8K prompt，一张卡只塞 1
    gradient_accumulation_steps = 4,        # eff_batch=1*4*8=32 prompts
    num_generations            = 4,          # G，每 prompt 生成 4 个 completion
    max_prompt_length          = 2048,
    max_completion_length      = 2048,
    learning_rate              = 5e-6,       # 比 SFT 低 20×，RL 必须小步
    num_train_epochs           = 2,
    beta                       = 0.04,       # KL 系数；太大(>0.1)会"锁死"policy
    temperature                = 0.9,        # 采样温度
    loss_type                  = "dapo",     # TRL 0.24 默认，DAPO 更稳定
    bf16                       = True,
    gradient_checkpointing     = True,
)
```

**要点**：
- `num_generations` 越大学得越稳，但显存/时长也成倍增加。**G=4 是最小可用**。
- `beta`：GRPO 会算 `KL(policy || sft_ref)`。`beta=0.04` 是 DeepSeek-R1 常用
  值，既保留 reward 信号又不跑偏。
- `loss_type="dapo"`：TRL 0.24 默认。相对传统 GRPO，DAPO 用了 token-level
  重要性采样，长样本更稳。

## 4. 代码骨架

见 `code/grpo/train_grpo.py`，关键片段：

```python
# 1. 加载 SFT LoRA
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=bf16)
model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)

# 2. 数据：每条是 {"prompt": [system, user], "enabled_tools": [...], "gold_claims": [...]}
ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")

# 3. reward 三件套（TRL 自动把 dataset 的非 prompt 列透传进 kwargs）
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, tool_legality_reward, claim_coverage_reward],
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
)
trainer.train()
```

## 5. Rollout 是单轮还是多轮

**TRL 的 GRPOTrainer 原生单轮 rollout**：
```
prompt → generate 完整 completion → 算 reward → 更新
```

完整多轮 tool-use（模型 tool_call → env → tool_response → 模型继续）
在当前 TRL 里需要**自定义 `rollout` hook**。我们这版**先做单轮**：
- 训练数据 prompt = `[system(含工具清单), user(任务)]`
- 模型一次性生成 `<think>...</think><tool_call>...</tool_call>...答案` 一整段
- reward 看 **格式 + 工具合法性 + claim 命中**

**SFT 已经教过模型格式**，所以 GRPO 阶段它已经会写出正确的 `<tool_call>` 结构；
我们只用 reward 往"工具选对 + 答案命中更多 claim"方向推。

> 想升级到真多轮 rollout？在 `train_grpo.py` 里继承 `GRPOTrainer`，
> 重写 `_prepare_inputs` / `_generate_completions`：在 generate 内部，
> 每当遇到 `<tool_call>`，暂停、查 `MCPEpisode.call_tool`、把返回拼进
> prompt，再 generate。TRL 源码里 `grpo_trainer.py::_generate_and_score_completions`
> 是可替换点。我们在 `07_pitfalls.md` 里详述。

## 6. 启动

8 卡（accelerate + DeepSpeed ZeRO-3）：
```bash
bash /storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/code/run_grpo.sh
```

背后执行的是：
```bash
accelerate launch \
    --config_file configs/accelerate_ds_z3.yaml \
    code/grpo/train_grpo.py
```

`configs/accelerate_ds_z3.yaml` 里：
```yaml
distributed_type: DEEPSPEED
num_processes: 8
mixed_precision: bf16
deepspeed_config:
  deepspeed_config_file: configs/ds_z3_bf16.json
  zero3_init_flag: true
```

## 7. 预期曲线

```
step  1  reward=0.12  kl=0.00  format=0.5  legal=-0.1  coverage=0.3
step 10  reward=0.38  kl=0.02  format=0.7  legal=0.2   coverage=1.1
step 40  reward=0.96  kl=0.05  format=0.8  legal=0.6   coverage=1.8
step 80  reward=1.15  kl=0.08  format=0.8  legal=0.7   coverage=2.0   ← plateau
```

reward 平台 + KL 增长到 `≈ 0.1` 左右就应该停，否则 policy 开始"作弊"
（拼命堆 claim 关键词而失去自然语言质量）。

## 8. 小坑 & 心得

1. **GRPO 不训 value model**，但**依然需要 reference model** 来算 KL。
   TRL 默认用 `model.copy()`，显存立刻翻倍。解决：
   - 加 `peft_config`：TRL 检测到 LoRA 后**会把 reference 设为"没 LoRA 的基础模型"**，不再复制权重；
   - 或设 `sync_ref_model=True`，用主模型当 ref（等价无 KL，crash 风险大）。
   我们的脚本走第一条路：PEFT + LoRA → 自动节省。
2. **rollout 是显存大头**。`max_completion_length=2048` 和 `num_generations=4`
   加起来一次前向就 8K 新 token × 4 generations × bs。要么降 G、要么开 vLLM。
3. **开 vLLM 加速 rollout**：把 `--use-vllm` 打开。但 TRL 的 `vllm_mode="server"`
   要**独立起一个 vLLM server 进程**（见 `06_distributed.md`）；或者
   `vllm_mode="colocate"` 和训练共享 GPU，显存调度变难。
4. **reward 绝对值别太大**：GRPO 组内归一化，但 `reward > 10` 时方差会压
   advantage，训练缓慢。我们把每个子 reward 控制在 ~[0, 2] 范围。
5. **SFT 必须先跑透**，否则 GRPO 阶段模型根本不会输出 `<tool_call>` 结构，
   format_reward 永远 0，后续 reward 都归零。

## 9. 复现命令一览

```bash
# 1) 准备 GRPO 数据
python code/prepare/parquet_to_grpo.py \
    --src data/raw/MCP-Atlas.parquet \
    --out data/grpo/mcp_atlas.jsonl

# 2) 启动训练
bash code/run_grpo.sh
#   默认超参：G=4, lr=5e-6, epochs=2, beta=0.04
#   自定义例：bash code/run_grpo.sh --num-generations 8 --use-vllm
```

下一篇 → `06_distributed.md`（多卡并行 & 加速踩坑）
