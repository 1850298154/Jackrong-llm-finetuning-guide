# 08 · 评测：base / SFT / GRPO 三档对比

> 目标：拿同一批 50 条 eval 样本，分别跑 **base Qwen3.6-27B**、
> **SFT checkpoint**、**GRPO checkpoint**，看 reward 和 claim_coverage 的提升。

## 1. 评测指标

我们复用训练时的三个 reward：

- `format_reward`：输出结构合法性（`<think>` / `<tool_call>` 配对）
- `tool_legality`：工具调用都在 `ENABLED_TOOLS` 白名单里
- `claim_coverage`：模型最终答案里覆盖 `GTFA_CLAIMS` 的比例

以及它们的**和**（总 reward）作为综合指标。这是"训练信号一致的评测"，
反映的是**我们 RL 优化的目标达成度**。

想要更权威的评测？官方提供 MCP-Atlas 的 eval harness（在
[mcp-atlas github](https://github.com/scaleapi/mcp-atlas/tree/main)）；
它用 claims-based rubric 对 model response 打 coverage 分数，和我们的
`claim_coverage_reward` 思路一致。

## 2. 评测脚本

`code/eval/evaluate.py`：

- 加载 base 模型（+ 可选 LoRA adapter）
- 对每条 eval 样本 **greedy decode**（temperature=0）生成 completion
- 计算 3 个 sub-reward 并写入 `logs/eval_results.jsonl`
- 打印 mean 每项

## 3. 复现三档评测

```bash
BASE=/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas
PY=$BASE/venv/bin/python
export PYTHONPATH=$BASE/code

# (A) base
$PY $BASE/code/eval/evaluate.py \
    --checkpoint /storage/RL/models/download/Qwen3.6-27B \
    --out $BASE/logs/eval_base.jsonl

# (B) SFT
$PY $BASE/code/eval/evaluate.py \
    --checkpoint $BASE/checkpoint/sft --is-lora \
    --out $BASE/logs/eval_sft.jsonl

# (C) GRPO
$PY $BASE/code/eval/evaluate.py \
    --checkpoint $BASE/checkpoint/grpo --is-lora \
    --out $BASE/logs/eval_grpo.jsonl
```

## 4. 期望的提升幅度（参考值）

根据参考 notebook（Llama-3.2-3B-R1-Zero-GRPO）的趋势 + 我们这里的配置：

| 阶段       | total mean | format | legal | coverage | 典型表现       |
|-----------|------------|--------|-------|----------|----------------|
| base      | 0.3–0.7    | 0.1    | -0.3  | 0.5–1.0  | 偶尔能输出 tool_call，格式乱，claim 命中少 |
| +SFT      | 1.8–2.2    | 0.7    | 0.6   | 0.5–0.9  | 格式稳定，工具合法，但 claim 覆盖仍不足 |
| +GRPO     | 2.3–2.8    | 0.8    | 0.8   | 0.7–1.2  | 格式 + 工具 + claim 都改善，尤其 coverage |

**为什么 GRPO 能继续涨**：reward 中 `claim_coverage` 的权重推动模型把
最终答案里的关键事实**讲得更密**——这是 SFT 冷启动 teacher 数据没法
直接刻画的。

## 5. 可视化

tensorboard 会记录训练过程里的 reward 曲线：

```bash
tensorboard --logdir $BASE/checkpoint/grpo --port 6006
```

建议看的面板：
- `reward_format / reward_legal / reward_coverage`：分项是否都在涨
- `reward_std / advantage_mean`：GRPO 健康度
- `kl`：policy 偏离 SFT 多远（<=0.1 安全）

## 6. 整个项目复盘

三段流水线全部打通：

```
parquet (500 条)
   │  parquet_to_sharegpt.py (filter 8K tok → 201 条)
   ▼
SFT LoRA (LLaMA-Factory + ZeRO-3)
   │  load_best_model_at_end=true → 抛物线最低点
   ▼
checkpoint/sft (LoRA adapter, ~400MB)
   │  + parquet_to_grpo.py (450 条 prompt-only)
   ▼
GRPO (TRL + 3-way reward)
   │  num_generations=4, beta=0.04, lr=5e-6
   ▼
checkpoint/grpo (LoRA adapter, ~400MB)
   │
   ▼
evaluate.py → 三档对比
```

## 7. 下一步可以探索的方向

1. **多轮 rollout**（见 07_pitfalls.md §F）：模型调工具 → env 回复 → 继续。
2. **PPO 替代 GRPO**：需要 value model，显存翻倍但 reward shaping 更灵活。
3. **真实 MCP server**：把 env.py 里的 replay 改成 HTTP 调 scaleapi 的 MCP 服务器。
4. **奖励模型 (RM)**：用 GTFA_CLAIMS 训一个小打分模型替代规则 reward。
5. **课程学习**：先训只有 1 个 tool 的简单任务，再逐步放开。

---

**祝学得愉快 🚀**

所有代码在 `/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/code/`，
所有笔记在 `/home/z/zyt/Jackrong-llm-finetuning-guide/learn/`。
