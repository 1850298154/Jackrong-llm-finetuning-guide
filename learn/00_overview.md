# 00 · 项目总览：用 MCP-Atlas 对 Qwen3.6-27B 做 SFT + GRPO

这个 `learn/` 目录是我（AI 助手）边做边写给你的**学习笔记合集**。每一篇都遵循
`nn_xx.md` 编号，**按顺序**阅读即可复现整条流水线。

## 1. 我们要做什么

> 用 **MCP-Atlas**（500 条工具调用轨迹数据集）微调 **Qwen3.6-27B**，
> 让模型具备“自己决定调哪些 MCP 工具、按什么顺序、传什么参数”的能力。

训练分两个阶段：

1. **SFT（监督微调）——冷启动**
   让模型先"会说这种话"：模仿 MCP-Atlas 里人类/大模型生成的完整工具调用轨迹。
   框架：**LLaMA-Factory**（封装好 DeepSpeed / FSDP / LoRA，适合新手上多卡）。
2. **GRPO（Group Relative Policy Optimization）——强化学习**
   让模型"用得更好"：在一个**模拟 MCP 环境**里 rollout，根据 reward（格式
   是否合法、工具名是否在允许列表、claims 是否命中）来优化 policy。
   框架：**TRL GRPOTrainer**（Hugging Face 官方的 DeepSeek-R1 同款训练器）。

SFT 的产物 → 作为 GRPO 的初始 policy；两者都会在 **eval loss / reward**
稳定后停下（你说的“抛物线最低点”）。

## 2. 为什么 SFT + GRPO，而不是只做其中一个

| 只做 SFT                         | 只做 GRPO                       | SFT + GRPO（我们选的）               |
|----------------------------------|---------------------------------|--------------------------------------|
| 学会"格式"，但不会 explore        | 冷启动 reward 太稀疏，根本学不动 | SFT 打底 → GRPO 精修                 |
| eval loss 最低即可停              | 需要上千步才能出好效果          | 两段 loss/ reward 曲线，清晰可追踪 |

> **一句话**：SFT 教模型"长什么样"；GRPO 教它"怎么答才好"。
> 参考 Jackrong 的 `Llama-3.2-3B-R1-Zero-GRPO.ipynb` 就是这一套路。

## 3. 硬件资源

```
8 × NVIDIA GPU（驱动 580.105.08）
/storage/RL/models/download/Qwen3.6-27B          # 约 54 GB 权重（FP16，15 个 shard）
/storage/RL/data/download/03_mcp/dataset/MCP-Atlas/MCP-Atlas.parquet  # 500 行
/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/                    # 我们的工作目录
```

## 4. 目录布局（最终形态）

```
/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/
├── venv/                    # uv 虚拟环境（下一章会建）
├── data/
│   ├── raw/                 # 指向原始 parquet 的 symlink
│   └── sft/                 # 预处理后 JSONL + LLaMA-Factory dataset_info.json
├── code/
│   ├── LLaMA-Factory/       # clone 自 hiyouga/LLaMA-Factory
│   ├── mcp_env/             # 我们自己写的 MCP 模拟环境 + reward
│   └── grpo/                # TRL GRPOTrainer 脚本
├── configs/
│   ├── sft_qwen27b_lora_zero3.yaml
│   └── grpo_qwen27b.yaml
├── checkpoint/
│   ├── sft/                 # SFT 产物（LoRA adapters + merged）
│   └── grpo/                # GRPO 产物
└── logs/
```

## 5. 章节索引

| 文件                    | 内容                                                               |
|-------------------------|--------------------------------------------------------------------|
| `00_overview.md`        | 本文：整体规划                                                     |
| `01_env_setup.md`       | uv 虚拟环境 + 依赖锁定 + CUDA/NCCL 验证                            |
| `02_data_prep.md`       | 从 parquet 到 LLaMA-Factory 支持的 ShareGPT JSONL                  |
| `03_sft_training.md`    | LLaMA-Factory + LoRA + ZeRO-3 在 8 卡上 SFT                        |
| `04_mcp_env.md`         | 我们自己搭的 MCP 模拟环境（mock tools, deterministic replay）      |
| `05_grpo_training.md`   | TRL GRPOTrainer：reward 设计、rollout、超参                        |
| `06_distributed.md`     | 多卡并行全景：DP / TP / PP / ZeRO / FSDP 的关系和选择依据          |
| `07_pitfalls.md`        | 踩坑实录：OOM、tokenizer 模板、packing、checkpoint 恢复…           |
| `08_evaluation.md`      | base / SFT / GRPO 三档对比，给出最终提升百分比                     |

## 6. 学习路径建议（给你 / 未来的自己）

- **第一遍**：跟着 01~03 跑完 SFT，看懂 LoRA 和 ZeRO-3 日志 → 够用了。
- **第二遍**：读 04~05，把 GRPO 跑通，重点理解 reward 怎么造。
- **第三遍**：06 扫盲分布式；07 当字典翻；08 看效果。

下一篇 → `01_env_setup.md`
