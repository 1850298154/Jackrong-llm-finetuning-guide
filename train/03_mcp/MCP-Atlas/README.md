# Qwen3.6-27B GRPO on MCP-Atlas — Ray Job

## 运行结果 (最新)
- **Job id**: `qwen36-27b-grpo-20260425-230652`
- **集群**: 32 × H100 (4 节点 × 8 GPU), `PACK` 放置, NCCL
- **模型**: `/storage/RL/models/download/Qwen3.6-27B` (`Qwen3_5ForCausalLM`, transformers 5.6.2 via runtime_env)
- **可训练**: 79,691,776 (LoRA r=16 / α=32, 目标 q,k,v,o,gate,up,down) / 26,975,690,240 = **0.295%**
- **Global batch**: per_device=1 × grad_accum=2 × world=32 = **64**
- **步速**: ~76 s/step, `max_steps=3000`, 早停监控 loss 最低点

### Step 进展
```
step 1 : loss 0.0592, reward 3.262
step 3 : loss 0.0161, reward 3.216
step 4 : loss -0.003, reward 3.347
```

## 目录结构
```
MCP-Atlas/
├── ray_job/                     # Ray --working-dir, 所有节点分发此目录
│   ├── ray_driver.py            # 驱动 Ray Train TorchTrainer (32 workers × 1 GPU)
│   ├── Qwen3_6_27B_GRPO.py      # 训练主体 (train_code 的运行态副本)
│   ├── runtime_env.yaml         # pip: transformers==5.6.2, huggingface_hub>=1.5,<2
│   ├── mcp_atlas_data/          # MCP-Atlas.parquet (15 MB, 随 working_dir 分发)
│   └── submit.sh
├── checkpoint/                  # GRPOTrainer 检查点 (save_steps=25, keep=3)
├── adapter_final/               # 最终 LoRA adapter
├── processed_data/              # rank0 保存的过滤后数据集
├── hf_cache/                    # HF_HOME, 跨节点共享
├── logs/                        # 每 rank 日志
├── ray_results/                 # Ray Train run metadata
└── README.md (此文件)
```

训练主源码存放于 `/storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train_code/Qwen3.6-27B-GRPO.py`，在 `ray_job/Qwen3_6_27B_GRPO.py` 保持同步副本（文件名下划线以保证可作为 python 模块导入）。

## 提交命令
```bash
export RAY_ADDRESS=http://172.17.193.214:8265
BASE=/storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train/03_mcp/MCP-Atlas
STAMP=$(date +%Y%m%d-%H%M%S)
SUB="qwen36-27b-grpo-$STAMP"
ray job submit --submission-id "$SUB" --no-wait \
    --working-dir "$BASE/ray_job" \
    --runtime-env "$BASE/ray_job/runtime_env.yaml" \
    -- python ray_driver.py
```

## 常用查询
```bash
# 状态
ray job status $SUB
# 实时 loss / 指标
ray job logs $SUB 2>&1 | grep -E "loss|reward|step_time" | tail -20
# 停止
ray job stop  $SUB
```

## 关键踩坑备忘 (已解决)
1. **镜像 transformers 4.57 不认 `qwen3_5`** → `runtime_env.pip: transformers==5.6.2`。
2. **transformers 5.6 `_is_package_available()` 返回 `(bool, None)` 元组**，trl 0.27 `is_<foo>_available()` 写法是 `return _is_package_available(pkg)`，使 tuple 永远为真，触发 `vllm_ascend`/`weave`/`liger_kernel`/... 等可选依赖连锁 import → 崩溃。修复：在 `_run_training()` 里 **只包装 `trl.import_utils` 的 15 个 `is_*_available()` 辅助**，取 `[0]` 强转 bool；**不要动 `transformers`**（它自己调用处都已 `[0]`）。
3. **`'Qwen3_5ForCausalLM' has no attribute warnings_issued`**：trl 0.27 `GRPOTrainer.__init__` 里 `model.warnings_issued["estimate_tokens"] = True`，而新移植的 Qwen3_5 模型类少了这个类属性。修复：构造 trainer 前显式设置 `model.warnings_issued = {}` （含 PEFT 包装层）。
4. **工作目录**：head pod 不挂 `/storage/RL`，所以代码 + 数据 全部打包进 `--working-dir`（15 MB parquet 直接塞进 ray_job/mcp_atlas_data/）。
5. **ray.init("http://...")** 在 job 内部会误判为 client 地址，job 内部统一 `ray.init(address="auto")`。

## 早停逻辑
`MinLossStopCallback` 基于 EMA 平滑 loss：经过 `min_loss_warmup=40` 步后，若 EMA 已连续 `min_loss_patience=15` 个 logging step 高于历史最低点，则 `control.should_training_stop = True`——"抛物线" 过了最低点即退出。不会训满 3000 步除非 loss 一直单调下降。
