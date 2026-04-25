# 07 · 踩坑实录 & 学习心得（SFT / GRPO / 分布式）

> 这一篇就是"字典 + 吐槽"。真实出现过的坑以 **现象 → 根因 → 修法** 的形式
> 记录。看到熟悉的报错可以直接搜索。

## A. 环境 / 安装

### A1. `nvidia-smi: command not found` 但驱动明明装了
- 现象：`/proc/driver/nvidia/` 有内容，但 `nvidia-smi` 找不到、
  `torch.cuda.is_available()=False`。
- 根因：容器/pod 没挂 `nvidia-container-runtime`，设备节点 `/dev/nvidia*` 不存在。
- 修法：让运维给 pod 加 `--gpus all` / `runtime=nvidia`。**应用侧无解**。

### A2. `uv` 跨文件系统 hardlink 失败
- 现象：`warning: Failed to hardlink files; falling back to full copy`
- 修法：`export UV_LINK_MODE=copy`，噪声消失。

### A3. `~/.bashrc: Permission denied`
- 现象：uv 装完想写 PATH 到 .bashrc 失败。
- 修法：用绝对路径调 `$HOME/.local/bin/uv`；或 `export PATH=$HOME/.local/bin:$PATH` 写到项目 Makefile。

## B. 数据 / tokenizer

### B1. `apply_chat_template(tokenize=True)` 返回 `[Encoding(...)]`
- 现象：transformers 5.x 下拿到的 `ids` 长度永远是 1，全部样本被误判为"太短"。
- 根因：新 `TokenizersBackend` 返回 `BatchEncoding` 列表，不再是 `list[int]`。
- 修法：先 `tokenize=False` 拿 text，再 `tok(text).input_ids`。

### B2. `GTFA_CLAIMS` 不是 JSON
- 现象：`json.JSONDecodeError`。
- 根因：原始字段是 Python `repr` 字符串（单引号 list）。
- 修法：
  ```python
  try: claims = json.loads(raw)
  except: import ast; claims = ast.literal_eval(raw)
  ```

### B3. 长轨迹样本被全部砍掉
- 现象：`max-seq-len=4096` 下只保留 110/500 条，22%。
- 根因：TRAJECTORY 字段 p95 462K 字符，一个完整轨迹就塞满。
- 修法：提 max-seq-len 到 8192（留 201 条）；若还嫌少，改为**按轮次截断**
  保前 N 个 tool_call，训练目标只到那个位置。

## C. SFT / LLaMA-Factory

### C1. `save_strategy != eval_strategy` 阻断 `load_best_model_at_end`
- 现象：`ValueError: --load_best_model_at_end requires the save and eval strategy to match`。
- 根因：transformers 5.2 校验变严。
- 修法：把 `save_strategy` 设成和 `eval_strategy` 一样（都用 `epoch` 或都用
  `steps`，且 save_steps 必须是 eval_steps 的整数倍）。

### C2. `warmup_ratio is deprecated`
- 现象：warning，v5.2 后移除。
- 修法：统一用 `warmup_steps: <int>`。

### C3. packing=True 导致 tool_call / tool_response 错位
- 现象：训练 loss 正常下降但推理时 assistant 乱插别人的 tool_response。
- 根因：packing 把多个样本拼进同一个 seq，`<tool_call>` 后面可能接的是别
  人的 `<tool_response>`，loss mask 把它错当成 label。
- 修法：**有 role=tool 数据的都关 packing**。`packing=false`。

### C4. ZeRO-3 保存的 checkpoint 载入报 shape mismatch
- 现象：`RuntimeError: size mismatch for ...`
- 根因：`stage3_gather_16bit_weights_on_model_save=false` 默认，保存的是 shard。
- 修法：DS JSON 里显式 `"stage3_gather_16bit_weights_on_model_save": true`。

### C5. LoRA 合并后 dtype 不匹配
- 现象：`merge_and_unload()` 后做 generate 崩掉。
- 根因：LoRA A/B 默认 fp32，和 bf16 base merge 后某些 linear 变 fp32。
- 修法：`model.merge_and_unload().to(torch.bfloat16)`。

## D. GRPO / TRL

### D1. `reward_funcs` 签名错误
- 现象：`TypeError: format_reward() got unexpected keyword argument 'enabled_tools'`。
- 根因：TRL 会把 dataset 里非 prompt 列**全部**当 kwargs 传；你所有 reward fn
  都要 `**kwargs`。
- 修法：每个 reward fn 定义为 `def fn(prompts, completions, **kwargs)`，
  按需从 kwargs 里取。

### D2. reference model 让显存翻倍
- 现象：GRPO 启动瞬间 OOM。
- 根因：TRL 默认 `copy.deepcopy(model)` 当 reference，加倍权重。
- 修法：用 LoRA + `is_trainable=True`，TRL 会**把 reference 改成"没 LoRA 的基础 frozen model"**，不复制权重。

### D3. reward 曲线一直在 0 附近
- 现象：GRPO 跑 50 step reward mean ≈ 0.05。
- 可能根因：
  1. SFT 没跑透，模型不会写 `<tool_call>` → 所有 reward 零。
  2. `temperature` 太低 (<0.5)，组内 completion 几乎一样 → advantage 为 0。
  3. `beta` 太大（>0.2），KL 拉得太狠，policy 动不了。
- 修法：依次把 temperature 调到 0.9、beta 降到 0.02、SFT 多跑两个 epoch。

### D4. rollout 超慢，GPU 利用率 20%
- 现象：每 step 40 s 以上，`nvidia-smi` 显示 gpu-util ~20%。
- 根因：TRL 默认用 HF `generate`，逐 token 同步走，batch 利用率差。
- 修法：打开 vLLM：`--use-vllm`，开一个独立 GPU 做 vLLM server。

### D5. `num_generations=8` 但 `per_device_train_batch_size=1`
- 现象：`ValueError: num_generations must divide total batch size`。
- 根因：TRL 要求 `per_device_train_batch_size * num_processes` 能被
  `num_generations` 整除。
- 修法：调整到匹配，比如 `bs=1, n_proc=8, G=4` → total=8，8 % 4 == 0 ✅。

## E. 分布式通信

### E1. NCCL timeout at first step
- 现象：`Watchdog caught collective operation timeout`。
- 根因：rank 之间初始化慢（27B 权重广播要几十秒），超时 600s 不够。
- 修法：`--ddp-timeout 1800` 或 `ddp_timeout: 180000000` 写进 yaml。

### E2. `NCCL_IB_DISABLE=1` 在单机里反而快
- 现象：8 卡单机，开 IB 通信反而慢。
- 根因：单机之间走 NVLink 更快，IB 只在多机有用。
- 修法：单机直接 `export NCCL_IB_DISABLE=1`；多机别乱加。

### E3. CUDA OOM 但 `nvidia-smi` 显示只占一半
- 现象：模型放得下但一起 step 就炸。
- 根因：activation 没算进去；gradient checkpointing 没开。
- 修法：
  1. `gradient_checkpointing: true`
  2. 检查 `max_seq_len` 是不是比数据实际 p95 还大一大截
  3. `per_device_train_batch_size=1` 保底，加 `grad_accum` 凑 effective batch

## F. 多轮 rollout（进阶）

### F1. 想让 GRPO 真做"模型 ↔ env" 多轮
- 现象：TRL 默认单轮 rollout，模型只有一次机会。
- 升级路线：继承 `GRPOTrainer`，覆盖 `_generate_and_score_completions`：
  ```python
  def _generate_and_score_completions(self, inputs):
      # 原版是一个 generate
      # 改造：循环
      chat = inputs["prompt"]
      for _ in range(MAX_TURNS):
          out = self.model.generate(chat, max_new_tokens=512)
          tcs = parse_tool_calls(out)
          if not tcs:
              break
          for t in tcs:
              status, resp = env.call_tool(t.name, t.args)
              chat = chat + tool_response_turn(resp)
      return super()._score(chat, out)
  ```
- 风险：长度膨胀、KL 计算对 multi-turn 语义要小心。建议先跑通单轮
  baseline，再迭代到多轮。

## G. 调试技巧

1. **把 batch 开到 2、dataset 砍到 4 条先打通流水**（`num_train_epochs=1, max_steps=5`），再上真参。
2. **`RANK=0` 专门打日志**，其他 rank 只打 warning：`transformers.utils.logging.set_verbosity_error()`。
3. **优先用 tensorboard 看曲线**：`tensorboard --logdir checkpoint/` 秒开。
4. **出错贴整栈 + `deepspeed --version` + `torch.__version__`** 去 LLaMA-Factory / TRL 的 GitHub Issues 搜，95% 已经有人遇到。

下一篇 → `08_evaluation.md`
