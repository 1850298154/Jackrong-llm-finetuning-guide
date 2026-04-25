# 02 · 数据预处理：MCP-Atlas parquet → LLaMA-Factory ShareGPT

> 目标：把 `/storage/RL/data/download/03_mcp/dataset/MCP-Atlas/MCP-Atlas.parquet`
> （500 行，单文件 15 MB）转成 **LLaMA-Factory 官方支持的 ShareGPT 工具调用
> 格式**，并做长度过滤，写出
> `/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/data/sft/mcp_atlas.json`。

## 1. 原始数据长什么样

```python
cols = ['TASK', 'ENABLED_TOOLS', 'PROMPT', 'GTFA_CLAIMS', 'TRAJECTORY']
```

- `PROMPT`：用户一句话任务（平均 300 字符）
- `ENABLED_TOOLS`：JSON 字符串 `["github_search", "fetch", ...]`（per-task 白名单）
- `TRAJECTORY`：JSON list，OpenAI messages 风格
  - `role=assistant`，可能带 `tool_calls=[{"function":{"name":..,"arguments":..}}]`
  - `role=tool`，`content` 是工具返回的 list-of-parts
- `GTFA_CLAIMS`：评测用，SFT 不需要

**长度分布**（字符）：
```
TRAJECTORY: mean=99656  p50=30397  p95=462036  max=4429593
```
尾巴非常长——有的单条轨迹 4M 字符。必须做 **token 长度截断**。

## 2. 我们选的目标格式：LLaMA-Factory "sharegpt + tools"

LLaMA-Factory 支持工具调用数据的 ShareGPT 方言（见 `data/glaive_toolcall_en_demo.json`）：

```json
{
  "conversations": [
    {"from": "human",         "value": "Find me a recipe"},
    {"from": "gpt",           "value": "Sure, let me search."},
    {"from": "function_call", "value": "{\"name\":\"search\",\"arguments\":{...}}"},
    {"from": "observation",   "value": "{\"recipes\":[...]}"},
    {"from": "gpt",           "value": "I found two: ..."}
  ],
  "system": "You are an MCP agent.",
  "tools": "[{\"type\":\"function\",\"function\":{...}}, ...]"
}
```

LLaMA-Factory 内部会把这条样本喂给 **Qwen 的官方 chat template**，模板会
把它自动渲染成 Qwen 认识的 XML 形式：

```
<tool_call>
<function=search>
<parameter=query>Bell peppers</parameter>
</function>
</tool_call>
```

**所以我们不用手写任何特殊 token**，照着原始 OpenAI schema 输出即可。

## 3. 转换脚本

完整源码：`/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/code/prepare/parquet_to_sharegpt.py`

核心映射：

| MCP-Atlas            | LLaMA-Factory ShareGPT               |
|----------------------|--------------------------------------|
| `PROMPT`             | 第一条 `from=human`                  |
| `TRAJECTORY[].role=assistant.content`  | `from=gpt`        |
| `TRAJECTORY[].role=assistant.tool_calls[i]` | `from=function_call`（一个工具调用一条） |
| `TRAJECTORY[].role=tool.content`       | `from=observation`|
| `ENABLED_TOOLS`      | 顶层 `tools` 字段（JSON 字符串）     |

我们额外做了 3 件小事：
1. **content 类型归一化**：原 content 可能是 `str` / `list[{"text":..}]` / `None`，
   统一成一行字符串。
2. **空 trajectory 过滤**：没有 assistant 回复的丢掉。
3. **结尾保护**：保证最后一轮是 `gpt` 或 `function_call`，否则追加一条占位
   （LLaMA-Factory SFT 要求 labels 不能全为 -100）。

## 4. 长度过滤

为了"跟训练时看到的一样"，我直接用 **Qwen3.6 的 tokenizer + chat_template**
来测量每条样本的 token 数：

```python
text = tok.apply_chat_template(messages, tools=tools, tokenize=False)
n = len(tok(text, add_special_tokens=False).input_ids)
```

然后丢掉 `n > 8192` 的样本。8192 这个数是一个折中：
- 再大 → 单卡显存吃不消；
- 再小 → 留下样本太少（实测 4096 只留 110 条）。

最终：
```
kept=201  dropped_parse=0  dropped_len=299
token-len  mean=4844  p50=4822  p90=7330  p99=8063  max=8149
```

train / eval = **181 / 20**（随机 shuffle + 10 % eval）。

## 5. ⚠️ 一个真实的踩坑：`apply_chat_template(tokenize=True)` 返回值变了

transformers **5.x** 引入新的 `TokenizersBackend`，
`tok.apply_chat_template(..., tokenize=True)` 返回的不是 `list[int]`，
而是 `[Encoding(num_tokens=...)]`（长度永远是 1，内含一个 Encoding 对象）。

如果你像 4.x 那样直接 `len(ids)` 就会全返回 1 或 2，过滤完全失效。

解决：**改用 `tokenize=False` 拿 text，再显式 tok(text)**。

## 6. 注册到 LLaMA-Factory

新建 `data/sft/dataset_info.json`：

```json
{
  "mcp_atlas_train": {
    "file_name": "mcp_atlas.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
    "tags": {
      "role_tag": "from", "content_tag": "value",
      "user_tag": "human", "assistant_tag": "gpt",
      "observation_tag": "observation", "function_tag": "function_call",
      "system_tag": "system"
    }
  },
  "mcp_atlas_eval": { /* same, file_name=mcp_atlas_eval.json */ }
}
```

训练时通过 `dataset_dir=/storage/.../data/sft` + `dataset=mcp_atlas_train` +
`eval_dataset=mcp_atlas_eval` 即可调用。

## 7. 复现命令

```bash
BASE=/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas
$BASE/venv/bin/python $BASE/code/prepare/parquet_to_sharegpt.py \
    --src /storage/RL/data/download/03_mcp/dataset/MCP-Atlas/MCP-Atlas.parquet \
    --out $BASE/data/sft/mcp_atlas.json \
    --tokenizer /storage/RL/models/download/Qwen3.6-27B \
    --max-seq-len 8192 --eval-ratio 0.1
```

下一篇 → `03_sft_training.md`
