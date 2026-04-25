# 04 · 搭一个"假"的 MCP 交互环境（给 GRPO 用）

> GRPO 需要**在线 rollout**：模型写一段包含 `<tool_call>` 的回复，环境执行它，
> 返回 `<tool_response>`，模型再接着写，形成多轮对话。真要去连 36 个真实的
> MCP 服务器（GitHub、Fetch、Memory…）在训练循环里太不现实——网络不稳定、
> 需要密钥、非确定性。本章展示**如何用数据集里现成的 golden TRAJECTORY 搭
> 一个可复放的模拟环境**，既廉价又确定性。

## 1. 设计思路

MCP-Atlas 每条样本自带：
- `ENABLED_TOOLS`：本任务允许的工具白名单
- `TRAJECTORY`：人类标注的"正确轨迹"，含 `(tool_call, tool_response)` 对

把 trajectory 拆成一张 **内存表**：

```python
tool_memory = {
  "github_search_repositories": [
      ({"query":"assaultcube"}, "<raw JSON response>"),
      ...
  ],
  "whois_whois_domain": [
      ({"domain":"assault.jpn.org"}, "..."),
  ],
  ...
}
```

当模型在 rollout 里调用 `tool_X(args)`：
- **若 tool_X ∉ enabled_tools** → 返回 `[ENV-ERROR] tool not allowed`，reward 扣分
- **若 tool_X ∈ enabled_tools 但 tool_memory[tool_X] 为空** → `[ENV-ERROR] no recording`
- **否则**：在 `tool_memory[tool_X]` 里挑"和模型 args 最像"的一条，返回其 response

匹配用最简的 overlap 相似度（字符串相等 +1，包含 +0.5，归一到 key 数量）。

## 2. 为什么这是"对的"近似

1. **Tool selection 是学习的主要瓶颈**：模型要学的是"用 github_search 而不是
   whois"这种决策，不是去 argue "query 字段该写什么"。Replay env 对工具选择
   这一维度 100% 真实，对 argument 细节容忍但打分。
2. **Reward 可设计**：即使 response 和 args 不完全匹配，我们也有确定的
   `ENV-ERROR` 文本当负信号——对比真实环境里的"timeout / 500"噪声更好控。
3. **无副作用**：训练期反复 rollout 不会去真的去 GitHub 查 API，不被 rate
   limit 打爆。

## 3. 模块结构

```
code/mcp_env/
├── __init__.py
├── env.py        # MCPEpisode + parse_tool_calls()
└── reward.py     # 3 个 TRL-兼容的 reward function
```

### env.py

```python
@dataclass
class MCPEpisode:
    task_id: str
    prompt: str
    enabled_tools: list[str]
    tool_memory: dict[str, list[(args, response_text)]]
    gold_claims: list[str]

    @classmethod
    def from_raw_row(cls, row):
        # 解析 TRAJECTORY，把每个 assistant.tool_calls[i] 和紧跟的
        # tool message 配对，存进 tool_memory
        ...

    def call_tool(self, name, arguments) -> (status, content):
        if name not in self.tool_memory:
            return "invalid_tool", "[ENV-ERROR] ..."
        records = self.tool_memory[name]
        if not records:
            return "no_record", "[ENV-ERROR] ..."
        best = max(records, key=lambda r: arg_sim(r[0], arguments))
        return "ok", best[1]
```

### reward.py — 三段求和

```python
total = format_reward + tool_legality_reward + claim_coverage_reward
```

| 子 reward              | 信号                                                          | 范围         |
|------------------------|---------------------------------------------------------------|--------------|
| `format_reward`        | 有 `<tool_call>` (+0.3) / `<think>` (+0.2) / 收尾文本 (+0.3) / 配对错 (-0.5) | ≈ [-0.5, 0.8] |
| `tool_legality_reward` | 每个合法工具名 +0.2（≤1.0），每个非法 -0.5                    | ≈ [-∞, 1.0] |
| `claim_coverage_reward`| 最终答案里出现的 gold-claim 的比例 × 2.0                      | [0, 2.0]     |

**注意签名**：TRL 的 `GRPOTrainer` 要求 `reward_fn(prompts, completions, **kwargs)`。
我们把 `enabled_tools` 和 `gold_claims` 作为 dataset 的额外列，TRL 会自动
透传到 kwargs。

## 4. Sanity check

```python
ep = MCPEpisode.from_raw_row(df.iloc[0])
status, content = ep.call_tool("fetch_fetch", {"query":"hello"})
# -> status='ok', content='Content type application/json ...' (真实返回值)

status, content = ep.call_tool("NONEXIST_TOOL", {})
# -> status='invalid_tool', content='[ENV-ERROR] tool `NONEXIST_TOOL` not in enabled_tools=[...]'

completion = (
  "<think>\nLet me search.\n</think>\n"
  "<tool_call>\n<function=fetch_fetch>\n<parameter=query>\n\"hello\"\n</parameter>\n</function>\n</tool_call>\n"
  "The AssaultCube repo was created in 2013, domain registered 2006, diff 7."
)
format_reward([""],[completion])        # -> [0.8]
tool_legality_reward([""],[completion], enabled_tools=[ep.enabled_tools])  # -> [+0.2]
claim_coverage_reward([""],[completion], gold_claims=[ep.gold_claims])     # -> [2.0]
```

完美，reward 行为符合直觉。

## 5. 踩坑：GTFA_CLAIMS 不是 JSON

原始数据里 claim 字段是 **Python repr**（单引号 list），`json.loads` 死。
解决：先 try json，失败后 fallback 到 `ast.literal_eval`。

```python
try:
    claims = json.loads(raw)
except Exception:
    import ast
    claims = ast.literal_eval(raw)
```

## 6. 在真实多轮 rollout 里串起来（伪代码）

完整逻辑放到 `05_grpo_training.md`，大致框架：

```python
for step in range(max_rollout_turns):
    text = model.generate(chat_history)      # 单轮前向
    tc = parse_tool_calls(text)
    if tc:
        for call in tc:
            status, resp = ep.call_tool(call["name"], call["arguments"])
            chat_history += tool_response_turn(resp)
    else:
        break                                 # 模型给出最终答案
final = text
rewards = sum(fn([prompt],[final], ...) for fn in ALL_REWARDS)
```

下一篇 → `05_grpo_training.md`
