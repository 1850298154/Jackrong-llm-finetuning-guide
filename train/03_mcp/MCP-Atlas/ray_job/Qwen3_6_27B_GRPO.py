#!/usr/bin/env python3
"""
GRPO fine-tuning for Qwen3.6-27B on the MCP-Atlas tool-use benchmark.

Design notes
------------
* Modeled after train_code/Llama-3.2-3B-R1-Zero-GRPO.ipynb (format+accuracy
  reward shaping, anti-silence floor, repetition penalty), but adapted to
  the tool-use / claim-coverage setting and to a much larger local model.
* Runs under Ray (``ray job submit``) -- see train_code/ray_submit.sh.
* Uses HF Accelerate + DeepSpeed ZeRO-3 so the 27B model shards across
  all visible GPUs. LoRA keeps the trainable count tiny.
* Trains with trl.GRPOTrainer, stops automatically once the smoothed
  training loss has passed its minimum (parabola) via MinLossStopCallback.
* All artefacts (checkpoints, logs, final adapter, processed dataset)
  live under:
    /storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train/03_mcp/MCP-Atlas/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Make TRL / HF happy under Ray workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
# Force offline model / dataset loads -- everything is on local disk.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ---------------------------------------------------------------------------
# Defaults / paths
# ---------------------------------------------------------------------------
MODEL_PATH_DEFAULT = "/storage/RL/models/download/Qwen3.6-27B"
DATA_DIR_DEFAULT = "/storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train/03_mcp/MCP-Atlas/dataset"
WORK_ROOT_DEFAULT = (
    "/storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/"
    "train/03_mcp/MCP-Atlas"
)

# ---------------------------------------------------------------------------
# System prompt + format tags.  We ask the model to emit <think>..</think>
# for chain-of-thought, then one or more Qwen <tool_call>..</tool_call>
# blocks, and finally a <final_answer>..</final_answer> block.  The reward
# functions below parse exactly this shape.
# ---------------------------------------------------------------------------
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
FINAL_OPEN, FINAL_CLOSE = "<final_answer>", "</final_answer>"

SYSTEM_PROMPT_TMPL = """You are an autonomous tool-using assistant.

For every user task you MUST produce your answer in this exact layout (no
text outside the tags):

{THINK_OPEN}
concise step-by-step plan about which tools to call and why
{THINK_CLOSE}
{TOOL_OPEN}
<function=TOOL_NAME>
<parameter=PARAM_NAME>value</parameter>
...
</function>
{TOOL_CLOSE}
(... repeat the <tool_call> block once per tool invocation you plan ...)
{FINAL_OPEN}
natural-language answer that covers every required claim
{FINAL_CLOSE}

Rules:
* Only call tools from the ENABLED_TOOLS list the user supplies.
* Emit between {MIN_TOOLS} and {MAX_TOOLS} tool calls; most tasks need 3-6.
* Do not invent tool outputs -- the tool blocks just declare intent.
* The <final_answer> must be a self-contained response grounded in the
  tools you invoked.
"""

MIN_TOOLS, MAX_TOOLS = 3, 6

def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TMPL.format(
        THINK_OPEN=THINK_OPEN, THINK_CLOSE=THINK_CLOSE,
        TOOL_OPEN=TOOL_OPEN, TOOL_CLOSE=TOOL_CLOSE,
        FINAL_OPEN=FINAL_OPEN, FINAL_CLOSE=FINAL_CLOSE,
        MIN_TOOLS=MIN_TOOLS, MAX_TOOLS=MAX_TOOLS,
    )


# ---------------------------------------------------------------------------
# MCP-Atlas loading + prompt construction
# ---------------------------------------------------------------------------
def _safe_json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    s = str(value).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            # GTFA_CLAIMS is a python-repr list -- tolerate single quotes.
            import ast
            return ast.literal_eval(s)
        except Exception:
            return None


def _extract_tool_names(trajectory_raw: Any) -> List[str]:
    traj = _safe_json_loads(trajectory_raw) or []
    names: List[str] = []
    if isinstance(traj, list):
        for turn in traj:
            if not isinstance(turn, dict):
                continue
            for call in turn.get("tool_calls") or []:
                fn = (call or {}).get("function") or {}
                name = fn.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
    return names


def _keyword_set(claim: str) -> List[str]:
    """Lower-cased content words (>=3 chars) pulled from a single claim."""
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|\d{2,}", str(claim).lower())
    stop = {"the", "and", "for", "was", "are", "from", "that", "this",
            "with", "has", "have", "its", "been", "were", "his", "her",
            "can", "get", "got", "about", "into", "out", "not", "but",
            "any", "all", "one", "two", "new", "old", "year", "years",
            "date", "between", "there", "their", "they", "them", "you",
            "your", "our", "which", "when", "where", "what", "who",
            "how", "why", "now", "also", "onto", "per", "upon"}
    return [t for t in toks if t not in stop]


def load_mcp_atlas(data_dir: str) -> pd.DataFrame:
    """Locate the MCP-Atlas parquet by checking several candidate paths.

    Ray workers don't always mount the proxy pod's /storage/RL/data volume,
    so we let the training script fall back to a parquet shipped inside
    the Ray working_dir (see ``ray_job/mcp_atlas_data/``) or, finally, to
    a copy stashed in the user workspace.
    """
    candidates = []
    # 1. Explicit --data_dir
    candidates.append(Path(data_dir) / "MCP-Atlas.parquet")
    # 2. Next to this script (ray working_dir upload)
    here = Path(__file__).resolve().parent
    candidates.append(here / "mcp_atlas_data" / "MCP-Atlas.parquet")
    candidates.append(here / "MCP-Atlas.parquet")
    # 3. A few known-good mount points on the KubeRay cluster
    for p in (
        "/storage/RL/models/download/MCP-Atlas/MCP-Atlas.parquet",
        "/storage/RL/dataset/ytzhao02/benchmark/MCP-Atlas/MCP-Atlas.parquet",
        "/storage/RL/data/download/03_mcp/dataset/MCP-Atlas/MCP-Atlas.parquet",
    ):
        candidates.append(Path(p))

    parquet_path = None
    for c in candidates:
        try:
            if c.exists():
                parquet_path = c
                break
        except Exception:
            pass
    if parquet_path is None:
        raise FileNotFoundError(
            "MCP-Atlas parquet not found. Tried:\n  "
            + "\n  ".join(str(x) for x in candidates)
        )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[load_mcp_atlas] using {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)

    enabled_list = df["ENABLED_TOOLS"].apply(_safe_json_loads)
    df["enabled_tools_list"] = enabled_list.apply(
        lambda xs: [str(x) for x in xs] if isinstance(xs, list) else []
    )

    claims_list = df["GTFA_CLAIMS"].apply(_safe_json_loads)
    df["gt_claims"] = claims_list.apply(
        lambda xs: [str(x) for x in xs] if isinstance(xs, list) else []
    )

    df["gt_tool_names"] = df["TRAJECTORY"].apply(_extract_tool_names)
    df["claim_keywords"] = df["gt_claims"].apply(
        lambda cs: [kw for c in cs for kw in _keyword_set(c)]
    )
    return df


def build_user_message(prompt: str, enabled_tools: Sequence[str]) -> str:
    tools_txt = ", ".join(enabled_tools) if enabled_tools else "(none)"
    return (
        "ENABLED_TOOLS:\n"
        f"{tools_txt}\n\n"
        "TASK:\n"
        f"{prompt.strip()}"
    )


def build_grpo_dataset(df: pd.DataFrame, tokenizer, max_prompt_tokens: int):
    from datasets import Dataset
    system_prompt = build_system_prompt()

    records: List[Dict[str, Any]] = []
    kept = 0
    for _, row in df.iterrows():
        user_msg = build_user_message(row["PROMPT"], row["enabled_tools_list"])
        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            ids = tokenizer.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=True,
            )
        except Exception:
            continue
        if len(ids) > max_prompt_tokens:
            continue
        records.append({
            "prompt": prompt_msgs,
            "enabled_tools": list(row["enabled_tools_list"]),
            "gt_claims": list(row["gt_claims"]),
            "gt_tool_names": list(row["gt_tool_names"]),
            "claim_keywords": list(row["claim_keywords"]),
        })
        kept += 1

    if kept == 0:
        raise RuntimeError("No MCP-Atlas rows survived tokenization.")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Reward functions
#
# Each reward callable takes GRPOTrainer's kwargs and returns one float per
# completion.  We build closures so they can read per-sample metadata
# (enabled_tools, gt_claims, ...) that we threaded through the dataset.
# ---------------------------------------------------------------------------
_TOOLCALL_BLOCK_RE = re.compile(rf"{re.escape(TOOL_OPEN)}(.*?){re.escape(TOOL_CLOSE)}",
                                flags=re.DOTALL)
_FUNCTION_NAME_RE = re.compile(r"<function=([A-Za-z0-9_\-\.]+)>")
_THINK_RE = re.compile(rf"{re.escape(THINK_OPEN)}(.*?){re.escape(THINK_CLOSE)}",
                       flags=re.DOTALL)
_FINAL_RE = re.compile(rf"{re.escape(FINAL_OPEN)}(.*?){re.escape(FINAL_CLOSE)}",
                       flags=re.DOTALL)
_STRICT_RE = re.compile(
    rf"^\s*{re.escape(THINK_OPEN)}\s*(?P<think>.+?)\s*{re.escape(THINK_CLOSE)}"
    rf"\s*(?P<tools>(?:{re.escape(TOOL_OPEN)}.*?{re.escape(TOOL_CLOSE)}\s*)+)"
    rf"{re.escape(FINAL_OPEN)}\s*(?P<final>.+?)\s*{re.escape(FINAL_CLOSE)}\s*$",
    flags=re.DOTALL,
)


def _completion_text(completion: Any) -> str:
    """GRPOTrainer feeds either list[dict] (chat) or plain strings."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    return str(completion)


def _strip_trailing_specials(text: str, extras: Sequence[str]) -> str:
    s = (text or "").rstrip()
    changed = True
    while changed and s:
        changed = False
        for token in extras:
            if token and s.endswith(token):
                s = s[: -len(token)].rstrip()
                changed = True
    return s


def _parse_structure(text: str, specials: Sequence[str]) -> Dict[str, Any]:
    raw = text or ""
    stripped = _strip_trailing_specials(raw, specials)

    think_matches = _THINK_RE.findall(stripped)
    final_matches = _FINAL_RE.findall(stripped)
    tool_blocks = _TOOLCALL_BLOCK_RE.findall(stripped)

    tool_names: List[str] = []
    for block in tool_blocks:
        m = _FUNCTION_NAME_RE.search(block)
        if m:
            tool_names.append(m.group(1))

    strict = _STRICT_RE.match(stripped) is not None
    has_think = bool(think_matches) and len(think_matches[0].strip()) >= 5
    has_final = bool(final_matches) and len(final_matches[0].strip()) >= 5
    n_tool_calls = len(tool_blocks)

    return {
        "raw": raw,
        "stripped": stripped,
        "think_text": think_matches[0] if think_matches else "",
        "final_text": final_matches[0] if final_matches else "",
        "tool_names": tool_names,
        "n_think": len(think_matches),
        "n_final": len(final_matches),
        "n_tool_calls": n_tool_calls,
        "strict_ok": strict,
        "has_think": has_think,
        "has_final": has_final,
    }


def make_reward_funcs(tokenizer):
    trailing = [t for t in (
        getattr(tokenizer, "eos_token", None),
        getattr(tokenizer, "pad_token", None),
        "<|im_end|>", "<|endoftext|>",
    ) if t]

    # Tunable weights -- match the Llama-R1 notebook's "anti-silence" vibe.
    FMT_EMPTY = -2.0
    FMT_NONEMPTY_FLOOR = -1.5
    FMT_PARTIAL_CAP = 0.6
    FMT_STRICT_BONUS = 1.0
    DUP_PENALTY = 0.6
    TAIL_PENALTY = 0.4

    def print_samples_reward(prompts, completions, **kwargs):
        try:
            text = _completion_text(completions[0])
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print("\n" + "=" * 20 + "  sample completion  " + "=" * 20)
                tail = str(prompts[0])[-400:] if prompts else ""
                print(f"prompt tail: ...{tail}")
                print("-" * 10)
                print(text[:2000])
                print("=" * 60 + "\n", flush=True)
        except Exception as exc:
            print(f"[print_samples_reward] {exc!r}", flush=True)
        return [0.0] * len(completions)

    def format_reward(prompts, completions, **kwargs):
        scores: List[float] = []
        for comp in completions:
            info = _parse_structure(_completion_text(comp), trailing)
            if not info["stripped"].strip():
                scores.append(FMT_EMPTY)
                continue

            score = 0.05
            score += 0.15 if info["has_think"] else -0.05
            score += 0.15 if info["has_final"] else -0.10
            if info["n_tool_calls"] >= 1:
                score += 0.15
            else:
                score -= 0.15
            if MIN_TOOLS <= info["n_tool_calls"] <= MAX_TOOLS:
                score += 0.15

            # Duplication / stray sections -> mild penalties.
            if info["n_think"] > 1:
                score -= DUP_PENALTY
            if info["n_final"] > 1:
                score -= DUP_PENALTY

            # Tail after the last </final_answer> is forbidden.
            tail_idx = info["stripped"].rfind(FINAL_CLOSE)
            if tail_idx >= 0:
                tail = info["stripped"][tail_idx + len(FINAL_CLOSE):].strip()
                if tail:
                    score -= TAIL_PENALTY

            if info["strict_ok"]:
                score += FMT_STRICT_BONUS
            else:
                score = min(score, FMT_PARTIAL_CAP)

            score = max(score, FMT_NONEMPTY_FLOOR)
            scores.append(score)
        return scores

    def _pair(lst, n):
        if lst is None:
            return [None] * n
        if len(lst) == n:
            return list(lst)
        if len(lst) > 0 and n % len(lst) == 0:
            k = n // len(lst)
            out = []
            for x in lst:
                out.extend([x] * k)
            return out
        return [lst[0] if lst else None] * n

    def tool_name_reward(prompts, completions,
                         enabled_tools=None, gt_tool_names=None, **kwargs):
        n = len(completions)
        enabled_all = _pair(enabled_tools, n)
        gt_all = _pair(gt_tool_names, n)
        scores: List[float] = []
        for comp, enabled, gt_names in zip(completions, enabled_all, gt_all):
            info = _parse_structure(_completion_text(comp), trailing)
            called = info["tool_names"]
            enabled_set = set(enabled or [])
            gt_set = set(gt_names or [])

            if not called:
                scores.append(-0.5)
                continue

            in_enabled = sum(1 for t in called if t in enabled_set)
            enabled_ratio = in_enabled / max(1, len(called))
            # Reward fraction of calls that are in the allowed set...
            s = 0.8 * enabled_ratio
            # ... plus recall of the "gold" tools from TRAJECTORY.
            if gt_set:
                hit = sum(1 for t in gt_set if t in called)
                s += 1.2 * (hit / len(gt_set))
            scores.append(s)
        return scores

    def claim_coverage_reward(prompts, completions,
                              gt_claims=None, claim_keywords=None, **kwargs):
        n = len(completions)
        claims_all = _pair(gt_claims, n)
        kws_all = _pair(claim_keywords, n)
        scores: List[float] = []
        for comp, claims, kws in zip(completions, claims_all, kws_all):
            info = _parse_structure(_completion_text(comp), trailing)
            haystack = (info["final_text"] or info["stripped"]).lower()

            if not claims:
                scores.append(0.0)
                continue

            # Per-claim coverage: fraction of that claim's keywords present.
            per_claim = []
            for claim in claims:
                kw = _keyword_set(claim)
                if not kw:
                    per_claim.append(0.0)
                    continue
                hits = sum(1 for k in kw if k in haystack)
                per_claim.append(hits / len(kw))
            coverage = sum(per_claim) / max(1, len(per_claim))

            # Bonus when most claims are (almost) fully covered.
            fully = sum(1 for x in per_claim if x >= 0.6)
            coverage_bonus = fully / max(1, len(per_claim))

            score = 3.0 * coverage + 1.5 * coverage_bonus
            if info["strict_ok"]:
                score += 0.5
            scores.append(score)
        return scores

    def repetition_penalty(prompts, completions, **kwargs):
        scores: List[float] = []
        for comp in completions:
            text = _completion_text(comp)
            toks = re.findall(r"[A-Za-z]+|\d+|[\u4e00-\u9fff]+", text)
            if len(toks) < 20:
                scores.append(0.0)
                continue

            def rep_rate(n: int) -> float:
                if len(toks) < n + 5:
                    return 0.0
                grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
                return 0.0 if not grams else (len(grams) - len(set(grams))) / len(grams)

            penalty = -(0.008 * rep_rate(3) + 0.012 * rep_rate(4))
            scores.append(max(penalty, -0.2))
        return scores

    return [print_samples_reward, format_reward, tool_name_reward,
            claim_coverage_reward, repetition_penalty]


# ---------------------------------------------------------------------------
# Minimum-loss (parabola) early stopping callback
# ---------------------------------------------------------------------------
try:
    from transformers import TrainerCallback  # type: ignore
except Exception:  # pragma: no cover
    TrainerCallback = object  # type: ignore


class MinLossStopCallback(TrainerCallback):
    """Stops training once the smoothed loss has clearly passed its minimum.

    We keep an EMA over the ``loss`` metric reported in state.log_history.
    After ``warmup`` steps we record the running minimum; once the EMA has
    climbed above (min + rel_tol * |min|) for ``patience`` consecutive
    logging steps we flip TrainerControl.should_training_stop.  This traces
    the downward+upward parabola the user described.
    """

    def __init__(self, warmup: int = 40, patience: int = 15,
                 ema_alpha: float = 0.1, rel_tol: float = 0.05):
        self.warmup = warmup
        self.patience = patience
        self.ema_alpha = ema_alpha
        self.rel_tol = rel_tol

        self.ema: Optional[float] = None
        self.min_ema: Optional[float] = None
        self.min_step: int = 0
        self.rises: int = 0
        self.seen: int = 0

    def _tol(self, v: float) -> float:
        return max(1e-4, abs(v) * self.rel_tol)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore
        if not logs:
            return control
        loss = logs.get("loss")
        if loss is None or not np.isfinite(float(loss)):
            return control
        loss = float(loss)
        self.seen += 1

        if self.ema is None:
            self.ema = loss
        else:
            self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * loss

        if self.seen < self.warmup:
            return control

        if self.min_ema is None or self.ema < self.min_ema - 1e-8:
            self.min_ema = self.ema
            self.min_step = state.global_step
            self.rises = 0
        elif self.ema > self.min_ema + self._tol(self.min_ema):
            self.rises += 1
        else:
            # Still near the minimum, reset the rise counter softly.
            self.rises = max(0, self.rises - 1)

        if self.rises >= self.patience:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    f"[MinLossStopCallback] Stopping: ema={self.ema:.4f}, "
                    f"min_ema={self.min_ema:.4f} at step {self.min_step}, "
                    f"rises={self.rises} (patience={self.patience}).",
                    flush=True,
                )
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
class RayTrainReportCallback(TrainerCallback):
    """Forward HF Trainer `loss` to Ray Train's metric stream on every
    log step.  Checkpoints are also reported alongside metrics on each
    `save_steps` boundary so Ray Train can keep the best-K by loss.
    """

    def __init__(self):
        try:
            import ray.train as _rt
            self._report = _rt.report
        except Exception:
            self._report = None
        self._last_loss: Optional[float] = None

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore
        if self._report is None or not logs:
            return control
        loss = logs.get("loss")
        if loss is not None and np.isfinite(float(loss)):
            self._last_loss = float(loss)
        try:
            payload: Dict[str, Any] = {
                "step": int(getattr(state, "global_step", 0)),
                "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            }
            if self._last_loss is not None:
                payload["loss"] = self._last_loss
            for k in ("learning_rate", "grad_norm", "reward",
                      "reward_std", "kl"):
                if k in logs:
                    payload[k] = logs[k]
            self._report(payload)
        except Exception as exc:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"[RayTrainReportCallback] report failed: {exc!r}",
                      flush=True)
        return control

    def on_save(self, args, state, control, **kwargs):  # type: ignore
        if self._report is None:
            return control
        try:
            import ray.train as _rt
            from ray.train import Checkpoint
            ckpt_dir = args.output_dir
            payload = {"step": int(state.global_step)}
            if self._last_loss is not None:
                payload["loss"] = self._last_loss
            self._report(payload, checkpoint=Checkpoint.from_directory(ckpt_dir))
        except Exception as exc:
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"[RayTrainReportCallback] checkpoint report failed: "
                      f"{exc!r}", flush=True)
        return control


# Model loading -- text-only path for Qwen3_5ForConditionalGeneration.
# ---------------------------------------------------------------------------
def _select_torch_dtype():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_tokenizer(model_path: str, attn_impl: str):
    """Load the Qwen3_5 text tower for GRPO.

    Transformers >= 5.6 has native ``Qwen3_5ForCausalLM`` /
    ``Qwen3_5ForConditionalGeneration``.  On older transformers we fall back
    to ``trust_remote_code`` in case the checkpoint ships Python modules.
    """
    import importlib
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _select_torch_dtype()
    load_kwargs: Dict[str, Any] = dict(
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )

    model = None
    last_err: Optional[Exception] = None

    # 1) Prefer Qwen3_5ForCausalLM directly (pure text path, smallest memory).
    for class_name in ("Qwen3_5ForCausalLM",
                       "Qwen3_5ForConditionalGeneration"):
        try:
            tx = importlib.import_module("transformers")
            cls = getattr(tx, class_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(model_path, **load_kwargs)
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"[load_model_and_tokenizer] loaded via {class_name}",
                      flush=True)
            break
        except Exception as exc:
            last_err = exc
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(f"[load_model_and_tokenizer] {class_name} failed: "
                      f"{exc!r}", flush=True)

    # 2) Fall back to AutoModelForCausalLM (old transformers + remote code).
    if model is None:
        from transformers import AutoModelForCausalLM, AutoModel
        kwargs_rc = dict(load_kwargs, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs_rc)
        except Exception as exc:
            last_err = exc
            print(f"[load_model_and_tokenizer] AutoModelForCausalLM failed: "
                  f"{exc!r}\n  trying AutoModel", flush=True)
            try:
                model = AutoModel.from_pretrained(model_path, **kwargs_rc)
            except Exception as exc2:
                last_err = exc2
                print(f"[load_model_and_tokenizer] AutoModel failed: {exc2!r}",
                      flush=True)

    if model is None:
        raise RuntimeError(
            f"Could not load {model_path}. Last error: {last_err!r}"
        )

    # If we ended up with a multimodal wrapper, project down to its text tower
    # for pure-text GRPO to avoid needlessly updating vision params.
    text_model = getattr(model, "language_model", None)
    if text_model is not None and hasattr(text_model, "generate"):
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print("[load_model_and_tokenizer] using model.language_model",
                  flush=True)
        model = text_model

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    return model, tokenizer


def maybe_wrap_with_lora(model, lora_rank: int, lora_alpha: int,
                        lora_dropout: float) -> Any:
    from peft import LoraConfig, get_peft_model
    target = ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO fine-tuning of Qwen3.6-27B on MCP-Atlas."
    )
    p.add_argument("--model_path", default=MODEL_PATH_DEFAULT)
    p.add_argument("--data_dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--work_root", default=WORK_ROOT_DEFAULT)

    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_completion_length", type=int, default=768)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--max_steps", type=int, default=3000,
                   help="Upper bound; MinLossStopCallback usually stops earlier.")
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--attn_impl",
                   default=os.environ.get("ATTN_IMPL", "flash_attention_2"))
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--min_loss_warmup", type=int, default=40)
    p.add_argument("--min_loss_patience", type=int, default=15)
    p.add_argument("--report_to", default="none")

    # Ray Train / distribution knobs.
    p.add_argument("--no_ray", action="store_true",
                   help="Skip Ray Train and run a single-process training "
                        "(useful for debugging).")
    p.add_argument("--num_workers", type=int, default=-1,
                   help="Number of Ray Train workers (GPUs).  -1 = use all "
                        "GPUs Ray can see.")
    p.add_argument("--cpus_per_worker", type=int, default=-1,
                   help="CPU resources per worker.  -1 = split all Ray CPUs "
                        "evenly across workers.")
    p.add_argument("--ray_address", default=os.environ.get("RAY_ADDRESS"),
                   help="Ray cluster address (e.g. 'auto' or "
                        "'http://<head>:8265').")
    return p.parse_args()


def _setup_logging(work_root: Path) -> None:
    log_dir = work_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rank = os.environ.get("RANK", "0")
    log_path = log_dir / f"train-{stamp}-rank{rank}.log"

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    fh = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[setup] logging to {log_path}", flush=True)


def _run_training(args: argparse.Namespace, *,
                  is_ray_worker: bool = False) -> Any:
    """Core training loop used by both the Ray Train worker and the local
    fallback path.  Assumes distributed env vars (RANK/WORLD_SIZE/LOCAL_RANK)
    are already set by the caller (either Ray Train or torchrun)."""
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "checkpoint").mkdir(exist_ok=True)
    (work_root / "adapter_final").mkdir(exist_ok=True)
    (work_root / "processed_data").mkdir(exist_ok=True)
    _setup_logging(work_root)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"[worker] rank={rank} world={world} local_rank={local_rank} "
          f"ray_worker={is_ray_worker}", flush=True)
    print(f"[worker] args: {vars(args)}", flush=True)

    import torch
    from transformers import set_seed
    set_seed(args.seed)

    # ---- Load model + tokenizer ----
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.attn_impl)
    model = maybe_wrap_with_lora(
        model, args.lora_rank, args.lora_alpha, args.lora_dropout,
    )

    # ---- Build dataset (rank 0 saves to disk once) ----
    df = load_mcp_atlas(args.data_dir)
    print(f"[worker] loaded MCP-Atlas rows: {len(df)}", flush=True)
    train_ds = build_grpo_dataset(df, tokenizer, args.max_prompt_length)
    print(f"[worker] GRPO training rows after length filter: {len(train_ds)}",
          flush=True)
    if rank == 0:
        try:
            train_ds.save_to_disk(str(work_root / "processed_data"))
        except Exception as exc:
            print(f"[worker] could not save processed_data: {exc!r}",
                  flush=True)

    # ---- GRPO config ----
    # ------------------------------------------------------------
    # trl 0.27 compat shim.
    # In trl 0.27 most ``is_<foo>_available()`` helpers are written as
    # ``return _is_package_available(pkg_name)`` -- but
    # ``_is_package_available`` actually returns a ``(bool, None)`` tuple,
    # which is *always* truthy.  Result: trl thinks vllm_ascend, weave,
    # liger_kernel, ... are installed and pulls them in at import time,
    # crashing GRPOTrainer's import chain.
    # Fix: wrap every ``is_*_available()`` so its return value is coerced
    # to a real bool -- if it's a tuple, take ``tuple[0]``.  Do NOT touch
    # transformers or trl._is_package_available itself (both are
    # internally consistent with their own call-sites).
    try:
        import trl.import_utils as _trl_imp_utils
        def _make_coerced(orig):
            def _wrapped(*a, **kw):
                try:
                    v = orig(*a, **kw)
                except Exception:
                    return False
                if isinstance(v, tuple):
                    return bool(v[0]) if v else False
                return bool(v)
            _wrapped.__wrapped__ = orig
            return _wrapped
        _patched = []
        for _name in list(vars(_trl_imp_utils)):
            if not (_name.startswith("is_") and _name.endswith("_available")):
                continue
            _orig = getattr(_trl_imp_utils, _name)
            if not callable(_orig):
                continue
            setattr(_trl_imp_utils, _name, _make_coerced(_orig))
            _patched.append(_name)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"[compat] coerced {len(_patched)} trl is_*_available()"
                  f" helpers to real booleans", flush=True)
    except Exception as _exc:
        print(f"[compat] trl import_utils patch skipped: {_exc!r}",
              flush=True)
    from trl import GRPOConfig, GRPOTrainer
    ckpt_dir = work_root / "checkpoint"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # Only pass kwargs that the installed GRPOConfig dataclass actually supports.
    # trl 0.22-0.24 keeps `max_prompt_length`; trl 1.x dropped it.  The rest
    # of the fields are widely available.
    import dataclasses as _dc
    _grpo_field_set = {f.name for f in _dc.fields(GRPOConfig)}

    def _pick(name, value):
        return (name, value) if name in _grpo_field_set else None

    _kw_items = [
        ("output_dir", str(ckpt_dir)),
        ("per_device_train_batch_size", args.per_device_train_batch_size),
        ("gradient_accumulation_steps", args.gradient_accumulation_steps),
        ("num_generations", args.num_generations),
        ("max_prompt_length", args.max_prompt_length),
        ("max_completion_length", args.max_completion_length),
        ("learning_rate", args.learning_rate),
        ("adam_beta1", 0.9),
        ("adam_beta2", 0.99),
        ("weight_decay", 0.1),
        ("warmup_ratio", args.warmup_ratio),
        ("lr_scheduler_type", "cosine"),
        ("optim", "adamw_torch"),
        ("logging_steps", args.logging_steps),
        ("max_steps", args.max_steps),
        ("save_steps", args.save_steps),
        ("save_total_limit", 3),
        ("save_strategy", "steps"),
        ("seed", args.seed),
        ("bf16", use_bf16),
        ("fp16", (not use_bf16) and torch.cuda.is_available()),
        ("gradient_checkpointing", True),
        ("temperature", 1.0),
        ("report_to", args.report_to),
        ("log_level", "info"),
        ("disable_tqdm", False),
        ("remove_unused_columns", False),
    ]
    _kwargs = {k: v for k, v in _kw_items if k in _grpo_field_set}
    _dropped = [k for k, _ in _kw_items if k not in _grpo_field_set]
    if _dropped and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[main] GRPOConfig ignoring unsupported kwargs: {_dropped}",
              flush=True)
    grpo_args = GRPOConfig(**_kwargs)

    reward_funcs = make_reward_funcs(tokenizer)
    # trl 0.27 + transformers 5.6 compat: trl's GRPOTrainer.__init__
    # mutates ``model.warnings_issued["estimate_tokens"] = True`` but
    # that attribute only exists on classes that inherit from
    # ``transformers.modeling_utils.PreTrainedModel`` *and* set the
    # class-level ``warnings_issued = {}`` dict.  The freshly ported
    # ``Qwen3_5ForCausalLM`` in transformers 5.6.2 forgets that attr,
    # and the PEFT LoraModel wrapper defers ``__getattr__`` to the
    # base model -> AttributeError.  Manually set the dict on every
    # layer that trl might reach.
    for _obj in (model, getattr(model, "base_model", None),
                 getattr(getattr(model, "base_model", None), "model", None)):
        if _obj is None:
            continue
        if not hasattr(_obj, "warnings_issued") or _obj.warnings_issued is None:
            try:
                _obj.warnings_issued = {}
            except Exception:
                pass
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_ds,
    )
    trainer.add_callback(
        MinLossStopCallback(
            warmup=args.min_loss_warmup,
            patience=args.min_loss_patience,
        )
    )
    if is_ray_worker:
        trainer.add_callback(RayTrainReportCallback())

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print("[main] starting GRPO training", flush=True)
    train_result = trainer.train()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"[main] train_result: {train_result}", flush=True)
        adapter_out = work_root / "adapter_final"
        try:
            trainer.save_model(str(adapter_out))
        except Exception as exc:
            print(f"[main] save_model failed: {exc!r}", flush=True)
            try:
                trainer.model.save_pretrained(str(adapter_out))
            except Exception as exc2:
                print(f"[main] save_pretrained fallback failed: {exc2!r}",
                      flush=True)
        try:
            tokenizer.save_pretrained(str(adapter_out))
        except Exception as exc:
            print(f"[main] tokenizer.save_pretrained failed: {exc!r}",
                  flush=True)
        print(f"[main] final adapter saved to {adapter_out}", flush=True)


def _local_fallback(args: argparse.Namespace) -> None:
    """Run a single-process / torchrun-driven training directly, bypassing
    Ray Train.  Useful for debugging or when RAY_ADDRESS is not set."""
    _run_training(args, is_ray_worker=False)


def _ray_train_loop(config: Dict[str, Any]) -> None:
    """Entry point executed inside every Ray Train worker.

    Ray Train wires RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
    before calling this, plus sets CUDA_VISIBLE_DEVICES so each worker
    sees exactly its assigned GPU.
    """
    args = argparse.Namespace(**config)
    _run_training(args, is_ray_worker=True)


def _resolve_scaling(args: argparse.Namespace) -> Dict[str, Any]:
    import ray
    cluster = ray.cluster_resources()
    n_gpus_cluster = int(cluster.get("GPU", 0))
    n_cpus_cluster = int(cluster.get("CPU", 1))
    if n_gpus_cluster == 0:
        raise RuntimeError(
            "Ray cluster reports 0 GPUs. Start Ray with `--num-gpus=N` or "
            "run this node with NVIDIA devices attached."
        )
    num_workers = args.num_workers if args.num_workers > 0 else n_gpus_cluster
    if num_workers > n_gpus_cluster:
        raise RuntimeError(
            f"--num_workers={num_workers} exceeds Ray GPUs ({n_gpus_cluster})."
        )
    if args.cpus_per_worker > 0:
        cpus = args.cpus_per_worker
    else:
        cpus = max(1, (n_cpus_cluster - 1) // max(1, num_workers))
    return {
        "num_workers": num_workers,
        "cpus_per_worker": cpus,
        "n_gpus_cluster": n_gpus_cluster,
        "n_cpus_cluster": n_cpus_cluster,
    }


def _run_with_ray_train(args: argparse.Namespace) -> None:
    import ray
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer, TorchConfig

    # Connect to (or start) Ray.
    if not ray.is_initialized():
        address = args.ray_address
        if address:
            ray.init(address=address, ignore_reinit_error=True,
                     log_to_driver=True)
        else:
            try:
                ray.init(address="auto", ignore_reinit_error=True,
                         log_to_driver=True)
            except Exception:
                ray.init(ignore_reinit_error=True, log_to_driver=True)

    scaling = _resolve_scaling(args)
    print(f"[driver] Ray cluster: GPUs={scaling['n_gpus_cluster']} "
          f"CPUs={scaling['n_cpus_cluster']}", flush=True)
    print(f"[driver] launching TorchTrainer: num_workers="
          f"{scaling['num_workers']} cpus_per_worker="
          f"{scaling['cpus_per_worker']} gpus_per_worker=1", flush=True)

    work_root = Path(args.work_root)
    (work_root / "ray_results").mkdir(parents=True, exist_ok=True)

    # Ray Train v2 (ray>=2.51) only exposes a minimal RunConfig.  We keep
    # `num_to_keep` for rolling checkpoints; best-K-by-loss would require
    # wrapping the run as a Tune trial, which we skip to stay single-trial.
    run_config = RunConfig(
        name="qwen3-27b-grpo",
        storage_path=str(work_root / "ray_results"),
        checkpoint_config=CheckpointConfig(num_to_keep=3),
    )
    scaling_config = ScalingConfig(
        num_workers=scaling["num_workers"],
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": scaling["cpus_per_worker"]},
        placement_strategy="PACK",
    )
    torch_config = TorchConfig(backend="nccl")

    trainer = TorchTrainer(
        train_loop_per_worker=_ray_train_loop,
        train_loop_config=vars(args),
        torch_config=torch_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    result = trainer.fit()
    print(f"[driver] Ray Train finished: {result}", flush=True)
    try:
        print(f"[driver] best checkpoint: {result.checkpoint}", flush=True)
        print(f"[driver] best metrics:    {result.metrics}", flush=True)
    except Exception:
        pass


def main() -> None:
    args = _parse_args()
    if args.no_ray:
        _local_fallback(args)
    else:
        _run_with_ray_train(args)


if __name__ == "__main__":
    main()
