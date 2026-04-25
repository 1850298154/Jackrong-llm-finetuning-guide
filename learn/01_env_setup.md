# 01 · 环境搭建：uv + LLaMA-Factory + TRL + DeepSpeed

> 目标：在 `/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas/venv` 下建一个
> Python 3.11 的虚拟环境，把 **SFT（LLaMA-Factory）** 和 **GRPO（TRL）**
> 所需的训练栈一次装齐，之后不用再动。

## 1. 为什么选 uv 而不是 conda

- **快**：uv 用 Rust 写的 resolver + 并行下载，比 pip/conda 快 5~10×。
- **纯 venv**：产物就是一个 `venv/` 目录，**搬迁/复制/备份都友好**，不会像
  conda 那样把自己钉死在机器上。
- **lockfile**：后面我们可以 `uv pip freeze > requirements.lock` 复现环境。

## 2. 一键装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/home/z/.local/bin:$PATH   # ~/.bashrc 不可写时，手工 export
```

## 3. 建虚拟环境（放到训练输出目录里，和 checkpoint 同宿主）

```bash
BASE=/storage/RL/user/ytzhao02/train/03_mcp/MCP-Atlas
cd $BASE
export UV_LINK_MODE=copy        # /home 与 /storage 跨文件系统，不能 hardlink
uv venv venv --python 3.11 --seed
```

> 小坑：`uv` 默认想用 hardlink 加速；跨文件系统时会降级到复制并打 warning，
> 加 `UV_LINK_MODE=copy` 就干净了。

## 4. 克隆并安装 LLaMA-Factory（editable）

```bash
cd /home/z/zyt
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
ln -sfn /home/z/zyt/LLaMA-Factory $BASE/code/LLaMA-Factory

uv pip install --python $BASE/venv/bin/python -e /home/z/zyt/LLaMA-Factory
```

`editable` 模式的好处：后面想读源码、改 trainer 行为、加自定义 callback
时**直接改文件就生效**，不用重装。

## 5. 装补充依赖：deepspeed / bitsandbytes / wandb / mcp

```bash
uv pip install --python $BASE/venv/bin/python \
    deepspeed bitsandbytes wandb mcp pyarrow
```

- **deepspeed**：ZeRO-1/2/3、offload、BF16 混合精度——27B 多卡训练的引擎。
- **bitsandbytes**：LoRA + 4/8-bit 量化，省显存。
- **wandb**：可选，训练曲线远端可视化；不需要的话 `report_to=none`。
- **mcp**：官方 Model Context Protocol SDK，我们用它的 schema 定义模拟工具。
- **pyarrow**：读 parquet 用。

## 6. Sanity check

```python
import torch, transformers, trl, peft, accelerate, deepspeed, bitsandbytes
import llamafactory, mcp, datasets
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
```

本项目此刻装到的版本（仅供参考）：

```
torch           2.11.0+cu130
transformers    5.2.0
trl             0.24.0
peft            0.18.1
accelerate      1.11.0
deepspeed       0.18.9
bitsandbytes    0.49.2
llamafactory    0.9.5.dev0
datasets        4.0.0
mcp             1.27.0
```

## 7. 学到 / 踩到的

1. **shell 里 `nvidia-smi` 和 `/dev/nvidia*` 都可能缺失**，哪怕机器上 8 张
   GPU 实打实在 `/proc/driver/nvidia/gpus/` 里。这种情况常见于：
   - 容器没挂 `--device /dev/nvidia*`；
   - nvidia-container-toolkit 没配；
   - 你在宿主的 non-GPU namespace 里。

   训练前一定要在目标 pod 里先跑一次 `nvidia-smi` 和
   `python -c "import torch; print(torch.cuda.device_count())"`。

2. **torch 是 cu130 预编译**。实测与驱动 580.x 兼容。如果 driver 老于
   545.x，会出现 `undefined symbol`，要降级 torch 或升级驱动。

3. **LLaMA-Factory 的 `pyproject.toml` 会拉一堆 UI/API 依赖**（gradio、
   fastapi、uvicorn）。我们不用 WebUI，但多装这些不贵，保持完整性。

下一篇 → `02_data_prep.md`
