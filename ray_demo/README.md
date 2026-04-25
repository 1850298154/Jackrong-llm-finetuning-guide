# Ray GPU 作业提交指南（代理 Pod → KubeRay 集群）

这份文档面向第一次在本项目的 **KubeRay 集群** 上提交作业的同学。你不用自己装 Ray、不用自己起集群，只需要：

1. 登录**代理 pod**（镜像里已经装好了与集群同版本的 Ray 和 CUDA 版 PyTorch）；
2. 把要跑的代码放在当前目录；
3. 用 `ray job submit` 命令提交。

本目录已经准备好 3 个可直接跑的脚本，照着「快速开始」走一遍就行。文档里每条命令都在本集群实测通过，附录里有真实日志。

> **官方原始文档**：[`ray-submit.md`](./ray-submit.md)（平台方提供的简版说明，本 README 是在它之上的**完整可执行版**）。
> 本 README 不改 `ray-submit.md`，只引用并补充：真实集群 IP、可直接跑的脚本、每张物理 GPU 的实测验收。

---

## 🚀 TL;DR（先解决 3 个最常见的困惑）

1. **"启动真实 GPU 需要加什么参数？"**
   —— **不需要任何命令行参数**。真实 GPU 是靠代码里 `@ray.remote(num_gpus=1)` 申请的，`ray job submit` 只要能连上集群就行。

2. **"那 `--entrypoint-num-cpus / --entrypoint-num-gpus / --entrypoint-memory` 是干嘛的？"**
   —— 它们是给**入口脚本（driver）自己**预留资源用的，与你 task/actor 要几张 GPU **没有关系**。绝大多数情况**保持默认不加**；若错手加 `--entrypoint-num-gpus 1`，反而会白白占一张卡让它闲着。（见 §2.2、§3）

3. **"怎么把集群里所有 GPU 都跑起来？"**
   —— 代码里用 `ray.cluster_resources()["GPU"]` 读总数 N，然后投 N 个 `@ray.remote(num_gpus=1)` task。本目录 `gpu_task.py` 就是这样做的，实测把本集群 **32 张 H100 物理卡全部独立占用并完成了计算**（§附录）。

---

## 0. 集群信息（复制即用）

| 参数 | 值 |
|---|---|
| Ray Dashboard / Job Submit 地址 | `http://172.17.193.214:8265` |
| Ray 版本 | `2.54.0`（代理 pod 与集群一致） |
| GPU 型号 | `NVIDIA H100 80GB HBM3` |
| 集群总资源 | **32 GPU / 544 CPU**（4 个 GPU worker pod × 每 pod 8×H100） |
| worker 镜像自带 | `torch 2.10.0+cu129`、`nvidia-smi`、CUDA 12.9 |

> ⚠️ 这是 **K8s 内网 IP，只能在代理 pod 里访问**。本机直连需自行做端口转发 / Service 暴露，协议仍是 `http://<ip>:8265`。

所有命令统一用环境变量，方便替换：

```bash
export RAY_ADDRESS=http://172.17.193.214:8265
```

设置后，后续 `ray job ...` 命令会自动读它，不用再写 `--address`。

---

## 1. 快速开始（3 步跑起来）

### Step 1 — 登录代理 pod，拉代码

```bash
# 在代理 pod 内
git clone <your-repo>
cd <your-repo>/ray_demo
```

### Step 2 — 验证链路（不占 GPU，几秒钟）

```bash
export RAY_ADDRESS=http://172.17.193.214:8265

ray job submit \
  --working-dir . \
  -- python hello.py
```

成功会打印 4 行 `hello from host=...`，并以 `Job '...' succeeded` 结尾。

### Step 3 — 把集群里**所有** GPU 全部跑起来（本集群 32 张 H100）

```bash
export RAY_ADDRESS=http://172.17.193.214:8265

ray job submit \
  --working-dir . \
  -- python gpu_task.py
```

预期输出（本集群实测节选）：

```
[driver] cluster resources: GPU=32, CPU=544
[driver] launching 32 tasks, each with num_gpus=1

== host verl-vllm017-gpu-workers-worker-sp6hr  (8 GPUs used) ==
  task=  0  CVD= 0  NVIDIA H100 80GB HBM3  uuid=GPU-04dacd5a-...  0.11s   ~99.8 TFLOPs(fp16)
  ...（共 8 行，CVD 0..7 各一张卡）
...（共 4 个 worker pod，每个 pod 8 张卡，合计 32 条）

[driver] requested GPUs=32, tasks completed=32,
         unique (host,CVD)=32, unique UUIDs=32,
         aggregate ~2730 TFLOPs(fp16), wall=10.82s
[driver] OK: 32 张物理 GPU 全部被独立占用并完成了计算。
Job 'raysubmit_xxx' succeeded
```

校验靠这两行**同时成立**才算"真·全部跑起来"：
- `unique (host, CUDA_VISIBLE_DEVICES) == 集群 GPU 总数`；
- `unique physical GPU UUIDs == 集群 GPU 总数`。

`gpu_task.py` 里内置了这两条 `assert`，跑完会自动校验；校验不过直接让 job 失败。

---

## 2. `ray job submit` 命令详解

### 2.1 命令模板（**不需要**任何 `--entrypoint-*` 就能启动全部 GPU）

```bash
# 最小版：足够启动集群全部物理 GPU
ray job submit \
  --address http://172.17.193.214:8265 \     # 或 export RAY_ADDRESS=...
  --working-dir . \                          # 当前目录打包上传给各 worker
  -- python gpu_task.py                      # "--" 前后要有空格
```

完整可选参数总览（**以下参数都是可选调优项**）：

```bash
ray job submit \
  --address http://172.17.193.214:8265 \
  --working-dir . \
  --runtime-env runtime_env.yaml \           # 可选：pip / env_vars 等
  --entrypoint-num-cpus 1 \                  # 可选：driver 预留 CPU（见 §2.2 警告）
  --entrypoint-num-gpus 0 \                  # 可选：driver 预留 GPU（几乎总是 0）
  --entrypoint-memory 2147483648 \           # 可选：driver 预留内存（字节），此处 2GiB
  --no-wait \                                # 可选：提交后立刻返回（后台跑）
  -- python your_script.py --your-args
```

### 2.2 `--entrypoint-*` 参数专题（这组参数最容易误解）

> 官方原话（`ray-submit.md` 里的定义）：
> - `--entrypoint-num-cpus`: reserve for the **entrypoint command**, separately from any tasks or actors that are launched by it
> - `--entrypoint-num-gpus`: reserve for the **entrypoint command**, separately from any tasks or actors that are launched by it
> - `--entrypoint-memory`: reserve for the **entrypoint command**, separately from any tasks or actors that are launched by it

关键词是 **"entrypoint command"** 和 **"separately from any tasks or actors"**：

| 东西 | 作用对象 | 何时用 |
|---|---|---|
| `@ray.remote(num_gpus=N)`（**代码里**） | **task / actor 进程** — 真正跑 CUDA kernel 的 worker | 几乎永远需要，它决定每个任务要几张物理卡 |
| `--entrypoint-num-gpus N`（**提交命令**） | **driver 进程** — 也就是 `python your_script.py` 这个入口进程本身 | 极少用，只有入口脚本**自己**直接跑 CUDA 时才需要 |
| `--entrypoint-num-cpus N` | 同上，给 driver 预留 CPU | driver 本身吃 CPU 时才加 |
| `--entrypoint-memory N`（字节） | 同上，给 driver 预留内存 | driver 本身吃内存时才加 |

记忆口诀：
- **`@ray.remote(...)` = "干活的 worker 需要多少资源"**；
- **`--entrypoint-*` = "发号施令的 driver 自己需要多少资源"**。

结论：
- **启动真实物理 GPU ≠ 加 `--entrypoint-num-gpus`。** 真实 GPU 是 `@ray.remote(num_gpus=...)` 调度出来的。
- driver 只做"提交 task、收集结果"时根本不摸 GPU，`--entrypoint-num-gpus` 保持默认 0 即可。
- 错手填 `--entrypoint-num-gpus 1`，相当于让 driver **多占一张 H100 坐着不动**，还可能拖慢调度。

### 2.3 其它参数说明

| 参数 | 含义 | 什么时候用 |
|---|---|---|
| `--address` | Dashboard 地址 `http://172.17.193.214:8265` | 每次都要（或设 `RAY_ADDRESS`） |
| `--working-dir .` | 把当前目录打包上传到 worker，作为进程工作目录 | 几乎每次都需要 |
| `--runtime-env <file>` | 从 YAML 读 runtime env | 需要额外依赖/环境变量时 |
| `--runtime-env-json '<json>'` | 行内 JSON 写法 | 想一行搞定 |
| `--no-wait` | 提交后立刻返回，不 tail 日志 | 长作业建议加 |
| `--` | 分隔符，后面是入口命令 | 每次都要 |

### 2.4 `--working-dir` 的规矩（`ray-submit.md` §3 要点 + 补充）

- 该目录里的所有文件会被打包上传到每个 worker，worker 解压后作为工作目录；
- **目录大小建议 ≤ 5MB**（平台方在 `ray-submit.md` 里明确提醒），越小启动越快；
- **不要把模型权重/数据集放进去**。大文件走共享存储（PVC）、对象存储、HF Hub，代码里按路径/URL 读；
- 也可以用远端 zip：`--working-dir "https://bucket/path/code.zip"`（`ray-submit.md` 里也列出了）。

### 2.5 `runtime_env` 的用法

**方式 A：YAML 文件**（见本目录 `runtime_env.yaml`）

```yaml
env_vars:
  HF_HUB_ENABLE_HF_TRANSFER: "1"
  HF_HOME: "/mnt/shared/hf-cache"
pip:
  - "transformers==4.46.0"
  - "accelerate>=0.34"
```

```bash
ray job submit --working-dir . --runtime-env runtime_env.yaml -- python train.py
```

**方式 B：行内 JSON**

```bash
ray job submit \
  --working-dir . \
  --runtime-env-json '{"pip":["transformers==4.46.0"],"env_vars":{"HF_HUB_ENABLE_HF_TRANSFER":"1"}}' \
  -- python train.py
```

- `pip`: worker 启动时临时追加安装；worker 镜像已有的不用写；
- `env_vars`: 每个 worker 进程都看到这些环境变量；
- `working_dir`: 也可以写在 runtime_env 里，和命令行 `--working-dir` 二选一。

---

## 3. 代码里怎么"真的拿到 GPU"

**核心一句话：在任务/actor 的装饰器里写 `num_gpus=...`。和 `--entrypoint-*` 无关。**

```python
import ray, torch

@ray.remote(num_gpus=1)                 # ← 关键：向调度器申请 1 张物理 GPU
def work(i):
    assert torch.cuda.is_available()    # worker 里会自动只看到 1 张卡
    # Ray 已经把 CUDA_VISIBLE_DEVICES 设好
    return torch.randn(4096, 4096, device="cuda").sum().item()

ray.init(address="auto")                # 在 ray job submit 里，自动连到当前集群
print(ray.get([work.remote(i) for i in range(32)]))   # 32 张卡全跑起来
```

实测事实：

- Ray 会自动设置 `CUDA_VISIBLE_DEVICES`，worker 里 `torch.cuda.device_count()` = 你申请到的卡数；
- 一个 task 独占多卡（单进程 DDP/FSDP）：`@ray.remote(num_gpus=8)`；
- 跨 pod 多机多卡训练：推荐 `ray.train.torch.TorchTrainer` + `ScalingConfig(num_workers=N, use_gpu=True)`。

### 想跑满**全部** GPU 的正确姿势

```python
ray.init(address="auto")
total_gpu = int(ray.cluster_resources().get("GPU", 0))   # 当前 32
futures = [work.remote(i) for i in range(total_gpu)]      # 一次性投 N 个 task
ray.get(futures)
```

校验是否真·独占了 N 张不同的物理卡：

- 每个 task 里拿 `(socket.gethostname(), os.environ["CUDA_VISIBLE_DEVICES"])` 去重后应 = N；
- 更严格：用 `nvidia-smi -i $CVD --query-gpu=uuid --format=csv,noheader` 拿那一张卡的**物理 UUID**，去重后也应 = N。

本目录的 `gpu_task.py` 已经包含这两条断言。

---

## 4. 作业生命周期管理

```bash
export RAY_ADDRESS=http://172.17.193.214:8265

ray job list                    # 列出所有作业
ray job status <job_id>         # 查询状态（PENDING/RUNNING/SUCCEEDED/FAILED/STOPPED）
ray job logs  <job_id>          # 打印全部日志
ray job logs  <job_id> --follow # 实时 tail
ray job stop  <job_id>          # 主动停止
```

`<job_id>` 是提交时返回的 `raysubmit_xxxxxxxx` 串。
Dashboard（浏览器打开 `http://172.17.193.214:8265`）能看节点、资源、作业、Actor、日志。

---

## 5. 本目录文件说明

| 文件 | 作用 |
|---|---|
| `ray-submit.md` | **平台方官方简版文档**（原样保留，本 README 在它之上做的补充） |
| `README.md` | 本文档（完整可执行指南） |
| `hello.py` | 最小样例，不占 GPU，验证提交链路 |
| `probe.py` | 集群探针：打印 `cluster_resources` / `nodes`，并在每张 GPU 上落一个探针任务返回 `nvidia-smi -L`、torch CUDA 可用性等 |
| `gpu_task.py` | **把集群里所有 GPU 全部跑起来**的示例（本集群 32 张 H100），自带双重校验 |
| `runtime_env.yaml` | `--runtime-env` 示例文件，按需改 `env_vars` / `pip` |

### 三条"一键运行"命令（在代理 pod、本目录下执行）

```bash
export RAY_ADDRESS=http://172.17.193.214:8265

# 1) 最简链路验证（不占 GPU）
ray job submit --working-dir . -- python hello.py

# 2) 集群/GPU 探针（会在每张 GPU 上落一个探针任务）
ray job submit --working-dir . -- python probe.py

# 3) 真 GPU 作业：占满全部 32 张 H100，自动校验"每张物理卡都用到了"
ray job submit --working-dir . -- python gpu_task.py
```

---

## 6. 常见坑

1. **`Invalid address format: http://...`**
   `ray status` / `ray attach` 这些**集群级命令**走的是 GCS 端口（`:6379`），不是 dashboard（`:8265`）。要看集群状态，要么打开 dashboard 网页，要么像本目录 `probe.py` 那样通过作业查。
   只有 `ray job *` 命令吃 `http://<ip>:8265`。

2. **`torch.cuda.is_available()` 在 worker 里是 False**
   - 忘了写 `@ray.remote(num_gpus=...)`：Ray 没给该进程分配 GPU，`CUDA_VISIBLE_DEVICES` 是空；
   - 或者作业被调度到了非 GPU 节点（比如 head）。看 `ray job logs` 里打印的 `host=...`，应为 `...-gpu-workers-worker-...`。

3. **用 `nvidia-smi` 校验 UUID 时看到"同一个 UUID 出现多次"**
   默认 `nvidia-smi --query-gpu=uuid` 列出**该 pod 里所有可见的物理 GPU**（不是 Ray 划给你的那一张）。
   正确做法：用 `-i <CUDA_VISIBLE_DEVICES>` 明确指定物理索引，见 `gpu_task.py`。

4. **代码上传很慢 / 很大**
   `--working-dir` 里混进了 `.git`、venv、模型权重、数据集。
   解决：
   - 建一个干净的提交子目录；
   - 大文件放共享存储 / 对象存储，代码里按路径读；
   - Ray 会自动读 `working_dir` 父级起的 `.gitignore` 来过滤。

5. **Ray 版本不一致**
   代理 pod 和集群 head 必须同一大版本。本集群是 `2.54.0`。如果在别处提交，先 `ray --version` 对一下。

6. **`--entrypoint-num-gpus` 填错**（最高频踩坑）
   别习惯性填 1，它只给 driver 自己占卡，会白白浪费一张。除非入口脚本在 driver 进程里直接跑 CUDA，否则保持 0。

7. **长作业日志断流**
   用 `--no-wait` 提交，之后 `ray job logs <id> --follow` 重新接上即可；作业本身在集群里继续跑，不受终端影响。

---

## 7. 参考

- **本目录** `ray-submit.md` — 平台方原版简述（IP、`--entrypoint-*`、`working_dir` 要求）。
- Ray Jobs CLI 快速开始: https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html
- `ray job submit` 全部选项: https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-submit
  （中文镜像：https://docs.rayai.org.cn/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-submit ）
- Runtime Environments: https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
- 在 Ray 中使用 GPU: https://docs.ray.io/en/latest/ray-core/tasks/using-ray-with-gpus.html
- 多机多卡训练（Ray Train）: https://docs.ray.io/en/latest/train/train.html
- KubeRay: https://github.com/ray-project/kuberay

---

## 附：实测验收

**验收 job：`raysubmit_VJcu9rzcsqYr4PFF` → `SUCCEEDED`**（本仓库最后一次跑的）

- 提交命令（**不含任何 `--entrypoint-*`**）：
  ```bash
  export RAY_ADDRESS=http://172.17.193.214:8265
  ray job submit --working-dir . -- python gpu_task.py
  ```
- 集群识别资源：`GPU=32, CPU=544`；
- 提交 32 个 `@ray.remote(num_gpus=1)` task，全部完成；
- 按 host 分布：4 个 worker pod（`gz5lp` / `m6j8k` / `pqkh6` / `sp6hr`），每 pod 恰好 8 张卡，`CVD=0..7` 一个不落；
- **`unique (host, CUDA_VISIBLE_DEVICES) = 32`、`unique physical GPU UUIDs = 32`**，严格证明 32 张不同的物理 H100 都跑到了；
- 单卡 fp16 matmul ~40–105 TFLOPs（8192×8192 × 10 iters），聚合 **~2730 TFLOPs**，墙钟 **10.82s**；
- 另验证 `probe.py`：每张 H100 都从 worker 侧返回了自己的 `GPU-UUID` 与 `torch.cuda.is_available()=True`。

**额外对照实验（证明 `--entrypoint-num-gpus` 与"启动 GPU"无关）**：
- 前次 job `raysubmit_hPkrvexC4vgec5QX`：同样**不带**任何 `--entrypoint-*` 参数，同样 32 张 H100 全部独立占用、`SUCCEEDED`（aggregate ~2651 TFLOPs, wall 11.17s）；
- 结论一致：**真实物理 GPU 的调度完全由 `@ray.remote(num_gpus=...)` 决定**，`--entrypoint-*` 只影响 driver 自己的资源预留。

结论：**代理 pod → `ray job submit` → KubeRay 集群 → 32 张 H100 物理 GPU 全部调起并完成计算**的链路走通，且无需任何 `--entrypoint-*` 参数。可以直接按本文跑自己的业务脚本。
