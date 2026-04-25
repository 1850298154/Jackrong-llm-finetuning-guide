"""Ray GPU 作业示例：把集群里**所有** GPU 全部跑起来。

- 读集群 GPU 总数 N，起 N 个 @ray.remote(num_gpus=1) task，每个 task 独占一张物理卡；
- 每个 task 在自己那张 GPU 上做 fp16 矩阵乘，打印 host / CUDA_VISIBLE_DEVICES / GPU UUID；
- 结束后校验：(host, CUDA_VISIBLE_DEVICES) 对必须 N 个都不同，且 GPU UUID 也要有 N 个不同，
  证明 N 张不同的物理 GPU 都被独立调度并完成了计算。
"""
import os
import socket
import subprocess
import time

import ray
import torch


@ray.remote(num_gpus=1)
def matmul_on_gpu(idx: int, size: int = 8192, iters: int = 10) -> dict:
    assert torch.cuda.is_available(), "CUDA 不可用"
    dev = torch.device("cuda")
    # Ray 设置了 CUDA_VISIBLE_DEVICES，让本进程只看到 1 张卡，映射成 cuda:0
    gpu_name = torch.cuda.get_device_name(0)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    # 用 nvidia-smi 的 -i 精确查询"本 task 分到的那一张"的 UUID。
    # 注意：-i 读的是宿主上的物理索引，所以要取 CUDA_VISIBLE_DEVICES 里的原始编号，
    # 而不是 cuda:0。
    try:
        phys_idx = cvd.split(",")[0].strip() if cvd else "0"
        uuid = subprocess.check_output(
            ["nvidia-smi", "-i", phys_idx,
             "--query-gpu=uuid", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()[0]
    except Exception as e:
        uuid = f"uuid-unavailable: {e!r}"

    a = torch.randn(size, size, device=dev, dtype=torch.float16)
    b = torch.randn(size, size, device=dev, dtype=torch.float16)

    torch.cuda.synchronize()
    t0 = time.time()
    c = a
    for _ in range(iters):
        c = c @ b
    torch.cuda.synchronize()
    dt = time.time() - t0

    # 2 * N^3 FLOPs per matmul
    tflops = (2.0 * size ** 3 * iters) / dt / 1e12

    return {
        "task": idx,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "CUDA_VISIBLE_DEVICES": cvd,
        "gpu_name": gpu_name,
        "gpu_uuid": uuid,
        "mem_alloc_MB": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
        "size": size,
        "iters": iters,
        "sum": float(c.float().sum().item()),
        "seconds": round(dt, 3),
        "tflops_fp16": round(tflops, 1),
    }


def main() -> None:
    ray.init(address="auto")
    res = ray.cluster_resources()
    total_gpu = int(res.get("GPU", 0))
    total_cpu = int(res.get("CPU", 0))
    print(f"[driver] cluster resources: GPU={total_gpu}, CPU={total_cpu}")
    assert total_gpu > 0, "集群里拿不到 GPU，检查集群或 @ray.remote(num_gpus=...)"

    print(f"[driver] launching {total_gpu} tasks, each with num_gpus=1")
    t0 = time.time()
    futures = [matmul_on_gpu.remote(i) for i in range(total_gpu)]
    results = ray.get(futures)
    wall = time.time() - t0

    # 按 host 分组打印
    by_host: dict[str, list[dict]] = {}
    for r in results:
        by_host.setdefault(r["host"], []).append(r)
    for host in sorted(by_host):
        rs = sorted(by_host[host], key=lambda x: x["CUDA_VISIBLE_DEVICES"] or "")
        print(f"\n== host {host}  ({len(rs)} GPUs used) ==")
        for r in rs:
            print(
                f"  task={r['task']:>3}  CVD={r['CUDA_VISIBLE_DEVICES']:>2}  "
                f"{r['gpu_name']}  uuid={r['gpu_uuid']}  "
                f"{r['seconds']}s  ~{r['tflops_fp16']} TFLOPs(fp16)"
            )

    # 校验 1：每张卡的 (host, CVD) 必须唯一
    host_cvd_pairs = {(r["host"], r["CUDA_VISIBLE_DEVICES"]) for r in results}
    # 校验 2：物理 UUID 也要唯一
    unique_uuids = {r["gpu_uuid"] for r in results}
    total_tflops = sum(r["tflops_fp16"] for r in results)

    print(
        f"\n[driver] requested GPUs={total_gpu}, tasks completed={len(results)}, "
        f"unique (host,CVD)={len(host_cvd_pairs)}, unique UUIDs={len(unique_uuids)}, "
        f"aggregate ~{total_tflops:.0f} TFLOPs(fp16), wall={wall:.2f}s"
    )
    assert len(host_cvd_pairs) == total_gpu, (
        f"(host, CUDA_VISIBLE_DEVICES) 去重后只有 {len(host_cvd_pairs)} 个，"
        f"应为 {total_gpu}"
    )
    assert len(unique_uuids) == total_gpu, (
        f"物理 GPU UUID 去重后只有 {len(unique_uuids)} 个，应为 {total_gpu}"
    )
    print(f"[driver] OK: {total_gpu} 张物理 GPU 全部被独立占用并完成了计算。")


if __name__ == "__main__":
    main()
