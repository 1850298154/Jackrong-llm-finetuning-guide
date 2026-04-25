"""探测 Ray 集群：节点、资源、每节点 GPU，以及 worker 里是否真能看到物理 GPU。"""
import json, os, socket, subprocess, shutil
import ray

ray.init(address="auto")

print("=== cluster_resources ===")
print(json.dumps(ray.cluster_resources(), indent=2, default=str))
print("=== available_resources ===")
print(json.dumps(ray.available_resources(), indent=2, default=str))

print("=== nodes ===")
for n in ray.nodes():
    print(json.dumps({
        "NodeID": n.get("NodeID"),
        "NodeManagerAddress": n.get("NodeManagerAddress"),
        "NodeName": n.get("NodeName"),
        "Alive": n.get("Alive"),
        "Resources": n.get("Resources"),
    }, indent=2, default=str))

@ray.remote(num_gpus=1)
def probe_gpu(i):
    info = {
        "task": i,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch.cuda.is_available"] = torch.cuda.is_available()
        info["torch.cuda.device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["torch_error"] = repr(e)
    if shutil.which("nvidia-smi"):
        try:
            info["nvidia-smi"] = subprocess.check_output(
                ["nvidia-smi", "-L"], text=True, timeout=10
            ).strip()
        except Exception as e:
            info["nvidia-smi_error"] = repr(e)
    else:
        info["nvidia-smi"] = "not-found"
    return info

n_gpu = int(ray.cluster_resources().get("GPU", 0))
print(f"=== launching {max(n_gpu,1)} probe tasks (num_gpus=1 each) ===")
futs = [probe_gpu.remote(i) for i in range(max(n_gpu, 1))]
for r in ray.get(futs):
    print(json.dumps(r, indent=2, default=str))
