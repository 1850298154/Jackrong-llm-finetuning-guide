"""最简单的 Ray 作业：不需要 GPU，用来验证提交链路是否通。"""
import os, socket, ray

ray.init(address="auto")

@ray.remote
def hello(i: int) -> str:
    return f"[task {i}] hello from host={socket.gethostname()} pid={os.getpid()}"

for msg in ray.get([hello.remote(i) for i in range(4)]):
    print(msg)

print("cluster_resources =", ray.cluster_resources())
