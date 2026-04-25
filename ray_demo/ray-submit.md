## 3. 提交作业
基于步骤 2 登陆代理 pod 之后，可以通过 ray 的命令提交（pod 的镜像已经安装了正确版本的 ray，不需要自己安装），比如：
**[代理 pod 上]执行：**
`ray job submit --address http://<k8s-rayhead-internal-ip>:8265 -- echo hello`
* **可以自定义入口脚本**
* **ray job submit 支持指定执行这个脚本需要的资源：**
| 参数 | 类型 | 描述 |
| :--- | :--- | :--- |
| `--entrypoint-num-cpus` | FLOAT | the quantity of CPU cores to reserve for the entrypoint command, separately from any tasks or actors that are launched by it |
| `--entrypoint-num-gpus` | FLOAT | the quantity of GPUs to reserve for the entrypoint command, separately from any tasks or actors that are launched by it |
| `--entrypoint-memory` | INTEGER | the amount of memory to reserve for the entrypoint command, separately from any tasks or actors that are launched by it |
* **ray job submit 支持定制作业每一个进程的 runtime env，包括：**
    * 环境变量
    * 运行时的目录，支持在提交侧指定本地目录/`http://xxxx.zip` 等方式指定。比较常见的方式是用户在自己的项目编程，把代码放到当前目录，然后指定 `working_dir` 为 `.`。但是请注意，指定本地目录的时候最好只包含代码，不要包含模型，这个目录不宜超过 5MB。
    * pip 安装包
* 这里只是第一次接触 ray 可能会比较需要的功能，更加具体的可见：
    [https://docs.rayai.org.cn/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-submit](https://docs.rayai.org.cn/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-submit)
* 提交完之后，可以在本地浏览器查看作业情况：
http://127.0.0.1:8265/#/overview