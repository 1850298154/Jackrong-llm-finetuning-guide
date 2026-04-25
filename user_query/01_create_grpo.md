现在需要你做的是参考 /storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train_code/Llama-3.2-3B-R1-Zero-GRPO.ipynb  代码，
实现一个 对 /storage/RL/models/download/Qwen3.6-27B 下载好的模型 进行 GRPO 的训练，
代码存放到 /storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train_code/Qwen3.6-27B-GRPO.py 你自己实现。

其中checkpoint存储到 /storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train/03_mcp/MCP-Atlas/checkpoint 下面 ，
需要任何存储到你自行组织就行了，在 /storage/RL/user/ytzhao02/zyt/Jackrong-llm-finetuning-guide/train/03_mcp/MCP-Atlas/ 下面，
你的训练数据是  /storage/RL/data/download/03_mcp/dataset/MCP-Atlas （15M，7 个文件） .
你需要训练，直到达到loss最低点（抛物线）的。

路径没有的你需要自己创建。
训练必须提交 ray 进行训练，使用最多的GPU和经可能多的CPU。
