# Megakernels 学习文档

本目录包含 Megakernels 项目的完整学习文档，帮助你从零理解这个高性能 GPU LLM 推理框架的原理与代码实现。

## 推荐阅读顺序

| 文档 | 内容 |
|------|------|
| [01-overview.md](01-overview.md) | 项目背景、问题与核心思想 |
| [02-architecture.md](02-architecture.md) | 整体技术架构与层次图 |
| [03-instructions.md](03-instructions.md) | 指令系统：Opcode、Globals、序列化 |
| [04-scheduling.md](04-scheduling.md) | DAG 调度：构建、SM 分配、张量化 |
| [05-llama-model.md](05-llama-model.md) | Llama 模型实现：权重堆叠、KV 缓存、张量并行 |
| [06-execution-backends.md](06-execution-backends.md) | 三种执行后端：PyTorch / PyVM / CUDA MK |
| [07-cuda-megakernel.md](07-cuda-megakernel.md) | CUDA 状态机深度解析 |
| [08-demo-walkthrough.md](08-demo-walkthrough.md) | 完整 Demo：从加载模型到逐步生成一个 Token |

## 前置知识

- Python 3.10+，熟悉 PyTorch
- 了解 Transformer/Llama 模型结构（Attention、MLP、RoPE、LayerNorm）
- 对 CUDA 编程有基本概念（SM、warp、shared memory）有助于理解第 7 章

## 快速上手

```bash
# 安装依赖
pip install -e .

# 运行 PyVM 模式（无需编译 CUDA）
python -m megakernels.scripts.generate \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode pyvm \
    --schedule wave \
    --ntok 20

# 运行原生 PyTorch 模式（对照基准）
python -m megakernels.scripts.generate \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode torch \
    --ntok 20
```

## 源码目录速览

```
megakernels/               # 核心 Python 包
  instructions.py          # 指令基类、BaseGlobals
  scheduler.py             # DAG 节点、调度算法、张量化
  generators.py            # 生成器：PyTorch / MK / PyVM
  dispatch.py              # 工厂函数（按 mode 路由）
  llama.py                 # Llama 模型完整实现
  model_types.py           # BatchState、ExtraModelConfig
  demos/
    latency/               # 低延迟模式（7 个 opcode）
      instructions.py
      scheduler.py
      python_vm.py
      mk.py
    throughput/            # 高吞吐模式

include/                   # CUDA 框架头文件
  megakernel.cuh           # GPU VM 入口 mk_internal()
  controller/controller.cuh # 指令调度主循环

demos/low-latency-llama/   # 完整 CUDA Megakernel 实现
  llama.cuh                # globals_t、硬件配置
  llama.cu                 # pybind11 绑定
  attention_partial.cu     # 流水线注意力核
  ...
```
