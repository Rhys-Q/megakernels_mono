# 01 — 项目概述：Megakernels 是什么，为什么存在

## 背景：LLM 推理的性能瓶颈

大语言模型（LLM）的推理分两个阶段：

- **Prefill（预填充）**：一次性处理整个 prompt，计算量大，GPU 通常能高效利用。
- **Decode（逐词解码）**：每次只生成一个 token，需要完整地跑一遍 Transformer forward pass，但计算量极小（只有一个 token 的隐状态需要变换）。

Decode 阶段是低延迟推理的瓶颈。问题出在哪里？

### 传统框架的问题

在 PyTorch 或大多数推理框架中，每一步 Decode 的执行路径大致是：

```
for layer in transformer_layers:
    hidden = attention(hidden, kv_cache)   # kernel 1
    hidden = mlp(hidden, ...)              # kernel 2, 3, 4...
logits = lm_head(hidden)
```

每个 `kernel` 都是独立调度到 GPU 的。对于一个 32 层的 Llama 模型，一次 Decode 可能需要启动 **100+ 个 kernel**。每次 kernel launch 都有固定开销（CPU→GPU 调度延迟），加起来就是数毫秒的额外耗时。

更深层的问题是：**SM（流处理器）利用率**。Decode 时每层的矩阵向量乘法（matvec）规模很小，单个 kernel 往往只能占用少数 SM，其他 SM 空转等待。

### Megakernels 的核心思想

Megakernels 的解法是：**把整个 forward pass 表达成一张指令 DAG，然后用一个 CUDA kernel（"megakernel"）在 GPU 上自己调度并执行这些指令**。

```
传统方式:  CPU → [kernel1] → [kernel2] → [kernel3] → ... → CPU

Megakernels:  CPU → [megakernel: 内部自己执行所有指令] → CPU
```

关键特性：

1. **只有一次 kernel launch**：CPU 把指令表（一个 3D 张量）传给 GPU，GPU 上的所有 SM 并行执行各自的指令队列，全程不需要返回 CPU。

2. **细粒度 SM 分配**：每个 SM 有自己的指令队列，不同 SM 可以在同一时间执行不同操作（有的做 QKV projection，有的做 attention，有的做 MLP）。调度器在 Python 端预先计算好最优分配方案。

3. **多后端验证**：同一套指令 DAG 可以用三种方式执行：
   - **PyTorch 后端**：直接 `model.forward()`，作为正确性基准
   - **PyVM 后端**：Python 解释器逐条执行指令，便于调试
   - **CUDA 后端**：编译好的 `.so` megakernel，生产级性能

## 类比：GPU 上的虚拟机

Megakernels 本质上是在 GPU 上实现了一个**字节码虚拟机（VM）**：

| 概念 | 传统 VM | Megakernels |
|------|---------|-------------|
| 字节码 | `.class` 文件 | `instructions` 张量 `[num_sms, max_queue, 32]` |
| 虚拟机 | JVM | `mk_internal()` CUDA kernel |
| 指令集 | JVM bytecode | 7 个 opcode（QKV、Attention、MLP 等） |
| 执行单元 | 线程 | SM（流处理器） |
| 共享状态 | 堆内存 | `Globals`（权重 + 激活值 + KV 缓存） |

## 目标硬件

项目针对 NVIDIA Hopper（H100，sm_90a）和 Blackwell（B200，sm_100a）架构优化，同时支持 A100（sm_80）和 RTX 4090（sm_89）。

H100 有 **132 个 SM**，这意味着指令调度器需要把模型的计算分成约 132 份并行执行。

## 项目结构定位

```
问题层次                     对应代码
─────────────────────────────────────────────
LLM 推理逻辑              megakernels/llama.py
指令抽象                  megakernels/instructions.py
                          megakernels/demos/latency/instructions.py
DAG 调度                  megakernels/scheduler.py
                          megakernels/demos/latency/scheduler.py
执行后端                  megakernels/generators.py
                          megakernels/demos/latency/python_vm.py
CUDA 实现                 include/megakernel.cuh
                          demos/low-latency-llama/
```

## 小结

- **问题**：Decode 阶段 kernel launch 多、SM 利用率低
- **方案**：把 forward pass 编译成指令 DAG，用一个 megakernel 在 GPU 内部执行
- **好处**：消除 CPU 调度开销，细粒度 SM 并行，统一多种执行后端

下一步：[02-architecture.md](02-architecture.md) — 整体技术架构
