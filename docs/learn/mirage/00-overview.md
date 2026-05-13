# Mirage Persistent Kernel (MPK) 概述

## 什么是 Mirage？

Mirage Persistent Kernel（简称 MPK）是一个**编译器和运行时系统**，专门用于加速大语言模型（LLM）的推理。

用一句话说：**MPK 把 LLM 推理所需要的所有计算，打包成一个超级大的 GPU 内核（MegaKernel），然后持续运行，极大地降低推理延迟。**

---

## 为什么需要 MPK？

### 传统方式的问题

在通常的 LLM 推理中，每生成一个 token，需要依次调用很多 GPU 核函数（kernel）：

```
每个 decode 步骤：
  ┌────────────────────────────────────────────────────────────────┐
  │ Embedding → RMSNorm → Linear(QKV) → Attention → Linear(O)     │
  │ → AllReduce → RMSNorm → Linear(Gate+Up) → SiLU → Linear(Down)  │
  │ → AllReduce → (重复 N 层) → RMSNorm → LM Head → Argmax         │
  └────────────────────────────────────────────────────────────────┘
```

每调用一次 GPU kernel，都需要：
1. CPU 通过 CUDA 驱动，向 GPU 发送命令（这有延迟！）
2. GPU 从全局内存读取数据，计算，再写回去
3. 等待 GPU 完成，继续下一个

对于 LLM 来说，每次 decode 只处理 **1~几个 token**，计算量极小，但 kernel 启动的开销（overhead）却不变。这就导致 GPU 大量时间都在等待 CPU 指令，而不是真正在做计算。

### MPK 的解决方案

MPK 的核心思想：**让 GPU 自己循环，不要等 CPU**。

MPK 把所有计算编译成一个单一的持久化 GPU 内核（Persistent Kernel），这个内核**一直运行在 GPU 上**，GPU 内部通过一个"任务调度器"来决定下一步做什么。

```
传统方式：
  CPU → [启动 kernel1] → [等待] → [启动 kernel2] → [等待] → ...

MPK 方式：
  CPU → [启动 MegaKernel（一次性）] → GPU 自己循环执行所有任务
```

这样消除了反复的 CPU-GPU 通信开销，实现了 **1.2× 到 6.7× 的推理加速**。

---

## MPK 能做什么？

| 功能 | 说明 |
|------|------|
| 单 GPU 推理 | 自动融合所有层 |
| 多 GPU 推理 | 通过 NVSHMEM 实现高效通信 |
| 分页 KV-Cache | 支持 Paged Attention |
| FP8 量化 | Blackwell 架构 FP8 矩阵乘 |
| MoE 模型 | Mixture of Experts 路由与计算 |
| 投机解码 | Speculative Decoding（lookahead/promptlookup） |
| 多轮对话 | online/offline/multi-turn 模式 |

---

## MPK 支持哪些硬件？

| GPU 架构 | CUDA 计算能力 | 说明 |
|----------|--------------|------|
| Ampere | SM80 (A100) | 基础支持 |
| Hopper | SM90 (H100/H200) | TMA 优化，最佳性能 |
| Blackwell | SM100 (B200) | FP8/MLA/MTP 等最新特性 |

---

## 一个直观的类比

把 LLM 推理想象成一条**流水线生产线**：

- **传统方式**：每道工序做完，工人跑去告诉老板，老板再通知下一个工人开始干活。每次沟通都有等待时间。

- **MPK 方式**：老板只需在开始时说"开工"，工人们按照预先制定好的流程表，自行协调，一直干活直到完成。中间没有沟通开销。

MPK 就是那张"预先制定好的流程表"——它在编译阶段分析好所有计算步骤、依赖关系，运行时 GPU 按图索骥，高效执行。

---

## 项目来源

MPK 是 CMU（卡内基梅隆大学）Zhihao Jia 团队开发的研究项目，发表于顶级学术会议：

- **Mirage: A Multi-Level Superoptimizer for Tensor Programs** — OSDI 2025
- **Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs** — arXiv 2025

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [01-architecture.md](./01-architecture.md) | 整体技术架构 |
| [02-components.md](./02-components.md) | 主要组件详解 |
| [03-implementation.md](./03-implementation.md) | 关键实现细节 |
| [04-workflow.md](./04-workflow.md) | 系统运行流程 |
| [05-demo-walkthrough.md](./05-demo-walkthrough.md) | Qwen3 Demo 全流程串讲 |
