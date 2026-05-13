# DFlash 和 PFlash 简介

本文简要介绍 Lucebox Hub 的另外两个子项目，帮助理解它们与 Megakernel 的关系和整体技术思路。

---

## DFlash：大模型推测解码

### 背景：为什么需要推测解码？

大模型（如 27B 参数）解码速度慢，因为：
- 每次只能生成 1 个 token（自回归）
- 大矩阵乘法 = 慢

**推测解码（Speculative Decoding）** 的思路：
1. 用一个小的 **draft 模型**（快但不准）先猜测接下来的 N 个 token
2. 让大的 **target 模型** 一次性验证这 N 个猜测
3. 接受与 target 模型分布一致的猜测，拒绝不一致的
4. 平均每次 target 模型调用可以接受多个 token → 速度提升

### DFlash 的具体方案

- **Target 模型**：Qwen3.5-27B Q4_K_M（GGUF 格式，~16GB）
- **Draft 模型**：Qwen3.5-27B-DFlash（专门训练的轻量版本，~3.5GB BF16）
- **Draft 算法**：DFlash（2026 年 z-lab 提出）— 基于 block-diffusion 的推测生成
- **Verify 算法**：DDTree（2026 年 Ringel 等提出）— 树形验证，比链式验证更高效

```
           ┌─────────────────┐
           │   Draft Model   │
           │ (Qwen3.5-27B    │
           │  DFlash, BF16)  │
           └────────┬────────┘
                    │ 猜测 N 个 token（树形分支）
                    ▼
           ┌─────────────────┐
           │  Target Model   │
           │ (Qwen3.5-27B    │
           │  Q4_K_M, GGUF)  │
           └────────┬────────┘
                    │ 验证 + 接受/拒绝
                    ▼
              生成 k 个 token（k ≥ 1）
```

### 自研组件

Lucebox 移植并实现了：
1. **C++/CUDA 解码引擎**：基于 ggml（llama.cpp 的底层库）
2. **3 个自定义 CUDA kernel**（用于树形 SSM 状态回滚）：
   - `ggml_ssm_conv_tree`
   - `ggml_gated_delta_net_tree`
   - `ggml_gated_delta_net_tree_persist`
3. **DDTree 超参数调优**：在 RTX 3090 上，budget=22 是最优的树大小

### 性能结果

| 测试集 | AR（自回归） | DFlash+DDTree | 加速比 |
|--------|------------|---------------|--------|
| HumanEval | 37.8 tok/s | 129.5 tok/s | **3.43×** |
| Math500 | 37.7 tok/s | 110.5 tok/s | 2.93× |
| GSM8K | 37.7 tok/s | 96.2 tok/s | 2.55× |

---

## PFlash：长上下文推测预填充

### 背景：Prefill 的时间问题

处理 128K token 的长 prompt，用普通方式需要约 257 秒（llama.cpp）。

**推测预填充** 的思路：
1. 用小 draft 模型快速"浏览"整个 prompt
2. 给每个 token 打重要性分数
3. 让大 target 模型只处理分数高的 token（跳过不重要的）
4. 保持效果基本不变，大幅减少 target 模型的计算量

### PFlash 的实现

- **Draft 模型**：Qwen3-0.6B BF16（GGUF 格式）
- **评分算法**：Cross-Family Speculative Prefill（ICLR 2026）
- **稀疏注意力**：FlashPrefill（2026）—— 用 BSA（Block-Sparse Attention）在 draft 模型上做稀疏注意力前向

### 4 个自定义 CUDA kernel（`flashprefill_kernels.cu`）

```
mean_K  → 计算每个 block 的平均 K 向量
score   → 给每个 token 打重要性分数
select  → 根据 keep_ratio 选出重要 token
sparse_fwd → 稀疏注意力前向（基于 BSA）
```

### Daemon 模式

PFlash 以 daemon 进程运行，通过 stdin 协议接受命令：

```
compress <ids.bin> <keep_x1000> <drafter.gguf>
→ 输出压缩后的 token id 流

generate <...>
→ 接着做推测解码，输出生成的 token 流

park/unpark drafter
→ 将 draft 模型权重换入/换出显存（与 target 共享 24GB）
```

### 性能结果

| 上下文长度 | DFlash TTFT | llama.cpp 基线 | 加速比 | NIAH |
|-----------|------------|----------------|--------|------|
| 64K | 13.5 s | 134.95 s | **10.0×** | ✅ |
| 128K | 24.8 s | ~257 s | **~10.4×** | ✅ |

NIAH = Needle In A Haystack，测试模型是否真的保留了关键信息。

---

## 三个项目的组合使用

DFlash + PFlash 组合使用可以同时加速 prefill 和 decode：

```
长 Prompt（128K）
     ↓ PFlash（稀疏预填充）
  24.8 秒完成 prefill（vs 257 秒）
     ↓ DFlash + DDTree（推测解码）
  ~130 tok/s 解码（vs ~38 tok/s AR）
```

---

## 与 Megakernel 的关系

| 项目 | 目标模型 | 技术方向 | 核心优化 |
|------|---------|---------|---------|
| Megakernel | Qwen3.5-0.8B（小模型）| 完全融合 | 单 kernel dispatch，消除 CPU 往返 |
| DFlash | Qwen3.5-27B（大模型）| 算法加速 | 推测解码，减少 target 模型调用次数 |
| PFlash | Qwen3.6-27B（大模型）| 算法加速 | 稀疏预填充，减少 target 模型的 prefill 计算量 |

三者代表了 LLM 推理优化的不同层次：
- **Megakernel**：内核级优化（如何高效执行一个模型的前向）
- **DFlash/PFlash**：算法级优化（如何减少模型调用次数）

---

## 代码位置速查

```
lucebox_hub/
├── megakernel/
│   ├── kernel.cu          ← Decode kernel（核心）
│   ├── prefill.cu         ← Prefill kernel
│   ├── model.py           ← Python 接口
│   └── setup.py           ← 编译配置
│
├── dflash/
│   ├── CMakeLists.txt     ← C++/CUDA 构建
│   ├── src/
│   │   ├── test_dflash.cpp    ← 主程序入口
│   │   ├── dflash.cu          ← DFlash 推测解码
│   │   └── flashprefill.cu    ← PFlash 预填充
│   └── scripts/
│       ├── run.py         ← 一键生成脚本
│       └── bench_llm.py   ← 基准测试
│
└── pflash/
    └── README.md          ← PFlash 文档（实现在 dflash/）
```
