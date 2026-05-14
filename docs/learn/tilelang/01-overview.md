# TileLang 概述：它是什么，为什么需要它

## 一句话介绍

**TileLang** 是一个用 Python 写 GPU kernel 的领域专用语言（DSL），让你能用接近算法描述的代码写出媲美手写 CUDA 的高性能 kernel。

---

## 为什么需要 TileLang？

### 传统 GPU 编程的困境

写一个高性能的 GEMM（矩阵乘法）kernel，手写 CUDA 需要：

- 手动管理共享内存（Shared Memory）的分配与生命周期
- 手动计算每个线程应该读哪块数据、写哪块结果
- 手动插入 `__syncthreads()` 同步屏障
- 手动做软件流水线（Software Pipeline）来隐藏内存延迟
- 手动适配不同 GPU 架构（V100 / A100 / H100 / Blackwell）的指令差异

这非常繁琐，一个 FlashAttention kernel 动辄几千行 CUDA 代码。

### TileLang 的解法

TileLang 让你**用"块（tile）"为单位来思考和描述计算**，编译器自动把它翻译成高效的 GPU 代码：

```python
# 用 TileLang 写一个 GEMM + ReLU，只需要 20 行
@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def matmul_relu_kernel(A, B, C):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)   # 声明共享内存
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)  # 声明寄存器片段

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):  # 自动流水线！
                T.copy(A[by * block_M, ko * block_K], A_shared)  # 自动异步拷贝
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)               # 自动映射 TensorCore

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)          # ReLU

            T.copy(C_local, C[by * block_M, bx * block_N])
    return matmul_relu_kernel
```

编译器会自动完成：共享内存布局推断、软件流水线注入、异步拷贝、TensorCore 指令映射等。

---

## TileLang 在 GPU 编程栈中的位置

```
用户代码层
┌─────────────────────────────────────────────┐
│  PyTorch / JAX / NumPy                      │  ← 用户通常在这里工作
│  (框架层，自动微分 + 调度)                    │
└─────────────────────────────────────────────┘
        ↓ 调用
┌─────────────────────────────────────────────┐
│  TileLang DSL (Python)                      │  ← TileLang 所在的层
│  (tile 级别的 kernel 描述语言)               │
└─────────────────────────────────────────────┘
        ↓ 编译（基于 TVM）
┌─────────────────────────────────────────────┐
│  CUDA PTX / HIP GCN / Metal Shader          │  ← 编译产物
│  (GPU 汇编或中间代码)                        │
└─────────────────────────────────────────────┘
        ↓ 执行
┌─────────────────────────────────────────────┐
│  GPU 硬件                                   │
│  (NVIDIA H100 / A100 / AMD MI300X / ...)    │
└─────────────────────────────────────────────┘
```

TileLang 比 PyTorch 更底层（你需要关心内存层次和并行结构），但比手写 CUDA 更高层（不需要关心每个线程的精确指令）。

---

## 技术架构总览

TileLang 的整体架构分为三层：

```
┌──────────────────────────────────────────────────────────────────┐
│                     Python 前端（Frontend）                       │
│                                                                  │
│  @tilelang.jit          T.Kernel / T.copy /                     │
│  @T.prim_func           T.gemm / T.Pipelined /                  │
│  (JIT 装饰器)           T.alloc_shared / T.alloc_fragment        │
│                         (语言原语，tile 级操作)                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 构建 TVM TIR IRModule
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     编译器中端（Midend）                           │
│                                                                  │
│  LayoutInference          LowerTileOp                           │
│  （推断寄存器/共享内存布局） （高级 tile 操作 → TVM 低级操作）       │
│                                                                  │
│  InjectSoftwarePipeline   WarpSpecialization                    │
│  （注入软件流水线）         （生产者/消费者 warp 分工）              │
│                                                                  │
│  VectorizeLoop            LoopUnrolling                         │
│  （向量化循环）             （循环展开）                            │
└────────────────────────────┬─────────────────────────────────────┘
                             │ 目标相关优化
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     后端（Backend）                               │
│                                                                  │
│  CUDA (PTX)    HIP (GCN)    Metal    WebGPU    CPU (C/LLVM)     │
│                                                                  │
│  nvcc 编译      hipcc 编译   Metal 编译 ...                       │
└──────────────────────────────────────────────────────────────────┘
```

### 核心依赖：TVM

TileLang 建立在 [Apache TVM](https://tvm.apache.org/) 之上，复用了 TVM 的：
- **TIR（Tensor Intermediate Representation）**：用于表示 kernel 的中间形式
- **Pass 基础设施**：用于做编译优化
- **代码生成框架**：用于生成目标平台代码

TileLang 在 TVM 之上额外添加了面向 tile 编程的语言原语和专用编译 Pass。

---

## 支持的硬件

| 硬件 | 支持状态 | 特殊功能 |
|------|---------|---------|
| NVIDIA H100 / Hopper | 完整支持 | TMA、WGMMA、软件流水线 |
| NVIDIA A100 / Ampere | 完整支持 | async copy、TensorCore |
| NVIDIA RTX 4090 / 3090 | 完整支持 | TensorCore |
| AMD MI300X / MI250 | 完整支持 | MatrixCore、Async Copy |
| Apple Metal | 支持 | Metal Shader |
| WebGPU | 支持 | WGSL |
| CPU (x86/ARM) | 支持 | LLVM/C |
| Huawei Ascend NPU | 实验性 | AscendC |
| NVIDIA Blackwell | 支持 | TCGEN05、2-CTA 协作 |

---

## 主要能力

TileLang 已经能实现以下高性能 kernel：

- **矩阵乘法（GEMM）**：包括 FP16、BF16、FP8、INT4、稀疏等
- **FlashAttention**：前向/反向传播
- **FlashMLA**：DeepSeek 的 Multi-head Latent Attention
- **线性注意力（Linear Attention）**
- **反量化 GEMM（Dequantize GEMM）**
- **卷积（Convolution）**
- **Flash Decoding**：KV cache 解码

---

## 下一步

- [02-tile-concept.md](./02-tile-concept.md) — Tile 是什么概念，和 CTA 有什么区别？
- [03-compilation-pipeline.md](./03-compilation-pipeline.md) — 代码是怎么被编译成 GPU 指令的？
- [04-components.md](./04-components.md) — 主要组件详解
- [05-demo-walkthrough.md](./05-demo-walkthrough.md) — 用一个 GEMM 例子串讲完整流程
- [06-key-questions.md](./06-key-questions.md) — 深度问题解答
