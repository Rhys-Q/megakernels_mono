# Tile 是什么概念？它和 CTA 有什么区别？

## 先理解 GPU 的并行结构

在深入 TileLang 的 tile 概念之前，我们先复习一下 NVIDIA GPU 的并行层次结构：

```
整个 GPU Kernel 执行
├── Grid（网格）：由多个 Block 组成
│   ├── Block（线程块）= CTA（Cooperative Thread Array）
│   │   ├── Warp（线程束）：32 个线程，硬件调度单位
│   │   │   ├── Thread（线程）
│   │   │   └── Thread ...（共 32 个）
│   │   ├── Warp ...
│   │   └── Warp ...（一个 Block 通常有 4~8 个 Warp）
│   ├── Block ...
│   └── Block ...（Grid 可以有成千上万个 Block）
```

关键记忆点：
- **CTA = Block = Thread Block**：这三个词指同一个东西
- 同一个 CTA 内的线程可以：共享 Shared Memory、用 `__syncthreads()` 同步
- 不同 CTA 之间**不能**直接通信（除非借助全局内存或 Cluster）

---

## TileLang 中 "Tile" 的含义

**Tile（块/瓦片）** 在 TileLang 中是一个**数据块**的概念，而不是执行单元的概念。

> Tile = 一个由并行单元（CTA / Warp / 线程）协作处理的**矩形数据区域**

理解这个区别很关键：
- **CTA** 是执行单元（一组线程）
- **Tile** 是数据块（一片矩形数据）
- **一个 CTA 处理一个（或多个）Tile**

### 直观类比

想象一张大地图（全局矩阵），你有很多工人（CTA），把地图切成小格子分配给工人处理：

```
全局矩阵 A（1024 × 1024）
┌────┬────┬────┬────┐
│ T  │ T  │ T  │ T  │  ← 每个格子 = 一个 Tile（128×128）
├────┼────┼────┼────┤     每个 Tile 由一个 CTA 负责处理
│ T  │ T  │ T  │ T  │
├────┼────┼────┼────┤
│ T  │ T  │ T  │ T  │
├────┼────┼────┼────┤
│ T  │ T  │ T  │ T  │
└────┴────┴────┴────┘
         ↕
   8×8 = 64 个 Tile
   64 个 CTA 并行处理
```

---

## TileLang 中的多层 Tile

TileLang 支持多个层次的 tile，每层对应 GPU 硬件的不同并行级别：

### 第 1 层：Block Tile（CTA 级别）

这是最核心的层次，一个 CTA 处理一个 Block Tile：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    #            ↑                       ↑
    #   Grid 的 X 维度大小         Grid 的 Y 维度大小
    #   = N/block_N 个 CTA         = M/block_M 个 CTA
    
    # bx = CTA 在 X 方向的索引（对应输出矩阵的列块）
    # by = CTA 在 Y 方向的索引（对应输出矩阵的行块）
    
    # 这个 CTA 负责计算的输出 tile：
    # C[by*block_M : (by+1)*block_M, bx*block_N : (bx+1)*block_N]
```

Block Tile 的大小（`block_M × block_N`）由用户指定，通常选 128×128 这样的值。

### 第 2 层：Shared Memory Tile

CTA 内的 Shared Memory 中存放的数据也是一个 Tile：

```python
# 共享内存 tile：block_M × block_K 的矩阵 A 片段
A_shared = T.alloc_shared((block_M, block_K), dtype)
# 共享内存 tile：block_K × block_N 的矩阵 B 片段
B_shared = T.alloc_shared((block_K, block_N), dtype)
```

这两个 Tile 存放在 Shared Memory 里，CTA 内所有 128 个线程共同访问。

### 第 3 层：Fragment Tile（寄存器级别）

每个线程（或线程组）在寄存器里维护的数据片段：

```python
# 寄存器 fragment：block_M × block_N 的累加结果
# 注意：这个大 fragment 在物理上被 128 个线程分摊存储
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
```

`alloc_fragment` 看起来分配了一个 128×128 的矩阵，但实际上这是被 128 个线程**分片持有**的，每个线程只持有其中一小部分（大约 128 个元素）。

### 多层 Tile 全图

```
全局内存（Global Memory）
┌─────────────────────────────────────┐
│      矩阵 A (M × K)                 │
│  ┌──────────┐                       │
│  │ A_block  │ ← Block Tile（block_M × block_K）
│  │ tile     │   由一个 CTA 负责     │
│  └──────────┘                       │
└─────────────────────────────────────┘
        ↓ T.copy()（异步拷贝）
共享内存（Shared Memory，CTA 内共享）
┌──────────────────┐
│   A_shared       │ ← Shared Tile（block_M × block_K）
│   (block_M×block_K) │
└──────────────────┘
        ↓ T.gemm()（TensorCore 计算）
寄存器（Registers，每线程私有，协作持有）
┌──────────────────┐
│   C_local        │ ← Fragment Tile（block_M × block_N，分散在 128 个线程的寄存器里）
│   (block_M×block_N) │
└──────────────────┘
        ↓ T.copy()（写回）
全局内存（Global Memory）
┌──────────────────┐
│   矩阵 C (M × N) │
└──────────────────┘
```

---

## Tile 和 CTA 的区别与联系

| 维度 | CTA（Cooperative Thread Array） | Tile |
|------|--------------------------------|------|
| **本质** | 执行单元（一组线程） | 数据单元（一片矩形数据） |
| **属于** | 计算模型（硬件视角） | 数据模型（算法视角） |
| **数量** | Grid 中有多少个 CTA | 矩阵可以切成多少个 Tile |
| **同步** | CTA 内线程可以同步 | Tile 是数据划分，无同步概念 |
| **关系** | 一个 CTA 通常处理一个 Block Tile | 一个 Tile 通常由一个 CTA 处理 |

关键区分：TileLang 中 `T.Kernel(...)` 定义的是 **Grid 的大小**（有多少个 CTA），而 `block_M × block_N` 定义的是每个 CTA 处理的 **Tile 的大小**。

### 相似之处

- CTA 和 Block Tile 通常是**一对一**的关系（一个 CTA 处理一个 Block Tile）
- CTA 的边界决定了 Tile 的边界（`bx * block_N` 就是当前 CTA 对应的 Tile 的起始列）
- 从这个角度看，"一个 CTA 就是一个 Block Tile 的执行者"——所以有时候你看到别人混用这两个概念，但严格来说它们是不同层面的概念

### 重要区别

1. **CTA 是 CUDA 的概念，Tile 是 TileLang 的抽象**
   - CUDA 里没有 "tile" 这个关键字，这是 TileLang 引入的编程抽象
   
2. **Tile 可以存在于不同的内存层次**
   - Block Tile → 全局内存中的数据区域
   - Shared Tile → 共享内存中的缓存
   - Fragment Tile → 寄存器中的累加器
   - 它们是同一数据在不同内存层次的**投影**，而 CTA 只是执行单元

3. **Tile 的大小是算法参数，CTA 的形状是硬件约束**
   - `block_M = 128` 是算法选择（可以调）
   - `threads = 128` 是 CTA 的线程数（同样可以调，但受硬件限制）

---

## 代码中的 Tile 概念示例

```python
# M=1024, N=1024, K=1024
# block_M=128, block_N=128, block_K=32

with T.Kernel(T.ceildiv(N, block_N),    # Grid.x = 1024/128 = 8
              T.ceildiv(M, block_M),    # Grid.y = 1024/128 = 8
              threads=128) as (bx, by): # 每个 CTA 有 128 个线程
    # 此时：
    # - Grid 共有 8×8 = 64 个 CTA
    # - 每个 CTA 负责 128×128 的输出 Tile
    # - 总共 64 个输出 Tile，覆盖整个 1024×1024 矩阵
    
    A_shared = T.alloc_shared((block_M, block_K), dtype)  # 128×32 的共享内存 Tile
    B_shared = T.alloc_shared((block_K, block_N), dtype)  # 32×128 的共享内存 Tile
    C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)  # 128×128 的寄存器 Tile

    T.clear(C_local)
    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):  # 迭代 K/32 = 32 次
        # 每次迭代：
        # - 从 A 中取一个 128×32 的 Block Tile，放入 A_shared
        # - 从 B 中取一个 32×128 的 Block Tile，放入 B_shared
        # - 用这两个 Tile 更新 C_local（128×128 的累加结果）
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)

    T.copy(C_local, C[by * block_M, bx * block_N])
```

---

## 总结

- **Tile** 是 TileLang 中数据划分的基本单位，是一片矩形数据区域
- **CTA** 是 CUDA 中线程组织的基本单位，是一组可以协作的线程
- 在 TileLang 的编程模型中，**一个 CTA 负责处理一个 Block Tile**
- Tile 可以出现在不同的内存层次（全局内存、共享内存、寄存器），而 CTA 是执行单元
- 它们是同一个 kernel 从不同角度（数据 vs 执行）的描述，既相关又不同
