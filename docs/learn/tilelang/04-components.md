# TileLang 主要组件详解

## 组件全图

```
tilelang/
├── tilelang/              # Python 包（用户接口 + 编译器中端）
│   ├── language/          # DSL 语言原语（T.Kernel, T.copy, T.gemm, ...）
│   ├── jit/               # JIT 编译器（@tilelang.jit）
│   ├── transform/         # 编译 Pass（Layout Inference, Pipeline, ...）
│   ├── ir.py              # IR 节点定义（Python 侧）
│   └── ...
└── src/                   # C++ 后端
    ├── backend/           # 代码生成器（CUDA, HIP, Metal, ...）
    ├── transform/         # C++ 编译 Pass
    ├── ir.cc              # IR 构建（C++ 侧）
    └── tl_templates/      # CUDA/PTX 代码模板
```

---

## 组件 1：语言层（`tilelang/language/`）

这是用户直接接触的 API 层，提供 TileLang 的 DSL 原语。

### 1.1 Kernel 上下文：`T.Kernel`

```python
# 源码：tilelang/language/kernel.py

with T.Kernel(
    *extents,          # Grid 大小（1~3 维）
    threads=128,       # 每个 CTA 的线程数
    cluster_dims=None, # 可选：线程块集群（Hopper+）
) as (bx, by, ...):   # CTA 索引变量
    ...
```

**作用**：定义 kernel 的 Grid/Block 结构，相当于 CUDA 的：
```cuda
// 等价于：
dim3 grid(extents[0], extents[1], extents[2]);
dim3 block(threads, 1, 1);
my_kernel<<<grid, block>>>(...);
```

内部会绑定 `blockIdx.x/y/z` 和 `threadIdx.x/y/z`。

### 1.2 内存分配：`T.alloc_*`

```python
# 共享内存（CTA 内所有线程共享）
A_shared = T.alloc_shared((block_M, block_K), dtype)
# 内部实现：allocate in "shared.dyn" scope

# 寄存器片段（被所有线程分摊持有）
C_local = T.alloc_fragment((block_M, block_N), dtype)
# 内部实现：allocate in "local.fragment" scope

# 本地变量（每个线程私有）
x = T.alloc_local((1,), dtype)
# 内部实现：allocate in "local" scope

# 全局内存（整个 grid 可访问）
buf = T.alloc_global((M, N), dtype)
# 内部实现：allocate in "global" scope

# Tensor Memory（Hopper 的特殊片上存储）
D = T.alloc_tmem((block_M, block_N), dtype)
# 内部实现：allocate in "shared.tmem" scope（Blackwell TCGEN05）
```

不同的 scope 对应 GPU 的不同内存层次：

| scope | 物理位置 | 容量（典型） | 延迟 |
|-------|---------|-------------|------|
| `local` | 寄存器 | ~255 寄存器/线程 | 1 周期 |
| `local.fragment` | 寄存器（分摊） | 同上 | 1 周期 |
| `shared.dyn` | L1/SMEM | 48-228 KB/CTA | ~32 周期 |
| `shared.tmem` | TMEM（Blackwell） | 512 KB/SM | 低延迟 |
| `global` | HBM | 数十 GB | ~200 周期 |

### 1.3 数据移动：`T.copy`

```python
# 从全局内存拷贝到共享内存
T.copy(A[by * block_M, ko * block_K], A_shared)

# 从共享内存/寄存器写回全局内存
T.copy(C_local, C[by * block_M, bx * block_N])

# 异步拷贝（在 T.Pipelined 中会自动变成 cp.async）
T.async_copy(src, dst)

# TMA 拷贝（Hopper，需要 T.use_tma_descriptor）
T.tma_copy(desc, mbarrier, dst)
```

`T.copy` 的实现很智能：
- 在 `T.Pipelined` 循环内，自动优化为 `cp.async`（异步拷贝）
- 在 H100 上且开启 TMA，优化为 `cp.async.bulk.tensor`（TMA 拷贝）
- 否则，生成普通的线程并行拷贝代码

### 1.4 矩阵计算：`T.gemm`

```python
# 基础 GEMM：C += A × B
T.gemm(A_shared, B_shared, C_local)

# 可选参数
T.gemm(A, B, C,
    transpose_A=False,  # 是否转置 A
    transpose_B=False,  # 是否转置 B
    policy=GemmWarpPolicy.FullRow,  # warp 分配策略
)

# 稀疏 GEMM（2:4 结构化稀疏）
T.gemm_sp(A_sparse, E_metadata, B, C)

# Warp Group MMA（显式 WGMMA，Hopper）
T.wgmma_gemm(A, B, C)

# TCGEN05（Blackwell）
T.tcgen05_gemm(A, B, C)
```

`T.gemm` 会根据 GPU 架构自动选择最优指令：
- **Hopper（H100）**：`wgmma.mma_async` 指令
- **Ampere（A100）**：`wmma` 或 `mma.sync` 指令
- **AMD MI300X**：Matrix Core 指令
- **其他**：标准 FP16 乘法

### 1.5 规约：`T.reduce`

```python
# 全规约
T.reduce(C_local, out, T.ReduceType.Sum)

# 行规约（沿 axis=1 方向）
T.reduce(C_local, out, T.ReduceType.Max, dim=1)

# Warp 内规约
result = T.warp_reduce_sum(val)
result = T.warp_reduce_max(val)
```

### 1.6 循环：`T.Pipelined` / `T.Parallel` / `T.Serial`

```python
# 流水线循环（自动注入软件流水线）
for ko in T.Pipelined(num_iterations, num_stages=3):
    T.copy(...)  # 会自动变成异步预取
    T.gemm(...)  # 会自动与下一轮的拷贝重叠

# 并行循环（所有线程并行执行）
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)  # 每个线程处理部分元素

# 串行循环（单线程串行执行，或所有线程执行相同代码）
for i in T.Serial(4):
    ...

# 持久化循环（Wave 级别的调度）
for wave_id in T.Persistent(num_waves):
    ...
```

---

## 组件 2：JIT 编译器（`tilelang/jit/`）

### 2.1 `@tilelang.jit` 装饰器

这是用户的编译入口，使用方式：

```python
# 方式 1：直接装饰（无参数）
@tilelang.jit
def my_func(M, N, K, ...):
    ...

# 方式 2：带参数
@tilelang.jit(
    out_idx=[-1],           # 指定哪个参数是输出
    target="cuda",          # 编译目标
    execution_backend="dlpack",  # 执行后端
    pass_configs={...},     # 编译器 Pass 配置
)
def my_func(M, N, K, ...):
    ...
```

### 2.2 `JITKernel`：编译后的 kernel 对象

`@tilelang.jit` 编译后返回一个 `JITKernel` 对象：

```python
kernel = matmul(1024, 1024, 1024, 128, 128, 32)
# kernel 是一个 JITKernel 对象

# 执行 kernel
c = kernel(a, b)

# 获取 CUDA 源码
cuda_code = kernel.get_kernel_source()

# 获取性能分析器
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
latency = profiler.do_bench()  # 返回毫秒数
```

### 2.3 执行后端（Execution Backend）

TileLang 支持多种执行方式：

| 后端 | 说明 | 适用场景 |
|------|-----|---------|
| `dlpack` | 通过 DLPack 协议传递张量 | 默认，框架无关 |
| `torch` | 原生 PyTorch 集成 | PyTorch 用户 |
| `tvm_ffi` | TVM 的原生 FFI | 低开销 |
| `cython` | Cython 扩展模块 | 高频调用优化 |
| `nvrtc` | NVIDIA 运行时编译 | CuTe 模板编译 |
| `cutedsl` | CUTLASS CuTe DSL 后端 | 生成 CuTe 代码 |

---

## 组件 3：编译 Pass（`tilelang/transform/`）

编译 Pass 是编译器中端的核心，以下是最重要的几个：

### 3.1 `LayoutInference`

- **作用**：推断所有 buffer 的内存布局（layout）
- **输入**：包含 `T.gemm`、`T.copy` 等高层操作的 IRModule
- **输出**：每个 buffer 都标注了 `tl.Fragment` 布局信息的 IRModule
- **关键性**：没有这个 Pass，无法生成正确的 TensorCore 代码

### 3.2 `LowerTileOp`

- **作用**：把高层 tile 操作（`T.copy`、`T.gemm`）降级为 TVM 能理解的操作
- **细节**：根据当前 GPU 架构选择具体指令：
  - `T.copy` → `cp.async` / TMA / 普通 load-store
  - `T.gemm` → `wgmma` / `mma.sync` / wmma

### 3.3 `InjectSoftwarePipeline`

- **作用**：把 `T.Pipelined` 循环变换成实际的流水线代码
- **细节**：在循环前添加 prologue（预热）、在循环中添加异步等待、在循环后添加 epilogue（收尾）

### 3.4 `ProducerConsumerWarpSpecialized`

- **作用**：把一个 CTA 内的 warp 分为 Producer（负责拷贝）和 Consumer（负责计算）两组
- **适用**：Hopper（H100）上的 TMA + WGMMA 流水线
- **效果**：Producer warp 用 TMA 异步拷贝，Consumer warp 同时做计算，真正并发

---

## 组件 4：C++ 后端（`src/backend/`）

### 4.1 CUDA 代码生成器（`src/backend/cuda/`）

`codegen_cuda.h` 负责把 TVM TIR 翻译成 CUDA PTX 代码。关键职责：
- 把 `wgmma.mma_async` 节点生成 PTX 内联汇编
- 把 `cp.async.bulk.tensor` 节点生成 TMA 指令
- 处理共享内存 swizzle 模式（避免 bank conflict）

### 4.2 CuTe DSL 代码生成器（`src/backend/cuda/codegen_cutedsl.h`）

为支持 CUTLASS CuTe DSL 的代码生成，输出 CuTe C++ 代码（而不是 PTX）。

### 4.3 代码模板（`src/tl_templates/`）

这里存放常用的 CUDA 代码模板，在代码生成时填充具体参数：
- GEMM 运算模板
- Reduction 模板
- Softmax 模板

---

## 组件 5：IR 系统（`tilelang/ir.py`，`src/ir.cc`）

TileLang 在 TVM TIR 的基础上定义了额外的 IR 节点：

```python
# Python 侧（tilelang/ir.py）
class Fill:    # T.fill() 操作
class Copy:    # T.copy() 操作
class Gemm:    # T.gemm() 操作
class ReduceOp:# T.reduce() 操作
class ParallelOp: # T.Parallel() 操作

# C++ 侧（src/ir.cc）
# KernelLaunch() - 建立 kernel 执行上下文（grid/block/threads）
# ParallelFor()  - 创建并行循环
# PipelinedFor() - 创建流水线循环
# WarpSpecialize()- 创建 warp 专一化区域
```

这些节点在 Layout Inference 和 LowerTileOp 之前是"高层 IR"，在 LowerTileOp 之后变成 TVM 标准的 TIR 节点（Buffer 访问、Loop 等）。

---

## 组件交互图

```
用户代码
    │
    │ 调用 @tilelang.jit
    ▼
tilelang/jit/__init__.py
    │ compile() → JITKernel
    │
    │ 1. 调用 @T.prim_func 解析器
    ▼
tilelang/language/eager/builder.py
    │ 构建 TVM TIR IRModule
    │
    │ 2. 执行编译 Pass 序列
    ▼
tilelang/transform/ (Python Pass) + src/transform/ (C++ Pass)
    │ LayoutInference → LowerTileOp → InjectSoftwarePipeline → ...
    │
    │ 3. 后端代码生成
    ▼
src/backend/cuda/codegen_cuda.h
    │ TIR → PTX 汇编
    │
    │ 4. 调用 nvcc 编译
    ▼
已编译的 CUBIN 二进制
    │
    │ 5. JIT 加载
    ▼
tilelang/jit/kernel.py: JITKernel.__call__()
    │ 接收 PyTorch 张量，执行 kernel
    ▼
GPU 执行结果
```
