# TileLang 编译流水线：Python 代码如何变成 GPU 指令

## 全局视图

TileLang 的编译过程可以分为 6 个主要阶段：

```
【用户写的 Python 代码】
         ↓  阶段 1: 前端解析
【TVM TIR IRModule（中间表示）】
         ↓  阶段 2: 合法化
【合法化后的 IRModule】
         ↓  阶段 3: 关键优化（Layout Inference、软件流水线等）
【优化后的 IRModule】
         ↓  阶段 4: 目标优化（设备相关）
【分离的 Host + Device IRModule】
         ↓  阶段 5: 代码生成（CUDA/HIP/Metal/...）
【PTX / GCN 汇编 / Metal Shader】
         ↓  阶段 6: 后端编译（nvcc / hipcc）
【可执行 CUBIN / HSACO / ...】
         ↓  JIT 加载与执行
【GPU 运行时执行】
```

下面详细解释每个阶段。

---

## 阶段 1：前端解析——Python 代码 → TIR IRModule

### 入口：`@tilelang.jit` 装饰器

当你写：
```python
@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(A, B, C):
        with T.Kernel(...) as (bx, by):
            ...
    return gemm
```

`@tilelang.jit` 是一个**延迟编译**的装饰器。真正的编译发生在**首次调用**时，这时才知道具体的 M、N、K 等参数值。

### `@T.prim_func` 的作用

`@T.prim_func` 是 TVM 的 TIR Script Builder，它把你的 Python 函数体**逐行解析**成 TVM TIR（Tensor Intermediate Representation）语法树。

TIR 是 TVM 的低级 IR，类似于 LLVM IR，专门用于表示循环和数组访问：

```
Python 代码:                    TIR 等价表示:
T.Kernel(8, 8, threads=128)  →  attr[...] "thread_extent"  blockIdx.x = 8
                                 attr[...] "thread_extent"  blockIdx.y = 8
                                 attr[...] "thread_extent"  threadIdx.x = 128

T.alloc_shared((128, 32))    →  allocate A_shared: float16[128 * 32] in shared.dyn

T.copy(A[...], A_shared)     →  // 暂时表示为 Copy 操作节点
T.gemm(A_shared, B_shared, C_local) →  // 暂时表示为 Gemm 操作节点
```

此时 `T.copy` 和 `T.gemm` 还没有被展开成具体的线程操作，只是一个"占位符"。

---

## 阶段 2：合法化（Legalization）

这一阶段进行基础检查和规范化：

1. **目标绑定（Target Binding）**：把 `target="cuda"` 或 `target="hip"` 等信息附加到 IRModule 上
2. **前端合法化（Frontend Legalization）**：把 TileLang 特有的构造转换成 TVM 兼容的形式
3. **负索引规范化**：处理 `-1` 这样的反向索引
4. **并行循环验证**：检查 `T.Parallel` 中有没有数据竞争（race condition）
5. **假设注入（Inject Assumes）**：添加 Z3 SMT 求解器能理解的约束，加速后续的符号推理

---

## 阶段 3：关键优化 Pass

这一阶段是 TileLang 编译器的"心脏"，包含几个核心优化：

### 3.1 布局推断（Layout Inference）—— 最关键的 Pass

这是整个编译器最重要的 Pass。

**问题背景**：GPU TensorCore 对数据的内存布局有严格要求。例如：
- WGMMA（Hopper）要求矩阵以特定的 `swizzle` 模式存储在共享内存中
- 每个线程应该持有哪些寄存器，是有硬件规定的格式的

如果布局不对，轻则性能下降，重则计算结果错误。

**Layout Inference 的作用**：自动分析代码中的 `T.gemm`、`T.copy` 操作，推断出每个 buffer 的最优内存布局，自动在代码中插入布局变换。

```
输入（高层 tile 操作）:
    T.gemm(A_shared[128×32], B_shared[32×128], C_local[128×128])

Layout Inference 分析:
    ↓ A_shared 被 WGMMA 用作 A 操作数
    ↓ 在 H100 上，WGMMA A 操作数需要 swizzled 共享内存布局
    ↓ C_local 是 WGMMA 的累加器
    ↓ 每个线程持有 C_local 中的特定行列（按照 WGMMA 寄存器规范）

输出（推断出的布局信息）:
    A_shared: tl.Fragment(shape=[128,32], layout=ColMajorLayout, swizzle=128B)
    C_local:  tl.Fragment(shape=[128,128], layout=WGMMAFragmentLayout)
```

**为什么这很重要**？手写 CUDA 时，你需要查阅几十页的 PTX 文档来确定正确的 swizzle 模式；TileLang 自动替你完成这件事。

### 3.2 软件流水线（Software Pipeline）

**问题背景**：GPU 的内存访问延迟很高（全局内存到共享内存：约 200-400 个时钟周期）。如果每次都等内存拷贝完成再做计算，GPU 大部分时间都在等待。

**解决方案**：软件流水线——在做第 `k` 轮计算的同时，预取第 `k+1` 轮的数据。

```
没有流水线:                有流水线（3级）:
  轮次 0: 拷贝数据           预取: 拷贝轮次 0 数据
  轮次 0: 计算              预取: 拷贝轮次 1 数据
  轮次 1: 拷贝数据           轮次 0: 拷贝轮次 2 数据，同时计算轮次 0
  轮次 1: 计算              轮次 1: 拷贝轮次 3 数据，同时计算轮次 1
  ...（串行，GPU 大量空闲）   ...（并行，GPU 利用率高）
```

当你写 `T.Pipelined(T.ceildiv(K, block_K), num_stages=3)` 时，TileLang 会自动注入这个流水线。

### 3.3 Warp 专一化（Warp Specialization）

在 Hopper（H100）上，TileLang 可以把 CTA 内的 Warp 分工：
- **Producer Warp**：专门负责用 TMA（Tensor Memory Accelerator）从全局内存拷贝数据
- **Consumer Warp**：专门负责用 WGMMA 进行矩阵计算

这种分工让拷贝和计算真正并发，而不只是流水线上的理论重叠。

### 3.4 Tile 操作降级（Lower Tile Op）

把高层的 `T.copy`、`T.gemm` 等操作展开成 TVM 能理解的低级操作：

```
T.gemm(A_shared, B_shared, C_local)
    ↓ 在 H100 上展开为
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 (...)  # WGMMA 指令
```

---

## 阶段 4：目标相关优化

这一阶段针对具体 GPU 架构做进一步优化：

- **共享内存管理**：合并多个共享内存分配为一块大的分配（减少 bank conflict）
- **屏障融合（Barrier Fusion）**：把多个 `mbarrier` 合并，减少同步开销
- **循环向量化**：把标量循环改成 `float4` 等向量操作
- **循环展开（Loop Unroll）**：把小循环体完全展开，减少分支开销
- **LDG/STG 指令**：把内存访问优化为 128-bit 的 `ldg128`/`stg128` 指令
- **只读参数**：把只读输入标注为 `__restrict__ const __ldg`，使用 L1 纹理缓存
- **Host/Device 分离**：把 kernel 启动代码（在 CPU 上运行）和 kernel 主体（在 GPU 上运行）分开

---

## 阶段 5：代码生成

根据目标平台，生成不同的代码：

| 目标 | 代码生成器 | 产物 |
|------|-----------|------|
| CUDA | `tilelang_cuda` codegen | PTX 汇编 |
| HIP (AMD) | `tilelang_hip` codegen | GCN 汇编 |
| Metal (Apple) | Metal codegen | Metal Shader |
| WebGPU | WebGPU codegen | WGSL |
| CPU | `tilelang_c` codegen | C 代码 |

以 CUDA 为例，生成的 PTX 代码会包含：
```ptx
// WGMMA 指令
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {%f0, %f1, ...},  // C 寄存器（累加器）
    [%smem_ptr],       // A 的共享内存地址
    %B_desc,           // B 的描述符
    1, -1, 1, 1, 1;

// 异步拷贝
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%smem], [%gmem], ...
```

---

## 阶段 6：后端编译

生成的 PTX / C 代码再经过编译器编译成可执行文件：

- **CUDA**：`nvcc -arch=sm_90 -O3 ...` → CUBIN（CUDA 二进制）
- **HIP**：`hipcc -arch=gfx90a ...` → HSACO（HIP 二进制）
- **CPU**：GCC/Clang → 共享库

---

## 编译缓存

TileLang 使用 `@cached` 装饰器缓存编译结果。如果用相同参数再次调用，直接返回缓存的 kernel，无需重新编译：

```python
# 第一次调用：完整编译流程（可能需要几秒）
kernel1 = matmul(1024, 1024, 1024, 128, 128, 32)

# 第二次调用相同参数：直接从缓存返回（毫秒级）
kernel2 = matmul(1024, 1024, 1024, 128, 128, 32)

# 不同参数：重新编译
kernel3 = matmul(2048, 2048, 2048, 128, 128, 32)
```

---

## 查看生成的代码

你可以用 `get_kernel_source()` 查看生成的 CUDA 代码，帮助调试和理解：

```python
kernel = matmul(1024, 1024, 1024, 128, 128, 32)
cuda_code = kernel.get_kernel_source()
print(cuda_code)
```

---

## 编译时间参考

| 操作 | 典型耗时 |
|------|---------|
| 首次编译（CUDA，含 nvcc） | 5-30 秒 |
| 缓存命中 | < 100 毫秒 |
| 调试模式（含 verbose） | 10-60 秒 |

编译时间较长是因为：TileLang 需要做 Layout Inference、Pipeline Planning 等复杂分析，然后调用 nvcc 编译 PTX。这是一次性成本，运行时直接使用缓存。
