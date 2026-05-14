# 深度问题解答

## 问题一：Tile 是什么概念？和 CTA 有什么区别？

> 已在 [02-tile-concept.md](./02-tile-concept.md) 中详细解答，这里给出简要总结。

**Tile = 数据块**（一片矩形数据区域），**CTA = 执行单元**（一组可协作的线程）。

在 TileLang 的编程模型中，**一个 CTA 负责处理一个 Block Tile**，两者通常一一对应，但本质不同：
- Tile 是算法抽象（数据视角），描述"处理哪片数据"
- CTA 是硬件抽象（执行视角），描述"由谁来处理"

---

## 问题二：TileLang 写的 kernel 能否拆分为多个 tile 独立调度？

### 2.1 当前实现分析

TileLang kernel 在启动时，通过 `T.Kernel(grid_x, grid_y, threads=...)` 一次性启动所有 CTA（也就是所有 tile）：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # 这等价于:
    # cuLaunchKernel(func, grid=(N/block_N, M/block_M), block=(128,), ...)
    # 一次性启动所有 tile，无法单独调度某一个
```

**现有调度机制**：
- 所有 tile（CTA）**同时提交**给 GPU 的 GigaThread Engine
- GPU 调度器根据 SM 空闲情况，**自动分配** CTA 到 SM 上运行
- 用户无法干预"哪个 tile 先执行"或"某个 tile 在哪个 SM 上执行"

**结论：当前实现中，每个 tile 确实是独立执行的（CTA 之间无依赖），但不能"独立调度"——你无法单独启动某一个 tile，它们是作为一个整体 kernel launch 一起发出的。**

### 2.2 TileLang 提供的更灵活调度方式

TileLang 已经提供了 `T.Persistent` 来实现更灵活的 tile 调度：

```python
# Persistent Kernel 示例
@tilelang.jit
def persistent_gemm(M, N, K, block_M, block_N, block_K, ...):
    @T.prim_func
    def kernel(A, B, C):
        num_tiles = T.ceildiv(M, block_M) * T.ceildiv(N, block_N)
        
        # T.Persistent：启动固定数量的 CTA（"wave"），
        # 每个 CTA 从工作队列中反复取 tile 执行，直到所有 tile 完成
        for tile_id in T.Persistent(num_tiles):
            bx = tile_id % T.ceildiv(N, block_N)
            by = tile_id // T.ceildiv(N, block_N)
            
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel
```

`T.Persistent` 的优势：
- 启动较少的 CTA（只有 SM 数量那么多），每个 CTA 处理多个 tile
- 可以实现**动态负载均衡**：慢的 SM 处理少量 tile，快的 SM 处理多量 tile
- 减少 kernel launch overhead

但这仍然是一次性启动，不是"逐 tile 独立调度"。

### 2.3 如何改造实现真正的逐 tile 独立调度？

**方案 A：每个 tile 单独 kernel launch（最简单，开销最大）**

```python
# 伪代码：对每个 tile 单独发起一次 kernel launch
for by in range(M // block_M):
    for bx in range(N // block_N):
        single_tile_kernel[grid=(1,1), block=(128,)](A, B, C, bx, by)
```

**缺点**：CUDA kernel launch 有约 5-20 微秒开销，1024×1024 矩阵有 64 个 tile，总开销 320-1280 微秒，远超计算时间。

**方案 B：Programmatic Dependent Launch（PDL）**

TileLang 已支持 `T.pdl`（Programmatic Dependent Launch），这是 CUDA 12 引入的功能，允许一个 kernel 在运行时动态触发另一个 kernel，实现 tile 级别的流水线：

```python
# TileLang 的 PDL 支持（tilelang/language/pdl.py）
# 允许 kernel A 在完成某个 tile 后，立即触发 kernel B 处理该 tile 的结果
# 实现跨 kernel 的 tile 级别数据流水线
```

这是目前 TileLang 中最接近"独立 tile 调度"的机制。

**方案 C：CUDA 工作队列（Work Queue）**

结合 `T.Persistent` + 原子操作，在全局内存里维护一个 tile 任务队列：

```python
# 概念：改造后的 Persistent Kernel，支持外部往队列中插入 tile 任务
# 每个 CTA 不断轮询队列，取到 tile_id 就处理，处理完就取下一个
# 外部可以按任意顺序、任意时机插入 tile 任务

# 注意：这需要 GPU 上的同步原语（atomicAdd 等），
# TileLang 提供了 T.atomic_add() 等原子操作支持这种模式
```

**总结**：TileLang 当前不原生支持"逐 tile 独立调度"，但通过 `T.Persistent`（波浪调度）和 `T.pdl`（跨 kernel 流水线）可以实现部分需求。完整的独立调度需要在 TileLang 框架外用 CUDA 工作队列实现。

---

## 问题三：一个 tile 能否抽象为 Load、Compute、Store 三个阶段？

### 3.1 当前代码中已经隐含了这个抽象

观察标准 GEMM kernel 的结构：

```python
with T.Kernel(...) as (bx, by):
    # ====== 准备 ======
    A_shared = T.alloc_shared(...)
    C_local  = T.alloc_fragment(...)
    T.clear(C_local)  # 初始化累加器

    # ====== 主循环：Load + Compute 交织 ======
    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
        T.copy(A[...], A_shared)  # Load 阶段：从 Global 加载到 SMEM
        T.copy(B[...], B_shared)  # Load 阶段
        T.gemm(A_shared, B_shared, C_local)  # Compute 阶段

    # ====== Store 阶段：将结果写回 ======
    T.copy(C_local, C[...])
```

**已经存在的 Load/Compute/Store 分离！**
- **Load**：`T.copy(Global → Shared)` — 数据从 HBM 加载到 SMEM
- **Compute**：`T.gemm(Shared, Shared → Fragment)` — TensorCore 计算
- **Store**：`T.copy(Fragment → Global)` — 结果写回 HBM

### 3.2 软件流水线把这个抽象变得更显式

当你写 `T.Pipelined(num_stages=3)` 时，编译器内部就是按照 Load/Compute 阶段来建模流水线的：

```
软件流水线内部的阶段划分（num_stages=3）:

Stage 0 (Load)：         cp.async 发起异步加载（Global → SMEM）
Stage 1 (Load Done)：    等待 Stage 0 的加载完成（mbarrier.wait）
Stage 2 (Compute)：      wgmma 执行矩阵乘法（SMEM → Fragment）
                         同时 Stage 0 发起下一轮加载
```

编译器的 `PipelinePlanning` + `InjectSoftwarePipeline` Pass 正是按照这个 Load/Compute 的两阶段模型来分析和变换代码的。

Store 阶段（写回全局内存）通常在整个 K 维度循环结束后进行，不在流水线内。

### 3.3 当前的限制

虽然逻辑上存在 Load/Compute/Store 三阶段，但在 TileLang 的 **API 层面**，这个抽象是**隐式的**，而不是显式的：

- 用户在同一个循环体内写 `T.copy` 和 `T.gemm`，没有明确的"阶段"标注
- 编译器通过分析操作的内存来源/目的地来推断哪些是 Load，哪些是 Compute

如果需要更精细的控制（比如让 Load 和 Compute 在不同的 warp 上执行），需要使用 **Warp Specialization**：

```python
# 通过 T.ws（warp specialization）显式分离 Producer 和 Consumer warp
# 这在 tilelang/language/warpgroup.py 中实现
with T.Kernel(..., threads=256) as (bx, by):
    # 前 128 个线程（Producer warp group）负责 Load
    # 后 128 个线程（Consumer warp group）负责 Compute
    # 通过 Hopper TMA 和 mbarrier 同步
```

`ProducerConsumerWarpSpecialized` 这个编译 Pass 会自动识别 `T.Pipelined` 中的 TMA copy 和 WGMMA 计算，把它们分配给不同的 warp，实现 Load/Compute 的真正并发（Producer warp 做 Load 的时候，Consumer warp 在做上一轮的 Compute）。

### 3.4 如何改造：更显式的 Load/Compute/Store 抽象

如果想让 TileLang 提供更显式的三阶段接口，可以添加类似这样的语法糖：

```python
# 设想中的 API（当前不存在，这是改造建议）
class TileScheduler:
    def load(self, *srcs):
        """声明 Load 阶段：把指定 tensor 加载到 tile 的共享内存"""
        for src in srcs:
            T.copy(src, self._shared_buf[src])
    
    def compute(self, op, *inputs, output):
        """声明 Compute 阶段：在 tile 内执行指定操作"""
        T.gemm(*[self._shared_buf[x] for x in inputs], output)
    
    def store(self, dst, src):
        """声明 Store 阶段：把 tile 的计算结果写回"""
        T.copy(src, dst)

# 使用方式
tile = TileScheduler(block_M, block_N, block_K)
for ko in tile.pipelined_loop(K):
    tile.load(A_slice, B_slice)     # 明确标记 Load 阶段
    tile.compute("gemm", A_slice, B_slice, output=C_local)  # 明确标记 Compute
tile.store(C, C_local)              # 明确标记 Store 阶段
```

**实现这个改造的关键修改点**：
1. 在 `tilelang/language/` 下添加 `TileScheduler` 类
2. 在 `tilelang/transform/pipeline_planning.py` 中识别带阶段标注的操作
3. 这本质上只是**语法糖**，底层编译逻辑不需要太大修改（因为编译器已经能识别 Load/Compute/Store 的语义了）

**结论**：一个 tile 完全可以（且在实现上已经是）被分解为 Load、Compute、Store 三个阶段。当前 API 是隐式的，可以通过添加语法糖使其显式化，核心编译逻辑无需大幅改动。

---

## 问题四：TileRT 和 TileLang 有什么关系？TileRT 的 kernel 是基于 TileLang 实现的吗？

> **结论先行**：TileRT 的生产环境 kernel **不是**基于 TileLang 实现的。TileLang 只出现在 TileRT 的参考实现（`refs/`）中。以下是基于代码分析的详细说明。

### 4.1 TileRT 是什么？

TileRT（TileRT v0.1.x）是一个**超低延迟 LLM 推理引擎**，专门针对 8× NVIDIA B200 GPU 的 DeepSeek-V3.2 和 GLM-5 模型推理。它的核心卖点是亚毫秒级 per-token 延迟。

### 4.2 TileRT 的 kernel 如何实现的？

**生产 kernel：预编译 C++/CUDA，通过 PyTorch custom ops 调用**

TileRT 的所有生产 kernel 都打包在 `libtilert.so`（预编译的 C++ 共享库）中，通过 PyTorch 的 custom ops 接口调用：

```python
# 以 QKV Rope 操作为例
# 文件：tilert/python/models/deepseek_v3_2/ops/qkv_rope.py

def qkv_rope(...):
    if q_pe.shape[2] == 16:
        torch.ops.tilert.qkv_rope_op(pe_cache, kv_cache, rope_freqs, cur_pos, profile_logs)
    else:
        torch.ops.tilert.qkv_rope_glm5_op(pe_cache, kv_cache, rope_freqs, cur_pos, profile_logs)
```

这里 `torch.ops.tilert.qkv_rope_op` 是 C++ 注册的 op，不涉及 TileLang 的任何调用。

TileRT 的初始化代码（`tilert/python/__init__.py`）明确显示：
```python
_load_library("libtilert.so")  # 加载预编译 C++ 库
```

**TileLang 在 TileRT 中的角色：仅用于参考实现**

TileLang 只出现在 `tilert/python/models/deepseek_v3_2/refs/kernel.py` 中：

```python
# 文件：tilert/python/models/deepseek_v3_2/refs/kernel.py
try:
    import tilelang
    import tilelang.language as T
except ImportError:
    raise ImportError("Cannot import tilelang, please install tilelang.") from None
```

这个文件包含：
- 用 `@tilelang.jit` 编写的 FP8 GEMM 参考实现
- 用 `@triton.jit` 编写的 weight dequant 参考实现

**这些是参考实现（reference implementation）**，用于：
1. 算法正确性验证（对比生产 kernel 的计算结果）
2. 在 `libtilert.so` 不可用时的 fallback（不影响生产部署）

### 4.3 TileRT 的依赖关系

从 `tilert/requirements.txt` 可以确认：
```
torch>=2.6.0
numpy
transformers
```

**TileLang 没有列在依赖中**。TileLang 是可选的，仅用于参考实现。

从 `tilert/pyproject.toml` 确认：
```toml
[project]
dependencies = ["torch"]  # 只依赖 torch！
```

### 4.4 两者的关系总结

| 维度 | TileLang | TileRT |
|------|---------|--------|
| **定位** | GPU kernel 开发 DSL | 超低延迟 LLM 推理引擎 |
| **目标用户** | kernel 开发者 | LLM 推理服务部署者 |
| **实现层** | Python DSL + TVM 编译器 | 预编译 C++ 引擎 |
| **相互依赖** | TileLang 不依赖 TileRT | TileRT **不依赖** TileLang（生产环境） |
| **关系** | — | TileRT 的 refs/ 中有 TileLang 参考实现 |
| **同团队** | 是，tile-ai 团队 | 是，tile-ai 团队 |

**推断**（基于代码结构，非官方说法）：
- TileLang 可能用于**早期开发阶段**快速验证算法正确性
- 一旦算法确认后，用手写 CUDA/C++ 重写为高度优化的生产 kernel，打包进 `libtilert.so`
- refs/ 中的 TileLang 实现保留用于回归测试和算法对比

这是工程上很常见的模式：用高层语言（TileLang）快速原型，然后用低层语言（CUDA C++）做极致优化。
