# MegaKernel 技术调研报告

> 日期：2026-05-14  
> 分析对象：`Megakernels`（Hazy Research）、`mirage`（MPK）、`tilelang`、`tilert`

---

## 一、背景：为什么需要 MegaKernel

传统 LLM 推理框架对每个算子单独 launch 一个 CUDA kernel：

```
RMSNorm → QKV Proj → RoPE → FlashAttention → O_Proj → Up/Gate → SiLU → Down_Proj → ...
```

每次 kernel launch 的代价链路：CPU 发起 → 驱动队列 → GPU 启动线程 → 从 HBM 重新加载权重 → 计算 → 写回 → SM 释放。

对于 batch=1 的 decode 阶段，每个算子的计算量极小（matvec 而非 matmul），但每次 launch 的固定开销无法摊薄。以 Llama-3B 为例，一个 token 约 100 次 launch，累计约 1-2ms 的纯开销，实际计算只需要不到 1ms。

**MegaKernel 的核心思路：把整个模型前向传播编译成一个 CUDA kernel，只 launch 一次，所有层在 GPU 内部执行，无 CPU 干预。**

---

## 二、Megakernels（Hazy Research）

> `Megakernels/`  
> 定位：**研究原型**，专为 Llama 1B/3B batch=1 decode 设计

### 2.1 整体架构

这是一个**静态指令表驱动的 persistent kernel**。核心思路是：在 Python 侧把整个 Llama 前向传播编译成一张"指令表"，每个 SM 持有自己的指令队列，kernel launch 后 GPU 按表执行，CPU 完全不参与。

```
Python 侧（launch 前）：
  1. 加载模型权重，堆叠为连续 Tensor（stack_params）
  2. 构建 DAG（make_dag）
  3. 分配指令到 SM（assign_to_sms）
  4. 序列化为 Tensor[num_sms, queue_len, 32]

↓ 一次 launch

GPU 侧（persistent kernel）：
  每个 SM 按 instructions[sm_id, :, :] 顺序执行，直到队列尾
```

### 2.2 模型构建

文件：`megakernels/llama.py`

`LlamaForCausalLM.from_pretrained()` 从 HuggingFace 加载权重后，调用 `stack_params()` 把所有层的同类权重堆叠成一个大 Tensor：

```python
stacked_qkv_proj = torch.stack([layer.attn.qkv_proj for layer in layers])
# 形状：[num_layers, 3*num_heads*head_dim, hidden_dim]
```

堆叠的目的是让 CUDA 侧用 `layer_idx` 做 offset 寻址，只需要一个 TMA descriptor，而不是为每层单独传指针。

KV Cache 预分配为 `[num_layers, batch_size, max_len, num_kv_heads, head_dim]`，decode 阶段 in-place 写入。

### 2.3 指令系统

文件：`megakernels/demos/latency/instructions.py`

每个"指令"是一个 Python dataclass，描述一个融合算子：

| Opcode | 类名 | 内容 |
|--------|------|------|
| 1 | `LayerNorm_QKV_MatVecRopeAppend` | RMSNorm + QKV 投影 + RoPE + 写入 KV Cache |
| 2 | `PartialAttention` | FlashAttention 的一个 KV 分区 |
| 3 | `AttentionReduction` | 多分区 attention 结果的 tree-reduce 归并 |
| 4 | `O_ProjResidual` | O 投影 + 残差相加 |
| 5 | `LayerNormDoubleMatVecSiLU` | RMSNorm + Up/Gate 投影 + SiLU |
| 6 | `DownProjResidual` | Down 投影 + 残差相加 |
| 7 | `RMS_LM_Head` | 最后一层 RMSNorm + LM Head 投影 |

每条指令有三个关键方法：
- `opcode()`：唯一标识符，用于 GPU 侧 dispatch
- `serialize()`：序列化为 32 个 int（固定宽度），写入 instructions Tensor
- `cost()`：计算量估算，用于 SM 分配时的负载均衡

### 2.4 调度：Python 侧静态分配

文件：`megakernels/scheduler.py`

调度完全在 Python 侧完成，GPU 上没有任何调度逻辑。

**DAG 构建（`make_dag()`）：**

遍历所有层，为每个 opcode 创建 `DAG_Node`，记录依赖关系（例如 opcode=2 依赖 opcode=1 的同一层结果）。

**SM 分配（`assign_to_sms()`）：**

支持多种策略：

- `rr`（round-robin）：依次轮转分配
- `zz`（zig-zag）：交错分配，增加局部性
- `dag`：基于 `cost()` 和依赖关系的贪心堆调度，负载均衡最好
- `pool`：将 SM 分为内存池和计算池两组

分配结果是每个 SM 的一个指令队列 `sm_queues[sm_idx]`，序列化后写入 `Tensor[num_sms, max_queue_len, 32]`。

**跨 SM 依赖靠全局 barrier 计数器解决**（见后文同步章节）。

### 2.5 编译与加载

文件：`demos/low-latency-llama/Makefile`、`llama.cu`

```makefile
nvcc -O3 -use_fast_math -arch=sm_90a \
     -I ${THUNDERKITTENS_ROOT}/include \
     -I ${MEGAKERNELS_ROOT}/include \
     llama.cu -o mk_llama.so
```

`llama.cu` 是主入口，`#include` 所有算子的 `.cu` 文件，然后通过 pybind11 暴露 Python 接口：

```cpp
PYBIND11_MODULE(mk_llama, m) {
    bind_kernel<mk<config_t, globals_t, op1, op2, op3, op4, op5, op6, op7>>(m, "mk_llama");
}
```

模型维度（`HIDDEN_DIM`、`HEAD_DIM`、`NUM_LAYERS` 等）作为 C++ 模板参数编译时固化，因此每种模型尺寸需要重新编译一个 `.so`。

Python 侧使用：

```python
import mk_llama
mk_llama(barriers, instructions, timings, *weights, activations, kv_cache)
```

### 2.6 CUDA Kernel 内部结构

文件：`include/megakernel.cuh`、`include/config.cuh`

每个 SM（Thread Block）内有 **20 个 warp，分 5 类角色**：

```
┌────────────────────────────────────────────────────────┐
│  Consumer warps  (warpid 0-15,  共 16 个)              │
│  → 矩阵计算，调用 WGMMA TensorCore                     │
├────────────────────────────────────────────────────────┤
│  Loader   warp   (warpid 16)                           │
│  → 从 Global Memory 用 TMA 加载权重到 SMEM              │
├────────────────────────────────────────────────────────┤
│  Storer   warp   (warpid 17)                           │
│  → 将结果从 SMEM 写回 Global Memory                    │
├────────────────────────────────────────────────────────┤
│  Launcher warp   (warpid 18)                           │
│  → 预发 TMA 异步加载（下一条指令的数据）               │
├────────────────────────────────────────────────────────┤
│  Controller warp (warpid 19)                           │
│  → 取指、分配 SMEM page、驱动其余 warp                 │
└────────────────────────────────────────────────────────┘
640 线程 / SM，所有 SM 结构完全对称
```

**Controller 主循环（`controller.cuh`）：**

```cpp
for (instruction_index = 0; instruction_index < num_iters; instruction_index++) {
    load_instructions(&state[ring], instruction_index, g);  // 从 GM 取指
    dispatch_op<page_allocator>(opcode, g, mks);            // 分配 SMEM page
    dispatch_op<semaphore_constructor>(opcode, g, mks);     // 构造 semaphore
    arrive(instruction_arrived[ring], 1);                   // 通知其余 warp 开始执行
    wait(instruction_finished[ring], ...);                  // 等待执行完成
}
```

**Opcode Dispatch（`util.cuh`）：**

```cpp
// 编译时模板展开的 if-else 链，运行时无跳转开销
template<typename... ops>
struct dispatch_op {
    __device__ static void run(int opcode, ...) {
        if (opcode == op0::opcode) { op0::consumer::run(...); }
        else if (opcode == op1::opcode) { op1::consumer::run(...); }
        // ...
    }
};
```

**SMEM Page 管理：**

SMEM 分为 13 个 16KB 的 page，Controller 在指令粒度动态分配/回收 page。不同算子对 page 的需求不同（例如 QKV 指令需要 3 个 page 分别存 Q、K、V tile，Attention 需要 KV cache 的 page）。Page 回收通过 `page_finished` semaphore 追踪。

### 2.7 Kernel 怎么写

每个算子（opcode）在独立的 `.cu` 文件中，实现 4 类 worker 的 `run()` 方法：

```cpp
// rms_matvec_rope_append.cu（示意）
namespace rms_matvec_rope_append {
    struct loader {
        static __device__ void run(const globals &g, state<config> &mks) {
            // 用 TMA 加载 RMSNorm 权重和 QKV 权重到 SMEM
            tma_load(mks.pages[pid].qkv_weight, g.stacked_qkv_proj[inst.layer_idx]);
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &mks) {
            // 16 个 consumer warp 协作：RMSNorm → WGMMA MatVec → RoPE → KV Cache 写入
            wgmma(q_tile, qkv_weight, hidden_states);
            apply_rope(q_tile, pos_id);
            kv_cache[inst.layer_idx][pos_id] = k_tile;
        }
    };
    struct storer { ... };   // 写回 hidden_states 到 GM
    struct launcher { ... }; // 预发下一条指令的 TMA 加载
}
```

底层使用 **ThunderKittens** 库封装 TMA、WGMMA、mbarrier 等 Hopper 硬件原语。

### 2.8 同步方式

两层同步：

**① SM 内 warp 间：SMEM semaphore（底层 mbarrier）**

```cpp
// Controller 通知其余 warp
arrive(instruction_arrived[ring], 1);
// Loader/Consumer/Storer 等待
wait(instruction_arrived[ring], ...);
```

**② 跨 SM：全局内存原子计数器 + volatile spin-poll**

```cpp
// 写端（某 SM 完成 opcode=1 后更新 barrier）
atomicAdd(&g.Bar[{layer, opcode - 1, 0}], inst.iters);

// 读端（另一个 SM 要执行 opcode=2 前，先等 opcode=1 全部完成）
while (*(volatile int *)&g.Bar[{layer, prev_opcode - 1, idx}] < EXPECTED_COUNT)
    __nanosleep(32);  // 让出 warp slot，避免死等
```

### 2.9 性能数据

| 方案 | Decode (tok/s) | Prefill (tok/s) |
|------|--------------|----------------|
| PyTorch HuggingFace | 108 | 7,578 |
| llama.cpp BF16 | 267 | 11,247 |
| **Megakernels** | **413** | **21,347** |

（来源：Lucebox Hub RESULTS.md，Qwen 3.5-0.8B，RTX 3090）

---

## 三、Mirage MPK（Mirage Persistent Kernel）

> `mirage/`  
> 定位：**自动编译框架 + 动态调度 runtime**，支持主流生产模型

### 3.1 整体架构

Mirage 的本质是一个**编译器**：用户用 Python 描述模型计算图，编译器自动生成一个单一的 persistent megakernel。区别于 Megakernels 的"Python 侧静态调度"，Mirage 将调度逻辑搬到了 **GPU 侧**，由专用的 Scheduler SM 在运行时动态驱动 Worker SM。

```
用户 Python 代码（定义 KNGraph）
      ↓
C++ 三层图优化（Computation Graph → Kernel Graph → Thread Block Graph）
      ↓
代码生成（task 函数 .cu）
      ↓
NVCC 编译 → persistent_kernel.so
      ↓
运行时：
  Worker SM  ←──任务队列──  Scheduler SM
  （执行 task）             （事件驱动分发）
```

### 3.2 三层图层次

Mirage 内部用三层图表示计算：

**① Computation Graph（用户定义层）**

用户通过 Python API `KNGraph` 定义 tensor 操作：

```python
import mirage as mi
graph = mi.new_kernel_graph()
X = graph.new_input(...)
Y = graph.matmul(X, W)
Z = graph.rms_norm(Y, ...)
graph.mark_output(Z)
```

**② Kernel Graph（多 GPU 算子层）**

`KNGraph` 内部是一张由 `KNOperator` 节点组成的 DAG，每个节点对应一个算子（matmul、add、silu、rms_norm、all_reduce 等），节点间的边是 `DTensor`（Device Tensor）。

**③ Thread Block Graph（单 thread block 计算层）**

每个 `KNOperator` 被分解为若干 `TBGraph`（Thread Block Graph），每个 TBGraph 描述一个 thread block 内的 tile-level 计算，操作的是 `STensor`（Shared Memory Tensor）。

### 3.3 超优化搜索

文件：`src/search/search.cc`

Mirage 的一个特色功能是对 thread block 分解方式做**自动搜索**（超优化）：

- 生成候选 TBGraph 结构
- 用 Z3 SMT solver 做形式化验证（确保语义等价）
- 用随机输入做概率验证（快速剔除错误候选）
- 选出最优的 tile 分块策略

这让 Mirage 能自动发现非显然的算子融合方案（例如把 RMSNorm + Linear 融合为一个 thread block 的计算），无需手工写融合代码。

### 3.4 模型搭建（Python API）

文件：`python/mirage/mpk/mpk.py`、`python/mirage/mpk/models/deepseek_v3/builder.py`

用户（或模型 builder）通过 `MPKMetadata` 配置后，调用模型 builder 构建任务图：

```python
from mirage.mpk import MPK

mpk = MPK(MPKMetadata(
    mode="offline",
    model_name="deepseek_v3",
    world_size=1,
    num_workers=96,
    max_seq_length=4096,
    page_size=16,
))
mpk.compile()   # 触发代码生成 + NVCC 编译
mpk.run(tokens) # 执行推理
```

内部的模型 builder（`deepseek_v3/builder.py`）用 `KNGraph` API 构建完整模型的任务图，包含：
- 每层的 `TASK_RMS_NORM_LINEAR`（RMSNorm + 投影）
- `TASK_MLA_DECODE_SM100`（MLA 注意力，Blackwell 专用）
- `TASK_MOE_W13_FP8_SM100`（MoE Up/Gate，FP8）
- `TASK_NVSHMEM_TILE_ALLREDUCE`（跨 GPU All-Reduce）
- 等 100+ 种 task

### 3.5 Task 系统与编译产物

**TaskDesc 格式（`runtime_header.h`）：**

```cpp
struct TaskDesc {
    TaskType  task_type;         // 算子类型（100+ 种枚举）
    unsigned  variant_id;        // 架构变体（Ampere/Hopper/Blackwell）
    void*     input_ptrs[8];     // 输入 tensor 指针（或 TMA descriptor）
    void*     output_ptrs[4];    // 输出 tensor 指针
    EventId   dependent_event;   // 前驱事件 ID（等待前驱完成再执行）
    EventId   trigger_event;     // 完成后触发的事件 ID
};
```

编译器生成的代码为每种 task type 生成一个对应的 CUDA 函数，task graph 序列化为 `all_tasks[]` 数组存在 GPU 内存中。

**编译流程：**

```
Python 模型 builder（构建 KNGraph）
      ↓ C++ transpiler（src/transpiler/）
生成 task 实现代码（.cu）+ task graph JSON
      ↓ NVCC 编译
persistent_kernel.so
```

### 3.6 Runtime：GPU 侧动态调度

文件：`include/mirage/persistent_kernel/persistent_kernel.cuh`

一次 `mpk.run()` 触发两个并发 kernel（双流）：
- `worker_kernel`：占用大部分 SM，循环取任务执行
- `scheduler_kernel`：占用少量 SM（约 10%），循环监听事件、分发任务

**Scheduler 主循环：**

```cpp
// execute_scheduler（persistent_kernel.cuh）
while (true) {
    event_id = poll(sched_queue_last_ready_event_id[sched_id]);
    switch (event_type) {
        case EVENT_LAUNCH_TASKS:
            // round-robin 把 [first_task, last_task) 分发到 worker 队列
            for (pos = first; pos < last; pos++)
                enqueue(worker_queues[pos % num_workers], tasks[pos]);
        case EVENT_END_OF_TASK_GRAPH:
            prepare_next_batch();  // 动态组装下一个 batch 的任务图
    }
}
```

**Worker 主循环：**

```cpp
// execute_worker（persistent_kernel.cuh）
while (true) {
    task = dequeue(worker_queue[worker_id]);
    // 等待前驱 task 完成（acquire 语义读事件计数器）
    while (ld_acquire_gpu_u64(&all_event_counters[task.dependent_event]) < expected)
        __nanosleep(SLEEP_NANOS);
    _execute_task(task);   // 分发到具体 task 实现函数
    // 通知后继 task（release 语义写事件计数器）
    atom_add_release_gpu_u64(&all_event_counters[task.trigger_event], 1);
}
```

`_execute_task()` 根据 `task_type` 做 switch dispatch 到具体实现（`tasks/ampere/`、`tasks/hopper/`、`tasks/blackwell/` 下的 `.cuh` 文件）。

### 3.7 同步方式

**跨 SM（task 间依赖）：PTX acquire/release 原子**

```cpp
// mpk_atoms.cuh
// 完成方（release 语义，确保 task 计算结果对所有 SM 可见）：
atom.add.release.gpu.u64 [event_counter], 1

// 等待方（acquire 语义，确保读到最新的事件计数）：
ld.acquire.gpu.u64 value, [event_counter]
```

相比普通 C++ `atomicAdd`，PTX inline 精确控制 acquire/release 内存序，避免编译器引入多余 fence。

**跨 GPU（Tensor Parallel All-Reduce）：NVSHMEM**

```cpp
// 等待远端 GPU 的事件
nvshmem_signal_wait_until(event_ptr, NVSHMEM_CMP_GE, expected_count);

// 向远端 GPU 写数据并发信号
nvshmem_put_signal(dst, src, size, signal, signal_val, NVSHMEM_SIGNAL_ADD, peer_gpu);
```

事件 ID bit 63 设为 1 表示远端事件（`EVENT_NVSHMEM_TAG`），worker 主循环根据此标志选择本地等待还是 NVSHMEM 等待。

### 3.8 支持的模型与硬件

**模型**：DeepSeek V3、Qwen3、Llama 系列，支持 MoE、MLA、MTP（投机解码）、FP8、Paged KV Cache、动态 batch、多 turn 对话。

**硬件**：
- Ampere (sm_80)：基础 PTX 内核，`tasks/ampere/`
- Hopper (sm_90)：TMA + WGMMA 内核，`tasks/hopper/` + `tasks/cute/hopper/`
- Blackwell (sm_100)：TCGEN05 + FP8，`tasks/blackwell/`
- 多 GPU：NVSHMEM，最多 16 GPU

---

## 四、TileLang

> `tilelang/`  
> 定位：**面向 tile 的 GPU Kernel DSL**，用于编写单个高性能算子

### 4.1 整体定位

TileLang **不是 megakernel 框架**，而是一个用于编写 GPU kernel 的 Python DSL。它解决的是**单个算子的高效实现问题**：开发者用 Python 描述 tile-based 的计算逻辑，TileLang 自动生成针对特定硬件优化的 CUDA/HIP 代码。

TileLang 生成的 kernel 是 megakernel 框架的**基础积木**——例如 TileRT 的底层算子实现就使用了 TileLang 的编译技术。

### 4.2 Tile 编程模型

TileLang 把 GPU 内存层次和计算资源抽象为 tile：

| 原语 | 含义 | 对应硬件 |
|------|------|---------|
| `T.Kernel(bx, by, threads=N)` | 定义 thread block 网格 | CUDA grid/block |
| `T.alloc_shared((M, K), dtype)` | SMEM 中的 tile 缓冲 | Shared Memory |
| `T.alloc_fragment((M, N), dtype)` | 寄存器中的 tile 片段 | Register File |
| `T.alloc_local((M,), dtype)` | 线程私有寄存器 | Register |
| `T.copy(src, dst)` | 数据移动（自动向量化、并行化到所有线程） | ld/st / cp.async / TMA |
| `T.gemm(A_s, B_s, C_r)` | TensorCore GEMM | WMMA / WGMMA / MFMA |
| `T.Pipelined(n, num_stages=k)` | k 级软件流水线 | 自动 overlap load/compute |
| `T.Parallel(M, N)` | 跨线程并行循环 | 自动线程映射 |

### 4.3 编写一个 GEMM Kernel

```python
@tilelang.jit
def matmul(M, N, K, block_M=128, block_N=128, block_K=32, dtype="float16"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)  # Global → SMEM
                T.copy(B[k * block_K, bx * block_N], B_shared)  # Global → SMEM
                T.gemm(A_shared, B_shared, C_local)              # SMEM → Registers（TensorCore）
            T.copy(C_local, C[by * block_M, bx * block_N])      # Registers → Global
    return kernel
```

`T.Pipelined` 自动生成 load/compute overlap 的流水线代码，`T.gemm` 根据目标 GPU 架构自动选择 WMMA（Ampere）、WGMMA（Hopper）或 MFMA（AMD）。

### 4.4 编译流程

文件：`tilelang/engine/lower.py`、`tilelang/jit/kernel.py`

```
Python @tilelang.jit
      ↓ 解析为 TVM PrimFunc（TIR）
PreLowerSemanticCheck   （语义合法性检验）
      ↓
LowerAndLegalize        （TIR pass：tile 语义 → 标准 TIR）
      ↓
OptimizeForTarget       （架构 pass：bank conflict 消除、layout 推断、向量化）
      ↓
Codegen                 （CUDA / HIP / LLVM / WebGPU / NVRTC / CuTeDSL）
      ↓
编译产物：
  - JIT 路径：NVRTC 在线编译，缓存 .cubin
  - AOT 路径：NVCC 编译，产出 .so
```

**TileLang 的内部是 TVM/TIR**，TileLang 的高层 tile 原语最终会被 lower 成标准的 TVM TIR，再走 TVM 的 codegen。TileLang 的价值在于它提供了更符合 GPU 编程直觉的 tile 抽象，屏蔽了 TIR 的复杂性。

### 4.5 支持的硬件原语

**NVIDIA：**
- TensorCore：WMMA（Ampere）、WGMMA（Hopper）、TCGEN05（Blackwell）
- TMA：异步加载（`T.tma_copy()`）
- Async Copy：`cp.async`（`T.async_copy()`）

**AMD：**
- MFMA（Matrix Fused Multiply-Add）

**其他：**
- Metal（Apple GPU）、WebGPU（浏览器）、CPU（LLVM + OpenMP）

### 4.6 典型使用场景

TileLang 的 `examples/` 目录包含大量高性能 kernel 实现：

- **FlashAttention**（MHA/GQA/MLA）：80 行 Python，H100 上与手写 CUDA 性能相当
- **FP8/INT4 GEMM**：利用 TCGEN05 实现 Blackwell FP8 矩阵乘
- **DeepSeek MLA Decoding**：针对 DeepSeek 的 Multi-head Latent Attention 解码
- **2:4 Sparse GEMM**：稀疏张量核心加速
- **Block-Sparse Attention**：Native Sparse Attention（NSA）

---

## 五、TileRT

> `tilert/`  
> 定位：**生产级低延迟 LLM 推理 runtime**，专为 8×B200 设计

### 5.1 整体定位

TileRT 是一个**针对超低延迟 batch=1 推理的生产级系统**，不是通用框架，也不是编译器。它的目标是把 8 张 B200 上推理 DeepSeek-V3.2 / GLM-5 的 decode 延迟压到极致。

核心创新是**算子级别的 tile-task 分解 + 跨 GPU 动态 overlap**：把每个 LLM 算子分解为细粒度的 tile-level task，runtime 在 8 GPU 间动态调度，把 computation、I/O、cross-GPU communication 最大程度地重叠起来。

### 5.2 模型搭建

**权重准备：**

TileRT 使用预转换的权重格式，不直接用 HuggingFace 原始格式：

```bash
python -m tilert.models.preprocess.weight_converter \
  --model_type deepseek-v32 \
  --model_dir /path/to/DeepSeek-V3.2 \
  --save_dir /path/to/DeepSeek-V3.2-TileRT
```

转换后权重按 device 分片：每个 `.safetensors` 文件中的权重 key 以 `_dev_0` 到 `_dev_7` 结尾，每张 GPU 只加载自己的分片。

**模型初始化（`modules/end2end.py`）：**

```python
# 8 个 GPU 并行加载权重（每 GPU 一个线程）
for device_id in range(8):
    thread = Thread(target=__load_weights, args=(device_id, model_path))
    thread.start()

# 每个 GPU 上：加载权重 → 分配中间变量 → 调用 prepare_money
with torch.cuda.device(device_id):
    state_dicts = load_device_weights(model_path, device_id, ...)
    dsa.init_tilert_weights(state_dicts)
    dsa_show_hands_prepare_money(params, intermediates, caches, profile_logs, ...)
```

`dsa_show_hands_prepare_money` 是 `libtilert.so` 暴露的 C++ 函数，通过 `torch.ops.tilert` 注册，内部完成 CUDA Graph 的捕获和 tile-task 图的初始化。

### 5.3 算子分解：50+ 个细粒度 Op

文件：`python/models/deepseek_v3_2/ops/`

每个 LLM 层被分解为多个独立的 tile-level op（Python 中的 `TileRTModule` 子类）：

| Op 文件 | 功能 |
|---------|------|
| `rmsnorm_projx_wqkvia.py` | RMSNorm + 压缩 KV/Q 投影（MLA 特有） |
| `qkv_rope.py` | Q/K RoPE 旋转位置编码 |
| `flash_sparse_mla.py` | FlashAttention + 稀疏 MLA |
| `expert_select.py` | MoE 专家路由（top-k 选择） |
| `expert_sel_up_gate_silu.py` | 所选专家的 Up/Gate + SiLU |
| `expert_down_allreduce.py` | Down 投影 + All-Reduce（跨 GPU）|
| `rmsnorm_head_proj.py` | 最后一层 RMSNorm + LM Head |
| `top_p.py` | Top-P 采样 |

每个 op 内部知道自己的设备分片方式（哪些权重在哪些 GPU 上）。

### 5.4 中间变量管理

文件：`python/models/deepseek_v3_2/temp_var_indices.py`

TileRT 预分配 **51 个中间变量**（Tensor），用连续内存 + 下标索引管理：

```python
class Idx:
    Q           = 0   # Query 向量
    KV          = 1   # KV 压缩向量（MLA）
    TOKEN_OUT   = 2   # 当前 token 输出
    SAMPLING_CONFIG = 3  # 采样参数（temperature, top_p, top_k）
    NEXT_DRAFT_TOKENS = 4  # MTP 投机预测的下一批 token
    ACCEPTED_TOKENS   = 5  # MTP 接受的 token 数
    # ...共 51 个
```

`generate_params_with_continuous_storage()` 将所有中间变量分配为一块大的连续内存，再切 view，保证 cache locality。

### 5.5 推理执行

文件：`python/models/deepseek_v3_2/generator.py`

**decode 循环（无 MTP）：**

```python
while len(generated_tokens) < max_new_tokens:
    token_tensor = torch.tensor([current_token])
    dsa_show_hands(token_tensor.cpu())   # 触发一步前向
    # 从 intermediates[Idx.TOKEN_OUT] 读取下一个 token
    next_token = get_predicted_tokens(device_id=0)[0]
    generated_tokens.append(next_token)
    current_token = next_token
```

`dsa_show_hands` 是 `libtilert.so` 的主执行函数，内部按 tile-task 图执行完整的一步前向，包括所有层的计算和 all-reduce。

**decode 循环（MTP 投机解码，`with_mtp=True`）：**

```python
while len(generated_tokens) < max_new_tokens:
    token_tensor = torch.tensor([current_token])
    dsa_show_hands(token_tensor.cpu(), with_mtp=True)
    
    # 主模型给出 token N 的预测
    predicted = get_predicted_tokens()[0]
    # MTP heads 给出 token N+1、N+2、N+3 的预测
    draft = get_next_draft_tokens()
    num_accepted = get_num_accepted()  # 实际接受了几个
    
    generated_tokens.extend(predicted + draft[:num_accepted])
    current_token = (predicted + draft[:num_accepted])[-1]
    # 平均 ~2.77 个 token/call
```

MTP（Multi-Token Prediction）是投机解码的一种形式：主模型预测下一个 token，附加的 MTP heads 并行预测后续 token，只要预测正确就都接受，有效提高 tokens/s。

### 5.6 同步方式

TileRT 的跨 GPU 同步封装在 `libtilert.so` 中，通过 "show hands" 语义抽象对外不暴露细节。从 Python 层能看到的是：

- 8 GPU 的权重加载是 Python 级多线程并行（`threading.Thread`）
- 每次 `dsa_show_hands()` 调用是同步的（返回后 token 已准备好）
- 内部 8 GPU 的通信通过 NVLink + 自定义 all-reduce 协议完成

### 5.7 性能数据

| 系统 | DeepSeek-V3.2 decode (tok/s) | 备注 |
|------|------------------------------|------|
| vLLM v0.16.0rc2 | ~200 | MTP=1 |
| SGLang v0.5.9 | ~350 | MTP=3 |
| **TileRT v0.1.3** | **600** | MTP=3，8×B200 |

（输入 1K tokens，输出 1K tokens，batch=1）

---

## 六、四个 Repo 的横向对比

### 6.1 定位区别

| | Megakernels | Mirage MPK | TileLang | TileRT |
|--|------------|-----------|---------|--------|
| **本质** | 研究原型 megakernel | 自动编译器 + 动态 runtime | Kernel DSL | 生产级推理 runtime |
| **输入** | 手写算子 + Python 调度 | Python 计算图 | Python tile 程序 | 预转换模型权重 |
| **输出** | `mk_llama.so` | `persistent_kernel.so` | `.cubin` / `.so` | `libtilert.so`（闭源）|
| **使用方式** | `import mk_llama; mk_llama(...)` | `mpk.run(tokens)` | `@tilelang.jit` 装饰 | `dsa_show_hands(token)` |

### 6.2 Runtime 类型

| | 类型 | 调度位置 | 动态能力 |
|--|------|---------|---------|
| Megakernels | **静态** | Python 侧（launch 前完全确定）| 不支持动态 batch/shape |
| Mirage MPK | **动态** | GPU 侧 Scheduler SM（运行时事件驱动）| 支持动态 batch、paged KV cache |
| TileLang | N/A（单算子）| 编译期确定 | N/A |
| TileRT | **动态** | C++ runtime（tile-task 级，跨 GPU）| 支持 MTP、动态长度 |

### 6.3 调度对象

| | 调度粒度 | 计算粒度 |
|--|---------|---------|
| Megakernels | 指令（一个融合算子）分配到 SM | SM 内 Consumer warps 的 tile（SMEM page 级）|
| Mirage MPK | TaskDesc（一个 thread block 的计算）分配到 Worker SM | Worker SM 内的 tile（SMEM 级）|
| TileLang | 不涉及（单 kernel 内部由编译器决定）| Thread Block Tile → Fragment（寄存器）|
| TileRT | Tile-level task 动态分配到 8 GPU | 单 GPU 内的 SMEM tile |

### 6.4 同步原语

| | SM 内 warp 间 | 跨 SM | 跨 GPU |
|--|-------------|-------|-------|
| Megakernels | mbarrier（ThunderKittens 封装）| `atomicAdd` + `volatile` poll | 不支持 |
| Mirage MPK | thread block 内 `__syncthreads()` | PTX `atom.add.release` + `ld.acquire` | NVSHMEM 信号量 |
| TileLang | mbarrier（自动生成）| N/A（单 kernel）| N/A |
| TileRT | 封装在 `libtilert.so` 中 | 封装在 `libtilert.so` 中 | NVLink + 自定义 all-reduce |

### 6.5 依赖的芯片能力

| 能力 | Megakernels | Mirage MPK | TileLang | TileRT |
|------|------------|-----------|---------|--------|
| TensorCore（WMMA）| ✓ | ✓ | ✓ | ✓ |
| TMA | ✓（Hopper）| ✓（Hopper+）| ✓（可选）| ✓（B200）|
| WGMMA | ✓（Hopper）| ✓（Hopper+）| ✓ | ✓（B200）|
| TCGEN05/FP8 | ✗ | ✓（Blackwell）| ✓ | ✓（B200）|
| `__nanosleep` | ✓ | ✓ | ✗ | 封装 |
| mbarrier | ✓ | ✓ | ✓ | 封装 |
| NVSHMEM | ✗ | ✓ | ✗ | 封装 |
| NVLink | ✗ | ✓ | ✗ | ✓ |

### 6.6 硬件支持

| | Ampere (A100) | Hopper (H100) | Blackwell (B200) | AMD | 多 GPU |
|--|:---:|:---:|:---:|:---:|:---:|
| Megakernels | ✗ | ✓（主要）| ✗ | ✗ | ✗ |
| Mirage MPK | ✓ | ✓ | ✓ | ✗ | ✓（≤16）|
| TileLang | ✓ | ✓ | ✓ | ✓ | ✗ |
| TileRT | ✗ | ✗ | ✓（8×B200）| ✗ | ✓（固定 8）|

---

## 七、性能收益的共同来源

不管哪个系统，megakernel 的性能收益来自以下几个方面：

**① 消除 kernel launch overhead**

一次 launch 约 5-20μs，Llama-3B 每 token 约 100 次，累计约 1ms。MegaKernel 归零此开销。

**② 权重在缓存中保持热态**

同一个 persistent kernel 内，SM 对权重的重复访问可以命中 L2 cache。传统多 kernel 模式下，每个 kernel 都会驱逐其他 kernel 的权重缓存。

**③ Load/Compute/Store 流水线重叠**

Megakernels 的 5 类 warp 实现了完整的流水线；TileLang 的 `T.Pipelined` 自动生成 overlap 代码。

**④ 跨算子 SMEM 数据复用**

RMSNorm 的输出直接在 SMEM 中传给下一个 Linear，不需要写回 HBM 再读取。

**⑤ 跨设备 overlap（TileRT 特有）**

把 computation、I/O（权重读取）、all-reduce 在 tile 粒度上交错，消除多 GPU 推理中的通信气泡。

---

## 八、参考文件索引

| 系统 | 文件 | 说明 |
|------|------|------|
| **Megakernels** | `include/megakernel.cuh` | persistent kernel 入口，5 类 warp 分工 |
| | `include/controller/controller.cuh` | Controller 取指主循环 |
| | `megakernels/scheduler.py` | Python DAG 构建，SM 分配策略 |
| | `megakernels/demos/latency/instructions.py` | 7 个 opcode 定义 |
| | `demos/low-latency-llama/llama.cuh` | Globals 模板，TMA descriptor |
| | `demos/low-latency-llama/matvec_adds.cu` | 跨 SM 同步（barrier_update）|
| **Mirage MPK** | `include/.../persistent_kernel.cuh` | Worker/Scheduler 主循环，事件驱动调度 |
| | `include/.../runtime_header.h` | TaskDesc, EventDesc 格式定义 |
| | `include/.../mpk_atoms.cuh` | PTX acquire/release 原子原语 |
| | `python/mirage/mpk/mpk.py` | MPKMetadata，Python 配置入口 |
| | `python/mirage/mpk/models/deepseek_v3/builder.py` | DeepSeek V3 模型 builder |
| **TileLang** | `tilelang/language/kernel.py` | `T.Kernel` context manager |
| | `tilelang/language/gemm_op.py` | `T.gemm`，WGMMA/TCGEN05 dispatch |
| | `tilelang/engine/lower.py` | 编译 pipeline 入口 |
| | `tilelang/jit/__init__.py` | `@tilelang.jit` 装饰器 |
| | `examples/flash_attention/` | FlashAttention 系列实现 |
| | `examples/deepseek_mla/` | DeepSeek MLA decoding |
| **TileRT** | `python/models/deepseek_v3_2/modules/end2end.py` | 端到端初始化，prepare_money |
| | `python/models/deepseek_v3_2/generator.py` | 生成循环，MTP 投机解码 |
| | `python/models/deepseek_v3_2/modules/dsa.py` | DSA 层组合 |
| | `python/models/deepseek_v3_2/ops/` | 50+ tile-level op 实现 |
| | `python/models/deepseek_v3_2/temp_var_indices.py` | 51 个中间变量索引 |
