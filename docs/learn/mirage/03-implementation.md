# Mirage MPK 关键实现细节

## 1. Worker/Scheduler 分配模型

### SM 的划分

GPU 的 SM（Streaming Multiprocessor，流多处理器）被静态分配为两类：

```
GPU（以 H100 为例，共 132 个 SM）
┌──────────────────────────────────────────────────┐
│  Scheduler SM × 4 (num_local_schedulers=4)       │
│  ┌────┐  ┌────┐  ┌────┐  ┌────┐                 │
│  │Sch0│  │Sch1│  │Sch2│  │Sch3│                 │
│  └────┘  └────┘  └────┘  └────┘                 │
│                                                  │
│  Worker SM × 96 (num_workers=96)                 │
│  ┌────┐┌────┐┌────┐ ... ┌────┐                  │
│  │Wkr0││Wkr1││Wkr2│     │Wkr95│                 │
│  └────┘└────┘└────┘     └────┘                  │
└──────────────────────────────────────────────────┘
```

**规则**：`num_workers + (num_local_schedulers + num_remote_schedulers) / 4 == GPU 总 SM 数`

Hopper H100：96 + 4/4*4 = 96 + 4 = 132 ✓  
（每 4 个 Scheduler SM 一组）

### Worker 线程数

- Ampere (A100)：每个 Worker SM 128 线程（1 warp group）
- Hopper/Blackwell：每个 Worker SM 256 线程（2 warp groups，支持 WGMMA）

---

## 2. 任务队列机制

### 队列结构

每个 Worker 有一个**独立的任务队列**，避免多 Worker 竞争同一队列：

```
worker_queues[0] → [TaskId_A, TaskId_B, ...]  // Worker 0 的队列
worker_queues[1] → [TaskId_C, TaskId_D, ...]  // Worker 1 的队列
...
worker_queues[95] → [...]                      // Worker 95 的队列
```

Scheduler 决定将任务分配给哪个 Worker 队列（轮询或负载均衡）。

### TaskId 编码

TaskId 是一个 64 位整数，高 32 位是迭代编号，低 32 位是任务位置索引：

```
TaskId (64bit) = [iteration_num (32bit)] | [position_index (32bit)]
```

这样，同一个任务在不同的 decode 迭代中有不同的 TaskId，避免跨迭代的混淆。

### EventId 编码

EventId 也是 64 位，包含多 GPU 标记：

```
EventId (64bit) = [nvshmem_tag (16bit)] | [owner_gpu_id (16bit)] | [event_idx (32bit)]
```

如果 `nvshmem_tag != 0`，说明这是一个跨 GPU 事件，需要通过 NVSHMEM 处理。

---

## 3. 事件计数器原子操作

事件使用**原子计数器**来判断是否就绪：

```cpp
// 当一个 Worker 完成任务时：
atomicAdd(&all_event_counters[event_idx], 1);

// Scheduler 检查事件是否就绪：
if (all_event_counters[event_idx] >= all_event_num_triggers[event_idx]) {
    // 事件就绪，触发后继任务
    enqueue_tasks(event.first_task_id, event.last_task_id);
}
```

`num_triggers` 表示这个事件需要被触发多少次才算"就绪"。这支持多对一的依赖：例如，一个 AllReduce 任务需要等到所有 4 个 GPU 都完成 Linear 层后才能开始，此时 `num_triggers = 4`。

---

## 4. Paged Attention 实现

### 什么是 Paged Attention

传统 KV Cache 为每个请求预分配一段连续内存，容易造成浪费。Paged Attention 将 KV Cache 分成等大小的**页（Page）**，按需分配，类似操作系统的虚拟内存页。

### MPK 中的 Paged Attention 参数

```python
mpk = mi.PersistentKernel(
    max_num_pages=1024,  # KV Cache 最多 1024 页
    page_size=64,        # 每页 64 个 token 的 KV
)
```

### 运行时数据结构

```cpp
// RuntimeConfig 中的 Paged Attention 相关字段
int *paged_kv_indptr_buffer;     // 每个请求的 page 起始索引（CSR 格式）
int *paged_kv_indices_buffer;    // page 索引数组
int *paged_kv_last_page_len_buffer;  // 最后一页的实际长度
int *page_queue;                 // 空闲页队列
int *page_queue_head, *page_queue_tail;  // 队列指针
```

### Split-KV 优化

对于长序列，MPK 支持 `paged_attention_split_kv_layer`：
- 将 KV Cache 分成多个 chunk，并行计算各 chunk 的 attention
- 最后用 `paged_attention_split_kv_merge_layer` 合并结果

这是为了在 decode 阶段利用更多 SM 并行度。

---

## 5. TMA（Tensor Memory Accelerator）优化

Hopper (H100) 和 Blackwell (B200) 支持 TMA，这是一种**异步批量内存传输**机制，可以在不占用计算线程的情况下，预取数据到共享内存（shared memory）。

### TMA 在 MPK 中的使用

启用条件：`-DMPK_ENABLE_TMA`（Hopper 和 Blackwell 默认启用）

TMA 描述符（TMA Descriptor）预先在 CPU 端创建，运行时传入 kernel：

```cpp
struct TensorDesc {
    void *base_ptr;
    void *tma_desc_ptrs[MAX_TMA_DESC_PER_TENSOR];  // 最多几个 TMA 描述符
    // ...
};
```

在 Hopper 的 Linear 实现中（`tasks/hopper/linear_hopper.cuh`），TMA 用于将权重矩阵的 tile 预取到共享内存，与 WGMMA 配合实现高效的矩阵乘法。

---

## 6. WGMMA — Hopper Tensor Core

Hopper 引入了 WGMMA（Warpgroup Matrix Multiply Accumulate）指令，比 Ampere 的 WMMA 效率更高。

MPK 在 Hopper 的 Linear kernel（`tasks/hopper/linear_hopper.cuh`）和 Attention kernel 中大量使用 WGMMA，实现高吞吐量的矩阵乘法：

```cpp
// 简化的 WGMMA 使用示意
wgmma::gemm_ss<...>(smem_A, smem_B, accum);  // 从 shared memory 读取
```

配合 TMA 的数据预取和 WGMMA 的异步执行，实现计算和内存访问的流水线重叠。

---

## 7. NVSHMEM 多 GPU 通信

### NVSHMEM 是什么

NVSHMEM 是 NVIDIA 提供的 GPU 间对等通信库，基于 OpenSHMEM 标准。它允许一个 GPU 直接读写另一个 GPU 的内存，无需经过 CPU。

### MPK 中的使用方式

**AllReduce 实现**：

当 `world_size > 1` 时，每层 Linear 之后需要做 AllReduce（把所有 GPU 的计算结果求和）。

MPK 的 AllReduce 实现（`tasks/hopper/allreduce.cuh`）：
1. 每个 GPU 计算自己的 Local Linear 结果
2. 通过 NVSHMEM 的 `nvshmem_put`，将本 GPU 的结果写到所有其他 GPU 的 `allreduce_buf` 中
3. 等待所有 GPU 写完（NVSHMEM barrier）
4. 本地 GPU 将 `allreduce_buf` 中所有 GPU 的结果加起来

```cpp
// 简化示意
nvshmem_put(dst_on_remote_gpu, src_local, size, remote_pe);
nvshmem_barrier_all();
// 然后做本地 reduce
```

### nvshmem_tensor vs cuda_tensor

```python
# 需要被所有 GPU 访问的张量（AllReduce buffer）
allreduce_buf = mpk.new_tensor(
    dims=(world_size, batch_size, hidden_size),
    dtype=mi.bfloat16,
    name="all_reduce_buf",
    io_category="nvshmem_tensor",  # ← 使用对称内存
)

# 只在本 GPU 使用的张量
hidden = mpk.new_tensor(
    dims=(batch_size, hidden_size),
    dtype=mi.bfloat16,
    name="hidden",
    io_category="cuda_tensor",     # ← 普通 CUDA 内存
)
```

---

## 8. 编译期常量优化

MPK 的很多配置（如最大批大小、页大小）被编译为 C++ 宏定义，而不是运行时变量。这样编译器可以做更多优化：

```cpp
// 编译期常量（通过 -D 传入）
#define MPK_MAX_NUM_BATCHED_REQUESTS 16
#define MPK_MAX_NUM_BATCHED_TOKENS 64
#define MPK_MAX_NUM_PAGES 1024
#define MPK_PAGE_SIZE 64
#define MPK_TARGET_CC 90
```

例如，循环展开（loop unrolling）和 register 分配都可以利用这些常量。

---

## 9. 共享内存（Shared Memory）管理

每个 Worker SM 的共享内存是最宝贵的资源，MPK 精确控制其分配：

```cpp
// runtime_header.h
#if MPK_TARGET_CC >= 90
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    207 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 80
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
```

Worker kernel 以动态共享内存方式申请尽可能多的共享内存，然后在不同任务之间复用（因为不同任务不会同时在同一个 SM 上运行）。

---

## 10. 性能分析（Profiling）

MPK 内置了 **任务级性能分析器**，通过 `--profiling` 启用：

```python
profiler_tensor = torch.zeros(3000 * 128, dtype=torch.uint64, device="cuda")
mpk = mi.PersistentKernel(..., profiler_tensor=profiler_tensor)
```

Profiler 记录每个任务的：
- 开始时间戳（GPU clock）
- 结束时间戳
- 任务类型和 Worker ID

最终可以生成 Perfetto 格式的 trace，在 Chrome 的 `chrome://tracing` 中可视化，直观看到每个 SM 的时间线。

```
Worker 0:  [Linear 0] [Attention 0] [Linear 1] ...
Worker 1:  [Linear 2] [Attention 1] [Linear 3] ...
...
Scheduler: [Schd] [Schd] [Schd] ...
```

---

## 11. FP8 量化（Blackwell 专属）

Blackwell (B200) 支持 FP8（8位浮点）矩阵乘法，MPK 通过专门的 task type 支持：

- `TASK_LINEAR_FP8_SM100`：FP8 矩阵乘
- `TASK_QUANTIZE_FP8_SM100`：量化到 FP8
- `TASK_MOE_W13_FP8_SM100`：MoE 门控网络 FP8

使用方式：
```python
mpk.linear_fp8_layer(
    input=x,
    weight=w_fp8,
    scale=w_scale,
    output=y,
    grid_dim=(96, 1, 1),
    block_dim=(256, 1, 1),
)
```

FP8 可以将显存占用减半，同时利用 Blackwell 的 FP8 Tensor Core，实现更高的 FLOPS/s。

---

## 12. MLA（Multi-head Latent Attention）支持

MLA 是 DeepSeek V3 使用的注意力变体，通过**低秩压缩 KV Cache** 大幅减少显存占用。

MPK 为 Blackwell 专门实现了多种 MLA 变体：
- `TASK_MLA_DECODE_SM100`：TP=1 的 MLA decode
- `TASK_MLA_MTP_DECODE_TP2_SM100`：TP=2 的 MLA + MTP（多 token 预测）
- `TASK_MLA_PREFILL_SM100`：MLA prefill（处理提示词）

这些实现利用了 Blackwell 的 tcgen05 指令和 TMA 特性，达到最优性能。
