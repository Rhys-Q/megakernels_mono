# Mirage MPK 主要组件详解

## 组件总览

MPK 由以下核心组件构成：

```
┌─────────────────────────────────────────────────────────────┐
│  [1] PersistentKernel  ─────  用户接口，协调整个生命周期       │
│        │                                                    │
│        ├──[2] KNGraph  ─────  计算图（层定义的容器）           │
│        │      │                                             │
│        │      └──[3] TBGraph ─  线程块级子图（自定义算子）     │
│        │                                                    │
│        ├──[4] Tensor ──────  张量描述（名字、形状、指针）       │
│        │                                                    │
│        └──[5] 编译器  ──────  task_graph.json + kernel.cu   │
│                                                             │
│  [6] 运行时（GPU 端）                                         │
│        ├──[7] Scheduler ──  事件管理，任务分发                │
│        ├──[8] Worker ────  任务执行（计算 kernel）            │
│        ├──[9] TaskDesc ──  任务描述结构                       │
│        ├──[10] EventDesc ─  事件描述结构                     │
│        └──[11] RuntimeConfig ─  全局运行时配置               │
└─────────────────────────────────────────────────────────────┘
```

---

## [1] PersistentKernel — 主入口类

**文件**：`python/mirage/mpk/persistent_kernel.py`

`PersistentKernel` 是用户直接使用的主类，它封装了整个 MPK 的生命周期：

### 构造参数

```python
mpk = mi.PersistentKernel(
    mode="offline",           # 运行模式
    world_size=1,             # GPU 总数
    mpi_rank=0,               # 当前 GPU 编号
    num_workers=96,           # Worker SM 数量
    num_local_schedulers=4,   # 本地 Scheduler SM 数量
    num_remote_schedulers=0,  # 远程 Scheduler SM 数量（多机）
    max_seq_length=512,       # 最大序列长度
    max_num_batched_requests=8,  # 最大批请求数
    max_num_batched_tokens=8,    # 最大批 token 数
    max_num_pages=1024,       # KV Cache 最大页数
    page_size=64,             # 每页大小（token 数）
    eos_token_id=...,         # 结束符 token ID
    meta_tensors={...},       # 运行时元数据张量
    profiler_tensor=None,     # 可选：性能分析张量
)
```

**运行模式说明**：

| 模式 | 说明 |
|------|------|
| `offline` | 离线批处理：填满请求队列后统一处理 |
| `online` | 在线服务：动态添加请求，实时处理 |
| `online_notoken` | 在线服务，返回 hidden states（用于 vLLM 集成） |
| `onepass` | 只执行一次迭代（测试用） |
| `online_multi_turn` | 多轮对话模式 |

### 关键方法

**张量管理**：
```python
# 绑定已有的 PyTorch 张量（模型权重）
x = mpk.attach_input(torch_tensor=weight, name="embed_tokens")

# 分配新的中间张量
y = mpk.new_tensor(
    dims=(batch_size, hidden_size),
    dtype=mi.bfloat16,
    name="embed_out",
    io_category="cuda_tensor",  # 或 "nvshmem_tensor"
)
```

**层定义（Layer API）**：
```python
mpk.embed_layer(input, weight, output, grid_dim, block_dim)
mpk.rmsnorm_layer(input, weight, output, grid_dim, block_dim)
mpk.linear_layer(input, weight, output, grid_dim, block_dim)
mpk.rmsnorm_linear_layer(input, weight_norm, weight_linear, output, grid_dim, block_dim)
mpk.paged_attention_layer(input, k_cache, v_cache, ..., output, grid_dim, block_dim)
mpk.silu_mul_layer(input, output, grid_dim, block_dim)
mpk.linear_with_residual_layer(input, weight, residual, output, grid_dim, block_dim)
mpk.allreduce_layer(input, buffer, output, grid_dim, block_dim)
mpk.argmax_partial_layer(input, output, grid_dim, block_dim)
mpk.argmax_reduce_layer(input, output, grid_dim, block_dim)
```

**编译与执行**：
```python
mpk.compile(output_dir="./kernels")  # 生成并编译 CUDA 代码
mpk()                                 # 执行 MegaKernel（阻塞直到完成）
```

---

## [2] KNGraph — 计算图

**文件**：`python/mirage/kernel.py`

`KNGraph`（Kernel-level Node Graph）是计算图的核心数据结构，对应 C++ 的 `CyKNGraph`。

它记录了：
- 所有算子（operator）节点
- 算子之间的数据流依赖
- 输入/输出张量

每个 `mpk.rmsnorm_layer(...)` 的调用，实际上是在 `KNGraph` 中添加一个 `RMSNorm` 算子节点。

**KNGraph 内部的算子类型**：

```
KNOperator（基类）
├── KNInputOp       — 输入张量（attach_input 创建）
├── KNOutputOp      — 输出张量
├── KNMatMulOp      — 矩阵乘法
├── KNElementUnaryOp — 逐元素一元运算（ReLU, SiLU, Exp 等）
├── KNElementBinaryOp — 逐元素二元运算（Add, Mul, Div）
├── KNReductionOp   — 归约运算（Sum, Max）
├── KNRMSNormOp     — RMSNorm 归一化
├── KNAllReduceOp   — 多 GPU 全归约
└── KNCustomizedOp  — 自定义算子（内嵌 TBGraph）
```

---

## [3] TBGraph — 线程块级图（高级用法）

**文件**：`python/mirage/threadblock.py`

`TBGraph`（Thread-Block level Graph）用于定义**单个线程块（Thread Block）内部的计算**，是针对高级用户的接口，允许用户自定义融合算子。

例如，用户可以把 RMSNorm + Linear 融合成一个自定义算子：
```python
bg = mi.new_threadblock_graph(
    grid_dim=(96, 1, 1),
    block_dim=(128, 1, 1),
    forloop_range=hidden_size // 64,
    reduction_dimx=hidden_size
)
input = bg.new_input(...)
norm = bg.rms_norm(input)
output = bg.matmul(norm, weight)
```

大多数用户**不需要直接使用** TBGraph，使用 `PersistentKernel` 的高层 API 即可。

---

## [4] Tensor — 张量描述

**结构**：
```python
class Tensor:
    name: str          # 张量名称（生成的 CUDA 代码中使用）
    dims: tuple        # 形状，例如 (8, 4096)
    strides: tuple     # 内存步长（可选，默认 row-major）
    dtype: DataType    # 数据类型：bfloat16, float32, int64 等
    io_category: str   # "cuda_tensor" 或 "nvshmem_tensor"
    base_ptr: int      # GPU 内存地址（运行时填充）
```

**`io_category` 的区别**：
- `cuda_tensor`：普通 CUDA 设备内存，只本机 GPU 可访问
- `nvshmem_tensor`：通过 NVSHMEM 分配的对称内存，所有 GPU 都可直接访问（多 GPU AllReduce 时需要）

---

## [5] 编译器 — 代码生成

**文件**：`python/mirage/mpk/persistent_kernel.py:compile()`

编译过程分两步：

### 步骤 1：生成任务图和 CUDA 代码

```python
results = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
# results["json_file"]   ← task_graph.json 内容
# results["cuda_code"]   ← kernel.cu CUDA 源代码
```

`generate_task_graph` 是 C++ transpiler 的 Python 绑定，它：
1. 遍历 `KNGraph` 中的所有算子
2. 为每个算子生成对应的 `TaskDesc`（任务描述）
3. 建立任务间的事件依赖关系
4. 生成完整的 CUDA 代码（包含 `init_persistent_kernel`、`launch_persistent_kernel` 函数）

### 步骤 2：nvcc 编译

```python
cmd = get_compile_command(mpk, target_cc=90, ...)
subprocess.run(cmd)
# 输出：kernel.cpython-311-x86_64-linux-gnu.so
```

关键编译参数：
- `-DMPK_TARGET_CC=90`：目标 GPU 计算能力
- `-DMODE_OFFLINE`：运行模式
- `-DMPK_MAX_NUM_BATCHED_REQUESTS=16`：最大批请求数（编译期常量，提升性能）
- `-DUSE_NVSHMEM`：启用多 GPU 支持
- `-rdc=true`：relocatable device code（Hopper/Ampere 启用）

---

## [6] 运行时 — GPU 端核心

**文件**：`include/mirage/persistent_kernel/persistent_kernel.cuh`

运行时由三个 GPU kernel 组成：

### init_kernel

在程序开始时执行一次，负责：
- 初始化请求队列
- 设置页面分配器（Page Allocator，用于 Paged Attention）
- 向 Worker 队列推入第一批任务

### scheduler_kernel

Scheduler SM 运行的 kernel，主循环如下：
```
while (true):
  从 sched_queue 取出一个 EventId
  if (EventId == TERMINATE):
    退出
  if (EventId 是 NVSHMEM 事件):
    处理多 GPU 同步
  else:
    找到 EventDesc（包含要触发的任务范围）
    将这些任务 ID 推入 Worker 队列
```

### worker_kernel

Worker SM 运行的 kernel，主循环如下：
```
while (true):
  从 worker_queue 取出一个 TaskId
  if (TaskId == TERMINATE):
    退出
  找到 TaskDesc（任务描述）
  等待 dependent_event 就绪
  调用 _execute_task() 执行计算
  触发 trigger_event
```

---

## [7] Scheduler — 调度器

Scheduler 是任务的"指挥官"，它：
- 持有一个 **事件队列（sched_queue）**
- 监听事件就绪状态（`EventCounter` 达到 `num_triggers`）
- 一旦事件就绪，将关联的任务批量推入各 Worker 的队列

**事件类型**（`EventType`）：

| 事件类型 | 说明 |
|---------|------|
| `EVENT_LAUNCH_TASKS` | 触发一批任务 |
| `EVENT_LAUNCH_MASSIVE_TASKS` | 触发大量任务（优化版） |
| `EVENT_LAUNCH_DEPENDENT_TASKS` | 触发有依赖的任务 |
| `EVENT_END_OF_TASK_GRAPH` | 一次迭代结束 |
| `EVENT_TERMINATION` | 退出信号 |

---

## [8] Worker — 工作者

Worker 是实际执行计算的 SM，它：
- 持有一个 **任务队列（worker_queue）**
- 从队列中取任务，调用 `_execute_task()` 执行
- 执行完毕后，原子地递增对应事件的计数器

`_execute_task()` 根据 `task_type` 分发到具体的 kernel 实现：

```cpp
switch (task_desc->task_type) {
  case TASK_LINEAR_HOPPER:
    execute_linear_hopper(task_desc, runtime_config);
    break;
  case TASK_RMS_NORM_HOPPER:
    execute_rmsnorm_hopper(task_desc, runtime_config);
    break;
  case TASK_PAGED_ATTENTION_HOPPER:
    execute_paged_attention_hopper(task_desc, runtime_config);
    break;
  // ...
}
```

---

## [9] TaskDesc — 任务描述

**文件**：`include/mirage/persistent_kernel/runtime_header.h`

`TaskDesc` 是运行时中任务的核心数据结构：

```cpp
struct TaskDesc {
    TaskType task_type;         // 任务类型（Linear, RMSNorm, Attention 等）
    unsigned variant_id;        // 变体 ID（选择具体实现）
    EventId trigger_event;      // 任务完成后触发的事件
    EventId dependent_event;    // 需要等待的事件（依赖）
    void *input_ptrs[7];        // 输入张量指针（最多 7 个）
    void *output_ptrs[3];       // 输出张量指针（最多 3 个）
    TaskMetadata task_metadata; // 任务附加信息（expert_offset, request_id 等）
};
```

**支持的 TaskType（部分）**：

| TaskType | 说明 |
|---------|------|
| `TASK_EMBEDDING` | Token Embedding |
| `TASK_RMS_NORM` | RMSNorm 归一化 |
| `TASK_LINEAR` | 矩阵乘法（Linear 层） |
| `TASK_PAGED_ATTENTION_HOPPER` | Hopper 上的 Paged Attention |
| `TASK_SILU_MUL` | SiLU 激活 × 门控 |
| `TASK_ARGMAX_PARTIAL` | Argmax 局部归约 |
| `TASK_ALLREDUCE` | 多 GPU AllReduce |
| `TASK_LINEAR_FP8_SM100` | Blackwell FP8 矩阵乘 |
| `TASK_MLA_DECODE_SM100` | MLA 解码（DeepSeek V3） |

---

## [10] EventDesc — 事件描述

```cpp
struct EventDesc {
    EventType event_type;    // 事件类型
    int num_triggers;        // 需要被触发多少次才就绪
    TaskId first_task_id;    // 关联的第一个任务
    TaskId last_task_id;     // 关联的最后一个任务
};
```

事件的设计用于表达**多对多的任务依赖**：
- 一个事件可以由多个任务触发（`num_triggers > 1`）
- 一个事件可以关联多个后继任务（`first_task_id` 到 `last_task_id`）

---

## [11] RuntimeConfig — 运行时全局配置

```cpp
struct RuntimeConfig {
    int num_workers, num_local_schedulers;   // SM 分配
    int num_gpus, my_gpu_id;                 // 多 GPU 信息
    TaskDesc *all_tasks;    // 所有任务的静态描述表
    EventDesc *all_events;  // 所有事件的静态描述表
    TaskId **worker_queues; // 每个 Worker 的任务队列
    EventId **sched_queues; // 每个 Scheduler 的事件队列
    int *step;              // 当前 decode 步骤
    long long *tokens;      // 所有请求的 token 序列
    // ... KV Cache 相关指针
    // ... 页面分配器状态
};
```

`RuntimeConfig` 在 `init_kernel` 中被初始化，然后以值传递的方式传给 `worker_kernel` 和 `scheduler_kernel`，是所有运行时状态的"总账本"。

---

## 组件交互总结

```
用户代码
  ↓ attach_input / new_tensor / rmsnorm_layer(...)
PersistentKernel ─→ KNGraph（算子列表）
  ↓ compile()
generate_task_graph()
  ↓
task_graph.json + kernel.cu
  ↓ nvcc
kernel.so（包含：RuntimeConfig 初始化代码 + 所有 TaskDesc/EventDesc）
  ↓ mpk()
init_kernel（初始化）
  ↓
scheduler_kernel  ←──────────────────────┐
  │ 推入任务                              │
  ↓                               触发事件│
worker_kernel → _execute_task() ─────────┘
  (Linear / RMSNorm / Attention / ...)
  ↓（一个 decode 步骤完成）
循环（下一个 decode 步骤）
  ↓（收到终止信号）
退出
```
