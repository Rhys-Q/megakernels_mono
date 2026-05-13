# Mirage MPK 技术架构

## 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户代码（Python）                        │
│  import mirage as mi                                        │
│  mpk = mi.PersistentKernel(...)                             │
│  mpk.rmsnorm_layer(...)                                     │
│  mpk.linear_layer(...)                                      │
│  mpk.compile()                                              │
│  mpk()                                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │  Python API
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Python API 层                               │
│  python/mirage/mpk/persistent_kernel.py                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PersistentKernel  ←──  计算图定义 (KNGraph)          │   │
│  │  attach_input / new_tensor                           │   │
│  │  layer 定义方法（rmsnorm/linear/attention/...）        │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │  调用 generate_task_graph()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  代码生成层（编译器）                           │
│  include/mirage/transpiler/                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  task_graph.json  ──→  kernel.cu (CUDA 源代码)        │    │
│  │  任务描述 + 事件依赖  ──→  Worker/Scheduler 代码        │    │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                    nvcc 编译                                  │
│                          │                                   │
│                  *.cpython-xx.so                             │
└─────────────────────────┼───────────────────────────────────┘
                          │  dlopen / import
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  运行时层（GPU 端）                             │
│  include/mirage/persistent_kernel/persistent_kernel.cuh     │
│                                                             │
│  ┌────────────────┐    ┌────────────────────────────────┐   │
│  │  Scheduler SM  │    │     Worker SM（×N 个）          │   │
│  │  读取 Event 队列│───→│  执行 Task（计算核函数）         │   │
│  │  分发 Task     │←───│  触发完成 Event                 │   │
│  └────────────────┘    └────────────────────────────────┘   │
│                                                             │
│  任务类型：Linear / RMSNorm / Attention / AllReduce / ...    │
└─────────────────────────────────────────────────────────────┘
```

---

## 三层架构详解

### 第一层：Python API 层

**位置**：`python/mirage/`

这是用户直接接触的层，负责：

1. **接收用户定义的计算图**：用户通过调用 `mpk.rmsnorm_layer()`、`mpk.linear_layer()` 等方法，描述模型的计算结构。
2. **管理张量（Tensor）**：`attach_input()` 绑定 PyTorch 张量，`new_tensor()` 申请中间缓冲区。
3. **触发编译**：调用 `mpk.compile()` 时，将计算图转换为 CUDA 代码并编译。

关键文件：
- `python/mirage/mpk/persistent_kernel.py` — `PersistentKernel` 类
- `python/mirage/kernel.py` — `KNGraph` 计算图类
- `python/mirage/__init__.py` — 模块入口

### 第二层：代码生成层（编译器）

**位置**：`include/mirage/transpiler/`、`src/transpiler/`

这是 MPK 的"大脑"，负责：

1. **分析计算图**：理解各层之间的依赖关系。
2. **生成任务图（Task Graph）**：将计算图转换为任务（Task）和事件（Event）的有向图，输出为 `task_graph.json`。
3. **生成 CUDA 代码**：将任务图实例化为具体的 CUDA kernel 代码（`kernel.cu`）。
4. **调用 nvcc 编译**：将 `.cu` 文件编译为 Python 可加载的 `.so` 共享库。

编译命令示例（简化）：
```bash
nvcc kernel.cu \
  -O3 -arch=sm_90a \
  -DMODE_OFFLINE \
  -DMPK_MAX_NUM_BATCHED_REQUESTS=16 \
  -shared -fPIC \
  -o kernel.cpython-311.so
```

### 第三层：运行时层（GPU 端）

**位置**：`include/mirage/persistent_kernel/persistent_kernel.cuh`

这是运行在 GPU 上的核心代码，包含：

- **`init_kernel`**：初始化内核，设置运行时配置（只运行一次）
- **`worker_kernel`**：Worker SM 运行的函数，循环取任务、执行计算
- **`scheduler_kernel`**：Scheduler SM 运行的函数，循环处理事件、分发任务

---

## 关键设计原则

### 原则一：计算图 → 任务图

用户定义的"层"（如 RMSNorm、Linear）在编译时被转换为"任务"（Task）。每个任务：
- 有明确的输入/输出张量指针
- 有"触发事件"（当前任务完成后触发哪个事件）
- 有"依赖事件"（需要等待哪个事件才能开始）

### 原则二：Worker/Scheduler 分离

GPU 的 SM（流多处理器）被分为两类：
- **Worker SM**：专门执行计算任务（矩阵乘法、注意力等）
- **Scheduler SM**：专门管理任务调度，决定哪个 Worker 执行哪个任务

这种分离让调度开销不干扰计算，最大化 GPU 利用率。

### 原则三：事件驱动

任务间的依赖通过"事件（Event）"来表达：
- 一个任务完成后，触发（Signal）某个事件
- 下一个任务等待（Wait）这个事件
- Scheduler 监听事件，事件就绪时将关联的任务推入 Worker 队列

### 原则四：持久化运行

整个 MegaKernel **不会退出**，它持续循环，每次迭代处理一个 decode 步骤：

```
┌──────────────────────────────────┐
│  初始化（init_kernel）             │
│         │                        │
│         ▼                        │
│  ┌─────────────────────────┐     │
│  │  一次 decode 迭代        │     │
│  │  (全部层的计算)          │ ←───┤ 循环
│  └─────────────────────────┘     │
│         │                        │
│         ▼                        │
│  收到终止信号 → 退出               │
└──────────────────────────────────┘
```

---

## 多 GPU 架构（NVSHMEM）

当 `world_size > 1` 时，MPK 使用 NVSHMEM 实现多 GPU 间的通信：

```
GPU 0                           GPU 1
┌─────────────────────┐        ┌─────────────────────┐
│  MegaKernel (rank=0)│        │  MegaKernel (rank=1)│
│                     │        │                     │
│  Attention →        │        │  Attention →        │
│  Linear →           │        │  Linear →           │
│  AllReduce ─────────┼────────┼─→ AllReduce         │
│      ↑              │        │                     │
│  NVSHMEM 通信       │        │  NVSHMEM 通信        │
└─────────────────────┘        └─────────────────────┘
          ▲                              ▲
          └──────── MPI 协调 ────────────┘
```

每个 GPU 上都运行一份完整的 MegaKernel，通过 NVSHMEM（NVIDIA Symmetric Memory）在 GPU 间直接通信，无需经过 CPU。

---

## 目录结构总览

```
mirage/
├── python/mirage/          # Python API 层
│   ├── __init__.py         # 模块入口，导出 PersistentKernel, MPK 等
│   ├── kernel.py           # KNGraph 计算图
│   ├── threadblock.py      # TBGraph 线程块图
│   └── mpk/
│       ├── persistent_kernel.py  # PersistentKernel 主类
│       ├── mpk.py                # MPK 高层包装
│       └── multigpu.py           # 多 GPU AllReduce 选择
│
├── include/mirage/         # C++ 头文件
│   ├── kernel/             # 算子定义（DTensor, KNOperator 等）
│   ├── threadblock/        # 线程块级算子
│   ├── transpiler/         # 代码生成器接口
│   └── persistent_kernel/  # 运行时核心
│       ├── persistent_kernel.cuh  # 主运行时逻辑
│       ├── runtime_header.h       # 数据结构定义
│       └── tasks/
│           ├── ampere/     # Ampere (A100) 的 kernel 实现
│           ├── hopper/     # Hopper (H100) 的 kernel 实现（TMA 优化）
│           └── blackwell/  # Blackwell (B200) 的 kernel 实现（FP8 等）
│
├── src/                    # C++ 实现
│   ├── kernel/             # 算子实现
│   ├── transpiler/         # 代码生成实现
│   └── search/             # 超优化搜索
│
└── demo/                   # 使用示例
    ├── qwen3/              # Qwen3 模型 Demo
    ├── llama3/             # Llama3 模型 Demo
    └── deepseek_v3/        # DeepSeek V3 Demo
```
