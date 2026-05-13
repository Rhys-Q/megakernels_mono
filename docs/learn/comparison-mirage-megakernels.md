# Mirage MPK vs Megakernels 深度对比分析

> 分析基于 `/root/tw/megakernels_mono/mirage/` 和 `/root/tw/megakernels_mono/Megakernels/`

---

## 一、核心问题与设计出发点

两者都试图解决同一个根本问题：

> **传统 LLM 推理为每个算子单独 launch 一个 CUDA kernel，导致大量 kernel launch overhead、SM 利用率低、跨算子数据在 L2 cache 中失效。**

但两者的"解法直觉"不同：

| | Megakernels | Mirage MPK |
|--|------------|-----------|
| **核心思路** | 把整个前向传播编译成一张**静态指令表**，一次 launch 让所有 SM 按表执行 | 把整个前向传播编译成一张**动态 DAG**，由专用 Scheduler SM 在运行时驱动 Worker SM 按依赖顺序执行 |
| **调度发生在哪里** | **Python 侧**（launch 前完全确定） | **GPU 侧**（Scheduler SM 在运行时做任务分发） |
| **类比** | 静态编译 + 直接执行 | JIT 调度 + 运行时调度器 |

---

## 二、编译/构建流水线

### Megakernels

```
Python 模型描述
      │
      ▼ make_dag() + assign_to_sms()
  DAG_Node 列表（拓扑排序 + SM 分配）
      │
      ▼ tensorize_instructions()
  globs.instructions: Tensor [num_sms, max_queue, 32]
      │
      ▼ (launch once)
  CUDA mk<<<num_sms, threads_per_block>>>
```

- 编译期：CUDA 模板参数固化（`HIDDEN_DIM`, `HEAD_DIM` 等），需要为不同模型重新编译 `.so`
- 运行期：GPU 直接按指令表执行，无额外开销
- 关键文件：`megakernels/scheduler.py`, `include/megakernel.cuh`

### Mirage MPK

```
Python KNGraph/TBGraph
      │
      ▼ C++ 代码生成 (src/kernel/runtime.cc)
  _init_persistent_kernel()
  _execute_task() dispatcher
  task_graph.json
      │
      ▼ nvcc 编译 → .so
      │
      ▼ (launch once)
  worker_kernel + scheduler_kernel（两个流）
```

- 编译期：生成 C++ 调度代码，任务图序列化为 JSON/二进制
- 运行期：Scheduler SM 解析 event counter，动态分发任务给 Worker SM
- 关键文件：`mirage/python/mirage/mpk/mpk.py`, `persistent_kernel.cuh`

---

## 三、调度模型

### Megakernels：静态预分配

每个 SM 的指令序列在 Python 侧完全确定：

```python
# scheduler.py
sm_queues[sm_idx].append(instruction)
# → globs.instructions[sm_idx, step, :] = serialized_instruction
```

GPU 上的 Controller warp 只是顺序取指执行，没有运行时决策：

```cpp
// controller.cuh
for (instruction_index = 0; instruction_index < num_iters; instruction_index++) {
    load_instructions(&state[ring], instruction_index, g);  // 取指
    dispatch_op<page_allocator_op_dispatcher>(...);         // 分配 SMEM
    num_semaphores[ring] = dispatch_op<semaphore_constructor_op_dispatcher>(...);
    arrive(instruction_arrived[ring], 1);                   // 通知 warp 执行
}
```

### Mirage MPK：运行时事件驱动

Scheduler SM 持续监听事件队列，动态分发任务：

```cpp
// persistent_kernel.cuh:811-1082（execute_scheduler）
while (true) {
    event_id = poll(sched_queue_last_ready_event_id[sched_id]);
    switch (event_type) {
        case EVENT_LAUNCH_TASKS:
            // Round-robin 分发 task_range 到 worker queues
            for (pos = first; pos < last; pos++)
                enqueue(worker_queues[worker_id % num_workers], task_id);
        case EVENT_END_OF_TASK_GRAPH:
            prepare_next_batch();  // 动态重组 batch
    }
}
```

Workers 自行等待依赖完成后执行任务：

```cpp
// persistent_kernel.cuh:640-673（execute_worker）
while (true) {
    task = dequeue(worker_queue[worker_id]);
    // 等待前驱任务完成
    while (ld_acquire_gpu_u64(&all_event_counters[task.dependent_event])
           < expected_count) {
        __nanosleep(SLEEP_NANOS);
    }
    _execute_task(task);
    atom_add_release_gpu_u64(&all_event_counters[task.trigger_event], 1);
}
```

---

## 四、同步机制

### Megakernels：两层同步

**跨 SM（cross-block）——全局内存原子计数器 + volatile spin-poll：**

```cpp
// matvec_adds.cu:174 — 写端
atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], inst.iters);

// matvec_adds.cu:129-133 — 读端
while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1, idx}]
       < EXPECTED_ARRIVAL_COUNT) {
    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
}
```

**SM 内部（warp group 间）——共享内存 semaphore（底层 Hopper mbarrier）：**

```cpp
// megakernel.cuh
__shared__ kittens::semaphore
    instruction_arrived[PIPELINE_STAGES],
    instruction_finished[PIPELINE_STAGES],
    page_finished[NUM_PAGES][PIPELINE_STAGES_BITS];
```

### Mirage MPK：acquire/release 原子语义 + NVSHMEM

**跨 SM——PTX atomic with memory ordering：**

```cpp
// mpk_atoms.cuh
atom.add.release.gpu.u64  // 完成方：release 语义写
ld.acquire.gpu.u64        // 等待方：acquire 语义读
atom.cas.release.gpu.u64  // 队列指针 CAS 更新
```

**跨 GPU——NVSHMEM：**

```cpp
// persistent_kernel.cuh:791-804
nvshmem_signal_wait_until(event_ptr, NVSHMEM_CMP_GE, expected);
// 事件 ID 高位标记区分本地/远端
// EVENT_NVSHMEM_TAG = bit 63 of event_index
```

**对比：**

| 方面 | Megakernels | Mirage MPK |
|------|------------|-----------|
| 原语 | `atomicAdd` + volatile poll | PTX `atom.release/acquire` + `ld.acquire` |
| 内存序保证 | C++ 原子（编译器可能弱化） | PTX 内联，精确控制 acquire/release 语义 |
| 跨 GPU | 不支持 | NVSHMEM 原生支持 |
| 粒度 | 每个 opcode 一个计数器 | 每个 task/event 一个计数器 |

---

## 五、SM 角色组织

### Megakernels：每个 SM 同质，5 类 warp 分工

```
每个 SM（block）：
┌─────────────────────────────────────────────┐
│  Consumer (16 warps) — 矩阵计算             │
│  Loader   (1 warp)   — 权重 GMEM→SMEM       │
│  Storer   (1 warp)   — 结果 SMEM→GMEM       │
│  Launcher (1 warp)   — TMA 异步发起          │
│  Controller (1 warp) — 取指/SMEM分配/驱动   │
└─────────────────────────────────────────────┘
所有 132 个 SM 角色完全对称
```

### Mirage MPK：Worker SM 与 Scheduler SM 异质

```
Worker SM（如 144 个）：
┌──────────────────────────────┐
│  全部线程（256）协同执行任务  │
│  等待 event_counter 满足后   │
│  调用 _execute_task()        │
└──────────────────────────────┘

Scheduler SM（如 16 个，每个 SM 4 个 scheduler）：
┌──────────────────────────────┐
│  每个 scheduler 1 warp       │
│  仅 thread 0 活跃，轮询事件  │
│  向 worker_queue 追加任务    │
└──────────────────────────────┘
```

---

## 六、内存管理

### Megakernels：共享内存 Page 池

```
SM 共享内存（228 KB）：
├── 静态区（52 KB）：semaphore、instruction_state、scratch
└── 动态 Page 池（176 KB）：13 pages × 16 KB
     ├── page 0: 权重缓冲 A
     ├── page 1: 权重缓冲 B（double buffering）
     ├── page 2-N: 激活中间值
     └── Controller 动态分配/回收
```

- Page 由 Controller 在指令级别分配，通过 `page_finished` semaphore 追踪回收
- 不同算子对 page 需求不同，通过 `release_lid()` 描述布局

### Mirage MPK：Paged KV Cache + 全局内存

```
GPU 全局内存（预分配）：
├── weights（所有层，LazyTensor or PyTorch Tensor）
├── all_event_counters[MAX_EVENTS]  — 事件计数
├── worker_queue_*                  — 任务队列
├── sched_queue_*                   — 事件队列
└── paged_kv_cache：
     ├── paged_kv_indptr_buffer[requests+1]
     ├── paged_kv_indices_buffer[max_pages]
     └── paged_kv_last_page_len_buffer[requests]
```

- 每 decode step 调用 `prepare_next_batch()` 动态分配新 KV 页
- 不依赖共享内存做跨 SM 数据传递（全部通过全局内存）

---

## 七、动态 Shape / 批量支持

| 能力 | Megakernels | Mirage MPK |
|------|------------|-----------|
| 动态 shape | ✗ 编译时固化 | ✗ 维度也编译时固化，但 batch/seq 更灵活 |
| 动态 batch size | ✗ latency demo 固定 batch=1 | ✓ 支持动态请求增删（ONLINE 模式） |
| 可变序列长度 | ✗ | ✓ paged KV cache 支持每请求独立 seq len |
| 多 turn 对话 | ✗ | ✓ MODE_MULTI_TURN |
| 每 step 重调度 | 需要重跑 Python 调度 | ✓ GPU 侧 scheduler 自动处理 |

---

## 八、算子覆盖与模型支持

### Megakernels

专为 Llama 定制，7 个 opcode：

| Opcode | 算子 |
|--------|------|
| 1 | RMSNorm + QKV ProjectionMatVec + RoPE + KV Append |
| 2 | Partial Flash Attention（分块 online softmax） |
| 3 | Attention Reduction（多分区归并） |
| 4 | O_Proj + Residual Add |
| 5 | RMSNorm + UpGate + SiLU |
| 6 | DownProj + Residual Add |
| 7 | RMSNorm + LM Head |

- 支持模型：Llama 3.2-1B/3B（实验性）
- GQA、RoPE、Grouped attention 已支持

### Mirage MPK

100+ task type，覆盖主流模型需求：

| 分类 | 代表 Task |
|------|---------|
| 基础线性 | TASK_LINEAR, TASK_RMS_NORM_LINEAR |
| 注意力 | TASK_PAGED_ATTENTION_SPLIT_KV_SM100, TASK_MLA_DECODE_SM100 |
| MoE | TASK_MOE_W13_FP8_SM100, TASK_MOE_W2_LINEAR_SM100 |
| 投机解码 | TASK_MTP_VERIFY_STRICT, TASK_SAMPLING_SM100 |
| 多 GPU | TASK_NVSHMEM_ALLGATHER_STRIDED_PUT, TASK_NVSHMEM_TILE_ALLREDUCE |

- 支持模型：DeepSeek V3、Qwen3、Llama 系列
- MoE、MLA、MTP（Multi-Token Prediction）均有专用 task

---

## 九、硬件支持

| | Megakernels | Mirage MPK |
|--|------------|-----------|
| Ampere (sm_80) | ✗ | ✓（基础 PTX 内核） |
| Hopper (sm_90) | ✓（主要目标） | ✓（专用 Hopper 内核） |
| Blackwell (sm_100) | ✗ | ✓（sm_100 专用任务类） |
| 多 GPU | ✗ | ✓（NVSHMEM，最多 16 GPU） |
| TMA | ✓（TMA 异步加载） | ✓（TMA descriptor 嵌入 TaskDesc） |

---

## 十、优缺点总结

### Megakernels

**优点**

1. **极低的运行时开销**：所有调度决策在 Python 侧完成，GPU 上只做纯粹的计算，无调度开销
2. **可预测的执行时序**：静态指令流使得 profiling 和调试更直观
3. **精细的 SM 内 warp 分工**：Consumer/Loader/Storer/Launcher/Controller 角色专业化，重叠计算与数据搬运
4. **代码简洁清晰**：核心调度逻辑在 Python 中，CUDA 侧只负责执行
5. **TMA + 双缓冲流水线**：Launcher warp 预发 TMA，Consumer warp 算完当前块同时加载下一块

**缺点**

1. **不支持动态 shape 和动态 batch**：维度编译时固化，无法服务可变长请求
2. **每个新模型/尺寸都要重新编译 `.so`**：HIDDEN_DIM、HEAD_DIM 等作为模板参数
3. **无法跨 GPU 扩展**：缺乏多 GPU 通信支持
4. **算子集合受限**：仅覆盖 Llama-style 模型的 7 个算子
5. **批量推理能力弱**：当前 demo 固定 batch=1，无 paged KV cache
6. **研究原型阶段**：缺乏生产化部署能力（无 ONLINE 请求管理）

---

### Mirage MPK

**优点**

1. **完整的运行时调度器**：GPU 侧 Scheduler SM 动态驱动任务，支持变长 batch、动态请求到达
2. **Paged KV Cache 原生支持**：变长序列无需填充，内存利用率高
3. **硬件覆盖广**：Ampere/Hopper/Blackwell 均有针对性实现
4. **算子生态完整**：MoE、MLA、MTP、FP8、投机解码全覆盖，支持主流前沿模型
5. **多 GPU 原生支持**：NVSHMEM 实现 Tensor Parallel，最多 16 GPU
6. **生产导向**：ONLINE 模式支持连续请求、多 turn 对话，接近生产部署需求
7. **精确内存语义**：PTX inline atomic 保证正确的 acquire/release 内存序

**缺点**

1. **调度开销**：Scheduler SM 消耗 GPU 资源（约 10% SM 用于调度），对小模型/低延迟场景有损耗
2. **编译流水线复杂**：Python → C++ 代码生成 → nvcc 编译，调试难度较高
3. **全局内存中心化**：跨 SM 通信完全依赖全局内存 + 原子，带宽受限
4. **运行时开销不确定**：动态调度使得 tail latency 不如静态预分配稳定
5. **共享内存利用不充分**：Worker SM 无显式共享内存 page 管理机制，每个 task 独立管理
6. **队列深度上限**：worker_queue 深度 8192，scheduler_queue 深度 4096（`persistent_kernel.cuh:383`），深度复杂任务图可能溢出

---

## 十一、相同点

| 维度 | 共同点 |
|------|--------|
| **核心架构** | 单次 kernel launch，persistent kernel 常驻 GPU |
| **调度单元** | 一个 SM 对应一个 thread block |
| **跨 SM 同步** | 全局内存原子计数器 + spin-wait（`__nanosleep`） |
| **TMA 使用** | 均使用 TMA 异步加载（Hopper 特性） |
| **指令/任务格式** | 均有序列化的固定格式（Megakernels: 32 int；MPK: TaskDesc 16B 对齐） |
| **设计目标** | 消除 kernel launch overhead，提升 SM 利用率 |
| **精度** | 均主要使用 BF16 |
| **框架依赖** | 均依赖 PyTorch 做权重管理和 Python 接口 |

---

## 十二、差异点汇总

| 维度 | Megakernels | Mirage MPK |
|------|------------|-----------|
| **调度时机** | launch 前（Python 侧静态） | 运行时（GPU 侧动态） |
| **SM 异质性** | 所有 SM 同构 | Worker SM + Scheduler SM 异构 |
| **SM 内 warp 分工** | 5 类专用 warp | 全部线程协同执行 |
| **指令/任务分发** | 预写入内存表，顺序读取 | 事件触发，运行时入队 |
| **动态 batch** | ✗ | ✓ |
| **跨 GPU** | ✗ | ✓ NVSHMEM |
| **算子数量** | 7 | 100+ |
| **支持模型** | Llama 1B/3B | DeepSeek V3, Qwen3, Llama 等 |
| **动态形状** | ✗ | 部分（batch/seq 动态，维度静态） |
| **内存同步原语** | C++ `atomicAdd` | PTX `atom.release/acquire` |
| **硬件目标** | Hopper | Ampere/Hopper/Blackwell |
| **KV Cache** | 静态分配 | Paged（动态扩展） |
| **成熟度** | 研究原型 | 接近生产（Alpha/Beta） |
| **代码复杂度** | 中（CUDA 侧简洁） | 高（多层代码生成） |

---

## 十三、适用场景建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 研究新 SM 调度策略 | **Megakernels** | Python 侧调度算法清晰可插拔 |
| 单 token decode，极致低延迟 | **Megakernels** | 无运行时调度开销，warp 分工精细 |
| 在线服务，变长请求 | **Mirage MPK** | ONLINE 模式 + paged KV cache |
| 超大模型（MoE/MLA） | **Mirage MPK** | 覆盖 DeepSeek V3 所有算子 |
| 多 GPU Tensor Parallel | **Mirage MPK** | NVSHMEM 原生支持 |
| 快速验证新模型结构 | **Megakernels** | Python 调度层易于修改和调试 |

---

## 参考文件索引

| 文件 | 说明 |
|------|------|
| `Megakernels/include/megakernel.cuh` | persistent kernel 入口，warp 分工 |
| `Megakernels/include/controller/controller.cuh` | Controller 取指主循环 |
| `Megakernels/megakernels/scheduler.py` | Python DAG 构建与 SM 分配 |
| `Megakernels/demos/low-latency-llama/llama.cuh` | Llama globals_t，模板参数 |
| `Megakernels/demos/low-latency-llama/matvec_adds.cu` | 跨 SM 同步（barrier_update） |
| `mirage/include/mirage/persistent_kernel/persistent_kernel.cuh` | MPK worker/scheduler 主循环 |
| `mirage/include/mirage/persistent_kernel/runtime_header.h` | TaskDesc, EventDesc 格式 |
| `mirage/include/mirage/persistent_kernel/mpk_atoms.cuh` | PTX 原子操作原语 |
| `mirage/python/mirage/mpk/mpk.py` | Python MPK 接口，batch 管理 |
