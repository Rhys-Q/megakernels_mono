# 07 — CUDA Megakernel 深度解析

源文件：`include/megakernel.cuh`，`include/controller/controller.cuh`，`demos/low-latency-llama/`

本章讲解 CUDA 后端的核心机制，包括 warp group 角色分工、指令调度流水线、semaphore 同步，以及 TMA 异步数据加载。

---

## 整体架构：每个 SM 就是一台小型 VM

CUDA kernel `mk` 以 `<<<num_sms, num_threads_per_block>>>` 启动，其中：
- `num_sms` = GPU 的 SM 数量（H100 为 132）
- 每个 block 对应一个 SM
- 每个 block 内有多个 warp group（每个 warp group = 4 个 warp = 128 线程）

每个 SM 上的所有线程运行**同一份**代码（SPMD），但通过 `blockIdx.x` 读取不同的指令队列。

---

## Warp Group 角色分工

源文件：`include/megakernel.cuh:118`

```cpp
if (warpid() < NUM_CONSUMER_WARPS) {
    increase_registers<CONSUMER_REGISTERS>();
    consumer::main_loop<config, globals, ops...>(g, mks);
} else {
    decrease_registers<NON_CONSUMER_REGISTERS>();
    switch (warpgroup::warpid()) {
        case 0: loader::main_loop(g, mks);     break;
        case 1: storer::main_loop(g, mks);     break;
        case 2: launcher::main_loop(g, mks);   break;
        case 3: controller::main_loop(g, mks); break;
    }
}
```

| Warp Group | 角色 | 说明 |
|-----------|------|------|
| Consumer (0..N-1) | **计算核心** | 执行矩阵向量乘、attention 等实际计算；寄存器分配更多（`CONSUMER_REGISTERS`） |
| Loader (N) | **权重加载** | 从全局内存把权重加载到共享内存（SMEM）的 page |
| Storer (N+1) | **结果写回** | 把计算结果从 SMEM 写回到全局内存 |
| Launcher (N+2) | **TMA 发起** | 通过 TMA（Tensor Memory Accelerator）异步加载 KV cache 数据 |
| Controller (N+3) | **指令调度** | 取指、分配 SMEM page、构建 semaphore、通知其他 warp 执行 |

**寄存器分配策略**：Consumer 做计算需要更多寄存器（存中间结果），其他 warp group 把寄存器"退还"给 Consumer（`decrease_registers`），最大化 Consumer 的可用寄存器数。

---

## 共享内存 Pages 机制

每个 SM 的共享内存（SMEM）被组织成固定大小的 **pages**（类似操作系统的内存页）。

- `NUM_PAGES` 个 page，每页固定大小
- Controller 为每条指令分配所需的 page（`page_allocator.cuh`）
- Loader 把权重写入分配好的 page
- Consumer 从 page 读取数据进行计算
- Storer 把结果从 page 写回全局内存

Page 复用：一条指令完成后，其 page 被归还，可供下一条指令使用。`page_finished` semaphore 追踪每个 page 是否可安全复用。

---

## Controller 主循环

源文件：`include/controller/controller.cuh:14`

Controller 是整个 SM 的"大脑"，驱动指令执行流水线：

```cpp
for (instruction_index = 0; instruction_index < num_iters; instruction_index++) {

    // Step 0: 等待上一个用这个 ring slot 的指令完成（pipeline 防冲突）
    if (instruction_index >= INSTRUCTION_PIPELINE_STAGES) {
        wait(instruction_finished[ring], phasebit);
        invalidate_semaphore(...);  // 清理旧的 semaphore
        store_timings(...);         // 记录性能计时
    }

    // Step 1: 从全局内存取指（读取 globs.instructions[sm_idx][instruction_index]）
    load_instructions(&instruction_state[ring], instruction_index, g);

    // Step 2: 分配 SMEM pages（基于上一条指令的 page 使用模式复用）
    dispatch_op<page_allocator_op_dispatcher>(last_opcode, g, ...);

    // Step 3: 构建 semaphore（初始化本条指令需要的同步原语）
    num_semaphores[ring] = dispatch_op<semaphore_constructor_op_dispatcher>(...);

    // Step 4: 通知所有其他 warp group 本指令已就绪
    arrive(instruction_arrived[ring], 1);
}
```

**指令流水线（Instruction Pipelining）**：

`INSTRUCTION_PIPELINE_STAGES`（通常为 2 或 4）个 ring slot 允许 Controller 在等待当前指令完成时，已经为下一条指令做好准备（取指、分配 page、构建 semaphore），形成流水线重叠。

```
ring slot 0: [instruction A 执行中]
ring slot 1: [instruction B 取指、分配page、构建sem] ← Controller 正在处理
```

---

## Semaphore 同步机制

```cpp
// 共享内存中的 semaphore 数组
__shared__ kittens::semaphore
    page_finished[NUM_PAGES][INSTRUCTION_PIPELINE_STAGES_BITS],
    instruction_arrived[INSTRUCTION_PIPELINE_STAGES],  // Controller → 其他 warp
    instruction_finished[INSTRUCTION_PIPELINE_STAGES]; // 其他 warp → Controller
```

**执行流程**：

```
Controller                     Consumer / Loader / Storer / Launcher
    │                                    │
    │ arrive(instruction_arrived)        │
    ├──────────────────────────────────► │
    │                          wait(instruction_arrived)
    │                                    │ 开始执行
    │                                    │
    │                          arrive(instruction_finished)
    │ ◄────────────────────────────────── │
    │ wait(instruction_finished)         │
    │ （收到 NUM_WARPS-1 个 arrive）      │
    │ 继续下一条指令                      │
```

每条指令完成后，Consumer、Loader、Storer 各自 arrive 一次 `instruction_finished`，Controller 等待收到 `NUM_WARPS-1` 次后继续。

---

## TMA 异步数据加载（Hopper 特性）

对于 attention 计算，KV cache 数据量大，使用 **TMA（Tensor Memory Accelerator）** 异步加载可以让加载和计算重叠：

```
时间线：
Launcher:  [发 TMA 异步加载 K/V block 0] [发 TMA 异步加载 K/V block 1] ...
Consumer:  [等待 block 0 到达] → [计算 block 0 的 attention] → [等待 block 1] → ...
```

Launcher warp 提前发出 TMA 请求，Consumer warp 通过 `wait(kv_arrived_semaphore)` 等待数据就绪，然后立即开始计算，同时 Launcher 继续发下一批 TMA 请求。

这种 **double-buffering** 流水线使内存带宽和计算带宽能同时利用。

---

## attention_partial.cu 详解

源文件：`demos/low-latency-llama/attention_partial.cu`（Opcode 2）

这是最复杂的一个 kernel，实现了流水线化的 partial attention 计算。

### parsed_instruction 结构

```cpp
struct parsed_instruction {
    int layer_idx;     // 当前是哪层
    int kv_head_idx;   // 哪个 KV head
    int num_partials;  // 总共多少 partition
    int partial_idx;   // 本 SM 负责哪个 partition
};
```

### Consumer 的在线 softmax

```cpp
// 在线 softmax（不需要预先知道全部 QK 值）
float softmax_temp = attn_scale * 1.44269504089f;  // = 1/sqrt(D) * 1/ln(2)

float max_qk = -INFINITY;
float normalizer = 0.0f;
// accumulator = zeros

for (kv_block_idx = start; kv_block_idx < end; kv_block_idx++) {
    // wait for TMA to load K/V block
    wait(kv_arrived);

    // compute Q @ K.T
    qk = Q @ K_block.T  // [gqa_ratio, block_size]

    // apply right-fill mask（sequence 末尾的无效位置设为 -inf）
    right_fill_mask(qk, valid_tokens);

    // 在线更新 max 和 normalizer
    new_max = max(max_qk, max(qk * softmax_temp))
    normalizer = normalizer * exp2(max_qk - new_max) + sum(exp2(qk * softmax_temp - new_max))
    // 用 exp2 而不是 exp，因为 softmax_temp 包含了 1/ln(2) 因子

    // 更新 accumulator
    acc = acc * exp2(max_qk - new_max) + softmax(qk) @ V_block

    max_qk = new_max;
}

// 最终归一化
out = accumulator / normalizer;
lse = log2(normalizer) + max_qk;  // log-sum-exp，用于多 partition 归并
```

**为什么用 exp2 而不是 exp？**  
NVIDIA GPU 的 `exp2` 指令（2^x）比 `exp` 更快。通过把 `attn_scale` 预乘 `1/ln(2)` = `1.44269504089`，把自然指数转为以 2 为底的指数，在不损失精度的情况下加速计算。

### Storer 的写回逻辑

```cpp
// 如果是唯一的 partition（skip_attn_reduction = true），直接写 attn_out
// 否则写 attn_out_intermediates[head, partial_idx] 和 attn_lse_intermediates[head, partial_idx]
if (is_only_partition) {
    attn_out[head] = out;
    barrier_update(OPCODE_ATTN_REDUCTION, head);
} else {
    attn_out_intermediates[head][partial_idx] = out;
    attn_lse_intermediates[head][partial_idx] = lse;
    barrier_update(OPCODE_PARTIAL_ATTN, head);
}
```

---

## globals_t — CUDA 端的全局状态

源文件：`demos/low-latency-llama/llama.cuh`

CUDA 端有自己的 `globals_t` 结构体，用 ThunderKittens 的类型描述 GPU 内存布局：

```cpp
struct globals_t {
    // ThunderKittens 布局描述符（用于 TMA）
    using qkv_layout = kittens::gl<bf16, 1, -1, -1, HIDDEN_DIM>;
    using attn_out_layout = kittens::gl<bf16, 1, 1, -1, HEAD_DIM>;
    // ...

    // 权重指针（所有层堆叠）
    qkv_layout qkv_weights;
    attn_out_layout o_proj;
    // ...

    // KV cache
    kv_cache_layout k_cache, v_cache;

    // 激活缓冲
    hidden_layout hidden_states;
    q_layout post_rope_q;
    // ...

    // 指令表和计时
    instructions_layout instructions;
    timings_layout timings;

    // 标量
    int pos_id;
    float attn_scale;
    // ...
};
```

ThunderKittens 的 `gl<T, batch, rows, cols, dim>` 是一个泛型布局描述符，可以直接用于 TMA 异步加载，无需手动计算内存偏移。

---

## 编译与运行

```bash
# 编译 CUDA megakernel（H100 目标）
cd demos/low-latency-llama
export THUNDERKITTENS_ROOT=/path/to/ThunderKittens
export MEGAKERNELS_ROOT=/path/to/Megakernels
make GPU_TARGET=H100

# 产物：latency_megakernel.cpython-3xx-linux-gnu.so
```

`Makefile` 中的编译标志：
- `-arch=sm_90a`（H100 Hopper 架构）
- `-O3 -use_fast_math`
- `--maxrregcount=<N>`（控制每个线程最大寄存器数，影响 occupancy）

下一步：[08-demo-walkthrough.md](08-demo-walkthrough.md) — 完整端到端 Demo 步骤分析
