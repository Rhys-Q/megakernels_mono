# 09 — 设计解析：调度单元、同步机制与动态 Shape

---

## 一、每个 SM 的调度单元是 kernel 还是 thread block？

**结论：thread block（每个 SM 分配一个 thread block）**

CUDA megakernel 以如下方式启动：

```cpp
// include/megakernel.cuh
mk<<<num_sms, num_threads_per_block>>>(globals);
```

- `gridDim.x = num_sms`，即 H100 上为 132 个 block
- 每个 block 通过 `blockIdx.x` 读取 `globs.instructions[blockIdx.x]` ——自己那一行指令队列
- 整个推理过程只有**一次 kernel launch**（persistent kernel），不是"每条指令一个 kernel"

所以**调度单元是 thread block**，不是 kernel。每个 SM 上常驻一个 thread block，block 内部再通过 warp group 分工（Controller / Consumer / Loader / Storer / Launcher），循环执行指令队列里的所有指令直到 NoOp 结束。

```
一次 kernel launch
┌──────────────────────────────────────────┐
│  block 0 → SM 0 → 执行 instructions[0]  │
│  block 1 → SM 1 → 执行 instructions[1]  │
│  ...                                     │
│  block 131 → SM 131 → instructions[131] │
└──────────────────────────────────────────┘
```

---

## 二、同步机制是什么？

**结论：全局内存原子计数器 + 自旋等待（跨 SM），共享内存 semaphore（SM 内部）**

### 跨 SM 同步：global memory atomicAdd + spin-poll

所有跨 SM 的依赖同步都通过 `barriers` 张量（形状 `[num_layers, 10, num_heads_total]`，存放在全局内存）完成：

**写端（完成的 SM）：**

```cpp
// demos/low-latency-llama/matvec_adds.cu:174
atomicAdd(&g.Bar[{inst.layer, opcode - 1, 0}], inst.iters);

// demos/low-latency-llama/attention_partial.cu:658
atomicAdd(&g.Bar[{inst.layer_idx, OPCODE_PARTIAL_ATTN - 1, head}], 1);
```

**读端（等待的 SM）：**

```cpp
// demos/low-latency-llama/matvec_adds.cu:129-133
while (*(volatile int *)&g.Bar[{inst.layer, prev_opcode - 1,
                                inst.reduction_block_idx}]
       < EXPECTED_ARRIVAL_COUNT) {
    __nanosleep(Config::GMEM_SPIN_LOOP_SLEEP_NANOS);
}
```

等待 SM 用 `volatile` 读反复轮询计数器，配合 `__nanosleep()` 避免过度占用 L2 带宽。当计数达到 `EXPECTED_ARRIVAL_COUNT`（所有产生该 opcode 输出的 SM 都到达）时继续执行下一条指令。

**整体流程：**

```
SM-A (执行 QKV)         SM-B (执行 Attention，依赖 QKV)
      │                           │
      │ 完成 QKV block             │ 轮询 Bar[layer, QKV-1, head]
      │ atomicAdd(Bar, 1) ────────►│ < threshold → __nanosleep
      │                           │ = threshold → 继续执行
```

每次 decode step 开始前，Python 侧将 `barriers.zero_()` 重置所有计数器。

### SM 内部同步：ThunderKittens semaphore（共享内存 mbarrier）

block 内部的 warp group 之间（Controller ↔ Consumer/Loader/Storer/Launcher）通过共享内存中的 `kittens::semaphore` 同步，其底层是 Hopper 的 `mbarrier` 指令：

```cpp
// include/megakernel.cuh
__shared__ kittens::semaphore
    page_finished[NUM_PAGES][PIPELINE_STAGES_BITS],
    instruction_arrived[PIPELINE_STAGES],   // Controller → 其他 warp
    instruction_finished[PIPELINE_STAGES];  // 其他 warp → Controller
```

这些 semaphore 仅在 thread block 内部有效，不跨 SM。

### 小结

| 范围 | 机制 | 原语 |
|------|------|------|
| 跨 SM（cross-block） | 全局内存原子计数器 + volatile spin-poll | `atomicAdd` + `*(volatile int *)` + `__nanosleep` |
| SM 内部（warp group 间） | 共享内存 semaphore | `kittens::semaphore`（`mbarrier`） |

---

## 三、支持动态 Shape 吗？

**结论：不支持——所有张量维度在 Python 调度期静态确定，并编译进 CUDA 模板参数**

### 静态编译的形状

模型的核心维度（`hidden_size`、`head_dim`、`num_kv_heads` 等）作为 C++ 模板参数编译进 `.so`：

```cpp
// demos/low-latency-llama/llama.cuh
using qkv_layout = kittens::gl<bf16, 1, -1, -1, HIDDEN_DIM>;
//                                               ^^^^^^^^^^^^ 编译时常量
```

`HIDDEN_DIM`、`HEAD_DIM`、`NUM_KV_HEADS` 等均在 `Makefile` 或 `config.cuh` 中以宏定义传入，**不同模型尺寸需要重新编译**。

### 指令张量在 Python 调度期生成

`globs.instructions`（形状 `[num_sms, max_queue_len, 32]`）在 `tensorize_instructions()` 中由 Python 计算好后整体传给 GPU，每条指令的 `start_block_idx`、`end_block_idx` 等字段都已编码为具体数值：

```python
# megakernels/scheduler.py
globs.instructions = torch.tensor(flattened, dtype=torch.int32).view(num_sms, -1, 32)
```

GPU 上的 Controller 只是顺序读取这个静态表，无法在运行时根据实际 shape 调整分块。

### 唯一的动态部分

| 字段 | 是否动态 | 说明 |
|------|----------|------|
| `pos_id` | ✓ | 每个 decode step 递增 |
| KV cache 有效长度 | ✓ | 随生成 token 增长，由 `pos_id` 隐式控制 |
| 张量维度 | ✗ | 编译时固定 |
| 指令分块方式 | ✗ | Python 调度期静态决定 |
| batch size | ✗ | 当前 latency demo 固定 batch=1 |

因此 megakernels **不支持运行时动态 shape**。若需要支持不同序列长度或不同 batch size，需要在 Python 侧重新调用 `tensorize_instructions()` 生成新的指令张量（整个调度过程重跑），而非在 GPU 上动态调整。
