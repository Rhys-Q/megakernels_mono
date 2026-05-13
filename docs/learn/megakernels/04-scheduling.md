# 04 — 调度系统：DAG 构建、SM 分配与张量化

## 概述

调度层的工作分三步：

1. **构建 DAG**：把 Transformer 每一层的操作分解成若干 `Instruction` 节点，按数据依赖关系连接成有向无环图（DAG）
2. **分配到 SM**：选择一种调度策略，把指令分配给各 SM 的队列
3. **张量化**：把指令队列序列化成 GPU 可直接读取的整数张量

源文件：
- `megakernels/scheduler.py` — 通用框架（DAG_Node、Schedule、分配算法、tensorize）
- `megakernels/demos/latency/scheduler.py` — Latency 模式的具体 DAG 构建逻辑

---

## DAG_Node — 指令节点

源文件：`megakernels/scheduler.py:17`

```python
@dataclass
class DAG_Node:
    instruction: Instruction          # 携带的指令
    dependencies: list[DAG_Node]      # 必须先完成的前驱节点
    children: set[DAG_Node]           # 依赖本节点的后继节点
    start_time: float                 # 调度后的开始时间（用于 dag 模式）
    end_time: float                   # 调度后的结束时间
    priority: float                   # 优先级（关键路径长度，越大越先调度）
    remaining_dependencies: set       # 调度时动态维护：还有多少前驱未完成
```

`dependencies` 是**数据依赖**：如果指令 B 需要读取指令 A 的输出，那么 B 的 `dependencies` 包含 A。

---

## Step 1: 构建 DAG

### make_globals()

源文件：`megakernels/demos/latency/scheduler.py:38`

第一步是创建 `Globals` 对象，为 GPU 上的激活缓冲分配内存：

```python
globs = make_globals(model)
```

主要工作：
- 从 `model.stacked_params` 读取已堆叠的权重引用
- 分配激活缓冲张量（`hidden_states`、`post_ln_rope_q`、`attn_out`、`silu_out`、`logits`）
- 初始化 `barriers` 张量（形状 `[num_layers, 10, num_heads_total]`，全零）
- 填入 block 大小常量（如 `qkv_block_size=16`）

### make_dag()

源文件：`megakernels/demos/latency/scheduler.py:235`

```python
def make_dag(globs, stop_after_op=None, layer_limit=None):
    nodes = []
    last_outputs = []

    for layer_idx in range(globs.num_hidden_layers):
        new_nodes, new_outputs = make_dag_layer(
            globs, layer_idx, prev_layer_outputs=last_outputs
        )
        nodes.extend(new_nodes)
        last_outputs = new_outputs

    # 最后一层之后：LM Head
    lm_head_instructions = schedule_lm_head(globs)
    lm_head_nodes = [DAG_Node(ins, last_outputs) for ins in lm_head_instructions]
    nodes.extend(lm_head_nodes)

    end_node = DAG_Node(NoOp(), lm_head_nodes)
    return nodes, end_node
```

每层内部由 `make_dag_layer()` 构建，产生节点并建立依赖关系：

### make_dag_layer() 详解

源文件：`megakernels/demos/latency/scheduler.py:270`

```
Layer N 的节点结构：

┌─────────────────────────────────────────────────────────┐
│  前一层输出节点 (prev_layer_outputs)                      │
└────────────────────────┬────────────────────────────────┘
                         │ (依赖)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  QKV 节点 (num_sms 个)                                    │
│  schedule_qkv():                                         │
│  SM 0: QKV[layer, block 0..k0]                           │
│  SM 1: QKV[layer, block k0..k1]                          │
│  ...                                                     │
└────────────────────────┬────────────────────────────────┘
                         │
       ┌─────────────────┼───────────────┐
       ▼                 ▼               ▼
  PartialAttn        PartialAttn    PartialAttn
  kv_head=0,         kv_head=1,     ...
  partial=0          partial=0
  (依赖产生对应 K/V block 的 QKV 节点)
       │
       ▼
  O_ProjResidual × num_o_blocks
  (每个 o_block 独立, 依赖所有 PartialAttn)
       │
       ▼
  UpGateSiLU × num_sms
  (依赖所有 O_ProjResidual)
       │
       ▼
  DownProjResidual × num_sms
  (依赖所有 UpGateSiLU)
```

**精细依赖追踪**（QKV → Partial）：

每个 `PartialAttention(kv_head_idx)` 只需要读取对应 K/V head 的 QKV block。代码通过 `qkv_deps` 字典精确追踪：

```python
# 建立 block 级别的依赖索引
qkv_deps[(layer_idx, opcode1, block_idx)] = qkv_node

# PartialAttention 只依赖产生其 K/V block 的那些 QKV 节点
k_start_block = (num_attention_heads + kv_head_idx) * head_dim // qkv_block_size
v_start_block = (num_attention_heads + num_kv_heads + kv_head_idx) * head_dim // qkv_block_size
block_indices = k_blocks + v_blocks
deps = {qkv_deps[(layer, opcode1, b)] for b in block_indices}
```

这使得 PartialAttention 可以在部分 QKV 指令完成后就开始，无需等待所有 QKV 完成。

---

## Step 2: 分配到 SM

源文件：`megakernels/scheduler.py:94`

分配函数签名：
```python
def assign_to_sms(mode: str, schedule: Schedule, ...) -> list[list[Instruction]]
```

返回值：`sm_queues[sm_idx]` = 第 `sm_idx` 个 SM 要执行的指令列表（按执行顺序）。

### 策略 1: `dag` — DAG 感知贪心调度

源文件：`megakernels/scheduler.py:94`

最复杂也最精确的策略。维护两个堆：

```
sm_heap:    (sm_end_time, sm_idx)   # 下一个空闲的 SM 是哪个
ready_heap: (-cost, node_idx)       # 依赖已满足、可以执行的节点（按代价降序）
```

算法：
1. 初始化：所有没有依赖的节点放入 `ready_heap`
2. 循环：
   - 从 `ready_heap` 取出代价最大的节点（贪心：大任务优先）
   - 从 `sm_heap` 取出当前最空闲的 SM
   - 把节点分配给该 SM，更新 SM 的预期完成时间
   - 检查该节点的所有子节点，将依赖已全部满足的加入 `ready_heap`

**优点**：考虑数据依赖，理论上接近最优  
**缺点**：没有考虑不同 SM 指令间的同步（GPU 上指令顺序执行，barrier 保证跨 SM 同步）

### 策略 2: `rr` — Round Robin

```python
# SM 0 → SM 1 → SM 2 → ... → SM n → SM 0 → ...
sm_queues[i % sm_count].append(instruction)
```

最简单，完全不考虑代价或依赖，但对于均匀分布的工作负载效果不差。

### 策略 3: `zz` — Zigzag

```python
# SM 0 → 1 → 2 → ... → n → n-1 → ... → 0 → 1 → ...
```

连续两条指令分配到相邻 SM，提高 L2 缓存局部性（相邻 SM 共享部分 L2 缓存）。

### 策略 4: `wave` — Wave 调度

源文件：`megakernels/scheduler.py:194`

把 opcode 相同的连续指令归为一个"波次（wave）"，在每个波次内按代价均衡分配：

```
指令序列: [QKV,QKV,QKV,  Attn,Attn,  QKV,QKV, ...]
波次划分: [── wave 1 ──][─ wave 2 ─][─ wave 3 ─]
```

每个波次内，按代价从大到小排序，用最小堆选择最空闲 SM：

```python
for wave in waves:
    sorted_by_cost = sorted(wave, key=lambda x: x.cost(globs), reverse=True)
    for ins in sorted_by_cost:
        sm_cost, sm_idx = heapq.heappop(sm_heap)
        sm_queues[sm_idx].append(ins)
        heapq.heappush(sm_heap, (sm_cost + ins.cost(globs), sm_idx))
```

**优点**：避免同 opcode 的指令分散到不同轮次（减少 barrier 等待），同时在波次内做负载均衡。  
**推荐**：`generate.py` 默认使用此策略。

### 策略 5: `pool` — 内存/计算池

按 `tags()["pool"]` 把指令分成 "memory" 和 "compute" 两类，分别分配到不同比例的 SM 子集，类似 CUDA stream 隔离。需要指令实现 `tags()` 方法（目前仅部分指令支持）。

---

## Step 3: 张量化（Tensorize）

源文件：`megakernels/scheduler.py:281`

```python
def tensorize_instructions(globs, instruction_queues):
    max_queue_len = max(len(q) for q in instruction_queues)

    # padding: 队列短的 SM 用 NoOp 补齐
    for queue in instruction_queues:
        queue.extend([NoOp()] * (max_queue_len - len(queue)))

    # 序列化: 每条指令 → 32 个 int，不足补 0
    flattened = [serialize_and_pad(ins) for queue in instruction_queues
                                         for ins in queue]

    # 整形为 [num_sms, max_queue_len, 32]
    globs.instructions = torch.tensor(flattened, dtype=torch.int32, device=device)
                                      .view(num_sms, -1, 32)

    # 同时分配计时张量
    globs.timings = torch.zeros([num_sms, max_queue_len, 128], dtype=torch.int32)
```

最终产物 `globs.instructions` 是一个整数张量，GPU 上每个 SM（block）读取自己那一行（`[sm_idx, :, :]`），按顺序执行每条指令。

### 内存布局可视化

```
globs.instructions: shape [132, max_q, 32]

SM 0:  [op1,0,0,3,0,...,0]  [op2,0,0,1,0,...,0]  [op3,...] ...
SM 1:  [op1,0,3,6,0,...,0]  [op2,0,1,1,0,...,0]  [0,0,...] ...  ← NoOp 填充
...
SM 131:[op1,...]             [op5,...]             ...
        ^                    ^
        |                    |
        第 0 条指令          第 1 条指令
        (32 个 int)         (32 个 int)
```

---

## 完整调度示例：Llama-1B 的规模

对于 `meta-llama/Llama-3.2-1B`：
- `num_hidden_layers = 16`
- `num_attention_heads = 32`（Hq），`num_kv_heads = 8`（Hkv）
- `hidden_size = 2048`，`intermediate_size = 8192`
- H100 有 132 SM，`qkv_block_size = 16`

每层的指令数量（wave 策略）：
| 操作 | 指令数 |
|------|--------|
| QKV（按 SM 数分块） | 132 |
| PartialAttention（8 KV heads × 1 partition） | 8 |
| O_ProjResidual（2048/16 = 128 blocks） | 128 |
| UpGateSiLU（按 SM 数分块） | 132 |
| DownProjResidual（按 SM 数分块） | 132 |

共 16 层 × ~532 + LM_Head(132) ≈ **8644 条指令**，分布在 132 个 SM 的队列中。

下一步：[05-llama-model.md](05-llama-model.md) — Llama 模型实现细节
