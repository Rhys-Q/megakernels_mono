# 02 — 技术架构

## 整体层次图

```
┌──────────────────────────────────────────────────────────────┐
│                     用户代码 / 脚本                           │
│          megakernels/scripts/generate.py                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   模型层 (Python / PyTorch)                   │
│                   megakernels/llama.py                       │
│                                                              │
│  LlamaForCausalLM                                            │
│    ├── LlamaEmbeddings (token embedding)                     │
│    ├── LlamaModel                                            │
│    │     └── LlamaBlock × N                                  │
│    │           ├── LlamaAttention (GQA + RoPE + KV cache)   │
│    │           └── LlamaMLP (up/gate/down + SiLU)           │
│    └── LlamaLMHead (RMSNorm + lm_head)                      │
│                                                              │
│  stack_params(): 把 N 层权重纵向拼接 → GPU 连续内存          │
└──────────────────────────────┬───────────────────────────────┘
                               │ model
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   调度层 (Python)                             │
│                                                              │
│  ScheduleBuilder.build(model)                                │
│    │                                                         │
│    ├── make_globals(model)                                   │
│    │     └── Globals: 权重引用 + 激活缓冲 + 常量             │
│    │                                                         │
│    └── make_dag(globs)                                       │
│          └── DAG_Node 列表（拓扑顺序）                       │
│                                                              │
│  assign_to_sms(mode, schedule)                               │
│    └── 5 种策略: dag / rr / zz / wave / pool                │
│          └── list[list[Instruction]]  (每个 SM 的队列)      │
│                                                              │
│  tensorize_instructions(globs, queues)                       │
│    └── globs.instructions: Tensor[num_sms, max_q, 32]       │
└──────────────────────────────┬───────────────────────────────┘
                               │ Schedule + Globals
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   执行层 (三选一)                             │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ PyTorchGenerator│  │  PyVM_Generator │  │ MK_Generator│ │
│  │                 │  │                 │  │             │ │
│  │ model.forward() │  │ Python 解释器   │  │ CUDA .so    │ │
│  │ (正确性基准)    │  │ (调试用)        │  │ (生产级)    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└──────────────────────────────┬───────────────────────────────┘
                               │ (MK 路径)
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                   CUDA Megakernel                             │
│        demos/low-latency-llama/ + include/                   │
│                                                              │
│  每个 SM 并行运行同一个 kernel 二进制，但执行不同指令         │
│                                                              │
│  每个 SM 内部有 5 类 warp group:                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │ Consumer │ │  Loader  │ │  Storer  │ │  Launcher      │ │
│  │ (计算)   │ │ (加载权重)│ │ (写回结果)│ │ (TMA 异步拷贝) │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘ │
│  ┌──────────┐                                               │
│  │Controller│  fetch 指令 → 分配共享内存页 → 构建 semaphore │
│  │ (调度)   │  → 触发其他 warp 执行 → 循环                  │
│  └──────────┘                                               │
└──────────────────────────────────────────────────────────────┘
```

## 数据流：一次 Decode Step

```
1. 输入 token_id (int)
       │
       ▼
2. embedding lookup → hidden_states [hidden_size]  (存入 globs)
       │
       ▼
3. megakernel 启动 (一次 CUDA kernel launch)
       │
   ┌───┴──────────────────────────────────────────────────┐
   │  SM 0         SM 1         ...         SM 131         │
   │  queue[0]     queue[1]                 queue[131]     │
   │  ┌────────┐   ┌────────┐               ┌────────┐    │
   │  │QKV L0  │   │QKV L0  │               │QKV L0  │    │
   │  │Attn L0 │   │Attn L0 │               │Attn L0 │    │
   │  │O-Proj  │   │O-Proj  │   (不同分块)   │O-Proj  │    │
   │  │UpGate  │   │UpGate  │               │UpGate  │    │
   │  │...     │   │...     │               │...     │    │
   │  └────────┘   └────────┘               └────────┘    │
   └─────────────────────────────┬────────────────────────┘
                                 │ (所有 SM 完成后)
                                 ▼
4. logits [vocab_size] (在 globs.logits 中)
       │
       ▼
5. argmax → 下一个 token_id
```

## 关键数据结构速览

### Globals（全局共享状态）

```python
# megakernels/demos/latency/instructions.py
@dataclass
class Globals(BaseGlobals):
    # 模型权重（所有层已堆叠）
    qkv_proj_weights: Tensor   # [num_layers, qkv_outdim, hidden_size]
    o_proj_weights:   Tensor   # [num_layers, hidden_size, hidden_size]
    up_proj_weights:  Tensor   # [num_layers, intermediate_size, hidden_size]
    gate_proj_weights:Tensor   # [num_layers, intermediate_size, hidden_size]
    down_proj_weights:Tensor   # [num_layers, hidden_size, intermediate_size]
    k_cache: Tensor            # [num_layers, 1, max_seq, num_kv_heads, head_dim]
    v_cache: Tensor            # [num_layers, 1, max_seq, num_kv_heads, head_dim]

    # 激活缓冲（每步复用）
    hidden_states:  Tensor     # [hidden_size]          - 主激活流
    post_ln_rope_q: Tensor     # [hidden_size]          - Q after RoPE
    attn_out:       Tensor     # [hidden_size]          - attention 输出
    silu_out:       Tensor     # [intermediate_size]    - MLP 中间结果
    logits:         Tensor     # [vocab_size]           - 最终输出

    # 同步原语
    barriers: Tensor           # [num_layers, 10, num_heads] int32

    # 运行时状态
    instructions: Tensor       # [num_sms, max_queue, 32]  - 指令表
    pos_id: int                # 当前 token 位置
```

### Instruction（指令基类）

```python
# megakernels/instructions.py
@dataclass
class Instruction:
    @classmethod
    def opcode(cls) -> int: ...          # 唯一标识
    @classmethod
    def prev_opcode(cls) -> int: ...     # 依赖的前驱 opcode（用于 barrier 检查）
    def cost(self, globs) -> float: ...  # 估算执行代价（用于调度）
    def serialize(self) -> list[int]: ... # 转为 32 个 int（写入指令张量）
```

### Schedule（调度产物）

```python
# megakernels/scheduler.py
@dataclass
class Schedule:
    globs: BaseGlobals           # 共享状态
    dag_nodes: list[DAG_Node]    # 拓扑排序后的指令节点
    end_node: DAG_Node           # 终止节点（NoOp）
```

## 模式与文件对应关系

| 模式 | 调度构建器 | 解释器 | CUDA 目录 |
|------|-----------|--------|-----------|
| `latency` | `LatencyScheduleBuilder` | `LatencyMK_Interpreter` | `demos/low-latency-llama/` |
| `throughput` | `ThroughputScheduleBuilder` | `ThroughputMK_Interpreter` | (另一目录) |

`dispatch.py` 中的 `BUILDER_MAP` / `MK_INTERPRETER_MAP` 根据 `mode` 字符串路由到对应实现。

## 下一步

- [03-instructions.md](03-instructions.md) — 深入理解 7 个 opcode 和 Globals 的设计
- [04-scheduling.md](04-scheduling.md) — DAG 构建和 SM 分配算法
