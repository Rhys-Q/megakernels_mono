# 03 — 指令系统：Opcode、Globals 与序列化

## 设计思路

Megakernels 把 Transformer 的 forward pass 分解成一组**细粒度指令**。每条指令描述"哪个 SM 做什么计算"，并携带足够的参数让 GPU 在不返回 CPU 的情况下完成执行。

指令系统由两个核心部分组成：
1. **Globals**：所有指令共享的读写状态（权重、激活值、KV 缓存、barriers）
2. **Instruction**：描述单个计算任务的数据类，可序列化为整数数组

---

## BaseGlobals — 全局共享状态

源文件：`megakernels/instructions.py:10`

```python
@dataclass
class BaseGlobals:
    # ── 模型权重（所有层已堆叠，第 0 维是层索引）──
    qkv_proj_weights:   Tensor   # [L, (Hq+2Hkv)*D, hidden]
    attn_ln_weights:    Tensor   # [L, hidden]
    o_proj_weights:     Tensor   # [L, hidden, Hq*D]
    mlp_ln_weights:     Tensor   # [L, hidden]
    up_proj_weights:    Tensor   # [L, inter, hidden]
    gate_proj_weights:  Tensor   # [L, inter, hidden]
    down_proj_weights:  Tensor   # [L, hidden, inter]
    lm_head_norm_weights: Tensor # [hidden]
    lm_head_weights:    Tensor   # [vocab, hidden]

    # ── KV 缓存 ──
    k_cache: Tensor  # [L, 1, max_seq, Hkv, D]
    v_cache: Tensor  # [L, 1, max_seq, Hkv, D]

    # ── RoPE 预计算表 ──
    rope_cos: Tensor  # [max_seq, D]
    rope_sin: Tensor  # [max_seq, D]

    # ── 模型常量 ──
    num_hidden_layers: int
    num_attention_heads: int   # Hq（query heads）
    num_kv_heads: int          # Hkv（key/value heads，GQA 时 < Hq）
    head_dim: int              # D
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    attn_scale: float          # 1/sqrt(D)
    rms_norm_eps: float

    # ── 激活缓冲（每步 Decode 复用）──
    hidden_states: Tensor      # [hidden_size]  主激活流，贯穿整个 forward
    barriers: Tensor           # [L, 10, Hq+2Hkv]  跨指令同步计数器

    # ── 运行时 ──
    pos_id: int                # 当前生成位置（token 序号）
    instructions: Tensor       # [num_sms, max_queue, 32]  指令表（tensorize 后填充）
    timings: Tensor            # [num_sms, max_queue, 128]  性能计时（可选）
```

`Globals`（latency 模式）在此基础上增加了激活缓冲和 block 大小常量：

```python
# megakernels/demos/latency/instructions.py:10
@dataclass
class Globals(BaseGlobals):
    post_ln_rope_q: Tensor          # [hidden_size]  Q 经过 RoPE 后的结果
    attn_out: Tensor                # [hidden_size]  attention 输出
    attn_lse_intermediates: Tensor  # [Hq, max_partitions]  log-sum-exp 中间值
    attn_out_intermediates: Tensor  # [Hq, max_partitions, D]  attention 分块中间值
    silu_out: Tensor                # [inter_size]   MLP 中间激活
    logits: Tensor                  # [vocab_size]   最终输出

    skip_attn_reduction: bool       # True = 不需要 reduce（只有 1 个 partition）

    # block 大小（决定每条指令处理多少行/列）
    up_gate_proj_block_size: int   # = 16
    down_proj_block_size: int      # = 16
    qkv_block_size: int            # = 16
    o_proj_block_size: int         # = 16
    lm_head_block_size: int        # = 16
    matvec_reduction_size: int     # = 2048（reduction 分组大小）
    attn_kv_block_size: int        # = 16（每个 attention 分块处理的 token 数）
    attn_reduction_size: int       # = 4
```

**为什么要把所有层的权重堆叠（stack）在一起？**

GPU 做矩阵向量乘法（matvec）时，每条指令只处理某一层的一个小分块（block）。如果权重分散在 N 个独立张量中，指令序列化时就无法用简单的整数索引定位到正确的权重切片。堆叠后，用 `weights[layer_idx][start:end]` 就能直接寻址。

---

## Instruction 基类

源文件：`megakernels/instructions.py:84`

```python
@dataclass
class Instruction:
    @classmethod
    def opcode(cls) -> int:
        """唯一标识这类操作，0 = NoOp"""

    @classmethod
    def prev_opcode(cls) -> int:
        """前驱操作的 opcode（barrier 检查依据）"""

    def cost(self, globs: BaseGlobals) -> float:
        """估算计算代价，调度器用此做负载均衡"""

    def serialize(self) -> list[int]:
        """序列化为整数列表，最终 padding 到 32 个 int"""
```

序列化格式（`serialize()` 的实现）：
```
[opcode, field1, field2, ..., 0, 0, ...]  # 总共 32 个 int
```

对于 `list` 和 `tuple` 类型的字段，先序列化长度，再序列化每个元素：
```
[opcode, ..., len(my_list), elem0, elem1, ..., padding]
```

---

## Latency 模式的 7 个 Opcode

latency 模式（`megakernels/demos/latency/instructions.py`）实现了完整的 Llama Decode 所需的 7 种操作：

### Opcode 1 — LayerNorm_QKV_MatVecRopeAppend

```python
@dataclass
class LayerNorm_QKV_MatVecRopeAppend(Instruction):
    layer_idx: int
    start_output_block_idx: int   # 负责输出向量的起始 block
    end_output_block_idx: int     # 负责输出向量的终止 block
```

**功能**：对 `hidden_states` 做 RMS LayerNorm，再对 Q/K/V 各自做矩阵向量乘，对 Q 和 K 施加 RoPE，最后把 K 和 V 写入 KV cache。

**分块逻辑**：QKV 输出维度 = `(Hq + 2×Hkv) × D`，按 `qkv_block_size`（=16）分成若干 block，每条指令负责其中一段（`[start_block, end_block)`）。多个 SM 各自负责不同的输出段，并行完成整个 QKV 矩阵乘。

**代价模型**：`(end - start) × qkv_block_size × hidden_size`

---

### Opcode 2 — PartialAttention

```python
@dataclass
class PartialAttention(Instruction):
    layer_idx: int
    kv_head_idx: int    # 负责哪个 KV head
    num_partials: int   # 总共分了几个 partition
    partial_idx: int    # 本指令负责哪个 partition
```

**功能**：对指定 KV head 的 attention 做部分计算（KV cache 中的一段 token 范围）。计算 softmax、log-sum-exp（LSE），输出部分注意力结果。

**为什么要分 Partition？**  
KV cache 随序列长度增长。当序列很长时，一条指令处理全部 KV 会很慢。Megakernels 把 KV cache 沿序列维度切成 `num_partitions` 段，每段由一条指令处理，多个 SM 并行。最后用 `AttentionReduction` 归并结果。

**代价模型**：`seq_len / num_partials × head_dim × 2`（分别为 K 和 V）

---

### Opcode 3 — AttentionReduction

```python
@dataclass
class AttentionReduction(Instruction):
    layer_idx: int
    head_start_idx: int     # 负责哪批 head（每次处理 attn_reduction_size 个）
    num_partials: int       # 原始 partition 数量
    is_terminal: bool       # True = 最终结果直接写入 attn_out
    reduction_list: list[int]  # 要 reduce 的 partition 索引
    output_partial_idx: int | None  # 非终结时，写到哪个中间 slot
```

**功能**：把多个 `PartialAttention` 的结果用 log-sum-exp 技巧归并成最终 attention 输出。

**归并算法**（在线 softmax）：
```
max_lse = max(lse_i for each partition)
scale_i = exp2(lse_i - max_lse)
out = sum(out_i × scale_i) / sum(scale_i)
```

当 `skip_attn_reduction=True`（只有 1 个 partition）时，`PartialAttention` 直接写 `attn_out`，跳过这一步。

---

### Opcode 4 — O_ProjResidual

```python
@dataclass
class O_ProjResidual(Instruction):  # 继承自 MatVecAdd
    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int   # 当前从哪个 reduction chunk 读入
```

**功能**：`attn_out` 经过输出投影矩阵（O-proj），结果加到 `hidden_states`（残差连接）。

**数据依赖**：需要等待所有 `AttentionReduction` 完成（barrier 检查 `barriers[layer, opcode3-1][0] == num_attention_heads`）。

---

### Opcode 5 — LayerNormDoubleMatVecSiLU

```python
@dataclass
class LayerNormDoubleMatVecSiLU(Instruction):
    layer_idx: int
    block_idxs: list[int]   # 负责的 MLP intermediate 维度 block 索引列表
```

**功能**：对更新后的 `hidden_states` 做 RMS LayerNorm，然后同时做 `up_proj` 和 `gate_proj` 两个矩阵向量乘，最后 `SiLU(gate) × up` 写入 `silu_out`。

**为什么是"Double"？**  
SwiGLU 激活需要两路投影，这里把它们融合成一条指令减少内存访问。

**分块逻辑**：每个 SM 负责若干个 `block_idxs`（轮转分配，非连续），比其他 opcode 的连续区间分配更分散，有利于负载均衡。

---

### Opcode 6 — DownProjResidual

```python
@dataclass
class DownProjResidual(Instruction):  # 继承自 MatVecAdd
    layer_idx: int
    start_block_idx: int
    end_block_idx: int
    reduction_block_idx: int  # 读取 silu_out 的哪个 chunk（因 intermediate > hidden）
```

**功能**：`silu_out` 经过 `down_proj` 矩阵，结果加到 `hidden_states`（残差连接）。

**为什么有 `reduction_block_idx`？**  
`down_proj` 的输入维度是 `intermediate_size`（如 8192），输出是 `hidden_size`（如 2048）。当 `intermediate > hidden` 时，需要先用 reduction 把输入分成多段再累加，`reduction_block_idx` 指定当前是哪一段。

---

### Opcode 7 — RMS_LM_Head

```python
@dataclass
class RMS_LM_Head(Instruction):
    start_output_block_idx: int
    end_output_block_idx: int
```

**功能**：对最后一层的 `hidden_states` 做最终 RMS LayerNorm，再做 `lm_head` 矩阵乘，输出 logits。

**只在最后一层之后执行**：`make_dag()` 在完成所有 `num_hidden_layers` 层的循环后，统一创建 `RMS_LM_Head` 指令节点。

---

## Opcode 依赖关系图

```
                    Opcode 1 (QKV)
                        │
                    Opcode 2 (PartialAttention) ─ × num_kv_heads
                        │
                    Opcode 3 (AttentionReduction)  [可选，skip_attn_reduction=True 时省略]
                        │
                    Opcode 4 (O_ProjResidual)
                        │
                    Opcode 5 (UpGateSiLU)
                        │
                    Opcode 6 (DownProjResidual)
                        │
              (下一层的 Opcode 1, 或...)
                    Opcode 7 (LM_Head)  [最后一层后]
```

每个 opcode 通过 `prev_opcode()` 声明自己的前驱，PyVM 和 CUDA kernel 在执行前都会检查 barriers 计数是否满足条件。

---

## Barriers 机制

`globs.barriers` 的形状是 `[num_layers, 10, num_heads_total]`，用作原子计数器。

每条指令执行完毕后，将对应维度的 barrier 值加上它完成的 block 数量：
```python
# 示例：O_ProjResidual 完成后
barriers = globs.barriers[layer_idx, opcode - 1]
barriers[0] += end_block_idx - start_block_idx
```

下一条依赖指令在执行前检查：
```python
# 示例：LayerNormDoubleMatVecSiLU 开始前
prev_barriers = globs.barriers[layer_idx, prev_opcode - 1]
assert prev_barriers[0] == 128  # 需要 128 个 o_proj block 全部完成
```

这个简单的计数器机制替代了复杂的锁，依赖 PyVM 单线程执行的顺序保证，而 CUDA 版本则使用 semaphore 实现。

---

## 序列化示例

以 `LayerNorm_QKV_MatVecRopeAppend(layer_idx=0, start=0, end=3)` 为例：

```
serialize() 输出：
[1,   ← opcode=1
 0,   ← layer_idx=0
 0,   ← start_output_block_idx=0
 3,   ← end_output_block_idx=3
 0, 0, 0, ..., 0]   ← padding 到 32 个 int
```

所有指令统一 padding 到 32 int，便于 GPU 用固定步长读取指令表。

下一步：[04-scheduling.md](04-scheduling.md) — 如何把指令组织成 DAG 并分配到各 SM
