# 08 — Demo 完整步骤分析：从加载模型到生成一个 Token

本章以 `meta-llama/Llama-3.2-1B-Instruct`（16 层，hidden_size=2048）为例，**逐步追踪** PyVM 模式下生成第一个 Decode token 的完整执行过程。

---

## 模型规格（Llama-3.2-1B）

| 参数 | 值 |
|------|---|
| `num_hidden_layers` | 16 |
| `hidden_size` | 2048 |
| `intermediate_size` | 8192 |
| `num_attention_heads` (Hq) | 32 |
| `num_key_value_heads` (Hkv) | 8 |
| `head_dim` (D) | 64 |
| `vocab_size` | 128256 |
| H100 SM 数 | 132 |

---

## Step 1: 加载模型

```python
from megakernels.llama import LlamaForCausalLM
from megakernels.model_types import ExtraModelConfig

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device="cuda:0",
    dtype=torch.bfloat16,
    extra_config=ExtraModelConfig(interleave_rope=True),
)
```

`from_pretrained()` 内部执行的事情：

1. **下载并解析配置**：`LlamaConfig` 包含所有超参数
2. **创建空壳模型**：`init_empty_weights()` 创建模型结构但不分配 GPU 内存
3. **加载 safetensors**：从磁盘读取权重，自动映射 HF 参数名到内部参数名（如 `model.layers.0.input_layernorm.weight` → `model.layers.0.self_attn.input_layernorm.weight`）
4. **重排 RoPE 表和权重**（因为 `interleave_rope=True`）：把 Q/K 权重从 `[even, odd]` 布局改为交错布局，配合 CUDA kernel 的内存访问模式
5. **堆叠权重**：`stack_params()` 创建 `stacked_params`
   - `stacked_params.qkv_proj` shape: `[16, (32+8+8)*64, 2048]` = `[16, 3072, 2048]`
   - `stacked_params.o_proj` shape: `[16, 2048, 2048]`
   - `stacked_params.up_proj` / `gate_proj` shape: `[16, 8192, 2048]`
   - `stacked_params.down_proj` shape: `[16, 2048, 8192]`
6. **初始化 KV 缓存**：`setup_caches()` 分配 `k_cache` 和 `v_cache`
   - shape: `[16, 1, 2048, 8, 64]`（16层, batch=1, max_seq=2048, 8 KV heads, head_dim=64）

---

## Step 2: Prefill（处理 Prompt）

假设 prompt 有 50 个 token，tokenize 后得到 `input_ids: [50]`。

```python
# Prefill 使用 PyTorch Generator 做完整 forward pass
prefill_state = BatchState(
    input_ids=input_ids.unsqueeze(0),  # [1, 50]
    position_ids=torch.arange(50).unsqueeze(0),  # [1, 50]
    seq_len=50,
)
output = model(prefill_state)
output_tokens[0, 1] = output.output_ids[0, -1]  # 保存第 50+1=51 个 token
```

Prefill 期间，每一层的 `attention()` 调用会把 K 和 V 写入 KV cache（位置 0..49）：
```python
k_cache[layer_idx, 0, :50] = key_states    # shape: [50, 8, 64]
v_cache[layer_idx, 0, :50] = value_states  # shape: [50, 8, 64]
```

---

## Step 3: 构建 Schedule（仅需一次）

```python
from megakernels.dispatch import make_schedule_builder

builder = make_schedule_builder("latency")
schedule = builder.build(model)
```

### 3.1 make_globals()

分配激活缓冲：
```
hidden_states:           Tensor[2048]         bfloat16
post_ln_rope_q:          Tensor[2048]         bfloat16  (= 32头 × 64维)
attn_out:                Tensor[2048]         bfloat16
attn_out_intermediates:  Tensor[32, 132, 64]  float32   (32 Q-heads, 132 partitions max, 64 head_dim)
attn_lse_intermediates:  Tensor[32, 132]      float32
silu_out:                Tensor[8192]         bfloat16
logits:                  Tensor[128256]       bfloat16
barriers:                Tensor[16, 10, 48]   int32     (16层, 10 opcodes, 32+8+8=48 heads)
```

### 3.2 make_dag()

对每一层调用 `make_dag_layer()`，以**第 0 层**为例（其余层类似）：

**QKV 节点**（`schedule_qkv()`）：

QKV 输出维度 = (32+8+8) × 64 = 3072，按 `qkv_block_size=16` 分成 192 个 block。
分配给 132 个 SM，每 SM 约 192/132 ≈ 1.45 个 block（最终取整分配，SM 0 得 2 blocks，SM 100+ 得 1 block）：

```
SM 0:  LayerNorm_QKV_MatVecRopeAppend(layer=0, start=0, end=2)
SM 1:  LayerNorm_QKV_MatVecRopeAppend(layer=0, start=2, end=4)
...
SM 131: LayerNorm_QKV_MatVecRopeAppend(layer=0, start=190, end=192)
```

共 132 个 QKV 节点。

**PartialAttention 节点**（`skip_attn_reduction=True` → `num_partials=1`）：

每个 KV head 对应一条 PartialAttention 指令（因 num_partials=1，不分 partition）：

```
DAG_Node(PartialAttention(layer=0, kv_head=0, num_partials=1, partial=0),
         deps=[产生 K-head-0 和 V-head-0 的 QKV 节点])
DAG_Node(PartialAttention(layer=0, kv_head=1, ...), deps=[...])
...
DAG_Node(PartialAttention(layer=0, kv_head=7, ...), deps=[...])
```

共 8 个 PartialAttention 节点，每个只依赖产生对应 K/V 块的 QKV 节点（精细依赖，不等全部 QKV 完成）。

**O_ProjResidual 节点**：

输出维度 = 2048，按 `o_proj_block_size=16` 分成 128 个 block，每个 block 一条指令：

```
O_ProjResidual(layer=0, start=0, end=1, reduction=0)
O_ProjResidual(layer=0, start=1, end=2, reduction=0)
...
O_ProjResidual(layer=0, start=127, end=128, reduction=0)
```

共 128 个节点，全部依赖所有 8 个 PartialAttention 节点。

**UpGateSiLU 节点**（`schedule_upgate()`）：

intermediate_size = 8192，按 `up_gate_proj_block_size=16` 分成 512 个 block，按**轮转**分配给 132 个 SM：

```
SM 0:  LayerNormDoubleMatVecSiLU(layer=0, block_idxs=[0, 132, 264, 396])
SM 1:  LayerNormDoubleMatVecSiLU(layer=0, block_idxs=[1, 133, 265, 397])
...
```

共 132 个节点，依赖所有 128 个 O_ProjResidual 节点。

**DownProjResidual 节点**（`schedule_downproj()`）：

输出维度 = 2048/16 = 128 blocks，输入维度 = 8192/2048 = 4 个 reduction chunk：

```
SM 0:  DownProjResidual(layer=0, start=0, end=1, reduction=0)
SM 1:  DownProjResidual(layer=0, start=1, end=2, reduction=0)
...
SM 127: DownProjResidual(layer=0, start=127, end=128, reduction=0)
SM 128: DownProjResidual(layer=0, start=0, end=1, reduction=1)
...
```

**每一层总指令数**：132 + 8 + 128 + 132 + 132 = **532 条**

**全部 16 层 + LM Head（132条）**：532×16 + 132 = **8644 条**指令

---

## Step 4: 分配到 SM（wave 策略）

```python
from megakernels.scheduler import assign_to_sms

assigned = assign_to_sms("wave", schedule=schedule)
# assigned: list[list[Instruction]], len=132
```

Wave 策略按 opcode 分组：
- Wave 1（opcode 1）：132 个 QKV 指令（层 0）→ 每 SM 1 条
- Wave 2（opcode 2）：8 个 PartialAttention → 8 个 SM 各 1 条，其余 SM 空
- Wave 3（opcode 4）：128 个 O_ProjResidual → 大多数 SM 1 条
- Wave 4（opcode 5）：132 个 UpGate → 每 SM 1 条
- Wave 5（opcode 6）：132 个 DownProj → 每 SM 1 条
- Wave 6（opcode 1）：132 个 QKV（层 1）→ 每 SM 1 条
- ... 重复 16 层 ...
- 最后 Wave：132 个 LM_Head

每个 SM 的最终队列长度约为 `8644 / 132 ≈ 65` 条指令。

---

## Step 5: 张量化

```python
from megakernels.scheduler import tensorize_instructions

tensorize_instructions(schedule.globs, assigned)
# schedule.globs.instructions: Tensor[132, 65, 32]  int32
# schedule.globs.timings:      Tensor[132, 65, 128] int32
```

示例：SM 0 的第 0 条指令（QKV 层 0）：
```
instructions[0, 0, :] = [1, 0, 0, 2, 0, 0, ..., 0]
                         ^  ^  ^  ^
                         |  |  |  end_output_block_idx=2
                         |  |  start_output_block_idx=0
                         |  layer_idx=0
                         opcode=1
```

---

## Step 6: 创建生成器并执行 Decode

```python
from megakernels.dispatch import make_pyvm_interpreter
from megakernels.generators import PyVM_Generator

interpreter = make_pyvm_interpreter("latency")
generator = PyVM_Generator(model, interpreter, schedule)
```

执行第一个 Decode 步骤（位置 50，生成 token 51）：

```python
generator.run(input_ids=output_tokens[:, 1:2], pos_id=50)
```

### 6.1 Embedding Lookup

```python
hidden = model.model.embed_tokens(batch_state)
# hidden.hidden_states: Tensor[1, 1, 2048]  → squeeze → [2048]
schedule.globs.hidden_states[:] = hidden.hidden_states.squeeze(1)
# 现在 globs.hidden_states 存放了第 51 个 token 的初始 embedding
```

### 6.2 重置 Barriers 和位置

```python
schedule.globs.barriers.zero_()
schedule.globs.pos_id = 50
```

### 6.3 PyVM 执行指令序列

`interpreter.interpret(globs, instructions)` 按拓扑顺序逐条执行：

---

**第 1 条：`LayerNorm_QKV_MatVecRopeAppend(layer=0, start=0, end=2)`**

Solver：`layer_norm_matvec_rope_append(globs, instruction)`

```
输入:  globs.hidden_states[0:2048]
操作:
  1. RMS LayerNorm:
     post_ln = hidden_states / rms(hidden_states) * attn_ln_weights[0]
     → post_ln: Tensor[2048]

  2. 对 block 0 (维度 0..15):
     mode = "q"  (0 < k_start=2048)
     matmul_out = qkv_weights[0][0:16] @ post_ln   → [16]
     apply RoPE (interleaved):
       full_head = zeros[64]; full_head[0:16] = matmul_out
       apply_rotary_pos_emb_interleaved(full_head, rope_cos[50], rope_sin[50])
       out = full_head_with_rope[0:16]
     globs.post_ln_rope_q[0:16] = out

  3. 对 block 1 (维度 16..31):
     同上，写入 globs.post_ln_rope_q[16:32]

  4. Barrier 更新:
     barriers[0, 0][0] += 2   # block 0,1 已完成（block_idx // 4 = 0）
```

... 继续 130 条类似的 QKV 指令（SM 1..131 的部分）...

执行完所有 QKV 指令后：
- `globs.post_ln_rope_q[0:2048]`：Q 向量，含 RoPE，完整
- `globs.k_cache[0, 0, 50, 0:8, 0:64]`：当前 token 的 K，已写入 KV cache
- `globs.v_cache[0, 0, 50, 0:8, 0:64]`：当前 token 的 V，已写入 KV cache
- `barriers[0, 0][...]`：各段已更新

---

**第 9 条（开始）：`PartialAttention(layer=0, kv_head=0, num_partials=1, partial=0)`**

Solver：`partial_attention(globs, instruction)`

```
输入:
  q: globs.post_ln_rope_q[0:2048].view(32, 64)[0:4]  # GQA ratio=32/8=4，head 0..3
     shape: [4, 64]
  k: globs.k_cache[0, 0, 0:51, 0]   # seq_len = pos_id+1 = 51，kv_head=0
     shape: [51, 64]
  v: globs.v_cache[0, 0, 0:51, 0]
     shape: [51, 64]

操作:
  qk = q @ k.T / sqrt(64)    → [4, 51]
  softmax = softmax(qk, dim=-1)
  lse = log2(sum(exp(qk), dim=-1))
  out = softmax @ v            → [4, 64]

  skip_attn_reduction=True →
    globs.attn_out.view(32, 64)[0:4] = out  # 直接写最终 attention 输出
    barriers[0, 2][0] += 4   # opcode 3 的 barrier（0-indexed: opcode3=3，所以 barriers[layer, 2]）
```

... 8 条 PartialAttention（kv_head 0..7）全部完成 ...

执行后 `globs.attn_out[0:2048]`：完整的 attention 输出，含所有 32 个 Q-head 的结果。

---

**第 17 条（开始）：`O_ProjResidual(layer=0, start=0, end=1, reduction=0)`**

Barrier 检查：`barriers[0, 2][0] == 32`（32 个 attention head 全完成）→ 通过

```
操作:
  matvec_out = o_proj_weights[0][0:16, 0:2048] @ attn_out[0:2048]  → [16]
  hidden_states[0:16] += matvec_out
```

... 128 条 O_ProjResidual 全部完成 ...

执行后 `globs.hidden_states`：加上了 attention 贡献的残差。

---

**第 145 条：`LayerNormDoubleMatVecSiLU(layer=0, block_idxs=[0, 132, 264, 396])`**

```
操作:
  post_ln = rms_norm(hidden_states, mlp_ln_weights[0])  → [2048]

  for block_idx in [0, 132, 264, 396]:
    start = block_idx * 16
    up_out  = up_proj_weights[0][start:start+16] @ post_ln   → [16]
    gate_out= gate_proj_weights[0][start:start+16] @ post_ln → [16]
    silu_out[start:start+16] = silu(gate_out) * up_out
```

... 132 条 UpGateSiLU 全部完成 ...

执行后 `globs.silu_out[0:8192]`：MLP 中间激活（SwiGLU 输出）。

---

**第 277 条：`DownProjResidual(layer=0, start=0, end=1, reduction=0)`**

```
操作:
  chunk = silu_out[0:2048]  # reduction_block_idx=0，取前 2048 维
  matvec_out = down_proj_weights[0][0:16, 0:2048] @ chunk  → [16]
  hidden_states[0:16] += matvec_out  # 残差加法
```

... 132 条 DownProjResidual 全部完成 ...

**第 0 层完成！**`globs.hidden_states` 已更新为经过第 0 层 Transformer 处理后的激活值。

---

**重复 15 次**（层 1..15 的指令），每次都：
1. 读取上一层输出的 `hidden_states`
2. 执行 QKV + Attention + O_Proj + MLP
3. 输出新的 `hidden_states`

---

**最后：`RMS_LM_Head(start=0, end=k)`（132 条）**

```
操作（以 start=0, end=16 为例）:
  post_ln = rms_norm(hidden_states, lm_head_norm_weights)  → [2048]
  logits[0:16] = lm_head_weights[0:16] @ post_ln          → [16]
```

... 所有 LM Head 指令完成 ...

`globs.logits[0:128256]`：完整 logits 向量。

---

### 6.4 取 argmax → 输出 token

```python
output_id = torch.argmax(globs.logits)
# 例如：output_id = 29871 (对应 "▁" 即空格)
output_tokens[0, 2] = output_id
```

**整个 Decode 步骤完成！** 从 token 51 的 embedding 到 token 52 的预测，共执行了约 8644 条指令。

---

## 数据流汇总图

```
input_ids[51]
    │
    ▼
embed_tokens → hidden_states[2048]
    │
    │ (重复 16 层)
    ▼
┌─────────────────────────────────────────────────────────┐
│ Layer N                                                  │
│                                                          │
│ hidden_states ──→ rms_norm ──→ QKV matmul               │
│                                     │                    │
│                   post_ln_rope_q ←──┤ Q (+ RoPE)        │
│                   k_cache[N,50] ←──┤ K (+ RoPE)        │
│                   v_cache[N,50] ←──┘ V                  │
│                       │                                  │
│        (k_cache[N,0:51], v_cache[N,0:51])                │
│                       │                                  │
│                   Attention ───→ attn_out[2048]          │
│                       │                                  │
│ hidden_states ←── O_Proj + residual ──────────────────┐  │
│     │                                                  │  │
│     └→ rms_norm ──→ up_proj ─→ silu(gate) × up        │  │
│                   gate_proj ─↗        │                │  │
│                                   silu_out[8192]       │  │
│                                       │                │  │
│ hidden_states ←── down_proj + residual ────────────────┘  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
rms_norm + lm_head → logits[128256]
    │
    ▼
argmax → next_token_id
```

---

## 关键数字总结

| 步骤 | 描述 | 数量 |
|------|------|------|
| QKV 指令（每层） | 132（每 SM 1 条） | ×16层 = 2112 |
| PartialAttention（每层） | 8（每 KV head 1 条） | ×16层 = 128 |
| O_ProjResidual（每层） | 128（每 output block 1 条） | ×16层 = 2048 |
| UpGateSiLU（每层） | 132 | ×16层 = 2112 |
| DownProjResidual（每层） | 132 | ×16层 = 2112 |
| LM_Head | 132（vocab/SM）| 1次 = 132 |
| **总计** | | **~8644 条** |

每个 SM 平均执行约 **65 条**指令，完成整个 forward pass 的 1/132 计算量。

---

## 下一步

恭喜！你已经完整追踪了一次 Decode 步骤的执行路径。

进一步探索：
- **修改 block 大小**：调整 `qkv_block_size` 等常量观察指令数量变化
- **实现新 opcode**：参考 `demos/latency/instructions.py` 的结构，添加自定义融合算子
- **比较调度策略**：用 `bench_engines.py` 对比 `rr`、`wave`、`dag` 策略的性能差异
- **CUDA 优化**：研究 `attention_partial.cu` 的流水线实现，理解 TMA 和在线 softmax 的结合
