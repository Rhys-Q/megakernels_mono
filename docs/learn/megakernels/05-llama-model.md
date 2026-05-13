# 05 — Llama 模型实现：权重堆叠、KV 缓存与张量并行

源文件：`megakernels/llama.py`

## 模型类层次

```
LlamaForCausalLM
  ├── LlamaModel
  │     ├── LlamaEmbeddings       (embed_tokens)
  │     ├── LlamaBlock × N        (transformer layers)
  │     │     ├── LlamaAttention  (self_attn)
  │     │     │     ├── RMSNorm   (input_layernorm)
  │     │     │     ├── q_proj, k_proj, v_proj  (Linear, no bias)
  │     │     │     └── o_proj    (Linear, no bias)
  │     │     └── LlamaMLP        (mlp)
  │     │           ├── RMSNorm   (input_layernorm)
  │     │           ├── up_proj, gate_proj  (Linear, no bias)
  │     │           └── down_proj (Linear, no bias)
  │     └── rope_cos, rope_sin    (预计算 RoPE 表，注册为 buffer)
  └── LlamaLMHead
        ├── RMSNorm               (input_norm)
        └── lm_head               (Linear, no bias)
```

## 关键设计决策

### 1. BatchState — 激活流传递

`BatchState` 是一个数据类，在 forward pass 中从上到下传递，每层把自己的输出写回去：

```python
# megakernels/model_types.py:21
@dataclass
class BatchState:
    input_ids: Tensor
    position_ids: Tensor | None = None
    seq_len: int | None = None
    hidden_states: Tensor | None = None       # 主激活张量
    position_embeddings: tuple | None = None  # (cos, sin) RoPE 嵌入
    output_ids: Tensor | None = None          # 最终输出 token
    # KV cache 索引（FlashInfer 使用，MK 模式不用）
    kv_indices, kv_indptr, kv_last_page_lens, ...
```

每个模块的 `forward()` 接收并返回同一个 `BatchState` 对象，在原地更新 `hidden_states`。

### 2. stack_params() — 权重堆叠

源文件：`megakernels/llama.py:713`

这是 Megakernels 最关键的一个预处理步骤。

**问题**：GPU 指令（如 `LayerNorm_QKV_MatVecRopeAppend`）需要以简单的整数索引定位权重。如果权重分散在 16 个（对于 16 层模型）独立的 `nn.Linear` 对象中，就无法通过一个整数（`layer_idx`）快速定位。

**解法**：在加载模型后，把所有层的对应权重沿新的第 0 维拼接（stack）成一个大张量：

```python
def stack_params(self):
    layers = self.model.layers

    # 1. q/k/v 先在每层内部 cat（[Hq*D + Hkv*D + Hkv*D, hidden]），再 stack
    qkv_weights = []
    for self_attn in [l.self_attn for l in layers]:
        cat_weight = torch.cat([self_attn.q_proj.weight,
                                 self_attn.k_proj.weight,
                                 self_attn.v_proj.weight], dim=0)
        qkv_weights.append(cat_weight)
    stacked_qkv = torch.stack(qkv_weights, dim=0)  # [L, qkv_dim, hidden]

    # 2. 其他权重直接 stack
    stacked_o_proj   = torch.stack([l.self_attn.o_proj.weight   for l in layers], dim=0)
    stacked_up_proj  = torch.stack([l.mlp.up_proj.weight        for l in layers], dim=0)
    stacked_gate_proj= torch.stack([l.mlp.gate_proj.weight      for l in layers], dim=0)
    stacked_down_proj= torch.stack([l.mlp.down_proj.weight      for l in layers], dim=0)
    stacked_attn_ln  = torch.stack([l.self_attn.input_layernorm.weight for l in layers], dim=0)
    stacked_mlp_ln   = torch.stack([l.mlp.input_layernorm.weight       for l in layers], dim=0)

    # 3. 保存到 self.stacked_params
    self.stacked_params = StackedParams(qkv_proj=stacked_qkv, ...)
```

**注意**：`stack_params()` 并不复制数据，而是用 `stack` 创建一个**视图**（view）。原始 `nn.Linear` 的权重仍然有效，只是 `stacked_params` 也指向同一块内存。这样既不浪费内存，又提供了按层索引的访问接口。

堆叠后的权重形状：

| 权重 | 形状 |
|------|------|
| `qkv_proj` | `[L, (Hq+2Hkv)*D, hidden]` |
| `o_proj` | `[L, hidden, Hq*D]` |
| `up_proj` / `gate_proj` | `[L, inter, hidden]` |
| `down_proj` | `[L, hidden, inter]` |
| `attn_ln_weight` / `mlp_ln_weight` | `[L, hidden]` |

### 3. setup_caches() — KV 缓存初始化

源文件：`megakernels/llama.py:552`

```python
def setup_caches(self):
    # 创建两个大张量：k_cache 和 v_cache
    shape = (num_layers, max_batch_size, max_seq_len, num_kv_heads, head_dim)
    k_cache = torch.zeros(shape, device=device, dtype=dtype)
    v_cache = k_cache.clone()

    self.stacked_kv_cache = (k_cache, v_cache)

    # 为每一层的 LlamaAttention 设置对应的 cache 切片（共享内存）
    for layer_idx in range(num_hidden_layers):
        layer.self_attn.kv_cache = (
            k_cache[layer_idx],   # shape: [max_batch, max_seq, num_kv_heads, head_dim]
            v_cache[layer_idx],
        )
```

这样 `Globals` 里的 `k_cache` 和 `v_cache` 就是整个 `stacked_kv_cache`，通过 `[layer_idx, ...]` 索引到特定层。

### 4. from_pretrained() — 完整加载流程

源文件：`megakernels/llama.py:583`

```python
@classmethod
def from_pretrained(cls, model_name_or_path, extra_config, device, dtype):
    # 1. 下载/解析模型配置
    config = LlamaConfig.from_pretrained(model_name_or_path)

    # 2. 用 accelerate 的 init_empty_weights 创建空壳模型（不分配 GPU 内存）
    with init_empty_weights():
        model = cls(config, extra_config)

    # 3. 加载 safetensors 权重（含 TP sharding）
    model.load_from_safetensors(model_path)

    # 4. 移动到目标设备
    model.to(device=device)

    # 5. 如果使用交错 RoPE，重新排列 Q/K 权重和 RoPE 表
    if extra_config.interleave_rope:
        model.model.interleave_rope()

    # 6. 堆叠权重（为 GPU 指令做准备）
    model.stack_params()

    # 7. 初始化 KV 缓存
    model.setup_caches()

    return model
```

---

## RoPE：标准模式 vs 交错模式

RoPE（旋转位置编码）有两种实现方式，由 `ExtraModelConfig.interleave_rope` 控制：

### 标准模式（`interleave_rope=False`）

使用 HuggingFace 原版实现：
```python
# 把向量前半部分旋转，后半部分旋转，再拼接
q_embed = (q * cos) + (rotate_half(q) * sin)
# rotate_half: 把后半维度取负，然后与前半维度拼接
```

### 交错模式（`interleave_rope=True`）

CUDA kernel 按奇偶维度交错处理，因此需要先把权重和 RoPE 表重排：

```python
def interleave_rope(self):
    # 对每个 head: [0,1,2,...,D-1] → [0, D/2, 1, D/2+1, 2, D/2+2, ...]
    # 把 "前半 + 后半" 布局改成 "交错对" 布局
    indices = [even/odd interleaved indices]
    self.rope_cos = self.rope_cos[..., indices]
    self.rope_sin = self.rope_sin[..., indices]
    # 同时对 q_proj 和 k_proj 的权重做对应重排
    mod.q_proj.weight[:] = mod.q_proj.weight[indices_for_q]
    mod.k_proj.weight[:] = mod.k_proj.weight[indices_for_k]
```

重排后，CUDA kernel 可以用 `(real, imag)` 连续内存对操作每个旋转单元，避免非连续内存访问。

---

## Tensor Parallelism (TP)

`ExtraModelConfig` 中的 TP 参数：
```python
tp_size: int = 1    # 总 GPU 数
tp_rank: int = 0    # 当前 GPU 编号
tp_group: ProcessGroup | None = None
```

### 权重分片（load_from_safetensors）

`make_tp_map()` 定义了每个参数在哪个维度分片：

| 参数 | 分片维度 | 说明 |
|------|---------|------|
| `q/k/v_proj.weight` | dim=0（输出维） | 每个 GPU 负责不同的 head |
| `up/gate_proj.weight` | dim=0（输出维） | 每个 GPU 负责不同的中间维度 |
| `o_proj.weight` | dim=1（输入维） | 输入来自各自 GPU 的 attention head |
| `down_proj.weight` | dim=1（输入维） | 输入来自各自 GPU 的 MLP 中间维度 |

加载时，`load_safetensors_repo` 按 `tp_rank` 切取对应分片。

### 通信原语

```python
# Attention 前：all_gather hidden_states（让每个 GPU 看到完整 hidden）
hidden_states = all_gather(hidden_states, extra_config)

# Attention 后：reduce_scatter o_proj 输出（每个 GPU 得到部分结果）
o_proj = reduce_scatter(o_proj, extra_config)
```

tp_size=1 时这两个函数直接返回输入，无通信开销。

---

## PyTorch Forward Pass 流程（完整 Decode 步骤）

```python
def forward(self, batch_state: BatchState):
    # 1. 查 embedding 表
    out = self.model.embed_tokens(batch_state)     # hidden_states = embed[input_ids]

    # 2. 预取 RoPE 嵌入（按 position_ids 索引）
    cos = self.model.rope_cos[position_ids]        # [batch, seq, head_dim]
    sin = self.model.rope_sin[position_ids]
    out.position_embeddings = (cos, sin)

    # 3. 依次经过 16 个 LlamaBlock
    for layer in self.model.layers:
        out = layer(out)
        # 每个 LlamaBlock 内部:
        #   hidden = rms_norm(hidden)
        #   q,k,v = qkv_proj(hidden)   # + RoPE
        #   k,v → kv_cache
        #   attn_out = softmax(q @ k.T / sqrt(D)) @ v
        #   hidden = hidden + o_proj(attn_out)   # 残差
        #   hidden = hidden + down_proj(silu(gate_proj(rms_norm(hidden))) * up_proj(rms_norm(hidden)))  # 残差

    # 4. 最终归一化 + LM Head
    out = self.lm_head(out)  # logits → argmax → output_ids
    return out
```

下一步：[06-execution-backends.md](06-execution-backends.md) — 三种执行后端的差异与用途
