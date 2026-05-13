# Decode Kernel 深入分析

本文逐步解析 `kernel.cu` 中 `decode_kernel` 的完整执行流程，以及 Python 端如何调用它。

---

## Python 端：从 `step()` 到 CUDA

### 1. 初始化（`Decoder.__init__`）

```python
decoder = Decoder()  # 自动加载 Qwen3.5-0.8B
```

初始化时做了几件事：

**a) 加载权重**
```python
weights, tokenizer = load_weights("Qwen/Qwen3.5-0.8B")
```
从 HuggingFace 下载模型，转为 BF16，存到 CUDA 设备上。

**b) 打包权重指针**
```python
self._layer_weights_packed = _pack_layer_weights(weights["layer_data"])
```
把每层的权重 tensor 的**设备指针**（8字节整数）打包成一个连续的 bytes 块，传给 CUDA kernel。

```python
def _pack_layer_weights(layer_data):
    struct_size = 16 + 14 * 8  # 头部16字节 + 最多14个指针
    buf = bytearray(24 * struct_size)  # 24层
    for i, ld in enumerate(layer_data):
        # 写入 layer_type（DeltaNet=0, FullAttn=1）
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        # 写入各张量的 CUDA 设备指针
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + 16 + j*8, tensor.data_ptr())
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()
```

**c) 预分配缓冲区**
```python
# KV Cache（用于 Full Attention 层）
self._fa_k_cache = torch.zeros(6, 2, 2048, 256, dtype=bf16)  # [n_fa_layers, KV_heads, max_seq, head_dim]
self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

# DeltaNet 状态矩阵
self._dn_states = torch.zeros(18, 16, 128, 128, dtype=f32)  # [n_dn_layers, heads, key, val]
self._conv_bufs = torch.zeros(18, 6144, 4, dtype=f32)       # [n_dn_layers, channels, kernel_size]

# 激活值缓冲区
self._hidden = torch.empty(1024, dtype=bf16)  # 当前 token 的隐向量
self._activations = torch.empty(max_scratch, dtype=f32)
# ... 更多临时缓冲区
```

**d) 预融合权重（用于 prefill 加速）**
```python
# 将 FA 层的 Q/K/V 权重拼接（后续 prefill 用 cuBLAS 单次 GEMM 搞定）
fa_qkv_list = [torch.cat([q, k, v], dim=0) for each FA layer]
self._fused_fa_qkv = torch.stack(fa_qkv_list).contiguous()

# 将 MLP 的 gate/up 权重拼接
gate_up_list = [torch.cat([gate, up], dim=0) for each layer]
self._fused_gate_up = torch.stack(gate_up_list).contiguous()
```

---

### 2. 单步解码（`step(token_id)`）

```python
def step(self, token_id: int) -> int:
    _decode(
        self._out_token, token_id,
        self._embed_weight, self._layer_weights_packed,
        self._final_norm_weight, self._lm_head_weight,
        self._fa_k_cache, self._fa_v_cache,
        self._dn_states, self._conv_bufs,
        self._hidden, self._activations, self._residual,
        # ... 更多缓冲区
        self._position, self.max_seq_len,
    )
    self._position += 1
    return self._out_token.item()  # 从 GPU 读回 next token id
```

`_decode` 是通过 `torch.ops.qwen35_megakernel_bf16_C.decode` 注册的 CUDA 函数。

---

## CUDA 端：`launch_decode`

```cpp
extern "C" void launch_decode(..., int position, int max_seq_len, cudaStream_t stream) {
    // 1. 确定安全的 block 数（不超过 GPU 的驻留上限）
    int decode_blocks = min(NUM_BLOCKS, max_safe_blocks);

    // 2. 重置栅栏计数器
    cudaMemsetAsync(barrier_counter, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(barrier_generation, 0, sizeof(unsigned int), stream);

    // 3. 启动 decode kernel（唯一一次 CPU→GPU 调度）
    decode_kernel<<<decode_blocks, BLOCK_SIZE, 0, stream>>>(...);

    // 4. 启动 lm_head kernel（另一次调度，计算词表 logits）
    cudaMemsetAsync(lm_sync_counter, 0, sizeof(unsigned int), stream);
    lm_head_kernel<<<LM_NUM_BLOCKS, LM_BLOCK_SIZE, 0, stream>>>(...);
}
```

这里有 **两次** kernel 调度：
1. `decode_kernel`：处理 24 层（embeding → 所有层 → final norm）
2. `lm_head_kernel`：词表投影（248320 × 1024 的大矩阵乘法）

为什么 lm_head 独立？因为词表很大（248320 个词），需要更多的并行度，用 512 个 block 更合适，而不是 decode 用的 82 个。

---

## `decode_kernel`：主 Kernel 逻辑

```cpp
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel(... int input_token_id, int position, int max_seq_len)
{
    // 初始化栅栏
    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)gridDim.x, 0};

    // 共享内存（每个 block 独立，不跨 block 共享）
    __shared__ __align__(16) char shmem_raw[MAX_ACT_DIM * sizeof(float)];
    half_t *shmem_bf16 = reinterpret_cast<half_t *>(shmem_raw);

    // Token Embedding：取词嵌入矩阵的第 input_token_id 行
    const half_t *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

    // 记录 token 用于 repetition penalty
    if (block_id == 0 && threadIdx.x == 0) {
        seen_token_mask[input_token_id] = 1.0f;
    }

    int dn_layer_idx = 0, fa_layer_idx = 0;

    // 遍历所有 24 层
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const half_t *layer_input = (layer == 0) ? embed_row : hidden_buffer;

        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(grid, ..., layer_input, ..., dn_layer_idx, ...);
            dn_layer_idx++;
        } else {
            full_attention_layer(grid, ..., layer_input, ..., position, max_seq_len, ...);
            fa_layer_idx++;
        }
    }

    // Final RMSNorm（只有 block 0 做）
    if (block_id == 0) {
        // 计算最后一层输出的 RMSNorm，写入 g_normalized
        // g_normalized 作为 lm_head_kernel 的输入
    }
}
```

---

## 数据流追踪

以处理 token "好" 为例（假设 token_id = 12345）：

```
输入：token_id = 12345, position = 3（已有3个token）

Step 1: Token Embedding
  embed_row = embed_weight[12345, :] → 1024维 BF16 向量

Step 2: Layer 0（DeltaNet）
  embed_row → RMSNorm(s_norm)
  s_norm → qkv_proj → g_qkv [6144]   Conv1d → Q[2048], K[2048], V[2048]
  s_norm → z_proj   → g_z [2048]
  s_norm → beta_proj → g_beta [16]   sigmoid → 更新强度
  s_norm → alpha_proj → g_alpha [16] softplus+exp → decay

  对每个头 h=0..15（block h 负责）：
    Conv1d(conv_buf[h], Q[h]) → s_q [128]   (SiLU)
    Conv1d(conv_buf[h], K[h]) → s_k [128]   (SiLU)
    Conv1d(conv_buf[h], V[h]) → s_v [128]   (SiLU)
    L2归一化 s_q, s_k
    DeltaNet 递推：
      kq = s_k · s_q
      stk = state[h] · s_k
      sqv = state[h] · s_q
      error = (s_v - decay * stk) * beta
      output[h] = decay * sqv + error * kq
      state[h] = state[h] * decay + outer(s_k, error)
    RMSNorm(output[h]) * SiLU(z[h]) → g_dn_out[h]

  g_dn_out → out_proj → hidden_buffer [1024 BF16]
  hidden_buffer += residual
  hidden_buffer → post_norm → MLP → hidden_buffer

  grid.sync()

Step 3: Layer 1（DeltaNet，类似）
Step 4: Layer 2（DeltaNet，类似）

Step 5: Layer 3（Full Attention）
  hidden → RMSNorm
  → q_proj → g_q [2048]（含gate向量）
  → k_proj → g_kv[:512]
  → v_proj → g_kv[512:]
  grid.sync()

  block_0: k_norm(g_kv[:]) + RoPE → k_cache[fa_layer=0][pos=3]
  block_0: v_cache[fa_layer=0][pos=3] = g_kv[512:]
  all blocks: q_norm(g_q[:]) + RoPE

  grid.sync()

  计算注意力（查询所有 pos=0,1,2,3 的K/V）：
    score[pos] = Q · K[pos] / sqrt(256)
    online softmax → weights
    out = sum(weight[pos] * V[pos])
    out = out * sigmoid(gate)  → g_attn_out

  grid.sync()

  g_attn_out → o_proj → hidden
  hidden += residual
  → MLP → hidden

  grid.sync()

... (重复 Layer 4~23)

Final RMSNorm（block 0）：
  hidden → RMSNorm(final_norm_w) → g_normalized [1024 F32]

lm_head_kernel（独立）：
  g_normalized → logit[v] = g_normalized · lm_head_weight[v,:]  for v=0..248319
  argmax(logit) → output_token_id

输出：next_token_id（如 = 67890）
```

---

## generate 函数的完整流程

```python
def generate(self, prompt: str, max_tokens: int = 100) -> str:
    self.reset()  # 清空 KV Cache 和 DeltaNet 状态

    # Tokenize 输入
    ids = tokenizer.encode("你好，我是")  # → [12, 345, 67, 890, 1011]

    # 处理 prompt 的前 S-1 个 token（只做 step，不收集输出）
    for tid in ids[:-1]:
        self.step(tid)

    # 从最后一个 prompt token 开始生成
    next_id = ids[-1]
    out = []
    for _ in range(max_tokens):
        next_id = self.step(next_id)  # 每次生成一个 token
        if next_id == eos_token_id:
            break
        out.append(next_id)

    return tokenizer.decode(out)  # 还原为文本
```

注意：这里没有用 prefill kernel，而是用 decode kernel 逐个处理 prompt token。prefill kernel 只在需要高吞吐时使用（如 `bench_pp_tg.py` 和 `final_bench.py`）。

---

## 同步机制的正确性保证

`AtomicGridSync` 的实现防止了两类问题：

**问题 1：某 block 超前执行**

如果 block A 比 block B 快，在 block B 还没完成上一层时，block A 就开始下一层计算。  
→ `grid.sync()` 中每个 block 必须等所有 82 个 block 都到达栅栏才能继续。

**问题 2：虚假唤醒**

generation 计数器用来区分不同轮次的栅栏。block 等待的条件是 `*generation > local_gen`，而不是简单的"counter 归零"。这样即使某 block 的 `counter` 重置了，其他 block 也能正确区分是新一轮还是残余值。

**已知陷阱（README 提到）**：

> `grid.sync()` inside loops will deadlock silently.

曾经尝试在 DeltaNet 的逐 token 循环内部做 `grid.sync()`，结果无限挂起。原因：`grid.sync()` 要求所有 block 都执行到同一同步点，而 DeltaNet 循环内部不同 block 的迭代次数可能不同（block 15 只处理 16 个头，其他 block 空闲），导致死锁。

**正确做法**：只在层与层之间做 `grid.sync()`。
