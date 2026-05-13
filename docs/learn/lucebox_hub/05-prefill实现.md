# Prefill 实现

Prefill（预填充）是处理输入 prompt 中所有 token 的阶段。它与 decode（解码）有根本的不同：
- decode：每次处理 1 个 token，强调延迟
- prefill：一次性处理 S 个 token，强调吞吐量

---

## 为什么 Prefill 不能直接用 Decode Kernel？

Decode kernel 每次处理 1 个 token，如果 prompt 有 520 个 token，就需要调用 520 次 decode kernel。

Prefill 专用的优化：
1. **批量矩阵乘法**：S 个 token 同时做投影，用 cuBLAS `cublasGemmEx` 一次完成，效率远高于 S 次单向量乘法
2. **并行注意力**：S 个 token 的注意力可以并行计算（causal mask 保证正确性）
3. **DeltaNet 序列优化**：DeltaNet 的递推必须串行，但可以用"Chunk-Parallel"算法并行化

---

## Prefill 的两种路径

### 路径 A（`prefill.cu`，常规 GPU）

用于 RTX 3090 等 Ampere/Turing GPU。

```
token_ids → pf_embed → hidden [S, 1024]

for each layer:
    pf_rmsnorm → normalized [S, 1024]

    if DeltaNet:
        cuBLAS: normalized → qkv_proj → proj_buf [S, 6144]
        cuBLAS: normalized → z_proj → proj_buf2 [S, 2048]
        pf_bf16_matvec: normalized → beta_buf, alpha_buf

        pf_deltanet_preproject: conv1d+silu+L2norm → dn_pre_qkv [S, 6144]
        pf_dn_chunk_phase1: intra-chunk并行 → u_scratch, w_scratch, cs_scratch
        pf_dn_chunk_phase2: inter-chunk串行 → dn_out_buf [S, 2048]
        pf_deltanet_gated_rmsnorm: 门控归一化

        cuBLAS: dn_out_buf → out_proj → proj_buf
        pf_add_residual_bf16
        pf_mlp_chunked_fused: MLP

    if Full Attention:
        cuBLAS fused QKV GEMM
        pf_qk_norm_rope_fused: QK 归一化 + RoPE + 写入 KV Cache
        tiled cuBLAS attention: 分块计算 QK^T 和 AttnV
        cuBLAS: out_proj
        pf_add_residual_bf16
        pf_mlp_chunked_fused: MLP

pf_final_norm → final_normed [1024]
pf_lm_head + pf_lm_reduce → output_token
```

---

## DeltaNet Prefill：Chunk-Parallel 算法

这是 prefill 最复杂的部分。普通的 DeltaNet 递推是完全串行的：

```
for t in [0, S):
    state = update(state, q[t], k[t], v[t])
    output[t] = query(state, q[t])
```

这没法并行。Chunk-Parallel 算法将序列分成若干小 chunk（每个 chunk 8 个 token），然后：
- **Phase 1（intra-chunk，并行）**：在每个 chunk 内部，用矩阵运算高效计算出 u 和 w（代表"这个 chunk 对状态的贡献"）
- **Phase 2（inter-chunk，串行 per head）**：跨 chunk 传播状态，计算每个 token 的最终输出

### Phase 1：Intra-Chunk（`pf_dn_chunk_phase1`）

每个 (chunk n, head h) 对应一个 GPU block，并行执行：

```
对 chunk n 中的 C=8 个 token：

1. 计算每个 token 的 beta、g（decay rate）
2. cs = cumsum(g)  →  cs_out

3. 构建 M[i,j]（严格下三角）：
   M[i,j] = beta[i] × exp(cs[i]-cs[j]) × (K[i]·K[j])
   （表示 token i 对 token j 的"误差传播"系数）

4. 初始化：
   u = beta × V        → "这个 chunk 对状态 V 分量的贡献"
   w = beta × exp(cs) × K  → "这个 chunk 对状态 K 分量的贡献"

5. 前向替换（Forward Substitution）：
   u[i] -= sum_{s<i} M[i,s] × u[s]
   w[i] -= sum_{s<i} M[i,s] × w[s]
   （去除 chunk 内部 token 间的自我影响）

输出：u_out, w_out [N_chunks, C, H, D]
      cs_out [N_chunks×C, H]
```

### Phase 2：Inter-Chunk（`pf_dn_chunk_phase2`）

每个 (head h, j_split) 对应一个 block，对 j 方向做分块（DN_PHASE2_J_SPLITS=4，每个 block 负责 32 个 value 维度）：

```
for each chunk n:
    1. 计算 d = u[n] - w[n] × state
       （"这个 chunk 受到之前状态影响后的净贡献"）
    
    2. 计算 QKt = Q[n] · K[n]^T（chunk 内的 Q-K 相关性）
    
    3. 计算输出：
       output[c,j] = exp(cs[c]) × (Q[c] · state[j]) +   ← 来自历史状态的贡献
                     sum_{s<=c} QKt[c,s] × exp(cs[c]-cs[s]) × d[s,j]  ← chunk 内贡献
    
    4. 更新状态：
       state = decay × state + sum_c(d_scaled[c] × K[c]^T)
```

**WMMA 优化路径**（可选，`pf_dn_chunk_phase2_wmma`）：

在 sm_80+ (Ampere+) 上，d-compute 和 o_inter 的内积可以用 Tensor Core（WMMA m16n16k16 指令）加速：
- d 的计算：`w × state` → GEMM(M=16, N=32, K=128)，用两个 warp 各算一个 16×16 块
- state 更新：`d^T × K` → GEMM(M=32, N=128, K=8)，四个 warp 各算 4 个 16×16 块

---

## Full Attention Prefill

使用 cuBLAS 的分块 tiled 注意力计算：

```cpp
void pf_causal_attn_fused_tiled_cublas(
    ..., int S, int query_block_tokens=4096, ...) {
    
    for (int qh = 0; qh < 8; qh++) {  // 每个 Q 头
        for (int q0 = 0; q0 < S; q0 += 4096) {  // 按 4096 分块
            int rows = min(4096, S - q0);
            int key_count = q0 + rows;  // 只看 causal 范围内的 K
            
            // Step 1: QK^T (cuBLAS GEMM) → scores [rows, key_count]
            cublas_bf16_qk_scores(Q_chunk, K_head, scores, ...);
            
            // Step 2: Causal Softmax → probs [rows, key_count]
            pf_causal_softmax_to_bf16<<<rows, 512>>>(scores, probs, ...);
            
            // Step 3: Attn × V (cuBLAS GEMM) → attn_out [rows, head_dim]
            cublas_bf16_probs_v(probs, V_head, attn_out, ...);
            
            // Step 4: 乘以 gate（Gated Attention）
            pf_apply_attention_gate_bf16_fused<<<...>>>(attn_out, qkv_fused, out, ...);
        }
    }
}
```

cuBLAS 利用 Tensor Core 加速矩阵乘法，在大 S 时效率很高。

---

## MLP 分块处理（`pf_mlp_chunked_fused`）

对于很长的序列，一次性做 MLP 计算可能超出显存。用分块策略：

```cpp
void pf_mlp_chunked_fused(
    ..., int S, int chunk_tokens=4096, ...) {
    
    for (int offset = 0; offset < S; offset += 4096) {
        int rows = min(4096, S - offset);
        
        // Gate+Up 融合 GEMM（一次计算两个矩阵）
        cublas_bf16_gemm(norm_chunk, fused_gate_up_w, gate_up_buf, rows, 2*INTER, HIDDEN);
        
        // SiLU(gate) × up
        pf_silu_mul_fused<<<...>>>(gate_up_buf, mlp_buf, rows, INTER);
        
        // Down projection
        cublas_bf16_gemm(mlp_buf, down_w, gate_up_buf, rows, HIDDEN, INTER);
        
        // 加残差
        pf_add_residual_bf16<<<...>>>(gate_up_buf, residual_chunk, hidden_chunk, ...);
    }
}
```

---

## Prefill vs Decode 对比

| 特性 | Decode | Prefill |
|------|--------|---------|
| 处理 token 数 | 1 | S（整个 prompt） |
| 矩阵乘法 | matvec（向量×矩阵） | GEMM（矩阵×矩阵） |
| DeltaNet | 单步递推 | Chunk-Parallel 算法 |
| Attention | Online Softmax 遍历 KV Cache | cuBLAS QK^T × probs × V |
| 实现文件 | `kernel.cu` | `prefill.cu` |
| 主要瓶颈 | 显存带宽（小 batch） | 计算（Tensor Core） |
| 调用方式 | `decode_kernel`（单次 dispatch） | 多个 kernel + cuBLAS |

---

## 为什么 Prefill 不是单次 dispatch？

Decode kernel 能做到单次 dispatch 是因为：
- batch_size = 1，矩阵乘法变成向量乘法，82 个 block 就足够
- 层间同步开销相对于计算量可以接受

Prefill 处理 S 个 token：
- 矩阵乘法太大，需要 cuBLAS 的 Tensor Core 最优化（cuBLAS 内部有更复杂的分块策略）
- DeltaNet 的 Chunk-Parallel 算法需要 phase1/phase2 两个不同的 kernel 形态
- 不同子操作（QK^T, softmax, AttnV）的最优线程配置不同

因此 prefill 是多个 kernel 的组合，而不是一个大 kernel。

（Blackwell GPU 的 `prefill_megakernel.cu` 实现了单次 dispatch 的 prefill，但那是 NVFP4 精度，是更进一步的优化。）
