# Megakernel 组件详解

本文深入讲解 `kernel.cu` 中每个核心函数的作用和实现细节。

---

## 1. 模型常量（Model Constants）

文件开头定义了 Qwen3.5-0.8B 的全部超参数：

```cpp
constexpr int HIDDEN_SIZE = 1024;       // 隐藏层维度
constexpr int INTERMEDIATE_SIZE = 3584; // MLP 中间层维度
constexpr int NUM_LAYERS = 24;          // 总层数

// Full Attention 参数
constexpr int FA_NUM_Q_HEADS = 8;    // Q 头数
constexpr int FA_NUM_KV_HEADS = 2;   // KV 头数（GQA：8个Q头共享2个KV头）
constexpr int FA_HEAD_DIM = 256;     // 每个头的维度

// DeltaNet 参数
constexpr int DN_NUM_HEADS = 16;     // DeltaNet 头数
constexpr int DN_KEY_DIM = 128;      // Key 维度
constexpr int DN_VALUE_DIM = 128;    // Value 维度
constexpr int DN_CONV_KERNEL = 4;    // 1D 卷积核长度
```

**GQA（Grouped Query Attention）**：8 个 Q 头共享 2 个 KV 头，节省 KV Cache 显存。每 4 个 Q 头用同一对 K/V。

---

## 2. 权重结构体

```cpp
// Full Attention 层的权重（11 个张量）
struct FullAttnWeights {
    const half_t *input_layernorm_weight; // [1024] 输入归一化
    const half_t *q_proj_weight;          // [4096, 1024] Q 投影（含 gate）
    const half_t *k_proj_weight;          // [512, 1024]  K 投影
    const half_t *v_proj_weight;          // [512, 1024]  V 投影
    const half_t *q_norm_weight;          // [256]  Q 归一化
    const half_t *k_norm_weight;          // [256]  K 归一化
    const half_t *o_proj_weight;          // [1024, 2048] 输出投影
    const half_t *post_attn_layernorm_weight; // 后归一化
    const half_t *gate_proj_weight;       // [3584, 1024] MLP gate
    const half_t *up_proj_weight;         // [3584, 1024] MLP up
    const half_t *down_proj_weight;       // [1024, 3584] MLP down
};

// DeltaNet 层的权重（14 个张量）
struct DeltaNetWeights {
    const half_t *input_layernorm_weight; // [1024]
    const half_t *qkv_proj_weight;        // [6144, 1024] 合并 QKV
    const half_t *z_proj_weight;          // [2048, 1024] gate 投影
    const half_t *beta_proj_weight;       // [16, 1024]   更新强度
    const half_t *alpha_proj_weight;      // [16, 1024]   decay 强度
    const half_t *conv1d_weight;          // [6144, 1, 4] 1D 卷积
    const half_t *a_log;                  // [16]  decay 底数对数
    const half_t *dt_bias;               // [16]  time step 偏置
    const half_t *norm_weight;            // [128] 内部归一化
    const half_t *out_proj_weight;        // [1024, 2048] 输出投影
    const half_t *post_attn_layernorm_weight; // 后归一化
    const half_t *gate_proj_weight;       // MLP gate
    const half_t *up_proj_weight;         // MLP up
    const half_t *down_proj_weight;       // MLP down
};
```

Python 端的 `_pack_layer_weights()` 将这些 PyTorch tensor 的设备指针打包成一个二进制 blob，传给 CUDA kernel。

---

## 3. RMSNorm

**作用**：对 hidden state 做均方根归一化，稳定训练和推理。

公式：`output = input / sqrt(mean(input²) + ε) × (1 + weight)`

```cpp
__device__ void rmsnorm_redundant(
    const half_t *input,    // 全局内存 BF16 输入 [HIDDEN_SIZE]
    const half_t *weight,   // 权重 [HIDDEN_SIZE]
    half_t *s_out,          // 共享内存 BF16 输出
    half_t *g_residual)     // 全局内存：保存残差连接用的副本
{
    // 步骤1：所有线程并行计算各自负责的元素的平方和
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(__ldg(input + i));  // __ldg: 通过只读缓存加载
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    // 步骤2：warp 内归约 → block 内归约 → 得到总平方和
    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float sum = warp_reduce_sum(...);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / HIDDEN_SIZE + RMS_EPS); // 1/sqrt(...)
    }
    __syncthreads();

    // 步骤3：用 rstd 缩放并乘以权重
    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));  // 注意：是 (1 + w)，不是 w
    }
}
```

**技巧细节**：
- 用 `__ldg`（load through global cache）加载只读数据，命中 L1 缓存
- 输出写到 **共享内存** `s_out`，供后续的矩阵乘法直接读取（避免二次全局内存访问）
- block_0 额外把结果写到全局内存 `g_residual`，用于后面的残差连接

---

## 4. 矩阵向量乘法（Matvec）

**作用**：计算 `output = weight × input`（模型的线性投影）

这里的 input 是一个向量（单 token 的 hidden state），weight 是矩阵。

```cpp
__device__ void matvec_bf16(
    const half_t *s_input,   // 共享内存中的输入向量 [in_dim]
    const half_t *weight,    // 权重矩阵 [out_dim, in_dim]
    float *output,           // 输出 [out_dim]，F32 累加
    int in_dim, int out_dim, int num_blocks)
{
    // 每个 block 负责 out_dim 的一段
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;

    // 每个 warp 负责一行（warp-per-row 策略）
    for (int m = row_start + warp_id; m < row_end; m += NUM_WARPS) {
        const half_t *w_row = weight + m * in_dim;
        float sum = 0.0f;

        // 每个 lane 处理 8 个元素（128位加载）
        for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
            uint4 w_u4 = load_128bit(...);  // 一次加载 8 个 BF16
            sum += dot8_bf16(w_u4, s_input + k);  // 点积
        }

        // warp 归约：32个 lane 的部分和 → 最终和
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) output[m] = sum;
    }
}
```

**关键优化**：
- **128位加载**：`load_128bit` 一次读 4 个 uint32（= 8 个 BF16），填满内存总线带宽
- **warp-per-row**：每个 warp 独立计算一行，warp 内 32 个 lane 并行处理该行的不同列段，最后 warp 内归约
- **BF16 权重，FP32 累加**：`dot8_bf16` 把 BF16 转为 FP32 再累加，保证精度

### 变体：融合 gate+up+SiLU

MLP 的 gate 和 up 投影可以融合计算，一次加载输入，同时乘以两个权重矩阵，顺便做 SiLU 激活：

```cpp
__device__ void matvec_gate_up_silu_bf16(...) {
    // 同时计算 gate 和 up 的点积
    float gate_sum = 0.0f, up_sum = 0.0f;
    for (int k = lane_id * 8; ...) {
        gate_sum += dot8_bf16(g_u4, s_input + k);
        up_sum   += dot8_bf16(u_u4, s_input + k);
    }
    // SiLU 激活后乘以 up
    if (lane_id == 0)
        output[m] = fast_silu(gate_sum) * up_sum;
}
```

---

## 5. Full Attention 层

**作用**：标准的多头注意力，用于 Layer 3, 7, 11, 15, 19, 23。

处理流程分 5 个阶段，用 `grid.sync()` 隔开：

### Phase 1：RMSNorm + QKV 投影

```
input → RMSNorm → normalized
normalized → q_proj → g_q    [FA_QPROJ_SIZE = 2048]  (含 gate 向量)
normalized → k_proj → g_kv   [FA_KV_SIZE = 512]
normalized → v_proj → g_kv+  [FA_KV_SIZE = 512]
grid.sync()
```

注意：`q_proj` 输出 `FA_QPROJ_SIZE = 2048`（= Q的1024 + gate的1024），是拼接在一起的。

### Phase 2：QK 归一化 + RoPE + KV Cache

```
g_q → q_norm → RoPE → 写回 g_q
g_kv → k_norm → RoPE → 写到 k_cache[position]
g_kv+ → 写到 v_cache[position]
grid.sync()
```

**RoPE（旋转位置编码）**：给 Q 和 K 加上位置信息，让模型知道每个 token 在序列中的位置。

```cpp
// 对头维度的前 FA_ROTARY_DIM=64 个维度做旋转
for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
    float normed = q[i] * scale * (1 + weight[i]);
    if (i < FA_ROTARY_DIM) {
        float freq = position / pow(10000000.0, 2*i / FA_ROTARY_DIM);
        float cos_v = cosf(freq), sin_v = sinf(freq);
        // 旋转变换：[cos, -sin; sin, cos] 乘以 [q_i, q_{i+half}]
        q[i] = normed * cos_v - q_paired * sin_v;
    }
}
```

### Phase 3：注意力计算（Online Softmax）

对 Q 头的每个 position，与所有历史 K 做点积，用 **online softmax** 计算注意力权重，加权聚合 V：

```cpp
// 遍历所有历史位置（0 到 position）
float max_score = -INFINITY, sum_exp = 0;
float out_acc[EPL];  // 输出累加器

for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
    float score = Q · K[pos] * scale;

    // Online softmax：动态更新 max 和 sum
    float old_max = max_score;
    max_score = fmaxf(max_score, score);
    float exp_diff = fast_exp(old_max - max_score);
    sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);

    // 加权聚合 V
    float wt = fast_exp(score - max_score);
    for (int e = 0; e < EPL; e++)
        out_acc[e] = out_acc[e] * exp_diff + wt * V[pos][e];
}

// 最终归一化：除以 sum_exp
// 并乘以 sigmoid(gate)：这是 Qwen3.5 的 gated attention 变体
for (int e = 0; e < EPL; e++)
    out_head[i] = out_acc[e] / sum_exp * sigmoid(gate[i]);
```

**online softmax**：不需要先算完所有 score 再做 softmax，边遍历边维护最大值，数值稳定。

### Phase 4：输出投影 + 残差

```
g_attn_out → o_proj → hidden_out
hidden_out = hidden_out + residual  (残差连接)
grid.sync()
```

### Phase 5：MLP

```
hidden_out → post_attn_rmsnorm → MLP（gate_up_silu + down） → hidden_out
grid.sync()
```

---

## 6. DeltaNet 层

**作用**：线性注意力层，用于 Layer 0,1,2, 4,5,6, ... 等 18 层。

DeltaNet 的核心思想：维护一个状态矩阵 `S[DN_KEY, DN_VALUE]`，每个 token 更新这个矩阵，并从中查询输出。

**更新公式**：
```
error = (v - decay × (S · k)) × beta
output = decay × (S · q) + error × (k · q)
S_new = S × decay + k_outer_product × error
```

其中：
- `decay`：控制历史记忆的衰减速度（0 ~ 1）
- `beta`：控制这个 token 对状态的更新强度

### 实现细节：Warp 协作状态更新

状态矩阵 S 的维度是 `[16 heads × 128 key × 128 value]`，约 1MB FP32。

为了效率，不同 warp 负责不同的 value 维度（j 方向），warp 内的不同 lane 负责不同的 key 维度（i 方向）：

```cpp
// J_PER_WARP = 128/16 = 8（每个 warp 负责 8 个 value 维度）
// I_PER_LANE = 128/32 = 4（每个 lane 负责 4 个 key 维度）

for (int jj = 0; jj < J_PER_WARP; jj++) {
    int j = warp_id * J_PER_WARP + jj;
    float s_regs[I_PER_LANE];  // 状态矩阵的一部分存在寄存器里

    // 从全局内存加载状态
    for (int ii = 0; ii < I_PER_LANE; ii++) {
        s_regs[ii] = state[j * DN_KEY + lane_id + ii * WARP_SIZE];
    }

    // 计算 stk = S[j,:] · k，sqv = S[j,:] · q
    float stk = 0, sqv = 0;
    for (int ii = 0; ii < I_PER_LANE; ii++) {
        stk += s_regs[ii] * s_k[lane_id + ii * WARP_SIZE];
        sqv += s_regs[ii] * s_q[lane_id + ii * WARP_SIZE];
    }
    stk = warp_reduce_sum(stk);  // warp 内归约
    sqv = warp_reduce_sum(sqv);

    // 计算误差和输出
    float error_j = (s_v[j] - decay * stk) * beta;
    float o_j = decay * sqv + error_j * kq;
    if (lane_id == 0) out_head[j] = o_j;

    // 更新状态矩阵（存回寄存器，最终写回全局内存）
    for (int ii = 0; ii < I_PER_LANE; ii++) {
        state[j * DN_KEY + i] = s_regs[ii] * decay + s_k[i] * error_j;
    }
}
```

**关键优化**：状态矩阵的一部分暂存在**寄存器**（`s_regs`）里，避免反复读写全局内存。

### Conv1d 前处理

DeltaNet 的 Q、K、V 在做注意力之前，先过一个 **1D 卷积**：

```cpp
// 维护 conv_buf：长度为 4 的滑动窗口（历史 3 个 + 当前 1 个）
float h0 = layer_conv[ch*4+1]; layer_conv[ch*4+0] = h0;  // 移位
float h1 = layer_conv[ch*4+2]; layer_conv[ch*4+1] = h1;
float h2 = layer_conv[ch*4+3]; layer_conv[ch*4+2] = h2;
layer_conv[ch*4+3] = g_qkv[ch];  // 新的输入

// 应用卷积核（Causal Conv1d）
float co = 0;
for (int t = 0; t < 4; t++)
    co += layer_conv[ch*4+t] * conv_w[ch*4+t];
dst[c] = fast_silu(co);  // SiLU 激活
```

这个操作让每个位置的表示包含了前 4 个 token 的信息（局部上下文），然后再做 DeltaNet 的长程记忆更新。

---

## 7. LM Head（语言模型头）

**作用**：把最后一层的 hidden state 映射到词表概率分布，找到概率最高的 token。

```cpp
__global__ void lm_head_kernel(...) {
    // 每个 block 负责词表的一段（约 248320/512 ≈ 485 个词）
    // 计算每个词的 logit = hidden · weight[vocab_id]
    // 同时做 argmax 找最大值
}
```

实现了两级 argmax：
1. 每个 block 找出自己负责范围内的最大值
2. block 0 等所有 block 完成（原子 counter 同步），然后找全局最大值

还支持 **repetition penalty**（重复惩罚）：对已经生成过的 token 降低其 logit，避免模型一直重复。

---

## 8. 快速数学函数

Megakernel 用 PTX 内联汇编实现了高效的近似数学函数：

```cpp
// 快速 exp（用 ex2 指令，以 2 为底）
__device__ float fast_exp(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504f));
    return y;
}

// 快速 sigmoid（利用 rcp 指令）
__device__ float fast_sigmoid(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(1.0f + fast_exp(-x)));
    return y;
}

// SiLU = x × sigmoid(x)
__device__ float fast_silu(float x) {
    return x * fast_sigmoid(x);
}
```

这些近似函数比标准的 `expf()`/`sigmoidf()` 快数倍，精度够用于 LLM 推理。
