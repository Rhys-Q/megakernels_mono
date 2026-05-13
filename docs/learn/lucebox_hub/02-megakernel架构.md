# Megakernel 架构设计

## 总体思路

Megakernel 的设计围绕一个核心问题：**如何在单次 GPU 调度中完成整个 24 层模型的前向传播？**

答案是：**持久化 Grid + 原子栅栏同步**。

---

## 整体架构图

```
Python 代码 (model.py)
    │
    ├── load_weights()     → 从 HuggingFace 加载 BF16 权重
    ├── _pack_layer_weights() → 打包权重指针为设备端 blob
    └── Decoder.step(token_id)
            │
            ▼
    torch_bindings.cpp
    decode()
            │
            ▼
    kernel.cu
    launch_decode()
    ┌─────────────────────────────────────┐
    │   decode_kernel<<<82 blocks, 512>>> │  ← 一次 dispatch
    │   ┌─────────────────────────────┐   │
    │   │  Layer 0  (DeltaNet)        │   │
    │   │  grid.sync()                │   │
    │   │  Layer 1  (DeltaNet)        │   │
    │   │  grid.sync()                │   │
    │   │  Layer 2  (DeltaNet)        │   │
    │   │  grid.sync()                │   │
    │   │  Layer 3  (Full Attention)  │   │
    │   │  grid.sync()                │   │
    │   │  ...                        │   │
    │   │  Layer 23 (Full Attention)  │   │
    │   │  grid.sync()                │   │
    │   │  Final RMSNorm (block 0)    │   │
    │   └─────────────────────────────┘   │
    └─────────────────────────────────────┘
            │
            ▼
    lm_head_kernel<<<512 blocks, 256>>>
    （词表投影 + argmax → 下一个 token）
```

---

## 关键设计决策

### 决策 1：82 个 Block，每个 512 线程

RTX 3090 有 82 个 SM，每个 SM 驻留 1 个 block（BLOCK_SIZE=512 时，寄存器和共享内存的压力使得每 SM 只能驻留 1 个）。

这样 82 个 block 恰好填满所有 SM，保证：
1. 所有 block 都在 GPU 上，`grid.sync()` 不会死锁
2. 所有 SM 保持满负荷运行

```cpp
#define NUM_BLOCKS 82   // RTX 3090 的 SM 数量
#define BLOCK_SIZE 512  // 每个 block 的线程数 = 16 个 warp
```

`setup.py` 在编译时自动检测当前 GPU 的 SM 数量，并在运行时通过 `query_max_safe_decode_blocks()` 动态核验。

### 决策 2：原子栅栏代替 cudaDeviceSynchronize

`cudaDeviceSynchronize()` 会让 CPU 等待 GPU，从而引入 CPU 往返延迟。

Megakernel 用纯 GPU 端的原子操作实现层间同步：

```cpp
struct AtomicGridSync {
    unsigned int *counter;    // 已到达栅栏的 block 数
    unsigned int *generation; // 当前代次（每过一次栅栏 +1）
    unsigned int nblocks;
    unsigned int local_gen;   // 本 block 记录的代次

    __device__ void sync() {
        __syncthreads();  // 先保证本 block 内部完成
        if (threadIdx.x == 0) {
            // 最后一个到达的 block 负责推进 generation
            unsigned int arrived = atomicAdd(counter, 1);
            if (arrived == nblocks - 1) {
                *counter = 0;
                atomicAdd(generation, 1);  // 通知其他 block 可以继续
            } else {
                // 其他 block 自旋等待 generation 变化
                while (*generation <= local_gen) {}
            }
            local_gen++;
        }
        __syncthreads();  // 保证 block 内所有线程看到新状态
    }
};
```

这个机制纯粹在 GPU 寄存器和全局内存中运作，完全不需要 CPU 介入。

### 决策 3：全局缓冲区而不是本地分配

由于 kernel 需要在层间传递数据（hidden state、KV cache、DeltaNet state 等），这些数据不能放在本地（kernel 退出就消失了），必须放在全局内存中。

Python 端在 `Decoder.__init__()` 中预分配所有缓冲区：

```python
# KV Cache（Full Attention 层）
self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, max_seq_len, FA_HEAD_DIM, ...)
self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

# DeltaNet 状态矩阵
self._dn_states = torch.zeros(n_dn, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM, ...)
self._conv_bufs  = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, ...)

# 各种 scratch 缓冲区（层间传递的临时激活值）
self._hidden     = torch.empty(HIDDEN_SIZE, ...)  # [1024] bf16
self._activations = torch.empty(max_scratch, ...)  # 最大激活尺寸
```

### 决策 4：冗余计算换通信

每个 block 都单独计算 RMSNorm（而不是 block 0 算完后广播给其他 block）。

这看起来浪费，但更快：避免了一次额外的 `grid.sync()`，让每个 block 独立运行。

```cpp
// 注意：rmsnorm_redundant 里所有 block 都算，没有条件分支
__device__ void rmsnorm_redundant(...) {
    // 所有 block 都执行以下代码
    ...
    // 但只有 block 0 更新全局的 g_residual（用于残差连接）
    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }
}
```

### 决策 5：矩阵乘法按 Block 分块

对于大矩阵乘法（如 gate_proj: [3584, 1024]），不同 block 负责不同的输出行：

```cpp
void matvec_bf16(..., int out_dim, int num_blocks) {
    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);
    // 只计算 [row_start, row_end) 范围内的输出行
}
```

82 个 block 同时并行计算矩阵的不同部分。

---

## 文件结构与职责

| 文件 | 职责 |
|------|------|
| `setup.py` | 编译配置：自动检测 GPU 架构，设置 NUM_BLOCKS、TARGET_SM 等宏 |
| `half_type.h` | 半精度类型抽象：sm_80+ 用 BF16，sm_75 用 FP16 |
| `model.py` | Python 层：加载权重、打包指针、`Decoder` 类 |
| `torch_bindings.cpp` | C++/PyTorch 胶水层：注册 CUDA 函数为 PyTorch op |
| `kernel.cu` | **核心 Decode Kernel**：所有 decode 逻辑 |
| `prefill.cu` | **Prefill Kernel**：序列预填充逻辑（cuBLAS GEMM） |
| `kernel_gb10_nvfp4.cu` | Blackwell GPU 专用 NVFP4 decode kernel |
| `prefill_megakernel.cu` | Blackwell GPU 专用单次 dispatch prefill |
| `final_bench.py` | 基准测试脚本（10次热身 + 20次计时） |

---

## 两种运行模式：Prefill vs Decode

### Prefill（预填充）

处理输入 prompt 的所有 token，为 decode 阶段做准备。

- 输入：完整的 token ID 序列，如 `[101, 234, 567, 890]`
- 使用 `prefill.cu`：cuBLAS GEMM 批量处理整个序列
- 产出：KV Cache 填充完毕，DeltaNet 状态更新完毕

### Decode（解码）

每次生成一个新 token。

- 输入：当前 token ID（单个）
- 使用 `kernel.cu`：单 token 的完整前向传播
- 产出：下一个 token ID

```
Prefill 阶段：
  [token_0, token_1, ..., token_S-1] → 并行处理 → KV Cache 就绪

Decode 阶段（循环）：
  token_S → decode_kernel → token_S+1
  token_S+1 → decode_kernel → token_S+2
  ...直到生成 EOS 或达到 max_tokens
```

---

## 层的调度逻辑

`decode_kernel` 主循环（`kernel.cu` 第 881 行）：

```cpp
int dn_layer_idx = 0, fa_layer_idx = 0;

for (int layer = 0; layer < NUM_LAYERS; layer++) {
    const half_t *layer_input = (layer == 0) ? embed_row : hidden_buffer;

    if (LAYER_TYPE[layer] == 0) {
        // DeltaNet 层
        deltanet_layer(grid, ..., dn_states + dn_layer_idx * stride, ..., dn_layer_idx, ...);
        dn_layer_idx++;
    } else {
        // Full Attention 层
        full_attention_layer(grid, ..., fa_k_cache + fa_layer_idx * stride, ..., fa_layer_idx, ...);
        fa_layer_idx++;
    }
}
```

第 0 层的输入是 embed_row（词嵌入），其余层的输入是上一层的 hidden_buffer 输出。

---

## 性能关键路径

### 显存带宽瓶颈

decode 阶段（batch=1）主要受**显存带宽**限制，而不是计算限制。

每生成一个 token，需要：
- 读取所有 24 层的权重（~1.5GB BF16 × 2 bytes = 3GB 读取量）
- RTX 3090 带宽 936 GB/s → 理论上每次 decode 需要 ~3.2ms → ~312 tok/s

实测 413 tok/s，高于理论值，原因：
- L2 缓存命中（小矩阵可以缓存）
- 权重加载和计算流水线化

### DVFS 功耗甜点

```
220W：411 tok/s，1.87 tok/J  ← 最佳效率点
300W：432 tok/s，1.44 tok/J
420W：433 tok/s，1.38 tok/J
```

Megakernel 的 tight 执行使得 GPU 在 220W 时就达到了计算上限，降频反而提升了能效。
