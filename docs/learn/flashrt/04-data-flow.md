# FlashRT 数据流与组件交互

## 数据的完整旅程

从一张摄像头图片到机器人的动作，数据要经历以下变换：

```
输入：
  images = [np.ndarray(224,224,3), np.ndarray(224,224,3)]
  prompt = "pick up the red block"

                    ↓
             [前端层处理]

  1. 图像 → GPU BF16 张量 (2,3,224,224)
  2. prompt → token IDs → BF16 嵌入 (prompt_len, 2048)

                    ↓
           [视觉编码器 Phase A]

  视觉特征：(num_views*256, 1152) BF16
  （每张图片被切成 16×16 = 256 个 patch，每个 patch 用 1152 维向量表示）

                    ↓
        [语言+视觉融合编码器 Phase B]

  K/V 缓存：18层 × (seq_len, 2048)
  （包含图像+文字的综合理解，等待解码器来查询）

                    ↓
           [动作解码器 Phase C]

  噪声 → 动作：(chunk_size, 32) BF16
  （通过 10步扩散，把随机噪声变成有意义的动作序列）

                    ↓
              [后处理]

  反归一化 → 实际关节角度

输出：
  actions = np.ndarray(10, 7)  # 10个时间步，7个关节DOF
```

---

## Phase A：视觉编码（SigLIP，27层）

**目标**：把摄像头图像变成高维特征向量

### 步骤 A1：图像分块嵌入（Patch Embedding）

```
输入：图像 (2, 3, 224, 224)  [2张图，RGB，224×224像素]

                ↓

切成 16×16 的格子（每格 14×14 像素）
→ 每张图 = 16×16 = 256 个 patch
→ 每个 patch 展平 = 14×14×3 = 588 个数值

                ↓

用线性投影把每个 patch 的 588 维 → 1152 维
（这步用的是 csrc/kernels/patch_embed.cu）

                ↓

加上位置编码（告诉模型每个 patch 在哪个位置）

                ↓

输出：(2*256, 1152) = (512, 1152)  [2张图共 512 个 patch]
```

### 步骤 A2-A6：27层 Transformer 自注意力

每一层做的事情（以第 i 层为例）：

```
输入：x (512, 1152)

1. RMSNorm(x) → x_norm         [归一化，让数值稳定]

2. QKV 投影：
   Q = x_norm @ W_q    (512, 1152) → (512, 1152)
   K = x_norm @ W_k    (512, 1152) → (512, 256)
   V = x_norm @ W_v    (512, 1152) → (512, 256)

3. 注意力：Attention(Q, K, V)   [每个 patch 关注其他 patch]
   → (512, 1152)

4. 输出投影：out @ W_o → (512, 1152)

5. 残差：x = x + out

6. FFN：
   a. RMSNorm(x)
   b. Gate: gate = GELU(x_norm @ W_gate)
   c. Up:   up   = x_norm @ W_up
   d. x = x + (gate * up) @ W_down

输出：x (512, 1152)
```

经过 27 层后，每个 patch 的特征已经融合了全局信息。

**输出结果**：`vision_x` = (512, 1152) BF16

---

## Phase B：语言+视觉融合编码（Gemma-2B，18层）

**目标**：把视觉特征和文字指令融合在一起，生成 K/V 缓存

### 步骤 B0：视觉特征投影

把视觉特征的维度从 1152 → 2048（Gemma-2B 的维度）：

```
vision_x (512, 1152) → projector (1152×2048) → encoder_x[:512] (512, 2048)
```

同时，文字嵌入（prompt_len, 2048）被放在 encoder_x 的后半部分：

```
encoder_x = [
    vision_features (512, 2048),   # 前 512 行：视觉
    language_embeds (L, 2048),     # 后 L 行：文字
]
总形状：(512+L, 2048)
```

### 步骤 B1-B5：18层编码器

每层包含：
1. RMSNorm → FP8 量化（如果已校准）
2. QKV 投影（GQA，分组查询注意力）
3. 自注意力（视觉和文字 patch 互相关注）
4. 输出投影
5. FFN（融合 gated activation）

**FP8 融合优化**：

当 FP8 已校准时，FlashRT 使用融合内核，把多个步骤合并成一个：

```
普通写法：
  残差加法 → 单独内核
  RMSNorm  → 单独内核
  FP8量化  → 单独内核

FlashRT 融合：
  residual_add_rms_norm_fp8() → 一个内核搞定三件事
```

这样减少了内存带宽消耗（中间结果不需要写回全局内存）。

**K/V 缓存**：编码器运行完后，每一层的 Key 和 Value 矩阵会被保存下来：

```
attn_backend.kv_cache["encoder"][layer_i] = {
    "k": (seq_len, num_kv_heads, head_dim),
    "v": (seq_len, num_kv_heads, head_dim),
}
```

这个 K/V 缓存在解码器的交叉注意力中使用。

**输出结果**：K/V 缓存，18 层 × 每层的 key/value 矩阵

---

## Phase C：动作解码（Gemma-300M，10步×18层）

**目标**：通过扩散过程，把随机噪声变成有意义的动作序列

### 扩散过程简介

Pi0.5 使用**扩散模型**生成动作。扩散模型的基本思路：

```
训练时：真实动作 + 越来越多的噪声 → 训练模型学会"去噪"
推理时：纯噪声 → 模型去噪10次 → 有意义的动作
```

每一步去噪：
```
输入：当前噪声状态 x_t (chunk_size=10, action_dim=32)
      + 时间步 t 的条件信息（AdaRMSNorm 调制参数）
      + 编码器的 K/V 缓存（视觉+语言信息）

输出：去噪后的 x_{t-1} (chunk_size=10, action_dim=32)
```

### 步骤 C0：动作输入投影

```
噪声 x (chunk_size, action_dim=32) → action_in_proj → (chunk_size, decoder_dim=1024)
```

### 步骤 C1：AdaRMSNorm（样式调制归一化）

普通 RMSNorm 的缩放因子是固定的。AdaRMSNorm 的缩放因子根据当前扩散步的时间步 t 动态调整：

```
style = precomputed_styles[step]  # (chunk_size, 3*decoder_dim)
scale, shift, gate = split(style)

# AdaRMSNorm
x_norm = RMSNorm(x) * (1 + scale) + shift
# gate 用于输出端的残差门控
```

预计算的 styles 在模型加载时就算好了（因为它们只依赖于 prompt，不依赖输入），所以推理时直接读取，不需要重新计算。

### 步骤 C2-C7：18层解码器

每层包含**自注意力**和**交叉注意力**两个注意力模块：

```
自注意力（只看动作序列内部）：
  Q, K, V 来自动作序列本身
  解码器的 chunk_size=10 个动作相互关注
  K/V 被追加到 K/V 缓存（用于后续层）

交叉注意力（看编码器的信息）：
  Q 来自解码器（动作）
  K/V 来自编码器的缓存（视觉+语言）
  动作在这里"查询"图像和文字的信息
```

这就是为什么机器人能根据图像和文字生成合适的动作——交叉注意力让动作生成时能参考视觉+语言信息。

### 步骤 C8：输出 + 残差

```
AdaRMSNorm(x) → action_out_proj → delta_x (chunk_size, action_dim=32)
x = x + delta_x  # 扩散步的残差更新
```

这个过程重复 10 次（10个扩散步），每次 x 都更接近真实动作。

**最终输出**：`action_buf` = (10, 32) BF16

---

## 后处理

### 维度转换

```
(chunk_size=10, action_dim=32) → (chunk_size=10, dof=7)
```

注意：Pi0.5 的 `action_dim=32` 但机器人只有 7 个 DOF（自由度），多余的维度在训练时被 mask 掉。实际使用时取前 7 维或特定维度。

### 反归一化

训练时动作被归一化了（均值0，标准差1）。推理时需要反归一化：

```python
actions_denorm = actions * action_std + action_mean
```

其中 `action_std` 和 `action_mean` 从数据集（如 LIBERO）的统计信息中加载。

**最终输出**：`actions` = (10, 7) numpy 数组，单位是弧度/米，代表未来 10 个时间步的关节角度

---

## 内存布局

理解内存布局有助于理解为什么 FlashRT 这么快：

```
设备内存（GPU RAM）
┌─────────────────────────────────────────────────────┐
│  权重缓冲区（只在加载时写入，推理时只读）              │
│  ├── FP8 视觉权重 [27层 × QKVO + FFN]               │
│  ├── FP8 编码器权重 [18层 × QKVO + FFN]              │
│  └── FP8 解码器权重 [18层 × QKVO + FFN]              │
├─────────────────────────────────────────────────────┤
│  推理工作缓冲区（每次推理时复用）                      │
│  ├── observation_images_normalized (2,3,224,224)    │
│  ├── vision_patches (2*256, 1152)                   │
│  ├── vision_x (2*256, 1152)                         │
│  ├── encoder_x (512+L, 2048)                        │
│  ├── decoder_x (chunk_size, 1024)                   │
│  └── action_buf (chunk_size, 32)                    │
├─────────────────────────────────────────────────────┤
│  注意力后端缓冲区                                     │
│  ├── Vision QKV [27层]                               │
│  ├── Encoder QKV + KV Cache [18层]                  │
│  └── Decoder QKV + Cross-Attn Cache [18层]          │
└─────────────────────────────────────────────────────┘
```

**关键**：所有工作缓冲区在加载时一次性分配好，推理时就地覆盖，完全没有动态内存分配。这是 CUDA Graph 能成功录制的前提。

---

## 数据类型的变化

```
加载时：                        推理时：
BF16 权重                       BF16 图像
  ↓                               ↓
FP8 权重（量化）              BF16 patch embedding
                                  ↓
                              BF16 vision features（Q/K/V/Attn/FFN 全程 BF16 或 FP8）
                                  ↓
                              FP8 GEMM（最大化吞吐量）
                                  ↓
                              BF16 累加（保持精度）
                                  ↓
                              BF16 输出动作
                                  ↓
                              FP32 反归一化（CPU 上）
                                  ↓
                              FP64 numpy 数组
```

---

## 并发和流

FlashRT 中的操作都在单个 CUDA 流上执行：

```python
stream = CUDAGraph.create_stream()
# 所有操作都在这个 stream 上
vision_encoder(stream)       # 顺序执行
transformer_encoder(stream)  # 等 A 完成后执行
transformer_decoder(stream)  # 等 B 完成后执行
```

虽然同一模型内部是串行的，但多个 `predict()` 调用可以用不同的 stream 并行（如果有多个 GPU）。

---

## 下一步

- [05-key-concepts.md](./05-key-concepts.md)：深入理解 FP8 量化、CUDA Graph 等技术原理
- [06-demo-walkthrough.md](./06-demo-walkthrough.md)：通过一个完整 Demo 串讲整个流程
