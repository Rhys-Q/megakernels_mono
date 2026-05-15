# FlashRT 主要组件详解

## 概览

FlashRT 由以下主要组件构成：

```
VLAModel（用户接口）
    ↓
load_model()（模型加载器）
    ├── detect_arch()（硬件检测）
    ├── [Frontend]（前端：权重加载 + 量化 + CUDA Graph）
    │       ├── WeightLoader（声明式权重加载）
    │       ├── FP8Calibrator（FP8 校准）
    │       └── CUDAGraph（计算图录制）
    └── [Pipeline]（流水线：推理核心）
            ├── VisionEncoder（视觉编码器）
            ├── TransformerEncoder（语言+视觉融合编码器）
            ├── TransformerDecoder（动作解码器）
            └── AttentionBackend（注意力后端）
```

---

## 组件1：VLAModel（用户接口）

**文件**：`flash_rt/api.py`

这是用户直接使用的对象，它是所有功能的"门面"。

```python
class VLAModel:
    def predict(self, images, prompt, state=None) -> dict:
        """运行推理，返回动作"""
    
    def calibrate(self, observations, percentile=99.9) -> None:
        """用真实数据做 FP8 校准"""
    
    def recalibrate(self) -> None:
        """清除旧校准，重新校准"""
```

**核心功能**：
- 管理 prompt 缓存（同一 prompt 不重复处理）
- 处理首次推理时的"懒校准"（第一次 predict 时自动完成校准）
- 统一不同后端的调用接口

---

## 组件2：硬件分发层

**文件**：`flash_rt/hardware/__init__.py`

### GPU 检测

```python
def detect_arch() -> str:
    cc = get_compute_capability()  # 例如 "12.0"
    if cc == "11.0": return "thor"
    if cc == "12.0": return "rtx_sm120"
    if cc == "8.9":  return "rtx_sm89"
```

### 分发表（_PIPELINE_MAP）

分发表是一个字典，键是 `(模型名, 框架, 硬件架构)` 的三元组，值是对应的前端类路径：

```python
_PIPELINE_MAP = {
    ("pi05", "torch", "thor"):        (..., "Pi05TorchFrontendThor"),
    ("pi05", "torch", "rtx_sm120"):  (..., "Pi05TorchFrontendRtx"),
    ("pi05", "jax",   "thor"):        (..., "Pi05JaxFrontendThor"),
    ("pi05", "torch", "rtx_sm89"):   (..., "Pi05TorchFrontendRtx"),
    ("groot", "torch", "rtx_sm120"): (..., "GrootTorchFrontendRtx"),
    # ... 还有很多
}
```

通过分发表，同一份 `load_model()` 代码可以处理所有模型、框架、硬件的组合。

---

## 组件3：前端（Frontend）

**文件**：`flash_rt/frontends/torch/pi05_rtx.py` 等

前端是最复杂的组件，负责"把一个裸权重文件变成一个可推理的系统"。以 `Pi05TorchFrontendRtx` 为例：

### 初始化流程

```
加载 safetensors 文件
    ↓
convert_pi05_safetensors()
（格式转换：合并 QKV、调整布局、折叠 RMSNorm）
    ↓
FP8 量化权重
（把 BF16 权重压缩成 FP8 E4M3，存为设备指针）
    ↓
预计算 Decoder Styles
（提前计算所有扩散步的 AdaRMSNorm 调制参数，见下方详解）
    ↓
创建 AttentionBackend
（分配 Q/K/V/O 缓冲区）
    ↓
创建 Pi05Pipeline
（传入所有权重指针和后端）
```

### set_prompt() 方法

当用户传入新的文字指令时：

```
文字指令（"pick up block"）
    ↓
PaliGemmaTokenizer.tokenize() → token IDs
    ↓
在 GPU 上用 embedding table 查嵌入 → BF16 张量
    ↓
如果 prompt 长度变了 → 重建流水线（避免 padding 浪费）
    ↓
上传 language_embeds 到编码器缓冲区
```

### infer() 方法

```
输入：摄像头图像 + （每次推理不同的）随机噪声
    ↓
把图像和噪声写入流水线缓冲区（设备内存）
    ↓
重放 CUDA Graph（或首次用 run_pipeline()）
    ↓
从设备内存读取动作数据（D2D 复制到 CPU）
    ↓
反归一化（乘以比例、加偏移）
    ↓
返回 actions: np.ndarray (10, 7)
```

---

## 组件4：模型流水线（Pipeline）

**文件**：`flash_rt/models/pi05/pipeline_rtx.py`

流水线是"推理计算本身"，是被 CUDA Graph 录制的主角。

### 关键设计：只接受指针

```python
class Pi05Pipeline:
    def __init__(self, gemm, fvk, attn_backend, weights: dict, ...):
        # weights 字典里存的是 int（GPU 内存地址），不是 tensor
        self.weights = weights
```

这样设计的原因：CUDA Graph 在捕获时会记录所有 GPU 操作，如果 Python 对象（如 torch.Tensor）参与了这些操作，可能导致捕获失败或产生不可预期的行为。使用原始指针可以完全规避这个问题。

### 三阶段推理

```python
def run_pipeline(self, stream: int):
    # 阶段A：视觉编码器（SigLIP，27层）
    self._vision_encoder(stream)
    
    # 阶段B：文本+视觉融合编码器（Gemma-2B，18层）
    self._transformer_encoder(stream)
    
    # 阶段C：动作解码器（Gemma-300M，10步×18层）
    self._transformer_decoder(stream)
```

详细内容见 [04-data-flow.md](./04-data-flow.md)。

---

## 组件5：注意力后端（AttentionBackend）

**文件**：`flash_rt/hardware/*/attn_backend.py`

### 协议接口

```python
class AttentionBackend(Protocol):
    def run(self, site: str, layer_idx: int, q_seq: int, kv_seq: int, stream: int) -> None:
        """在指定位置运行注意力计算"""
    
    def get_slot_ptrs(self, site: str, layer_idx: int) -> tuple[int, int, int, int]:
        """返回 Q/K/V/O 缓冲区的指针"""
```

`site` 参数指明这是哪种注意力：
- `"siglip"`：视觉编码器的自注意力
- `"encoder"`：文本编码器的自注意力
- `"decoder_self"`：动作解码器的自注意力
- `"decoder_cross"`：解码器和编码器之间的交叉注意力

### 不同硬件的实现

| 硬件 | 实现方式 | 特点 |
|------|---------|------|
| Thor (SM110) | CUTLASS FMHA | 针对 Jetson 优化 |
| RTX (SM120/89) | FlashAttention-2 | 标准高效实现 |

流水线代码完全不知道底层用的是哪种实现，只需调用 `attn_backend.run(...)` 即可。

---

## 组件6：权重加载器（WeightLoader）

**文件**：`flash_rt/executors/weight_loader.py`

这是一个"声明式权重加载框架"。与其手动写：

```python
# 命令式写法（丑陋）
weight_dict["vision.layer0.qkv"] = merge(q, k, v)
weight_dict["vision.layer0.qkv"] = quantize_fp8(weight_dict["vision.layer0.qkv"])
# ... 重复 27 次
```

不如声明：

```python
# 声明式写法（优雅）
WEIGHT_SPEC = ModelWeightSpec(
    blocks=[
        LayerBlock(
            prefix="vision.layer{i}",
            num_layers=27,
            items=[
                Item(key="qkv", transforms=[FuseQKV(), QuantFP8()], sink=TensorList("fp8_vision_qkv"))
            ]
        )
    ]
)
# 然后 WeightLoader 自动执行所有操作
```

### 三层架构

```
WeightSource（读取来源）
    safetensors 文件 / Orbax 目录
         ↓
Transform 管道（变换）
    FuseQKV、转置、量化FP8...
         ↓
WeightSink（存储目标）
    self.weights 字典 / CudaBuffer...
```

这个框架使 PyTorch 和 JAX 路径可以共享同一份权重规格，只需换不同的 Source 和 Sink。

---

## 组件7：FP8 校准器（Calibration）

**文件**：`flash_rt/core/calibration.py`

### 什么是校准？

FP8（8位浮点数）的动态范围比 FP16 小得多。为了不损失精度，我们需要知道每一层的激活值最大能到多少，然后据此缩放。这个过程叫"校准"。

```
实际激活值范围：[0, 150]
FP8 E4M3 能表示的范围：[0, 448]

缩放因子 = max_val / fp8_max = 150 / 448 = 0.335
量化时：fp8_value = round(bf16_value / 0.335)
反量化时：bf16_value ≈ fp8_value * 0.335
```

### 单帧校准 vs 多帧校准

**单帧校准**：
- 用一帧图像跑一次推理
- 收集每一层的激活值最大值（amax）
- 简单但可能不稳定（一帧可能是异常帧）

**多帧校准（推荐）**：
```python
# 收集多帧的 amax
per_sample_amax = []
for obs in observations:
    run_pipeline_with_dynamic_quant()
    per_sample_amax.append(collect_amax())

# 用 99.9 百分位数，而不是绝对最大值
final_amax = np.percentile(per_sample_amax, 99.9, axis=0)
# 这样可以消除偶然出现的异常大值对量化精度的影响
```

### 校准缓存

校准结果保存在本地文件里，下次加载时直接使用，不需要重新跑：

```
~/.flash_rt/calibration/{checkpoint_hash}_Se{N}.json
```

---

## 组件8：CUDA Graph

**文件**：`flash_rt/core/cuda_graph.py`

### 原理

CUDA Graph 是 NVIDIA 的一项技术：把一系列 GPU 操作录制成一个"程序"，之后每次执行只需发一条指令，CUDA 驱动自动重放所有操作。

```
录制阶段（只做一次）：
  graph.begin_capture(stream)
  vision_encoder(...)     → 记录到图中
  transformer_encoder(...)→ 记录到图中
  transformer_decoder(...)→ 记录到图中
  graph.end_capture()

重放阶段（每次推理）：
  graph.replay(stream)    → 一条指令，执行所有上面的操作
```

### 为什么快？

没有 CUDA Graph 时：
- Python 解释器逐个操作调度 GPU
- 每个 cuBLAS/cuDNN 调用都有 CPU 开销
- GPU 无法提前规划执行顺序

有 CUDA Graph 时：
- GPU 驱动拿到完整的执行图，可以最优化调度
- 零 CPU 调度开销
- 多个操作可以并行执行

**结果**：推理延迟从 100ms+ 降至 ~17ms（RTX 5090）

### FlashRT 中的使用

```python
# 前端中的 CUDA Graph 录制
def record_infer_graph(self, obs):
    stream = CUDAGraph.create_stream()
    
    # 预热（让所有内存分配稳定）
    for _ in range(3):
        self.run_pipeline(obs, stream)
    
    # 录制
    self.graph = CUDAGraph()
    self.graph.begin_capture(stream)
    self.run_pipeline(obs, stream)
    self.graph.end_capture()
    
    # 之后每次推理
    # self.graph.replay(stream)  ← 就这一行！
```

---

## 组件9：世界模型（flash_wm）

**文件**：`flash_wm/src/bagel_fp8_engine.py`

FlashRT 还包含一个"世界模型"组件，用于图像生成（而非动作预测）。它基于 BAGEL 模型：

```
BF16 文本 Prefill → KV 缓存
    ↓
FP8 融合扩散步骤（重复多次，使用 KV 缓存注意力）
    ↓
CFG（无分类器引导，3路并行：条件+文本无条件+图像无条件）
    ↓
VAE 解码 → PIL 图像
```

目标：3 视角 × 448×448，24 步 + CFG，在 Thor 上 < 4 秒完成。

---

## 组件交互图

```
用户调用 model.predict(images, prompt)
         │
         ▼
    VLAModel.predict()
         │
         ├─[首次调用]─→ pipe.calibrate(obs)
         │                   │
         │               收集 FP8 amax
         │               上传缩放因子
         │               录制 CUDA Graph
         │
         ├─[prompt 变化]─→ pipe.set_prompt(new_prompt)
         │                   │
         │               tokenize → embed
         │               上传到编码器缓冲区
         │               （如果长度变化，重建流水线）
         │
         └─[正常推理]─→ pipe.infer(obs)
                             │
                         暂存图像到设备缓冲区
                             │
                         graph.replay(stream)
                             │
                         ┌───▼──────────────────┐
                         │  GPU 执行（单条指令）  │
                         │  • 视觉编码 (27层)    │
                         │  • 语言融合 (18层)    │
                         │  • 动作解码 (10×18层) │
                         └───┬──────────────────┘
                             │
                         D2D 复制结果到 CPU
                             │
                         反归一化
                             │
                         返回 actions (10, 7)
```

---

## 附录：预计算 Decoder Styles 详解

这是前端初始化中最容易让人困惑的一步，这里从头讲清楚。

### 问题背景：解码器为什么需要"知道时间步"？

Pi0.5 的动作解码器是一个**扩散模型**。它通过 10 步去噪把随机噪声变成有意义的动作：

```
步骤0（t=1.0）：x₀ = 随机噪声
步骤1（t=0.9）：x₁ = x₀ - 去噪量₀
步骤2（t=0.8）：x₂ = x₁ - 去噪量₁
...
步骤9（t=0.1）：x₉ = x₈ - 去噪量₈ = 最终动作
```

每一步的去噪量不同。为了让模型知道"现在在第几步"，需要把时间步信息注入模型。注入的方式是通过 **AdaRMSNorm（自适应 RMS 归一化）** 的调制参数来实现的。

---

### 第一层：正弦时间嵌入

原始时间步 `t`（一个从 1.0 到 0.1 的浮点数）通过正弦函数编码成一个 1024 维向量：

```python
# 来自 csrc/frontends/torch/pi05_rtx.py
min_period, max_period = 4e-3, 4.0
fraction = linspace(0, 1, 512)               # 512个频率
period = min_period * (max_period / min_period) ** fraction  # 指数间隔的频率

for step in range(10):
    # t 从 1.0 按 -0.1 递减
    sinusoid_input = t / period * 2π          # 512维
    time_embed = [sin(sinusoid_input),         # 512维
                  cos(sinusoid_input)]         # 512维 → 合并为 1024维
```

**为什么用正弦？** 因为正弦函数可以用任意精度区分不同的时间步，而且两个相近时间步的嵌入也是相近的（连续性）。这和 Transformer 的位置编码思路一样。

**结果**：`decoder_time_embeds` = (10, 1024) — 10个时间步，每步一个 1024 维向量。

---

### 第二层：Time MLP（时间 MLP）

1024 维的原始时间嵌入通过一个两层 MLP 进行变换，让模型能学到更复杂的时间依赖关系：

```python
# 2层 MLP，激活函数是 SiLU（= x * sigmoid(x)）
hidden = SiLU(t_embed @ W_in + b_in)   # (1, 1024) → SiLU → (1, 1024)
time_emb = SiLU(hidden @ W_out + b_out)  # (1, 1024) → SiLU → (1, 1024)

# 扩展到 chunk_size（动作序列的每个位置都用同一个时间嵌入）
time_emb_expanded = repeat(time_emb, chunk_size=10)  # (10, 1024)
```

**结果**：`time_emb_out` = (num_steps=10, chunk_size=10, 1024)

---

### 第三层：Style 线性投影（调制参数生成）

时间嵌入需要投影成三个部分才能用于 AdaRMSNorm：
- **scale（缩放）**：控制归一化后每个维度的放大倍数
- **shift（偏移）**：控制归一化后每个维度的平移量
- **gate（门控）**：控制残差连接的权重

解码器有 18 层，每层有两处 AdaRMSNorm（注意力前和 FFN 前），所以需要 18×2 套投影权重。

```python
for step in range(10):
    te = time_emb_expanded[step]           # (chunk_size=10, 1024)

    for i in range(18):  # 每个解码器层
        # 注意力前的 AdaRMSNorm 调制
        style_attn[step, i] = te @ W_attn_mod[i] + b_attn_mod[i]
        # (10, 1024) @ (1024, 3072) + (3072,) → (10, 3072)
        # 3072 = 3 × 1024 = 3 份 1024 维向量 [scale | shift | gate]

        # FFN 前的 AdaRMSNorm 调制
        style_ffn[step, i] = te @ W_ffn_mod[i] + b_ffn_mod[i]
        # (10, 1024) → (10, 3072)

    # 最后一层归一化的调制（只有一套）
    style_final[step] = te @ W_final_mod + b_final_mod
    # (10, 1024) → (10, 3072)
```

**结果**：

| 缓冲区 | 形状 | 说明 |
|--------|------|------|
| `style_attn` | (10, 18, 10, 3072) | 10步 × 18层 × 10动作 × 3组参数 |
| `style_ffn` | (10, 18, 10, 3072) | 10步 × 18层 × 10动作 × 3组参数 |
| `style_final` | (10, 10, 3072) | 10步 × 10动作 × 3组参数（最终层） |

总存储量：`(10×18×10×3072 + 10×18×10×3072 + 10×10×3072) × 2字节`
= `(11,059,200 + 11,059,200 + 307,200) × 2 ≈ 44 MB` BF16 数据。

---

### 第四层：AdaRMSNorm 内核实际做什么？

拿到 style 后，CUDA 内核的计算公式（来自 `csrc/kernels/norm.cu`）是：

```c
// 每个 token 位置（row）独立计算
float rms = rsqrtf(mean(x²) + eps);       // 普通 RMSNorm 的分母

// style_row 是当前 step、当前 layer、当前 token 的 (3×1024) 向量
scale  = style_row[0 : 1024]              // 第1段
shift  = style_row[1024 : 2048]           // 第2段
gate   = style_row[2048 : 3072]           // 第3段，存入 gate_buf（后面用）

// AdaRMSNorm 输出（带调制）：
out[i] = x[i] * rms * weight[i]          // 普通 RMSNorm 部分
       * (1 + scale[i])                   // × 时间步相关的缩放
       + shift[i]                         // + 时间步相关的偏移
```

等价的 Python 表达：

```python
# 普通 RMSNorm
x_norm = x / sqrt(mean(x²) + 1e-6) * weight

# AdaRMSNorm = RMSNorm × (1 + scale) + shift
x_ada = x_norm * (1 + scale) + shift

# gate 单独存起来，用于后面的残差门控
# residual = x + gate * FFN(x_ada)
```

**直观理解**：
- `scale` 和 `shift` 让模型在不同时间步可以学到"这个层在 t=0.9 时应该放大某些维度，在 t=0.1 时应该压缩某些维度"。
- `gate` 控制这一层的输出对残差的贡献权重，也是时间步相关的。
- 整体效果：**同一组权重，通过调制参数，在不同扩散步表现出不同的行为**。

---

### 为什么要"预计算"而不是在推理时计算？

关键观察：**style 只依赖于时间步 t，不依赖于输入图像或动作噪声。**

因此，所有 10 步 × 18 层 × 2 类的 style 可以在加载模型时就算好，存在 GPU 缓冲区里。推理时直接按索引取用，完全不需要重新计算：

```
推理时（没有预计算）：
  每步：
    te = sinusoidal(t)                    ← 计算
    te = SiLU(te @ W_in) @ W_out         ← 计算
    for 18层 × 2次：
      style = te @ W_mod + b_mod          ← 计算（36次矩阵乘法）

推理时（有预计算）：
  每步：
    style_ptr = &precomputed[step, layer, :]  ← 只是一个指针偏移，零计算
```

省去了推理热路径上的 36+ 次矩阵乘法（尽管都是小矩阵，但在 CUDA Graph 内零开销更好）。

---

## 下一步

- [04-data-flow.md](./04-data-flow.md)：了解数据在各组件间如何流动
- [05-key-concepts.md](./05-key-concepts.md)：深入理解 FP8 量化、CUDA Graph 等技术
