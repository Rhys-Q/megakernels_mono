# FlashRT 技术架构

## 整体架构一览

FlashRT 的代码组织非常清晰，按照"职责分离"的原则分成几层：

```
┌─────────────────────────────────────────────────────────┐
│                    用户代码（3行API）                      │
│          flash_rt.load_model() / model.predict()         │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                   API层 (flash_rt/api.py)                 │
│   VLAModel 类  │  load_model() 函数  │  自动检测GPU        │
└──────┬──────────────────────────────────────────────────┘
       │
┌──────▼───────────────────────────────────────────────────┐
│              硬件分发层 (flash_rt/hardware/)               │
│  detect_arch()  │  _PIPELINE_MAP 分发表  │  插件注册        │
└──────┬────────────┬──────────────────────────────────────┘
       │            │
┌──────▼──────┐  ┌──▼──────────────────────────────────────┐
│  前端层      │  │           前端层                          │
│ (frontends/ │  │  框架特定：权重加载、FP8量化、CUDA Graph    │
│ torch/ jax/ │  │  Pi05TorchFrontendRtx / Pi05JaxFrontendThor│
└──────┬──────┘  └──────────────┬──────────────────────────┘
       │                        │
┌──────▼────────────────────────▼──────────────────────────┐
│              模型流水线层 (flash_rt/models/)                │
│   框架无关的前向传播  │  Pi05Pipeline / GrootPipeline        │
│   只接收设备指针（int），不依赖 torch/jax                   │
└──────────────────────────────┬───────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────┐
│             硬件后端层 (flash_rt/hardware/thor/rtx/)        │
│   注意力后端 (AttentionBackend)  │  Thor vs RTX 实现        │
└──────────────────────────────┬───────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────┐
│                    C++ CUDA 内核层 (csrc/)                  │
│   flash_rt_kernels.so  │  flash_rt_fa2.so                  │
│   RMSNorm / RoPE / FP8量化 / FlashAttention2 等             │
└──────────────────────────────────────────────────────────┘
```

---

## 各层的职责

### 1. API 层（用户直接接触的部分）

**文件位置**：`flash_rt/api.py`

这是用户唯一需要关心的层。它做两件事：

- **`load_model()`**：加载模型，返回一个 `VLAModel` 对象
- **`VLAModel`**：封装了推理接口，提供 `predict()` 方法

```python
# 这就是整个API层的使用方式
model = flash_rt.load_model(checkpoint="...", framework="torch")
actions = model.predict(images=[img], prompt="pick up block")
```

### 2. 硬件分发层

**文件位置**：`flash_rt/hardware/__init__.py`

这一层就像一个"路由器"——根据你的 GPU 类型和想用的框架，把请求转发给正确的实现：

```python
# 分发表的核心逻辑（简化版）
_PIPELINE_MAP = {
    ("pi05", "torch", "thor"):       "Pi05TorchFrontendThor",
    ("pi05", "torch", "rtx_sm120"): "Pi05TorchFrontendRtx",
    ("pi05", "jax",   "thor"):       "Pi05JaxFrontendThor",
    ("groot", "torch", "rtx_sm120"): "GrootTorchFrontendRtx",
    # ... 更多组合
}
```

GPU 检测逻辑也在这里：
- SM 11.0 → Jetson AGX Thor
- SM 12.0 → RTX 5090 (Blackwell)
- SM 8.9  → RTX 4090 (Ada)

### 3. 前端层（框架特定代码）

**文件位置**：`flash_rt/frontends/torch/` 或 `flash_rt/frontends/jax/`

这是最"重"的一层，负责所有需要 torch 或 jax 才能完成的工作：

| 工作 | 具体操作 |
|------|---------|
| **加载权重** | 从 safetensors 文件读取，转换格式 |
| **FP8 量化** | 把 BF16 权重压缩成 FP8 |
| **校准** | 收集激活值的最大值，确定量化参数 |
| **CUDA Graph 录制** | 把整个推理过程录制成可重放的图 |

### 4. 模型流水线层（框架无关代码）

**文件位置**：`flash_rt/models/pi05/pipeline_rtx.py` 等

这是推理的核心逻辑。**关键设计**：这一层完全不依赖 torch 或 jax，只接收整数指针（device pointers），因此可以安全地被 CUDA Graph 捕获。

```python
# 流水线只知道指针，不知道 tensor 是什么
class Pi05Pipeline:
    def run_pipeline(self, stream: int):
        # A: 视觉编码（27层 SigLIP）
        self._vision_encoder(stream)
        # B: 语言+视觉融合（18层 Gemma-2B）
        self._transformer_encoder(stream)
        # C: 动作解码（10步 × 18层 Gemma-300M）
        self._transformer_decoder(stream)
```

### 5. 硬件注意力后端

**文件位置**：`flash_rt/hardware/thor/attn_backend.py` 或 `rtx/attn_backend.py`

注意力计算在不同硬件上使用不同实现：

- **Thor**：使用 CUTLASS FMHA（针对 SM110 优化）
- **RTX（SM120/SM89）**：使用供应商内置的 FlashAttention-2

所有后端都实现同一个协议接口（`AttentionBackend`），流水线层不需要关心具体用了哪种实现。

### 6. C++ CUDA 内核层

**文件位置**：`csrc/`

这是最底层，包含所有手写的 GPU 内核：

| 内核 | 功能 |
|------|------|
| `norm.cu` | RMS 归一化 |
| `rope.cu` | 旋转位置编码 |
| `quantize.cu` | FP8 E4M3 量化 |
| `fusion.cu` | 融合操作（norm + 量化合并） |
| `patch_embed.cu` | SigLIP 图像分块嵌入 |
| `decoder_fused.cu` | 解码器融合操作 |

编译后生成两个共享库：
- `flash_rt_kernels.so`：所有自定义内核（约 3MB）
- `flash_rt_fa2.so`：FlashAttention-2（约 135MB，仅 RTX）

---

## 目录结构总览

```
flashrt/
├── flash_rt/                    # 主 Python 包
│   ├── api.py                   # 公共 API 入口
│   ├── hardware/                # 硬件检测与分发
│   │   ├── __init__.py          # detect_arch() + _PIPELINE_MAP
│   │   ├── thor/                # Jetson AGX Thor 实现
│   │   └── rtx/                 # RTX GPU 实现
│   ├── frontends/               # 框架特定前端
│   │   ├── torch/               # PyTorch 实现
│   │   └── jax/                 # JAX 实现
│   ├── models/                  # 框架无关流水线
│   │   ├── pi05/                # Pi0.5 模型
│   │   ├── pi0/                 # Pi0 模型
│   │   ├── groot/               # GROOT 模型
│   │   └── pi0fast/             # Pi0-FAST 模型
│   ├── executors/               # 声明式权重加载器
│   ├── core/                    # 共享基础设施
│   │   ├── cuda_graph.py        # CUDA Graph 录制/重放
│   │   ├── calibration.py       # FP8 校准
│   │   └── rl/                  # 强化学习推理
│   └── configs/                 # 模型配置文件 (YAML)
├── flash_wm/                    # 世界模型 (BAGEL)
├── csrc/                        # C++/CUDA 内核源码
├── examples/                    # 使用示例
└── tests/                       # 测试文件
```

---

## 设计哲学

FlashRT 有几个重要的设计原则，理解这些可以帮助你更好地理解代码：

### 原则1：流水线只接受指针

流水线层（`models/`）不接受 torch tensor 或 jax array，只接受整数形式的设备指针。这样做是为了能被 CUDA Graph 安全捕获——一旦 Python 对象参与了计算，CUDA Graph 就可能出错。

### 原则2：权重加载是声明式的

权重加载不是写一堆 `weights["layer0.qkv"] = ...`，而是定义一个 `ModelWeightSpec` 规格，然后让 `WeightLoader` 自动执行。这使得 torch 和 jax 的权重加载逻辑可以共享同一份规格。

### 原则3：注意力是可插拔的

通过 `AttentionBackend` 协议，流水线代码不需要知道底层是 FlashAttention-2 还是 CUTLASS，只需调用 `attn_backend.run(...)` 即可。

### 原则4：校准结果会被缓存

FP8 校准的结果保存在 `~/.flash_rt/calibration/` 目录下，下次加载同一个模型时不需要重新校准。

---

## 下一步

- [03-components.md](./03-components.md)：深入了解各个主要组件的实现细节
