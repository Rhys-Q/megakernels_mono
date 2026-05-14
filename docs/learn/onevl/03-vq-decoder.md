# VQ-VAE 视觉分词器：图像如何变成 token？

## 为什么需要把图像变成 token？

语言模型只能处理离散的 token（整数 ID），不能直接处理连续的像素值。  
为了让语言模型能"生成图像"，需要把图像转换成一个整数序列，就像把文字分词一样。

**VQ-VAE**（Vector Quantized Variational Autoencoder，向量量化变分自编码器）就是做这件事的工具。

---

## 直觉理解：像素画

想象你要用有限种颜色的马赛克来重现一幅画：

1. 把画切成很多小格子（patch）
2. 对每个格子，从调色板中找最接近的颜色编号
3. 记录所有格子的颜色编号列表
4. 要还原图像时，用编号查调色板，再拼回去

VQ-VAE 做的事情类似，但"调色板"是一个高维向量的码本，每个格子对应一个特征向量而不是颜色。

---

## Emu3.5 IBQ 的具体结构

OneVL 使用的是 **Emu3.5-VisionTokenizer**，采用 IBQ（Index-Based Quantization）方法：

```
输入图像 (512×512×3)
        │
        ▼
   ┌──────────┐
   │  Encoder │  ← 多层 ResNet + Attention 下采样
   └──────────┘
        │
   特征图 (32×32×z_channels)
        │
        ▼
   ┌──────────┐
   │quant_conv│  ← 1×1 卷积，调整维度到 embed_dim
   └──────────┘
        │
   量化特征 (32×32×embed_dim)
        │
        ▼
   ┌──────────────────────┐
   │ IndexPropagation     │  ← 核心量化模块
   │ Quantize             │
   │  码本: 131072 × 256  │
   └──────────────────────┘
        │
   token 索引网格 (32×32) ← 每个位置是 0~131071 的整数
        │
        ▼ （解码时）
   ┌──────────────┐
   │post_quant_conv│  ← 反向 1×1 卷积
   └──────────────┘
        │
        ▼
   ┌──────────┐
   │  Decoder │  ← 多层 ResNet + Attention 上采样
   └──────────┘
        │
   重建图像 (512×512×3)
```

---

## Encoder 结构详解

Encoder 是一个逐步下采样的 CNN，代码在 `vq_decoder/modules/encoder_decoder.py:117-181`：

```
输入 → conv_in (3×3 卷积，3→ch 通道)
     → 多个下采样阶段：
         - ResnetBlock × num_res_blocks
         - AttnBlock（在特定分辨率加自注意力）
         - Downsample（步长为2，分辨率减半）
     → 中间层 (mid)：
         - ResnetBlock
         - AttnBlock
         - ResnetBlock
     → norm_out → nonlinearity
     → conv_out → 输出特征图
```

**ResnetBlock** 是标准的残差块（带 GroupNorm 归一化和 Swish 激活函数）。  
**AttnBlock** 是自注意力块，让特征图中的每个位置都能看到其他所有位置。

---

## 量化模块（IndexPropagationQuantize）详解

代码在 `vq_decoder/modules/quantize.py:27-113`：

### 前向过程（推理时）

```python
# z: 连续特征图 (B, embed_dim, H, W)
# embedding.weight: 码本 (131072, embed_dim)

# 1. 计算每个位置与所有码本向量的相似度
logits = einsum('b d h w, n d -> b n h w', z, embedding)
# logits: (B, 131072, H, W)

# 2. 找最相似的码本向量索引
soft_one_hot = softmax(logits, dim=1)
ind = soft_one_hot.argmax(dim=1)  # (B, H, W)

# 3. 用索引查码本，获得量化后的特征向量
z_q = embedding[ind]  # (B, H, W, embed_dim) → (B, embed_dim, H, W)
```

### 训练时的技巧：直通估计器（Straight-Through Estimator）

量化操作（argmax）不可微分，无法直接反向传播。解决方案：

```python
# 训练时混合 soft 和 hard one-hot
one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot
# 前向用 hard（精确），反向梯度通过 soft（可微）
```

---

## Decoder 结构详解

Decoder 是 Encoder 的对称镜像，逐步上采样，代码在 `vq_decoder/modules/encoder_decoder.py:184-256`：

```
量化特征图 → conv_in
           → 中间层 (mid)
           → 多个上采样阶段：
               - ResnetBlock × (num_res_blocks + 1)
               - AttnBlock
               - Upsample（最近邻插值 × 2，然后 3×3 卷积）
           → norm_out → nonlinearity
           → conv_out → 输出图像 (值域 [-1, 1])
```

解码图像时，输出值域是 `[-1, 1]`，转换为 `[0, 255]` 的公式：

```python
arr = ((decoded + 1.0) * 127.5).clamp(0, 255)
```

---

## 视觉 token 的文字表示

在 OneVL 的推理输出中，视觉 token 以以下格式储存（`scripts/visualize_predict_image_tokens.py:40-80`）：

```
<|image start|><|image token|>
<|visual token 12345|><|visual token 67890|>...<|extra_200|>
<|visual token 11111|><|visual token 22222|>...<|extra_200|>
...（每行是图像的一行 patch）
<|image end|>
```

每个 `<|visual token XXXXX|>` 中的数字就是码本索引（0~131071）。

解析时用正则表达式提取：

```python
VISUAL_TOKEN_RE = re.compile(r"<\|visual token (\d+)\|>")
```

---

## 加载模型代码

```python
# vq_decoder/loader.py
from omegaconf import OmegaConf
from vq_decoder.ibq import IBQ

cfg = OmegaConf.load("Emu3.5-VisionTokenizer/config.yaml")
model = IBQ(**cfg)
ckpt = torch.load("Emu3.5-VisionTokenizer/model.ckpt", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

---

## 下一步

- [04-inference-flow.md](04-inference-flow.md)：推理流程详解
