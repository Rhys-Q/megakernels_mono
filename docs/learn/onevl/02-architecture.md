# OneVL 技术架构详解

## 整体架构图

```
输入图像
   │
   ▼
┌─────────────────────────────────────┐
│          Qwen3-VL-4B 主干模型        │
│                                     │
│  ┌──────────┐    ┌────────────────┐ │
│  │ ViT 视觉  │    │  语言模型      │ │
│  │ 编码器    │───▶│ (LLM)         │ │
│  └──────────┘    └───────┬────────┘ │
└─────────────────────────┼──────────┘
                           │ 隐藏状态
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   视觉 latent 位置  语言 latent 位置   答案位置
   (4个token)       (2个token)
          │                │
    （仅训练时）      （仅训练时）
          ▼                ▼
   ┌─────────────┐  ┌─────────────┐
   │ 视觉辅助    │  │ 语言辅助    │
   │ 解码器      │  │ 解码器      │
   │ (Visual Aux)│  │ (Lang Aux)  │
   └──────┬──────┘  └──────┬──────┘
          │                │
          ▼                ▼
   预测未来帧 token   重建推理文字
   (t+0.5s, t+1.0s)
```

---

## 组件一：Qwen3-VL-4B 主干模型

OneVL 基于 **Qwen3-VL-4B-Instruct**，这是一个多模态大语言模型，包含两个子模块：

### 1.1 ViT 视觉编码器

负责将输入图像转换成向量序列：

```
输入图像 (H × W × 3 像素)
    ↓  分块（patch）
图像 patch 序列
    ↓  Transformer 处理
视觉特征向量序列 (每个 patch 对应一个向量)
```

这些视觉特征后续会传给语言模型作为上下文，也会在辅助解码器中用作条件输入。

### 1.2 语言模型（LLM）

接收用户提示 + 视觉特征，按顺序处理以下 token 序列：

```
[系统提示] [用户问题] [图像特征] [assistant前缀]
    ↓
[<|start-latent-vis|>] [latent-vis×4] [<|end-latent-vis|>]
    ↓
[<|start-latent|>] [latent×2] [<|end-latent|>]
    ↓
[<answer>] [轨迹坐标...]
```

每个 latent token 位置都会输出一个**隐藏状态向量**（维度 = `hidden_size`），这就是后续辅助解码器的输入。

---

## 组件二：Latent Token 接口

### 为什么用普通词表 token？

OneVL **不新增特殊 token**，而是用现有词表中的普通词汇拼出标记：

- `|latent|` 对应三个 token：`|`、`latent`、`|`
- `|latent-vis|` 对应四个 token：`|`、`latent`、`-vis`、（加上前后界定符）

代码中通过模式匹配来定位这些位置（`infer_onevl.py:76-196`）：

```python
def _find_latent_keyword_positions(ids_list, latent_keyword_id, pipe_id):
    # 找到 | latent | 模式的位置
    for i in range(1, n - 1):
        if (ids_list[i] == latent_keyword_id
                and ids_list[i - 1] == pipe_id
                and ids_list[i + 1] == pipe_id):
            positions.append(i)
```

### Latent 块的完整格式

推理时，模型的 assistant 回复前缀是这样的字符串：

```
<|start-latent-vis|>|latent-vis||latent-vis||latent-vis||latent-vis|<|end-latent-vis|>
<|start-latent|>|latent||latent|<|end-latent|>
<answer>[
```

这段文字被**预填充（prefill）**到上下文，模型在一次并行前向传播中同时处理所有 latent token，然后**只生成** `<answer>` 后的轨迹坐标。

---

## 组件三：视觉辅助解码器（Visual Auxiliary Decoder）

### 作用

给定视觉 latent token 的隐藏状态，预测未来 0.5 秒和 1.0 秒的场景图像（以视觉 token 序列形式输出）。

### 结构

```
视觉 latent 隐藏状态 (4×hidden_size)
    ↓  投影层（Linear→GELU→Linear→LayerNorm）
    ↓  映射到视觉辅助解码器的隐藏维度
+ （可选）ViT 视觉特征拼接
    ↓
视觉辅助解码器（也是 Qwen3-VL 结构）
    ↓  自回归生成
视觉 token 序列（Emu3.5 IBQ 词表，约 131k 个 token）
```

投影层代码（`infer_onevl.py:258-273`）：

```python
proj = nn.Sequential(
    nn.Linear(in_dim, in_dim),
    nn.GELU(),
    nn.Linear(in_dim, out_dim),
    nn.LayerNorm(out_dim),
)
```

### 视觉 token 是什么？

视觉 token 来自 **Emu3.5 IBQ（Index-Based Quantization）** 视觉分词器。  
这是一个 VQ-VAE（向量量化变分自编码器），把图像压缩成离散的 token 序列，每个 token 是一个整数索引，指向一个包含 131072 个向量的码本（codebook）。

```
原始图像 → Encoder → 特征图 → 量化 → token 索引网格 (H×W)
token 索引网格 → 查码本 → 特征图 → Decoder → 重建图像
```

---

## 组件四：语言辅助解码器（Language Auxiliary Decoder）

### 作用

给定语言 latent token 的隐藏状态，重建人类可读的推理文字（Chain-of-Thought）。

### 结构

与视觉辅助解码器相似，但输出的是文字 token（普通语言词表）：

```
语言 latent 隐藏状态 (2×hidden_size)
    ↓  投影层
+ （可选）ViT 视觉特征拼接
    ↓
语言辅助解码器（也是 Qwen3-VL 结构）
    ↓  自回归生成
推理文字（如"前方有施工区域，需要减速..."）
```

---

## 训练 vs 推理的区别

| 阶段 | 辅助解码器 | 速度 |
|------|:---:|:---:|
| 训练 | 启用（提供监督信号） | 慢（需要解码） |
| 推理 | 丢弃（不运行） | 快（等同直接输出答案） |
| 推理 + 解释模式 | 可选启用 | 稍慢（用于可视化/调试） |

推理时**默认不运行**辅助解码器，只有加了 `--decoder_explain` 或 `--visual_decoder_explain` 参数才会启用（主要用于调试和可视化）。

---

## 下一步

- [03-vq-decoder.md](03-vq-decoder.md)：VQ-VAE 视觉分词器详解
- [04-inference-flow.md](04-inference-flow.md)：推理流程详解
