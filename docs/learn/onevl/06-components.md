# 各组件详解与参数说明

## 组件总览

```
OneVL 系统
├── 主干模型 (Qwen3-VL-4B)
│   ├── ViT 视觉编码器
│   └── 语言模型 (LLM)
├── 辅助组件（仅训练时起监督作用）
│   ├── 语言辅助解码器 + 投影层
│   └── 视觉辅助解码器 + 投影层
└── 视觉分词器 (Emu3.5 IBQ VQ-VAE)
    ├── Encoder（图像→token）
    ├── IndexPropagationQuantize（量化码本）
    └── Decoder（token→图像）
```

---

## 组件一：主干模型（Qwen3-VL-4B）

| 参数 | 值 |
|------|-----|
| 模型类 | `Qwen3VLForConditionalGeneration` |
| 参数量 | ~4B |
| 隐藏维度 (`hidden_size`) | 2560 |
| 精度 | bfloat16 |
| 图像 token ID | `model.config.image_token_id` |
| 最大图像尺寸 | 1792 × 1792 像素 |

**加载方式**（`infer_onevl.py:549-557`）：

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path, dtype=torch.bfloat16, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
processor.image_processor.max_pixels = 1792 * 1792
```

---

## 组件二：Latent Token 配置

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--num_latent` | 2 | 语言 latent token 数量 |
| `--num_latent_vis` | 4 | 视觉 latent token 数量 |

**Latent 块格式**（`infer_onevl.py:574-589`）：

```
<|start-latent-vis|>|latent-vis|×N<|end-latent-vis|>
<|start-latent|>|latent|×M<|end-latent|>
<answer>[前缀
```

这些 token 使用现有词表中的普通 token 拼接而成，无需扩展词表。

---

## 组件三：投影层（Projection）

从主干模型的隐藏维度映射到辅助解码器的隐藏维度，结构固定：

```python
# infer_onevl.py:260-266
nn.Sequential(
    nn.Linear(in_dim, in_dim),     # 恒等变换维持维度
    nn.GELU(),                      # 激活函数
    nn.Linear(in_dim, out_dim),    # 维度适配
    nn.LayerNorm(out_dim),         # 归一化
)
```

| 参数 | 来源 |
|------|------|
| `in_dim` | 主干模型 `hidden_size` = 2560 |
| `out_dim` | 辅助解码器 `hidden_size` |

**权重前缀**：
- 语言投影：`_latent_cot_latent_proj.`
- 视觉投影：`_latent_cot_visual_latent_proj.`

---

## 组件四：语言辅助解码器

| 参数 | 说明 |
|------|------|
| 架构 | 与主干模型相同（Qwen3-VL） |
| 权重前缀 | `_latent_cot_aux_decoder.` |
| 输入 | 语言 latent 隐藏状态（经投影）+ 可选 ViT 特征 |
| 输出 | 推理文字（普通语言 token） |

**关键参数**（`infer_onevl.py:517-522`）：

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--c_thought` | 2 | 送入辅助解码器的 latent token 数 |
| `--max_explain_tokens` | 512 | 最大生成 token 数 |
| `--aux_visual_condition` | False | 是否拼接 ViT 特征作为条件 |

---

## 组件五：视觉辅助解码器

| 参数 | 说明 |
|------|------|
| 架构 | 与主干模型相同（Qwen3-VL） |
| 权重前缀 | `_latent_cot_visual_aux_decoder.` |
| 输入 | 视觉 latent 隐藏状态（经投影）+ 可选 ViT 特征 |
| 输出 | 视觉 token 序列（Emu3.5 IBQ 词表，约 131k）|

**关键参数**（`infer_onevl.py:527-535`）：

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--c_thought_visual` | 4 | 送入视觉辅助解码器的 latent token 数 |
| `--max_visual_tokens` | 1024 | 最大生成视觉 token 数 |
| `--visual_aux_visual_condition` | False | 是否拼接 ViT 特征 |

**视觉 tokenizer 路径**：默认为脚本同目录下的 `visual_tokenizer/`，也可通过 `--visual_aux_tokenizer_path` 指定。

---

## 组件六：Emu3.5 IBQ VQ-VAE

| 参数 | 值 |
|------|-----|
| 码本大小 (`n_embed`) | 131072 |
| 嵌入维度 (`embed_dim`) | 256 |
| 配置文件 | `Emu3.5-VisionTokenizer/config.yaml` |
| 权重文件 | `Emu3.5-VisionTokenizer/model.ckpt` |

**核心接口**（`vq_decoder/ibq.py:31-45`）：

```python
# 编码（图像→量化特征+token索引）
quant, emb_loss, info = model.encode(image_tensor)

# 解码（量化特征→图像）
reconstructed = model.decode(quant)

# 直接从 token 索引解码
reconstructed = model.decode_code(token_indices, shape=(B, H, W, embed_dim))
```

---

## 各 Benchmark 的参数差异

| Benchmark | `answer_prefix` | `prefix_k` | 说明 |
|-----------|:---:|:---:|------|
| NAVSIM | `[` | 0 | 单层列表，如 `[x, y]` |
| APR1 | `[[` | 0 | 嵌套列表，如 `[[x, y], ...]` |
| ROADWork | `[[` | 可设为 >0 | 支持 GT 路点预填充 |
| Impromptu | `[[` | 0 | 嵌套列表 |

---

## 权重文件布局

OneVL checkpoint 目录结构：

```
/path/to/OneVL-checkpoint/
├── config.json                  # Qwen3-VL 配置
├── tokenizer.json               # 分词器
├── model-00001-of-XXXXX.safetensors
├── model-00002-of-XXXXX.safetensors
├── ...                          # 所有权重都在这里
│   ├── model.*                  # 主干模型权重
│   ├── _latent_cot_aux_decoder.*      # 语言辅助解码器
│   ├── _latent_cot_visual_aux_decoder.*  # 视觉辅助解码器
│   ├── _latent_cot_latent_proj.*     # 语言投影层
│   └── _latent_cot_visual_latent_proj.*  # 视觉投影层
```

`collect_state_dict_from_safetensors` 函数（`infer_onevl.py:203-210`）通过前缀过滤来分离各组件的权重：

```python
def collect_state_dict_from_safetensors(ckpt_dir, prefix):
    result = {}
    for sf in sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors'))):
        sd = load_file(sf)
        for k, v in sd.items():
            if k.startswith(prefix):
                result[k[len(prefix):]] = v  # 去掉前缀
    return result
```

---

## 下一步

返回 [01-overview.md](01-overview.md) 温习整体概念，或查看具体的推理命令示例。
