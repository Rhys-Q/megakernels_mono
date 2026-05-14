# 完整示例：一步步追踪 OneVL 的工作流程

我们用一个真实的 APR1 测试样本来演示 OneVL 从输入到输出的完整流程。

---

## 示例场景

**输入**（来自 `test_data/ar1_test.jsonl` 第一行）：

- **图像**：4 帧连续摄像头图像（时间间隔约 0.1 秒）
- **提问**：根据当前图像，预测自车未来可行驶路径
- **运动上下文**：
  - 历史轨迹：过去 8 个时刻的 (x, y) 坐标（米，自车坐标系）
  - 当前速度：10.0 m/s
  - 当前加速度：2.9 m/s²

**GT（标准答案）**：`[[1.0, 0.0], [8.6, 0.7], [14.3, 3.2], [18.1, 7.3], [20.4, 12.9], [21.6, 20.1], [22.7, 28.5], [23.8, 38.0]]`

---

## 步骤 1：图像预处理

**输入**：4 张 JPEG 图像文件路径  
**处理**：

```python
imgs = [Image.open(p).convert("RGB") for p in image_paths]
# processor 内部处理：
# - 调整大小（最大边长 ≤ 1792 像素）
# - 归一化到 [-1, 1]
# - 转换为 patch 序列
```

**产物**：像素张量，形状约 `(4, 3, H, W)`

---

## 步骤 2：构建输入文本

**输入**：用户提问 + latent 前缀字符串  
**处理**：

```python
text = processor.apply_chat_template(messages, ..., add_generation_prompt=True)
# 生成如下格式（伪代码）：
# <|im_start|>system\n你是一个自动驾驶助手<|im_end|>
# <|im_start|>user\n<image><image><image><image>
# Based on the current image, predict...
# history_xy: [[-17.4, 1.0], ..., [0.0, 0.0]]
# current_speed: 10.0 m/s
# ...<|im_end|>
# <|im_start|>assistant\n

text += "<|start-latent-vis|>|latent-vis||latent-vis||latent-vis||latent-vis|<|end-latent-vis|><|start-latent|>|latent||latent|<|end-latent|><answer>["
```

**产物**：完整的多模态输入文本字符串

---

## 步骤 3：Tokenization 和 ViT 编码

**输入**：文本字符串 + 图像张量  
**处理**：

```python
inputs = processor(text=[text], images=imgs, return_tensors="pt")
# inputs 包含：
# - input_ids:       文本 token ID 序列（含图像占位符）
# - attention_mask:  掩码
# - pixel_values:    图像像素值
# - image_grid_thw:  图像分辨率信息（用于位置编码）
```

ViT 编码器将图像转换为特征向量，替换文本中的图像占位符 token：

```
[系统提示 tokens] [用户文字 tokens] [图像特征 tokens × N] [latent 块 tokens] ...
```

**产物**：
- `input_ids`：整数 token ID 序列，shape `(1, seq_len)`
- `pixel_values`：处理后的图像特征，ViT 稍后在模型内部处理

---

## 步骤 4：定位 Latent Token 位置

**输入**：`input_ids` 序列  
**处理**（`infer_onevl.py:707-714`）：

```python
text_pos, vis_pos = compute_inference_latent_positions(
    inputs['input_ids'][0], tokenizer, pattern_ids, marker_component_ids)
# 示例结果（假设序列长度 512）：
# vis_pos  = [490, 491, 492, 493]   # 4 个视觉 latent token 位置
# text_pos = [496, 497]              # 2 个语言 latent token 位置
```

**位置检测原理**：
- 扫描 token 序列，找 `pipe_id, latent_keyword_id, pipe_id` 模式 → 语言 latent
- 找 `pipe_id, latent_keyword_id, vis_suffix_id` 模式 → 视觉 latent
- 展开相邻的标记符 token（`<|start-latent-vis|>` 等也算入位置集合）

**产物**：两个位置列表，指向序列中的 latent token

---

## 步骤 5（标准推理）：一次并行前向 + 自回归生成

这是核心推理步骤，分两个阶段：

### 阶段 A：Prefill（并行处理前缀）

```
[所有前缀 tokens，含图像特征 + latent 块]
              ↓ 一次并行前向传播
[每个 token 位置的隐藏状态向量]
              ↓
latent-vis 位置 [490~493] 的隐藏状态 → 已内化"未来场景"信息
latent 位置 [496~497] 的隐藏状态    → 已内化"推理逻辑"信息
<answer>[ 位置 [498] 的 KV cache   → 已准备好
```

### 阶段 B：自回归生成轨迹

```
[<answer>[ KV cache]
        ↓ 生成第一个坐标 token
[1.0]
        ↓ 生成 ,
[,]
        ↓ 继续...
最终输出：[1.0, 0.0], [8.6, 0.7], [14.3, 3.2], ...
```

```python
gen_outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
```

**为什么这么快？**  
关键在于 latent token 被**预填充**，而非自回归生成。相比显式 CoT 要自回归生成几百个文字 token，OneVL 只需自回归生成最终的坐标数字，大大减少了生成步骤数。

**产物**：
```
输出文本：[1.0, 0.0], [8.5, 0.7], [14.1, 3.1], [17.9, 7.2], [20.2, 12.8], [21.5, 20.0], [22.6, 28.4], [23.7, 37.9]
```

---

## 步骤 5（解释模式）：运行辅助解码器

如果开启 `--decoder_explain` 和 `--visual_decoder_explain`，在步骤 5 之前还有一步：

### 5a：提取 Latent 隐藏状态

```python
fwd_out = model(**inputs, output_hidden_states=True)
hidden_states = fwd_out.hidden_states  # 每层的隐藏状态
last_hidden = hidden_states[-1]        # 最后一层

# 提取视觉 latent 隐藏状态
vis_latent = last_hidden[0, vis_pos, :]   # shape: (4, 2560)

# 提取语言 latent 隐藏状态
txt_latent = last_hidden[0, text_pos, :]  # shape: (2, 2560)
```

### 5b：语言辅助解码器生成推理文字

```python
# 投影到语言辅助解码器维度
txt_latent_proj = lang_proj(txt_latent)   # (2, 2560) → (2, aux_hidden)

# 拼接 ViT 特征（可选）
combined = cat([vit_features, txt_latent_proj], dim=0)  # (N+2, aux_hidden)

# 自回归解码
generated = "前方有施工路障，需要减速并向左变道，预计弯道半径约 20 米"
```

### 5c：视觉辅助解码器生成未来帧 token

```python
# 投影到视觉辅助解码器维度
vis_latent_proj = vis_proj(vis_latent)   # (4, 2560) → (4, vis_hidden)

# 自回归解码（输出 Emu3.5 视觉词表的 token）
# 每帧约 32×40 = 1280 个 token
visual_output = """
<|image start|><|image token|>
<|visual token 45231|><|visual token 12089|>...<|extra_200|>
...（1280 个 token，表示 t+0.5s 的预测画面）
<|image end|>
<|image start|>...（t+1.0s 的预测画面）<|image end|>
"""
```

**产物**：
- `decoder_explain`：人类可读的推理文字
- `visual_decoder_explain`：视觉 token 字符串（可进一步解码成图像）

---

## 步骤 6：视觉 Token 解码为图像（可选）

使用 `scripts/visualize_predict_image_tokens.py`：

```python
# 解析 token 字符串 → 整数网格
grid = parse_token_block(visual_output)  # shape: (40, 32)

# 查码本，获取连续特征图
quant = vq_model.quantize.get_codebook_entry(grid, shape=(1, 40, 32, 256))
# shape: (1, 256, 40, 32)

# VQ-VAE Decoder 重建图像
img = vq_model.decode(quant)  # shape: (1, 3, 320, 256)

# 转换为 PIL Image
arr = ((img[0].permute(1,2,0) + 1.0) * 127.5).clamp(0, 255).numpy()
Image.fromarray(arr.astype(np.uint8)).save("predicted_t+0.5s.png")
```

**产物**：预测的未来帧图像（约 320×256 像素）

---

## 完整流程总结

```
4张摄像头图像 + 运动历史
         │
         ▼
   ViT 编码 + 分词
         │
         ▼
 prefill 所有 token（含 latent 块）
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼（隐藏状态）                      ▼（KV cache）
视觉latent隐藏态  语言latent隐藏态   自回归生成轨迹
    │                │
    ▼（仅解释模式）   ▼（仅解释模式）
视觉辅助解码器    语言辅助解码器
    │                │
    ▼                ▼
未来帧token序列  推理文字
    │
    ▼（可选可视化）
VQ-VAE Decoder
    │
    ▼
预测未来帧图像

最终输出：
- 轨迹坐标：[[1.0, 0.0], [8.5, 0.7], ...]
- 推理文字：前方有左弯，需要减速...（可选）
- 未来帧图像：decoded_from_tokens_00.png（可选）
```

---

## 下一步

- [06-components.md](06-components.md)：各组件的详细参数与配置说明
