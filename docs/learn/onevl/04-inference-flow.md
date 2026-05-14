# 推理流程详解

## 流程总览

```
测试数据 (.json/.jsonl)
        │
        ▼
  加载模型和处理器
        │
        ▼
  构建 assistant 前缀
  (<latent-vis×4><latent×2><answer>[)
        │
        ▼  对每个样本循环：
  ┌─────────────────────────────┐
  │  1. 读取图像 + 用户提问      │
  │  2. 构建消息（messages）     │
  │  3. 生成输入文本（含 latent  │
  │     前缀，已预填充）         │
  │  4. processor 处理图文输入  │
  │  5. （可选）前向传播获取     │
  │     隐藏状态，运行辅助解码器 │
  │  6. model.generate() 生成   │
  │     轨迹坐标                 │
  │  7. 保存结果                 │
  └─────────────────────────────┘
        │
        ▼
  输出 JSON 文件
```

---

## 第一步：加载模型

代码位置：`infer_onevl.py:549-565`

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path, dtype=torch.bfloat16, trust_remote_code=True)
model.to(device).eval()

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
```

`processor` 包含两个子组件：
- `processor.tokenizer`：文字分词器
- `processor.image_processor`：图像预处理器（调整大小、归一化）

模型权重存储在多个 `.safetensors` 文件中，包含：
- 主干 Qwen3-VL 权重（前缀无特殊标记）
- 语言辅助解码器权重（前缀 `_latent_cot_aux_decoder.`）
- 视觉辅助解码器权重（前缀 `_latent_cot_visual_aux_decoder.`）
- 语言投影层权重（前缀 `_latent_cot_latent_proj.`）
- 视觉投影层权重（前缀 `_latent_cot_visual_latent_proj.`）

---

## 第二步：构建 Latent 前缀

代码位置：`infer_onevl.py:574-589`

```python
latent_block = (
    "<|start-latent-vis|>"
    + "<|latent-vis|>" * 4          # 4 个视觉 latent token
    + "<|end-latent-vis|><|start-latent|>"
    + "<|latent|>" * 2              # 2 个语言 latent token
    + "<|end-latent|><answer>["     # answer 前缀，[ 是轨迹开始符
)
```

这个字符串会被**拼接**到 `apply_chat_template` 生成的对话文本末尾，成为 assistant 的"已知"前缀。

---

## 第三步：构建输入

代码位置：`infer_onevl.py:651-685`

```python
# 构建多模态消息格式
messages = [{"role": "user", "content": [
    {"type": "image", "image": img_path},  # 图像
    {"type": "text", "text": prompt},      # 文字提问
]}]

# 生成对话文本（已包含 latent 前缀）
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
text += assistant_prefix  # 拼接 latent 块

# processor 同时处理图文
inputs = processor(text=[text], images=imgs, return_tensors="pt")
```

**关键**：图像会被 ViT 编码为特殊的 image token，插入到文本序列中图像标记的位置。

---

## 第四步（可选）：辅助解码器解释

代码位置：`infer_onevl.py:688-746`

当开启 `--decoder_explain` 或 `--visual_decoder_explain` 时，需要先做一次完整前向传播来获取隐藏状态：

```python
# 注册 hook 捕获 ViT 嵌入
def _capture_hook(module, args, kwargs):
    _captured['embeds'] = kwargs.get('inputs_embeds').detach()

_hook = model.model.language_model.register_forward_pre_hook(_capture_hook, with_kwargs=True)
fwd_out = model(**inputs, output_hidden_states=True)
_hook.remove()

hidden_states = fwd_out.hidden_states  # 每一层的隐藏状态
```

然后找到 latent token 的位置：

```python
for b in range(batch_size):
    text_pos, vis_pos = compute_inference_latent_positions(
        inputs['input_ids'][b], tokenizer, pattern_ids, marker_component_ids)
```

**位置检测原理**（`infer_onevl.py:168-196`）：

1. 找到所有 `| latent |` 模式（前后都是 `|` 的 `latent` token）→ 语言 latent 位置
2. 找到所有 `| latent -vis` 模式 → 视觉 latent 位置
3. 用 `<|start-latent|>` 标记把两类位置分开
4. 展开相邻的标记符 token（如 `<|start-latent-vis|>` 等）

找到位置后，从最后一层的隐藏状态中提取对应向量，送入辅助解码器：

```python
latent_embeds = last_hidden[b, positions, :]  # 形状: (num_latent, hidden_size)
latent_embeds = latent_proj(latent_embeds)     # 投影到辅助解码器维度

# 拼接 ViT 视觉特征（可选条件）
combined = torch.cat([vit_cond, latent_embeds], dim=0).unsqueeze(0)

# 自回归生成（贪婪解码）
for _ in range(max_tokens):
    logits, past_kv = call_aux_decoder_lm(aux_decoder, cur_embeds, use_cache=True, ...)
    next_id = logits[:, -1, :].argmax(dim=-1)
    if next_id == eos_id: break
    cur_embeds = aux_embedding(next_id).unsqueeze(1)
```

---

## 第五步：生成轨迹

代码位置：`infer_onevl.py:750-795`

```python
gen_outputs = model.generate(
    **inputs,
    max_new_tokens=args.max_new_tokens,
    do_sample=False,  # 贪婪解码
    return_dict_in_generate=True,
    output_scores=True,
)
```

由于 assistant 前缀（含 latent 块）已经被预填充，模型**一次并行处理**所有 latent token，然后**只自回归生成** `<answer>[` 之后的坐标文字。

输出示例：

```
[[0.0, 0.0], [1.2, 0.1], [2.5, 0.3], [3.8, 0.5], [5.1, 0.6]]
```

这是未来几个时间步的 (x, y) 坐标列表。

---

## 第六步：保存结果

代码位置：`infer_onevl.py:773-811`

每个样本保存为一个字典：

```json
{
    "latency": 4.46,
    "messages": [...],
    "GT": "[[0.0, 0.0], [1.1, 0.05], ...]",
    "output_text": "[[0.0, 0.0], [1.2, 0.1], ...]",
    "avg_entropy": 0.1234,
    "avg_log_prob": -0.0567,
    "seq_confidence": 0.9449,
    "decoder_explain": "前方道路清晰，建议保持当前速度...",  // 可选
    "visual_decoder_explain": "<|image start|>..."           // 可选
}
```

同时计算置信度指标：
- `avg_entropy`：生成序列的平均熵（越低越确定）
- `seq_confidence`：序列的平均 token 概率（越高越自信）

---

## 多 GPU 并行推理

`run_infer.sh` 会：
1. 自动检测 GPU 数量
2. 将测试集均分成 N 份
3. 每张 GPU 跑一份子集
4. 最后合并结果

---

## 下一步

- [05-demo-walkthrough.md](05-demo-walkthrough.md)：用一个完整示例串讲整个工作流
