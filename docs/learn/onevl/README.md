# OneVL 学习文档索引

本系列文档用中文，从零开始讲解 OneVL 自动驾驶 VLA 框架的原理与实现。

## 阅读顺序

| 文档 | 内容 | 适合谁 |
|------|------|--------|
| [01-overview.md](01-overview.md) | OneVL 是什么，解决了什么问题，三种 CoT 范式对比 | 先从这里开始 |
| [02-architecture.md](02-architecture.md) | 技术架构：主干模型、Latent Token、双辅助解码器 | 想了解整体设计 |
| [03-vq-decoder.md](03-vq-decoder.md) | VQ-VAE 视觉分词器：图像如何变成 token | 想了解视觉编解码 |
| [04-inference-flow.md](04-inference-flow.md) | 推理流程：从图像到轨迹的每一步代码 | 想看具体实现 |
| [05-demo-walkthrough.md](05-demo-walkthrough.md) | 用真实示例串讲完整工作流，每步产物 | 想要端到端理解 |
| [06-components.md](06-components.md) | 各组件参数、权重布局、benchmark 差异 | 想上手配置和调试 |

## 核心概念速查

**Latent Token**：放在 assistant 回复前的几个特殊位置 token，主干模型在这些位置产生的隐藏状态会被辅助解码器读取。推理时只做 prefill，不自回归生成，所以不增加延迟。

**双辅助解码器**：
- **视觉辅助解码器**（Visual Aux Decoder）：读取视觉 latent 隐藏状态，预测未来 0.5s/1.0s 的场景 token
- **语言辅助解码器**（Language Aux Decoder）：读取语言 latent 隐藏状态，重建 CoT 推理文字

**IBQ VQ-VAE**：Emu3.5 的图像分词器，把图像压缩为整数 token 序列（码本大小 131072），可逆向解码回图像。

**Prefill 推理**：所有 latent token 在一次并行前向传播中处理完毕，只有最终轨迹坐标需要自回归生成，速度等同 "直接输出答案" 的 AR 模型。
