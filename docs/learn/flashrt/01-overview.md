# FlashRT 是什么？

## 一句话介绍

FlashRT 是一个专门为**实时机器人控制**场景打造的高性能 AI 推理引擎。它能让 Vision-Language-Action（VLA）模型在极短时间内完成一次推理——比如让机器人"拿起红色积木"这样的指令，从看到图像到输出动作，整个过程只需要 **17~44 毫秒**。

---

## 它解决了什么问题？

### 背景：机器人需要"快速思考"

想象一个机器人正在操作物品。它的大脑（AI 模型）需要：
1. 看清楚摄像头拍到的画面
2. 理解文字指令（"把瓶子放进箱子里"）
3. 计算出手臂每一步该怎么动

这个过程必须足够**快**，否则机器人动作会迟钝、不流畅。

### 普通推理框架的问题

如果用 PyTorch 直接跑 VLA 模型，会碰到这些问题：

| 问题 | 原因 | 影响 |
|------|------|------|
| 推理慢 | 每次推理都要重新调度 GPU 指令 | 延迟高，不适合实时控制 |
| 精度浪费 | 用 FP32/FP16 存储权重，占用内存大 | 无法跑大模型或跑得很慢 |
| CPU 开销大 | Python 解释器逐层调度 | 浪费时间在非 GPU 工作上 |

### FlashRT 的解决方案

FlashRT 用三个核心技术解决了这些问题：

```
普通推理                FlashRT 推理
-----------           -----------
Python调度层A          ┌─────────────────┐
Python调度层B    →     │  CUDA Graph      │
Python调度层C          │  (一条指令完成    │
...（数百层）          │   所有工作)      │
                       └─────────────────┘
耗时：100ms+           耗时：17ms
```

1. **CUDA Graph（计算图录制）**：把整个推理过程录制成一段"程序"，之后每次推理只需"播放"这段程序，消除所有 Python/CPU 调度开销。

2. **FP8 量化**：把原本用 16 位浮点数（BF16）存储的权重压缩成 8 位（FP8），内存减半，计算更快，精度几乎无损。

3. **手写 CUDA 内核**：针对每个操作（归一化、注意力、激活函数）写专用 GPU 程序，比通用框架快得多。

---

## 支持的模型

FlashRT 目前支持以下模型：

| 模型 | 类型 | 用途 |
|------|------|------|
| **Pi0.5** | 扩散模型 VLA | 通用机器人操作 |
| **Pi0** | 扩散模型 VLA | 机器人操作（连续状态） |
| **GROOT N1.6** | DiT + 语言模型 | 通用机器人操作 |
| **Pi0-FAST** | 自回归 VLA | 机器人操作（更快） |
| **Qwen3.6-27B** | 大语言模型 | 通用对话/推理 |

---

## 支持的硬件

| 硬件 | GPU 架构 | 说明 |
|------|----------|------|
| **Jetson AGX Thor** | SM110 | NVIDIA 机器人专用边缘计算模块 |
| **RTX 5090** | SM120 (Blackwell) | 桌面级高端 GPU，支持 FP4 量化 |
| **RTX 4090** | SM89 (Ada) | 桌面级高端 GPU |

---

## 使用有多简单？

只需三行代码：

```python
import flash_rt

model = flash_rt.load_model(checkpoint="/path/to/pi05", framework="torch")
actions = model.predict(images=[camera_image, wrist_image], prompt="pick up the red block")
# actions.shape = (10, 7)  # 10个时间步，7个关节角度
```

`load_model` 会自动：
- 检测你的 GPU 类型
- 加载对应的优化版模型
- 完成 FP8 校准
- 录制 CUDA Graph

之后每次调用 `predict`，就直接"播放"录制好的计算图，得到动作输出。

---

## 核心性能数据

| 模型 | Jetson AGX Thor | RTX 5090 |
|------|:-:|:-:|
| Pi0.5 (2视角) | **44ms** | **17.58ms** |
| Pi0 (2视角) | **46ms** | — |
| GROOT N1.6 | **45ms** | **13.08ms** |
| Pi0-FAST | **8.1ms/token** | **2.39ms/token** |

---

## 下一步

- [02-architecture.md](./02-architecture.md)：了解 FlashRT 的整体技术架构
- [03-components.md](./03-components.md)：深入了解各个主要组件
- [04-data-flow.md](./04-data-flow.md)：理解数据是如何在系统中流动的
- [05-key-concepts.md](./05-key-concepts.md)：掌握 FP8 量化、CUDA Graph 等关键技术
- [06-demo-walkthrough.md](./06-demo-walkthrough.md)：跟着一个完整 Demo 串讲整个工作流程
