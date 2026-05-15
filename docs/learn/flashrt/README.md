# FlashRT 学习文档

欢迎来到 FlashRT 学习文档！本系列文档面向初学者，用中文深入浅出地介绍 FlashRT 的原理与实现。

## 文档目录

| 文档 | 内容 | 适合读者 |
|------|------|---------|
| [01-overview.md](./01-overview.md) | FlashRT 是什么，解决什么问题，核心性能数据 | 所有人 |
| [02-architecture.md](./02-architecture.md) | 整体技术架构，各层职责，设计原则 | 想了解全局的读者 |
| [03-components.md](./03-components.md) | 9大主要组件的详细说明 | 想深入了解各组件的读者 |
| [04-data-flow.md](./04-data-flow.md) | 数据从图像到动作的完整旅程，内存布局 | 想理解推理流程的读者 |
| [05-key-concepts.md](./05-key-concepts.md) | FP8量化、CUDA Graph、FlashAttention等关键技术 | 想理解技术原理的读者 |
| [06-demo-walkthrough.md](./06-demo-walkthrough.md) | 完整 Demo 串讲，每步的输入/输出/作用 | 想上手使用的读者 |

## 推荐阅读顺序

### 快速上手（30分钟）
1. [01-overview.md](./01-overview.md) — 了解 FlashRT 是什么
2. [06-demo-walkthrough.md](./06-demo-walkthrough.md) — 直接看完整流程

### 深入理解（2小时）
1. [01-overview.md](./01-overview.md)
2. [02-architecture.md](./02-architecture.md)
3. [03-components.md](./03-components.md)
4. [04-data-flow.md](./04-data-flow.md)
5. [05-key-concepts.md](./05-key-concepts.md)
6. [06-demo-walkthrough.md](./06-demo-walkthrough.md)

## 关键知识点速查

| 如果你想了解... | 去看... |
|----------------|---------|
| FlashRT 支持哪些模型 | [01-overview.md](./01-overview.md) |
| 代码目录结构 | [02-architecture.md](./02-architecture.md) |
| load_model 做了什么 | [03-components.md](./03-components.md) |
| VLAModel / predict 怎么用 | [03-components.md](./03-components.md) |
| 图像是怎么变成动作的 | [04-data-flow.md](./04-data-flow.md) |
| Phase A/B/C 是什么 | [04-data-flow.md](./04-data-flow.md) |
| FP8 量化的原理 | [05-key-concepts.md](./05-key-concepts.md) |
| 为什么需要校准 | [05-key-concepts.md](./05-key-concepts.md) |
| CUDA Graph 是什么 | [05-key-concepts.md](./05-key-concepts.md) |
| 完整的 3 行 API 示例 | [06-demo-walkthrough.md](./06-demo-walkthrough.md) |
| 每步的耗时分布 | [06-demo-walkthrough.md](./06-demo-walkthrough.md) |

## FlashRT 代码位置

```
/root/tw/megakernels_mono/flashrt/
├── flash_rt/api.py              # 公共 API：load_model() + VLAModel
├── flash_rt/hardware/           # GPU 检测与分发
├── flash_rt/frontends/          # 权重加载、量化、CUDA Graph 录制
├── flash_rt/models/             # 推理流水线（框架无关）
├── flash_rt/core/               # CUDA Graph、FP8 校准等基础设施
├── csrc/                        # C++/CUDA 内核
└── examples/quickstart.py       # 最简示例
```
