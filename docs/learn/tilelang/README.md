# TileLang 学习文档

> 面向初学者的 TileLang 原理与实现解析（中文版）

## 文档目录

| 文档 | 内容 |
|------|------|
| [01-overview.md](./01-overview.md) | TileLang 是什么、为什么需要它、整体架构与支持的硬件 |
| [02-tile-concept.md](./02-tile-concept.md) | "Tile" 概念详解：它和 CTA 的区别与联系，多层 Tile 结构 |
| [03-compilation-pipeline.md](./03-compilation-pipeline.md) | 编译流水线：Python 代码如何被编译成 GPU 指令 |
| [04-components.md](./04-components.md) | 主要组件详解：语言层、JIT、编译 Pass、后端、IR 系统 |
| [05-demo-walkthrough.md](./05-demo-walkthrough.md) | Demo 串讲：GEMM kernel 从写代码到 GPU 执行的全过程 |
| [06-key-questions.md](./06-key-questions.md) | 深度问题解答（Tile vs CTA、独立调度、LCS抽象、与TileRT的关系） |

## 推荐阅读顺序

**快速入门（30 分钟）**：
1. [01-overview.md](./01-overview.md) — 先建立大局观
2. [05-demo-walkthrough.md](./05-demo-walkthrough.md) — 通过具体例子理解工作流程

**深入理解（2 小时）**：
3. [02-tile-concept.md](./02-tile-concept.md) — 理解 tile 这个核心概念
4. [03-compilation-pipeline.md](./03-compilation-pipeline.md) — 理解编译器做了什么
5. [04-components.md](./04-components.md) — 了解各模块的职责

**专题深度（按需阅读）**：
6. [06-key-questions.md](./06-key-questions.md) — 回答高频深度问题

## 关键概念速查

- **Tile**：数据块，一片矩形数据区域（见 02）
- **CTA**：执行单元，一组可协作的线程 = Thread Block（见 02）
- **Layout Inference**：自动推断寄存器/共享内存布局的核心编译 Pass（见 03、05）
- **Software Pipeline**：T.Pipelined，隐藏内存延迟的关键优化（见 03、05）
- **Fragment**：寄存器片段，被 CTA 内所有线程分摊持有的数据（见 02、04）
- **Warp Specialization**：Producer warp 做 Load，Consumer warp 做 Compute（见 03、06）
