# 任务描述
请帮我分析mirage repo的原理与实现。
我需要学习：
- 技术原理
- 实现细节
- 技术架构
- 主要组件
- 组件交互
- 系统流程
- 用一个demo来串讲工作流程，每一步的作用，每一步的产物，以及最终的输出结果。

最终，你需要用中文，将作者当成一个小白，将上述内容写成多个文档，争取让这个小白也能理解。
文档放在docs/learn/mirage/目录下。

# QA1
我在阅读时，有一些疑问：
1. mirage中，用户负责定义计算图。mirage会将计算图编译成任务图。任务图的节点 是什么？是一个cuda kernel？还是只是cuda kernel 中的一个thread block？即mirage runtime 调度对象是一个kernek 还是一个thread block。
2. mirage event 同步是怎么实现的？依赖cuda的什么同步机制？
3. mirage的kernel 是怎么生成的？ 我不太能想象如何将一个kernel 拆分为多个thread block，然后单独调度这些thread block。

# QA2
1. mirage 支持动态shape吗？
2. mirage的编译产物是什么？即编译产物ABI。
3. mirage runtime 是怎么工作的？怎样加载编译产物？如何将任务图交给scheduler？因为有多个scheduler，这里面是怎么工作的？scheduler又是怎样分配任务的？你可以用一个简单的demo模型来进行详细说明吗？