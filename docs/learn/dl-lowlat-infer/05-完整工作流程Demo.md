# 完整工作流程 Demo

> 本文以 `lstm_s` 模型为例，从头到尾串讲一次完整的推理流程。
> 我们使用最优化的模式：**持久化内核 + AsSingleStepGemv 算法**。

---

## 场景设定

假设我们要对一个实时金融信号做 LSTM 推理：

- 模型：`lstm_s`（2层，输入128维，隐藏96维，64个时间步）
- 算法：`AsSingleStepGemv`（持久化内核，最低延迟）
- 精度：FP16
- 硬件：单块 NVIDIA H100 GPU（PCIe 系统）

```bash
nvLstmInf --precision=16 --algo=AsSingleStepGemv \
    lstm_s ./data/lstm_s ./data/lstm_s.npy 30 ./output
```

---

## 阶段一：准备阶段

### 步骤 1：解析参数，创建配置

**位置**：`main.cu`

**输入**：命令行参数

**产物**：`ModelConfig` 结构体

```cpp
ModelConfig config;
config.algorithm = Algo::AsSingleStepGemv;  // 持久化内核模式
config.data_type = Data_kind::FP16;          // 半精度
config.as_supported = true;                  // 永久自旋
config.use_gdrcopy = false;                  // 不启用 GDRCopy
config.sm_arch = 90;                         // Hopper 架构
config.sm_count = 132;                       // H100 有 132 个 SM
```

---

### 步骤 2：加载模型权重

**位置**：`ModelDef::LoadFromDir()`

**输入**：`./data/lstm_s/` 目录下的 `.npy` 文件

**产物**：`ModelDef` 对象，包含所有权重矩阵

磁盘上的文件：
```
data/lstm_s/
├── lstm_0_W.npy  # 第0层输入权重 W: (384, 128) in FP32
├── lstm_0_R.npy  # 第0层循环权重 U: (384, 96)  in FP32
├── lstm_0_B.npy  # 第0层偏置 b: (384,)          in FP32
├── lstm_1_W.npy  # 第1层输入权重 W: (384, 96)   in FP32
├── lstm_1_R.npy  # 第1层循环权重 U: (384, 96)   in FP32
├── lstm_1_B.npy  # 第1层偏置 b: (384,)           in FP32
├── mul.npy        # 聚合权重: (96,)              in FP32
└── add.npy        # 聚合偏置: (1,)               in FP32
```

> **为什么是 384×128 而不是 4×96×128 = 49152？**
> 384 = 4 × 96（4 个 gate，每个 96 维）。这4个gate按 **iofc** 顺序拼接：
> `[i_gate(96), o_gate(96), f_gate(96), c_gate(96)]`

---

### 步骤 3：加载输入数据

**位置**：`DataIterator` 初始化

**输入**：`./data/lstm_s.npy`（1000条×128维的随机输入向量）

**产物**：主机内存中的输入数据数组

```
data/lstm_s.npy: shape=(1000, 128), dtype=float16
第 0 条: [0.42, -0.13, 0.89, ..., 0.07]  # 128个特征值
第 1 条: [-0.21, 0.56, 0.34, ..., -0.88]
...
```

---

### 步骤 4：初始化 CUDA 环境

**位置**：`CudaEnv` 构造

**产物**：CUDA 绿色上下文、流、cuBLAS 句柄

```
GPU 资源分配：
- 绿色上下文：为 lstm_s 分配 2 个 SM
- 高优先级流：用于推理计算
- Pinned memory：用于 CPU-GPU 信号传递（约 32 字节）
```

---

### 步骤 5：上传权重到 GPU

**位置**：`ModelAsSingleStepGemv` 构造和初始化

**输入**：`ModelDef` 中的 FP32 权重

**产物**：GPU 显存中的 FP16 权重矩阵

```
GPU 显存布局（lstm_s，FP16）：

权重区域（只读，整个测试期间不变）：
  layer0.W: 384×128 个 FP16 = 98,304 字节
  layer0.U: 384×96  个 FP16 = 73,728 字节
  layer0.B: 384 个 FP16     = 768 字节
  layer1.W: 384×96  个 FP16 = 73,728 字节
  layer1.U: 384×96  个 FP16 = 73,728 字节
  layer1.B: 384 个 FP16     = 768 字节
  mul:       96 个 FP32     = 384 字节
  add:        1 个 FP32     = 4 字节

状态区域（每步更新）：
  hx[2][64]: 2层 × 64步 × 96个FP16 = 24,576 字节
  cx[2][64]: 2层 × 64步 × 96个FP16 = 24,576 字节

I/O 区域（每步覆写）：
  dev_x: 128 个 FP16 = 256 字节
  host_signal: 1 个 float（pinned memory）= 4 字节
  host_start:  1 个 int  （pinned memory）= 4 字节
```

---

### 步骤 6：填充内核参数

**位置**：`ModelAsSingleStepGemv::InitParams()`

**产物**：`Lstm_s_bpt_params` 结构体，包含所有指针

```cpp
Lstm_s_bpt_params params;

// I/O 指针
params.gmem_input = dev_x;           // GPU 输入缓冲区
params.gmem_h_out = host_signal;     // CPU 可见的输出信号（pinned）
params.gmem_dot_prod_weights = dev_mul_weights;

// 每层权重
params.gmem_weights[0] = dev_W_layer0;
params.gmem_weights[1] = dev_W_layer1;
params.gmem_biases[0]  = dev_B_layer0;
params.gmem_biases[1]  = dev_B_layer1;

// 每层状态
params.gmem_c_in[0]  = &dev_cx[0][63];  // 指向最后一个时间步
params.gmem_c_out[0] = &dev_cx[0][63];
params.gmem_gemm_h[0] = &dev_hx[0][63];
// ... 第1层类似 ...

// 同步信号
params.host_ld_complete = host_signal; // CPU 轮询这里
params.gmem_start = dev_start_signal;  // GPU 等待这里

// 配置
params.always_spin = true;    // 永久自旋
params.use_gdrcopy = false;
params.sm_arch = 90;
params.sm_count = 2;  // 绿色上下文分配的 SM 数
```

---

### 步骤 7：启动持久化内核

**位置**：`ModelAsSingleStepGemv::PreProcess()`

```cpp
// 只调用一次！
run_lstm_s_bpt(params, stream);

// 等待 GPU 发出"就绪"信号
while (*host_signal != HOST_SPIN_0_UNTIL_SIGNAL_VALUE) {
    // CPU 自旋等待（约 10-50µs，只等一次）
}
```

**此刻 GPU 端发生的事情**（`gemv_lstm_bp_impl.cu`）：

```
1. 单个 CUDA block 启动（含 32 个 warp，共 1024 个线程）
2. 协作：所有线程把权重矩阵从显存加载到共享内存
   - layer0.W 加载到 smem（384×128 × 2字节 = 98KB）
   - layer0.U 加载到 smem（384×96 × 2字节 = 73KB）
   - layer0.B 加载到寄存器
   - layer1.W、layer1.U、layer1.B 同样加载
   （H100 每个 SM 有 228KB 共享内存，lstm_s 正好放得下）
3. Thread rank 0 原子写入"就绪"信号到 host_signal
4. 进入自旋等待循环...
```

---

## 阶段二：推理循环

以下循环每次处理一个新到来的时间步，重复数千次。

### 步骤 8：获取新输入

**位置**：`DataIterator::Next()`

**输入**：数据数组（循环遍历）

**产物**：指向第 k 条输入向量的指针

```
第1次推理：x_1 = [0.42, -0.13, 0.89, ..., 0.07]  (128维, FP16)
第2次推理：x_2 = [-0.21, 0.56, 0.34, ..., -0.88] (128维, FP16)
...（每秒约 10 万次）
```

---

### 步骤 9：滑动窗口状态更新

**位置**：内核内部，每次推理前执行

**作用**：把状态数组向前移动一位，腾出最新位置

```
【推理前的隐藏状态（共64个时间步）】
hx[0]: [h_oldest, h_1, h_2, ..., h_62, h_most_recent]
         ↑ 即将被丢弃                    ↑ 这是上一步的输出

【滑动后】
hx[0]: [0,        h_oldest, h_1, ..., h_62, h_most_recent]
         ↑ 新的最老位置（重置为0）         

实际上是把指针后移一位，不真正移动数据：
  params.gmem_c_in[0] = &dev_cx[0][step_idx];
  params.gmem_gemm_h[0] = &dev_hx[0][step_idx];
  step_idx = (step_idx + 1) % 64;
```

---

### 步骤 10：复制输入到 GPU

**位置**：`ModelInstance` 主循环

**输入**：主机上的 128维 FP16 向量

**产物**：`dev_x`（GPU 显存）中已更新的输入

```cpp
cudaMemcpyAsync(dev_x, host_x, 128 * sizeof(uint16_t), 
                cudaMemcpyHostToDevice, stream);
// 异步复制，256 字节，约 1-2µs
cudaStreamSynchronize(stream);  // 确保复制完成
```

---

### 步骤 11：记录开始时间，发出"开始"信号

**位置**：`ModelPersistentBase::LstmForward()`

```cpp
// 记录开始时间（纳秒精度）
start_time = std::chrono::high_resolution_clock::now();

// 发出开始信号
// PCIe 系统：信号值 = SIGNAL_VALUE × spin_switch（交替 0/1）
int signal_val = DEVICE_SPIN_0_UNTIL_SIGNAL_VALUE * cpu_spin_switch;
cpu_pingpong_handles::signal_release(dev_start_signal, signal_val);

// 翻转 CPU 侧开关
cpu_spin_switch = !cpu_spin_switch;
```

**此刻 GPU 端发生的事情**（仍在那个自旋循环里）：

```cuda
// GPU 检测到开始信号
flag = poll_load(dev_start_signal, 
                 SIGNAL_VALUE * device_spin_switch);
device_spin_switch = !device_spin_switch;

// 广播给 block 内所有线程
group.sync();
```

---

### 步骤 12：GPU 执行 LSTM 前向计算

**位置**：`gemv_lstm_bp_impl.cu`，已在共享内存中有权重

**输入**：
- `dev_x`：128维 FP16 输入向量
- 共享内存中的权重矩阵
- 上一步的细胞状态 `cx`、隐藏状态 `hx`（来自 `gmem_c_in`, `gmem_gemm_h`）

**计算过程**：

```
【第0层计算】

  GEMV（矩阵-向量乘法）：
  ┌─────────────────────────────────────────┐
  │  32个warp 并行计算 gates = W₀ × x + b₀  │
  │                                         │
  │  每个 warp 负责输出向量的 32 个元素       │
  │  （因为 4×96 = 384，384/32 = 12 warp）  │
  │                                         │
  │  + 加上预计算项：U₀ × h_{t-1}            │
  └─────────────────────────────────────────┘
  
  gates: (384,) FP16 向量（4 个门拼接在一起）
  
  拆分：
    [i_gate, o_gate, f_gate, c_gate] = gates.split(4)  # 各 96维
  
  逐元素激活（每个元素独立计算）：
    i_t = sigmoid(i_gate)  →  (96,)
    o_t = sigmoid(o_gate)  →  (96,)
    f_t = sigmoid(f_gate)  →  (96,)
    c'  = tanh(c_gate)     →  (96,)
  
  更新细胞状态：
    cx_new = f_t ⊙ cx_old + i_t ⊙ c'  →  (96,) 更新到 gmem_c_out[0]
  
  更新隐藏状态：
    hx_new = o_t ⊙ tanh(cx_new)        →  (96,) 更新到 gmem_gemm_h[0]

【第1层计算（以第0层输出为输入）】
  
  GEMV：
    gates = W₁ × hx_new_layer0 + b₁  +  U₁ × h_{t-1,layer1}
  
  激活和状态更新（同上，但输入是 96维而非 128维）
  
  最终隐藏状态：hx_final  (96,)

【聚合计算（融合在内核中）】
  
  output = dot(mul_weights, hx_final) + add_bias
         = Σ(mul_weights[i] × hx_final[i]) + add_bias
         = 一个浮点标量
  
  这一步通过 warp shuffle 完成规约：
    - 每个 warp 负责 mul_weights 的一部分，局部求和
    - warp 内用 __shfl_down_sync 规约
    - 最终 warp 0 汇总所有结果
```

**产物**：一个 `float` 标量，存在寄存器中，准备写出

---

### 步骤 13：GPU 写出结果，通知 CPU

**位置**：`gpu_cpu_sync_part_2()`

```cuda
// 把 float 输出写入 pinned memory（CPU 可直接读取）
store_atomic_relaxed((float*)host_ld_complete, output_value);
// 内存屏障确保写入对 CPU 可见
```

整个第12-13步的耗时：对于 lstm_s，约 **0.5-2 µs**。

---

### 步骤 14：CPU 读取输出，记录结束时间

**位置**：`ModelPersistentBase::Output()`

```cpp
// CPU 自旋等待：轮询 pinned memory 直到出现有效值
float output;
do {
    output = *(volatile float*)host_ld_complete;
} while (isnan(output) || output == INIT_VALUE);

// 记录结束时间
end_time = std::chrono::high_resolution_clock::now();

// 清空信号，为下次准备（写入 NaN 哨兵值）
*(volatile float*)host_ld_complete = NaN;

// 返回结果
return output;
```

---

### 步骤 15：存储推理结果

**位置**：`InferenceResult::Add()`

```cpp
result.outputs_.push_back(output);
result.start_times_.push_back(start_ns);
result.end_times_.push_back(end_ns);
```

**至此，一次完整推理完成！**

延迟 = `end_time - start_time`，预计约 **1-5 µs**（取决于硬件）。

---

### 步骤 16：继续下一次推理

回到步骤 8，循环执行直到达到 30 秒。在 30 秒内，预计完成约 **1000万次**推理。

---

## 阶段三：收尾阶段

### 步骤 17：保存推理结果

**位置**：`InferenceResult::SaveToNpy()`

**产物**：`./output/lstm_s.1.1.npy`（所有推理输出值的数组）

```
文件格式：NumPy array
shape = (N,)   # N = 推理次数（约1000万）
dtype = float32
内容 = 每次推理的输出标量
```

### 步骤 18：打印统计结果

```
实例 1 统计：
  推理次数: 10,234,567
  总时长:   30.012 秒
  平均延迟: 2.93 µs
  P99 延迟: 4.12 µs   ← 99% 的推理在 4.12 µs 内完成
  P50 延迟: 2.71 µs
```

### 步骤 19：验证结果正确性

```bash
python check.py \
    --ref=./output/lstm_s.1.1.ref.npy \  # SingleStep 算法的参考输出
    --out=./output/lstm_s.1.1.npy         # 持久化内核的输出

# 预期输出：
# Max absolute error: 0.0003  (FP16 精度损失）
# All close: True
```

---

## 完整流程时间线总结

```
时间（相对，不到比例）

0µs  ┌─ 步骤10：复制输入 (256字节, ~1µs)
     │
1µs  ├─ 步骤11：发出"开始"信号 (原子写, ~0.1µs)
     │            │
1µs  │            └──► GPU: 检测信号 (~0.2µs)
     │
1.3µ ├─ GPU: 第0层 GEMV + 激活 (~0.3µs)
     │
1.6µ ├─ GPU: 第1层 GEMV + 激活 (~0.3µs)
     │
1.9µ ├─ GPU: 聚合（点积规约）(~0.2µs)
     │
2.1µ ├─ GPU: 写出输出（原子写, ~0.1µs）
     │            │
2.2µ └─ CPU: 读到输出值 (~0.1µs)

总延迟：~2.2µs（理想情况下）
```

---

## 与 SingleStep（cuBLAS）方案的对比

| 阶段 | SingleStep (cuBLAS) | AsSingleStepGemv (持久化) |
|------|---------------------|--------------------------|
| 每次推理时 kernel 启动 | ~5-10µs | 0（已经在运行） |
| 权重加载 | 每次从显存读 | 一次性放入共享内存 |
| 同步机制 | CUDA 事件（~2µs） | 原子操作（~0.1µs） |
| 典型延迟 | ~10-20µs | ~2-5µs |
| 适用场景 | 通用、调试 | 极低延迟生产环境 |

---

## 真实执行示意（PingPong 基准）

项目还提供了一个"空操作"基准（PingPong 模型），只做 CPU-GPU 来回信号而不做 LSTM 计算，用于测量**框架本身的开销**：

```bash
nvLstmInf --ping-pong lstm_s ./data/lstm_s ./data/lstm_s.npy 2 ./output
```

典型结果：
- PCIe 系统：~2µs（主要是 PCIe 传输延迟）
- Grace Hopper：~0.5µs（统一内存，无 PCIe）

实际 LSTM 推理延迟 = PingPong 延迟 + 计算时间。
