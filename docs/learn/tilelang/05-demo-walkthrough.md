# Demo 串讲：一个 GEMM Kernel 从写代码到 GPU 执行的全过程

我们用 TileLang 的 quickstart 示例——一个带 ReLU 的矩阵乘法（GEMM + ReLU）——来串讲 TileLang 的完整工作流程。

## Demo 代码

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), dtype),    # 输入矩阵 A: M×K
        B: T.Tensor((K, N), dtype),    # 输入矩阵 B: K×N
        C: T.Tensor((M, N), dtype),    # 输出矩阵 C: M×N
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)   # ReLU

            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_relu_kernel


# === 使用 ===
M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32

# 第 1 步：生成 JITKernel（延迟编译）
kernel = matmul(M, N, K, block_M, block_N, block_K)

import torch
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)

# 第 2 步：执行（触发实际编译）
kernel(a, b, c)

# 第 3 步：验证正确性
ref_c = torch.relu(a @ b)
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("结果正确！")

# 第 4 步：查看生成的 CUDA 代码
print(kernel.get_kernel_source())

# 第 5 步：性能测试
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
latency = profiler.do_bench()
print(f"延迟: {latency:.3f} ms")
```

---

## 步骤 1：理解算法目标

我们要计算：**C = ReLU(A @ B)**，其中 A 是 1024×1024，B 是 1024×1024。

在 GPU 上高效执行这个计算的关键是：
1. **数据局部性**：每次从 HBM 加载一小块数据到共享内存，在共享内存里完成计算
2. **隐藏延迟**：内存拷贝时同步做计算（软件流水线）
3. **使用 TensorCore**：用专用硬件做矩阵乘法，比 FP32 CUDA core 快 ~8×

---

## 步骤 2：定义 tile 大小（Kernel 参数选择）

```python
block_M, block_N, block_K = 128, 128, 32
```

这三个参数决定了每个 CTA 处理多大的数据：
- **block_M = 128**：每个 CTA 负责 C 中 128 行
- **block_N = 128**：每个 CTA 负责 C 中 128 列  
- **block_K = 32**：每次从 K 维度取 32 列（分 1024/32=32 次累加）

**产物**：确定了数据分块策略和 Grid 大小：
- Grid = (1024/128, 1024/128) = (8, 8) → 共 64 个 CTA 并行

---

## 步骤 3：`@T.prim_func` 解析——Python → TIR AST

当你调用 `matmul(M, N, K, block_M, block_N, block_K)` 时，Python 解释器执行整个函数体，`@T.prim_func` 装饰器拦截这个过程，把 Python 代码**构建为 TVM TIR 语法树**。

**输入**：Python 源代码

**产物**：TVM TIR IRModule（概念性示意）：
```
primfunc matmul_relu_kernel(A: Buffer[1024, 1024, f16],
                              B: Buffer[1024, 1024, f16],
                              C: Buffer[1024, 1024, f16]) {
  // attr: thread_extent blockIdx.x = 8
  // attr: thread_extent blockIdx.y = 8
  // attr: thread_extent threadIdx.x = 128
  
  allocate A_shared: f16[128, 32] in "shared.dyn"
  allocate B_shared: f16[32, 128] in "shared.dyn"
  allocate C_local:  f32[128, 128] in "local.fragment"
  
  Fill(C_local, 0.0)  // T.clear
  
  for ko in pipelined_range(32, num_stages=3):
    Copy(A[by*128, ko*32] → A_shared)  // T.copy (未展开)
    Copy(B[ko*32, bx*128] → B_shared)  // T.copy (未展开)
    Gemm(A_shared, B_shared → C_local) // T.gemm (未展开)
  
  for i, j in parallel(128, 128):
    C_local[i,j] = max(C_local[i,j], 0.0)  // ReLU
  
  Copy(C_local → C[by*128, bx*128])  // T.copy (未展开)
}
```

此时 `Copy`、`Gemm` 只是"占位符"节点，还没有展开成具体的线程操作。

---

## 步骤 4：Layout Inference——推断内存布局

**输入**：上一步的 TIR IRModule，包含未展开的 `Gemm` 节点

**问题**：`C_local = T.alloc_fragment((128, 128), f32)` 声明了一个 128×128 的"寄存器片段"，但 128×128=16384 个 f32 元素、每个元素 4 字节，总共 64KB——一个线程的寄存器根本装不下！

**Layout Inference 的工作**：

1. 分析 `Gemm(A_shared, B_shared → C_local)` 操作
2. 确定目标 GPU（假设是 H100）上使用 `wgmma.mma_async` 指令
3. 查阅 WGMMA 指令规范：一个 warp group（128 线程）执行 `wgmma.m64n128k16`，每个线程持有 C 中 **64 个 f32 寄存器**
4. 计算 128×128×4 / 128线程 = 512 bytes / 线程 = 128 个 f32 → 实际每个线程持有 128 个 f32（两个 wgmma tile 的累加器）
5. 同时确定 A_shared 需要 **swizzled 布局**（行内按 128B 为单位 XOR 位移）以避免 bank conflict

**产物**：带布局标注的 TIR IRModule：
```
// A_shared 的布局：ColMajor + 128B swizzle
// C_local 的布局：每线程持有 [row/8*2, col/64*4 + lane_id/4/8] 的元素
// （实际布局更复杂，这里是简化描述）
```

---

## 步骤 5：软件流水线注入——隐藏内存延迟

**输入**：包含 `for ko in pipelined_range(32, num_stages=3)` 的 TIR

**问题**：从 HBM 加载数据到共享内存需要 ~200 个时钟周期。如果串行执行：
```
轮次 0: 等待 A_shared 加载完成 [200 cycles]
轮次 0: 等待 B_shared 加载完成 [200 cycles]
轮次 0: WGMMA 计算 [10 cycles]
轮次 1: 等待 A_shared 加载完成 [200 cycles]
...
```
GPU 90% 的时间在等内存，效率极低！

**软件流水线的解法**：`num_stages=3` 表示提前 3 轮预取数据：

```
预热阶段 (prologue):
  异步发起: 拷贝 ko=0 的 A_shared, B_shared
  异步发起: 拷贝 ko=1 的 A_shared, B_shared  
  异步发起: 拷贝 ko=2 的 A_shared, B_shared

主循环 (ko = 0..31):
  等待: ko=0 的拷贝完成
  计算: WGMMA 使用 ko=0 的数据        ← 和下一行同时进行！
  异步发起: 拷贝 ko=3 的数据

收尾阶段 (epilogue):
  等待: ko=30 的拷贝完成，计算 ko=30
  等待: ko=31 的拷贝完成，计算 ko=31
```

**产物**：包含 `cp.async`、`mbarrier` 的 TIR 代码，同时维护多个共享内存 buffer（stage 0、1、2 各一份，循环使用）

---

## 步骤 6：Tile 操作降级——展开具体指令

**输入**：包含 `Gemm`、`Copy` 占位符节点的 TIR

**LowerTileOp Pass 的工作**：

1. **T.copy 展开**（全局内存 → 共享内存）：
   ```
   // 128 个线程并行地各自拷贝一部分元素
   // 每个线程的拷贝变成 cp.async 指令
   cp.async.ca.shared.global [A_shared_ptr + tid*8], [A_ptr + offset], 16
   ```

2. **T.gemm 展开**（在 H100 上）：
   ```
   // 128 个线程组成一个 warp group，执行 WGMMA
   wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
     {acc0, acc1, ..., acc63},  // C 寄存器（每线程 64 个）
     [A_smem_desc],             // A 的共享内存描述符
     [B_smem_desc],             // B 的共享内存描述符
     1, -1, 1, 1, 1;
   ```

3. **T.copy 展开**（共享内存/寄存器 → 全局内存）：
   ```
   // 每个线程把自己持有的 C 寄存器元素写回全局内存
   stg.128 [C_ptr + offset], {acc0, acc1, acc2, acc3}
   ```

**产物**：完全展开的 TIR，所有操作都是线程级别的基本操作

---

## 步骤 7：后端代码生成——TIR → CUDA PTX

**输入**：完全展开的 TIR

**CUDA Codegen 的工作**：遍历 TIR 语法树，生成 CUDA C++ 和 PTX 混合代码

**产物**：生成的 CUDA C++ 代码（简化版）：
```cuda
__global__ void matmul_relu_kernel(half* A, half* B, half* C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    
    // 共享内存（3 个 stage，循环缓冲区）
    __shared__ alignas(128) half A_smem[3][128][32];  // swizzled
    __shared__ alignas(128) half B_smem[3][128][128]; // swizzled
    __shared__ uint64_t mbarrier[3];  // 同步屏障
    
    // 每线程的累加寄存器
    float C_frag[64];  // 128 个线程分摊持有 128×128×4 bytes
    
    // 初始化累加器为 0
    for (int i = 0; i < 64; i++) C_frag[i] = 0.0f;
    
    // 流水线 prologue：预取前 3 轮
    for (int s = 0; s < 3; s++) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                     " [%0], [%1], [%2], ...;" ...);
    }
    
    // 主循环（32 轮，每轮 block_K=32）
    for (int ko = 0; ko < 32; ko++) {
        // 等待当前 stage 的数据就绪
        asm volatile("mbarrier.try_wait.parity.shared::cta.b64 %0, %1, %2;" ...);
        
        // WGMMA 矩阵乘法
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16"
            " {%0,...,%63},"         // 输出：64 个 f32 寄存器
            " %64,"                   // A 的 smem 描述符
            " %65,"                   // B 的 smem 描述符
            " 1, -1, 1, 1, 1;" ...);
        
        // 预取下一轮数据（和计算重叠）
        if (ko + 3 < 32) { /* cp.async.bulk.tensor ... */ }
    }
    
    // ReLU 激活（并行施加到所有 C 元素）
    for (int i = 0; i < 64; i++) C_frag[i] = fmaxf(C_frag[i], 0.0f);
    
    // 将结果写回全局内存（128-bit 向量写）
    // 每个线程写回它负责的 C 元素
    asm volatile("stg.global.128 [%0], {%1, %2, %3, %4};" ...);
}
```

---

## 步骤 8：nvcc 编译——PTX → CUBIN

**输入**：CUDA C++ + PTX 源代码

**工作**：调用系统 nvcc：
```bash
nvcc -arch=sm_90 -O3 --use_fast_math \
     -ptx /tmp/tilelang_xxx.cu -o /tmp/tilelang_xxx.ptx
ptxas -arch=sm_90 /tmp/tilelang_xxx.ptx -o /tmp/tilelang_xxx.cubin
```

**产物**：`tilelang_xxx.cubin`——可以直接加载到 GPU 上执行的二进制文件

---

## 步骤 9：JIT 加载与执行

```python
kernel = matmul(1024, 1024, 1024, 128, 128, 32)
# 此时 kernel 是 JITKernel 对象，已持有编译好的 cubin

c = kernel(a, b, c)
# 内部流程：
# 1. 把 a, b, c 的 DLPack tensor 传给 TVM
# 2. 从 cubin 加载 GPU 函数
# 3. 调用 cuLaunchKernel(
#        func,
#        grid=(8, 8, 1),    # 64 个 CTA
#        block=(128, 1, 1), # 每个 CTA 128 线程
#        args=[a_ptr, b_ptr, c_ptr]
#    )
# 4. 等待执行完成
# 5. 返回 c
```

---

## 完整流程一图总结

```
用户写的 Python 代码（20 行）
            │
            ▼ @tilelang.jit + @T.prim_func
TVM TIR IRModule（中间表示，含 Copy/Gemm 占位符）
            │
            ▼ Layout Inference
带布局标注的 IRModule（知道每个线程持有哪些寄存器）
            │
            ▼ Inject Software Pipeline
带流水线的 IRModule（多阶段 cp.async + mbarrier）
            │
            ▼ Lower Tile Op
展开的 TIR（wgmma 指令、cp.async 指令、stg128 指令）
            │
            ▼ Vectorize / Unroll / MergeSmem 等优化
优化后的 TIR
            │
            ▼ CUDA Codegen
CUDA C++ + PTX 混合代码（~500 行）
            │
            ▼ nvcc -arch=sm_90
可执行 CUBIN 二进制
            │
            ▼ cuLaunchKernel(grid=(8,8), block=(128,))
GPU 并行执行（64 个 CTA，每个 128 线程，8192 个线程同时运行）
            │
            ▼
输出矩阵 C（已完成 GEMM + ReLU）
```

---

## 每步的关键产物总结

| 步骤 | 输入 | 产物 | 核心工作 |
|------|-----|------|---------|
| 1. 算法设计 | 数学公式 | tile 大小参数 | 确定分块策略 |
| 2. Python 解析 | Python 源码 | TIR IRModule（高层） | 语法树构建 |
| 3. Layout Inference | 高层 TIR | 带布局的 TIR | 推断寄存器/SMEM 布局 |
| 4. 软件流水线 | 带布局的 TIR | 多阶段 TIR | 预取 + 异步等待 |
| 5. Tile 降级 | 多阶段 TIR | 展开的 TIR（线程级别） | Copy/Gemm 展开为指令 |
| 6. 进一步优化 | 展开的 TIR | 优化后 TIR | 向量化、展开、SMEM 合并 |
| 7. 代码生成 | 优化 TIR | CUDA C++ / PTX | TIR → 字符串代码 |
| 8. nvcc 编译 | PTX 代码 | CUBIN 二进制 | 汇编 → 机器码 |
| 9. 执行 | CUBIN + 输入张量 | 输出张量 | GPU 并行计算 |
