# Mirage MPK 系统运行流程

## 完整生命周期概览

MPK 的生命周期分为两个阶段：**编译期**和**运行期**。

```
编译期（只做一次）：
  用户定义模型 → 生成任务图 → nvcc 编译 → .so 文件

运行期（每次推理调用）：
  加载 .so → 初始化 → 启动 MegaKernel → GPU 自循环 → 返回结果
```

---

## 阶段一：构建计算图

### 1.1 创建 PersistentKernel 对象

```python
mpk = mi.PersistentKernel(
    mode="offline",
    world_size=1, mpi_rank=0,
    num_workers=96, num_local_schedulers=4,
    max_seq_length=512,
    max_num_batched_requests=8,
    max_num_batched_tokens=8,
    meta_tensors={"step": step, "tokens": tokens, ...}
)
```

这一步只是在 Python 层创建对象，并**记录配置参数**。内部会创建一个空的 `KNGraph`。

**产物**：Python 对象 `mpk`，内部有空的 `kn_graph`

---

### 1.2 绑定输入张量

```python
# 绑定模型权重（已在 GPU 上的 PyTorch 张量）
embed_weight = mpk.attach_input(
    torch_tensor=model.embed_tokens.weight, 
    name="embed_tokens"
)
```

`attach_input` 的作用：
- 在 `kn_graph` 中创建一个 `KNInputOp` 节点
- 记录张量的 GPU 内存地址（`data_ptr()`）、形状、数据类型
- 返回一个 `Tensor` 对象，后续层定义时使用

**产物**：`kn_graph` 中增加一个 Input 节点

---

### 1.3 分配中间张量

```python
y = mpk.new_tensor(
    dims=(batch_size, hidden_size),
    dtype=mi.bfloat16,
    name="embed_out",
    io_category="cuda_tensor",
)
```

`new_tensor` 的作用：
- 记录张量描述（形状、类型、名称）
- 但**不立即分配 GPU 内存**，内存在编译/初始化时分配
- 返回一个 `Tensor` 对象

**产物**：一个逻辑张量描述，尚无实际内存

---

### 1.4 定义计算层

```python
# 每调用一次 layer 方法，就在 kn_graph 中添加一个节点
mpk.embed_layer(input=x, weight=embed_weight, output=y, ...)
mpk.rmsnorm_layer(input=y, weight=w_norm, output=rmsnorm_out, ...)
mpk.linear_layer(input=rmsnorm_out, weight=w_qkv, output=attn_in, ...)
mpk.paged_attention_layer(input=attn_in, k_cache=k_cache, ..., output=attn_out, ...)
```

每个 `layer` 方法内部：
1. 创建对应的 `KNOperator` 子类对象（如 `KNLinearOp`）
2. 记录 `grid_dim`、`block_dim`（决定并行度）
3. 将算子添加到 `kn_graph`

**产物**：`kn_graph` 中的计算图节点链（形成 DAG）

---

## 阶段二：代码生成与编译

### 2.1 生成任务图和 CUDA 代码

```python
results = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
```

C++ transpiler 遍历 `KNGraph`，执行以下操作：

**步骤 A：算子到任务的映射**

每个 `KNOperator` 对应一类任务。由于 `grid_dim` 指定了并行度，一个算子可能对应多个任务：

```
RMSNorm layer (grid_dim=(8,1,1))
  → TaskDesc[0]: TASK_RMS_NORM_HOPPER (处理 batch 中第 0 行)
  → TaskDesc[1]: TASK_RMS_NORM_HOPPER (处理 batch 中第 1 行)
  ...
  → TaskDesc[7]: TASK_RMS_NORM_HOPPER (处理 batch 中第 7 行)

Linear layer (grid_dim=(64,1,1))
  → TaskDesc[8..71]: TASK_LINEAR_HOPPER × 64 (每个处理输出的一部分列)
```

**步骤 B：建立任务依赖关系（事件）**

分析数据依赖：Linear 层的输入是 RMSNorm 的输出，所以：
- 创建一个 `EventDesc`，`num_triggers = 8`（等待 8 个 RMSNorm 任务全部完成）
- Linear 的 64 个任务在该事件就绪后才能启动

**步骤 C：生成 RuntimeConfig 初始化代码**

将所有 `TaskDesc[]` 和 `EventDesc[]` 的初始化代码写入 `kernel.cu`。

**产物**：
- `task_graph.json`：JSON 格式的任务图描述
- `kernel.cu`：完整的 CUDA 源代码（约数千到数万行）

---

### 2.2 nvcc 编译

```python
subprocess.run([
    "nvcc", "kernel.cu",
    "-O3", "-arch=sm_90a",
    "-DMODE_OFFLINE",
    "-DMPK_MAX_NUM_BATCHED_REQUESTS=8",
    "-shared", "-fPIC",
    "-o", "kernel.cpython-311.so"
])
```

nvcc 编译生成 Python 可导入的共享库（`.so` 文件）。

该 `.so` 包含：
- `init_persistent_kernel()`：初始化函数
- `launch_persistent_kernel()`：启动函数
- `finalize_persistent_kernel()`：清理函数
- 所有 TaskDesc 和 EventDesc 的静态数据

**产物**：`kernel.cpython-311.so`

---

## 阶段三：初始化

### 3.1 加载编译好的模块

```python
import importlib.util
spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
```

### 3.2 调用 init_persistent_kernel

在 GPU 端执行 `init_kernel<<<1, 128>>>`：

```cpp
__global__ void init_kernel(RuntimeConfig config) {
    // 初始化请求队列（offline 模式）
    for (int i = 0; i < config.total_num_requests; i++) {
        request_ids[i] = i;  // 标记为待处理
    }
    
    // 初始化页面分配器
    for (int i = 0; i < max_num_pages; i++) {
        page_queue[i] = i;  // 所有页面都是空闲的
    }
    
    // 将第一个任务推入 Worker 队列
    worker_queues[0][0] = first_task_id;
}
```

**产物**：GPU 端运行时状态初始化完毕

---

## 阶段四：MegaKernel 执行

### 4.1 启动 MegaKernel

```python
mpk()  # 调用 launch_persistent_kernel
```

在 C++ 层：
```cpp
// 启动两个持久化 kernel
scheduler_kernel<<<num_schedulers/4, SCHEDULER_NUM_THREADS>>>(runtime_config);
worker_kernel<<<num_workers, WORKER_NUM_THREADS, shared_mem_size>>>(runtime_config);
// 等待完成
cudaDeviceSynchronize();
```

Worker 和 Scheduler 使用**不同的 CUDA stream**，可以并发执行。

### 4.2 Scheduler 循环

```cpp
__global__ void scheduler_kernel(RuntimeConfig config) {
    while (true) {
        // 从自己的事件队列取出一个事件
        EventId event_id = dequeue(config.sched_queues[sm_id]);
        
        if (is_termination_event(event_id)) break;
        
        EventDesc event = config.all_events[get_event_position_index(event_id)];
        
        if (event.event_type == EVENT_END_OF_TASK_GRAPH) {
            // 一次迭代完成，准备下一次迭代
            prepare_next_iteration(config);
            // 重新推入第一个任务
            enqueue(config.worker_queues[0], config.first_tasks[next_iter]);
        } else {
            // 将事件关联的任务分发到 Worker 队列
            for (TaskId t = event.first_task_id; t <= event.last_task_id; t++) {
                int target_worker = t % config.num_workers;
                enqueue(config.worker_queues[target_worker], t);
            }
        }
    }
}
```

### 4.3 Worker 循环

```cpp
__global__ void worker_kernel(RuntimeConfig config) {
    int sm_id = blockIdx.x;
    
    while (true) {
        // 从自己的任务队列取出一个任务 ID
        TaskId task_id = dequeue_spin(config.worker_queues[sm_id]);
        
        if (task_id == TERMINATE_ID) break;
        
        // 获取任务描述
        TaskDesc *task = &config.all_tasks[get_task_position_index(task_id)];
        
        // 等待依赖事件就绪（自旋等待）
        wait_for_event(task->dependent_event, config);
        
        // 执行计算
        _execute_task(task, config);
        
        // 触发完成事件
        trigger_event(task->trigger_event, config);
    }
}
```

### 4.4 任务执行（_execute_task）

根据任务类型调用具体实现：

```cpp
__device__ void _execute_task(TaskDesc *task, RuntimeConfig &config) {
    switch (task->task_type) {
        case TASK_EMBEDDING_HOPPER:
            // tasks/hopper/embedding_hopper.cuh
            execute_embedding(task->input_ptrs, task->output_ptrs, config);
            break;
        case TASK_RMS_NORM_HOPPER:
            // tasks/hopper/rmsnorm_hopper.cuh
            execute_rmsnorm(task->input_ptrs, task->output_ptrs, config);
            break;
        case TASK_LINEAR_HOPPER:
            // tasks/hopper/linear_hopper.cuh（使用 TMA + WGMMA）
            execute_linear(task->input_ptrs, task->output_ptrs, config);
            break;
        case TASK_PAGED_ATTENTION_HOPPER:
            // tasks/hopper/multitoken_paged_attention_hopper.cuh
            execute_paged_attention(task->input_ptrs, task->output_ptrs, config);
            break;
        // ...
    }
}
```

---

## 阶段五：迭代循环

### 5.1 一次 decode 迭代的流程

```
初始化 → 以下在 GPU 内部循环
┌────────────────────────────────────────────────────┐
│ 1. Scheduler 推入 "embed_task" 到 Worker 队列       │
│                                                    │
│ 2. Worker 0: 执行 embed_task                        │
│    完成后触发 event_embed_done                      │
│                                                    │
│ 3. Scheduler 监听 event_embed_done 就绪             │
│    推入 rmsnorm_task[0..7] 到多个 Worker 队列        │
│                                                    │
│ 4. Worker 0..7: 并行执行 rmsnorm_task               │
│    各自完成后原子地递增 event_rmsnorm_done 计数器     │
│                                                    │
│ 5. event_rmsnorm_done 计数器达到 8                  │
│    Scheduler 推入 linear_task[0..63] 到多个 Worker  │
│                                                    │
│ 6. Worker 0..63: 并行执行 linear_task               │
│    ...（重复 N 层 Transformer）                     │
│                                                    │
│ 7. 最终执行 argmax_task，确定下一个 token            │
│                                                    │
│ 8. 触发 EVENT_END_OF_TASK_GRAPH                    │
│                                                    │
│ 9. Scheduler 检查停止条件（EOS token？序列满？）     │
│    若否：更新 step，推入下一迭代的第一个任务          │
│    若是：推入终止任务                               │
└────────────────────────────────────────────────────┘
```

### 5.2 停止条件（offline 模式）

在 offline 模式下，所有请求都从 `tokens` 数组中读取 prompt，MegaKernel 持续 decode 直到：
1. 某个请求生成了 EOS token（结束符）
2. 达到 `max_seq_length`

所有请求都完成后，Scheduler 发送终止信号，Worker 和 Scheduler kernel 退出，`cudaDeviceSynchronize()` 返回。

---

## 数据流概览

```
输入数据流（用户提供）：
  tokens (input_ids) → [embed] → hidden_states

每层 Transformer 数据流：
  hidden_states
    → [RMSNorm] → normed_hidden
    → [Linear(QKV)] → qkv
    → [PagedAttention] → attn_out
    → [Linear(O) + Residual] → hidden_states'
    → [AllReduce（多GPU）] → hidden_states''
    → [RMSNorm] → normed_hidden
    → [Linear(Gate+Up)] → gate_up
    → [SiLU×Mul] → activated
    → [Linear(Down) + Residual] → hidden_states'''
    → [AllReduce（多GPU）] → hidden_states

最终处理：
  hidden_states
    → [RMSNorm] → normed
    → [Linear(LM Head)] → logits
    → [Argmax] → next_token_id

next_token_id 写入 tokens 数组，进入下一个迭代
```
