# Mirage MPK Demo 串讲：用 Qwen3-8B 来理解完整流程

本文以 `demo/qwen3/demo.py` 为例，从头到尾串讲 MPK 如何编译并运行一个真实的 LLM（Qwen3-8B），解释每一步的作用和产物。

---

## 运行命令

```bash
# 普通方式（用 PyTorch 跑，作为对照）
python demo/qwen3/demo.py

# 使用 MPK 加速
python demo/qwen3/demo.py --use-mirage
```

---

## 完整流程图

```
用户输入 prompt："Give me a short introduction to large language model."
            │
            ▼
Step 1: 加载 Qwen3-8B 模型权重到 GPU
            │
            ▼
Step 2: 创建 PersistentKernel，指定配置
            │
            ▼
Step 3: 绑定输入张量（权重 + 输入 tokens）
            │
            ▼
Step 4: 定义计算图（逐层 attach）
        Embed → (RMSNorm → QKV Linear → Attention → O Linear → AllReduce
               → RMSNorm → Gate/Up Linear → SiLU → Down Linear → AllReduce) × 28层
        → RMSNorm → LM Head → Argmax
            │
            ▼
Step 5: 生成 task_graph.json 和 kernel.cu
            │
            ▼
Step 6: nvcc 编译 → kernel.so
            │
            ▼
Step 7: mpk() 执行 MegaKernel
        GPU 自循环 decode，直到 EOS 或 max_seq_length
            │
            ▼
Step 8: 输出生成的文本
```

---

## Step 1：加载模型

```python
model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B", world_size=1, ...).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

**作用**：下载并加载 Qwen3-8B 的所有权重到 GPU 显存中。

**产物**：
- `model.model.embed_tokens.weight`：shape `[151936, 4096]`，Embedding 表
- `model.model.layers[i].input_layernorm.weight`：shape `[4096]`，RMSNorm 权重（每层一个）
- `model.model.layers[i].self_attn.q_proj.weight`：shape `[4096, 4096]`，Q 投影权重
- `model.model.layers[i].self_attn.k_proj.weight`：shape `[512, 4096]`，K 投影权重（GQA）
- `model.model.layers[i].self_attn.v_proj.weight`：shape `[512, 4096]`，V 投影权重（GQA）
- `model.model.kv_cache[0][i]`：K Cache，shape `[max_num_pages, page_size, num_kv_heads, head_dim]`
- ...（共 28 层 × 约 10 个张量）

---

## Step 2：创建 PersistentKernel

```python
import mirage as mi

num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
# 在 H100 上：num_workers=96, num_schedulers=4

mpk = mi.PersistentKernel(
    mode="offline",
    world_size=1,
    mpi_rank=0,
    num_workers=96,
    num_local_schedulers=4,
    num_remote_schedulers=0,
    max_seq_length=512,
    max_num_batched_requests=1,
    max_num_batched_tokens=8,
    max_num_pages=16,
    page_size=4096,
    eos_token_id=model.config.eos_token_id,
    meta_tensors={
        "step": step,               # shape [1]，记录当前 decode 步骤
        "tokens": tokens,           # shape [1, 512]，存储所有 tokens
        "input_tokens": input_tokens,  # shape [8, 1]，当前步骤的输入
        "output_tokens": output_tokens, # shape [8, 1]，生成的 token
        "num_new_tokens": num_new_tokens,  # shape [1]
        "prompt_lengths": prompt_lengths,  # shape [1]，prompt 长度
        ...
    },
)
```

**作用**：记录配置，创建内部的 `KNGraph` 对象。

**产物**：Python `mpk` 对象（内存中的配置记录）

---

## Step 3：绑定输入张量和定义中间张量

### 3.1 中间张量（intermediate tensors）

```python
# embed 层的输出：[batch_tokens, hidden_size] = [8, 4096]
y = mpk.new_tensor(dims=(8, 4096), dtype=mi.bfloat16, name="embed_out", io_category="cuda_tensor")

# RMSNorm 输出：[8, 4096]
rmsnorm_out = mpk.new_tensor(dims=(8, 4096), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")

# QKV linear 输出：[8, 5120] = [8, (64+8+8) × 64]（Q×64头+K×8头+V×8头）
attn_in = mpk.new_tensor(dims=(8, 5120), dtype=mi.bfloat16, name="attn_in", io_category="cuda_tensor")

# attention 输出：[8, 4096]
attn_out = mpk.new_tensor(dims=(8, 4096), dtype=mi.bfloat16, name="attn_out", io_category="cuda_tensor")

# ... 其他中间张量
```

**作用**：为计算中产生的临时数据预留"名字"，实际内存在编译后分配。

**产物**：一批有名字的逻辑张量描述

---

### 3.2 绑定 Embedding 层

```python
# 绑定模型权重
w_embed = mpk.attach_input(
    torch_tensor=model.model.embed_tokens.weight,
    name="embed_tokens"
)
x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")

# 定义 embed 层
mpk.embed_layer(input=x, weight=w_embed, output=y, grid_dim=(1,1,1), block_dim=(128,1,1))
x = y  # 更新当前隐状态指针
```

**此时 kn_graph 中**：
```
InputOp("input_token") ──┐
                          ▼
InputOp("embed_tokens") → EmbedOp → OutputOp("embed_out")
```

---

## Step 4：定义 28 层 Transformer

以**第 0 层**为例，其余层完全相同：

### 4.1 RMSNorm + QKV Linear

```python
# 绑定权重
w_norm = mpk.attach_input(layer.input_layernorm.weight, "layer_0_input_layernorm")
w_q = mpk.attach_input(layer.self_attn.q_proj.weight, "layer_0_q_proj")
w_k = mpk.attach_input(layer.self_attn.k_proj.weight, "layer_0_k_proj")
w_v = mpk.attach_input(layer.self_attn.v_proj.weight, "layer_0_v_proj")

# 将 Q/K/V 权重混合排列（优化内存访问）
w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, ...)

# 定义 RMSNorm 层（grid_dim=(8,1,1)：对 8 个 token 并行处理）
mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, 
                  grid_dim=(8,1,1), block_dim=(128,1,1))

# 定义 QKV Linear 层（grid_dim=(80,1,1)：将 5120 维输出分 80 份并行）
mpk.linear_layer(input=rmsnorm_out, weight=w_qkv, output=attn_in,
                 grid_dim=(80,1,1), block_dim=(128,1,1))
```

**此刻的 kn_graph 新增节点**：
```
x ──→ RMSNormOp(layer_0) ──→ rmsnorm_out ──→ LinearOp(QKV, layer_0) ──→ attn_in
```

**实际意义**：
- RMSNorm：对每个 token 的 4096 维向量做归一化，消除数值尺度差异
- QKV Linear：把 4096 维隐状态投影到 5120 维（Q+K+V 的拼接）

---

### 4.2 Attention 层

```python
# KV cache（分页内存）
k_cache = mpk.attach_input(model.model.kv_cache[0][0], "layer_0_k_cache")
v_cache = mpk.attach_input(model.model.kv_cache[1][0], "layer_0_v_cache")

# Q/K 的额外 RMSNorm（Qwen3 特有）
w_q_norm = mpk.attach_input(layer.self_attn.q_norm.weight, "layer_0_q_norm")
w_k_norm = mpk.attach_input(layer.self_attn.k_norm.weight, "layer_0_k_norm")

mpk.paged_attention_layer(
    input=attn_in,
    k_cache=k_cache, v_cache=v_cache,
    q_norm=w_q_norm, k_norm=w_k_norm,
    cos_pos_embed=cos_pos_embed,  # RoPE 位置编码
    sin_pos_embed=sin_pos_embed,
    output=attn_out,
    grid_dim=(1, 8, 1),   # (batch=1, kv_heads=8, chunks=1)
    block_dim=(128, 1, 1),
)
```

**实际意义**：
- 从 KV Cache 中读取历史的 K、V 向量（分页存储）
- 对当前 token 的 Q 向量，计算与所有历史 K 的 attention score
- 用 softmax 权重汇总 V 向量，得到 attention 输出

**Paged Attention 的特点**：KV Cache 不是连续内存，而是通过 `paged_kv_indices` 索引一个个 page，类似虚拟内存页表。

---

### 4.3 Output Projection（Linear + Residual）

```python
w_o = mpk.attach_input(layer.self_attn.o_proj.weight, "layer_0_o_proj")
mpk.linear_with_residual_layer(
    input=attn_out, weight=w_o, residual=x,
    output=attn_proj_out,
    grid_dim=(64, 1, 1), block_dim=(128, 1, 1),
)
x = attn_proj_out  # 更新残差连接
```

**实际意义**：
- 将 attention 输出投影回 hidden_size（4096）
- 加上残差（input x），实现 Transformer 的跳跃连接

---

### 4.4 MLP 层（FFN）

```python
# MLP 权重
w_norm = mpk.attach_input(layer.post_attention_layernorm.weight, "layer_0_post_attn_layernorm")
w_gate_proj = mpk.attach_input(layer.mlp.gate_proj.weight, "layer_0_gate_proj")
w_up_proj = mpk.attach_input(layer.mlp.up_proj.weight, "layer_0_up_proj")
w_gatedup = mpk.shuffle_tensors([w_gate_proj, w_up_proj], shuffled_dim=0, ...)

# RMSNorm → Linear(Gate+Up 合并) → SiLU × Mul → Linear(Down) + Residual
mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, ...)
mpk.linear_layer(input=rmsnorm_out, weight=w_gatedup, output=mlp_mid, ...)
mpk.silu_mul_layer(input=mlp_mid, output=silu_mul_out, ...)
w_down = mpk.attach_input(layer.mlp.down_proj.weight, "layer_0_down_proj")
mpk.linear_with_residual_layer(input=silu_mul_out, weight=w_down, residual=x, output=mlp_out, ...)
x = mlp_out
```

**实际意义**：
- Gate + Up：两路平行 Linear，将 4096→28672
- SiLU×Mul：Gate 路经过 SiLU 激活，与 Up 路相乘（SwiGLU 变体）
- Down：将 28672→4096，加残差

这 4 步完成一层 Transformer 的前馈网络（FFN）。

---

### 4.5 重复 28 层

```python
for i, layer in enumerate(model.model.layers):  # 28 层
    # ... 上述 4.1~4.4 的代码，只是 i 不同
```

**此时 kn_graph 中**：约有 28 × 10+ = 280+ 个算子节点

---

## Step 4.6：LM Head 和 Argmax

```python
# 最终的 RMSNorm + 语言模型 Head
w_norm = mpk.attach_input(model.model.norm.weight, "model_norm_weight")
w_lm_head = mpk.attach_input(lm_head_weight, "lm_head")  # shape [153600, 4096]

mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out, grid_dim=(8,1,1), ...)
mpk.linear_layer(input=rmsnorm_out, weight=w_lm_head, output=argmax_in, grid_dim=(96,1,1), ...)

# Argmax：从 153600 个词里找最高概率的 token
mpk.argmax_partial_layer(input=argmax_in, output=(argmax_part_value, argmax_part_index),
                          grid_dim=(96,1,1), ...)
mpk.argmax_reduce_layer(input=(argmax_part_value, argmax_part_index), output=argmax_out,
                         grid_dim=(1,1,1), ...)
```

**实际意义**：
- LM Head（Linear）：把 4096 维的 hidden state 映射到整个词表（153600 个词）的 logits
- Argmax：选出 logits 最大的 token ID，就是模型预测的下一个词

**产物**：`argmax_out` 中存储了预测的下一个 token ID，会被写入 `tokens` 数组

---

## Step 5：生成任务图和 CUDA 代码

```python
results = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
with open("task_graph_0.json", "w") as f:
    f.write(results["json_file"])
with open("kernel_0.cu", "w") as f:
    f.write(results["cuda_code"])
```

**作用**：C++ transpiler 分析 280+ 个算子节点，生成：

**task_graph.json 内容示例**（简化）：
```json
{
  "tasks": [
    {"id": 0, "type": "EMBED", "inputs": ["input_token", "embed_tokens"], 
     "outputs": ["embed_out"], "trigger_event": 1},
    {"id": 1, "type": "RMSNORM", "inputs": ["embed_out", "layer_0_input_layernorm"],
     "outputs": ["rmsnorm_out"], "dependent_event": 1, "trigger_event": 2},
    ...
    {"id": 5000, "type": "ARGMAX_REDUCE", "inputs": [...], 
     "outputs": ["output_token"], "trigger_event": "END_OF_GRAPH"}
  ],
  "events": [
    {"id": 1, "num_triggers": 1, "tasks": [1]},   // embed_done → 触发 rmsnorm
    {"id": 2, "num_triggers": 8, "tasks": [9..71]}, // 8个rmsnorm全完成 → 触发64个linear
    ...
  ]
}
```

**kernel.cu 代码结构**（简化）：
```cuda
// 任务和事件的静态数据
static TaskDesc all_tasks[] = { /* 5000+ 个任务描述 */ };
static EventDesc all_events[] = { /* 500+ 个事件描述 */ };

// 初始化函数（被 Python 调用）
void init_persistent_kernel(...) {
    // 初始化 RuntimeConfig，设置队列，分配 KV Cache 页
    init_kernel<<<1, 128>>>(config);
}

// 启动函数（被 Python 调用）
void launch_persistent_kernel(cudaStream_t stream) {
    scheduler_kernel<<<4, 32>>>(config);
    worker_kernel<<<96, 256, smem_size>>>(config);
    cudaDeviceSynchronize();
}
```

**产物**：
- `task_graph_0.json`：5000+ 行 JSON，描述任务和事件
- `kernel_0.cu`：数万行 CUDA 代码

---

## Step 6：nvcc 编译

```python
mpk.compile(output_dir="./kernels")
```

内部执行：
```bash
nvcc kernel_0.cu \
  -O3 -gencode=arch=compute_90a,code=sm_90a \
  -DMODE_OFFLINE \
  -DMPK_MAX_NUM_BATCHED_REQUESTS=1 \
  -DMPK_MAX_NUM_BATCHED_TOKENS=8 \
  -DMPK_MAX_NUM_PAGES=16 \
  -DMPK_PAGE_SIZE=4096 \
  -DMPK_MAX_SEQ_LENGTH=512 \
  -DMPK_TARGET_CC=90 \
  -shared -fPIC \
  -lcuda -lcudart \
  -o ./kernels/kernel_0.cpython-311.so
```

**编译时长**：约 30~120 秒（主要是 CUDA 代码的优化编译）

**产物**：`kernel_0.cpython-311.so`（约 50~200 MB），包含：
- 所有任务实现（Linear、Attention 等 CUDA kernel）
- 所有 TaskDesc 和 EventDesc 静态数据
- Python C API 接口函数

---

## Step 7：执行 MegaKernel

```python
starter.record()
mpk()  # 单次调用，直到所有 token 生成完毕
ender.record()
torch.cuda.synchronize()
run_time = starter.elapsed_time(ender)
```

`mpk()` 内部：

```python
# 调用编译好的 .so 中的函数
launcher.launch_func(stream_ptr)
# 该函数启动 scheduler_kernel 和 worker_kernel
# GPU 自循环 decode，直到序列完成
# cudaDeviceSynchronize() 等待 GPU 完成
```

**GPU 端发生的事情**（以 Qwen3-8B decode 100 个 token 为例）：

```
init_kernel（初始化，约 1ms）
  ↓
[迭代 0]（prefill，处理 prompt 的所有 token）
  Embed → 28×(RMSNorm→QKV→Attention→O_Proj→RMSNorm→FFN) → LM_Head → Argmax
  生成第 1 个新 token，写入 tokens[0, prompt_len]
  ↓
[迭代 1]（decode，处理 1 个 token）
  同上流程
  生成第 2 个新 token
  ↓
...（循环 99 次）
  ↓
[迭代 99]
  生成第 100 个 token（或 EOS token）
  Scheduler 检测到停止条件
  推入 TERMINATE 任务
  ↓
Worker 和 Scheduler kernel 退出
cudaDeviceSynchronize() 返回
```

**产物**：`tokens` 张量中已填充了所有生成的 token ID

---

## Step 8：读取结果

```python
generated_ids = tokens[0, : step[0] + 1]  # step 记录了最终步骤数
response = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(response)

# 打印延迟
print(f"generate length {step.max() + 1 - prompt_lengths[0]}, "
      f"per-token latency: {run_time / (step.max() + 1):.3f} ms")
```

**示例输出**：
```
Give me a short introduction to large language model.

Large language models (LLMs) are sophisticated artificial intelligence systems trained on 
vast amounts of text data. They can understand and generate human-like text...

Prompt length 42, generate length 100, per-token latency: 2.3 ms
```

---

## 性能分析（可选）

如果启用了 `--profiling`：

```python
profiler_tensor = torch.zeros(3000 * 128, dtype=torch.uint64, device="cuda")
mpk = mi.PersistentKernel(..., profiler_tensor=profiler_tensor)
```

执行后可以导出 Perfetto trace：
```python
mpk.export_trace("qwen3_trace.json")
```

在 `chrome://tracing` 中打开可以看到：
```
Worker SM 0:  [Embed] [RMSNorm0] [Linear_QKV0] [Attn0] [O_Proj0] ...
Worker SM 1:  [RMSNorm0] [Linear_QKV0_part] [Attn1] ...
...
Scheduler:    [Sched] [Sched] [Sched] ...
```

直观显示每个 SM 的工作时间线，帮助发现性能瓶颈（如某些 SM 空闲、任务不均衡等）。

---

## 关键参数调优指南

| 参数 | 影响 | 调优建议 |
|------|------|---------|
| `num_workers` | Worker SM 数量 | = GPU SM 总数 - num_schedulers/4 |
| `grid_dim` for Linear | 并行度 | = 输出维度 / 64 或 96 |
| `max_num_batched_tokens` | 批处理大小 | decode 时通常 = batch_size |
| `page_size` | KV Cache 分页 | 越大越省 GPU 内存，越小越灵活 |

---

## 总结：MPK vs 普通 PyTorch 推理

| 对比点 | PyTorch 推理 | MPK 推理 |
|--------|-------------|---------|
| 每步 decode 的 kernel 调用次数 | ~300 次 | 1 次（MegaKernel 内部循环） |
| CPU-GPU 通信 | 每 kernel 一次 | 仅初始化时一次 |
| Kernel 启动开销 | ~1ms per kernel × 300 = 300ms | ~0.1ms（一次性） |
| 内存访问效率 | 每 kernel 单独加载 | 可以复用 Shared Memory |
| 多 GPU 通信 | NCCL（需 CPU 协调） | NVSHMEM（GPU 直接通信） |
| 延迟改善 | 基准 | 1.2× ~ 6.7× 更快 |
