# 06 — 三种执行后端：PyTorch / PyVM / CUDA Megakernel

Megakernels 提供三种方式执行同一套指令调度计划，各有其用途和权衡。

源文件：`megakernels/generators.py`，`megakernels/dispatch.py`

---

## 架构概览

```
dispatch.py
  make_schedule_builder("latency") → LatencyScheduleBuilder
  make_mk_interpreter("latency", mk_dir) → LatencyMK_Interpreter
  make_pyvm_interpreter("latency") → PyVM_Interpreter

generators.py
  PyTorchGenerator  → 直接 model.forward()
  PyVM_Generator    → PyVM_Interpreter.interpret(globs, instructions)
  MK_Generator      → MK_Interpreter.interpret(globs)  ← 调用 CUDA .so
```

---

## 后端 1: PyTorchGenerator

源文件：`megakernels/generators.py:61`

```python
class PyTorchGenerator(Generator):
    def __init__(self, model: LlamaForCausalLM):
        self.model = model

    def generate(self, output_tokens, prompt_len, ntok, ntok_already_generated=1):
        for i in range(ntok):
            position_ids = (prompt_len + ntok_already_generated + i - 1)
            decode_inp = BatchState(
                input_ids=output_tokens[:, input_token_pos:input_token_pos+1],
                position_ids=position_ids,
                seq_len=...,
            )
            decode_output = self.model(decode_inp)   # ← 标准 PyTorch forward
            output_tokens[:, output_pos] = decode_output.output_ids
```

**特点**：
- 不使用指令系统，直接调用 `model.forward()`
- 每次 Decode 步骤会产生多次 kernel launch（每层的 attention、MLP 等各一次）
- 结果是**正确性的黄金标准**，用于验证其他后端
- 使用 `F.scaled_dot_product_attention` (FlashAttention) 做 attention

**适用场景**：开发初期验证功能正确性，或对比基准性能。

---

## 后端 2: PyVM_Generator

源文件：`megakernels/generators.py:166`，`megakernels/python_vm.py`，`megakernels/demos/latency/python_vm.py`

```python
class PyVM_Generator(MK_Generator):
    def run(self, input_ids, pos_id):
        # 1. Embedding lookup
        post_embedding = self.model.model.embed_tokens(batch_state)
        self.schedule.globs.hidden_states[:] = post_embedding.hidden_states

        # 2. 重置 barriers
        self.schedule.globs.barriers.zero_()
        self.schedule.globs.pos_id = pos_id

        # 3. Python VM 执行所有指令
        self.interpreter.interpret(self.schedule.globs, self.instructions)

        # 4. 取 logits → argmax（注意：PyVM 不执行 LM Head，还是走 PyTorch）
        post_embedding.hidden_states = self.schedule.globs.hidden_states
        post_lm_head = self.model.lm_head(post_embedding)
        return post_lm_head.output_ids
```

### PyVM_Interpreter 工作原理

源文件：`megakernels/python_vm.py`

```python
class PyVM_Interpreter:
    def __init__(self, instruction_to_solver: dict):
        self.instruction_to_solver = instruction_to_solver

    def interpret(self, globs, instructions):
        for instruction in instructions:
            if isinstance(instruction, NoOp):
                continue
            solver = self.instruction_to_solver[type(instruction)]
            solver(globs, instruction)
```

它接收一个**拓扑排序后的指令列表**（不是 SM 分配后的张量化版本），逐条调用对应的 Python solver 函数。

`INSTRUCTION_TO_SOLVER` 字典（`demos/latency/python_vm.py:396`）：

```python
INSTRUCTION_TO_SOLVER = {
    LayerNorm_QKV_MatVecRopeAppend:   layer_norm_matvec_rope_append,
    PartialAttention:                  partial_attention,
    AttentionReduction:                attention_reduction,
    O_ProjResidual:                    o_proj_residual,
    LayerNormDoubleMatVecSiLU:         layer_norm_double_matvec_silu,
    DownProjResidual:                  down_proj_residual,
    RMS_LM_Head:                       rms_lm_head,
}
```

### Solver 函数结构（以 `o_proj_residual` 为例）

```python
def o_proj_residual(globals: Globals, instruction: O_ProjResidual):
    # 1. Barrier 检查：确认前驱指令已全部完成
    op_barriers = globals.barriers[instruction.layer_idx, instruction.prev_opcode() - 1]
    assert op_barriers[0] == globals.num_attention_heads  # 所有 attn head 完成

    # 2. 执行实际计算（使用 einops.einsum 做矩阵向量乘）
    matvec_with_residual(
        mat=globals.o_proj_weights[instruction.layer_idx],
        vec=globals.attn_out,
        residual=globals.hidden_states,
        ...
    )

    # 3. Barrier 更新：通知后继指令本指令已完成多少 block
    next_op_barriers = globals.barriers[instruction.layer_idx, instruction.opcode() - 1]
    next_op_barriers[0] += instruction.end_block_idx - instruction.start_block_idx
```

**特点**：
- 完全用 Python + PyTorch 实现，无 CUDA 编译
- 单线程执行，**不模拟 SM 并行**（所有 SM 的指令被展平为一个顺序列表）
- Barrier 是简单的 Python assert，失败即报错（便于调试）
- 速度比 PyTorch 后端慢得多（每条指令处理的数据量很小，kernel launch overhead 极大）
- 执行结果应与 PyTorch 后端数值基本一致（可用 `diff_test.py` 对比）

**适用场景**：调试指令逻辑、验证 DAG 依赖关系、开发新 opcode 时。

---

## 后端 3: MK_Generator（CUDA Megakernel）

源文件：`megakernels/generators.py:95`，`megakernels/demos/latency/mk.py`

```python
class MK_Generator(Generator):
    def __init__(self, model, interpreter: MK_Interpreter, schedule: Schedule):
        self.model = model
        self.interpreter = interpreter
        self.schedule = schedule
        self.fill()  # 初始化 barriers

    def run(self, input_ids, pos_id):
        # 1. Embedding lookup（仍在 Python/PyTorch 中）
        post_embedding = self.model.model.embed_tokens(batch_state)
        self.schedule.globs.hidden_states[:] = post_embedding.hidden_states.squeeze(1)

        # 2. 重置 barriers、更新 pos_id
        self.fill()  # barriers.fill_(0)
        self.schedule.globs.pos_id = pos_id

        # 3. 调用 CUDA Megakernel（一次 kernel launch）
        self.interpreter.interpret(self.schedule.globs)

        # 4. logits 现在在 globs.logits 中，直接 argmax
        return torch.argmax(self.schedule.globs.logits, dim=-1)
```

### MK_Interpreter 的实现

源文件：`megakernels/demos/latency/mk.py`

```python
class LatencyMK_Interpreter(MK_Interpreter):
    def __init__(self, mk_dir: Path):
        # 动态加载编译好的 .so 文件（pybind11 扩展）
        import importlib.util
        spec = importlib.util.spec_from_file_location(...)
        self.mk_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mk_module)

    def interpret(self, globs: Globals):
        # 调用 C++ 绑定函数，传入 Globals 对象
        self.mk_module.run(globs.instructions, globs.timings, ...)
```

CUDA kernel 的入口在 `demos/low-latency-llama/llama.cu`，它接收 `globals_t` 结构体（包含所有权重、缓冲、指令张量的 GPU 指针），然后启动 `megakernel::mk<config, globals_t, op1, op2, ...>` kernel：

```cpp
// llama.cu
void run(globals_t g) {
    megakernel::mk<llama_config, globals_t,
        RMS_QKV_Rope_Append,   // opcode 1
        PartialAttention,       // opcode 2
        AttentionReduction,     // opcode 3
        O_Proj,                 // opcode 4
        RMS_UpGate_SiLU,        // opcode 5
        DownProj,               // opcode 6
        RMS_LM_Head             // opcode 7
    ><<<num_blocks, num_threads>>>(g);
}
```

**特点**：
- 生产级性能，只有一次 kernel launch
- 所有 SM 并发执行各自的指令队列
- Barrier 通过 GPU semaphore 实现跨 SM 同步
- 需要编译 CUDA 代码（`make` in `demos/low-latency-llama/`）
- 支持 Hopper TMA（Tensor Memory Accelerator）异步数据加载

**适用场景**：生产部署，性能测量。

---

## 三种后端对比

| 维度 | PyTorch | PyVM | CUDA MK |
|------|---------|------|---------|
| 执行方式 | 标准 PyTorch | Python 逐条解释 | CUDA kernel |
| Kernel launch 次数 | N×num_layers | N×(instructions) | **1** |
| SM 并行 | 由 PyTorch 自动管理 | 无（顺序） | 细粒度控制 |
| 需要编译 | 否 | 否 | **是** |
| 调试便利性 | 高 | **最高**（Python trace） | 低 |
| 速度 | 中 | **最慢** | **最快** |
| 正确性验证 | 黄金标准 | 应与 PyTorch 一致 | 应与 PyTorch 一致 |

---

## generate_with_eos() — 带终止条件的生成

所有后端都继承 `Generator.generate_with_eos()`：

```python
def generate_with_eos(self, output_tokens, prompt_len, ntok,
                      eos_token_check_interval, eos_token_ids):
    for chunk_start in range(1, ntok, eos_token_check_interval):
        # 生成一段 tokens
        self.generate(output_tokens, prompt_len=prompt_len,
                      ntok=chunk_interval, ntok_already_generated=chunk_start)

        # 检查这段里是否出现了 EOS token
        for j, token in enumerate(generated_chunk):
            if token in eos_token_ids:
                return (eos_position, total_generated)

    return (ntok, ntok - 1)
```

`eos_token_check_interval` 控制每生成多少 token 检查一次 EOS（每次检查需要把数据搬回 CPU），在延迟和提前终止之间权衡。

---

## 如何运行

```bash
# PyTorch 后端（无需编译 CUDA）
python -m megakernels.scripts.generate \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode torch --ntok 50

# PyVM 后端（无需编译 CUDA）
python -m megakernels.scripts.generate \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode pyvm --schedule wave --ntok 50

# CUDA MK 后端（需先 make）
cd demos/low-latency-llama && make
python -m megakernels.scripts.generate \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode mk --schedule wave --mk-dir demos/low-latency-llama --ntok 50

# 正确性验证（对比 torch 和 pyvm 的输出是否一致）
python -m megakernels.scripts.diff_test \
    --model meta-llama/Llama-3.2-1B-Instruct
```

下一步：[07-cuda-megakernel.md](07-cuda-megakernel.md) — CUDA 状态机内部结构
