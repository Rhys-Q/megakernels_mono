# Demo 串讲：从 Prompt 到生成文字

本文用一个具体的例子，串联 Megakernel 的完整工作流程。

**示例**：用 Qwen3.5-0.8B 回答 "你好，请介绍一下自己"

---

## 完整流程概览

```
用户输入: "你好，请介绍一下自己"
           ↓
    Step 1: Tokenize（分词）
           ↓
    Step 2: Prefill（处理 Prompt）
           ↓
    Step 3: Decode Loop（生成回复）
           ↓
用户看到: "你好！我是一个AI助手，由..."
```

---

## Step 0：准备工作（只做一次）

```python
from model import Decoder

# 加载模型（第一次运行需要从 HuggingFace 下载 ~1.5GB）
decoder = Decoder()
```

**发生了什么**：
1. 下载 Qwen3.5-0.8B 的 BF16 权重
2. 把 24 层的权重指针打包成设备端 blob
3. 在 GPU 上预分配所有缓冲区（KV Cache、DeltaNet State、激活值等）
4. 等待下一个输入

**产物**：
- GPU 上已有 ~1.5GB 的 BF16 权重
- 空的 KV Cache（全零）
- 空的 DeltaNet States（全零）

---

## Step 1：Tokenize（分词）

```python
ids = decoder.tokenizer.encode("你好，请介绍一下自己", add_special_tokens=True)
# 假设结果：[151643, 13048, 11, 14507, 67869, 26920, 14284]
# （实际 token id 因模型而异）
```

**发生了什么**：  
把文字切成模型认识的"词片段"（subword），每个片段对应一个整数 token ID。

词表大小：248,320（Qwen3.5-0.8B）

**产物**：一个整数列表 `ids`，假设长度为 7。

---

## Step 2：Prefill（预填充 6 个 token）

```python
# 处理 prompt 的前 S-1 个 token
for tid in ids[:-1]:  # 处理前 6 个 token
    decoder.step(tid)
```

**每次 `step(tid)` 发生了什么**（以 `tid = 13048` 为例）：

### 2.1 Token Embedding

```
embed_row = embed_weight[13048, :]  → 1024维 BF16 向量
```

**作用**：把 token ID 转为一个向量表示，即这个词的"基础含义"。

**产物**：`embed_row`，shape = [1024], dtype = BF16

### 2.2 通过 Layer 0~2（DeltaNet 层）

以 Layer 0 为例，`deltanet_layer` 执行：

```
a) RMSNorm(embed_row) → s_norm [1024 BF16，在共享内存中]

b) 线性投影：
   s_norm × qkv_proj^T → g_qkv [6144 F32]   ← Q/K/V 的原始投影
   s_norm × z_proj^T   → g_z   [2048 F32]    ← 门控向量
   s_norm × beta_proj^T → g_beta [16 F32]    ← 更新强度
   s_norm × alpha_proj^T → g_alpha [16 F32]  ← decay 强度

c) Conv1d + SiLU（对 16 个头，每头 128 个 key/value 维度）：
   conv_buf[layer_0, head_h] 是长度为 4 的滑动窗口
   新的 token 输入入队，旧的挤出
   → 卷积结果 SiLU 后得到 s_q[128], s_k[128], s_v[128]

d) L2 归一化 s_q 和 s_k

e) 计算 decay（根据 alpha 和 a_log）
   beta = sigmoid(g_beta[h])
   decay = exp(-exp(a_log[h]) × softplus(alpha[h] + dt_bias[h]))

f) DeltaNet 状态更新（核心递推）：
   stk = state[head_h] · s_k  （状态与 k 的内积）
   sqv = state[head_h] · s_q  （状态与 q 的内积）
   error = (s_v - decay × stk) × beta
   output[h] = decay × sqv + error × (s_k · s_q)
   state[head_h] = state[head_h] × decay + outer(s_k, error)

   [状态矩阵 [128×128] 更新了，记住了这个 token 的信息]

g) Gated RMSNorm：output = RMSNorm(output) × SiLU(z)

h) 输出投影：output × out_proj^T → hidden_buffer [1024 BF16]
   hidden_buffer += residual  （残差连接）

i) Post-Attention MLP：
   hidden_buffer → RMSNorm → gate_up → SiLU(gate)×up → down → hidden_buffer
```

**产物**：更新后的 `hidden_buffer` [1024 BF16]，以及更新后的 `dn_states[0]` [128×128 F32]。

### 2.3 Layer 3（Full Attention 层）

```
a) RMSNorm → s_norm

b) QKV 投影：
   s_norm → q_proj → g_q [2048 F32]   （含 gate 向量）
   s_norm → k_proj → g_kv[:512]
   s_norm → v_proj → g_kv[512:]

c) QK 归一化 + RoPE（位置编码）：
   Q[head] → L2归一化 × (1+q_norm_weight) → 旋转变换（基于 position=当前位置）
   K[head] → L2归一化 × (1+k_norm_weight) → 旋转变换

d) 写入 KV Cache：
   k_cache[fa_layer=0][position] = K[head 0], K[head 1]
   v_cache[fa_layer=0][position] = V[head 0], V[head 1]

e) 注意力计算（解码时，遍历所有已存入的历史 K/V）：
   for pos in [0, current_position]:
       score[pos] = Q · K_cache[pos] / sqrt(256)
   probs = online_softmax(scores)
   output = sum(probs[pos] × V_cache[pos])
   output *= sigmoid(gate)  ← Gated Attention

f) 输出投影 + 残差 + MLP
```

**产物**：更新的 `hidden_buffer`，以及 `fa_k_cache[0]` 和 `fa_v_cache[0]` 中新增了一个 position 的 KV 数据。

### 2.4 重复 Layer 4~23

Layer 4,5,6 = DeltaNet（各自的 state 独立更新）  
Layer 7 = Full Attention（fa_k_cache[1], fa_v_cache[1] 更新）  
...以此类推。

### 2.5 Final RMSNorm（只有在需要产生 token 时才用到）

实际上在 prefill 阶段，`step()` 会调用完整的 decode_kernel，因此每次都会产生一个 output_token（但这些中间 token 我们不需要，只需要最后一个）。

### 2.6 六次 step 后的状态

经过 6 次 `step`（处理 ids[0] 到 ids[5]）：
- 18 个 DeltaNet states 都包含了前 6 个 token 的压缩历史
- 6 个 Full Attention 层的 KV Cache 中各有 6 个 position 的数据
- conv_bufs 记录了各层最近 4 个 token 的卷积缓冲

---

## Step 3：Decode Loop（生成回复）

```python
out = []
next_id = ids[-1]  # 从最后一个 prompt token 开始
eos = decoder.tokenizer.eos_token_id

for _ in range(100):  # 最多生成 100 个 token
    next_id = decoder.step(next_id)
    if next_id == eos:
        break
    out.append(next_id)

reply = decoder.tokenizer.decode(out)
```

### 第一次 Decode：处理 ids[-1]，生成第 8 个 token

调用 `decode_kernel`：

**输入**：
- `input_token_id = ids[-1]`（prompt 最后一个 token）
- `position = 6`（当前在第 7 个位置，0-indexed）

**Layer 0 (DeltaNet) 执行**：
- dn_state[0] 已经包含了前 6 个 token 的历史
- 处理第 7 个 token（ids[-1]），状态再次更新
- 输出这个 token 的 DeltaNet 响应

**Layer 3 (Full Attention) 执行**：
- 计算 Q，K，V
- K 和 V 写入 k_cache[fa_layer=0][position=6]
- 查询所有 position 0~6 的 KV Cache → 注意力输出

**LM Head**：
- 最后的 hidden state [1024] → 投影到 [248320] 维 logits
- argmax → 返回概率最大的 token id

假设返回 `token_id = 151622`（对应 "你好"）。

### 后续 Decode 迭代

每次：
- 输入上一次的 output token
- position 递增（6, 7, 8, ...）
- KV Cache 不断增长（每次 Full Attention 层都写入新位置）
- DeltaNet States 不断更新（每次都更新状态矩阵）

直到模型输出 EOS token 或达到 max_tokens=100。

**产物**：一系列 token ids，解码为文字：
```
"你好！我是一个人工智能助手，我可以帮助你..."
```

---

## 全流程时序图

```
时间 →

加载模型:          [下载权重 ~10s][初始化缓冲区]
                   ↓
Prefill:
  step(ids[0])     [embedding][Layer0 DN][Layer1 DN][Layer2 DN]
                   [Layer3 FA][Layer4 DN]...[Layer23 FA][LM Head]  → 忽略输出
  step(ids[1])     [同上，position=1，KV Cache+1]                  → 忽略输出
  ...
  step(ids[5])     [同上，position=5，KV Cache+1]                  → 忽略输出

Decode:
  step(ids[6])     [同上，position=6，收集输出] → token_7
  step(token_7)    [同上，position=7]           → token_8
  step(token_8)    [同上，position=8]           → token_9
  ...
  step(token_n)    [EOS] → 停止

Tokenize:          [ids → 文字]
                   ↓
输出: "你好！我是..."
```

---

## 每一步的产物汇总

| 步骤 | 产物 | 格式 |
|------|------|------|
| Tokenize | token IDs | `[int]`，长度 S |
| Token Embedding | embed_row | `[1024] BF16` |
| DeltaNet Layer | 更新 dn_state，输出 hidden | state: `[128×128] F32`；hidden: `[1024] BF16` |
| Full Attn Layer | 更新 KV Cache，输出 hidden | kv_cache: 新增一行；hidden: `[1024] BF16` |
| Final RMSNorm | normalized | `[1024] F32` |
| LM Head | 下一个 token | `int` (0~248319) |
| Decode Loop | 生成的 token 序列 | `[int]` |
| Detokenize | 最终文字回复 | `str` |

---

## 运行基准测试

```bash
cd lucebox_hub/megakernel
pip install -e . --no-build-isolation

# 运行完整基准（pp520 预填充 + tg128 解码）
python final_bench.py
```

输出示例：
```
Method                      Prefill pp520   Decode tg128   tok/J
------------------------------------------------------------------
Megakernel (BF16, @220W)       21,347          413        1.87
```

`pp520` 表示处理 520 个 token 的 prompt。  
`tg128` 表示生成 128 个 token。  
`tok/J` 是能效（每焦耳生成的 token 数）。
