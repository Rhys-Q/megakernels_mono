# FlashRT Demo 串讲：从代码到机器人动作

## Demo 场景设定

假设我们有一个机器人正在执行桌面操作任务，我们想让它"拿起红色积木"。

**硬件环境**：RTX 5090 (SM120)
**模型**：Pi0.5
**框架**：PyTorch

---

## 第一步：安装和导入

```python
# 安装（只需一次）
# pip install -e ".[torch]"

import flash_rt
import numpy as np
from PIL import Image

# 准备假的摄像头图像（真实使用时从摄像头读取）
camera_image = np.random.uint8(np.zeros((224, 224, 3)))   # 主摄像头（RGB，224×224）
wrist_image  = np.random.uint8(np.zeros((224, 224, 3)))   # 腕部摄像头（RGB，224×224）
```

**这一步的产物**：
- 导入了 `flash_rt` 包
- 准备了两张输入图像（numpy 数组格式）

---

## 第二步：加载模型

```python
model = flash_rt.load_model(
    checkpoint="/path/to/pi05_checkpoint",   # 模型权重目录
    framework="torch",                        # 使用 PyTorch
    num_views=2,                              # 摄像头数量
    autotune=3,                               # GEMM 自动调优迭代次数
)
```

**这一步幕后发生了什么？**（大约需要 30-60 秒）

### 2.1 检测 GPU 架构

```python
# flash_rt/hardware/__init__.py 中
cc = get_compute_capability()  # → "12.0"
arch = "rtx_sm120"             # RTX 5090
```

**产物**：确定使用 RTX 路径

### 2.2 查询分发表，确定前端类

```python
# 查分发表
pipeline_class = _PIPELINE_MAP[("pi05", "torch", "rtx_sm120")]
# → Pi05TorchFrontendRtx
```

**产物**：知道要实例化哪个前端类

### 2.3 前端初始化（Pi05TorchFrontendRtx.__init__）

#### 2.3.1 加载 safetensors 权重文件

```python
# 从磁盘读取权重文件
raw_weights = load_safetensors("/path/to/pi05_checkpoint/*.safetensors")
# raw_weights 是一个大字典：
# {
#   "vision_model.encoder.layers.0.self_attn.q_proj.weight": tensor(1152, 1152),
#   "vision_model.encoder.layers.0.self_attn.k_proj.weight": tensor(256, 1152),
#   ...
# }
```

**产物**：原始权重字典（BF16 格式，在 GPU 内存中）

#### 2.3.2 权重格式转换

```python
# convert_pi05_safetensors() 做了几件事：
# 1. 合并 Q、K、V 为单个 QKV 权重（减少内核调用次数）
q = raw_weights["...q_proj.weight"]  # (1152, 1152)
k = raw_weights["...k_proj.weight"]  # (256, 1152)
v = raw_weights["...v_proj.weight"]  # (256, 1152)
qkv = torch.cat([q, k, v], dim=0)   # (1664, 1152) ← 合并后

# 2. 调整内存布局（行优先 vs 列优先，影响 GEMM 性能）
# 3. 折叠 RMSNorm（把 FP32 的 norm_weight 转成 BF16，避免 -1.0 的精度问题）
```

**产物**：格式对齐的权重字典

#### 2.3.3 FP8 量化权重

```python
# 把所有大矩阵从 BF16 量化成 FP8 E4M3
for layer_i in range(27):  # 视觉编码器 27 层
    fp8_qkv, qkv_scale = quantize_to_fp8(weights[f"vision.layer{layer_i}.qkv"])
    fp8_weights[f"vision.layer{layer_i}.qkv"] = fp8_qkv
    fp8_scales[f"vision.layer{layer_i}.qkv"] = qkv_scale
# 编码器 18 层、解码器 18 层同理
```

**产物**：FP8 权重字典（内存是原来的 50%）

#### 2.3.4 预计算解码器样式

```python
# 为所有扩散步和所有层预计算 AdaRMSNorm 调制参数
# 这些参数只依赖于时间步，不依赖于输入
# 预计算后存为 CudaBuffer，推理时直接读取
styles = precompute_decoder_styles(
    time_mlp_weights,
    num_steps=10,
    num_layers=18,
    chunk_size=10,
    decoder_dim=1024,
)
# styles.shape = (10, 18, 10, 3*1024) 的 BF16 数据
```

**产物**：预计算好的样式缓冲区

#### 2.3.5 创建注意力后端

```python
attn_backend = RtxFlashAttnBackend(
    num_views=2,          # 2个摄像头 → vs = 2*256 = 512
    max_prompt_len=48,    # 最大 prompt 长度
    chunk_size=10,        # 动作序列长度
    num_encoder_layers=18,
    num_decoder_layers=18,
)
# 分配 Q/K/V/O 缓冲区（各种大小的 BF16 张量）
# 总计约 200MB GPU 内存
```

**产物**：包含所有 QKV 缓冲区的注意力后端

#### 2.3.6 创建流水线

```python
self.pipeline = Pi05Pipeline(
    gemm=GemmRunner(),
    fvk=flash_rt_kernels,      # C++ CUDA 内核模块
    attn_backend=attn_backend,
    weights=fp8_weights,        # 只有 int 指针，没有 tensor
    num_views=2,
    chunk_size=10,
    use_fp8=True,
)
```

**产物**：完整的推理流水线（尚未校准，不能最快速度运行）

---

## 第三步：首次推理（触发自动校准）

```python
# 第一次调用 predict 时，会自动做校准
actions = model.predict(
    images={"image_primary": camera_image, "image_wrist": wrist_image},
    prompt="pick up the red block"
)
```

**这一步幕后发生了什么？**

### 3.1 VLAModel.predict() 的逻辑

```python
def predict(self, images, prompt):
    # 检测是否需要做校准
    if self._needs_real_data_calibration:
        self.pipe.calibrate_with_real_data([observations])
        self._needs_real_data_calibration = False
    
    # 检测 prompt 是否变了
    if prompt != self._current_prompt:
        self.pipe.set_prompt(prompt)
        self._current_prompt = prompt
    
    return self.pipe.infer(observations)
```

### 3.2 set_prompt("pick up the red block")

```python
# 1. Tokenize
tokens = tokenizer.encode("pick up the red block")
# → [3, 8919, 701, 278, 2654, 2908, 2] （假设的 token ID 序列）
# prompt_len = 7

# 2. 查 embedding table 得到嵌入向量
lang_embeds = embedding_table[tokens]
# shape: (7, 2048) BF16，在 GPU 上

# 3. 如果 prompt 长度与上次不同，重建流水线
# 假设是第一次设置，需要重建
self.pipeline = rebuild_pipeline(prompt_len=7)
# 重建会分配新的缓冲区（按实际 prompt 长度，不浪费 padding）

# 4. 把语言嵌入写入编码器缓冲区
# encoder_x 的布局：
# [vision_features: 512行] [language_embeds: 7行]
# encoder_x[512:519] = lang_embeds
```

**产物**：
- 流水线知道了文字指令的嵌入表示
- 编码器的输入缓冲区准备好了（512+7=519 行）

### 3.3 校准（calibrate_with_real_data）

```python
# 这一步用真实图像数据收集 FP8 激活值统计
for _ in range(1):  # 单帧校准
    # 暂存图像到缓冲区
    stage_images(camera_image, wrist_image)
    stage_random_noise()
    
    # 用"动态量化"模式运行（每层实时计算激活值最大值）
    run_pipeline_with_dynamic_quant(stream)
    
    # 收集每层的激活值最大值（amax）
    # layer0_qkv_amax = max(abs(activation))
    per_sample_amax.append(collect_amax())

# 计算最终缩放因子
final_scales = amax_to_scales(per_sample_amax[-1])
# 例如：
# encoder.layer0.qkv_input_scale = 0.0035 (激活值最大约 127)
# encoder.layer0.ffn_gate_scale  = 0.0028 (激活值最大约 160)
# ...

# 把缩放因子写入流水线的持久缓冲区
upload_scales_to_pipeline(final_scales)

# 标记"已校准"
self.fp8_calibrated = True

# GEMM 自动调优（autotune=3）
autotune_gemms(num_trials=3)
# cuBLASLt 会为每个 GEMM 尺寸测试不同的算法，选最快的

# 录制 CUDA Graph
record_infer_graph(obs)
# 这一步把整个 run_pipeline() 录制成 CUDA Graph
```

**产物**：
- 每层都有了精确的 FP8 缩放因子
- cuBLASLt 已知道最优的 GEMM 算法
- CUDA Graph 录制完成（`self.graph` 保存了图）

### 3.4 infer(observations) — 真正的推理

现在所有准备工作都做完了，开始真正的推理：

```python
def infer(self, obs):
    # 暂存输入到设备缓冲区
    stage_images(obs)
    stage_random_noise(seed=random.randint())  # 每次用不同的噪声
    
    # 重放 CUDA Graph！（只有这一行有实际意义）
    self.graph.replay(self.stream)
    
    # 等待 GPU 完成（同步）
    torch.cuda.synchronize()
    
    # 从 GPU 内存复制结果到 CPU
    actions_raw = self.pipeline.action_buf.copy_to_cpu()
    # shape: (10, 32) FP16
    
    # 反归一化
    actions = actions_raw[:, :7] * action_std + action_mean
    # 取前 7 维，并还原到真实量纲
    
    return {"actions": actions}  # shape: (10, 7)
```

**这 17ms 内发生了什么？**（CUDA Graph 重放阶段）

```
[0ms]    CUDA Graph replay() 发送到 GPU
[0.1ms]  阶段A开始：视觉编码器
         A1: patch_embed (图像 → 512 个 patch 向量)
         A2-A6: 27层 SigLIP Transformer
              每层：RMSNorm + QKV(FP8) + FlashAttn + OutProj(FP8) + FFN(FP8)
[6ms]    阶段A完成：vision_x = (512, 1152) BF16

[6ms]    阶段B开始：语言+视觉编码器
         B0: vision_proj (1152 → 2048)
             encoder_x[:512] = projected vision
             encoder_x[512:] = language_embeds (已预置)
         B1-B5: 18层 Gemma-2B Encoder
              每层：FP8融合内核(RMSNorm+量化) + QKV(FP8) + GQA(FlashAttn) + 融合FFN
              每层同时把 K/V 存入 KV 缓存
[9ms]    阶段B完成：KV缓存已填充

[9ms]    阶段C开始：动作解码器（扩散）
         初始状态：x = 随机噪声 (10, 32)
         
         Step 0 (t=T):
           C0: action_in_proj (32 → 1024)
           C1-C7: 18层 Gemma-300M Decoder
                每层：AdaRMSNorm(style[0,i]) + QKV(FP8) + 自注意力
                      + 交叉注意力（查编码器KV缓存）+ FFN(FP8)
           C8: AdaRMSNorm + action_out_proj (1024 → 32)
               x = x + delta_x
         
         Step 1 (t=T-1): [同上，使用 style[1,:]]
         ...
         Step 9 (t=1): [最后一步]
         
[17ms]   阶段C完成：action_buf = (10, 32) BF16

[17ms]   GPU完成，通知CPU
[17.1ms] CPU读取结果，反归一化
[17.2ms] 返回 actions (10, 7)
```

**最终产物**：
```python
actions = np.array([
    [0.12, -0.05, 0.30, 0.01, 0.15, -0.02, 0.85],  # 时间步0的7个关节角度
    [0.14, -0.04, 0.32, 0.02, 0.16, -0.01, 0.88],  # 时间步1
    ...
    [0.30,  0.10, 0.45, 0.05, 0.20,  0.05, 0.95],  # 时间步9
])
# 单位：弧度（旋转关节）或米（平移关节）
```

---

## 第四步：后续推理（极速重放）

```python
# 第二次调用开始，由于：
# - prompt 没变 → 不需要重新处理文字
# - CUDA Graph 已录制 → 直接重放

for _ in range(1000):
    # 读取新图像
    camera_image = capture_from_camera()
    wrist_image = capture_from_wrist_camera()
    
    # 推理（纯 CUDA Graph 重放）
    actions = model.predict(
        images={"image_primary": camera_image, "image_wrist": wrist_image},
        prompt="pick up the red block"  # 同一 prompt，不重新处理
    )
    
    # 发给机器人执行
    robot.execute_actions(actions)
    # 注意：每次只执行第一个时间步的动作，然后再推理
    # （Model Predictive Control 模式）
```

**每次推理的时间分解（RTX 5090）**：
```
图像暂存到 GPU 缓冲区：   ~0.5ms
CUDA Graph replay 发送：  ~0.1ms
GPU 执行（17ms）：        ~17ms
等待同步：                ~0.2ms
结果复制 GPU→CPU：        ~0.1ms
反归一化：                ~0.1ms
─────────────────────────────────
总计：                    ~18ms
```

---

## 第五步：多帧校准（可选，更精确）

如果你有数据集，可以做更精确的多帧校准：

```python
# 加载数据集中的多帧样本
from flash_rt.datasets.libero import load_libero_frames
observations = load_libero_frames(n=8)  # 加载 8 帧

# 多帧校准
model.calibrate(observations, percentile=99.9)
```

**幕后过程**：

```python
# 对每帧独立运行推理，收集激活值统计
per_sample_amax = []
for obs in observations:  # 8 帧
    stage_images(obs)
    run_pipeline_with_dynamic_quant(stream)
    per_sample_amax.append(collect_amax())  # 每帧的各层 amax

# 用 99.9 百分位数取代最大值
final_amax = np.percentile(per_sample_amax, 99.9, axis=0)
# 消除 1 帧异常对整体校准的影响

# 计算并上传缩放因子
upload_scales(amax_to_scales(final_amax))

# 重新录制 CUDA Graph
record_infer_graph(observations[0])
```

**对比单帧 vs 多帧校准的效果**：

| 指标 | 单帧校准 | 8帧校准 |
|------|---------|--------|
| 校准时间 | ~2s | ~15s |
| 精度（cos sim vs FP16）| ~0.995 | ~0.998 |
| 对异常帧的鲁棒性 | 低 | 高 |

---

## 整体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FlashRT 工作流程                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ① load_model()                                                      │
│     ├── 检测GPU架构 (detect_arch → "rtx_sm120")                      │
│     ├── 加载权重 (safetensors → BF16 → FP8)                         │
│     ├── 预计算 Decoder Styles (10步×18层×调制参数)                   │
│     └── 创建 Pipeline + AttentionBackend + 分配所有缓冲区            │
│                           ↓                                           │
│  ② model.predict() [首次调用]                                        │
│     ├── set_prompt()                                                  │
│     │    ├── Tokenize → Embed → 写入编码器缓冲区                     │
│     │    └── (如长度变化)重建流水线                                   │
│     ├── calibrate()                                                   │
│     │    ├── 用真实数据跑动态量化推理                                 │
│     │    ├── 收集每层激活值最大值                                     │
│     │    ├── 计算 FP8 缩放因子并上传                                  │
│     │    └── 录制 CUDA Graph                                          │
│     └── infer()                                                       │
│          ├── 暂存图像到 GPU 缓冲区                                    │
│          ├── graph.replay(stream) ← 单条指令触发 GPU                 │
│          │    ├── Phase A: 视觉编码 (SigLIP 27层)                    │
│          │    │    视觉特征: (512, 1152) BF16                        │
│          │    ├── Phase B: 语言+视觉融合 (Gemma-2B 18层)             │
│          │    │    K/V缓存: 18层 × (519, 2048) BF16                 │
│          │    └── Phase C: 动作解码 (10步 × Gemma-300M 18层)        │
│          │         动作输出: (10, 32) BF16                           │
│          └── 反归一化 → 返回 (10, 7) numpy                           │
│                           ↓                                           │
│  ③ model.predict() [后续调用，同一 prompt]                           │
│     └── infer() ← 只有这一步！                                       │
│          ├── 暂存新图像                                               │
│          └── graph.replay() → (10, 7) actions [~17ms]               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 每步的输入/输出总结

| 步骤 | 输入 | 操作 | 输出 | 耗时 |
|------|------|------|------|------|
| load_model | checkpoint路径 | 加载+量化+预计算 | VLAModel | 30-60s |
| set_prompt | 文字指令 | tokenize+embed | 编码器缓冲 | <10ms |
| calibrate | 1-8帧图像 | 动态量化推理 | FP8缩放因子 | 2-15s |
| record_graph | 一帧图像 | CUDA Graph录制 | 可重放的图 | ~1s |
| infer(首次) | 图像+噪声 | 完整推理 | 动作序列 | ~17ms |
| infer(后续) | 图像+噪声 | CUDA Graph重放 | 动作序列 | **~17ms** |

注：首次和后续推理延迟相同——校准和录制的开销是一次性的，之后每次推理都是最优速度。

---

## 完整可运行代码

```python
import flash_rt
import numpy as np

# 1. 加载模型
model = flash_rt.load_model(
    checkpoint="/path/to/pi05_checkpoint",
    framework="torch",
    num_views=2,
    autotune=3,
)

# 2. （可选）用真实数据校准，提高精度
# model.calibrate([obs1, obs2, ..., obs8], percentile=99.9)

# 3. 推理循环
prompt = "pick up the red block"
while True:
    # 读取摄像头
    camera = get_camera_image()   # numpy (224,224,3) uint8
    wrist  = get_wrist_image()    # numpy (224,224,3) uint8
    
    # 推理
    result = model.predict(
        images={"image_primary": camera, "image_wrist": wrist},
        prompt=prompt,
    )
    
    actions = result["actions"]   # (10, 7) float64
    
    # 执行第一步动作
    robot.execute(actions[0])
    
    # 检查任务是否完成
    if task_complete():
        break
```

---

## 常见问题

**Q：第一次 predict 比后续慢很多，这正常吗？**

是的，完全正常。第一次会触发：
1. FP8 校准（~2秒）
2. CUDA Graph 录制（~1秒）

之后每次都是纯 CUDA Graph 重放，非常快。

**Q：换了 prompt 会重新录制 CUDA Graph 吗？**

如果新 prompt 的 token 数量与旧 prompt 相同（或者已经为这个长度录制过），就不会重新录制。如果长度不同，需要重建流水线并重新录制（约几秒）。

**Q：每次推理的噪声必须随机吗？**

是的，扩散模型从随机噪声出发生成动作，固定的噪声会导致每次都输出相同的动作。FlashRT 在每次 infer() 前会生成新的随机噪声，确保多样性。

**Q：(10, 7) 的动作怎么用？**

通常有两种方式：
1. **只执行第一步**：`robot.execute(actions[0])`，然后立即推理下一帧
2. **执行多步后再推理**：`robot.execute_chunk(actions[:3])`，这样可以降低推理频率要求

Pi0.5 在 44ms/帧的控制频率下可以很好地工作。
