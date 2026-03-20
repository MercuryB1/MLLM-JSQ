# AGENT.md — mllm-jsq 设计文档

---

## hessian 分支：Modal-Aware Block-level JSQ (MA-JSQ)

### 一、核心思想

**原始 JSQ 的局限**：per-layer 独立优化，每层稀疏度全局固定（如 43.75%），clip 用 layer 级 MSE 评估。各 pass 串联，误差累积，且完全不感知多模态结构。

**MA-JSQ 的改进**：以 block 为单位做联合优化。在固定 block 总稀疏度预算下，搜索 block 内各层的最优稀疏度分配，以 **Hessian 加权 + 模态解耦的 block 输出重建误差**为统一目标函数。

---

### 二、目标函数

对第 $i$ 个 block，设原始输出为 $Y$，压缩后输出为 $\hat{Y}$：

$$\mathcal{L}_{block}^{(i)} = \frac{1}{N_v} \left\| \sqrt{H_v} \odot (Y_v - \hat{Y}_v) \right\|_F^2 + \frac{\gamma}{N_t} \left\| \sqrt{H_t} \odot (Y_t - \hat{Y}_t) \right\|_F^2$$

其中：
- $H_v, H_t$：视觉/文本 token 的对角 Fisher（$\approx \text{mean}(Y^2)$，纯前向近似，无需 backward）
- $M_v$：视觉 token 掩码，从 `input_ids` 中 `<|image_pad|>` 位置提取，与 ViT merger 输出 1:1 对应
- $\gamma$：模态平衡因子（可搜索或固定）
- 对纯文本模型，退化为标准 Hessian 加权 MSE

---

### 三、搜索设计

#### 搜索对象：block 内各层的稀疏度分配

一个 block 含 $L$ 个 Linear 层（如 q/k/v/o/gate/up/down），总预算约束：

$$\frac{\sum_l s_l \cdot n_l}{\sum_l n_l} = s_{target}$$

每层 $s_l$ 从小候选集中取值，例如 $\{s_{target} - 0.1,\ s_{target} - 0.05,\ s_{target},\ s_{target} + 0.05,\ s_{target} + 0.1\}$，预算约束大幅削减实际候选数量。

#### 与原始 JSQ 的关系

JSQ importance score 依然负责**决定哪些权重被剪**（mask 内部排序不变），搜索只决定**每层剪多少**。两者正交，MA-JSQ 是 JSQ 的自然扩展而非替代。

#### 候选生成策略

不做全组合枚举，而是用 JSQ score 引导生成少量有意义的候选：

1. **Uniform**：所有层 $s_l = s_{target}$（baseline）
2. **Attn-light**：注意力层稀疏度 $-\delta$，MLP 层 $+\delta$（补偿预算）
3. **MLP-light**：反之
4. **Sensitivity-driven**：按每层 $\text{tr}(H_l) = \sum_j H_{lj}$（从 input_feat 计算）做反比分配，灵敏度高的层少剪

候选数量控制在 **5-10 个**，每个候选跑一次 block forward 即可评估，总开销约为当前 clip grid search 的同量级。

---

### 四、每个 Block 的完整流程

```
输入：inps（当前 block 输入），layer_kwargs

Step 1  [免费] 原始前向
    Y_orig = block(inps)
    H = mean(Y_orig²)                        # Fisher proxy，shape [seq, hidden]
    H_v = H[vision_mask],  H_t = H[~vision_mask]   # 模态分离

Step 2  [免费] 计算各层 Hessian trace
    tr(H_l) = sum(mean(feat_l²))             # 从 input_feat 直接算，不额外跑前向

Step 3  [搜索] 生成候选稀疏度分配
    configs = generate_candidates(s_target, tr_H_per_layer)   # ~5-10 个

Step 4  [搜索] 评估每个候选
    for config in configs:
        block_copy = deepcopy(block)
        apply prune(config.s_l) + smooth + clip + quant  →  block_copy
        Ŷ = block_copy(inps)
        err = hessian_block_error(Y_orig, Ŷ, H_v, H_t, γ)

Step 5  [应用] 选最优 config，写入原始 block
    apply best config to block (in-place)

Step 6  更新 inps = block(inps)，进入下一个 block
```

---

### 五、与现有框架的关系

| 组件 | 现有行为 | MA-JSQ 改动 |
|------|---------|------------|
| `PruningPass` | 全局固定 `sparsity_ratio` | 接受 per-layer `{name: s_l}` dict |
| `ClippingPass` | layer 级 MSE grid search | 作为 Step 4 内部子步骤，不变 |
| `SmoothingPass` | 全局固定 `alpha` | 暂不搜索，保持固定（可后续扩展）|
| `CompressionPipeline` | 顺序调用各 pass | 外层加 block 级搜索循环 |
| `collector.py` | 收集 input_feat | 同时输出 Y_orig（已有） |
| `ModelAdapter` | — | 新增 `get_vision_token_mask()` |

---

### 六、需要新增/修改的文件

| 文件 | 改动 |
|------|------|
| `jsq/config.py` | 新增 `gamma: float = 1.0`，`n_search_candidates: int = 8` |
| `jsq/compression/block_search.py` | **新文件**：`BlockSearcher`，封装 Step 1-5 |
| `jsq/compression/pipeline.py` | 用 `BlockSearcher` 替换原有 per-block 逻辑 |
| `jsq/compression/passes/prune.py` | `PruningPass.apply()` 支持 per-layer sparsity dict |
| `jsq/compression/passes/clip.py` | 误差函数改为 Hessian 加权（`vision_mask` 注入）|
| `jsq/models/base.py` | 新增 `get_vision_token_mask(calib_samples, processor)` |
| `jsq/models/qwen2_vl.py` 等 | 实现 vision mask 提取 |
| `run.py` | 计算 `vision_mask`，传给 pipeline |
| `main.py` | 新增 `--gamma`，`--n_search_candidates` 参数 |

---

### 七、开放问题

1. **deepcopy 代价**：Step 4 每个候选需要 copy block 权重，7B 模型单 block 约 500MB，需要确认显存是否够用，或改为 save/restore patch 的方式（只存 diff）
2. **clip 在搜索内层的位置**：clip grid search 本身较慢，是否在外层搜索时用简化版（跳过 clip 只做 prune+quant），赢家出来后再做完整 clip？
3. **γ 的设定**：固定为 1.0，还是按 $N_v/N_t$ 自适应，还是作为一个可搜索的超参？
4. **vision mask 准确性**：Qwen2.5-VL 中 `<|image_pad|>` 与 ViT merger 输出是否严格 1:1，需要代码验证

---

### 八、实现进度（hessian 分支）

#### 已完成

所有设计文件均已实现并提交到 `hessian` 分支：

| 文件 | 状态 | 关键改动 |
|------|------|---------|
| `jsq/config.py` | ✅ | `gamma`, `n_search_candidates` |
| `jsq/compression/block_search.py` | ✅ | `BlockSearcher`，含分块 trace_H、lite_feat、per-sample forward |
| `jsq/compression/pipeline.py` | ✅ | 使用 `BlockSearcher` |
| `jsq/compression/passes/prune.py` | ✅ | `per_layer_sparsity` dict 支持 |
| `jsq/compression/passes/clip.py` | ✅ | Hessian 加权误差 |
| `jsq/compression/collector.py` | ✅ | `_slice_kw_for_sample`（支持 tuple），所有 text 模式改为 per-sample forward |
| `jsq/models/base.py` | ✅ | `get_vision_token_mask()` 默认 no-op |
| `jsq/models/qwen2_vl.py` | ✅ | vision mask 提取，`run_forward_for_calibration` 修复 |
| `jsq/models/qwen2_5_vl.py` | ✅ | 同上 |
| `run.py` | ✅ | vision_mask 计算，tokenizer unwrap |
| `main.py` | ✅ | `--gamma`、`--n_search_candidates` 参数 |

#### 核心 Bug 修复记录

1. **`collector.py` per-sample forward**（最新修复）
   - 问题：text 模式用全 batch（128）跑 block forward，Qwen2-VL 的 SDPA math backend 分配 56GB attention matrix → OOM
   - 修复：`collect_block_input_feat_and_output`、`collect_block_input_feat`、`run_block` 均改为逐样本 forward
   - 关键：`_slice_kw_for_sample` 需处理 `position_embeddings = (cos, sin)` tuple，旧版只处理 Tensor 导致 batch=128 的 tuple 未被 slice，新版正确 slice

2. **`block_search.py` `_slice_kw_for_sample` tuple 处理**
   - 旧版不处理 tuple，`_run_forward_lite` 传入 batch=1 input 但 batch=128 的 position_embeddings → OOM
   - 修复：统一用 `collector.py` 中的 `_slice_kw_for_sample`（已处理 tuple）

3. **`_compute_layer_trace_H` 分块计算**
   - 问题：`feat.float().pow(2).mean()` 一次性 fp32 转换 → 3.5GB OOM
   - 修复：512 token 分块，每块 fp32 后释放

4. **`run_forward_for_calibration` device 问题**
   - 问题：text 模式用 `next(model.parameters()).device` 取到 ViT 的 CPU device
   - 修复：改用 `model.model.language_model.embed_tokens.weight.device`

5. **`block_search.py` `_hessian_block_error` device 不一致**
   - 问题：multimodal 模式下 `_run_forward_lite` 输出保留在 GPU（list of GPU tensors），`_flatten` 后 `diff_sq` 在 GPU；但 `vision_mask_flat` 由 `_build_flat_vision_mask` 在 CPU 构建，`diff_sq[vision_mask_flat]` 触发 device mismatch
   - 修复：`_hessian_block_error` 内部在索引前将 `mask` 移到 `diff_sq.device`

6. **`block_search.py` FP16 overflow in H**
   - 问题：深层 block 的 hidden states 量级可达 100–1000，FP16 最大值仅 65504；`H = Y_orig_flat.pow(2)` 在 FP16 下溢出为 inf，导致 block 2 以后所有 candidate 的 err=nan/inf
   - 修复：`H = Y_orig_flat.float().pow(2)`（FP32）；`_hessian_block_error` 内部也把 Y 张量转为 FP32 再做差

7. **`_generate_candidates` 预算补偿公式导致极端稀疏度（GQA 架构特有）**
   - 问题：MLP-light 候选用 `a_s = (budget - m_s * n_mlp) / n_attn` 做预算补偿；Qwen2-VL 是 GQA（4 KV heads），`n_attn ≪ n_mlp`，补偿后 attn 层稀疏度高达 78%，压缩后 attention 质量崩溃
   - 修复：改用对称 delta（`a_s = s_target ± d, m_s = s_target ∓ d`），由 `scale_to_budget` 做轻微均匀调整，最大偏离仍受控在 ≈ ±5%

#### MME 实验结果（Qwen2-VL-7B-Instruct, s=0.4375, W8A8）

| 版本 | Cognition | Perception | **Total** | 备注 |
|------|-----------|------------|-----------|------|
| Baseline (n=1, uniform) | 623.9 | 1615.7 | **2239.6** | 纯均匀稀疏 |
| MA-JSQ v1 (n=8, 无限幅) | 545.7 | 1547.0 | 2092.7 | attn 最高 78%，严重退化 |
| MA-JSQ v2 (n=8, sens cap=0.1) | 600.0 | 1589.7 | 2189.7 | 仍有 MLP-light 问题 |
| **MA-JSQ v3 (n=8, 对称δ)** | **617.9** | **1633.1** | **2250.9** | ✅ 超过 baseline +11.3 |

MA-JSQ v3 在 Perception 上超 baseline +17.4 分，总分 +11.3 分。

---

## 基础架构（main 分支，不在此分支修改）

### 目录结构

```
mllm-jsq/
├── main.py
├── run.py
├── jsq/
│   ├── config.py
│   ├── models/           base · registry · llama · qwen2 · qwen2_vl · qwen2_5_vl · qwen3_vl
│   ├── compression/      pipeline · collector · passes/{prune,smooth,clip,quantize}
│   ├── quant/            ops · linear
│   ├── calibration/      text · multimodal
│   └── eval/             ppl · lmms_eval
├── configs/
└── scripts/
```

### 压缩流程（main 分支）

```
校准数据 → collect_first_layer_inputs
         → for each block:
               collect_block_input_feat   (Linear 输入特征 + block 输出)
               PruningPass   (JSQ/WANDA，固定全局稀疏度)
               SmoothingPass (固定 α)
               ClippingPass  (layer 级 MSE grid search)
               QuantizationPass (W8A8)
               block.cpu()
```
