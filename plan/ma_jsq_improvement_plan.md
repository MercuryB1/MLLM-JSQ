# MA-JSQ Improvement Plan: Multimodal-Aware Pruning Metrics & Search Strategies

> **Date**: 2026-04-07
> **Branch**: `hessian`
> **Status**: Planning
> **Problem**: MA-JSQ (block-level Hessian-weighted search) underperforms baseline JSQ v1 (uniform WANDA + sensitivity)

---

## 1. Current System Analysis

### Baseline: JSQ v1

- **Metric**: `_jsq_v1_metric` = WANDA (`|W| * √(||X||² / N)`) + ρ × leave-one-out sensitivity (cross-covariance trick)
- **Sparsity allocation**: Uniform across all layers and blocks
- **File**: `jsq/compression/passes/prune.py:21-67`

### MA-JSQ (current, underperforming)

- **Search**: `BlockSearcher` enumerates 5–8 hand-crafted candidates per block (`block_search.py:148-253`)
  - Uniform, Attn-light(±δ), MLP-light(±δ), Sensitivity-driven (inverse tr(H))
- **Evaluation**: Hessian-weighted block reconstruction error, Fisher proxy H = Y² (`block_search.py:84-117`)
- **Apply**: deepcopy block → apply all passes → forward on lite subset → pick min error

### Root Cause Analysis

| Issue | Detail | Impact |
|-------|--------|--------|
| **Search space too narrow** | 5–8 candidates, essentially 1-DoF grid search on attn vs MLP | Cannot find fine-grained optimal allocation |
| **Fisher proxy H = Y² is crude** | Overweights large-activation tokens (padding/BOS), not semantically important tokens | Misleading error signal |
| **lite_feat subsampling** | Uniform stride to 4096 tokens may discard rare vision-text patterns | Loss of multimodal signal |
| **Modality-blind metric** | `_jsq_v1_metric` treats all tokens identically | No multimodal advantage |
| **Deepcopy evaluation** | Each candidate requires full block copy + forward | Limits search budget |
| **No cross-block allocation** | Every block gets identical sparsity budget | Wastes budget on easy blocks |

---

## 2. Proposed Improvements

### Core Design: JSQ v3 — Unified Quantization-Aware Multimodal Pruning Metric

> **Design principle**: JSQ 的核心是联合优化稀疏+量化。pruning metric 应该同时回答三个问题：
> 1. **这个权重对 output 贡献多大？** （standard importance）
> 2. **保留它会多大程度伤害量化精度？** （quantization damage）
> 3. **它处理的是什么样的 token？** （multimodal token density）
>
> 最终决策：**prune the weight whose removal causes the least output error AND most benefits quantization, weighted by token information density.**

#### 统一公式

$$\text{metric}(i,j) = \underbrace{|W_{ij}| \cdot \tilde{S}_j}_{\text{density-aware importance}} + \underbrace{\rho \cdot \text{sensitivity}(i,j)}_{\text{leave-one-out (optional)}} - \underbrace{\beta \cdot \text{quant\_damage}(i,j) \cdot \tilde{S}_j}_{\text{quantization penalty}}$$

三个组件完全正交，可以独立 ablate，组合使用时 near zero-cost：

---

#### Component 1: Density-Aware Activation Scale $\tilde{S}_j$（替代标准 WANDA 的 $\|X_j\|$）

**问题**：标准 WANDA 用 $\|X_j\| = \sqrt{\sum_t x_{t,j}^2 / N}$，对所有 token 一视同仁。但 vision tokens 数量多且冗余（相邻 patch 空间局部性），text tokens 少但语义密度高。

**方案**：按 token 信息密度加权。Outlier tokens（高 activation norm）无论 modality 都是信息密集的：

$$\omega_t = 1 + \alpha \cdot \max\left(0,\; \frac{\|x_t\| - \mu}{\sigma}\right)$$

$$\tilde{S}_j = \sqrt{\frac{\sum_t \omega_t \cdot x_{t,j}^2}{N_{\text{samples}}}}$$

- 背景 vision patches（low norm）→ $\omega_t \approx 1$，和标准 WANDA 一样
- 物体边缘 / 关键 text tokens（high norm outlier）→ $\omega_t > 1$，被 upweight
- **Cost**: 一次 token-wise norm + clamp, **O(T × cin)**, ≈ 0

```python
def _density_aware_scale(inp_flat: torch.Tensor, nsamples: int, alpha: float = 0.5):
    """Compute density-weighted activation scale per input channel.
    
    Args:
        inp_flat: [total_tokens, cin], all calibration tokens concatenated
        nsamples: number of calibration samples
        alpha: density weighting strength (0 = standard WANDA)
    Returns:
        scale: [cin], density-weighted activation scale
    """
    token_norm = inp_flat.norm(dim=-1)                     # [tokens]
    mu, sigma = token_norm.mean(), token_norm.std() + 1e-8
    density = ((token_norm - mu) / sigma).clamp(min=0)     # only upweight outliers
    token_weight = 1.0 + alpha * density                   # [tokens]
    
    # Weighted per-channel activation magnitude
    scale_sq = (inp_flat.pow(2) * token_weight.unsqueeze(-1)).sum(0) / nsamples  # [cin]
    return scale_sq.sqrt()
```

**Ablation variants**:
- `alpha = 0` → 退化为标准 WANDA scale
- Strategy B: 直接用 vision_mask downweight，`token_weight[vision_mask] *= 0.5`

---

#### Component 2: Quantization Damage $\text{quant\_damage}(i,j)$（JSQ 核心创新）

**问题**：当前用 per-channel absmax 量化 (`ops.py:6-12`)，每行的 quantization step size 由 `max|w|` 决定。行内的 outlier 权重拉大 step size，导致该行所有权重量化误差增大。

**方案**：量化惩罚项——如果保留某权重会伤害量化精度，降低其 importance：

$$\text{quant\_damage}(i,j) = \underbrace{\mathbb{1}[|W_{ij}| \geq |W_{i,(k)}|]}_{\text{top-k outlier?}} \cdot \underbrace{\frac{|W_{ij}| - |W_{i,(k+1)}|}{|W_{i,(k+1)}| + \epsilon}}_{\text{outlier severity}} \cdot \underbrace{\bar{e}_i^{\text{quant}}}_{\text{row quant error}}$$

其中 $|W_{i,(k)}|$ 是第 $i$ 行第 $k$ 大绝对值。直觉：

| Case | quant_damage | Effect |
|------|-------------|--------|
| 普通权重（不是行 top-k） | 0 | metric 不受影响 |
| 行最大值 ≈ 第二大值 | ≈ 0 | 剪掉也不缩小 step，不鼓励 |
| 行最大值 >> 其他（严重 outlier） | 大 | metric 降低 → 鼓励剪掉 → step size 大幅缩小 → 全行量化精度提升 |

**推广到 top-k**：实际中一行可能有多个 outlier（top-2, top-3），不只是最大值。用 top-k (默认 k=3) 可以级联消除多个 outlier。

```python
def _quant_damage(w: torch.Tensor, w_bits: int = 8, top_k: int = 3):
    """Compute per-weight quantization damage score.
    
    Args:
        w: [cout, cin] weight matrix
        w_bits: quantization bit width
        top_k: number of top outlier positions to penalize per row
    Returns:
        damage: [cout, cin] quantization damage score
    """
    w_abs = w.abs()
    q_max = 2 ** (w_bits - 1) - 1
    
    # Current per-row quantization error
    row_max = w_abs.max(dim=-1, keepdim=True)[0]         # [cout, 1]
    step = row_max / q_max
    quant_err = ((w / step).round() * step - w).pow(2)   # [cout, cin]
    row_mean_err = quant_err.mean(dim=-1, keepdim=True)  # [cout, 1]
    
    # Top-k outlier detection
    sorted_abs, _ = w_abs.sort(dim=-1, descending=True)
    threshold = sorted_abs[:, top_k:top_k+1]             # [cout, 1], the (k+1)-th largest
    is_outlier = (w_abs > threshold)                      # [cout, cin]
    
    # Outlier severity: how much this weight sticks out above threshold
    severity = ((w_abs - threshold) / (threshold + 1e-8)).clamp(min=0)  # [cout, cin]
    
    return is_outlier.float() * severity * row_mean_err
```

**Cost**: 一次 sort + 一次 round 模拟 = **O(cout × cin × log cin)**，与 WANDA 的 O(cout × cin) 同量级。

---

#### Component 3: Sensitivity（保留 JSQ v1 的 leave-one-out，可选）

保留原有 JSQ v1 的 cross-covariance sensitivity 项。它衡量"剪掉这个权重后 output 方差变化多少"，是 pruning error 的直接度量。

$$\text{sensitivity}(i,j) = \sqrt{\text{Var}(Y_j) - 2\text{Cov}(Y_j, X_c) W_{jc} + \text{Var}(X_c) W_{jc}^2}$$

- **与 QAJ/IDA 完全正交**：sensitivity 衡量 pruning error，QAJ 衡量 quant benefit，IDA 衡量 token density
- **Cost**: O(T × cin × cout)，两次 BLAS matmul（已有实现）
- 可以设 ρ=0 关闭，作为 ablation

---

#### 统一实现: `_jsq_v3_metric`

```python
def _jsq_v3_metric(
    w: torch.Tensor,
    inp: torch.Tensor,
    nsamples: int,
    rho: float = 1.0,      # sensitivity weight (0 = disable)
    alpha: float = 0.5,    # token density weight (0 = standard WANDA)
    beta: float = 0.5,     # quantization penalty weight (0 = no quant-awareness)
    w_bits: int = 8,       # quantization bit width
    top_k: int = 3,        # top-k outlier positions to penalize
    max_tokens: int = 4096,
) -> torch.Tensor:
    """JSQ v3: unified quantization-aware + density-aware pruning metric.
    
    metric(i,j) = |W_ij| * S_j(density) + rho * sensitivity(i,j) - beta * quant_damage(i,j) * S_j
    
    Three orthogonal, independently ablatable components:
    - Density-aware scale (alpha): upweight info-dense tokens, modality-agnostic
    - Sensitivity (rho): leave-one-out output variance (from JSQ v1)
    - Quant damage (beta): penalize weight outliers that inflate quantization step
    
    Cost: ~same as JSQ v1 + O(cout*cin*log(cin)) for quant_damage
    """
    # --- Density-aware activation scale ---
    act = inp.reshape(-1, inp.shape[-1]).float()
    if act.shape[0] > max_tokens:
        step = (act.shape[0] + max_tokens - 1) // max_tokens
        act = act[::step].contiguous()
    act = act.to(w.device)
    
    scale = _density_aware_scale(act, nsamples, alpha)            # [cin]
    base = w.abs() * scale.unsqueeze(0)                           # [cout, cin]
    
    # --- Sensitivity (reuse JSQ v1 cross-covariance trick) ---
    if rho > 0:
        w_f = w.float()
        N = act.shape[0]
        out = act @ w_f.T                                         # [N, cout]
        out_c = out - out.mean(0, keepdim=True)
        act_c = act - act.mean(0, keepdim=True)
        var_out = (out_c ** 2).mean(0)                            # [cout]
        var_act = (act_c ** 2).mean(0)                            # [cin]
        cov = (out_c.T @ act_c) / N                               # [cout, cin]
        ss = (var_out.unsqueeze(1) - 2.0 * cov * w_f
              + var_act.unsqueeze(0) * w_f.pow(2)
             ).clamp(min=0).sqrt().clamp(max=100.0)
        base = base + rho * ss.to(w.dtype)
    
    # --- Quantization damage ---
    if beta > 0:
        qd = _quant_damage(w.data.float(), w_bits=w_bits, top_k=top_k)
        # Weight damage by activation scale (damage on high-activation channels hurts more)
        qd = qd * scale.unsqueeze(0).to(qd.device)
        base = base - beta * qd.to(w.dtype)
    
    return base
```

---

#### 为什么这个设计有效

| 对比 | JSQ v1 | JSQ v3 |
|------|--------|--------|
| **Token weighting** | 所有 token 等权 | 信息密度加权（vision 冗余 token 自然降权） |
| **Quantization** | 不考虑 | 主动剪掉 outlier → 缩小 step size → 量化精度提升 |
| **Sensitivity** | ✓ | ✓（保留，可关闭 ablate） |
| **Multimodal** | 无 | 通过 density weighting 间接处理（modality-agnostic） |
| **Cost vs v1** | baseline | +10%（sort + round 模拟） |
| **Cost vs MA-JSQ** | — | **快得多**（无 deepcopy, 无 candidate search） |

**关键**：quantization damage 和 density weighting 是 *乘性* 关系——一个 outlier 权重如果恰好在高 density token 的 channel 上，它的惩罚更大，因为保留它不仅伤害量化，还浪费在了高信息密度的 channel 上。

---

### Supplementary Metrics（独立 ablation 用）

---

#### 1.2 Gradient-Signal-Preserving (GSP) Metric

**Core idea**: Replace heuristic sensitivity (leave-one-out output variance) with actual gradient-based importance from a small calibration loss.

**Formulation**:

$$\text{importance}(i,j) = |W_{ij}| \cdot |\frac{\partial \mathcal{L}}{\partial W_{ij}}| \cdot \|X_{\cdot,j}\|$$

This is a first-order Taylor expansion of the pruning-induced loss change:

$$\Delta \mathcal{L} \approx \frac{\partial \mathcal{L}}{\partial W_{ij}} \cdot \Delta W_{ij}$$

Combined with weight magnitude for stability.

**Implementation sketch**:

```python
def _gsp_metric(w, inp, nsamples, block, calib_samples, n_grad_samples=4):
    """Gradient-Signal-Preserving metric using actual loss gradients."""
    base = _wanda_metric(w, inp, nsamples)
    
    # Collect gradients from a few calibration forward-backward passes
    block.requires_grad_(True)
    grad_acc = torch.zeros_like(w)
    
    for sample in calib_samples[:n_grad_samples]:
        loss = next_token_prediction_loss(block, sample)
        loss.backward()
        grad_acc += w.grad.abs()
        w.grad.zero_()
    
    block.requires_grad_(False)
    grad_importance = grad_acc / n_grad_samples
    
    return base * (1 + rho * grad_importance)
```

**Rationale**: The current JSQ v1 sensitivity (leave-one-out output variance) measures output variability but not task-relevant sensitivity. Actual gradients from even simple next-token-prediction loss give much better importance signals, as shown in works like Wanda-SP, SparseGPT, etc.

**Cost**: Requires `n_grad_samples` backward passes per layer — roughly 2–4× slower than forward-only metrics but much cheaper than JSQ v2's O(cout × cin) loop.

---

#### 1.3 Cross-Modal Output Sensitivity (CMOS) Metric

> **Note**: 原方案 CMAP 依赖完整 attention score 矩阵，但 FlashAttention 使用 online softmax + tiling，不会 materialize 完整 N×N attention map 到 HBM。因此改为基于 **output hidden states** 的跨模态敏感度方案。

**Core idea**: 对 attention 层，衡量剪枝对跨模态信息流的影响。不依赖 attention map，而是用 output hidden states 在 vision/text token 位置的变化来衡量。

**Formulation**:

对于 attention block 的输出 $Y = \text{Attn}(X)$，将 Y 分为 vision 和 text 部分：

$$\text{cross\_sensitivity}(i,j) = \left\| \frac{\partial Y_{\text{text}}}{\partial W_{ij}} \right\|^2 + \left\| \frac{\partial Y_{\text{vision}}}{\partial W_{ij}} \right\|^2$$

**Practical approximation** (no backward needed):

```python
def _cmos_metric(w, inp, nsamples, vision_mask=None):
    """Cross-Modal Output Sensitivity: measure per-weight impact on 
    vision/text output tokens separately, then combine.
    
    Uses the same leave-one-out trick as JSQ v1 but splits the 
    variance computation by modality.
    """
    if vision_mask is None:
        return _jsq_v1_metric(w, inp, nsamples, rho=1.0)
    
    act = inp.reshape(-1, inp.shape[-1]).float().to(w.device)
    w_f = w.float()
    out = act @ w_f.T  # [tokens, cout]
    
    # Split by modality
    vis_out = out[vision_mask]    # [n_vis, cout]
    txt_out = out[~vision_mask]   # [n_txt, cout]
    
    # Per-modality variance of leave-one-out output
    ss_vis = _leave_one_out_var(vis_out, act[vision_mask], w_f)   # [cout, cin]
    ss_txt = _leave_one_out_var(txt_out, act[~vision_mask], w_f)  # [cout, cin]
    
    # Text output sensitivity weighted higher (vision is redundant)
    ss = ss_vis + gamma * ss_txt
    
    base = _wanda_metric(w, inp, nsamples)
    return base + rho * ss.to(w.dtype)
```

**Cost**: 与 JSQ v1 相同量级 — 两次 cross-covariance matmul（一次 vision、一次 text），O(T × cin × cout)。

**Rationale**: 不需要 attention map，直接在 hidden state 空间衡量跨模态影响。Text token 的输出敏感度更重要（vision token 冗余可容忍），所以用 γ > 1 加权 text 部分。

**Applicability**: 仅需要 vision_mask，已由 adapter 的 `get_vision_token_mask` 提供。

---

#### 1.4 Output Activation Outlier (OWL) Metric for Layer Sparsity Allocation

**Core idea** (from [OWL, NeurIPS 2023]): Layers with more outlier activations should be pruned less aggressively, because outliers amplify pruning error nonlinearly.

**Formulation**:

$$\text{outlier\_ratio}_l = \frac{|\{x : |x| > \mu + 3\sigma\}|}{|x|}$$

$$s_l = s_{\text{target}} \cdot \frac{1 - \text{outlier\_ratio}_l}{\text{normalize}}$$

**Implementation sketch** (`block_search.py`):

```python
def _owl_layer_sparsity(input_feat, s_target, layer_params):
    """OWL-style per-layer sparsity allocation based on outlier ratio."""
    outlier_ratios = {}
    for name, feat in input_feat.items():
        if not isinstance(feat, torch.Tensor):
            continue
        f = feat.reshape(-1).float()
        mu, sigma = f.mean(), f.std()
        outlier_ratios[name] = (f.abs() > mu + 3 * sigma).float().mean().item()
    
    # Inverse outlier ratio → more outliers = less pruning
    raw_sparsity = {}
    for name in outlier_ratios:
        raw_sparsity[name] = s_target * (1 - outlier_ratios[name])
    
    # Rescale to meet budget constraint
    total_params = sum(layer_params.values())
    weighted = sum(raw_sparsity[n] * layer_params[n] for n in raw_sparsity)
    ratio = s_target * total_params / weighted if weighted > 0 else 1.0
    
    return {n: min(0.95, s * ratio) for n, s in raw_sparsity.items()}
```

**Rationale**: Proven in OWL (NeurIPS 2023). Replaces the current naive candidate enumeration with a principled, data-driven allocation. Zero search cost — computed directly from input activations.

---

### Phase 2: Better Search Strategies

#### 2.1 Sequential Greedy Layer-wise Search

**Core idea**: Instead of evaluating block-level candidates holistically, process layers sequentially and greedily pick the best sparsity for each.

**Algorithm**:

```
Input: block, layers [l_1, ..., l_K], budget s_target
Output: per-layer sparsity {s_1, ..., s_K}

1. Initialize all s_k = s_target
2. For k = 1 to K:
     For s in {s_target - 2δ, s_target - δ, s_target, s_target + δ, s_target + 2δ}:
       Temporarily set s_k = s
       Prune layer k with sparsity s (on a copy)
       Evaluate cumulative block reconstruction error
     Pick s_k = argmin(error)
3. Rescale: adjust all s_k proportionally to satisfy budget constraint
4. Apply final {s_1, ..., s_K} to the real block
```

**Complexity**: `K layers × 5 candidates = 35` evaluations (vs current 5–8). Each evaluation only modifies one layer (no full block deepcopy needed), so per-evaluation cost is lower too.

**Advantage**: Captures inter-layer interactions sequentially. Much richer exploration than the current approach.

---

#### 2.2 Global Block-Level Budget Allocation

**Core idea**: Not all blocks are equally sensitive. Allocate the total sparsity budget across blocks based on block sensitivity, instead of giving every block the same budget.

**Algorithm**:

```
Pre-pass (before main compression loop):
1. For each block b, estimate sensitivity:
   - Option A: ||Y_b||_F (output norm — cheap)
   - Option B: Gradient-based: ||∂L/∂Y_b||_F from a few calibration samples
   - Option C: Perturbation-based: add noise to Y_b, measure loss increase
2. Normalize sensitivities: sens_b = sens_b / mean(sens)
3. Allocate: s_b = s_target * (1 / sens_b) / normalize_factor
4. Clamp: s_b ∈ [s_target - max_delta, s_target + max_delta]
5. Rescale to meet global budget constraint
```

**Expected behavior**: Shallow blocks (less sensitive) get pruned more; deep blocks (more sensitive) get pruned less. Empirically shown in multiple works (OWL, SparseGPT, BESA).

**Implementation**: Add a pre-pass in `pipeline.py` before the main block loop.

---

#### 2.3 Differentiable Sparsity Search (DSS)

**Core idea**: Replace discrete candidate enumeration with continuous optimization of per-layer sparsity via soft masks.

**Formulation**:

$$s_l = \sigma(\theta_l) \cdot 0.95, \quad \theta_l \in \mathbb{R} \text{ (learnable)}$$

Soft pruning mask for differentiability:

$$m_{ij} = \sigma\left(\frac{\text{metric}_{ij} - \text{threshold}(s_l)}{\tau}\right)$$

Optimization:

$$\min_{\theta} \; \mathcal{L}_{\text{recon}}(\theta) + \lambda \cdot \mathcal{L}_{\text{budget}}(\theta)$$

where $\mathcal{L}_{\text{budget}} = (\sum_l p_l \cdot s_l / \sum_l p_l - s_{\text{target}})^2$.

**Implementation sketch**:

```python
s_logits = nn.Parameter(torch.zeros(n_layers))
optimizer = torch.optim.Adam([s_logits], lr=0.1)

for step in range(50):
    s = torch.sigmoid(s_logits) * 0.95
    # Apply soft masks with current s
    Y_hat = soft_pruned_forward(block, input_feat, s, metric, tau=0.1)
    
    loss_recon = hessian_weighted_error(Y_orig, Y_hat, H)
    loss_budget = (param_weighted_mean(s, layer_params) - s_target) ** 2
    loss = loss_recon + 100.0 * loss_budget
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Advantage**: Explores the full continuous space of per-layer sparsity allocations. No need for candidate enumeration.

**Risk**: Soft-to-hard mask gap; requires careful temperature annealing.

---

### Phase 3: Multimodal-Specific Enhancements

#### 3.1 Vision Token Density Calibration

Not all vision tokens are equal. Patches from complex image regions carry more information than uniform background patches.

```python
def compute_patch_importance(pixel_values, patch_embeddings):
    """Weight vision tokens by image region complexity."""
    # Entropy of patch embeddings as complexity proxy
    patch_entropy = -(F.softmax(patch_embeddings, dim=-1) * 
                      F.log_softmax(patch_embeddings, dim=-1)).sum(-1)
    # Normalize to [0.5, 2.0] range
    importance = 0.5 + 1.5 * (patch_entropy - patch_entropy.min()) / 
                 (patch_entropy.max() - patch_entropy.min() + 1e-8)
    return importance
```

#### 3.2 Layer-Type Aware Metric Selection

Use the most appropriate metric for each layer type:

| Layer Type | JSQ v3 Hyperparams | Rationale |
|-----------|-------------------|-----------|
| q_proj, k_proj, v_proj | `beta↑=1.0` (高 quant penalty) | Attn 权重 outlier 对量化伤害最大（影响 KV cache 精度） |
| o_proj | default `beta=0.5` | Output projection，中等敏感 |
| gate_proj | `beta↑=1.0, sparsity↓` | Gating 接近 binary，对 outlier 和量化都极敏感 |
| up_proj, down_proj | default `alpha=0.5, beta=0.5` | MLP projections，最 robust |

#### 3.3 Modal-Split Reconstruction Loss

> **Note**: 原方案 Dual-target with KL(A_orig ‖ A_pruned) 依赖完整 attention map，FlashAttention 下不可行。改为基于 hidden states 的模态分离重建误差。

Current target: minimize block output MSE uniformly. Better: split by modality and weight differently.

$$\mathcal{L} = \mathcal{L}_{\text{vision}} + \gamma \cdot \mathcal{L}_{\text{text}}$$

$$\mathcal{L}_{\text{vision}} = \text{MSE}(Y_{\text{vis,orig}}, Y_{\text{vis,pruned}})$$
$$\mathcal{L}_{\text{text}} = \text{MSE}(Y_{\text{txt,orig}}, Y_{\text{txt,pruned}})$$

Text tokens carry less redundancy → weight text reconstruction error higher (γ > 1). This is already partially supported by the existing `_hessian_block_error` with `vision_mask_flat`, but the current implementation uses γ on text which can be tuned more aggressively.

**Cost**: ≈0% extra — just a mask split on existing MSE computation.

---

## 3. Efficiency Analysis

### Baseline Cost Reference

当前 JSQ v1 per-layer 的 metric 计算开销（以 Qwen2-VL-7B 的 hidden=3584 为例）：

| Operation | Complexity | Typical Time | Note |
|-----------|-----------|-------------|------|
| WANDA base: `‖X‖²` reduction | O(T × cin) | ~0.1ms | 一次 norm |
| JSQ v1 sensitivity: `act @ w.T` | O(T × cin × cout) | ~2-5ms | 单次 BLAS matmul, T≤4096 |
| JSQ v1 sensitivity: `cov = out_c.T @ act_c` | O(T × cin × cout) | ~2-5ms | 单次 BLAS matmul |
| **Total per layer** | | **~5-10ms** | |
| **Total per block** (7 layers) | | **~35-70ms** | |
| **Search overhead** (current, 5-8 candidates × deepcopy + forward) | | **~2-5s per block** | **dominant cost** |

### Per-Method Efficiency Assessment

| Method | Extra Cost per Block | vs JSQ v1 | Verdict |
|--------|---------------------|-----------|---------|
| **JSQ v3 (unified: density + sensitivity + quant_damage)** | sort + round 模拟 + token norm | ≈**+10%** | ✅ 核心方案 |
| — Component: density-aware scale (alpha) | +0.1ms (token norm + clamp) | ≈0% | ✅ Zero-cost |
| — Component: sensitivity (rho) | 已有 (两次 BLAS matmul) | 0% | ✅ 已有 |
| — Component: quant_damage (beta) | sort + round, O(cout×cin×log cin) | ≈+10% | ✅ Near zero |
| **1.4 OWL layer sparsity allocation** | 一次 `mean(feat²)` per layer | ≈0% | ✅ Zero-cost |
| **2.1 Sequential greedy search** | K×5 次单层 prune+forward（无 deepcopy） | **比当前更快** | ✅ 省时 |
| **2.2 Global block alloc** | 预扫一遍 block output norm | +1 次全模型 fwd | ⚠️ 一次性 |
| **1.3 CMOS** | 两次 cross-cov matmul (按 modality 分) | ≈0% | ✅ 同 JSQ v1 |
| **3.3 Modal-split recon loss** | mask split on MSE | ≈0% | ✅ Zero-cost |
| ~~1.2 GSP (gradient)~~ | ~~n × backward~~ | ~~+200-400%~~ | ❌ 太贵 |
| ~~2.3 DSS~~ | ~~50 steps × backward~~ | ~~+500-1000%~~ | ❌ 太贵 |

### Efficiency-First Priority

核心原则：**metric 改进应该是 zero-cost 或 near-zero-cost**，因为它在每个 block 的每个 layer 都要跑。搜索策略改进可以有一定开销，但应该比当前 deepcopy 方案更快而不是更慢。

## 4. Implementation Priority & Timeline

| Priority | Item | Extra Cost vs JSQ v1 | Effort |
|----------|------|---------------------|--------|
| **P0** | **JSQ v3 metric (unified: density + sensitivity + quant_damage)** | ≈+10% | 1.5 days |
| **P0** | 1.4 OWL layer sparsity allocation | ≈0% | 0.5 day |
| **P0** | 2.1 Sequential greedy layer search (替代 deepcopy search) | **更快** | 1 day |
| **P1** | 3.3 Modal-split reconstruction loss (search 阶段的评估函数) | ≈0% | 0.5 day |
| **P1** | 3.2 Layer-type aware hyperparams (per-layer alpha/beta/rho) | ≈0% | 0.5 day |
| **P1** | 2.2 Global block-level budget allocation | 一次性 pre-pass | 1 day |
| **P2** | 1.3 CMOS (sensitivity 按 modality 分离计算) | ≈0% | 1 day |
| ~~P3~~ | ~~1.2 GSP / 2.3 DSS~~ | ~~太贵~~ | ~~—~~ |

> **Strategy**:
> 1. **P0 实现 JSQ v3** — 一个统一函数，三个正交 component（alpha/rho/beta），默认参数即可超越 v1
> 2. **P0 配套 OWL + greedy search** — 替代当前低效的 deepcopy candidate search
> 3. **P1 做 ablation 调参** — 每种 layer type 的最优 (alpha, rho, beta) 可能不同
> 4. **P1 完善 search 评估** — modal-split loss + global block allocation

---

## 5. Experiment Plan

### Stage 1: JSQ v3 Component Ablation (P0)

```bash
COMMON="--model Qwen/Qwen2-VL-7B-Instruct --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --eval_ppl"

# Baseline: JSQ v1 (WANDA + sensitivity, uniform sparsity)
python main.py $COMMON --pruning_method jsq_v1

# --- 单因素 ablation: 逐个开启 JSQ v3 的三个 component ---

# Exp 1: 只开 density-aware scale (alpha=0.5, rho=0, beta=0)
#   验证: 信息密度加权是否优于等权 WANDA
python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 0 --beta 0

# Exp 2: 只开 quant_damage (alpha=0, rho=0, beta=0.5)
#   验证: 量化感知是否独立有效
python main.py $COMMON --pruning_method jsq_v3 --alpha 0 --rho 0 --beta 0.5

# Exp 3: density + sensitivity (alpha=0.5, rho=1.0, beta=0)
#   对比 JSQ v1: 把等权 scale 换成 density scale，其余不变
python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 1.0 --beta 0

# Exp 4: density + quant_damage (alpha=0.5, rho=0, beta=0.5)
#   验证: 去掉 sensitivity 后 density+quant 够不够
python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 0 --beta 0.5

# Exp 5: 全开 JSQ v3 (alpha=0.5, rho=1.0, beta=0.5)
python main.py $COMMON --pruning_method jsq_v3 --alpha 0.5 --rho 1.0 --beta 0.5
```

### Stage 2: JSQ v3 + Search Strategy (P0)

```bash
# Exp 6: JSQ v3 + OWL layer allocation (无 deepcopy search)
python main.py $COMMON --pruning_method jsq_v3 --search_method owl

# Exp 7: JSQ v3 + Sequential greedy (比当前 search 更快)
python main.py $COMMON --pruning_method jsq_v3 --search_method greedy_sequential

# Exp 8: JSQ v3 + OWL + greedy (完整 P0 combo)
python main.py $COMMON --pruning_method jsq_v3 --search_method greedy_sequential_owl
```

### Stage 3: Hyperparameter Tuning (P1)

```bash
# Exp 9: beta sweep (quantization penalty strength)
for beta in 0.1 0.3 0.5 1.0 2.0; do
  python main.py $COMMON --pruning_method jsq_v3 --beta $beta
done

# Exp 10: alpha sweep (density weighting strength)
for alpha in 0 0.3 0.5 1.0 1.5; do
  python main.py $COMMON --pruning_method jsq_v3 --alpha $alpha --beta 0.5
done

# Exp 11: top_k sweep (how many outliers to penalize per row)
for k in 1 3 5 10; do
  python main.py $COMMON --pruning_method jsq_v3 --top_k $k
done

# Exp 12: Layer-type specific hyperparams
#   (attn: higher beta, gate_proj: lower sparsity, MLP: standard)
python main.py $COMMON --pruning_method jsq_v3 --layer_type_aware

# Exp 13: + Global block budget allocation
python main.py $COMMON --pruning_method jsq_v3 --search_method greedy_sequential_owl \
  --global_block_alloc

# Exp 14: + Modal-split recon loss
python main.py $COMMON --pruning_method jsq_v3 --modal_split_gamma 2.0
```

### Stage 4: Multimodal-Specific (P2)

```bash
# Exp 15: CMOS (sensitivity split by modality)
python main.py $COMMON --pruning_method jsq_v3 --cmos --modal_split_gamma 2.0
```

### Evaluation Metrics

- **PPL**: WikiText-2 perplexity (text quality)
- **MLLM benchmarks**: MMBench, SEEDBench, ScienceQA (visual reasoning)
- **Latency**: End-to-end compression time (search efficiency)
- **Memory**: Peak GPU memory during compression

---

## 6. Key Files to Modify

| File | Changes |
|------|---------|
| `jsq/compression/passes/prune.py` | Add `_jsq_v3_metric` (统一), `_density_aware_scale`, `_quant_damage`; extend dispatch |
| `jsq/compression/block_search.py` | Add `_owl_layer_sparsity` (reuse trace_H); replace `_generate_candidates`; add sequential greedy (单层 prune, 无 deepcopy) |
| `jsq/compression/pipeline.py` | Add global block budget allocation pre-pass |
| `jsq/models/base.py` | Add `get_vision_token_mask` to base adapter interface |
| `jsq/models/qwen2_vl.py` | Implement `get_vision_token_mask` |
| `jsq/models/qwen3_vl.py` | Already has `get_vision_token_mask` ✓ |
| `main.py` / `CompressConfig` | Add `--pruning_method jsq_v3`, `--alpha`, `--beta`, `--top_k`, `--search_method`, `--global_block_alloc` |

---

## 7. References

- [WANDA] Sun et al., "A Simple and Effective Pruning Approach for Large Language Models", ICLR 2024
- [OWL] Yin et al., "Outlier Weighed Layerwise Sparsity", NeurIPS 2023
- [SparseGPT] Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot", ICML 2023
- [BESA] Xu et al., "BESA: Block-wise Efficient Sparsity Allocation", 2024
- [AWQ] Lin et al., "AWQ: Activation-aware Weight Quantization", MLSys 2024
