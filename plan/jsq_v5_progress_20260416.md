# JSQ v5 项目进展总结 — 2026-04-16

**Model**: Qwen2-VL-7B-Instruct | **Calib**: gqa (128 samples) | **Sparsity**: 0.4375 | **Quant**: W8A8
**Tasks**: MME (cognition + perception), MMStar (6 子项)
**工作目录**: `/mnt/disk3/wzn/mllm-jsq` | **分支**: `jsq-v5-mixture`

---

## 1. 背景与目标

plan (`jsq_v5_mixture_hessian.md`) 提出 v5 mixture-Hessian OBS metric，包含两个 innovation：

- **Innovation 1 — mixture-H per-element importance**:
  $I(i,j) = W_{ij}^2 / [H^{-1}]_{jj}$，其中 $H = \pi_t H_t + \pi_v H_v + \lambda I + \sigma_A^2 I$
  替换 v1/v4 的 WANDA / token-weighted 尺度。
- **Innovation 2 — 跨层非均匀稀疏率分配**:
  两层 Hessian 框架 = 外层块级搜索 {s_l} + 内层 per-element mask。

plan §13 目标：MME_cog ≥ 624, MME_per ≥ 1633 (即追回 v4 的 -7 pt cognition 损失)。

---

## 1.5 当前方法（v5 direct，实际跑通的 best 配置）

### Pipeline 概览

```
[Calibration]                     [Per-block prune pass]             [Quant]
  128 gqa samples                   For each decoder block l:          GPTQ-style
  ──────────────          ──►        1. collect X_t, X_v              per-column
  text & vision tokens               2. build H_t = X_t^T X_t          W8A8
  分离存储                            3. build H_v = X_v^T X_v
                                     4. H = π_t H_t + π_v H_v
                                          + λ_floor · I  + σ_A² · I
                                     5. I(i,j) = W_ij² / [H^-1]_jj
                                     6. per-row top-k mask with
                                        s_l = s_target (uniform)
                                     7. apply mask, forward to next block
```

### 1.5.1 Mixture-Hessian 构造（Innovation 1，核心）

对每个 block 的每个投影层 (q/k/v/o/gate/up/down)，基于**校准激活**分 modality 汇聚 Gram：

$$
H_t = \frac{1}{N_t}\sum_{x \in \text{text tokens}} x x^\top,\quad
H_v = \frac{1}{N_v}\sum_{x \in \text{vision tokens}} x x^\top
$$

混合 + 数值稳定化：

$$
H = \pi_t \cdot H_t + \pi_v \cdot H_v + \lambda_\text{floor}\cdot I + \sigma_A^2 \cdot I
$$

- **π_t ∈ [0, 1]**: 文本通道权重，当前 best π_t = 0.3（vision-biased）
- **π_v = 1 − π_t**
- **λ_floor**: 对角保底（当前 1e−2 × mean(diag(H))），防 H 奇异
- **σ_A²**: 量化激活噪声项（plan §13.5），把 W8A8 的激活扰动注入对角

实现：`jsq/compression/collector.py::collect_block_input_feat` 产出 2D feat + vision_mask；
`jsq/compression/passes/prune.py::_jsq_v5_metric` 组装 H 并求 `[H^{-1}]_{jj}`（用 Cholesky + triangular solve 取对角）。

### 1.5.2 Per-element importance → per-row top-k mask

对 layer 权重 $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$，计算：

$$
I_{ij} = \frac{W_{ij}^2}{[H^{-1}]_{jj}}
$$

按**每行**取最小的 $k = \lfloor d_\text{in} \cdot s_l \rfloor$ 个置零：

$$
M_i = \text{bottom-k}(I_{i,:}),\quad \hat W_{ij} = 0 \text{ if } j \in M_i
$$

per-row top-k 保证每个 output neuron 的可计算容量相同（同 WANDA）。
理论依据：OBS (Hassibi & Stork) — 单元素置零的二阶损失为 $W_{ij}^2 / [H^{-1}]_{jj}$。

### 1.5.3 稀疏率分配（Innovation 2，**当前使用 uniform**）

原设想：外层块级搜索 {s_l}（Fisher proxy candidate search or analytic water-fill）。
实验结果（§2-3）：两条搜索路线都不如均匀分配 → **目前 s_l ≡ s_target = 0.4375** for all l。
这意味着 Innovation 2 在当前代码库里**未启用**，只在 A 组 direct 模式下跑。

### 1.5.4 量化（后续 pass，与 v4 保持一致）

稀疏化后对**保留权重**做 W8A8 per-channel 对称量化（`jsq/quant/ops.py::quantize_per_channel`），
activation 按 per-token 动态量化。σ_A² 的值从量化前向后反推塞进 H 对角。

### 1.5.5 关键超参（实际命令行）

```
--model Qwen/Qwen2-VL-7B-Instruct
--calib_dataset gqa --n_samples 128
--pruning_method jsq_v5
--sparsity_ratio 0.4375
--w_bits 8 --a_bits 8
--pi_t 0.3                   # best；也跑了 0.5, 0.7
--lambda_floor 1e-2
--max_tokens 4096            # 每层用的 token 上限（内存控制）
```

### 1.5.6 方法 vs v1/v4 baseline

| 维度 | v1 (WANDA) | v4 (token-weighted) | **v5 (mixture-H OBS)** |
|---|---|---|---|
| Importance | $\lvert W_{ij}\rvert \cdot \lVert X_{:,j}\rVert_2$ | $\lvert W_{ij}\rvert \cdot \sqrt{\sum_t w_t X_{t,j}^2}$ | $W_{ij}^2 / [H^{-1}]_{jj}$ |
| Token 加权 | 均匀 | per-token (modality-aware) | per-token (进 H 里) |
| Modality 处理 | 无 | 权重乘 token | **Hessian 层面线性组合** |
| 二阶信息 | ✗ | ✗ | ✓ (OBS) |
| 跨层 budget | uniform | Fisher search (B_fix) | **uniform (best)** |

### 1.5.7 方法当前局限

- **π_t 全局单值**：未做 per-layer 自适应（见 §7 Option D 下 P0b）
- **σ_A² 贡献未拆解**：论文需要 ablation（§7 Option D 下 P0）
- **Innovation 2 未跑通**：均匀分配在本 model × 本 calib 上已是经验最优

### 1.5.8 完整推导

#### Step 1 — Task loss → Layer reconstruction surrogate

原始 task loss $\mathcal{L}_\text{task}(\theta)$（CE over VL outputs）对某层权重 $W$ 展开：

$$
\mathcal{L}_\text{task}(W + \Delta W) \approx \mathcal{L}_\text{task}(W) + g^\top \text{vec}(\Delta W) + \tfrac{1}{2}\text{vec}(\Delta W)^\top H_\text{task} \text{vec}(\Delta W)
$$

在剪枝/量化 finetune-free 场景下，$g \approx 0$（已收敛），主导项是二阶。full $H_\text{task}$ 难算，OBS/SparseGPT 换**层级 surrogate**：

$$
\mathcal{L}_\text{layer}(W) = \mathbb{E}_{x \sim \mathcal{D}_\text{calib}} \lVert (W - W_0) x \rVert_2^2
$$

其中 $W_0$ 是原始权重，$(W - W_0)x$ 是剪枝造成的 pre-activation 扰动。

**合理性**：残差 + LayerNorm 架构下，每层扰动在下一层几乎线性传播；$\mathcal{L}_\text{layer}$ 是 $\mathcal{L}_\text{task}$ 的 Gauss-Newton 近似。

#### Step 2 — Layer surrogate 的 Hessian

对 $W$ 的 row $w_i$（每行独立），

$$
\mathcal{L}_\text{layer}^{(i)}(w_i) = \mathbb{E}_x \lvert (w_i - w_{0,i})^\top x \rvert^2 = (w_i - w_{0,i})^\top \mathbb{E}_x[x x^\top] (w_i - w_{0,i})
$$

所以

$$
\boxed{\ H := \frac{\partial^2 \mathcal{L}_\text{layer}^{(i)}}{\partial w_i \partial w_i^\top} = 2\, \mathbb{E}_x[x x^\top] = \frac{2}{N} X^\top X\ }
$$

Hessian 对所有 row 相同（layer 级共享），这是 OBS 的 "diagonal Hessian across rows" 假设的真正含义。

#### Step 3 — Modality mixture

Qwen2-VL 的 calibration batch 有 text + vision 两种 token，分布差异大（vision 多是平滑连续、text 多是稀疏 one-hot-like）。

设 calib 分布 $\mathcal{D} = \pi_t \mathcal{D}_t + \pi_v \mathcal{D}_v$（$\pi_t + \pi_v = 1$），则 $\mathcal{L}_\text{layer}$ 线性分裂：

$$
\mathcal{L}_\text{layer} = \pi_t \cdot \mathbb{E}_{x \sim \mathcal{D}_t} \lVert \Delta W x \rVert^2 + \pi_v \cdot \mathbb{E}_{x \sim \mathcal{D}_v} \lVert \Delta W x \rVert^2
$$

对应 Hessian：

$$
\boxed{\ H = \pi_t \underbrace{\mathbb{E}_{\mathcal{D}_t}[xx^\top]}_{= H_t} + \pi_v \underbrace{\mathbb{E}_{\mathcal{D}_v}[xx^\top]}_{= H_v}\ }
$$

**注意**：v4 "token-weighted" 等价于给每个 token 的 outer product 加个标量权 $w_t$ 再求和。v5 的 mixture 是**分 modality 求期望再线性组合**——理论上更严格，且允许 $\pi_t$ 作为独立超参。

#### Step 4 — 加入激活量化噪声

推理时 $\tilde x = Q(x) = x + \epsilon$，假设

$$
\mathbb{E}[\epsilon] = 0,\quad \mathbb{E}[\epsilon \epsilon^\top] = \sigma_A^2 I
$$

（uniform quantizer + 噪声各维独立同方差假设）。期望 reconstruction loss：

$$
\mathbb{E}_{x, \epsilon}\lVert \Delta W (x+\epsilon)\rVert^2 = \mathbb{E}_x \lVert \Delta W x \rVert^2 + \sigma_A^2 \cdot \text{tr}(\Delta W^\top \Delta W)
$$

第二项对 $w_i$ 的 Hessian 是 $2\sigma_A^2 I$，合并得：

$$
\boxed{\ H_\text{full} = \pi_t H_t + \pi_v H_v + \sigma_A^2 I\ }
$$

工程上再加 $\lambda_\text{floor} I$ 保证正定（对理论无影响，只保 Cholesky）：

$$
H = \pi_t H_t + \pi_v H_v + (\sigma_A^2 + \lambda_\text{floor}) I
$$

#### Step 5 — OBS per-element importance

给定 $H$，把 $w_{ij}$ 置零但让其他元素做补偿（OBS 的核心）：

$$
\min_{\Delta w_i}\ \Delta w_i^\top H \Delta w_i \quad \text{s.t.}\ (w_i + \Delta w_i)_j = 0
$$

Lagrangian 写成 $\mathcal{L} = \Delta w_i^\top H \Delta w_i - 2\mu (e_j^\top(w_i + \Delta w_i))$，
KKT：$H \Delta w_i = \mu e_j$，$\Delta w_i = \mu H^{-1} e_j$，$\mu = -w_{ij} / [H^{-1}]_{jj}$。

最优误差：

$$
\boxed{\ \Delta \mathcal{L}_{ij} = \Delta w_i^\top H \Delta w_i = \frac{w_{ij}^2}{[H^{-1}]_{jj}}\ }
$$

**这就是 v5 importance**。

推广到同时置零多个元素 $S_i \subset \{1, \dots, d_\text{in}\}$，**二阶近似** + **假设被剪 column 在 $H$ 其余部分几乎不耦合**：

$$
\Delta \mathcal{L}_i^{(S_i)} \approx \sum_{j \in S_i} \frac{w_{ij}^2}{[H^{-1}]_{jj}} = \sum_{j \in S_i} I_{ij}
$$

即**可加近似**。这个近似在 $|S_i|/d_\text{in} \ll 1$ 时准确，在 $s \to 1$ 时失效（§3 F8 根因）。

#### Step 6 — Per-row top-k 预算分配

对 row $i$，要剪掉 $k = \lfloor d_\text{in} \cdot s_l \rfloor$ 个元素最小化 total error：

$$
S_i^* = \arg\min_{|S_i| = k} \sum_{j \in S_i} I_{ij} = \text{bottom-k}(I_{i,:})
$$

即**取 I 值最小的 k 个置零**。per-row 独立，是 WANDA/SparseGPT 通用做法。

#### Step 7 — 跨层 budget（未实现的 Innovation 2）

**理论**：块级总误差 $E_\text{block}(\{s_l\}) = \sum_l E_l(s_l)$，约束 $\sum_l s_l \cdot p_l = s_\text{target} \sum_l p_l$（$p_l$ = 层 param 数）。

每层的 error curve：

$$
E_l(s_l) = \sum_i \sum_{j \in \text{bottom-}k_l(\text{row }i)} I_l[i,j] \quad (k_l = \lfloor d_\text{in}^l s_l \rfloor)
$$

**water-filling** (已实现但失败)：KKT → $\partial E_l / \partial s_l = \lambda \cdot p_l$ for all l → 二分搜 $\lambda$。
**为什么失败**（§3 F8）：Step 5 的可加近似在 $s_l \to 0.9$ 不合法；且不同层 $I_l$ 尺度不可比，water-fill 会把便宜层推到极端。

---

**Summary of derivation**（一图一行）：

$$
\mathcal{L}_\text{task} \xrightarrow{\text{GN approx}} \mathcal{L}_\text{layer} \xrightarrow{\text{modality split}} \pi_t \mathcal{L}_t + \pi_v \mathcal{L}_v \xrightarrow{\text{act quant}} +\sigma_A^2 \lVert \Delta W\rVert_F^2
$$

$$
\Rightarrow H = \pi_t H_t + \pi_v H_v + \sigma_A^2 I \xrightarrow{\text{OBS}} I_{ij} = \frac{W_{ij}^2}{[H^{-1}]_{jj}} \xrightarrow{\text{row top-k}} \text{mask}
$$

---

## 2. 实验矩阵（已完成）

### 2.1 Raw results

| ID | method | π_t | γ | search | MME_cog | MME_per | MME_total | MMStar | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| **A0** | v5 | 0.5 | — | ✗ | 617.50 | 1654.52 | **2272.02** | 0.5307 | direct |
| **A1** | v5 | 0.7 | — | ✗ | 608.93 | 1650.55 | 2259.48 | 0.5322 | direct |
| **A2** | v5 | 0.3 | — | ✗ | **625.00** | 1640.22 | 2265.22 | **0.5378** | direct, best |
| B0 | v5 | 0.5 | 1.0 | Fisher | 610.36 | 1565.38 | 2175.74 | 0.5055 | 原始 (mask bug) |
| B1 | v5 | 0.7 | 1.0 | Fisher | 620.71 | 1582.61 | 2203.32 | 0.5067 | 原始 (mask bug) |
| B2 | v5 | 0.7 | 3.0 | Fisher | 596.07 | 1581.30 | 2177.38 | 0.5157 | 原始 (mask bug) |
| B0_fix | v5 | 0.5 | 1.0 | Fisher | 608.21 | 1620.67 | 2228.88 | 0.5146 | P0 修复后 |
| B1_fix | v5 | 0.7 | 1.0 | Fisher | 617.14 | 1581.58 | 2198.72 | 0.5125 | P0 修复后 |
| B2_fix | v5 | 0.7 | 3.0 | Fisher | 603.21 | 1597.65 | 2200.86 | 0.5078 | P0 修复后 |
| C0 | v1 | — | 1.0 | Fisher | 566.43 | 1607.47 | 2173.90 | 0.5467 | v1 baseline |
| C1 | v4 | — | 3.0 | Fisher | 576.07 | 1621.81 | 2197.88 | 0.5456 | v4 baseline |
| **D0** | v5 | 0.5 | — | **analytic** | **434.29** | **1417.51** | **1851.80** | **0.4067** | 崩盘 |
| **D1** | v5 | 0.7 | — | **analytic** | **463.21** | **1411.12** | **1874.33** | **0.4385** | 崩盘 |
| **D2** | v5 | 0.3 | — | **analytic** | **410.00** | **1387.61** | **1797.61** | **0.3922** | 崩盘 |

### 2.2 Key deltas

- **v5 direct vs v4 baseline**: A2 (625) − C1 (576) = **+48.93 MME_cog** → plan §13 目标达成
- **v5 direct vs v1 baseline**: A0 (2272) − C0 (2174) = **+98 MME_total**
- **v5 search vs direct**: B0_fix (2229) − A0 (2272) = **−43 MME_total**（搜索反而变差）
- **v5 analytic vs direct**: D0 (1852) − A0 (2272) = **−420 MME_total**（搜索灾难性变差）

---

## 3. 核心发现

### F1. Innovation 1（mixture-H metric）✅ 成立
v5 per-element importance 在 direct 模式下显著优于 v1/v4，MME 两项全部达到 plan 目标。
MMStar 仍落后 C 组 ~0.9 pp（reasoning 偏科，与 calib 分布有关）。

### F2. π_t 敏感性反直觉
π_t ∈ {0.3, 0.5, 0.7} 下 MMStar {0.5378, 0.5307, 0.5322}，**vision-biased (π_t=0.3) 最优**。
提示 §3.3 子空间假设成立：vision H 给文本通道提供有效正则。

### F3. search × v5 失配（已排除 P0 假设）
初诊：`block_search.py:530` 候选评估 `vision_mask_for_metric=None` → v5 退化为纯文本 mask。
**修复后**（B_fix）：B0 MME +53 / MMStar +0.9 pp；但仍劣于 A direct 43 MME → **修复必要不充分**。

### F4. F6 假设被证伪（analytic water-fill 崩盘）
假设"外层 Fisher proxy vs 内层 input-space v5 的流形失配是主因"。
实现 `analytic_search.py`（water-filling on Σ_l E_l(s_l)，严格对齐 v5 OBS input-space）。
D 结果：budget 严格满足（`mean_s=0.4375`, ΣE 最小化），但 **MME −420 / MMStar −10~15 pp**。

### F5. 真正根因（F8 重构）
1. OBS additive 假设在 s_l > 0.5 时失效；
2. 各层 I 值**尺度不可比**（W_l Frobenius × H_l 对角分布跨层差 10×）；
3. water-fill 把便宜层推至 s_max=0.9 → 该层输出崩 → block 崩；
4. Fisher proxy 在输出空间天然惩罚极端分配 → 隐式正则，这是 B 组不崩的原因。

### F6. Innovation 2 需重新定位
| 配置 | per-element mask | 跨层 budget | MME_total |
|---|---|---|---|
| A (direct) | v5 mixture-H ✓ | uniform ✓ | **2272** |
| B_fix (Fisher search) | v5 mixture-H ✓ | ±0.05 扰动 | 2229 |
| D (analytic) | v5 mixture-H ✓ | full-range water-fill | 1852 |

跨层非均匀分配**理论上应有效**，但当前两种实现都不好。
非均匀分配的核心挑战 = scale normalization + tight bounds。

---

## 4. 代码资产

### 已修改 / 新增
- `jsq/compression/block_search.py`:
  - 新增 `_subsample_vision_mask(full_mask, n_feat_tokens, max_tokens)`
  - `_evaluate_candidate(..., vision_mask_for_metric=lite_vision_mask)` (P0 修复)
  - 新增 `_analytic_search_and_apply_v5()` + 在 `search_and_apply` 里 dispatch (`config.pruning_method == "jsq_v5"`)
- `jsq/compression/analytic_search.py` (新, ~170 行):
  - `compute_layer_error_curve` — 每行排序累加
  - `_optimal_s_for_lambda` — searchsorted 找 KKT 交点
  - `water_fill_allocation` — λ 二分搜索 + 最终 rescale
  - `allocate_v5` — 端到端入口

### Commits
- `d4501ab`: size-align vision_mask for full-sample input_feat in block search
- `4258098`: wire jsq_v5 + pi_t/lambda_floor flags through argparse
- `971c373`: feat: JSQ v5 mixture-Hessian OBS metric
- `0a89a1a`: feat: improve Hessian block search quality
- `4d2c580`: fix: unsqueeze 2D multimodal inputs in block_search _run_forward_lite

（analytic_search.py 及相关改动**尚未提交**。）

### 日志
- `logs/v5_A{0,1,2}_direct_pt*.log` — A 组 direct
- `logs/v5_B{0,1,2}_search_pt*_g*.log` — B 组 原始
- `logs/v5_B{0,1,2}_fix_search_pt*_g*.log` — B 组 P0 修复后
- `logs/v5_C{0,1}_v{1,4}_search.log` — baseline
- `logs/v5_D{0,1,2}_analytic_pt*.log` — D 组 analytic

---

## 5. 论文叙事影响

### 当前可讲的 story
- **v5 mixture-Hessian OBS metric** 是一个稳健的 per-element 重要性（Innovation 1）。
- 相对 v4 WANDA/token-weighted baseline 在 MME cognition +48.9 / perception +18.4。
- 机制上：vision 和 text token 有独立 Hessian 结构（§2 推导），混合能抵消 calib 分布偏置。

### 需要修正 / 删除的 claim
- ❌ plan §13.6 "两层 Hessian 乘法增益" — 被 B/D 组共同否定。
- ❌ plan §13.3 关于 search 带来的增益 — 删或降权。
- ⚠️ plan §3.5 跨层 allocation 的推导 — 保留作为 motivation，但实验 section 需展示 water-fill 反例作为负面 ablation。

### 缺的实验 / writing（影响投稿）
- **σ_A² ablation** — 论文必备（A2 w/ vs w/o σ_A²）
- **π_t 自适应** — 解释为何 π_t=0.3 最优
- **子空间夹角实证** — §8 writing 前置，决定"mixture"章节是否成立
- **Innovation 2 救赎** — 需要一个有效的跨层分配机制才能保住 2 个 innovation

---

## 6. 已排除的搜索路线

| 路线 | 代价 | 结果 | 教训 |
|---|---|---|---|
| Fisher proxy + ±0.05 扰动 | 低 (~100 min calib) | B_fix ≤ A direct | 扰动太小，贴近 uniform 无增益 |
| Analytic water-fill [0, 0.9] | 中 (~10 min calib) | D ≪ A direct | scale 不可比 + s_max=0.9 超出 OBS 合法区间 |

---

## 7. 下一步候选

### ⭐ Option E — Per-block Multimodal Sensitivity Allocation（**当前首选，待实现**）

**设计哲学**：v5 的 mixture 原则贯穿两个粒度——element 级混 Hessian，block 级混 sensitivity，共享同一个 $\pi_t$。

#### E.1 对称结构

- Innovation 1 (element): $H = \pi_t H_t + \pi_v H_v + (\sigma_A^2 + \lambda_\text{floor}) I$
- Innovation 2 (block): $c_l = \pi_t\, c_l^{(t)} + \pi_v\, c_l^{(v)}$

#### E.2 Per-modality block score（三候选）

对每个 decoder block $l$，基于 calib 前向（可选反向）分 modality 计算：

**Score-A: Block Influence (BI，纯前向)**
$$
c_l^{(m)} = 1 - \frac{\langle X_l^{(m)},\ Y_l^{(m)}\rangle}{\lVert X_l^{(m)}\rVert\cdot\lVert Y_l^{(m)}\rVert},\quad m \in \{t, v\}
$$

**Score-B: Output-change energy**
$$
c_l^{(m)} = \frac{\mathbb{E}\lVert Y_l^{(m)} - X_l^{(m)}\rVert^2}{\mathbb{E}\lVert X_l^{(m)}\rVert^2}
$$

**Score-C: Taylor first-order（需反向）**
$$
c_l^{(m)} = \lvert \mathbb{E}[g_l^{(m)\top}(Y_l^{(m)} - X_l^{(m)})]\rvert
$$

#### E.3 Mixture + Fusion bonus
$$
c_l = \pi_t c_l^{(t)} + \pi_v c_l^{(v)} + \beta \cdot \min(c_l^{(t)}, c_l^{(v)})
$$
$\beta \geq 0$；$\beta > 0$ 显式保护 cross-modal fusion 层。

#### E.4 Score → sparsity mapping

$$
s_l = \text{rescale}\!\left(\text{clip}\!\left(s_\text{target} - \alpha\cdot\frac{c_l - \bar c}{\bar c},\ s_\text{target}\pm 0.1\right)\right)
$$

tight clip（±0.1）避开 D 组 water-fill 崩盘；rescale 保证总预算严格满足。

#### E.5 Block 内仍 uniform

block 内所有投影层 (q/k/v/o/gate/up/down) 共享 $s_l$，用 v5 per-element mask 剪枝。
继承 A direct 的稳定性。

#### E.6 实验矩阵（E 组）

| ID | Score | π_t | α | β | 目的 |
|---|---|---|---|---|---|
| E0 | BI mixture | 0.3 | 0.3 | 0 | 主方法 |
| E1 | BI text-only | 0.3 | 0.3 | 0 | 消融 vision 信号 |
| E2 | BI vision-only | 0.3 | 0.3 | 0 | 消融 text 信号 |
| E3 | BI mixture + fusion | 0.3 | 0.3 | 0.5 | fusion-aware |
| E4 | Output-energy | 0.3 | 0.3 | 0 | score 变体 |
| E5 | Taylor | 0.3 | 0.3 | 0 | 一阶 score |
| E6 | BI mixture 强扰 | 0.3 | 0.5 | 0 | α sweep（反证 clip 必要性）|

**预期**：E3 ≥ E0 > E4, E5 ≥ A direct > E1, E2, E6

#### E.7 实现成本

| 文件 | 改动 | 行数 |
|---|---|---|
| `jsq/compression/collector.py` | `collect_block_sensitivity()` hook | ~60 |
| `jsq/compression/block_vl_allocator.py` | 新文件：三种 score + mapping | ~100 |
| `jsq/compression/pipeline.py` | 接入 per-block sparsity dict | ~15 |
| `jsq/configs.py` / `main.py` | CLI flag `--block_alloc_method` | ~10 |
| `scripts/sweep_v5_vl_block.sh` | E0–E6 脚本 | ~40 |

**合计 ~225 行**。calib 每个 config ~5–10 min + eval ~30 min/GPU。3 GPU 并行跑完 E 组约 3 小时。

#### E.8 论文叙事影响

若 E0/E3 > A direct：
- Innovation 2 救活，paper 从"单 innovation"回到"dual innovation"
- Selling point：**dual-level mixture**（element + block）都用同一个 $\pi_t$
- §4 改写：§4.1 element mixture, §4.2 block mixture, §4.3 unified framework
- §3 Motivation 加"VL block 三角色"（text-specialized / vision-specialized / fusion）散点图

若 E0 ≤ A direct：Innovation 2 降级为 ablation（"跨层分配在此模型/calib 组合下收敛到 uniform，验证 v5 metric 本身已接近最优"）。

---

### Option A — Layer-type grouped allocation（最推荐）
按架构分组：`{q,k,v}`、`{gate,up}`、`{down,o}` 各共享一个 s（每 block 3 旋钮）。
用 B_fix 机制搜（Fisher proxy + ±0.05 扰动）。
- **优点**：搜索空间从 7^n 降到 3^n；架构先验自带正则；论文有"layer-type sensitivity"叙事
- **实现**：~30 行（`block_search.py` 候选生成处）
- **预期**：MME ≥ A direct，可能 +20~50

### Option B — Normalized marginal + tight bounds
在 water-fill 前对每层 I 归一化：$\tilde I_l = I_l / E_l(s_\text{target})$，
并把 allocation clip 在 $[s_\text{target} - 0.08, s_\text{target} + 0.08]$。
- **优点**：直接解决 scale 不可比 + 避免 s_l→0.9 病态
- **实现**：~50 行（改 `analytic_search.py`）
- **风险**：若 tight bound 太窄，退化为 uniform（同 B 组）

### Option C — Global layer-type allocation（跨 block）
7 个 s（per layer type，全模型共享），一次性搜出。
- **优点**：story 最干净；搜索空间最小
- **缺点**：实现最多（~150 行），失去 per-block adaptivity

### Option D — 非搜索增强（并行）
- **P0 σ_A² ablation**（~30 min）
- **P0b π_t 逐层 oracle 网格**（~4 小时）
- **P1 子空间夹角实证**（~1 小时脚本 + 半小时作图）

**决定**：**优先跑 Option E**（per-block multimodal allocation），因为它：
1. 直接利用 v5 的 mixture 框架（story 对称）
2. 避开 D 的 scale 不可比陷阱（block 输出空间单位可比）
3. 继承 A direct 的 block 内稳定性（tight clip + 内部 uniform）
4. 实现成本 225 行、3 小时跑完

**次选**：若 E 不过 → Option A (layer-type grouped) 作为 fallback。
**并行**：P0 σ_A² ablation（论文必备，不阻断主路径）。

---

## 8. 文档索引

- `plan/jsq_v5_mixture_hessian.md` — v5 原始 plan（§1-§13）
- `plan/jsq_v5_zero_bit_allocation.md` — v5 第二贡献设计：Zero-Bit Rate-Distortion Allocation
- `plan/exp_v5_sweep_20260414.md` — A/B/C/D sweep 结果与 F1-F9 findings
- `plan/exp_v5_next_steps_20260414.md` — 优先级清单（updated 2026-04-15）
- `plan/jsq_v5_progress_20260416.md` — **本文件（最新进展总结）**

---

## Status: decision-pending (Innovation 2 路径待选)
