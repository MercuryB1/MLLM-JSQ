# JSQ v5 第二贡献设计：分歧正则的混合 Hessian

**状态**：设计草案，未实现  
**日期**：2026-04-23

---

## 1. 动机

当前 v5 的核心 metric 是平均风险形式：

$$
H_l^{mix} = \pi_l C_{t,l} + (1-\pi_l) C_{v,l} + \lambda_l I
$$

其中 `C_t, C_v` 是 text / vision token 的二阶统计。这个设计已经能稳定提升剪枝 mask，但它默认一个前提：**两模态的高曲率方向大体一致**。  

问题在于，多模态 LLM 的中后层往往包含 text-specialized、vision-specialized 和 fusion 三类 block。对 fusion 层来说，`C_t` 和 `C_v` 的主方向可能并不一致。此时简单 mixture 更像“平均掉分歧”，会优先保住共同方向，却可能牺牲某一模态独有但重要的方向。

已有实验表明，沿着 block-level 非均匀稀疏率分配去补救效果不稳定，说明问题不在外层 budget，而在 **metric 本身还缺少对跨模态分歧方向的建模**。因此第二个 contribution 不再做 allocation，而是直接扩展同一个 joint pruning-quantization metric。

---

## 2. 方法

我们将 v5 的 mixture Hessian 扩展为 **分歧正则的 robust Hessian**：

$$
H_l^{rob} = \pi_l C_{t,l} + (1-\pi_l) C_{v,l} + \beta_l D_l + \lambda_l I
$$

逐元素重要性保持不变：

$$
I_{ij}^{rob} = \frac{W_{ij}^2}{[(H_l^{rob})^{-1}]_{jj}}
$$

也就是说，剪枝与量化仍然由**同一个 Hessian**统一描述；新项 `D_l` 只负责保护 text / vision 分歧较大的方向。

### 2.1 二阶统计

$$
C_{t,l} = \frac{X_{t,l}^\top X_{t,l}}{n_t}, \qquad
C_{v,l} = \frac{X_{v,l}^\top X_{v,l}}{n_v}
$$

其中 `X_t, X_v` 为当前层输入激活的 text / vision 子集。

### 2.2 分歧项 `D_l`

先定义模态差分矩阵：

$$
G_l = C_{t,l} - C_{v,l}
$$

目标形式使用 `G_l` 的低秩谱包络：

$$
D_l = \sum_{k=1}^{r} |\sigma_{lk}| u_{lk} u_{lk}^\top
$$

其中 `(σ_{lk}, u_{lk})` 是 `G_l` 的 top-r 特征对。这样 `D_l` 是 PSD 的，并显式保护“两个模态曲率差异最大”的方向。

为降低首版实现风险，可先用对角近似：

$$
D_l^{diag} = \mathrm{Diag}\left(|\mathrm{diag}(C_{t,l}) - \mathrm{diag}(C_{v,l})|\right)
$$

首版先验证 `D_l^{diag}` 是否有效；若有效，再升级到低秩 `D_l`。

---

## 3. 自校准参数

为了避免引入新的手调超参，`pi_l` 与 `beta_l` 都由层统计量直接估计。

### 3.1 模态先验 `pi_l`

以当前最优全局先验 `\pi_0 = 0.3` 为中心，只做小幅残差修正：

$$
\pi_l =
\mathrm{clip}\left(
(1-\alpha)\pi_0 +
\alpha \frac{\mathrm{tr}(C_{t,l})}{\mathrm{tr}(C_{t,l}) + \mathrm{tr}(C_{v,l}) + \varepsilon},
\pi_{min}, \pi_{max}
\right)
$$

默认建议：`alpha = 0.3`, `pi_min = 0.1`, `pi_max = 0.5`。

### 3.2 分歧强度 `beta_l`

用模态子空间重叠度控制分歧项的强弱：

$$
o_l =
\frac{\|X_{t,l}^\top X_{v,l}\|_F}
{\|X_{t,l}^\top X_{t,l}\|_F^{1/2}\|X_{v,l}^\top X_{v,l}\|_F^{1/2} + \varepsilon}
$$

$$
\beta_l = \beta_{max}(1 - o_l)
$$

含义很直接：

- overlap 高：两模态子空间接近，退回普通 mixture
- overlap 低：更可能是 fusion / specialized 层，增强分歧保护

默认建议：`beta_max = 0.2`。

---

## 4. 这条设计为什么和主线一致

- **仍然只有一个 joint metric**：剪枝 mask 仍由 `W^2 / diag(H^{-1})` 决定。
- **量化仍在 metric 内部**：`lambda_l` 继续吸收激活量化噪声与 shrinkage，不额外引入独立的 quant module。
- **第二贡献是 metric 的 robust 扩展**：不是外层 budget search，也不是另挂一个后处理技巧。

论文叙事可以直接写成：

1. v5 用 mixture Hessian 统一建模 pruning 与 quantization。  
2. 在 fusion 层，average-case curvature 不足以保护跨模态分歧方向，因此引入 disagreement regularization。  

---

## 5. 实现落点

- `jsq/compression/passes/prune.py`
  - 在 `_jsq_v5_metric()` 上扩展 `pi_l`、`beta_l`、`D_l`
- `jsq/compression/hessian_utils.py`
  - 先支持对角 `D_l`
  - 后续若做低秩版，可把 `\sqrt{\beta_l |\sigma_k|} u_k^\top` 作为额外虚拟行拼到 Woodbury 的 `U` 中
- `jsq/config.py` / `main.py`
  - 新增 `pi_alpha`, `beta_max`, `gap_rank`

首版不改 search、不改 allocation、不改 smooth/clip/quant pass。

---

## 6. 最小实验矩阵

- `A2`：当前 v5 baseline，固定 `pi_t = 0.3`
- `R1`：只开 `pi_l` 自校准，`beta_l = 0`
- `R2`：固定 `pi = 0.3`，只开 `D_l^{diag}`
- `R3`：`pi_l + D_l^{diag}` 全开

如果 `R2 / R3` 优于 `A2`，说明 novelty 来自“分歧正则”本身，而不只是自适应权重。
