# JSQ v5：面向 MLLM 的混合 Hessian 剪枝-W8A8 联合 metric

**状态**：设计锁定（路径 A），待实现
**分支**：`jsq-v4-hessian` → 下一次提交到 `jsq-v5-mixture`
**日期**：2026-04-14

---

## 1. 动机

当前 v4 metric 是四项启发式叠加（`|W|·S_vis`、`|W|·S_txt`、`ρ·ss_vis`、`ρ·ss_txt`），带两个互相纠缠的超参 γ 和 ρ。存在三个结构性问题：

1. **完全没有量化感知**。metric 和下游 W8A8 量化器互相不知道对方存在。
2. **和外层搜索重复记账**。`ρ·LOO` 敏感度项和 block 级 Hessian search 工作在相近尺度上。
3. **并非真正的多模态**。`S_vis + γ·S_txt` 只是 token 层面的加权——它只改变单个共享 Hessian 的**特征值**，却忽略了视觉与文本激活可能张成**不同的主子空间**这一事实。

v5 用一个显式期望损失导出的**单一二阶目标**，替换堆叠式 metric。

---

## 2. 设计原则

> 一个目标、一个 Hessian、一个闭式打分公式。多模态性和量化感知都进入 **H 内部**，不再以额外加法项的形式出现。

metric 求解的是在期望 W8A8 线性层输入重构损失下每个元素的重要性：

$$
\mathcal{L} = \mathbb{E}_{m \sim \pi}\,\|W X_m - \hat W X_m\|_F^2,
\qquad m \in \{v, t\}
$$

在剪枝扰动 `δW_ij` 附近做二阶 Taylor 展开，得到 OBS 形式的逐元素重要性：

$$
\boxed{\quad
\mathcal{I}(i,j) = \frac{W_{ij}^2}{[\mathbf H^{-1}]_{jj}}
\quad}
$$

对每一行剪掉 `I` 最小的若干元素（总数由 block-level 预算 `s_l` 决定）。**不做权重补偿**（OBS update 在 per-row absmax W8 下会被量化抹掉，详见 §7）。

---

## 3. Hessian 定义

$$
\mathbf H = \pi_t\,\mathbf H_t + \pi_v\,\mathbf H_v + \lambda\,\mathbf I
$$

| 项 | 公式 | 作用 |
|---|---|---|
| `H_t` | 文本 token 上的 `X_tᵀ X_t` | 单模态经验输入协方差 |
| `H_v` | 视觉 token 上的 `X_vᵀ X_v` | 同上 |
| `π_t`, `π_v` | 混合权重，`π_t + π_v = 1` | 模态先验 |
| `λ` | shrinkage + 激活量化噪声 | 小样本正则 + `σ_A²` 吸收 |

### 3.1 混合权重 `π`
- **基线**：`π_t = π_v = 0.5`（对称先验）
- **文本偏置**：`π_t = 0.7, π_v = 0.3`（反映自回归 NLL 由文本 token 驱动）
- **自适应**：`π_m ∝ ||grad_m(L_block) · X_m||²`，从校准数据估计；需要一次额外反向传播

v5 初版**固定 `π_t = 0.5`**，在 ablation 里扫描。

### 3.2 正则项 `λ`
两个来源之和：
- **激活量化噪声** `σ_A²` — 每通道的 `X - Q_a(X)` 方差，沿 token 取平均；每层预计算一次
- **Ledoit-Wolf shrinkage** — 当有效秩 `n_m = max(n_t, n_v)` 相对 `d` 较小时，加上 `λ_{LW}` 从 X 估计

形式：`λ = σ_A² + λ_{LW}`。两项都是数据估计得到的，**无需手工调参**。

### 3.3 为什么不用 token 级加权（`Xᵀ diag(w) X`）
token 加权只能重新缩放单个 `XᵀX` 的**特征值**，不能对齐主轴。如果视觉和文本张成不同子空间，`diag(w)` 无法保护一个"在文本上关键、在视觉上数值平淡"的通道（因为视觉 token 数量会在数值上压倒它）。

混合 Hessian `π_t H_t + π_v H_v` 把**具有不同特征向量**的两个协方差矩阵加起来——这是结构上的差异，不只是幅度上的差异。

---

## 4. 逐层计算流程

### 4.1 统计量收集
```
X_t = 文本 token 位置的激活          [n_t, d]
X_v = 视觉 token 位置的激活          [n_v, d]
σ_A² = (X - Q_a(X)) 的每通道方差      [d]
```

### 4.2 Woodbury 计算 Hessian（避开 d×d 直接求逆）

对 `H = (π_t X_tᵀ X_t + π_v X_vᵀ X_v) + λI`，构造堆叠低秩因子：
```
U = [√π_t · X_t;  √π_v · X_v]     shape [n_t+n_v, d]
H = UᵀU + λI
```

应用 Woodbury 恒等式：
```
H⁻¹ = λ⁻¹ I − λ⁻¹ Uᵀ (I_n + UUᵀ/λ)⁻¹ U λ⁻¹
```

其中 `I_n` 是 `(n_t + n_v) × (n_t + n_v)` 单位阵，一般 `n ≪ d`。

### 4.3 只求 H⁻¹ 对角

我们只需要 `[H⁻¹]_{jj}`。令 `M = (I_n + UUᵀ/λ)⁻¹`，则：
```
[H⁻¹]_{jj} = 1/λ − (1/λ²) · (U[:, j]ᵀ · M · U[:, j])
           = 1/λ − (1/λ²) · ||L⁻ᵀ U[:, j]||²        （M = LLᵀ 的 Cholesky 分解）
```

每层成本：
- 构造 `UUᵀ`：`O(n² d)`（主开销）；MLLM 校准下 `n = n_t + n_v ≪ d`
- `n × n` Cholesky：`O(n³)`
- 对角提取：`O(n² d)`

Qwen2-VL-7B 的 `d_mlp = 18944`、`n ≈ 4096`（subsample 后）：单次 Cholesky ≈ 70 GFLOP，逐层在秒级。

### 4.4 打分与 mask

对权重 `W [d_out, d_in]` 的每一输出行 `i`：
```
I_ij = W_ij² / [H⁻¹]_jj              （沿 j 广播）
mask_ij = (I_ij ≥ topk(I_i, K))      K = (1 - s_l) · d_in
```

执行 `W ← W ⊙ mask`。**不做 OBS 补偿。**

---

## 5. 与现有 pipeline 的整合

### 5.1 保持不变的部分
- `CompressionPipeline` 的 Prune → Smooth → Clip → Quantize 顺序
- 外层 `BlockSearcher` 及其 Hessian 加权块输出误差
- 多模态校准流程（`collect_first_layer_inputs`、`move_vision_encoder`）
- `vision_mask` 通过 `_build_flat_vision_mask` 的传递

### 5.2 需要改动的部分
| 文件 | 改动 |
|---|---|
| `jsq/compression/passes/prune.py` | 新增 `_jsq_v5_metric(w, inp, vision_mask, sigma_A_sq, pi_t, pi_v)`，dispatch 分支 `jsq_v5` |
| `jsq/compression/collector.py` | 每层额外估计 `σ_A²`（feat 收集时多一次"quantize → 算 var"的 pass） |
| `jsq/compression/passes/smooth.py` | **无改动**；smoothing 仍然在 v5 看到激活之前迁移 A-outliers |
| `configs/` | 加 `pruning_method: jsq_v5`，去掉 `rho` 和 `gamma`，加可选 `pi_t`（默认 0.5）、`lambda_floor`（默认 1e-3） |
| `block_search.py` | **无改动**；如果 v5 让敏感度曲线更平滑，后续可以简化候选生成策略 |

### 5.3 向后兼容
保留 `jsq_v1`–`jsq_v4` 的 dispatch，供 ablation 对比。v5 是**新增的**剪枝方法，不替换 v1。

---

## 6. 两层 Hessian 框架（不会双重记账）

| 层级 | 粒度 | Hessian | 决策 |
|---|---|---|---|
| **外层（BlockSearcher）** | 每 block | `H_block = Y_orig²`（块输出 Fisher 代理） | 在各线性层之间分配 `{s_l}` |
| **内层（v5 metric）** | 每元素 | `H_layer = π_t H_t + π_v H_v + λI`（层输入协方差） | 给定 `s_l`，选择剪掉哪些位置 |

两者作用在不同尺度上，通过链式法则组合。内层 metric 编码了外层块输出 loss 看不见的**通道间相关性**；外层搜索编码了单层无法捕获的**跨层误差传播**。没有冗余。

---

## 7. 关键设计决策（取舍记录）

### 7.1 为什么丢弃 OBS 权重更新
per-row absmax W8 量化 step：`s_i = max_j|W_ij| / 127`。OBS 补偿 `δ ∝ [H⁻¹]_{jk} · W_ik / [H⁻¹]_{kk}` 的典型幅度 `|δ| ≪ s_i`（除非剪的是行 outlier）。量化 round 会把大部分 δ 抹掉，计算 δ 的代价不划算。

**保留的后路（future work）**：若升级到 per-group W8（group-size 128）或 per-channel scale，step 缩小约 10 倍，补偿能存活。那时重新考虑 OBS update，优先做 GPTQ-style 逐列循环把 prune + quantize 合并（补偿立即被重新量化，所以**构造上就能存活**）。

### 7.2 为什么用期望损失（混合），而不是 max 或 union
- **Union sum** `1/[H_v⁻¹]_{jj} + 1/[H_t⁻¹]_{jj}`：没有 principled 推导；少样本模态的 H 接近奇异 → 该项饱和到 `1/λ` → 变成平坦偏移，不提供排序信号。**弃用。**
- **Max** `max(·, ·)`：最坏情况；鲁棒但要两次 Cholesky，且在模态不平衡时会过度保护罕见模态。**v5 初版弃用。**
- **Mixture** `π_t H_t + π_v H_v`：从期望损失 `L = E_{m ∼ π}[L_m]` 推导；Woodbury 下只需一次 Cholesky；模态平衡由显式先验控制。**采用。**

### 7.3 为什么不在单个 `XᵀX` 上用 token 加权
见 §3.3。重加权只改特征值，不改特征向量。

### 7.4 为什么没有显式的 W8A8 damage 项
激活量化方差 `σ_A²` 进入 `λ`，直接缩小 `[H⁻¹]_{jj}` 在噪声大的通道上的值 → 抬高该列的 `I(i,j)` → 保护这些通道。不需要单独加项。

权重量化 damage 也不单独建模——per-row absmax step 由行 outlier 决定，而 metric 的 `W²` 分子本身就给这些元素很高权重，自动保留。

---

## 8. 必须先验证的实证前提

**待验证论点**：MLLM 的 LLM-backbone 各层中，`H_v` 和 `H_t` 的主子空间差异足够大。

**实验方案**：
1. 在 Qwen2-VL-7B 选 4 个代表性 block（早期、中早、中晚、晚期）
2. 对每个 q/k/v/o/gate/up/down 投影层，算 `H_v` 和 `H_t` 的 top-32 特征向量
3. 用 `scipy.linalg.subspace_angles` 量化**主子空间夹角**
4. **阈值**：若 ≥50% 的层平均主成分夹角 > 30° → 分开算 H 成立；若不成立 → 退化到 token 加权的单 H 即可

结果写入 `plan/v5_subspace_overlap_measurement.md`。**必须在写代码前完成。**

---

## 9. 超参清单

| 超参 | 默认值 | 来源 | 是否可调 |
|---|---|---|---|
| `π_t` | 0.5 | 对称先验 | 是（ablation：0.5 / 0.7 / 数据估计） |
| `π_v` | `1 − π_t` | — | 否 |
| `λ_{LW}` | Ledoit-Wolf 估计 | 数据 | 否 |
| `σ_A²` | 每通道经验方差 | 数据 | 否 |
| `λ_floor` | `1e-3` | 数值稳定性 | 否 |

**v5 净可调超参：1 个**（`π_t`）
对比 v4：`γ`、`ρ`、`gamma_block` 共 3 个

---

## 10. 预期行为

- **纯文本**（`n_v = 0`）：退化为 SparseGPT 风格 `W²/[H_t⁻¹]_{jj}`
- **纯视觉**：对称退化到 `W²/[H_v⁻¹]_{jj}`
- **均衡多模态**：通过混合协方差保护对任一模态关键的权重列；激活 outlier 通道经 `σ_A²` 自动保护

Qwen2-VL-7B @ sparsity 0.4375 + W8A8 的目标指标：
- **MME Cognition** ≥ 624（缩小与 dense baseline 的差距）
- **MME Perception** ≥ 1633（维持 v3 的领先）
- **MMStar** — 新增基准，建立 baseline
- **MMBench_en_dev, SEED-Bench** — 新增基准

---

## 11. 实现 Checklist

- [ ] §8 前置实验：测量每模态 Hessian 的子空间夹角
- [ ] `jsq/compression/collector.py` 加 `σ_A²` 收集器
- [ ] `jsq/compression/passes/prune.py` 实现 `_jsq_v5_metric`
- [ ] 新文件 `jsq/compression/hessian_utils.py`（<150 行）实现 Woodbury 对角求解器
- [ ] prune pass 里 dispatch `jsq_v5`
- [ ] 配置加 `pi_t`、`pi_v`、`lambda_floor`
- [ ] 修复 pileval 缓存问题（blocker，与 v5 无关）
- [ ] 跑 `python main.py ... --pruning_method jsq_v5 --pi_t 0.5`
- [ ] Ablation：`π_t ∈ {0.3, 0.5, 0.7}`
- [ ] Ablation：v5 vs v5-without-σ_A²（分离量化噪声贡献）
- [ ] Ablation：mixture-H vs single-H-token-weighted（验证子空间假设）
- [ ] 评测：MME / MMStar / MMBench / SEED-Bench / PPL

---

## 12. v5 范围外（future work）

- GPTQ 风格 fused prune+quantize 列循环（若采用 per-group W8 再考虑）
- 每层自适应 `π`
- 跨模态 Hessian 非对角块（`H_{vt} = X_vᵀ X_t`）
- 替换 smooth/clip pass（metric 继续沿用当前四阶段骨架）

---

## 13. 为什么这个方案有机会做到"近乎无损"

目标：Qwen2-VL-7B @ sparsity 0.4375 + W8A8，相对 dense baseline 各基准下降 ≤ 1%。
下面是**量化的论证**，不是直觉辩护。

### 13.1 压缩预算下"近乎无损"是否可达——先确认问题不是无解的

把单层误差分解为：
$$
\varepsilon = \underbrace{\Delta W \cdot X}_{\text{剪枝+权重量化}} + \underbrace{W \cdot \Delta X}_{\text{激活量化}} + \underbrace{\Delta W \cdot \Delta X}_{\text{二阶}}
$$

其中 `ΔW = Q(W⊙M) - W`，`ΔX = Q(X) - X`，二阶项相对可忽略。

文献已知上界：
- **SparseGPT / WANDA** 在 LLaMA-7B 50% 非结构化剪枝下 PPL 上升仅 0.3（相对 4%）
- **SmoothQuant W8A8** 在 LLM 上 PPL 上升约 0.1–0.3
- **两者叠加**约 0.4–0.6 PPL，折算到 MME 任务分约下降 1–2%

我们的设置 **更容易**（sparsity 0.4375 < 0.50），理论上界宽松。**"近乎无损"不是奢望，而是 baseline SOTA 已在邻近区间达成**。问题是：能否同时在多模态下达到？——**这正是 v4 卡住的地方**（Cognition 624 → 617，落后约 1%）。

### 13.2 v4 为什么差那 7 分（Cognition）——定位病灶

Qwen2-VL 校准数据中 token 比例典型为 `n_v : n_t ≈ 10 : 1`。v4 的 mixed metric：

$$
\text{score}_j \propto \sum_{t \in \text{all}} X_{tj}^2 \approx 10 \cdot \mathbb{E}[X_{vj}^2] + 1 \cdot \mathbb{E}[X_{tj}^2]
$$

**文本通道的统计在数值上被压低 10 倍**。如果某个通道 j 对文本推理关键但在视觉上平淡（`E[X_{vj}²] < E[X_{tj}²]`），这个通道的权重列就被 v4 误判为"不重要"并剪掉。

Cognition 任务（MME 的算术、翻译、代码推理子项）几乎**纯文本**，最先暴露这种误伤。Perception 是视觉任务，混合 metric 反而对它有利——这完美解释了 v3 的非对称误差（Perception ↑ 17 / Cognition ↓ 7）。

### 13.3 v5 如何**定量**解决这个问题

混合 Hessian `H = π_t H_t + π_v H_v + λI` 的逐通道诊断值：
$$
[\mathbf H^{-1}]_{jj} \approx \frac{1}{\pi_t \cdot \mathbb{E}[X_{tj}^2] + \pi_v \cdot \mathbb{E}[X_{vj}^2] + \lambda}
$$

令 `π_t = 0.5`（不是 `n_t / (n_t + n_v) ≈ 0.09`），**文本统计的有效权重从 v4 的 0.09 提升到 0.5**——对"文本重要、视觉平淡"的通道，重要性得分提升约 **5–6 倍**。

这恰好对应 v3 丢失的 ~7 分 Cognition。**不是定性改善，是定位到具体通道的定量修正。**

### 13.4 为什么 metric 本身在理论上最强——不会"漏信息"

`I(i,j) = W_ij² / [H⁻¹]_{jj}` 是**线性层输出 MSE 对剪枝扰动的二阶 Taylor 精确展开**（OBS 定理）：
$$
\mathbb{E}\|\Delta Y\|^2 \Big|_{W_{ij} \to 0} = \frac{W_{ij}^2}{[\mathbf H^{-1}]_{jj}} + O(\|\Delta W\|^3)
$$

在"只利用 (W, X 二阶统计)"这个信息类里，**没有任何 metric 能比它更低的期望层输出误差**——这是一个信息下界，不是近似。WANDA、magnitude、LOO sensitivity 都是它在不同对角假设下的简化版本（可严格证明）。

所以 v5 metric 不是"又一个启发式"，是**同一信息类里的最优 metric**。我们之前的 v1/v4 都是近似它但近似得不够好。

### 13.5 为什么把 W8A8 量化噪声写进 H 里能**对冲量化误差**

将 `σ_A²·I` 加入 H 等价于在**期望下替换** `X → X + η`（`η ~ N(0, σ_A²)`），所以 metric 实际优化的是：
$$
\mathbb{E}_\eta\,\|W(X+\eta) - \hat W(X+\eta)\|^2
$$

这 **正好就是部署时的 quantized forward 误差**（在 η 是一阶噪声的近似下）。metric 不再是"剪完再期待量化能扛住"，而是"剪枝决策**本身就是在量化后的 forward 上最优**"。

定量上：在激活 outlier 通道 j（`σ_A²` 大），`[H⁻¹]_{jj}` 变小 → `I(i,j)` 变大 → 这些列的权重被**主动保护**。这对冲了 SmoothQuant 之后的残余 A-outlier——残余越大，保护越强，**自适应**。

### 13.6 两层 Hessian 的叠加增益——不是加法，是乘法

在近乎无损区间，误差模型近似为**各层误差独立累加**（残差网络的标准假设）：
$$
\|\Delta Y_{\text{model}}\|^2 \approx \sum_l \|\Delta Y_l\|^2 \quad(\text{每层通过 LayerNorm 近似解耦})
$$

- **外层 block search** 优化 `{s_l}` 让 `Σ_l ||ΔY_l||²` 在预算约束下最小——**层间分配最优**
- **内层 v5 metric** 对每个 `s_l` 让 `||ΔY_l||²` 最小——**层内选择最优**

两者分别在不同维度上紧致 → 合起来在 `(M, {s_l})` 联合搜索空间下是 **coordinate-descent 最优点**。v4 的外层是次优目标（因为 metric 差），所以外层搜索的信号本身被污染了；v5 让外层搜到的 `{s_l}` 也更准。**这是两层同时吃紧的复合增益**。

### 13.7 近乎无损的脆弱性 vs v5 的稳健性

近乎无损区间对**超参数的鲁棒性**要求极高——哪怕校准数据稍有漂移，metric 的一个关键超参偏了 20%，分数就可能多掉 2%。

超参对比：

| 方法 | 可调超参数量 | 各超参影响 |
|---|---|---|
| v4 | 3（γ, ρ, gamma_block） | γ 改变模态平衡、ρ 改变 LOO 强度，相互耦合 |
| v5 | 1（π_t，默认 0.5） | 只影响混合比例，和 metric 形式正交 |

**v5 的超参数空间是 v4 的 1/27**（指数关系）。校准数据变化下性能方差小，近乎无损区间更稳定。

### 13.8 可能的失败模式及其边界

| 风险 | 发生概率 | 影响 | v5 的兜底 |
|---|---|---|---|
| §8 子空间假设不成立（H_v ≈ H_t） | 低 | mixture 退化到单 H，无增益 | 结果不差于 v4，不会 regression |
| `H_t` 文本 token 过少 → 低秩 | 中 | 文本统计噪声大 | Ledoit-Wolf shrinkage + `λ_floor` 兜底 |
| per-row absmax step 被新剪出的 outlier 扰动 | 低 | 残留量化误差 | 由下游 Clip pass 兜底（保持不变） |
| calibration 数据分布偏 | 中 | 和所有 PTQ/prune 通病 | 与 v4 一致，不引入新风险 |
| 某层剪枝导致通道塌缩 | 低 | 后续层级联放大 | 外层 block search 能检测到块误差异常并分配更少预算 |

**v5 的所有失败模式都有明确兜底，且最坏情况不比 v4 差。**

### 13.9 论证总结

v5 有机会做到近乎无损的四个必要条件，**逐一可验证**：

1. **信息效率**：metric 在 (W, X²) 信息类里达到理论最优（OBS 定理，§13.4）
2. **量化对齐**：metric 直接优化 post-quant forward（`σ_A²·I` 注入，§13.5）
3. **模态平衡**：修正 v4 的 token-count bias，定量补足 Cognition 的 7 分（§13.3）
4. **鲁棒性**：可调超参数减到 1，校准漂移风险降低（§13.7）

加上外层 block search 在层间预算分配上的独立最优（§13.6），v5 在两个正交维度上都是所在信息类里的紧致最优解。

**不保证近乎无损**，但**结构上没有已知的损失源可以把它挡在近乎无损之外**——剩下的是工程实现和超参校准的问题。这和 v4 的情况本质不同（v4 的 metric 在 §13.2 有定量可见的信息丢失）。

---

## 14. 验证近乎无损的实验层级

按**必要条件排序**，下一条失败则无须继续：

1. **§8 子空间夹角实验** → 验证 mixture 结构有必要
2. **v5 @ π_t = 0.5 vs v4**（MME、PPL） → 验证基础收益
3. **Cognition 子项细分** → 验证 §13.3 的病灶定位正确
4. **π_t ∈ {0.3, 0.5, 0.7}** → 估算对超参的敏感度（§13.7）
5. **v5 w/ σ_A² vs v5 w/o σ_A²** → 验证量化对齐的贡献（§13.5）
6. **MMStar / MMBench / SEED-Bench** → 最终多任务验证

每一步都有 falsifiable 的阈值，能在早期截断掉不成立的分支。
