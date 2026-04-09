# JSQ v3 Metric 开发进展

> 日期: 2026-04-09 | 分支: hessian | 模型: Qwen2-VL-7B-Instruct | 校准数据: GQA

---

## 1. Metric 设计

### 背景: WANDA 和 JSQ v1

**WANDA** (基础剪枝指标):
$$\text{metric}(i,j) = |W_{ij}| \cdot S_j$$
其中 $S_j = \sqrt{\sum_t x_{t,j}^2 / N}$ 是每个输入通道的激活幅度，对所有 token 等权求和。

**JSQ v1** 在 WANDA 基础上加了 sensitivity 项:
$$\text{metric}(i,j) = |W_{ij}| \cdot S_j + \rho \cdot \text{sensitivity}(i,j)$$
sensitivity 衡量的是"去掉权重 $(i,j)$ 后输出方差的变化"（leave-one-out），通过 cross-covariance 技巧用单次矩阵乘法实现。

**问题**: 两者都对所有校准 token（视觉、文本、padding）一视同仁。

---

### JSQ v3: 信息密度感知的剪枝指标

#### 动机

多模态模型的校准数据同时包含视觉和文本 token。视觉 token 数量多但空间冗余度高（相邻 patch 相似），文本 token 少但语义密度高。标准 WANDA/JSQ v1 等权处理所有 token，导致剪枝指标被大量冗余视觉 token 的统计量主导。

#### 组件 1: 信息密度感知的激活尺度 (alpha)

用密度加权尺度 $\tilde{S}_j$ 替代等权 WANDA 尺度 $S_j$:

**第一步** — 计算每个 token 的信息密度权重:
$$\omega_t = 1 + \alpha \cdot \max\left(0,\; \frac{\|x_t\| - \mu}{\sigma}\right)$$
其中 $\|x_t\|$ 是 token $t$ 的 L2 范数，$\mu$ 和 $\sigma$ 是所有 token 范数的均值和标准差。

**直觉**: 激活范数高的 token（outlier token）不论是哪种模态，信息量都更大:
- 背景视觉 patch（低范数）→ $\omega_t \approx 1$，和标准 WANDA 一样
- 物体边缘、关键文本 token（高范数 outlier）→ $\omega_t > 1$，被上调权重

**第二步** — 加权的每通道尺度:
$$\tilde{S}_j = \sqrt{\frac{\sum_t \omega_t \cdot x_{t,j}^2}{N_{\text{samples}}}}$$

**特殊情况**: $\alpha = 0$ 时退化为标准 WANDA 尺度。

**开销**: 一次 token-wise L2 范数 + clamp，$O(T \times c_{in})$，可忽略。

#### 组件 2: Leave-One-Out Sensitivity (rho)

继承自 JSQ v1。衡量去掉单个权重后，该层输出的变化:

$$\text{sensitivity}(i,j) = \sqrt{\text{Var}(Y_i) - 2\text{Cov}(Y_i, X_j) W_{ij} + \text{Var}(X_j) W_{ij}^2}$$

通过 cross-covariance 技巧实现为两次 BLAS 矩阵乘法（非逐元素循环）。

**归一化**: v3 中对 sensitivity 做归一化，使其与 base 项量级一致，从而 $\rho$ 控制的是相对贡献比例而非依赖绝对数值:
$$\text{sensitivity\_norm} = \text{sensitivity} \times \frac{\text{mean}(\text{base})}{\text{mean}(\text{sensitivity})}$$

**开销**: $O(T \times c_{in} \times c_{out})$，两次矩阵乘法。token 数上限 4096 以控制显存。

#### 统一公式

$$\text{metric}(i,j) = \underbrace{|W_{ij}| \cdot \tilde{S}_j}_{\text{密度感知的重要性}} + \underbrace{\rho \cdot \text{sensitivity\_norm}(i,j)}_{\text{输出敏感度}}$$

**参数含义**:
- $\alpha$: 密度加权强度。0 = 标准 WANDA，越大越强调信息密集 token 的通道。
- $\rho$: 敏感度权重。0 = 纯密度感知 WANDA，越大越依赖 leave-one-out 输出变化。

**特殊情况**:
- $\alpha=0, \rho=0$ → WANDA
- $\alpha=0, \rho>0$ → 近似 JSQ v1（带归一化的 sensitivity）

---

## 2. 实验结果

### 实验设置

- 模型: Qwen2-VL-7B-Instruct
- 校准: GQA（128 样本，多模态）
- 压缩: sparsity=0.4375（非结构化）+ W8A8 量化
- 搜索: 无（uniform sparsity，无 block search）
- 评测: MMStar, MME (cognition + perception)

### Round 1: 组件消融

| 方法 | alpha | rho | beta | MMStar | MME_cog | MME_perc |
|------|-------|-----|------|--------|---------|----------|
| **JSQ v1 (baseline)** | - | 2.1 | - | 0.5296 | 625.3 | 1617 |
| +density | 0.5 | 0 | 0 | **0.5549** | 615 | 1615 |
| +quant_damage | 0 | 0 | 0.5 | 0.5029 | 571 | 1540 |
| +density+sensitivity | 0.5 | 1.0 | 0 | 0.5438 | 621 | **1631** |
| +density+quant_damage | 0.5 | 0 | 0.5 | 0.4915 | 562.8 | 1548.6 |
| +全部三项 | 0.5 | 1.0 | 0.5 | 0.4767 | 502.8 | 1501.4 |

### Round 2: Alpha/Rho Sweep（beta=0，进行中）

| 实验 | alpha | rho | MMStar | MME_cog | MME_perc |
|------|-------|-----|--------|---------|----------|
| alpha=0.3 | 0.3 | 0 | | | |
| alpha=0.5 | 0.5 | 0 | 0.5549 | 615 | 1615 |
| alpha=1.0 | 1.0 | 0 | | | |
| alpha=1.5 | 1.5 | 0 | | | |
| rho=0.5 | 0.5 | 0.5 | | | |
| rho=1.0 | 0.5 | 1.0 | 0.5438 | 621 | 1631 |
| rho=2.0 | 0.5 | 2.0 | | | |
| rho=3.0 | 0.5 | 3.0 | | | |

---

## 3. 分析

### 有效的部分

- **密度感知尺度 (alpha)**: MMStar 0.5296 → 0.5549（+2.5%）。对信息密集 token 的上调权重有效地保护了重要通道。
- **密度 + 敏感度组合**: MME_perc 1617 → 1631（最佳），MMStar 也优于 baseline。整体最均衡。

### 失败的部分: 量化损伤惩罚 (quant_damage)

所有 beta>0 的配置全线下降，最差时 MMStar 降至 0.4767。

**失败原因**: quant_damage 的设计是"减去一个惩罚项，鼓励剪掉行内的 outlier 权重以缩小量化步长"。但问题是——**行内最大的权重既是量化损伤最大的，也是最重要的**。两者正相关，减法惩罚直接把最重要的权重剪掉了，模型精度崩塌远超量化步长改善带来的收益。

**教训**: 剪枝重要性和量化损伤都与 $|W|$ 正相关，减法机制无法将两者解耦。

### 当前不足

v3 (density + sensitivity) 是更好的**剪枝**指标，但仍未考虑量化。目前 pruning 和 quantization 仅在 pipeline 层面是"联合"的（顺序执行），metric 层面没有联合优化。

---

## 4. 下一步计划

### 近期: 根据 Round 2 sweep 结果确定最优 alpha/rho

### 量化感知剪枝方向（待选）

| 方案 | 思路 | 额外开销 | 风险 |
|------|------|----------|------|
| **A. 量化后权重算指标** | 用 $W_q = \text{round}(W/\text{step}) \times \text{step}$ 代替 $W$ 计算 metric，使重要性反映量化后的真实贡献 | 约 0 | 低 |
| B. 乘性量化调节 | `metric *= (1 + beta * quant_benefit)`，仅在重要性相近的权重间做 tiebreak，不会反转重要性排序 | 约 0 | 中 |
| C. 剪枝+量化联合搜索 | 在 block search 中，每个 candidate 同时做 prune + quantize，用量化后的输出算 reconstruction error | 较高 | 低 |

**推荐优先尝试方案 A**: 改动量最小（一行代码），理论上合理——metric 评估的是权重量化后的实际贡献，而非量化前的原始值。
