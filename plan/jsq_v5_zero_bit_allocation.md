# JSQ v5 第二贡献设计：Zero-Bit Rate-Distortion Allocation

**状态**：设计草案，未实现  
**日期**：2026-04-23

---

## 1. 定位

第一贡献保持不变：v5 用同一个 mixture-Hessian metric 统一建模 pruning 与 quantization：

$$
H_l=\pi_t C_{t,l}+\pi_v C_{v,l}+\lambda_l I
$$

第二贡献不再改 metric，也不再做 proxy-based block allocation，而是把 **pruning 视为 0-bit quantization**，在固定 block 稀疏率约束下，用同一个 distortion 为 block 内各 layer 求解最优稀疏率分配。

---

## 2. 动机

之前的 block allocation 路线没有稳定收益，核心问题不是“没有找到更好的 searcher”，而是 **allocation signal 与真实 prune+quant objective 不一致**：

- BI / damage / conflict 都是 proxy
- 这些 proxy 默认只回答“哪里敏感”，却没有回答“剪掉 vs 保留为 W8 哪个更划算”
- uniform `s_l` 又忽略了 layer 间 quantization error 与 pruning error 的真实 tradeoff

如果把一个权重的最终状态只看成两个动作：

- `0`：prune，等价于 `0-bit`
- `q = Q_8(w)`：保留并量化为 W8

那么每层稀疏率不应由外部 heuristic 决定，而应由同一个 mixture-Hessian distortion 直接推出来。

---

## 3. 优化问题

### 3.1 单个权重的两种代价

对 layer `l` 中的权重 `w_{ij}`，定义：

$$
d^0_{ij}=\frac{w_{ij}^2}{[H_l^{-1}]_{jj}}
$$

$$
d^8_{ij}=\frac{(w_{ij}-q_{ij})^2}{[H_l^{-1}]_{jj}}, \qquad q_{ij}=Q_8(w_{ij})
$$

其中：

- `d^0_{ij}` 是把该权重直接置零的二阶代价
- `d^8_{ij}` 是把该权重保留并量化到 W8 的二阶代价

于是保留该权重的联合收益定义为：

$$
u_{ij}=d^0_{ij}-d^8_{ij}
$$

`u_{ij}` 越大，说明“保留为 W8”相对“直接剪掉”越划算。

### 3.2 每层的离散 rate-distortion 曲线

保持当前非结构化 top-k 形式不变：每层每行保留相同个数的权重。

对某一行 `i`，将 `u_{ij}` 按从大到小排序为 `u_{ij_{(1)}}, u_{ij_{(2)}}, ...`。若该行保留 `k` 个权重，则该行最小二阶失真为：

$$
E_{li}(k)=\sum_j d^0_{ij}-\sum_{m=1}^{k}u_{ij_{(m)}}
$$

于是 layer `l` 的失真曲线为：

$$
E_l(k)=\sum_i E_{li}(k)
$$

其中 `k` 是“每行保留多少个权重”，对应 layer sparsity：

$$
s_l = 1-\frac{k_l}{d_{in,l}}
$$

这一步很关键：**我们不直接搜索 `s_l`，而是先由 same-objective 的 per-weight utility 构造每层的离散 rate-distortion 曲线。**

### 3.3 固定 block budget 下的 layer allocation

设 block `b` 的目标 sparsity 为 `s_b`，总保留预算为：

$$
K_b=(1-s_b)\sum_{l\in b} d_{out,l}d_{in,l}
$$

则 block 内最优 layer allocation 写成：

$$
\min_{\{k_l\}} \sum_{l\in b} E_l(k_l)
$$

$$
\text{s.t.}\quad \sum_{l\in b} d_{out,l}k_l = K_b,\qquad 0\le k_l \le d_{in,l}
$$

输出是每层的 `k_l`，从而自然得到每层 `s_l`。  
求解方式可以是离散 DP，也可以是拉格朗日水位法；两者都属于 **objective-consistent allocation**，不是 heuristic search。

---

## 4. 这条设计为什么比已有 allocation 更合理

- **同一个目标**：`d^0` 和 `d^8` 都由同一个 mixture-Hessian 给出，不再依赖 proxy score
- **联合考虑 pruning 与 quantization**：不是 prune-only 的 damage，也不是 quant-only 的 rounding error
- **最终 mask 仍然是非结构化的**：只是在 block 内重新分配 layer 稀疏率，不引入结构化约束
- **`s_l` 是解出来的，不是调出来的**：layer sparsity 是 block-level budget 下的最优解，而不是手工设 uniform 或额外搜索超参

---

## 5. 与 GPTQ / 旧 allocation 的区别

这条方法不是 GPTQ 变体，原因有三点：

- GPTQ 求的是“给定全量保留时的最终量化权重”，这里求的是“固定 block budget 下每层应保留多少权重”
- GPTQ 依赖逐列贪心补偿，这里只用 mixture-Hessian 构造离散 rate-distortion 曲线，再做 layer budget allocation
- 最终输出不是新的 quant solver，而是 **block 内 layer-wise sparsity allocation**

同时，它也不同于之前失败的 block allocation：

- 旧方法：先构造 BI / damage / conflict，再映射成 `s_l`
- 新方法：先定义 prune-vs-W8 的 joint distortion，再由该 distortion 直接推导 `s_l`

---

## 6. 最小实验矩阵

- `A2`：当前 baseline，uniform layer sparsity + v5 metric
- `Z1`：只用 `d^0` 构造曲线，验证“纯 pruning allocation”是否仍然无效
- `Z2`：用 `d^0` 和 `d^8` 的 joint utility 构造曲线，做 block 内最优 allocation

如果 `Z2 > A2` 且 `Z1` 不明显优于 `A2`，说明收益来自“把 pruning 视作 0-bit quantization”的联合优化，而不是普通的 sparsity reallocation。
