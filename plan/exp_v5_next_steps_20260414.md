# v5 改进方向 — 基于 2026-04-14 sweep 分析

**Linked experiment**: `plan/exp_v5_sweep_20260414.md`
**Current best**: A2 (v5 direct, π_t=0.3) — MME_cog 625.00 / MME_per 1640.22 / MMStar 0.5378

---

## 优先级排序（按预期 ROI × 实现成本）

### P0 — 修复 search × v5 失配（F3，一定做）

**假设**：传递 vision_mask 到候选评估后，B 组应≥ A 组对应配置 + 获得 §13.6 两层乘法增益。

**改动**：`jsq/compression/block_search.py`
- 在 `_subsample_feat` 对齐阶段，为 lite_feat 同步构造一个 `lite_vision_mask`（按相同 stride subsample）。
- `_evaluate_candidate(..., vision_mask_for_metric=lite_vision_mask)`。
- 保证 lite_feat 的 token 顺序与 vision_mask 的对齐（`_subsample_feat` 目前用 `f[::step]`，vision_mask 也按同 step subsample 即可）。

**验证**：重跑 B0/B1/B2，预期 MMStar ≥ A 组对应 π_t 的结果；若 ≥ 0.54 则 §13.6 验证通过。

**代码量**：约 30 行，风险低（向后兼容，v1/v4 传 None 不变）。

---

### P1 — σ_A² 贡献分离（plan §13.5 + checklist）

**假设**：§13.5 的量化对齐把 σ_A² 注入 λ 是 v5 的关键；若移除，MME_per 会显著下降。

**改动**：在 `configs` 加 `--disable_sigma_a` flag，prune pass 里跳过 σ_A² 项。

**实验**：A2 w/ σ_A² vs A2 w/o σ_A²，只跑 MME+MMStar。

**目的**：plan checklist line 239；建立 σ_A² 的定量贡献（论文 ablation 必需）。

---

### P2 — π_t 自适应（plan §3.1 option 3）

**现象**：π_t=0.3 (vision-biased) 在 Qwen2-VL 上反而最优，与直觉反的；但 π_t=0.5 也只差 0.7 pp。**提示**：最优 π_t 是**逐层**变化的，全局单值掩盖了异质性。

**改动**：在 collector 阶段对每层估计 `π_t^l ∝ ||grad_t^l · X_t^l||² / (||grad_t^l · X_t^l||² + ||grad_v^l · X_v^l||²)`，需要一次额外反向传播。

**成本**：collector 增加 ~1 次反向；前向方法可用 per-layer hook 绕开 full backward。

**风险**：π_t 估计噪声大会让 v5 退化。先跑一个"layer-wise oracle"：对 4 层做 π_t ∈ {0.1, 0.3, 0.5, 0.7, 0.9} 网格，看是否有显著的 per-layer 偏好，再决定是否实现自适应。

---

### P3 — 子空间夹角实证（plan §8，blocking 了 v5-plus）

**plan §8 是 writing 前置**：若 H_v 与 H_t 主子空间夹角 < 30°，mixture 理论上退化为单 H + token 加权。

**需产出**：`plan/v5_subspace_overlap_measurement.md`。

**成本**：跑 4 个 block × 7 投影层 × top-32 eigenvectors → `scipy.linalg.subspace_angles`。约 1 小时脚本 + 半小时作图。

**用途**：论文 §3 motivation 图 / appendix；结果如 ≥ 30° 为 v5 给出独立的理论合法性。

---

### P4 — MMStar 落后的细分原因

**假设**：MMStar 的 logical/science 子项偏 NLP，gqa calib 主要激活视觉通路 → v5 的 mixture 对 gqa-type 样本最优，对 MMStar-type 不最优。

**实验**：
1. 换用 `textvqa` 或 `llava-instruct` 作 calib，再跑 A 组（π_t=0.3, 0.5）。
2. 分 MMStar 子类对比：coarse/fine/instance vs logical/math/science。

**预期**：textvqa calib 下 A 组 MMStar 应追平 C 组；如仍落后则问题不在 calib。

---

### P5 — Fused prune + quant (plan §12 / future)

**前置**：P0 通过后再做。若 v5 + search 仍未到近乎无损（MME ≥ 98% dense），再考虑 GPTQ-style 逐列循环把 prune/quantize 融合（per-group W8 会让 OBS 补偿存活，见 plan §7.1）。

**这是下一篇论文或 v6 的范畴**；v5 当前阶段先不碰。

---

## 建议执行顺序

```
P0 (~0.5 day, 修 search 耦合)
  ↓ 若 B 组追平/超过 A 组
P1 (~0.5 day, σ_A² ablation，论文必备)
  ↓
P3 (~1 day, 子空间夹角，论文 motivation)
  ↓
P4 (~0.5 day, calib 消融)
  ↓ 若 MMStar 仍落后
P2 (~2 days, 自适应 π_t)
  ↓
P5 (future)
```

P0 是阻断的根因修复，其他均为 parallel-able。
建议先跑 P0，等结果后按日志再决定 P1–P4 的顺序。

---

## Decision points

- **P0 后 B 组仍不如 A 组** → 说明 block search 的 Fisher proxy (`Y_orig²`) 与 v5 metric 不一致，需把外层 loss 换成 v5 期望形式。改 `_hessian_block_error` 的 H 构造。
- **P1 w/o σ_A² 相近** → σ_A² 在 gqa calib 下不显著，章节 §13.5 的 claim 要弱化。
- **P3 夹角 < 30°** → 放弃 mixture，退回 token-weighted single H，v5 名字保留但公式换。

---

## Status: updated 2026-04-15

## 2026-04-15 Addendum — D 组结果倒逼重排

D 组 (v5 + analytic water-fill) 相比 A direct **全面崩溃**（MME −380~−470，MMStar −9~−15 pp）。
即把外层 loss 与 v5 内层完全对齐，效果依然劣于 uniform。说明：

- F6 "Fisher/input-space 流形失配" 假设**被反驳**。
- 真正原因：OBS additive 假设在 s_l > 0.5 时失效；跨层 water-fill 会把某些层推到 s_max=0.9 → 该层崩盘 → 整块崩盘。
- **v5 的唯一真正贡献是 per-element mask**；跨层 budget 不应做非均匀优化（或只做 ±0.03 扰动）。

### 新优先级

| 优先级 | 任务 | 状态 |
|---|---|---|
| P0 (新) | **σ_A² ablation** — A2 w/ 和 w/o σ_A² 对比，论文必备 | 未开始 |
| P0b (新) | **π_t 自适应** — gradient ratio per-layer，看是否进一步拉开 A2 | 未开始 |
| P1 (新) | **子空间夹角实证** — plan §8 writing 前置 | 未开始 |
| P2 (降级) | Search 机制研究（仅 B_fix 路径，Fisher proxy ±0.03）— 论文 §5 |  已跑 |
| **废弃** | Analytic water-fill / full-range search | 论文中不提，仅作为负例 ablation |

下一步（按顺序）：P0 σ_A² ablation → P0b π_t 自适应 → P1 子空间夹角。
