#!/bin/bash
# JSQ v5 Option E inverted: high BI → high sparsity (prune sensitive blocks)
# Qwen2-VL-7B-Instruct | GQA multimodal calibration | sparsity=0.4375 | W8A8
# Tasks: mme, mmstar
#
# Hypothesis: BI measures representation change magnitude, NOT weight
# compressibility. Inverting the allocation direction (prune MORE in
# high-BI blocks) may recover signal if high-BI blocks have redundant
# pathways doing large representational work.
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 \
        --w_bits 8 --a_bits 8 --tasks mmstar,mme \
        --pruning_method jsq_v5 --search_method none"

mkdir -p logs

# ================================================================
# F0: uniform baseline (same as E0, for same-GPU-load comparison)
# ================================================================
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method uniform \
  > logs/v5_F0_uniform.log 2>&1 &

# ================================================================
# F1: BI mixture inverted, mild spread (α=0.5)
# ================================================================
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_mixture --block_alloc_alpha 0.5 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  --block_alloc_invert \
  > logs/v5_F1_bimix_inv_a0.5.log 2>&1 &

# ================================================================
# F2: BI mixture inverted, default spread (α=1.0)
# ================================================================
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_mixture --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  --block_alloc_invert \
  > logs/v5_F2_bimix_inv_a1.0.log 2>&1 &

# ================================================================
# F3: BI mixture inverted, aggressive spread (α=2.0)
# ================================================================
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_mixture --block_alloc_alpha 2.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  --block_alloc_invert \
  > logs/v5_F3_bimix_inv_a2.0.log 2>&1 &

# ================================================================
# F4: BI vision inverted (the worst E-sweep config, invert it)
# ================================================================
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_vision --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  --block_alloc_invert \
  > logs/v5_F4_bivision_inv_a1.0.log 2>&1 &

wait
echo "All F sweep (inverted) experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v5_F*.log; do
  echo "--- $(basename $f) ---"
  grep -iE "mmstar|mme.*score|Overall LLM sparsity|accuracy|Per-block sparsity|target=.*realized" "$f" | tail -15 \
    || echo "(no results found)"
  echo ""
done
