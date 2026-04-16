#!/bin/bash
# JSQ v5 Option E: per-block multimodal sparsity allocation
# Qwen2-VL-7B-Instruct | GQA multimodal calibration | sparsity=0.4375 | W8A8
# Tasks: mme, mmstar
#
# Hypothesis: Allocate per-decoder-block sparsity from modality-split
# Block Influence (BI_t, BI_v); blocks with higher mixed sensitivity
# receive lower sparsity. Symmetric to the element-level mixture-H
# metric (Innovation 1).
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 \
        --w_bits 8 --a_bits 8 --tasks mmstar,mme \
        --pruning_method jsq_v5 --search_method none"

mkdir -p logs

# ================================================================
# E0: uniform baseline (sanity check — must match A0 result)
# ================================================================
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method uniform \
  > logs/v5_E0_uniform_pt0.5.log 2>&1 &

# ================================================================
# E1-E3: BI mixture allocation, sweep alpha
# ================================================================

# E1: BI mixture, mild spread
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_mixture --block_alloc_alpha 0.5 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_E1_bimix_pt0.5_a0.5.log 2>&1 &

# E2: BI mixture, default spread
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_mixture --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_E2_bimix_pt0.5_a1.0.log 2>&1 &

# E3: BI mixture, aggressive spread
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_mixture --block_alloc_alpha 2.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_E3_bimix_pt0.5_a2.0.log 2>&1 &

# ================================================================
# E4-E5: single-modality ablations
# ================================================================

# E4: text-only BI (BI_t)
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_text --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_E4_bitext_a1.0.log 2>&1 &

# E5: vision-only BI (BI_v)
CUDA_VISIBLE_DEVICES=5 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method bi_vision --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_E5_bivision_a1.0.log 2>&1 &

# ================================================================
# E6: text-biased prior + BI mixture (cross-mixed)
# ================================================================
CUDA_VISIBLE_DEVICES=6 python main.py $COMMON \
  --pi_t 0.7 \
  --block_alloc_method bi_mixture --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_E6_bimix_pt0.7_a1.0.log 2>&1 &

wait
echo "All E sweep experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v5_E*.log; do
  echo "--- $(basename $f) ---"
  grep -iE "mmstar|mme.*score|Overall LLM sparsity|accuracy|Per-block sparsity|target=.*realized" "$f" | tail -15 \
    || echo "(no results found)"
  echo ""
done
