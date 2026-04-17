#!/bin/bash
# JSQ v5 Option G: per-block sparsity from trial-pruning damage
# Qwen2-VL-7B-Instruct | GQA multimodal calibration | sparsity=0.4375 | W8A8
# Tasks: mme, mmstar
#
# Signal: for each decoder block, trial-prune at target sparsity, measure
# MSE(y_clean, y_pruned). High damage → protect (low sparsity).
# This is a direct compressibility signal, unlike BI which is semantic.
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 \
        --w_bits 8 --a_bits 8 --tasks mmstar,mme \
        --pruning_method jsq_v5 --search_method none"

mkdir -p logs

# ================================================================
# G0: uniform baseline (same-GPU-load comparison)
# ================================================================
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method uniform \
  > logs/v5_G0_uniform.log 2>&1 &

# ================================================================
# G1: damage, protect sensitive (high damage → low sparsity), α=0.5
# ================================================================
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 0.5 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_G1_damage_a0.5.log 2>&1 &

# ================================================================
# G2: damage, protect sensitive, α=1.0
# ================================================================
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_G2_damage_a1.0.log 2>&1 &

# ================================================================
# G3: damage, protect sensitive, α=2.0 (aggressive spread)
# ================================================================
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 2.0 \
  --block_alloc_s_min 0.20 --block_alloc_s_max 0.65 \
  > logs/v5_G3_damage_a2.0.log 2>&1 &

# ================================================================
# G4: damage, wider range [0.15, 0.70], α=1.0
# ================================================================
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.15 --block_alloc_s_max 0.70 \
  > logs/v5_G4_damage_a1.0_wide.log 2>&1 &

wait
echo "All G sweep (damage) experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v5_G*.log; do
  echo "--- $(basename $f) ---"
  grep -iE "mmstar|mme.*score|Overall LLM sparsity|accuracy|Per-block sparsity|target=.*realized|pruning_damage" "$f" | tail -40 \
    || echo "(no results found)"
  echo ""
done
