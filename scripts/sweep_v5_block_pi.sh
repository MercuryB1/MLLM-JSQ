#!/bin/bash
# JSQ v5 block-wise adaptive pi_t sweep
# Qwen2-VL-7B-Instruct | GQA multimodal calibration | sparsity=0.4375 | W8A8
# Tasks: mme, mmstar
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 \
        --w_bits 8 --a_bits 8 --tasks mmstar,mme \
        --pruning_method jsq_v5 --search_method none"

mkdir -p logs

# BPI0: pure block-wise dominance estimate
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pi_t 0.5 \
  --block_pi_method dominance \
  --block_pi_blend 1.0 \
  > logs/v5_BPI0_block_pi_dominance.log 2>&1 &

# BPI1: conflict-weighted block pi (pure estimate)
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pi_t 0.5 \
  --block_pi_method conflict_weighted \
  --block_pi_blend 1.0 \
  > logs/v5_BPI1_block_pi_conflict.log 2>&1 &

# BPI2: conflict-weighted block pi blended with global prior
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pi_t 0.5 \
  --block_pi_method conflict_weighted \
  --block_pi_blend 0.5 \
  > logs/v5_BPI2_block_pi_conflict_blend0.5.log 2>&1 &

wait
echo "All block-wise pi_t experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v5_BPI*.log; do
  echo "--- $(basename $f) ---"
  grep -iE "block_pi_t|mmstar|mme.*score|Overall LLM sparsity|accuracy" "$f" | tail -20 \
    || echo "(no results found)"
  echo ""
done
