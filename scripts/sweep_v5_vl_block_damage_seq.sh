#!/bin/bash
# JSQ v5 Option I: sequential damage estimation
# Propagates pruned outputs between blocks during damage pre-pass,
# so each block's damage is measured with degraded inputs from
# previous trial-pruned blocks. Captures cascading error effects.
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 \
        --w_bits 8 --a_bits 8 --tasks mmstar,mme \
        --pruning_method jsq_v5 --search_method none"

mkdir -p logs

# ================================================================
# I0: uniform baseline
# ================================================================
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method uniform \
  > logs/v5_I0_uniform.log 2>&1 &

# ================================================================
# I1: sequential damage + log, α=0.5 (best H sweep config, now with seq)
# ================================================================
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 0.5 \
  --block_alloc_s_min 0.25 --block_alloc_s_max 0.60 \
  --block_alloc_log --block_alloc_seq \
  > logs/v5_I1_seq_log_a0.5.log 2>&1 &

# ================================================================
# I2: sequential damage + log, α=1.0
# ================================================================
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.25 --block_alloc_s_max 0.60 \
  --block_alloc_log --block_alloc_seq \
  > logs/v5_I2_seq_log_a1.0.log 2>&1 &

# ================================================================
# I3: sequential damage (no log), α=0.5 — test if seq alone fixes the range issue
# ================================================================
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 0.5 \
  --block_alloc_s_min 0.25 --block_alloc_s_max 0.60 \
  --block_alloc_seq \
  > logs/v5_I3_seq_nolog_a0.5.log 2>&1 &

# ================================================================
# I4: sequential damage + log, α=0.5, tighter range [0.30, 0.55]
# ================================================================
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 0.5 \
  --block_alloc_s_min 0.30 --block_alloc_s_max 0.55 \
  --block_alloc_log --block_alloc_seq \
  > logs/v5_I4_seq_log_a0.5_tight.log 2>&1 &

wait
echo "All I sweep (sequential damage) experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v5_I*.log; do
  echo "--- $(basename $f) ---"
  grep -iE "mmstar|mme.*score|Overall LLM sparsity|target=.*realized|pruning_damage|sequential" "$f" | tail -40 \
    || echo "(no results found)"
  echo ""
done
