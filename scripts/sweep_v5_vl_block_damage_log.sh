#!/bin/bash
# JSQ v5 Option H: damage allocation with log-transform
# Log-transform compresses 2240x damage range to ~8x, preventing
# extreme over-protection of late blocks that destroyed G sweep.
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 \
        --w_bits 8 --a_bits 8 --tasks mmstar,mme \
        --pruning_method jsq_v5 --search_method none"

mkdir -p logs

# ================================================================
# H0: uniform baseline
# ================================================================
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method uniform \
  > logs/v5_H0_uniform.log 2>&1 &

# ================================================================
# H1: damage + log, α=0.5 (mild spread)
# ================================================================
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 0.5 \
  --block_alloc_s_min 0.25 --block_alloc_s_max 0.60 \
  --block_alloc_log \
  > logs/v5_H1_damage_log_a0.5.log 2>&1 &

# ================================================================
# H2: damage + log, α=1.0
# ================================================================
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.25 --block_alloc_s_max 0.60 \
  --block_alloc_log \
  > logs/v5_H2_damage_log_a1.0.log 2>&1 &

# ================================================================
# H3: damage + log, α=2.0
# ================================================================
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 2.0 \
  --block_alloc_s_min 0.25 --block_alloc_s_max 0.60 \
  --block_alloc_log \
  > logs/v5_H3_damage_log_a2.0.log 2>&1 &

# ================================================================
# H4: damage + log, α=1.0, tighter range [0.30, 0.55]
# ================================================================
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pi_t 0.5 \
  --block_alloc_method damage --block_alloc_alpha 1.0 \
  --block_alloc_s_min 0.30 --block_alloc_s_max 0.55 \
  --block_alloc_log \
  > logs/v5_H4_damage_log_a1.0_tight.log 2>&1 &

wait
echo "All H sweep (damage + log) experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v5_H*.log; do
  echo "--- $(basename $f) ---"
  grep -iE "mmstar|mme.*score|Overall LLM sparsity|target=.*realized|pruning_damage" "$f" | tail -40 \
    || echo "(no results found)"
  echo ""
done
