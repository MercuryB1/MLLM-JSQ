#!/bin/bash
# JSQ v5: mixture-Hessian OBS metric sweep
# Qwen2-VL-7B-Instruct | GQA multimodal calibration | sparsity=0.4375 | W8A8
# Tasks: mme, mmstar
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 \
        --w_bits 8 --a_bits 8 --tasks mmstar,mme"

mkdir -p logs

# ================================================================
# Group A: v5 direct (no search) — isolate metric contribution
# ================================================================

# A0: v5 direct, symmetric prior
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pruning_method jsq_v5 --pi_t 0.5 --search_method none \
  > logs/v5_A0_direct_pt0.5.log 2>&1 &

# A1: v5 direct, text-biased prior
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pruning_method jsq_v5 --pi_t 0.7 --search_method none \
  > logs/v5_A1_direct_pt0.7.log 2>&1 &

# A2: v5 direct, vision-biased prior (sanity check)
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pruning_method jsq_v5 --pi_t 0.3 --search_method none \
  > logs/v5_A2_direct_pt0.3.log 2>&1 &

# ================================================================
# Group B: v5 + Hessian block search (full pipeline)
# ================================================================

# B0: v5 + search, symmetric prior
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pruning_method jsq_v5 --pi_t 0.5 \
  --search_method candidate --gamma 1.0 --n_search_candidates 16 \
  > logs/v5_B0_search_pt0.5_g1.log 2>&1 &

# B1: v5 + search, text-biased
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pruning_method jsq_v5 --pi_t 0.7 \
  --search_method candidate --gamma 1.0 --n_search_candidates 16 \
  > logs/v5_B1_search_pt0.7_g1.log 2>&1 &

# B2: v5 + search, text-biased + text-biased block search
CUDA_VISIBLE_DEVICES=5 python main.py $COMMON \
  --pruning_method jsq_v5 --pi_t 0.7 \
  --search_method candidate --gamma 3.0 --n_search_candidates 16 \
  > logs/v5_B2_search_pt0.7_g3.log 2>&1 &

# ================================================================
# Group C: Reference baselines for head-to-head comparison
# ================================================================

# C0: v1 + search (previous best text-only metric + search)
CUDA_VISIBLE_DEVICES=6 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 \
  --search_method candidate --gamma 1.0 --n_search_candidates 16 \
  > logs/v5_C0_v1_search.log 2>&1 &

# C1: v4 + search (previous best multimodal metric + search)
CUDA_VISIBLE_DEVICES=7 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 3.0 \
  --search_method candidate --n_search_candidates 16 \
  > logs/v5_C1_v4_search.log 2>&1 &

wait
echo "All v5 sweep experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v5_*.log; do
  echo "--- $(basename $f) ---"
  grep -iE "mmstar|mme.*score|Overall LLM sparsity|accuracy" "$f" | tail -10 \
    || echo "(no results found)"
  echo ""
done
