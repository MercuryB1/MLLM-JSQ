#!/bin/bash
# JSQ v4 + Hessian block search combined sweep
# Tests modality-split metric with block-level sparsity allocation
# Qwen2-VL-7B-Instruct, GQA calib, sparsity=0.4375, W8A8
#
# gamma controls BOTH metric modality split AND search error weighting.
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --tasks mmstar,mme --beta 0 --alpha 0"
mkdir -p logs

# ================================================================
# Group A: Direct mode (search=none) — metric-only baselines
# ================================================================

# A0: JSQ v1 baseline (no search)
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 --search_method none \
  > logs/v4h_A0_v1_direct.log 2>&1 &

# A1: v4 modal-split, gamma=3.0 (best from previous sweep)
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 3.0 --search_method none \
  > logs/v4h_A1_v4_g3_direct.log 2>&1 &

# ================================================================
# Group B: Hessian block search — metric + search combined
# ================================================================

# B0: JSQ v1 + search (gamma=1.0, baseline search)
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 \
  --search_method candidate --gamma 1.0 --n_search_candidates 8 \
  > logs/v4h_B0_v1_search_g1.log 2>&1 &

# B1: JSQ v1 + search (gamma=3.0, upweight text error)
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 \
  --search_method candidate --gamma 3.0 --n_search_candidates 8 \
  > logs/v4h_B1_v1_search_g3.log 2>&1 &

# B2: v4 + search (gamma=1.0, modal-split metric + equal-weight search)
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 \
  --search_method candidate --gamma 1.0 --n_search_candidates 8 \
  > logs/v4h_B2_v4_search_g1.log 2>&1 &

# B3: v4 + search (gamma=3.0, full modal-aware: metric + search both upweight text)
CUDA_VISIBLE_DEVICES=5 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 \
  --search_method candidate --gamma 3.0 --n_search_candidates 8 \
  > logs/v4h_B3_v4_search_g3.log 2>&1 &

# B4: WANDA + search (gamma=1.0, isolate search contribution)
CUDA_VISIBLE_DEVICES=6 python main.py $COMMON \
  --pruning_method wanda --rho 0 \
  --search_method candidate --gamma 1.0 --n_search_candidates 8 \
  > logs/v4h_B4_wanda_search_g1.log 2>&1 &

# B5: WANDA + search (gamma=3.0)
CUDA_VISIBLE_DEVICES=7 python main.py $COMMON \
  --pruning_method wanda --rho 0 \
  --search_method candidate --gamma 3.0 --n_search_candidates 8 \
  > logs/v4h_B5_wanda_search_g3.log 2>&1 &

wait
echo "All v4+hessian sweep experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v4h_*.log; do
  echo "--- $(basename $f) ---"
  grep -i "mmstar\|mme\|accuracy\|score" "$f" || echo "(no results found)"
  echo ""
done
