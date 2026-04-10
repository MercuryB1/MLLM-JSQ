#!/bin/bash
# JSQ v4 + improved Hessian block search (v2)
# Improvements over v1:
#   - Full input_feat for final application (not lite_feat subsample)
#   - 16 diverse candidates (was 6)
#   - 16 eval samples (was 8)
#   - Sensitivity-proportional + OWL-style candidates
#
# Qwen2-VL-7B-Instruct, GQA calib, sparsity=0.4375, W8A8
set -euo pipefail

MODEL="Qwen/Qwen2-VL-7B-Instruct"
COMMON="--model $MODEL --calib_dataset gqa --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --tasks mmstar,mme --beta 0 --alpha 0"
mkdir -p logs

# ================================================================
# Group A: Direct mode baselines (no search)
# ================================================================

# A0: JSQ v1 (reference)
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 --search_method none \
  > logs/v4h2_A0_v1_direct.log 2>&1 &

# A1: v4 modal-split, gamma=3.0
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 3.0 --search_method none \
  > logs/v4h2_A1_v4_g3_direct.log 2>&1 &

# ================================================================
# Group B: Improved Hessian search (16 candidates, 16 eval samples)
# ================================================================

# B0: JSQ v1 + search (gamma=1.0)
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 \
  --search_method candidate --gamma 1.0 --n_search_candidates 16 \
  > logs/v4h2_B0_v1_search_g1.log 2>&1 &

# B1: JSQ v1 + search (gamma=3.0)
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 \
  --search_method candidate --gamma 3.0 --n_search_candidates 16 \
  > logs/v4h2_B1_v1_search_g3.log 2>&1 &

# B2: v4 + search (gamma=1.0)
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 \
  --search_method candidate --gamma 1.0 --n_search_candidates 16 \
  > logs/v4h2_B2_v4_search_g1.log 2>&1 &

# B3: v4 + search (gamma=3.0) — full pipeline
CUDA_VISIBLE_DEVICES=5 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 \
  --search_method candidate --gamma 3.0 --n_search_candidates 16 \
  > logs/v4h2_B3_v4_search_g3.log 2>&1 &

# B4: v4 + search (gamma=5.0) — aggressive text upweight
CUDA_VISIBLE_DEVICES=6 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 \
  --search_method candidate --gamma 5.0 --n_search_candidates 16 \
  > logs/v4h2_B4_v4_search_g5.log 2>&1 &

# B5: WANDA + search (gamma=1.0) — isolate search contribution
CUDA_VISIBLE_DEVICES=7 python main.py $COMMON \
  --pruning_method wanda --rho 0 \
  --search_method candidate --gamma 1.0 --n_search_candidates 16 \
  > logs/v4h2_B5_wanda_search_g1.log 2>&1 &

wait
echo "All v4+hessian-v2 sweep experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v4h2_*.log; do
  echo "--- $(basename $f) ---"
  grep -i "mmstar\|mme\|accuracy\|score" "$f" || echo "(no results found)"
  echo ""
done
