#!/bin/bash
# JSQ v4 ablation sweep: quantization-aware + modality-split metric
# GQA calibration + MMStar + MME evaluation
# Qwen2-VL-7B-Instruct, sparsity=0.4375, W8A8, search=none
set -euo pipefail

COMMON="--model Qwen/Qwen2-VL-7B-Instruct --calib_dataset gqa --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --search_method none --tasks mmstar,mme --beta 0 --alpha 0"
mkdir -p logs

# ================================================================
# Exp 0: JSQ v1 baseline (for fair comparison on same branch)
# ================================================================
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON \
  --pruning_method jsq_v1 --rho 2.1 \
  > logs/v4_exp0_jsq_v1.log 2>&1 &

# ================================================================
# Exp 1: v4 quant-aware only (no modality split, no sensitivity)
# Tests: (|W| - |W - W_q|) * S_mixed
# ================================================================
CUDA_VISIBLE_DEVICES=1 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 0 --gamma 1.0 \
  > logs/v4_exp1_quant_only.log 2>&1 &

# ================================================================
# Exp 2: v4 quant-aware + sensitivity (no modality split)
# Tests: (|W| - |W - W_q|) * S_mixed + rho * ss_mixed
# ================================================================
CUDA_VISIBLE_DEVICES=2 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 1.0 \
  > logs/v4_exp2_quant_sens.log 2>&1 &

# ================================================================
# Exp 3: v4 full (quant-aware + modality split, gamma=1.0)
# Tests: (|W| - |W - W_q|) * (S_vis + S_txt) + rho * (ss_vis + ss_txt)
# ================================================================
CUDA_VISIBLE_DEVICES=3 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 1.0 \
  > logs/v4_exp3_full_g1.0.log 2>&1 &

# ================================================================
# Exp 4: v4 full, gamma=2.0 (upweight text)
# ================================================================
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 2.0 \
  > logs/v4_exp4_full_g2.0.log 2>&1 &

# ================================================================
# Exp 5: v4 full, gamma=3.0 (strongly upweight text)
# ================================================================
CUDA_VISIBLE_DEVICES=5 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 3.0 \
  > logs/v4_exp5_full_g3.0.log 2>&1 &

# ================================================================
# Exp 6: v4 full, gamma=0.5 (downweight text / upweight vision)
# ================================================================
CUDA_VISIBLE_DEVICES=6 python main.py $COMMON \
  --pruning_method jsq_v4 --rho 2.1 --gamma 0.5 \
  > logs/v4_exp6_full_g0.5.log 2>&1 &

# ================================================================
# Exp 7: WANDA baseline (for reference)
# ================================================================
CUDA_VISIBLE_DEVICES=7 python main.py $COMMON \
  --pruning_method wanda --rho 0 \
  > logs/v4_exp7_wanda.log 2>&1 &

wait
echo "All v4 sweep experiments finished."
echo ""
echo "=== Results ==="
for f in logs/v4_exp*.log; do
  echo "--- $(basename $f) ---"
  grep -i "mmstar\|mme\|accuracy\|score" "$f" || echo "(no results found)"
  echo ""
done
