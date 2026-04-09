#!/bin/bash
# JSQ v3 alpha/rho sweep (beta=0, quant_damage disabled)
# GQA calibration + MMStar + MME evaluation
# 8 experiments on GPU 0-7 in parallel

set -euo pipefail

COMMON="--model Qwen/Qwen2-VL-7B-Instruct --calib_dataset gqa --sparsity_ratio 0.4375 --w_bits 8 --a_bits 8 --search_method none --tasks mmstar,mme --beta 0 --pruning_method jsq_v3"
mkdir -p logs

# --- Alpha sweep (rho=0, pure density effect) ---
CUDA_VISIBLE_DEVICES=0 python main.py $COMMON --alpha 0.3 --rho 0 \
  > logs/sweep_a0.3_r0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main.py $COMMON --alpha 0.5 --rho 0 \
  > logs/sweep_a0.5_r0.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main.py $COMMON --alpha 1.0 --rho 0 \
  > logs/sweep_a1.0_r0.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main.py $COMMON --alpha 1.5 --rho 0 \
  > logs/sweep_a1.5_r0.log 2>&1 &

# --- Rho sweep (alpha=0.5, density+sensitivity) ---
CUDA_VISIBLE_DEVICES=4 python main.py $COMMON --alpha 0.5 --rho 0.5 \
  > logs/sweep_a0.5_r0.5.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python main.py $COMMON --alpha 0.5 --rho 1.0 \
  > logs/sweep_a0.5_r1.0.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python main.py $COMMON --alpha 0.5 --rho 2.0 \
  > logs/sweep_a0.5_r2.0.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python main.py $COMMON --alpha 0.5 --rho 3.0 \
  > logs/sweep_a0.5_r3.0.log 2>&1 &

wait
echo "All sweep experiments finished."
grep -i "mmstar\|mme\|accuracy\|score" logs/sweep_*.log
