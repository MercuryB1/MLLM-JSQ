#!/usr/bin/env bash
# Compress Qwen2/2.5/3-VL: JSQ v1, 43.75% sparsity, W8A8
# Then evaluate on MMBench and SEED-Bench.

export CUDA_VISIBLE_DEVICES=4

set -e

MODEL=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}
SAVE_DIR=${2:-"./outputs/qwen2.5vl-3b-instruct"}
TASKS=${3:-"mme"}

python main.py \
    --model "${MODEL}" \
    --calib_dataset gqa \
    --nsamples 128 \
    --calib_batch_size 4 \
    --data_dir /mnt/disk3/wzn/datasets \
    --pruning_method jsq_v1 \
    --sparsity_ratio 0.4375 \
    --sparsity_type unstructured \
    --rho 2.1 \
    --w_bits 8 \
    --a_bits 8 \
    --weight_quant per_channel \
    --act_quant per_token \
    --smooth_alpha 0.8 \
    --save_dir "${SAVE_DIR}" \
    --tasks "${TASKS}"
