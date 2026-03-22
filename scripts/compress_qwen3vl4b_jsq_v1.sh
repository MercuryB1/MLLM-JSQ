#!/usr/bin/env bash
# Compress Qwen3-VL-4B-Instruct with JSQ v1 (W8A8, sparsity=0.4375) and run 5 eval tasks.
# Usage: bash scripts/compress_qwen3vl4b_jsq_v1.sh [model] [save_dir] [tasks] [log_dir]

export CUDA_VISIBLE_DEVICES=0

set -e

MODEL=${1:-"Qwen/Qwen3-VL-4B-Instruct"}
SAVE_DIR=${2:-"/mnt/disk3/wxj/JSQ4LMM/mllm-jsq/outputs/qwen3vl-4b-jsq-v1-w8a8-sp04375"}
TASKS=${3:-"gqa,mme,textvqa_val,mmstar,mmmu_val"}
LOG_DIR=${4:-"${SAVE_DIR}/logs"}
DATA_DIR=${5:-"/mnt/disk3/wxj/JSQ4LMM/mllm-jsq/storage/datasets"}

python main.py \
    --model "${MODEL}" \
    --calib_dataset gqa \
    --nsamples 128 \
    --calib_batch_size 4 \
    --data_dir "${DATA_DIR}" \
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
    --tasks "${TASKS}" \
    --log_dir "${LOG_DIR}"
