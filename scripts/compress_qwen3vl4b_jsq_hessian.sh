#!/usr/bin/env bash
# Compress Qwen3-VL-4B-Instruct with JSQ Hessian config (W8A8, sparsity=0.4375) and run 5 eval tasks.
# Usage: bash scripts/compress_qwen3vl4b_jsq_hessian.sh [model] [save_dir] [tasks] [data_dir]

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

set -e

MODEL=${1:-"Qwen/Qwen3-VL-4B-Instruct"}
SAVE_DIR=${2:-"/mnt/disk3/wxj/JSQ4LMM/mllm-jsq/outputs/qwen3vl-4b-jsq-v1-w8a8-sp04375"}
TASKS=${3:-"gqa,mme,textvqa_val,mmstar,mmmu_val"}
DATA_DIR=${4:-"/mnt/disk3/wxj/JSQ4LMM/mllm-jsq/storage/datasets"}

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
    --tasks "${TASKS}"
