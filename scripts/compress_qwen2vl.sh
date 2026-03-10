#!/usr/bin/env bash
# Compress Qwen2-VL-7B-Instruct: JSQ v1, 43.75% sparsity, W8A8
# Evaluates on MMBench and SEED-Bench after compression.

set -e

MODEL=${1:-"Qwen/Qwen2-VL-7B-Instruct"}
SAVE_DIR=${2:-"./outputs/qwen2vl-7b-jsq-w8a8-s0.4375"}
TASKS=${3:-"mmbench_en_dev,seedbench"}

python main.py \
    --model "${MODEL}" \
    --calib_dataset coco_captions \
    --nsamples 128 \
    --seqlen 2048 \
    --pruning_method jsq_v1 \
    --sparsity_ratio 0.4375 \
    --sparsity_type unstructured \
    --rho 2.1 \
    --w_bits 8 \
    --a_bits 8 \
    --weight_quant per_channel \
    --act_quant per_token \
    --smooth_alpha 0.8 \
    --eval_ppl \
    --tasks "${TASKS}" \
    --save_dir "${SAVE_DIR}"
