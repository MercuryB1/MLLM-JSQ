#!/usr/bin/env bash
# Compress a text-only LLM (LLaMA / Qwen2): JSQ v1, 43.75% sparsity, W8A8

set -e

MODEL=${1:-"Qwen/Qwen2-7B-Instruct"}
SAVE_DIR=${2:-"./outputs/qwen2-7b-jsq-w8a8-s0.4375"}

python main.py \
    --model "${MODEL}" \
    --calib_dataset pileval \
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
    --save_dir "${SAVE_DIR}"
