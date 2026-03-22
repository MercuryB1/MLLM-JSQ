#!/usr/bin/env bash
# Evaluate Qwen3-VL-4B-Instruct with lmms-eval.
# Usage: bash scripts/eval_qwen3vl4b.sh [model] [save_dir] [tasks] [log_dir]

export CUDA_VISIBLE_DEVICES=0

set -e

MODEL=${1:-"Qwen/Qwen3-VL-4B-Instruct"}
SAVE_DIR=${2:-"/mnt/disk3/wxj/JSQ4LMM/mllm-jsq/outputs/qwen3vl-4b"}
TASKS=${3:-"gqa,mme,textvqa_val,mmstar,mmmu_val"}
LOG_DIR=${4:-"${SAVE_DIR}/logs"}

python main.py \
    --model "${MODEL}" \
    --save_dir "${SAVE_DIR}" \
    --tasks "${TASKS}" \
    --log_dir "${LOG_DIR}" \
    --eval_only
