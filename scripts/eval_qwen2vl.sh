#!/usr/bin/env bash
# Evaluate a saved compressed Qwen2-VL model with lmms-eval.
# Usage: bash scripts/eval_qwen2vl.sh [save_dir] [tasks]

export CUDA_VISIBLE_DEVICES=5

set -e

MODEL=${1:-"Qwen/Qwen2-VL-7B-Instruct"}
SAVE_DIR=${2:-"/mnt/disk3/wzn/mllm-jsq/outputs/qwen2vl-7b"}
TASKS=${3:-"mme"}

python main.py \
    --model "${MODEL}" \
    --save_dir "${SAVE_DIR}" \
    --tasks "${TASKS}" \
    --eval_only
