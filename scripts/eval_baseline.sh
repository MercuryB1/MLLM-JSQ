#!/bin/bash
# Evaluate original (uncompressed) models on multimodal benchmarks
# Uses --no_compress to skip all compression passes

TASKS="mmstar,mme"

# Qwen2-VL-7B
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --no_compress \
  --tasks $TASKS \
  > logs/baseline_qwen2vl_7b.log 2>&1 &

# Qwen2.5-VL-7B
CUDA_VISIBLE_DEVICES=1 python main.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --no_compress \
  --tasks $TASKS \
  > logs/baseline_qwen2.5vl_7b.log 2>&1 &

# Qwen3-VL-8B
CUDA_VISIBLE_DEVICES=2 python main.py \
  --model Qwen/Qwen3-VL-8B \
  --no_compress \
  --tasks $TASKS \
  > logs/baseline_qwen3vl_8b.log 2>&1 &

wait
echo "All baseline evaluations finished."
grep -i "mmstar\|mme\|accuracy\|score" logs/baseline_*.log
