#!/usr/bin/env bash
# Qwen3-VL-4B 一键总控：串行执行 baseline -> jsq_v1 -> jsq_v2，并自动汇总结果。
# 用法：
#   bash scripts/run_qwen3vl4b_all.sh [model] [tasks] [exp_root]

set -euo pipefail

# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export CUDA_VISIBLE_DEVICES=2
echo "[GPU] 总控 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 模型未命中本地缓存时，优先通过 HF 镜像下载。
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

MODEL=${1:-"Qwen/Qwen3-VL-4B-Instruct"}
TASKS=${2:-"gqa,mme,textvqa_val,mmstar,mmmu_val"}
EXP_ROOT=${3:-"/mnt/disk3/wxj/JSQ4LMM/mllm-jsq/outputs/qwen3vl4b_all_runs"}

RUN_ID=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${EXP_ROOT}/${RUN_ID}"
LOG_ROOT="${RUN_DIR}/logs"

BASELINE_LOG_DIR="${LOG_ROOT}/baseline"
V1_SAVE_DIR="${RUN_DIR}/models/jsq_v1"
V1_LOG_DIR="${LOG_ROOT}/jsq_v1"
V2_SAVE_DIR="${RUN_DIR}/models/jsq_v2"
V2_LOG_DIR="${LOG_ROOT}/jsq_v2"

mkdir -p "${BASELINE_LOG_DIR}" "${V1_LOG_DIR}" "${V2_LOG_DIR}" "${RUN_DIR}/summary"

echo "[总控] 运行目录: ${RUN_DIR}"
echo "[总控] 任务列表: ${TASKS}"

# 第 1 步：评测 baseline 模型（不压缩，不加载 save_dir）
echo "[总控] 开始 baseline 评测（若未缓存将走 HF 镜像下载）"
python main.py \
  --model "${MODEL}" \
  --tasks "${TASKS}" \
  --no_compress \
  --log_dir "${BASELINE_LOG_DIR}"

# 第 2 步：JSQ v1 压缩 + 评测
echo "[总控] 开始 JSQ v1"
bash scripts/compress_qwen3vl4b_jsq_v1.sh "${MODEL}" "${V1_SAVE_DIR}" "${TASKS}" "${V1_LOG_DIR}"

# 第 3 步：JSQ v2 压缩 + 评测
echo "[总控] 开始 JSQ v2"
bash scripts/compress_qwen3vl4b_jsq_v2.sh "${MODEL}" "${V2_SAVE_DIR}" "${TASKS}" "${V2_LOG_DIR}"

# # 第 4 步：自动汇总
# echo "[总控] 开始自动汇总"
# python scripts/summarize_qwen3_logs.py \
#   --run "baseline:${BASELINE_LOG_DIR}" \
#   --run "jsq_v1:${V1_LOG_DIR}" \
#   --run "jsq_v2:${V2_LOG_DIR}" \
#   --output-md "${RUN_DIR}/summary/summary.md" \
#   --output-csv "${RUN_DIR}/summary/summary.csv"

echo "[总控] 全部完成"
# echo "[总控] Markdown 汇总: ${RUN_DIR}/summary/summary.md"
# echo "[总控] CSV 汇总: ${RUN_DIR}/summary/summary.csv"
