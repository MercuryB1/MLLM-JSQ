#!/usr/bin/env bash
# JSQ v5 Innovation 2 — zero-bit rate-distortion layer allocation sweep.
#
# Protocol aligned with plan/jsq_v5_progress_20260416.md (A2 baseline):
#   Model: Qwen2-VL-7B-Instruct
#   Calib: gqa, 128 samples
#   Sparsity: 0.4375 (unstructured)
#   Quant:   W8A8 (per-channel)
#   Tasks:   MME (cog + per), MMStar
#
# Experiment matrix (block-level uniform; layer-level method varies):
#   Z0_uniform_pt03   — A2 reproduction (uniform), reference baseline.
#   Z1_d0_pt03        — allocation only; metric d0 = w^2 / [H^-1]_jj.
#   Z2_joint_pt03     — allocation + per-element use u = d0 - d8 (joint).
#   Z2_joint_pt05     — π_t robustness check (symmetric prior).
#   Z2_joint_pt07     — π_t robustness check (text-biased).
#
# Usage:
#   bash scripts/sweep_v5_zero_bit.sh                         # parallel 5-GPU
#   RUN_MODE=sequential GPU0=0 bash scripts/sweep_v5_zero_bit.sh  # single GPU
#   GPU0=0 GPU1=1 GPU2=2 GPU3=3 GPU4=4 bash scripts/sweep_v5_zero_bit.sh
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2-VL-7B-Instruct"}
DATA_DIR=${DATA_DIR:-"storage/datasets"}
CALIB_DATASET=${CALIB_DATASET:-"gqa"}
TASKS=${TASKS:-"mmstar,mme"}
LOG_DIR=${LOG_DIR:-"logs/v5_zero_bit_sweep"}

NSAMPLES=${NSAMPLES:-128}
CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
SPARSITY_RATIO=${SPARSITY_RATIO:-"0.4375"}
SEED=${SEED:-42}

RUN_MODE=${RUN_MODE:-"parallel"}   # parallel | sequential
GPU0=${GPU0:-0}                    # Z0_uniform_pt03
GPU1=${GPU1:-1}                    # Z1_d0_pt03
GPU2=${GPU2:-2}                    # Z2_joint_pt03
GPU3=${GPU3:-3}                    # Z2_joint_pt05
GPU4=${GPU4:-4}                    # Z2_joint_pt07

mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --model "${MODEL}"
  --data_dir "${DATA_DIR}"
  --calib_dataset "${CALIB_DATASET}"
  --nsamples "${NSAMPLES}"
  --calib_batch_size "${CALIB_BATCH_SIZE}"
  --seed "${SEED}"
  --sparsity_ratio "${SPARSITY_RATIO}"
  --sparsity_type unstructured
  --pruning_method jsq_v5
  --search_method none
  --lambda_floor 1e-2
  --w_bits 8
  --a_bits 8
  --tasks "${TASKS}"
  --batch_size "${EVAL_BATCH_SIZE}"
)

run_exp() {
  local name="$1"
  local gpu="$2"
  shift 2
  echo "[run] ${name} (gpu=${gpu})  log=${LOG_DIR}/${name}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py "${COMMON_ARGS[@]}" "$@" \
    > "${LOG_DIR}/${name}.log" 2>&1
}

launch_all() {
  local mode="$1"
  local bg_op=""
  if [[ "${mode}" == "parallel" ]]; then bg_op="&"; fi

  eval run_exp "v5_Z0_uniform_pt03"   "${GPU0}" \
    --pi_t 0.3 --layer_alloc_method uniform "${bg_op}"

  eval run_exp "v5_Z1_d0_pt03"        "${GPU1}" \
    --pi_t 0.3 --layer_alloc_method zero_bit_d0 "${bg_op}"

  eval run_exp "v5_Z2_joint_pt03"     "${GPU2}" \
    --pi_t 0.3 --layer_alloc_method zero_bit_joint "${bg_op}"

  eval run_exp "v5_Z2_joint_pt05"     "${GPU3}" \
    --pi_t 0.5 --layer_alloc_method zero_bit_joint "${bg_op}"

  eval run_exp "v5_Z2_joint_pt07"     "${GPU4}" \
    --pi_t 0.7 --layer_alloc_method zero_bit_joint "${bg_op}"

  if [[ "${mode}" == "parallel" ]]; then wait; fi
}

launch_all "${RUN_MODE}"

echo ""
echo "All v5 zero-bit sweep runs finished."
echo ""
echo "=== Results ==="
for f in "${LOG_DIR}"/v5_Z*.log; do
  echo "--- $(basename "${f}") ---"
  grep -iE "zero-bit alloc|Overall LLM sparsity|mme.*score|mmstar|average|cognition|perception" "${f}" | tail -15 \
    || echo "(no results found)"
  echo ""
done
