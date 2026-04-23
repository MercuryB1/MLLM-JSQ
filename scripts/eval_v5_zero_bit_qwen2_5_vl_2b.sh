#!/usr/bin/env bash
# JSQ v5 zero-bit layer allocation eval on Qwen2.5-VL-2B-Instruct
# Default: sequential single-GPU runs for safety.
# Optional: RUN_MODE=parallel with separate GPU ids.
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-VL-2B-Instruct"}
DATA_DIR=${DATA_DIR:-"storage/datasets"}
CALIB_DATASET=${CALIB_DATASET:-"gqa"}
TASKS=${TASKS:-"mmstar,mme"}
LOG_DIR=${LOG_DIR:-"logs/qwen2_5_vl_2b_zero_bit"}

SPARSITY_RATIO=${SPARSITY_RATIO:-"0.4375"}
PI_T=${PI_T:-"0.3"}
NSAMPLES=${NSAMPLES:-128}
CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
SEED=${SEED:-42}

RUN_MODE=${RUN_MODE:-"sequential"}   # sequential | parallel
GPU_BASELINE=${GPU_BASELINE:-0}
GPU_D0=${GPU_D0:-0}
GPU_JOINT=${GPU_JOINT:-0}

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
  --pi_t "${PI_T}"
  --w_bits 8
  --a_bits 8
  --tasks "${TASKS}"
  --batch_size "${EVAL_BATCH_SIZE}"
)

run_exp() {
  local name="$1"
  local gpu="$2"
  shift 2

  echo "[run] ${name} (gpu=${gpu})"
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py "${COMMON_ARGS[@]}" "$@" \
    > "${LOG_DIR}/${name}.log" 2>&1
}

if [[ "${RUN_MODE}" == "parallel" ]]; then
  run_exp "v5_Z0_uniform" "${GPU_BASELINE}" \
    --layer_alloc_method uniform &
  pid0=$!

  run_exp "v5_Z1_zero_bit_d0" "${GPU_D0}" \
    --layer_alloc_method zero_bit_d0 &
  pid1=$!

  run_exp "v5_Z2_zero_bit_joint" "${GPU_JOINT}" \
    --layer_alloc_method zero_bit_joint &
  pid2=$!

  wait "${pid0}" "${pid1}" "${pid2}"
else
  run_exp "v5_Z0_uniform" "${GPU_BASELINE}" \
    --layer_alloc_method uniform

  run_exp "v5_Z1_zero_bit_d0" "${GPU_D0}" \
    --layer_alloc_method zero_bit_d0

  run_exp "v5_Z2_zero_bit_joint" "${GPU_JOINT}" \
    --layer_alloc_method zero_bit_joint
fi

echo "All zero-bit evaluation runs finished."
echo ""
echo "=== Results ==="
for f in "${LOG_DIR}"/v5_Z*.log; do
  echo "--- $(basename "${f}") ---"
  grep -iE "zero-bit alloc|Overall LLM sparsity|mme.*score|mmstar|average|logical reasoning|math|science & technology" "${f}" | tail -20 \
    || echo "(no results found)"
  echo ""
done
