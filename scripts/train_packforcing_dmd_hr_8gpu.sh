#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/beijing-c/workspace/hxj/miniconda3/envs/packforcing/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/packforcing_dmd_chunkwise_hr_8gpu_sink3_recent2.yaml}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/logs/packforcing_dmd_chunkwise_hr_8gpu_sink3_recent2}"
MASTER_PORT="${MASTER_PORT:-29553}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
NO_VISUALIZE="${NO_VISUALIZE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${GPU_LIST}" == *","* ]]; then
  NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "${GPU_LIST}")
else
  NPROC_PER_NODE=1
fi

mkdir -p "${LOGDIR}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CMD=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  train.py
  --config_path "${CONFIG_PATH}"
  --logdir "${LOGDIR}"
)

if [[ "${DISABLE_WANDB}" == "1" ]]; then
  CMD+=(--disable-wandb)
fi

if [[ "${NO_VISUALIZE}" == "1" ]]; then
  CMD+=(--no_visualize)
fi

cd "${ROOT_DIR}"

echo "[PackForcing HR DMD] config: ${CONFIG_PATH}"
echo "[PackForcing HR DMD] logdir: ${LOGDIR}"
echo "[PackForcing HR DMD] gpus: ${GPU_LIST}"
echo "[PackForcing HR DMD] nproc_per_node: ${NPROC_PER_NODE}"

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS_ARRAY=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

"${CMD[@]}"
