#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/beijing-c/workspace/hxj/miniconda3/envs/packforcing/bin/python"
CONFIG_PATH="${ROOT_DIR}/configs/packforcing_dmd_chunkwise_8gpu_10step.yaml"
LOGDIR="${ROOT_DIR}/logs/packforcing_dmd_chunkwise_8gpu_10step"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  train.py \
  --config_path "${CONFIG_PATH}" \
  --logdir "${LOGDIR}" \
  --disable-wandb \
  --no_visualize \
  --no_save
