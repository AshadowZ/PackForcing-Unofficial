#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/beijing-c/workspace/hxj/miniconda3/envs/packforcing/bin/python}"
GPU_FSDP="${GPU_FSDP:-0}"
GPU_NOFSDP="${GPU_NOFSDP:-1}"
PORT_FSDP="${PORT_FSDP:-29680}"
PORT_NOFSDP="${PORT_NOFSDP:-29681}"

FSDP_LOGDIR="${ROOT_DIR}/logs/packforcing_hr_strict_compare_fsdp"
NOFSDP_LOGDIR="${ROOT_DIR}/logs/packforcing_hr_strict_compare_nofsdp"
FSDP_STDOUT="${FSDP_LOGDIR}/stdout.log"
NOFSDP_STDOUT="${NOFSDP_LOGDIR}/stdout.log"

mkdir -p "${FSDP_LOGDIR}" "${NOFSDP_LOGDIR}"

echo "[compare] running FSDP strict compare on GPU ${GPU_FSDP}"
CUDA_VISIBLE_DEVICES="${GPU_FSDP}" PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=1 --master_port="${PORT_FSDP}" \
  train.py \
  --config_path configs/packforcing_dmd_chunkwise_hr_strict_compare_fsdp.yaml \
  --logdir "${FSDP_LOGDIR}" \
  --disable-wandb --no_visualize --no_save | tee "${FSDP_STDOUT}"

echo "[compare] running no-FSDP strict compare on GPU ${GPU_NOFSDP}"
CUDA_VISIBLE_DEVICES="${GPU_NOFSDP}" PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=1 --master_port="${PORT_NOFSDP}" \
  train.py \
  --config_path configs/packforcing_dmd_chunkwise_hr_strict_compare_nofsdp.yaml \
  --logdir "${NOFSDP_LOGDIR}" \
  --disable-wandb --no_visualize --no_save | tee "${NOFSDP_STDOUT}"

echo
echo "[compare] FSDP summary:"
grep "\\[step" "${FSDP_STDOUT}" | tail -n 1 || true
echo "[compare] no-FSDP summary:"
grep "\\[step" "${NOFSDP_STDOUT}" | tail -n 1 || true
