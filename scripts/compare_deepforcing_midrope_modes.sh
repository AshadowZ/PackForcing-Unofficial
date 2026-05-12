#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${ENV_PATH:-/beijing-c/workspace/hxj/miniconda3/envs/packforcing}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/deepforcing_chunkwise_sinkmid.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT/ckpt/causal_forcing_ckpt/causal_forcing.pt}"
PROMPT_PATH="${PROMPT_PATH:-$ROOT/output/deepforcing_compare_60s/prompts/sample1.txt}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-240}"
SEED="${SEED:-0}"
GPU_NONE="${GPU_NONE:-0}"
GPU_SEGMENT="${GPU_SEGMENT:-1}"
GPU_TOKEN="${GPU_TOKEN:-2}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ROOT/output/deepforcing_midrope_compare_${RUN_TAG}}"
PROMPT_LABEL="${PROMPT_LABEL:-$(basename "${PROMPT_PATH%.*}")}"

mkdir -p \
  "$OUT_DIR/logs" \
  "$OUT_DIR/raw/none" \
  "$OUT_DIR/raw/segment" \
  "$OUT_DIR/raw/token" \
  "$OUT_DIR/comparisons"

run_variant() {
  local mode="$1"
  local gpu="$2"
  local out_subdir="$3"
  shift 3
  CUDA_VISIBLE_DEVICES="$gpu" \
    conda run --no-capture-output -p "$ENV_PATH" \
    python -u inference_deepforcing.py \
      --config_path "$CONFIG_PATH" \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --data_path "$PROMPT_PATH" \
      --output_folder "$out_subdir" \
      --num_output_frames "$NUM_OUTPUT_FRAMES" \
      --seed "$SEED" \
      "$@" > "$OUT_DIR/logs/${mode}.log" 2>&1 &
  RUN_PID=$!
}

cd "$ROOT"

run_variant none "$GPU_NONE" "$OUT_DIR/raw/none"
pid_none="$RUN_PID"
run_variant segment "$GPU_SEGMENT" "$OUT_DIR/raw/segment" --pc_mid_rope_mode segment
pid_segment="$RUN_PID"
run_variant token "$GPU_TOKEN" "$OUT_DIR/raw/token" --pc_mid_rope_mode token
pid_token="$RUN_PID"

wait "$pid_none"
wait "$pid_segment"
wait "$pid_token"

video_none="$(find "$OUT_DIR/raw/none" -maxdepth 1 -type f -name '*.mp4' | sed -n '1p')"
video_segment="$(find "$OUT_DIR/raw/segment" -maxdepth 1 -type f -name '*.mp4' | sed -n '1p')"
video_token="$(find "$OUT_DIR/raw/token" -maxdepth 1 -type f -name '*.mp4' | sed -n '1p')"

if [[ -z "$video_none" || -z "$video_segment" || -z "$video_token" ]]; then
  echo "Failed to locate one or more rendered videos under $OUT_DIR/raw" >&2
  exit 1
fi

ffmpeg -y \
  -i "$video_none" \
  -i "$video_segment" \
  -i "$video_token" \
  -filter_complex hstack=inputs=3 \
  -c:v libx264 \
  -crf 18 \
  -preset veryfast \
  -pix_fmt yuv420p \
  "$OUT_DIR/comparisons/${PROMPT_LABEL}_none_vs_segment_vs_token.mp4" \
  > "$OUT_DIR/logs/ffmpeg_compare.log" 2>&1

echo "Output directory: $OUT_DIR"
echo "Comparison video: $OUT_DIR/comparisons/${PROMPT_LABEL}_none_vs_segment_vs_token.mp4"
