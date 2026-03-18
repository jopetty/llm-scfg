#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
INPUT_PATH="${1:-}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "usage: $0 <input-jsonl-or-batch-dir>" >&2
  exit 1
fi

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="${BASE_URL:-http://$HOST:$PORT/v1}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
CONCURRENCY="${CONCURRENCY:-32}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-600}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

vllm serve "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  $VLLM_EXTRA_ARGS &
SERVER_PID=$!

deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
until curl -fsS "$BASE_URL/models" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM at $BASE_URL" >&2
    exit 1
  fi
  sleep 2
done

if [[ -d "$INPUT_PATH" ]]; then
  uv run python open_weights.py run_batch_dir \
    --batch_dir="$INPUT_PATH" \
    --base_url="$BASE_URL" \
    --api_key=EMPTY \
    --model_override="$SERVED_MODEL_NAME" \
    --concurrency="$CONCURRENCY"
else
  uv run python open_weights.py run_batch_file \
    --input_file="$INPUT_PATH" \
    --base_url="$BASE_URL" \
    --api_key=EMPTY \
    --model_override="$SERVED_MODEL_NAME" \
    --concurrency="$CONCURRENCY"
fi
