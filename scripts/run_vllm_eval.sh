#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
INPUT_PATH="${1:-}"
DEFAULT_VENV_DIR="/scratch/$USER/venvs/llm-scfg"
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
PYTHON_BIN="${PYTHON_BIN:-}"
VLLM_BIN="${VLLM_BIN:-}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "usage: $0 <input-jsonl-or-batch-dir>" >&2
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -z "$VLLM_BIN" ]]; then
  if [[ -x "$VENV_DIR/bin/vllm" ]]; then
    VLLM_BIN="$VENV_DIR/bin/vllm"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/vllm" ]]; then
    VLLM_BIN="${VIRTUAL_ENV}/bin/vllm"
  else
    VLLM_BIN="$(command -v vllm)"
  fi
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "python executable not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$VLLM_BIN" || ! -x "$VLLM_BIN" ]]; then
  echo "vllm executable not found at $VLLM_BIN" >&2
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
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

VLLM_ARGS=(
  serve "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
)

if [[ -n "$ATTENTION_BACKEND" ]]; then
  VLLM_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
  # Intentionally allow shell-style extra flags for cluster-specific tuning.
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $VLLM_EXTRA_ARGS )
  VLLM_ARGS+=("${EXTRA_ARGS[@]}")
fi

"$VLLM_BIN" "${VLLM_ARGS[@]}" &
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
  "$PYTHON_BIN" open_weights.py run_batch_dir \
    --batch_dir="$INPUT_PATH" \
    --base_url="$BASE_URL" \
    --api_key=EMPTY \
    --model_override="$SERVED_MODEL_NAME" \
    --concurrency="$CONCURRENCY"
else
  "$PYTHON_BIN" open_weights.py run_batch_file \
    --input_file="$INPUT_PATH" \
    --base_url="$BASE_URL" \
    --api_key=EMPTY \
    --model_override="$SERVED_MODEL_NAME" \
    --concurrency="$CONCURRENCY"
fi
