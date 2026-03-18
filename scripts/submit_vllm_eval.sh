#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage: scripts/submit_vllm_eval.sh <input-path> [sbatch args...]

Examples:
  scripts/submit_vllm_eval.sh batches/wordorder_large_exp
  scripts/submit_vllm_eval.sh batches/wordorder_large_exp --export=MODEL_NAME=google/gemma-3-12b-it,TP_SIZE=1
EOF
  exit 1
fi

INPUT_PATH="$1"
shift

mkdir -p logs

SBATCH_ARGS=()
USER_EXPORT=""

for arg in "$@"; do
  if [[ "$arg" == --export=* ]]; then
    USER_EXPORT="${arg#--export=}"
  else
    SBATCH_ARGS+=("$arg")
  fi
done

if [[ -n "$USER_EXPORT" ]]; then
  MERGED_EXPORT="ALL,INPUT_PATH=${INPUT_PATH},${USER_EXPORT}"
else
  MERGED_EXPORT="ALL,INPUT_PATH=${INPUT_PATH}"
fi

sbatch --export="${MERGED_EXPORT}" "${SBATCH_ARGS[@]}" scripts/slurm_vllm_eval.sbatch
