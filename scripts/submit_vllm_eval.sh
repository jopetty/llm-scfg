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

sbatch --export=ALL,INPUT_PATH="${INPUT_PATH}" "$@" scripts/slurm_vllm_eval.sbatch
