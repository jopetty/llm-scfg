#!/bin/bash
#SBATCH --job-name=llm-scfg-vllm
#SBATCH --account=torch_pr_287_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Example:
#   sbatch --export=ALL,MODEL_NAME=google/gemma-3-12b-it,INPUT_PATH=batches/wordorder_large_exp scripts/slurm_vllm_eval.sbatch

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8000}"
export MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_NAME}"
export TP_SIZE="${TP_SIZE:-1}"
export CONCURRENCY="${CONCURRENCY:-32}"
export VENV_DIR="${VENV_DIR:-/scratch/$USER/venvs/llm-scfg}"
export PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
export VLLM_BIN="${VLLM_BIN:-$VENV_DIR/bin/vllm}"

# L40S usually fits Gemma 3 12B with TP=1. Larger Gemma 3 variants typically
# need more GPUs or a higher-memory node; override TP_SIZE and gres as needed.
INPUT_PATH="${INPUT_PATH:?Set INPUT_PATH to an inputs_*.jsonl file or a batch dir}"

bash scripts/run_vllm_eval.sh "$INPUT_PATH"
