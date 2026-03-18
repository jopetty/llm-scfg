#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-/share/apps/images/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
UV_ROOT="${UV_ROOT:-/scratch/$USER/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/scratch/$USER/uv-python}"
VENV_DIR="${VENV_DIR:-/scratch/$USER/venvs/llm-scfg}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

mkdir -p "${UV_CACHE_DIR}" "$(dirname "${VENV_DIR}")"

apptainer exec --nv \
  "${IMAGE}" \
  /bin/bash -lc "
    set -euo pipefail
    export UV_UNMANAGED_INSTALL='${UV_ROOT}'
    export UV_NO_MODIFY_PATH=1
    export UV_PYTHON_INSTALL_DIR='${UV_PYTHON_INSTALL_DIR}'
    export UV_CACHE_DIR='${UV_CACHE_DIR}'
    export UV_LINK_MODE=copy
    export UV_PROJECT_ENVIRONMENT='${VENV_DIR}'
    mkdir -p '${UV_CACHE_DIR}' '$(dirname "${VENV_DIR}")'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH='${UV_ROOT}':\$PATH
    uv python install --install-dir '${UV_PYTHON_INSTALL_DIR}' --force '${PYTHON_VERSION}'
    PYTHON_BIN=\$(find '${UV_PYTHON_INSTALL_DIR}' -path '*/bin/python3.12' | head -n 1)
    if [[ -z \"\${PYTHON_BIN}\" ]]; then
      echo 'Failed to locate installed Python 3.12 in ${UV_PYTHON_INSTALL_DIR}' >&2
      exit 1
    fi
    rm -rf '${VENV_DIR}'
    uv venv --python \"\${PYTHON_BIN}\" '${VENV_DIR}'
    cd '${REPO_ROOT}'
    uv sync --python \"\${PYTHON_BIN}\" --group cluster
    '${VENV_DIR}/bin/python' -c 'import openai, wandb; print(openai.__version__, wandb.__version__)'
    '${VENV_DIR}/bin/vllm' --help >/dev/null
  "
