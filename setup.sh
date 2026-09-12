#!/usr/bin/env bash
# Create a project-local inference environment and expose it through conda activate.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"
conda_exe=${CONDA_EXE:-conda}
prefix="$script_dir/.conda/envs/scenetracker"
export CONDA_PKGS_DIRS="$script_dir/.conda/pkgs"
export PIP_CACHE_DIR="$script_dir/.cache/pip"
export TORCH_EXTENSIONS_DIR="$script_dir/.cache/torch_extensions"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MAX_JOBS=${MAX_JOBS:-4}
# Do not inherit unrelated user pip indexes or cache-disable settings.
export PIP_CONFIG_FILE=/dev/null
if [[ "${1:-}" != "" && "${1:-}" != "--check" ]]; then
    echo 'Usage: bash setup.sh [--check]' >&2
    exit 2
fi
if [[ "${1:-}" != "--check" ]]; then
    if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
        echo 'Set CUDA_HOME to a CUDA 12.x toolkit containing bin/nvcc.' >&2
        exit 1
    fi
    mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
    if [[ ! -f "$prefix/conda-meta/history" ]]; then
        "$conda_exe" env create --prefix "$prefix" --file environment.yml --yes
    fi
    "$prefix/bin/python" -m pip install -r requirements.txt
    base=$("$conda_exe" info --base)
    link="$base/envs/scenetracker"
    mkdir -p "$base/envs"
    if [[ -e "$link" || -L "$link" ]]; then
        if [[ "$(readlink -f "$link")" != "$(readlink -f "$prefix")" ]]; then
            echo "Refusing to replace another environment at $link" >&2
            exit 1
        fi
    else
        ln -s "$prefix" "$link"
    fi
    "$conda_exe" env config vars set --prefix "$prefix" \
        "CUDA_HOME=$CUDA_HOME" "CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS" \
        "PIP_CACHE_DIR=$PIP_CACHE_DIR" "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" \
        "MAX_JOBS=$MAX_JOBS"
    cat > "$prefix/pip.conf" <<EOF
[global]
index-url = https://pypi.org/simple
extra-index-url =
no-cache-dir = false
cache-dir = $PIP_CACHE_DIR
EOF
fi
if [[ ! -x "$prefix/bin/python" ]]; then
    echo 'No local environment; run bash setup.sh first.' >&2
    exit 1
fi
"$prefix/bin/python" - <<'PY'
import torch
from knn_cuda import KNN
from torch_scatter import scatter_mean
from lib.network.scene_tracker import SceneTracker
assert torch.cuda.is_available(), 'A CUDA GPU is required for inference'
x = torch.rand(1, 64, 3, device='cuda')
d, _ = KNN(k=4, transpose_mode=True)(x, x[:, :8].contiguous())
assert d.shape == (1, 8, 4) and torch.isfinite(d).all()
y = scatter_mean(x, torch.zeros_like(x, dtype=torch.long), dim=1)
assert torch.allclose(y, x.mean(1, keepdim=True))
print('CUDA KNN/scatter passed:', torch.__version__, torch.cuda.get_device_name())
PY
"$prefix/bin/python" -m pip check
printf '\nReady: conda activate scenetracker\n'
