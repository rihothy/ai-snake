#!/usr/bin/env bash
# Rebuild the C++ game engine module (needs: uv venv with pybind11 installed).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-../.venv/bin/python}"

"$PY" -c "import pybind11" 2>/dev/null || {
    echo "pybind11 not installed in $PY; run: uv pip install --python ../.venv pybind11" >&2
    exit 1
}

SUFFIX="$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"

g++ -O3 -march=native -std=c++17 -pthread -shared -fPIC \
    $("$PY" -m pybind11 --includes) \
    engine.cpp \
    -o "engine$SUFFIX"

echo "built engine$SUFFIX"
