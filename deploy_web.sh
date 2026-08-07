#!/usr/bin/env bash
# Deploy a trained Keras model to the web frontend (TensorFlow.js format).
#
# Usage: ./deploy_web.sh [model.h5]
#   default model: train/model/model4.h5
#
# Steps: convert h5 -> tfjs layers-model (isolated venv), verify weights are
# bit-identical by name, back up the current web model, then deploy.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-train/model/model4.h5}"
VENV=".venv-tfjs"
PY="$VENV/bin/python"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [ ! -f "$MODEL" ]; then
    echo "error: model not found: $MODEL" >&2
    exit 1
fi

# --- 1. make sure the conversion venv exists ---
if [ ! -x "$PY" ]; then
    echo "creating conversion venv $VENV ..."
    uv venv "$VENV" --python 3.12
fi
if ! "$PY" -c "import tensorflowjs" >/dev/null 2>&1; then
    echo "installing tensorflowjs into $VENV ..."
    uv pip install --python "$VENV" tensorflowjs
fi

OUT=$(mktemp -d /tmp/ai-snake-tfjs.XXXXXX)
trap 'rm -rf "$OUT"' EXIT

# --- 2. convert keras h5 -> tfjs layers model ---
echo "converting $MODEL -> tfjs ..."
"$PY" - "$MODEL" "$OUT" <<'PY'
import sys, types
# Stub optional deps that tensorflowjs imports at package level but this
# conversion never uses.
for name in ("tensorflow_decision_forests", "yggdrasil_decision_forests",
             "tensorflow_hub"):
    sys.modules[name] = types.ModuleType(name)
from tensorflow import keras
from tensorflowjs.converters import keras_h5_conversion as conv

model = keras.models.load_model(sys.argv[1])
conv.save_keras_model(model, sys.argv[2])
print("converted:", len(model.layers), "layers,",
      len(model.get_weights()), "weight tensors")
PY

# --- 3. verify weights are identical by name ---
echo "verifying weights ..."
"$PY" - "$MODEL" "$OUT" <<'PY'
import json, sys
import numpy as np
from tensorflow import keras

model = keras.models.load_model(sys.argv[1])
keras_weights = {
    layer.name + "/" + w.name: w.numpy()
    for layer in model.layers
    for w in layer.weights
    if "kernel" in w.name or "bias" in w.name
}

with open(sys.argv[2] + "/model.json") as f:
    manifest = json.load(f)["weightsManifest"][0]
buf = []
for path in manifest["paths"]:
    buf.append(np.fromfile(sys.argv[2] + "/" + path, dtype=np.float32))
shard = np.concatenate(buf)

off = 0
maxdiff = 0.0
for entry in manifest["weights"]:
    shape = entry["shape"]
    n = int(np.prod(shape))
    got = shard[off:off + n].reshape(shape)
    off += n
    if entry["name"] not in keras_weights:
        raise SystemExit("weight not in keras model: " + entry["name"])
    maxdiff = max(maxdiff,
                  float(np.abs(got - keras_weights[entry["name"]]).max()))

print(f"weights match, max abs diff = {maxdiff:.3e}")
if maxdiff > 1e-5:
    raise SystemExit("weight mismatch, aborting deploy")
PY

# --- 4. back up current web model and deploy ---
BACKUP="src/model_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
if compgen -G "src/model/*" >/dev/null; then
    mv src/model/* "$BACKUP/"
    echo "backed up previous web model to $BACKUP/"
fi
cp "$OUT"/* src/model/

echo "deployed:"
ls -la src/model/
echo
echo "serve with:  python3 -m http.server 8899 --bind 0.0.0.0  (project root)"
