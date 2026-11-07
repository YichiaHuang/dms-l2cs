#!/bin/bash
# One-time environment setup on the training machine (TWCC container or Taiwania-2 login node).
# Idempotent: safe to re-run. Works in two modes:
#   (a) the machine already has a CUDA-enabled torch (e.g. TWCC pytorch image): just installs the rest
#   (b) no torch: creates/uses conda env "l2cs" and installs torch for the CUDA version in $TORCH_INDEX
# Usage: bash scripts/remote/setup_env.sh            (run from anywhere; repo root is derived from this file)
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"   # cu118 for very old drivers, cu124/cu128 for new

have_cuda_torch() { python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if ! have_cuda_torch; then
  echo "== no CUDA torch in current python; using conda env l2cs"
  if ! command -v conda >/dev/null; then
    module load miniconda3 2>/dev/null || { echo "conda not found: install Miniconda to ~/miniconda3 first"; exit 1; }
  fi
  eval "$(conda shell.bash hook)"
  conda env list | grep -q '^l2cs ' || conda create -n l2cs python=3.10 -y -c conda-forge --override-channels
  conda activate l2cs
  have_cuda_torch || pip install torch torchvision --index-url "$TORCH_INDEX"
fi

echo "== python deps"
pip install -r "$ROOT/requirements.txt"
# L2CS-Net package (pulls face_detection from GitHub for RetinaFace; only needed by run_l2cs.py / bench scripts,
# training imports l2cs via sys.path and needs just the deps above). Do not fail the setup if GitHub is unreachable.
pip install -e "$ROOT/L2CS-Net" --no-deps || echo "WARN: pip install -e L2CS-Net failed (ok for training)"
pip install "face_detection@git+https://github.com/elliottzheng/face-detection" || echo "WARN: face_detection not installed (only needed for inference demo)"

echo "== pre-download ImageNet ResNet-50 weights (compute nodes may have no internet)"
python - <<'PY'
import torch
torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", progress=False)
print("resnet50 cached at", torch.hub.get_dir())
PY

echo "== sanity"
python - <<'PY'
import torch, torchvision, numpy, PIL, cv2, scipy
print("torch", torch.__version__, "torchvision", torchvision.__version__, "cuda", torch.version.cuda)
print("gpus:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
try:
    print("native bf16:", torch.cuda.is_bf16_supported(including_emulation=False))
except TypeError:
    print("native bf16 (by cc):", torch.cuda.get_device_capability(0)[0] >= 8)
PY
echo "setup done"
