# GPU-day training stack check/install ON THE POD, any provider (a gpupulse setup step: config "setup_steps": [{"name": "gpuday_stack",
# "script_file": "scripts/gpuday/pod_setup.sh", "timeout_s": 1800}]). Runs in remote_dir after the runner has verified the models.
#
# Works on any SSH-able Linux NVIDIA image with Python 3.10+ and pip:
#   preferred  an image that already has torch + Unsloth (Vast "Unsloth Studio" vastai/unsloth-studio:<date>-cuda-12.9-py312)  -> ~1-2 min
#   fine       a plain PyTorch image (Vast "PyTorch" vastai/pytorch:cuda-12.8.1-auto, RunPod "PyTorch" runpod/pytorch:<v>-cu1290-torch2xx-
#              ubuntu2204) -> torch kept when it already sees the GPU with a matching CUDA, then Unsloth/TRL/PEFT pip-installed (~3-6 min)
#   slowest    a bare CUDA image (Vast "Llama.cpp" vastai/llama-cpp:<b>-cuda-12.9): torch is installed from the PyTorch index that matches
#              the DRIVER (cu128 if the driver supports CUDA >= 12.8 - required for Blackwell sm_120 cards like the RTX 5090 - else cu126 /
#              cu124), then Unsloth (~5-15 min)
# Then: the llama.cpp SOURCE at the runner's tag (convert_hf_to_gguf.py / convert_lora_to_gguf.py + gguf-py) and llama-quantize (from the
# runner's prebuilt release, else a CPU build of that one tool). Every phase is timed (@@t_<phase>=s); the card is reported (@@gpu=...) and
# written to gpuday/gpu.json (name, MiB, compute capability, driver CUDA) - day_jobs takes it as config 'gpuday_gpu'.
set -u
T0=$(date +%s)
export PIP_NO_CACHE_DIR=1                      # small pod disks (32 GB): no pip wheel cache
mkdir -p gpuday/runs gpuday/data
# The Python every GPU-day job uses (written to gpuday/python; the jobs read it): $GPUDAY_PY, else the image's venv (/venv/main on Vast
# images), else gpuday/venv, else the system python3 - but never a PEP 668 externally-managed system Python (Ubuntu 24): then a venv is made
# with --system-site-packages (never pip --break-system-packages).
managed() { "$1" -c "import os, sys, sysconfig; sys.exit(0 if sys.prefix == sys.base_prefix and os.path.exists(os.path.join(sysconfig.get_paths()['stdlib'], 'EXTERNALLY-MANAGED')) else 1)" 2>/dev/null; }
PY=""
for c in "${GPUDAY_PY:-}" /venv/main/bin/python "$PWD/gpuday/venv/bin/python"; do
  if [ -n "$c" ] && [ -x "$c" ]; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  SYS=$(command -v python3 || command -v python)
  if managed "$SYS"; then
    "$SYS" -m venv --system-site-packages gpuday/venv || { echo "system Python is externally managed and 'python3 -m venv' failed (apt install python3-venv)"; exit 3; }
    PY="$PWD/gpuday/venv/bin/python"
  else
    PY="$SYS"
  fi
fi
echo "$PY" > gpuday/python
echo "@@python=$PY"
# llama.cpp source (convert scripts + gguf-py) at the SAME build as the pod's llama-server when it can be read (a newer converter can write
# GGUF metadata an older server does not know); $GPUDAY_LLAMA_TAG overrides; the runner's fallback tag last.
SRV_VER="$("$(cat run/exe 2>/dev/null || echo llama-server)" --version 2>&1 | grep -o 'version: [0-9]*' | head -1 | awk '{print $2}')"
TAG="${GPUDAY_LLAMA_TAG:-${SRV_VER:+b$SRV_VER}}"
TAG="${TAG:-b11379}"
echo "@@llama_tag=$TAG"
GPU="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1 | awk '{print $3}')"
CC="$(echo "$GPU" | awk -F', ' '{print $3}')"
echo "@@gpu=$GPU"; echo "@@driver_cuda=$DRV_CUDA"
printf '{"gpu": "%s", "driver_cuda": "%s"}\n' "$GPU" "$DRV_CUDA" > gpuday/gpu.json
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }      # ver_ge A B: A >= B
if ver_ge "${CC:-0}" "12.0" && ! ver_ge "${DRV_CUDA:-0}" "12.8"; then
  echo "Blackwell card (cc $CC) needs a driver with CUDA >= 12.8 (have $DRV_CUDA): rent another host"; exit 2
fi
torch_ok() { "$PY" -c "import torch; x=torch.ones(64,64,device='cuda'); assert float((x@x).sum())==64**3" >/dev/null 2>&1; }   # runs a kernel: catches wheels without this arch
t=$(date +%s)
if ! torch_ok; then
  if ver_ge "${DRV_CUDA:-0}" "13.0"; then IDXS="cu130 cu128"; elif ver_ge "${DRV_CUDA:-0}" "12.8"; then IDXS=cu128; elif ver_ge "${DRV_CUDA:-0}" "12.6"; then IDXS=cu126; else IDXS=cu124; fi
  for IDX in $IDXS; do
    echo "@@torch_index=$IDX"
    "$PY" -m pip install -q --upgrade torch --index-url "https://download.pytorch.org/whl/$IDX" >/dev/null 2>&1 && torch_ok && break
  done
  torch_ok || { echo "torch from $IDXS does not run on this card"; exit 3; }
fi
echo "@@t_torch=$(( $(date +%s) - t ))"
t=$(date +%s)
if ! "$PY" -c "import unsloth, trl, peft, datasets" >/dev/null 2>&1; then
  echo "@@unsloth_preinstalled=0"
  "$PY" -m pip install -q unsloth unsloth_zoo trl peft datasets >/dev/null 2>&1 || exit 4
  torch_ok || { echo "installing Unsloth replaced torch with one that does not run on this card"; exit 4; }
else
  echo "@@unsloth_preinstalled=1"
fi
echo "@@t_unsloth=$(( $(date +%s) - t ))"
"$PY" - <<'PYEOF'
import torch, trl, peft, transformers
cc = torch.cuda.get_device_capability(0)
print("@@versions=torch %s cuda %s trl %s peft %s transformers %s gpu %s cc %d.%d bf16 %s" % (torch.__version__, torch.version.cuda,
      trl.__version__, peft.__version__, transformers.__version__, torch.cuda.get_device_name(0), cc[0], cc[1], torch.cuda.is_bf16_supported()))
PYEOF
t=$(date +%s)
if [ "$(cat gpuday/llama_tag 2>/dev/null)" != "$TAG" ] || [ ! -f gpuday/llama.cpp/convert_hf_to_gguf.py ]; then
  rm -rf gpuday/llama.cpp
  git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp gpuday/llama.cpp || exit 5
  echo "$TAG" > gpuday/llama_tag
fi
"$PY" -m pip install -q ./gpuday/llama.cpp/gguf-py >/dev/null 2>&1 || "$PY" -m pip install -q gguf >/dev/null 2>&1 || exit 5
echo "@@t_llama_src=$(( $(date +%s) - t ))"
Q=""
for c in "$(dirname "$(cat run/exe 2>/dev/null)" 2>/dev/null)/llama-quantize" "$(command -v llama-quantize 2>/dev/null)" "$(find llama gpuday /opt/llama.cpp -name llama-quantize -type f 2>/dev/null | head -1)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then Q="$c"; break; fi
done
if [ -z "$Q" ] && command -v cmake >/dev/null 2>&1; then         # quantizing needs no CUDA: a CPU build of the one tool (~1-3 min)
  t=$(date +%s)
  cmake -S gpuday/llama.cpp -B gpuday/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
    && cmake --build gpuday/llama.cpp/build --target llama-quantize -j "$(nproc)" >/dev/null 2>&1 \
    && Q="$(find gpuday/llama.cpp/build -name llama-quantize -type f | head -1)"
  echo "@@t_build_quantize=$(( $(date +%s) - t ))"
fi
echo "$Q" > gpuday/quantize_path
echo "@@quantize=$Q"
df -Pk . | awk 'NR==2 {printf "@@disk_free_gb=%.1f\n", $4/1048576}'
echo "@@t_total=$(( $(date +%s) - T0 ))"
[ -n "$Q" ] || { echo "llama-quantize not found and could not be built (install cmake + build-essential)"; exit 6; }
