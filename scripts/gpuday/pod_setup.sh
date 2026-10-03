# GPU-day training stack check/install ON THE POD (a gpupulse setup step: config "setup_steps": [{"name": "gpuday_stack",
# "script_file": "scripts/gpuday/pod_setup.sh", "timeout_s": 1800}]). Runs in remote_dir after the runner has verified the models.
# Rented image (owner checklist): vastai/unsloth-studio:2026.9.14-cuda-12.9-py312 - torch + Unsloth + TRL are already there, so this
# step only (1) proves it, (2) fetches the llama.cpp SOURCE at the runner's tag for convert_hf_to_gguf.py / convert_lora_to_gguf.py and
# pip-installs its gguf-py, (3) finds llama-quantize (the runner's prebuilt ggml-org CUDA release, run/exe's directory). Each phase is
# timed (@@t_<phase>=seconds). On any other image it installs Unsloth with pip (slow path, timed, 5-15 min) - not the plan.
set -u
T0=$(date +%s)
PY=$(command -v python3 || command -v python)
TAG="${GPUDAY_LLAMA_TAG:-b11379}"
mkdir -p gpuday/runs gpuday/data
if ! "$PY" -c "import unsloth, trl, peft, torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "@@unsloth_preinstalled=0"
  t=$(date +%s); "$PY" -m pip install -q --upgrade unsloth unsloth_zoo trl peft datasets >/dev/null 2>&1 || exit 3
  echo "@@t_pip_unsloth=$(( $(date +%s) - t ))"
else
  echo "@@unsloth_preinstalled=1"
fi
"$PY" - <<'PYEOF'
import torch, trl, peft, transformers
print("@@versions=torch %s cuda %s trl %s peft %s transformers %s gpu %s" % (torch.__version__, torch.version.cuda, trl.__version__,
      peft.__version__, transformers.__version__, torch.cuda.get_device_name(0)))
PYEOF
t=$(date +%s)
if [ ! -f gpuday/llama.cpp/convert_hf_to_gguf.py ]; then
  rm -rf gpuday/llama.cpp
  git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp gpuday/llama.cpp || exit 4
fi
"$PY" -m pip install -q ./gpuday/llama.cpp/gguf-py >/dev/null 2>&1 || "$PY" -m pip install -q gguf >/dev/null 2>&1 || exit 5
echo "@@t_llama_src=$(( $(date +%s) - t ))"
Q=""
for c in "$(dirname "$(cat run/exe 2>/dev/null)" 2>/dev/null)/llama-quantize" "$(command -v llama-quantize 2>/dev/null)" "$(find llama gpuday -name llama-quantize -type f 2>/dev/null | head -1)"; do
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
[ -n "$Q" ] || { echo "llama-quantize not found (the runner's llama.cpp install must run first)"; exit 6; }
