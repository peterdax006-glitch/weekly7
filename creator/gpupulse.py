"""GPU PULSE RUNNER (owner, 3 Oct 2026: rent short GPU 'pulses' on Vast.ai, about $8 in all, and spend every paid minute on THINKING work).

Loaded ON DEMAND only (scripts/gpu_pulse.py, or creator.device when a pulse is switched on); never part of the swarm's eager start load.

The pod is a Linux Docker container (Vast.ai template 'Llama.cpp', image vastai/llama-cpp) reached over SSH. Everything heavy happens THERE at
datacenter speed (llama.cpp is preinstalled; the GGUF files come straight from Hugging Face and are checked against pinned sha256), and
Nupen's own code on this PC talks to the remote llama-servers through an SSH port-forward exactly as it talks to a local one:

  setup      probe the pod, stop the template's own server, make sure a CUDA llama-server exists (image binary; fallback: the official
             ggml-org Linux CUDA 12.8 release; last resort: cmake -DGGML_CUDA=ON), download + verify the models, start the servers.
             Idempotent: a verified model is never fetched twice, a healthy server with the same arguments is left running.
  tunnel     one `ssh -N -L` process; the forwarded ports are written to <runtime>/gpu/tunnel.json. creator.device / creator.generator
             attach to them ONLY when env NUPEN_GPU_PULSE names that file (or device setting 'gpu_pulse' is true): off by default.
  run        jobs (thinkbench / judgment rounds / reasoning epochs) as child processes of this PC with the pulse switched on; their records
             land in Nupen's normal state files carrying 'gpu_pulse' so CPU and GPU timings never mix in throughput statistics.
  teardown   stop the servers and the tunnel, close the pulse in the ledger, print (or, when asked, run) the destroy command.
  budget     elapsed pulse time x the listed $/hr (plus bandwidth when given) in a ledger OUTSIDE the repo; a run stops at the cap
             (default $8 across all pulses) and the pod carries its own dead-man switch (vastai stop/destroy of itself) in case this PC dies.

SECURITY: the repository is public. The config (host, port, key path, optional API key), the ledger, known_hosts and the tunnel file live
under <runtime>/gpu (default ~/creator_runtime/gpu) and every writer here refuses a path inside the repository. NO FILE is uploaded to the
pod: setup sends shell commands only (model URLs and hashes); jobs send model prompts built from public material (the public-repo text
caches, Nupen's public git history and its kernel package records). Thinkbench part d (owner directives from ~/Masterstock) is skipped on a
pulse, and the remote client refuses any prompt that carries a private marker (PRIVATE_MARKERS)."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------------ pinned facts (sources: report of 3 Oct 2026)
# sha256 = the Hugging Face LFS oid (api/models/<repo>/tree/main, read 3 Oct 2026); the local 4B/8B copies were fetched from the Qwen repos
# and verified against the same values (models/dl/q4b.log, q8b.log). <runtime>/models/MODELS.json wins when it lists a file with a sha256.
CATALOG: dict[str, dict[str, Any]] = {
    "Qwen3-1.7B-Q4_K_M.gguf": {"repo": "unsloth/Qwen3-1.7B-GGUF", "bytes": 1107409472,
                               "sha256": "b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897", "kv_mib_per_1k": 112},
    "Qwen3-4B-Q4_K_M.gguf": {"repo": "Qwen/Qwen3-4B-GGUF", "bytes": 2497280256,
                             "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", "kv_mib_per_1k": 144},
    "Qwen3-8B-Q4_K_M.gguf": {"repo": "Qwen/Qwen3-8B-GGUF", "bytes": 5027783488,
                             "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785", "kv_mib_per_1k": 144},
    "Qwen3-14B-Q4_K_M.gguf": {"repo": "Qwen/Qwen3-14B-GGUF", "bytes": 9001752960,
                              "sha256": "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0", "kv_mib_per_1k": 160},
}
# KV cache (f16) per 1024 tokens = 2 (K,V) x layers x kv_heads x head_dim x 2 bytes x 1024: 1.7B 28x8x128, 4B/8B 36x8x128, 14B 40x8x128.
HF_BASE = "https://huggingface.co"
FALLBACK_TAG = "b11379"     # ggml-org/llama.cpp release of 3 Oct 2026 with llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz (runs on any >= 12.8 driver)
FALLBACK_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
FALLBACK_CUDART = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz"
VRAM_GB = 24.0
COMPUTE_BUFFER_GB = 0.8     # llama.cpp CUDA compute buffers + context, per server (generous)
PRIVATE_MARKERS = ("Latest owner directives", "Masterstock", "livesim", "BEGIN OPENSSH PRIVATE KEY", "OPEN_BUTTON_TOKEN")
PRIVATE_PATHS = ("state/livesim", "oldpc", "Masterstock", ".ssh", "secrets", "pulse.json", ".env")

DEFAULTS: dict[str, Any] = {
    "host": "", "port": 22, "user": "root", "key_path": "", "instance_id": "",
    "usd_per_hr": 0.343, "bandwidth_usd_per_tb": 0.0, "budget_usd": 8.0, "reserve_minutes": 3.0,
    "remote_dir": "/workspace/nupen", "models": ["Qwen3-1.7B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf"],
    "slots": "auto", "ctx_per_slot": 8192, "best_model": "Qwen3-8B-Q4_K_M.gguf", "monitor_s": 60.0, "low_util_pct": 70.0, "low_util_abort_minutes": 8, "instances": 1, "remote_port_base": 18100, "local_port_base": 18100,
    "allow_build": False, "deadman": "stop", "hf_base": HF_BASE,
    "inflight_factor": 3,
}


class PulseError(RuntimeError):
    pass


class BudgetExceeded(PulseError):
    pass


# ------------------------------------------------------------------------------------------------ paths and config (never inside the repo)
def gpu_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    from creator import device as DEV
    return DEV.runtime_dir(env) / "gpu"


def outside_repo(p: Path, root: Path = ROOT) -> Path:
    """`p` itself when it lies outside the repository; PulseError otherwise (the repo is PUBLIC: nothing of a pulse is ever written in it)."""
    rp, rr = Path(p).resolve(), Path(root).resolve()
    if rp == rr or rr in rp.parents:
        raise PulseError(f"refusing to write pulse data inside the repository: {rp}")
    return Path(p)


def _write_json(p: Path, obj: Any) -> None:
    outside_repo(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def config_path(env: Optional[Mapping[str, str]] = None) -> Path:
    e = os.environ if env is None else env
    return Path(e["NUPEN_PULSE_CONFIG"]) if e.get("NUPEN_PULSE_CONFIG") else gpu_dir(e) / "pulse.json"


def load_config(path: Optional[Path] = None) -> dict[str, Any]:
    p = path or config_path()
    outside_repo(p)                                      # a config inside the public repo is refused even for reading (it would get committed)
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    cfg["_path"] = str(p)
    return cfg


def catalog(rt: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """CATALOG with MODELS.json entries (repo/file/bytes/sha256) of the runtime dir laid over it."""
    from creator import device as DEV
    out = {k: dict(v) for k, v in CATALOG.items()}
    try:
        mj = json.loads(((rt or DEV.runtime_dir()) / "models" / "MODELS.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        mj = {}
    for name, e in mj.items():
        if isinstance(e, dict) and e.get("sha256") and e.get("repo"):
            out.setdefault(name, {}).update({k: e[k] for k in ("repo", "bytes", "sha256") if k in e})
    return out


_POD_LOCAL: dict[str, dict[str, Any]] = {}            # config 'extra_models' seen by extra_models(): sizes for the VRAM fit of pod-local files


def extra_models_file(cfg: Optional[Mapping[str, Any]] = None, env: Optional[Mapping[str, str]] = None) -> Path:
    """Where jobs register pod-local models mid-run: config 'extra_models_file', else env NUPEN_GPU_EXTRA_MODELS, else
    <runtime>/gpu/extra_models.json. It is laid over the config's 'extra_models' on every read (the runner holds the config in memory for a
    whole run, and pulse.json is never rewritten by a job)."""
    e = os.environ if env is None else env
    v = (cfg or {}).get("extra_models_file") or e.get("NUPEN_GPU_EXTRA_MODELS")
    return Path(str(v)) if v else gpu_dir(e) / "extra_models.json"


def extra_models(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Config 'extra_models' (plus extra_models_file()): {name: {"bytes": int, "sha256": hex}} - model files that are ALREADY ON THE POD under <remote_dir>/models
    (e.g. a GPU-day fine-tune's GGUF, made there by an external job). They are never downloaded; serve() and setup() check the file's size
    and sha256 on the pod first and refuse a mismatch. A malformed entry is an error (an unverifiable model is never served)."""
    out: dict[str, dict[str, Any]] = {}
    entries = dict(cfg.get("extra_models") or {})
    try:                                                  # registered mid-run by a job (creator.gpuday:register_tuned_job), read fresh each call
        entries.update(json.loads(extra_models_file(cfg).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    for name, e in entries.items():
        if not isinstance(e, Mapping):
            raise PulseError(f"extra_models entry {name!r} must be {{\"bytes\": int, \"sha256\": hex}}")
        sha, size = str(e.get("sha256", "")).lower(), int(e.get("bytes", 0) or 0)
        bad_name = "/" in name or "\\" in name or name.startswith(".") or not name.endswith(".gguf")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or size <= 0 or bad_name:
            raise PulseError(f"extra_models entry {name!r} needs a plain *.gguf name, 'bytes' > 0 and a 64-hex 'sha256'")
        out[name] = {"bytes": size, "sha256": sha, "pod_local": True}
        if e.get("lora") or e.get("base"):       # h61: a LoRA adapter served on a base GGUF (llama-server -m <base> --lora <adapter>)
            lora, base = str(e.get("lora") or ""), str(e.get("base") or "")
            if any("/" in x or "\\" in x or x.startswith(".") or not x.endswith(".gguf") for x in (lora, base)):
                raise PulseError(f"extra_models entry {name!r}: 'lora' and 'base' must both be plain *.gguf names (files in models/)")
            out[name].update(lora=lora, base=base)     # bytes / sha256 are the ADAPTER file's (what is verified before serving)
    _POD_LOCAL.update(out)
    for name, e in out.items():                  # the VRAM fit counts the base weights for an adapter entry
        if e.get("base"):
            b = CATALOG.get(e["base"]) or _POD_LOCAL.get(e["base"]) or {"bytes": 0}
            _POD_LOCAL[name] = dict(e, bytes=int(b.get("bytes") or 0) + int(e["bytes"]))
    return out


def pod_model_script(rdir: str, name: str, size: int, sha: str) -> str:
    """Verify a pod-local model (extra_models): present, the recorded size, and the sha256 computed NOW from the file (a stale .ok marker is
    not trusted). A mismatch is refused, and the file is left in place (it is the job's output; the caller decides)."""
    q, f = shlex.quote(rdir), shlex.quote(name)
    return f"""set -u
cd {q}/models 2>/dev/null || {{ echo "@@error=no models directory on the pod"; exit 3; }}
[ -f {f} ] || {{ echo "@@error=pod-local model {name} is not on the pod"; exit 3; }}
sz="$(stat -c %s {f} 2>/dev/null || wc -c < {f} | tr -d ' ')"
[ "$sz" = "{size}" ] || {{ rm -f {f}.ok; echo "@@error=size mismatch: got $sz want {size}"; exit 4; }}
got="$(sha256sum {f} | cut -d' ' -f1)"
[ "$got" = "{sha}" ] || {{ rm -f {f}.ok; echo "@@error=sha256 mismatch: got $got want {sha}"; exit 4; }}
echo "{sha}" > {f}.ok && echo "@@verified=1"
"""


def verify_pod_models(cfg: Mapping[str, Any], models: Sequence[str], sh: "Shell") -> list[str]:
    """Check every pod-local model among `models` on the pod; PulseError on the first that is missing or differs. Returns those checked."""
    extra = extra_models(cfg)
    done = []
    for m in models:
        if m in extra:
            f = str(extra[m].get("lora") or m)          # an adapter entry: the adapter file is what is verified (the base is a catalog model)
            r = _kv(sh.run(pod_model_script(str(cfg["remote_dir"]), f, int(extra[m]["bytes"]), str(extra[m]["sha256"])), timeout=1800,
                           check=False)[1])
            if not r.get("verified"):
                raise PulseError(f"pod-local model {m}: {r.get('error', 'not verified')} - refusing to serve it")
            done.append(m)
    return done


def model_url(name: str, cat: Mapping[str, Mapping[str, Any]], base: str = HF_BASE) -> str:
    e = cat[name]
    return f"{base.rstrip('/')}/{e['repo']}/resolve/main/{e.get('file', name)}"


def vram_need_gb(name: str, slots: int, ctx_per_slot: int, cat: Optional[Mapping[str, Mapping[str, Any]]] = None) -> float:
    """GB of VRAM one server needs: weights + f16 KV cache of all slots + compute buffers (MODELS.json entries count; unknown KV size: 160 MiB/1k)."""
    e = (cat or CATALOG).get(name) or _POD_LOCAL.get(name) or catalog().get(name) or {"bytes": 0}
    kv = float(e.get("kv_mib_per_1k", 160)) * slots * ctx_per_slot / 1024 / 1024
    return round(float(e["bytes"]) / 2**30 + kv + COMPUTE_BUFFER_GB, 2)


def fits(names: Sequence[str], slots: int, ctx_per_slot: int, vram_gb: float = VRAM_GB) -> bool:
    return sum(vram_need_gb(n, slots, ctx_per_slot) for n in names) <= vram_gb - 1.0


SLOT_CHOICES = (16, 12, 8)      # blueprint rule 2: -np 8..16 with continuous batching
MIN_CTX = 4096                  # per-slot context is never cut below this (the longest judgment prompts with 5 shots stay under ~3k tokens)


def auto_slots(name: str, ctx_per_slot: int = 8192, vram_gb: float = VRAM_GB, copies: int = 1) -> tuple[int, int]:
    """(slots, ctx per slot) for one model on `vram_gb`: the most of 16 / 12 / 8 slots whose f16 KV cache fits beside the weights; when not even
    8 fit, the per-slot context is halved (not below MIN_CTX) first; a model too big for 8 slots at MIN_CTX gets as many as fit (at least 1).
    RTX 4090 24 GB, 8k context: 1.7B and 4B -> 16, 8B -> 12, 14B -> 8."""
    ctx = int(ctx_per_slot)
    while True:
        for n in SLOT_CHOICES:
            if fits([name] * copies, n, ctx, vram_gb):
                return n, ctx
        if ctx // 2 < MIN_CTX:
            break
        ctx //= 2
    n = SLOT_CHOICES[-1]
    while n > 1 and not fits([name] * copies, n, ctx, vram_gb):
        n -= 1
    return n, ctx


SHORT_SLOT_CHOICES = (32, 24) + SLOT_CHOICES


def short_ctx_slots(name: str, prompt_tokens: int, reply_tokens: int, vram_gb: float = VRAM_GB, copies: int = 1,
                    margin: float = 1.25) -> tuple[int, int]:
    """PROPOSAL (h52, 3 Oct 2026; not wired into serve): (slots, ctx per slot) sized to the job's real requests instead of the 8k default. The
    per-slot context is the smallest power of two (>= 2048) holding margin x (prompt + reply); then the most of 32 / 24 / 16 / 12 / 8 slots
    that fit. Traces prompts are ~1.4k tokens with replies capped at 260 (kept ones ~50-60): 14B on a 32 GB card -> 32 slots x 4096, the
    same KV cache as today's 16 x 8192, twice the requests decoding at once. Longer jobs (judgment shots, thinkbench) keep 8k."""
    need = int(margin * (max(0, int(prompt_tokens)) + max(0, int(reply_tokens))))
    ctx = 2048
    while ctx < need:
        ctx *= 2
    for n in SHORT_SLOT_CHOICES:
        if fits([name] * copies, n, ctx, vram_gb):
            return n, ctx
    return auto_slots(name, ctx, vram_gb, copies)


def inflight(cfg: Mapping[str, Any], slots: int) -> int:
    """Requests a job keeps in flight by default: inflight_factor x the served slots. Measured 3 Oct 2026 (14B, 16 slots, SSH tunnel): each
    request loses ~1 s in the tunnel, so with in-flight == slots only 3.7-5 of 16 slots were busy; with 48 in flight all 16 were busy (207 tok/s,
    GPU 69%). An explicit --workers still wins."""
    try:
        f = float(cfg.get("inflight_factor", DEFAULTS["inflight_factor"]))
    except (TypeError, ValueError):
        f = float(DEFAULTS["inflight_factor"])
    return max(1, int(round(max(1, int(slots)) * max(1.0, f))))


def served_slots(cfg: Mapping[str, Any], models: Sequence[str]) -> dict[str, tuple[int, int]]:
    """{model: (slots, ctx per slot)} for models served together: config 'slots' = 'auto' (default) shares the VRAM evenly; a number is used as given."""
    inst = max(1, int(cfg.get("instances", 1)))
    ctx = int(cfg.get("ctx_per_slot", 8192))
    if str(cfg.get("slots", "auto")) != "auto":
        return {m: (int(cfg["slots"]), ctx) for m in models}
    share = float(cfg.get("vram_gb", VRAM_GB)) / max(1, len(models))
    return {m: auto_slots(m, ctx, share, inst) for m in models}


# ------------------------------------------------------------------------------------------------ the remote shell
class Shell:
    """Runs a bash script on the pod: `argv` gets the script on stdin (ssh ... bash -s). Tests pass a local `bash -s` as argv."""

    def __init__(self, argv: Sequence[str]) -> None:
        self.argv = list(argv)
        self.log: list[str] = []

    def run_bytes(self, script: str, timeout: float = 600.0) -> tuple[int, bytes]:
        self.log.append(script)
        r = subprocess.run(self.argv, input=script.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
        return r.returncode, r.stdout

    def run(self, script: str, timeout: float = 600.0, check: bool = True) -> tuple[int, str]:
        self.log.append(script)
        r = subprocess.run(self.argv, input=script.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        out = r.stdout.decode("utf-8", "replace")
        if check and r.returncode != 0:
            raise PulseError(f"remote step failed (rc {r.returncode}): {out[-1500:]}")
        return r.returncode, out


def ssh_base(cfg: Mapping[str, Any]) -> list[str]:
    if not cfg.get("host"):
        raise PulseError(f"no 'host' in {cfg.get('_path')}: put the instance's SSH host and port there (see `gpu_pulse.py checklist`)")
    known = outside_repo(gpu_dir() / "known_hosts")
    argv = [str(cfg.get("ssh", "ssh")), "-p", str(cfg["port"]), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={known}", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
    if cfg.get("key_path"):
        argv += ["-i", str(Path(str(cfg["key_path"])).expanduser())]
    return argv + [f"{cfg.get('user', 'root')}@{cfg['host']}"]


def shell_for(cfg: Mapping[str, Any]) -> Shell:
    if cfg.get("shell_argv"):                             # tests: a local bash standing in for the pod
        return Shell(list(cfg["shell_argv"]))
    return Shell(ssh_base(cfg) + ["bash -s"])


def _kv(out: str) -> dict[str, str]:
    d: dict[str, str] = {}
    for ln in out.splitlines():
        if ln.startswith("@@") and "=" in ln:
            k, _, v = ln[2:].partition("=")
            d[k.strip()] = v.strip()
    return d


# ------------------------------------------------------------------------------------------------ remote scripts (bash; also run by the tests)
def probe_script(rdir: str) -> str:
    q = shlex.quote(rdir)
    return f"""set -u
mkdir -p {q}/models {q}/run {q}/llama
exe=""
for c in "$(cat {q}/run/exe 2>/dev/null)" "$(command -v llama-server 2>/dev/null)" /opt/llama.cpp/llama-server "$(find {q}/llama -name llama-server -type f 2>/dev/null | head -1)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then exe="$c"; break; fi
done
echo "@@exe=$exe"
if [ -n "$exe" ]; then echo "@@devices=$("$exe" --list-devices 2>&1 | grep -iE 'CUDA[0-9]' | head -3 | tr '\\n' ';')"; echo "@@version=$("$exe" --version 2>&1 | head -2 | tr '\\n' ' ')"; fi
if command -v nvidia-smi >/dev/null 2>&1; then echo "@@gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader | head -1)"; echo "@@cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1)"; fi
echo "@@disk_free_gb=$(df -Pk {q} | awk 'NR==2 {{printf "%.1f", $4/1048576}}')"
if command -v supervisorctl >/dev/null 2>&1; then echo "@@template_llama=$(supervisorctl status llama 2>/dev/null | awk '{{print $2}}')"; fi
CID="${{CONTAINER_ID:-$(tr '\\0' '\\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p')}}"
echo "@@container_id=$CID"
echo "@@vastai=$(command -v vastai 2>/dev/null)"
"""


def stop_template_script() -> str:
    """The template's own supervisor program 'llama' (autostarts only when LLAMA_MODEL is set; port 18000): stop it so it holds no VRAM."""
    return """if command -v supervisorctl >/dev/null 2>&1 && supervisorctl status llama 2>/dev/null | grep -q RUNNING; then
  supervisorctl stop llama && echo "@@template_stopped=1"
fi
"""


def fallback_install_script(rdir: str, tag: str = FALLBACK_TAG, allow_build: bool = False) -> str:
    q = shlex.quote(rdir)
    url, cudart = FALLBACK_URL.format(tag=tag), FALLBACK_CUDART.format(tag=tag)
    build = f"""
  if [ -z "$exe" ] && [ "{int(allow_build)}" = "1" ]; then
    t0=$(date +%s); apt-get update -qq && apt-get install -y -qq git cmake build-essential >/dev/null
    git clone --depth 1 --branch {tag} https://github.com/ggml-org/llama.cpp {q}/llama/src && cmake -S {q}/llama/src -B {q}/llama/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 >/dev/null \\
      && cmake --build {q}/llama/build --target llama-server -j "$(nproc)" >/dev/null && exe={q}/llama/build/bin/llama-server
    echo "@@build_seconds=$(( $(date +%s) - t0 ))"
  fi"""
    return f"""set -u
cd {q}/llama
exe=""
if curl -fsSL --retry 3 -o llama.tgz {shlex.quote(url)}; then
  tar -xzf llama.tgz && exe="$(find {q}/llama -name llama-server -type f | head -1)"
  if [ -n "$exe" ] && ! ldconfig -p 2>/dev/null | grep -q libcudart.so.12; then
    curl -fsSL --retry 3 -o cudart.tgz {shlex.quote(cudart)} && tar -xzf cudart.tgz -C "$(dirname "$exe")"
  fi
fi{build}
if [ -n "$exe" ]; then chmod +x "$exe"; echo "$exe" > {q}/run/exe; fi
echo "@@exe=$exe"
"""


def model_script(rdir: str, name: str, url: str, size: int, sha: str) -> str:
    """Fetch one model unless a verified copy is there (size + recorded hash). A file whose sha256 differs is DELETED and the step fails.
    A copy the template's `-hf` download left under /workspace/llama.cpp is checked and reused before anything is fetched."""
    q, f = shlex.quote(rdir), shlex.quote(name)
    return f"""set -u
cd {q}/models
if [ -f {f} ] && [ "$(stat -c %s {f} 2>/dev/null || wc -c < {f})" = "{size}" ] && [ "$(cat {f}.ok 2>/dev/null)" = "{sha}" ]; then echo "@@cached=1"; exit 0; fi
rm -f {f}.ok
cand="$(find /workspace/llama.cpp -maxdepth 2 -name '*{name}' -size {size}c 2>/dev/null | head -1)"
if [ -n "$cand" ]; then cp -f "$cand" {f}.part; else
  curl -fL --retry 5 --retry-delay 3 -s -o {f}.part {shlex.quote(url)} || {{ rm -f {f}.part; echo "@@error=download failed"; exit 3; }}
fi
got="$(sha256sum {f}.part | cut -d' ' -f1)"
if [ "$got" != "{sha}" ]; then rm -f {f}.part; echo "@@error=sha256 mismatch: got $got want {sha}"; exit 4; fi
mv -f {f}.part {f} && echo "{sha}" > {f}.ok && echo "@@fetched=1"
"""


SERVER_FLAG = re.compile(r"^-{1,2}[A-Za-z][A-Za-z0-9-]*$|^[A-Za-z0-9_.]+$")   # flags and plain values only: no shell metacharacters


def server_flags(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Config 'server_flags' (a list or a space-separated string): extra llama-server arguments, e.g. ["-ub", "2048", "-fa", "on"]. h51 (3 Oct):
    the 14B traces job spent ~54% of server time in prompt processing at the default micro-batch (512); a larger -ub speeds prefill on a big
    GPU. Default none (unchanged behaviour). A token that is not a plain flag/value raises (it would reach a remote shell)."""
    raw = cfg.get("server_flags") or ()
    toks = tuple(str(raw).split()) if isinstance(raw, str) else tuple(str(x) for x in raw)
    bad = [t for t in toks if not SERVER_FLAG.match(t)]
    if bad:
        raise PulseError(f"server_flags: refusing {bad!r}")
    return toks


def server_script(rdir: str, exe: str, name: str, port: int, slots: int, ctx: int, flags: Sequence[str] = (), base: str = "",
                  lora: str = "") -> str:
    """Start (or keep) one llama-server on 127.0.0.1:port with every layer on the GPU. Kept when its recorded arguments are the same and
    /health answers; otherwise the old process is stopped and a new one started (nohup, survives the SSH session). `base` + `lora`: the served
    name is an adapter entry - the base GGUF with the adapter applied at load time (no merged model needed)."""
    q = shlex.quote(rdir)
    extra = "".join(f" {shlex.quote(f)}" for f in flags)
    if lora:
        extra += f" --lora {q}/models/{shlex.quote(lora)}"
    args = f"-m {q}/models/{shlex.quote(base or name)} --host 127.0.0.1 --port {port} -ngl 99 -np {slots} -c {slots * ctx} --metrics --no-webui{extra}"
    sig = f"{name}|{slots}|{ctx}" + (f"|{' '.join(flags)}" if flags else "")
    return f"""set -u
cd {q}/run
if [ "$(cat {port}.sig 2>/dev/null)" = "{sig}" ] && [ -f {port}.pid ] && kill -0 "$(cat {port}.pid)" 2>/dev/null \\
   && curl -sf -o /dev/null http://127.0.0.1:{port}/health; then echo "@@kept={port}"; exit 0; fi
if [ -f {port}.pid ]; then kill "$(cat {port}.pid)" 2>/dev/null; sleep 1; kill -9 "$(cat {port}.pid)" 2>/dev/null; rm -f {port}.pid; fi
nohup {shlex.quote(exe)} {args} > {port}.log 2>&1 &
echo $! > {port}.pid; echo "{sig}" > {port}.sig
echo "@@started={port}"
"""


def health_script(rdir: str, ports: Sequence[int], timeout_s: int = 300) -> str:
    q = shlex.quote(rdir)
    ps = " ".join(str(p) for p in ports)
    return f"""set -u
cd {q}/run
end=$(( $(date +%s) + {timeout_s} ))
for p in {ps}; do
  until curl -sf -o /dev/null http://127.0.0.1:$p/health; do
    if ! kill -0 "$(cat $p.pid)" 2>/dev/null; then echo "@@dead=$p"; tail -20 $p.log; exit 5; fi
    if [ $(date +%s) -gt $end ]; then echo "@@timeout=$p"; exit 6; fi
    sleep 1
  done
  echo "@@healthy=$p"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "@@vram_$p=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v P="$(cat $p.pid)" '$1==P {{print $2}}')"
  fi
done
"""


def stop_servers_script(rdir: str, keep: Sequence[int] = ()) -> str:
    q = shlex.quote(rdir)
    k = " ".join(str(p) for p in keep)
    return f"""cd {q}/run 2>/dev/null || exit 0
for f in *.pid; do
  [ -f "$f" ] || continue
  p="${{f%.pid}}"
  case "$p" in ''|*[!0-9]*) continue;; esac                     # servers are <port>.pid; deadman.pid (the budget guard) is never stopped here
  case " {k} " in *" $p "*) continue;; esac
  kill "$(cat $f)" 2>/dev/null; sleep 0.5; kill -9 "$(cat $f)" 2>/dev/null; rm -f "$f" "$p.sig"; echo "@@stopped=$p"
done
"""


def deadman_script(rdir: str, seconds: int, action: str) -> str:
    """The pod's own budget guard: after `seconds` it stops (or destroys) itself with the instance key the template ships with the Vast CLI.
    Re-armed on every setup/run (the old timer is killed first); action 'off' only disarms."""
    q = shlex.quote(rdir)
    arm = "" if action == "off" else f"""
CID="${{CONTAINER_ID:-$(tr '\\0' '\\n' < /proc/1/environ 2>/dev/null | sed -n 's/^CONTAINER_ID=//p')}}"
if command -v vastai >/dev/null 2>&1 && [ -n "$CID" ]; then
  nohup bash -c "sleep {int(seconds)}; vastai {action} instance $CID || vastai stop instance $CID" > {q}/run/deadman.log 2>&1 &
  echo $! > {q}/run/deadman.pid; echo "@@deadman_armed={int(seconds)}"
else echo "@@deadman_armed=0"; fi"""
    return f"""mkdir -p {q}/run
if [ -f {q}/run/deadman.pid ]; then kill "$(cat {q}/run/deadman.pid)" 2>/dev/null; rm -f {q}/run/deadman.pid; fi{arm}
"""


# ------------------------------------------------------------------------------------------------ the ledger and the budget guard
def ledger_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return gpu_dir(env) / "ledger.json"


@dataclasses.dataclass
class Budget:
    path: Path
    cap_usd: float
    clock: Callable[[], float] = time.time

    def load(self) -> dict[str, Any]:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {"pulses": []}
        except (OSError, ValueError):
            return {"pulses": []}

    def open_pulse(self) -> Optional[dict[str, Any]]:
        ps = self.load().get("pulses", [])
        return ps[-1] if ps and ps[-1].get("end") is None else None

    def begin(self, usd_per_hr: float, instance: str = "", started: Optional[float] = None) -> dict[str, Any]:
        d = self.load()
        cur = self.open_pulse()
        if cur is not None:
            return cur
        p = {"id": dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "instance": str(instance),
             "usd_per_hr": float(usd_per_hr), "start": float(started if started is not None else self.clock()), "end": None, "bandwidth_usd": 0.0}
        d.setdefault("pulses", []).append(p)
        _write_json(self.path, d)
        return p

    def end(self) -> Optional[dict[str, Any]]:
        d = self.load()
        ps = d.get("pulses", [])
        if not ps or ps[-1].get("end") is not None:
            return None
        ps[-1]["end"] = self.clock()
        _write_json(self.path, d)
        return dict(ps[-1])

    def add_bandwidth(self, usd: float) -> None:
        d = self.load()
        ps = d.get("pulses", [])
        if ps and ps[-1].get("end") is None:
            ps[-1]["bandwidth_usd"] = round(float(ps[-1].get("bandwidth_usd", 0.0)) + usd, 4)
            _write_json(self.path, d)

    def spent(self) -> float:
        now = self.clock()
        tot = 0.0
        for p in self.load().get("pulses", []):
            end = p.get("end") if p.get("end") is not None else now
            tot += max(0.0, float(end) - float(p["start"])) / 3600.0 * float(p["usd_per_hr"]) + float(p.get("bandwidth_usd", 0.0))
        return round(tot, 4)

    def remaining(self) -> float:
        return round(self.cap_usd - self.spent(), 4)

    def seconds_left(self, usd_per_hr: float, reserve_s: float = 0.0) -> float:
        return max(0.0, self.remaining() / max(1e-9, usd_per_hr) * 3600.0 - reserve_s)

    def check(self, usd_per_hr: float, reserve_s: float = 0.0) -> None:
        if self.seconds_left(usd_per_hr, reserve_s) <= 0.0:
            raise BudgetExceeded(f"GPU budget reached: spent ${self.spent():.2f} of ${self.cap_usd:.2f} (reserve {reserve_s / 60:.0f} min kept for teardown)")


def budget_for(cfg: Mapping[str, Any], clock: Callable[[], float] = time.time) -> Budget:
    return Budget(outside_repo(Path(str(cfg.get("ledger") or ledger_path()))), float(cfg.get("budget_usd", 8.0)), clock)


# ------------------------------------------------------------------------------------------------ setup / serve / teardown
def plan_ports(cfg: Mapping[str, Any], models: Sequence[str]) -> dict[str, list[int]]:
    """Remote ports per model: instance i of the k-th configured model -> base + 10 k + i (stable across calls)."""
    allm = list(cfg.get("models") or [])
    out: dict[str, list[int]] = {}
    for m in models:
        k = allm.index(m) if m in allm else len(allm) + list(models).index(m)
        out[m] = [int(cfg["remote_port_base"]) + 10 * k + i for i in range(max(1, int(cfg.get("instances", 1))))]
    return out


def setup(cfg: Mapping[str, Any], shell: Optional[Shell] = None, serve_models: Optional[Sequence[str]] = None,
          say: Callable[[str], None] = print) -> dict[str, Any]:
    """Idempotent pod bootstrap. Returns the probe facts, what was fetched/kept and the serving plan {model: [remote ports]}."""
    sh = shell or shell_for(cfg)
    rdir = str(cfg["remote_dir"])
    t0 = time.monotonic()
    budget = budget_for(cfg)                              # billing started when the owner rented: 'rented_at' (epoch s) when given
    budget.begin(float(cfg["usd_per_hr"]), str(cfg.get("instance_id", "")), started=cfg.get("rented_at"))
    budget.check(float(cfg["usd_per_hr"]))
    facts = _kv(sh.run(probe_script(rdir), timeout=120)[1])
    say(f"pod: gpu={facts.get('gpu', '?')} {facts.get('cuda', '')} llama-server={facts.get('exe') or 'none'} devices={facts.get('devices') or 'none'} "
        f"disk free={facts.get('disk_free_gb', '?')} GB template llama={facts.get('template_llama', 'n/a')}")
    if facts.get("template_llama") == "RUNNING":
        sh.run(stop_template_script(), timeout=60)
        say("stopped the template's own llama-server (supervisor program 'llama', port 18000) so it holds no VRAM")
    if not facts.get("exe") or not facts.get("devices"):
        say("no CUDA-capable llama-server in the image: installing the official ggml-org Linux CUDA 12.8 build" +
            (" (cmake build allowed as last resort)" if cfg.get("allow_build") else ""))
        got = _kv(sh.run(fallback_install_script(rdir, str(cfg.get("fallback_tag", FALLBACK_TAG)), bool(cfg.get("allow_build"))), timeout=1800)[1])
        facts.update({k: v for k, v in got.items() if v})
        facts.update({k: v for k, v in _kv(sh.run(probe_script(rdir), timeout=120)[1]).items() if v})
        if not facts.get("exe"):
            raise PulseError("no llama-server could be installed on the pod")
        if not facts.get("devices"):
            raise PulseError(f"llama-server at {facts.get('exe')} lists no CUDA device: it would serve from the CPU (paid GPU idle); stop here")
    cat = catalog()
    models = list(cfg.get("models") or [])
    fetched: list[str] = []
    cached: list[str] = []
    pod_local = extra_models(cfg)
    for m in models:
        if m in pod_local:                                # made on the pod (e.g. a fine-tune): verified, never downloaded
            verify_pod_models(cfg, [m], sh)
            cached.append(m)
            say(f"model {m}: pod-local, sha256 verified")
            continue
        if m not in cat:
            raise PulseError(f"model {m} has no pinned sha256 (CATALOG / MODELS.json): refusing to download an unverifiable file")
        r = _kv(sh.run(model_script(rdir, m, model_url(m, cat, str(cfg.get("hf_base", HF_BASE))), int(cat[m]["bytes"]), str(cat[m]["sha256"])),
                       timeout=3600, check=False)[1])
        if r.get("error"):
            raise PulseError(f"{m}: {r['error']} (the bad file was deleted)")
        (cached if r.get("cached") else fetched).append(m)
        say(f"model {m}: {'verified copy kept' if r.get('cached') else 'downloaded, sha256 verified'}")
    tb = float(cfg.get("bandwidth_usd_per_tb", 0.0))
    if fetched and tb > 0:
        budget_for(cfg).add_bandwidth(sum(int(cat[m]["bytes"]) for m in fetched) / 1e12 * tb)
    steps = run_setup_steps(cfg, sh, say)
    plan = serve(cfg, list(serve_models) if serve_models is not None else models[:1], sh, facts["exe"], say=say)
    armed = arm_deadman(cfg, budget, sh)
    say(f"pod dead-man switch: {'vastai ' + str(cfg.get('deadman', 'stop')) + ' of itself in ' + str(armed // 60) + ' min' if armed else 'not armed (no vastai CLI / CONTAINER_ID on the pod)'}")
    return {"facts": facts, "fetched": fetched, "cached": cached, "setup_steps": steps, "plan": plan, "seconds": round(time.monotonic() - t0, 1)}


SETUP_HOOKS: list[Callable[[Mapping[str, Any], "Shell", Callable[[str], None]], None]] = []   # in-process hooks (other modules append)


def setup_step_script(rdir: str, name: str, body: str, always: bool = False) -> str:
    """One optional install step on the pod, idempotent: it runs once and leaves run/step_<name>.done (unless `always`); a failing step
    leaves no marker and stops setup with its output."""
    q = shlex.quote(rdir)
    mark = f"{q}/run/step_{shlex.quote(name)}.done"
    guard = "" if always else f'if [ -f {mark} ]; then echo "@@step_kept=1"; exit 0; fi\n'
    return f"""set -u
mkdir -p {q}/run
{guard}( set -e
{body}
) || {{ echo "@@step_failed=1"; exit 7; }}
date +%s > {mark}; echo "@@step_done=1"
"""


def run_setup_steps(cfg: Mapping[str, Any], sh: "Shell", say: Callable[[str], None] = print) -> dict[str, str]:
    """The pluggable part of setup (e.g. a training stack next to llama.cpp): config 'setup_steps' = [{"name": ..., "script": "<bash>" or
    "script_file": "<local .sh, sent as text>", "always": false, "timeout_s": 3600}], then every callable in SETUP_HOOKS(cfg, shell, say).
    Runs after the models are verified and before the servers start. Returns {step name: 'done' | 'kept'}."""
    out: dict[str, str] = {}
    for st in cfg.get("setup_steps") or []:
        name = str(st["name"])
        body = str(st.get("script") or "")
        if st.get("script_file"):
            body = Path(str(st["script_file"])).expanduser().read_text(encoding="utf-8").replace("\r\n", "\n")  # CRLF breaks bash on the pod
        outbound_ok([{"content": body}])                  # a step script is sent to the pod: the private-marker guard applies
        rc, o = sh.run(setup_step_script(str(cfg["remote_dir"]), name, body, bool(st.get("always"))), timeout=float(st.get("timeout_s", 3600)),
                       check=False)
        kv = _kv(o)
        if rc != 0 or kv.get("step_failed"):
            raise PulseError(f"setup step {name} failed (rc {rc}): {o[-1500:]}")
        out[name] = "kept" if kv.get("step_kept") else "done"
        say(f"setup step {name}: {out[name]}")
    for hook in SETUP_HOOKS:
        hook(cfg, sh, say)
    return out


def serve(cfg: Mapping[str, Any], models: Sequence[str], shell: Optional[Shell] = None, exe: str = "",
          say: Callable[[str], None] = print) -> dict[str, list[int]]:
    """Exactly these models served (others stopped first, so their VRAM is free): one server per instance, -np slots each."""
    sh = shell or shell_for(cfg)
    rdir = str(cfg["remote_dir"])
    if not exe:
        exe = _kv(sh.run(probe_script(rdir), timeout=120)[1]).get("exe", "")
        if not exe:
            raise PulseError("no llama-server on the pod: run setup first")
    verify_pod_models(cfg, models, sh)                    # pod-local models (extra_models): size + sha256 checked before serving
    inst = max(1, int(cfg.get("instances", 1)))
    sl = served_slots(cfg, models)
    need = sum(vram_need_gb(m, *sl[m]) * inst for m in models)
    if need > float(cfg.get("vram_gb", VRAM_GB)) - 1.0:
        raise PulseError(f"{list(models)} x{inst} with slots {sl} need {need:.1f} GB > {float(cfg.get('vram_gb', VRAM_GB)):.0f} GB VRAM: serve fewer at once")
    plan = plan_ports(cfg, models)
    keep = [p for ps in plan.values() for p in ps]
    sh.run(stop_servers_script(rdir, keep), timeout=60)
    for m, ports in plan.items():
        for p in ports:
            ex = extra_models(cfg).get(m) or {}
            r = _kv(sh.run(server_script(rdir, exe, m, p, *sl[m], flags=server_flags(cfg), base=str(ex.get("base") or ""),
                                         lora=str(ex.get("lora") or "")), timeout=60)[1])
            say(f"server {m} on pod port {p} (-np {sl[m][0]}, {sl[m][1]} tokens per slot): {'kept (healthy, same arguments)' if r.get('kept') else 'started'}")
    h = _kv(sh.run(health_script(rdir, keep, int(cfg.get("health_timeout_s", 300))), timeout=float(cfg.get("health_timeout_s", 300)) + 30)[1])
    for p in keep:
        v = h.get(f"vram_{p}")
        if v is not None and v != "" and float(v) < 500:
            raise PulseError(f"server on port {p} holds only {v} MiB of VRAM: the model is NOT on the GPU (CUDA backend failed to load)")
    return plan


def arm_deadman(cfg: Mapping[str, Any], budget: Budget, shell: Optional[Shell] = None) -> int:
    """(Re)arm the pod's self-stop at the moment the budget runs out (minus nothing: the PC-side guard stops work `reserve_minutes` earlier)."""
    sh = shell or shell_for(cfg)
    secs = int(budget.seconds_left(float(cfg["usd_per_hr"])))
    r = _kv(sh.run(deadman_script(str(cfg["remote_dir"]), max(60, secs), str(cfg.get("deadman", "stop"))), timeout=60, check=False)[1])
    return int(r.get("deadman_armed", "0") or 0)


def teardown(cfg: Mapping[str, Any], shell: Optional[Shell] = None, destroy: bool = False, say: Callable[[str], None] = print) -> dict[str, Any]:
    sh = shell or shell_for(cfg)
    out: dict[str, Any] = {}
    try:
        r = sh.run(stop_servers_script(str(cfg["remote_dir"])), timeout=60, check=False)[1]
        out["stopped"] = [ln.split("=", 1)[1] for ln in r.splitlines() if ln.startswith("@@stopped=")]
    except Exception as e:                                           # noqa: BLE001 - an unreachable pod still gets its destroy command printed
        out["stop_error"] = f"{type(e).__name__}: {e}"
    close_tunnel()
    p = budget_for(cfg).end()
    out["pulse"] = p
    iid = str(cfg.get("instance_id") or "<INSTANCE_ID>")
    cmd = f"vastai destroy instance {iid}"
    say(f"servers stopped; ledger: ${budget_for(cfg).spent():.2f} spent of ${float(cfg.get('budget_usd', 8.0)):.2f}")
    say(f"TO STOP ALL BILLING (storage is billed while an instance is merely stopped) run:  {cmd}   (or Destroy in the Vast.ai console)")
    out["destroy_command"] = cmd
    if destroy:
        say("--destroy: asking the pod to destroy itself with the instance key the template ships (vastai destroy instance $CONTAINER_ID) ...")
        rc, o = sh.run('CID="${CONTAINER_ID:-$(tr \'\\0\' \'\\n\' < /proc/1/environ 2>/dev/null | sed -n \'s/^CONTAINER_ID=//p\')}"; '
                       'nohup bash -c "sleep 2; vastai destroy instance $CID" >/dev/null 2>&1 & echo "@@cid=$CID"', timeout=60, check=False)
        out["destroy_requested"] = rc == 0 and bool(_kv(o).get("cid"))
        say("destroy requested" if out["destroy_requested"] else f"could not request destroy from the pod; run the command above. ({o[-300:]})")
    return out


# ------------------------------------------------------------------------------------------------ the tunnel (and how Nupen attaches to it)
def tunnel_path(env: Optional[Mapping[str, str]] = None) -> Path:
    return gpu_dir(env) / "tunnel.json"


def write_tunnel(models: Mapping[str, Sequence[int]], pulse: str, pid: Optional[int] = None, path: Optional[Path] = None,
                 slots: Optional[Mapping[str, int]] = None, extra: Optional[Mapping[str, int]] = None) -> Path:
    """The tunnel file - the public interface for scripts outside this module: {"pulse": id, "models": {model file name: [local ports]},
    "slots": {model: parallel slots per server}, "extra": {name: local port}}. Every port listens on 127.0.0.1 only; each model port is an
    OpenAI-compatible llama-server (/v1/chat/completions, /v1/completions, /health, /metrics). Use endpoints() rather than parsing it."""
    p = path or tunnel_path()
    _write_json(p, {"pulse": pulse, "pid": pid, "created": time.time(), "models": {m: list(v) for m, v in models.items()},
                    "slots": dict(slots or {}), "extra": dict(extra or {})})
    return p


def endpoints(pf: Optional[Path] = None) -> dict[str, Any]:
    """For external scripts: {"pulse": id, "models": {model: {"urls": ["http://127.0.0.1:<port>/v1", ...], "slots": n}}, "extra": {name:
    "http://127.0.0.1:<port>"}} of the open tunnel ({} when none). Prompts sent there must pass outbound_ok() first (nothing private leaves)."""
    t = _tunnel(Path(pf) if pf else pulse_file())
    if not t:
        return {}
    return {"pulse": t.get("pulse"), "models": {m: {"urls": [f"http://127.0.0.1:{p}/v1" for p in ps], "slots": (t.get("slots") or {}).get(m)}
                                                for m, ps in (t.get("models") or {}).items()},
            "extra": {k: f"http://127.0.0.1:{v}" for k, v in (t.get("extra") or {}).items()}}


def pulse_file(cfg: Optional[Mapping[str, Any]] = None, env: Optional[Mapping[str, str]] = None) -> Path:
    """The tunnel file of the pulse that is switched on (creator.device.pulse_on): env NUPEN_GPU_PULSE (set by run_jobs for its job processes),
    else <runtime>/gpu/tunnel.json (device setting 'gpu_pulse')."""
    e = os.environ if env is None else env
    return Path(e["NUPEN_GPU_PULSE"]) if e.get("NUPEN_GPU_PULSE") else tunnel_path(e)


def open_tunnel(cfg: Mapping[str, Any], plan: Mapping[str, Sequence[int]], pulse: str) -> Path:
    """One `ssh -N -L 127.0.0.1:<local>:127.0.0.1:<remote>` per served port; nothing is exposed beyond this PC's loopback."""
    close_tunnel()
    fwd: list[str] = []
    local: dict[str, list[int]] = {}
    for m, ports in plan.items():
        for rp in ports:
            lp = int(cfg["local_port_base"]) + (int(rp) - int(cfg["remote_port_base"]))
            fwd += ["-L", f"127.0.0.1:{lp}:127.0.0.1:{rp}"]
            local.setdefault(m, []).append(lp)
    extra: dict[str, int] = {}
    for name, rp in dict(cfg.get("extra_forwards") or {}).items():        # e.g. a training dashboard: {"name": remote port}
        lp = int(cfg["local_port_base"]) + (int(rp) - int(cfg["remote_port_base"])) if int(rp) >= int(cfg["remote_port_base"]) else int(rp)
        fwd += ["-L", f"127.0.0.1:{lp}:127.0.0.1:{int(rp)}"]
        extra[str(name)] = lp
    argv = ssh_base(cfg)
    argv = argv[:-1] + ["-N", "-o", "ExitOnForwardFailure=yes"] + fwd + [argv[-1]]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30:
        if proc.poll() is not None:
            raise PulseError("the SSH tunnel exited at once (host/port/key wrong, or a local port in use)")
        if all(_healthy(p) for ps in local.values() for p in ps):
            break
        time.sleep(0.5)
    return write_tunnel(local, pulse, proc.pid, slots={m: n for m, (n, _c) in served_slots(cfg, list(plan)).items()}, extra=extra)


def close_tunnel(path: Optional[Path] = None) -> None:
    p = path or tunnel_path()
    try:
        pid = json.loads(p.read_text(encoding="utf-8")).get("pid")
    except (OSError, ValueError):
        return
    if pid:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=20)
            else:
                os.kill(int(pid), 15)
        except (OSError, subprocess.SubprocessError):
            pass
    p.unlink(missing_ok=True)


def _healthy(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return bool(r.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


_TCACHE: dict[str, Any] = {"key": None, "body": {}}
_RR = {"i": 0}


def _tunnel(pf: Path) -> dict[str, Any]:
    try:
        st = pf.stat()
    except OSError:
        return {}
    key = (str(pf), st.st_mtime_ns, st.st_size)
    if _TCACHE["key"] != key:
        try:
            _TCACHE["body"] = json.loads(pf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _TCACHE["body"] = {}
        _TCACHE["key"] = key
    return dict(_TCACHE["body"])


def serves(pf: Path, model: Any) -> bool:
    """Does the pulse in tunnel file `pf` serve this model (by file name)? No network."""
    return bool(_tunnel(pf).get("models", {}).get(Path(str(model)).name))


def attach(pf: Path, model: Any) -> Optional[tuple[int, str]]:
    """(local port, pulse id) of a healthy forwarded server of this model; None when the pulse does not serve it. A served model whose
    server does not answer is an ERROR (never a silent fall-back to the CPU: CPU and GPU timings must not mix)."""
    t = _tunnel(pf)
    ports = list(t.get("models", {}).get(Path(str(model)).name) or [])
    if not ports:
        return None
    for k in range(len(ports)):
        p = int(ports[(_RR["i"] + k) % len(ports)])
        if _healthy(p, 3.0):
            _RR["i"] += 1
            return p, str(t.get("pulse") or "pulse")
    raise PulseError(f"GPU pulse server(s) for {Path(str(model)).name} on {ports} do not answer (tunnel down or pod gone)")


def settings_overlay(pf: Path, env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Device settings while a pulse is on: the job's model (env NUPEN_PULSE_MODEL) becomes the thinker; as many thinkers as the pod has slots."""
    e = os.environ if env is None else env
    t = _tunnel(pf)
    out: dict[str, Any] = {}
    m = e.get("NUPEN_PULSE_MODEL", "")
    if m and m in t.get("models", {}):
        out["think_model"] = m
        out["think_servers"] = int(e.get("NUPEN_PULSE_SLOTS") or (t.get("slots") or {}).get(m) or 8) * len(t["models"][m])
    return out


def outbound_ok(messages: Sequence[Mapping[str, str]]) -> None:
    """Defence in depth at the PC -> pod boundary: a prompt carrying a private marker never leaves the machine."""
    for m in messages:
        c = str(m.get("content", ""))
        for k in PRIVATE_MARKERS:
            if k in c:
                raise PulseError(f"refusing to send a prompt containing '{k}' to the GPU pod (private material stays on this PC)")


# ------------------------------------------------------------------------------------------------ what leaves the machine
def manifest(cfg: Mapping[str, Any]) -> dict[str, Any]:
    cat = catalog()
    return {
        "files_uploaded": [],
        "commands_sent": "bash scripts of this module (probe, stop template server, llama.cpp fallback install, model fetch + sha256, servers, dead-man switch, "
                         "monitor) + config 'setup_steps' and external 'remote' job scripts (private-marker guard applied)",
        "setup_steps": [str(st.get("name")) for st in cfg.get("setup_steps") or []],
        "copied_home": "only the 'outputs' an external job names (tar over the SSH channel, unpacked under <runtime>/gpu/outputs)",
        "downloaded_by_the_pod": [model_url(m, cat, str(cfg.get("hf_base", HF_BASE))) for m in cfg.get("models", []) if m in cat]
        + [FALLBACK_URL.format(tag=cfg.get("fallback_tag", FALLBACK_TAG)) + " (only if the image has no CUDA llama-server)"],
        "sent_over_the_tunnel": ["thinkbench parts b and c prompts (Nupen's kernel package records; questions from public git histories)",
                                 "judgment prompts (kernel package records, Nupen's public git history, public-repo text caches)",
                                 "reasoning drill prompts (public repositories and Nupen's public repository history)"],
        "never": ["state/livesim", "~/oldpc", "~/Masterstock (thinkbench part d is skipped on a pulse)", "SSH keys / API keys / pulse.json"],
    }


def check_upload(paths: Sequence[Path]) -> list[Path]:
    """The upload list, refused whole when any entry is private (today the list is empty: nothing is uploaded)."""
    for p in paths:
        s = Path(p).as_posix()
        for k in PRIVATE_PATHS:
            if k in s:
                raise PulseError(f"refusing to upload private path {s}")
    return list(paths)


# ------------------------------------------------------------------------------------------------ jobs (run in a child process: python -m creator.gpupulse job ...)
def _drain(next_job: Callable[[], Optional[Callable[[], None]]], n: int, workers: int, deadline: float) -> int:
    """Run up to n jobs of a filler with `workers` threads (n = 0: until the filler has nothing more)."""
    done = {"n": 0, "handed": 0}
    lock = threading.Lock()

    def worker() -> None:
        idle = 0
        while time.monotonic() < deadline:
            with lock:
                if n and done["handed"] >= n:
                    return
                job = next_job()
                if job is not None:
                    done["handed"] += 1
            if job is None:
                idle += 1
                if idle > 20:
                    return
                time.sleep(0.5)
                continue
            idle = 0
            job()
            with lock:
                done["n"] += 1
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return done["n"]


M17, M4, M8, M14 = "Qwen3-1.7B-Q4_K_M.gguf", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf", "Qwen3-14B-Q4_K_M.gguf"
BEST = "best"                   # in a pulse plan: the config's 'best_model' (set it from pulse 1's result)
PULSES: dict[int, list[str]] = {
    1: [f"smoke:{M17}", f"probe:{M17}", f"thinkbench:{M17}", f"probe:{M4}", f"thinkbench:{M4}",
        f"probe:{M8}", f"thinkbench:{M8}", f"probe:{M14}", f"thinkbench:{M14}"],
    2: [f"smoke:{BEST}", f"judgment:{BEST}:0:290"],
    3: [f"smoke:{BEST}", f"drills:{BEST}:0:230"],
    4: [f"smoke:{BEST}", f"traces:{BEST}:0:350"],
}
KINDS = ("smoke", "probe", "thinkbench", "judgment", "drills", "traces")


def pulse_jobs(cfg: Mapping[str, Any], pulse: int) -> list[str]:
    if pulse not in PULSES:
        raise PulseError(f"no pulse {pulse} in the plan (1-{max(PULSES)}); pulse 5 is the reserve: name its jobs yourself")
    best = str(cfg.get("best_model") or M8)
    return [j.replace(f":{BEST}", f":{best}") for j in PULSES[pulse]]


def ext_job(spec: Mapping[str, Any]) -> dict[str, Any]:
    """An EXTERNAL job (other modules' GPU work, e.g. a fine-tune or an embedding index), given as a dict in any job list:
      name          unique label (results, outputs folder)
      exactly one of
        call        "package.module:function" - run in a child process on this PC with the pulse switched on; called with one dict
                    (state, repo, owner_dir, workers, deadline (monotonic), pulse, tunnel_file, model, endpoints); returns a JSON-able dict
        command     [argv...] - a local program run with the same environment (NUPEN_GPU_PULSE = the tunnel file, NUPEN_PULSE_MODEL)
        remote      "<bash script>" - run ON THE POD (cwd = remote_dir); print '@@result=<json>' to return a result
      model         model file name to serve for it ("" = none: no server needed)
      free_gpu      true: stop every server first (training needs the VRAM)
      minutes       expected minutes (the plan's estimate; required)
      max_minutes   hard cap (default 0 = none; the budget cap always applies)
      outputs       pod paths (relative to remote_dir) copied home after the job: <runtime>/gpu/outputs/<pulse>/<name>/ (or 'home')
      low_util_abort_minutes   override of the idle-GPU stop for this job (0 = never; e.g. a job with a long CPU-side phase)
      args          dict handed to a 'call' job as ctx['args'] (e.g. which earlier job's model to register)"""
    j = dict(spec)
    kinds = [k for k in ("call", "command", "remote") if j.get(k)]
    if not j.get("name") or len(kinds) != 1 or "minutes" not in j:
        raise PulseError(f"bad external job {spec!r}: needs name, minutes and exactly one of call / command / remote")
    if kinds[0] == "call" and ":" not in str(j["call"]):
        raise PulseError(f"external job {j['name']}: call must be 'package.module:function'")
    if not isinstance(j.get("args") or {}, Mapping):
        raise PulseError(f"external job {j['name']}: args must be a dict (passed to a 'call' job as ctx['args'])")
    j.update(kind="ext", via=kinds[0], model=str(j.get("model") or ""), n=0, max_minutes=float(j.get("max_minutes") or 0.0),
             minutes=float(j["minutes"]), spec=f"ext:{j['name']}", outputs=[str(o) for o in j.get("outputs") or []])
    return j


def job_list(cfg: Mapping[str, Any], jobs: Sequence[str] = (), pulse: int = 1, jobs_file: str = "", jobs_from: str = "") -> list[Any]:
    """The job list of a run/plan/prepare: explicit JOB strings, else a JSON file, else 'module:function' (called with the config), else the
    blueprint's pulse. Every entry is validated here (parse_job) so a bad list fails before any paid minute."""
    if jobs:
        out: list[Any] = list(jobs)
    elif jobs_file:
        out = list(json.loads(Path(jobs_file).read_text(encoding="utf-8")))
    elif jobs_from:
        import importlib
        mod, _, fn = jobs_from.partition(":")
        out = list(getattr(importlib.import_module(mod), fn)(cfg))
    else:
        out = pulse_jobs(cfg, pulse)
    for j in out:
        parse_job(j)
    return out


def parse_job(spec: Any) -> dict[str, Any]:
    """'<kind>:<model>[:<n>[:<max minutes>]]' with kind smoke | probe | thinkbench | judgment (n batches) | drills (n batches) | traces (n
    questions); n = 0 or absent: until the time cap (or the work) runs out. A dict is an external job (ext_job)."""
    if isinstance(spec, Mapping):
        return ext_job(spec)
    parts = str(spec).split(":")
    kind = parts[0]
    if kind not in KINDS or len(parts) < 2 or not parts[1]:
        raise PulseError(f"bad job {spec!r}: <{'|'.join(KINDS)}>:<model>[:<n>[:<max minutes>]]")
    return {"kind": kind, "model": parts[1], "n": int(parts[2]) if len(parts) > 2 and parts[2] else 0,
            "max_minutes": float(parts[3]) if len(parts) > 3 and parts[3] else 0.0, "spec": spec}


# ------------------------------------------------------------------------------------------------ the cost plan (dry run: nothing is contacted)
# Single-stream decode tok/s on an RTX 4090 (llama.cpp, Q4_K_M): 8B ~141 and 14B ~69-83 (cited 3 Oct 2026); 1.7B / 4B scaled by weight bytes
# (decode is memory-bound). Prompt processing ~40x decode. Aggregate with n parallel slots = single x BATCH_GAIN (assumed; the pulse-1 probe
# measures the real curve and later plans use it: <state>/thinking/gpu_probe.jsonl).
SINGLE_TPS = {M17: 330.0, M4: 220.0, M8: 141.0, M14: 76.0}
PP_FACTOR = 40.0
BATCH_GAIN = ((1, 1.0), (4, 3.0), (8, 4.5), (12, 5.2), (16, 6.0))
# Per job kind: (calls per unit, prompt tokens per call, output tokens per call); unit = one batch / one question / the whole job.
WORK = {"smoke": (2, 60, 10), "probe": (0, 40, 256), "thinkbench": (122, 1200, 80), "judgment": (5, 1200, 80), "drills": (4, 700, 180),
        "traces": (1, 350, 200)}
LOCAL_MIN = {"thinkbench": 2.5, "judgment": 0.5, "drills": 0.5, "traces": 0.5}   # PC-side work inside a job while the GPU waits (thinkbench:
# part a + the statistical predictions, measured 132 s at low priority on 3 Oct; the others: reading cases / git histories)
LOAD_MIN = 0.6                  # minutes to (re)start a server on another model (weights from the pod's disk)
SETUP_MIN = 3.0                 # ssh, probe, template check, dead-man switch
DOWNLOAD_MBPS = 100.0           # datacenter -> Hugging Face, MB/s (conservative)


def batch_gain(n: int) -> float:
    pts = list(BATCH_GAIN)
    if n <= pts[0][0]:
        return pts[0][1]
    for (a, ga), (b, gb) in zip(pts, pts[1:]):
        if n <= b:
            return ga + (gb - ga) * (n - a) / (b - a)
    return pts[-1][1]


def measured_tps(state: Optional[Path], model: str) -> Optional[float]:
    """The best aggregate tok/s a pulse-1 probe measured for this model (None before any probe)."""
    if state is None:
        return None
    best = None
    try:
        for ln in (Path(state) / "thinking" / "gpu_probe.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            if r.get("model") == model and r.get("best_tok_s"):
                best = max(best or 0.0, float(r["best_tok_s"]))
    except (OSError, ValueError):
        return None
    return best


def plan(cfg: Mapping[str, Any], jobs: Sequence[Any], state: Optional[Path] = None, include_setup: bool = True) -> dict[str, Any]:
    """Minutes and dollars per job of a job list - a dry run, nothing is contacted. Every model is served alone (the job's model), so each
    switch costs a server start; the first pulse also pays the model downloads."""
    rate = float(cfg.get("usd_per_hr", 0.343))
    rows: list[dict[str, Any]] = []
    served = ""
    cat = catalog() if state is not None else {k: dict(v) for k, v in CATALOG.items()}
    total = 0.0
    if include_setup:
        gb = sum(float(cat[m]["bytes"]) for m in dict.fromkeys(parse_job(j)["model"] for j in jobs) if m and m in cat) / 1e9
        mins = SETUP_MIN + gb * 1000 / DOWNLOAD_MBPS / 60 + gb / 0.5 / 60          # download + sha256 at ~0.5 GB/s
        rows.append({"job": "setup (probe pod, download + sha256 models, dead-man switch)", "minutes": round(mins, 1), "basis": f"{gb:.1f} GB models"})
        total += mins
    for spec in jobs:
        j = parse_job(spec)
        m = j["model"]
        if j["kind"] == "ext":
            mins = (LOAD_MIN if m and m != served else 0.0) + (min(j["minutes"], j["max_minutes"]) if j["max_minutes"] else j["minutes"])
            served = "" if j.get("free_gpu") else (m or served)
            rows.append({"job": j["spec"], "minutes": round(mins, 1), "usd": round(mins / 60 * rate, 3), "basis": f"declared by the job ({j['via']})"})
            total += mins
            continue
        slots, _ctx = served_slots(cfg, [m])[m] if m in cat else (8, 4096)
        single = SINGLE_TPS.get(m, 100.0)
        meas = measured_tps(state, m)
        agg = meas or single * batch_gain(slots)
        calls, tin, tout = WORK[j["kind"]]
        mins = LOAD_MIN if m != served else 0.0
        served = m
        basis = f"-np {slots}, {agg:.0f} tok/s aggregate ({'measured' if meas else 'assumed'})"
        if j["kind"] == "probe":
            levels = [c for c in (1, 4, 8, 16) if c <= slots] + ([slots] if slots not in (1, 4, 8, 16) else [])
            mins += sum(tout / (single * batch_gain(c) / c) for c in levels) / 60 + 0.2
            basis = f"concurrency {levels}, {tout} tokens each"
        elif j["kind"] in ("smoke", "thinkbench") or j["n"]:
            n_calls = calls * (j["n"] or 1)
            work = (n_calls * tout / agg + n_calls * tin / (single * PP_FACTOR)) / 60 + LOCAL_MIN.get(j["kind"], 0.0)
            mins += min(work, j["max_minutes"]) if j["max_minutes"] else work
            basis += f"; {n_calls} calls x ({tin} in, {tout} out)"
        else:
            mins += j["max_minutes"] or 60.0
            per_hr = 3600 / ((calls * tout / agg + calls * tin / (single * PP_FACTOR)) or 1)
            basis += f"; time-capped, ~{per_hr * (j['max_minutes'] or 60) / 60:.0f} units ({calls} calls each)"
        rows.append({"job": spec, "minutes": round(mins, 1), "usd": round(mins / 60 * rate, 3), "basis": basis})
        total += mins
    total += 1.0                                                     # teardown
    rows.append({"job": "teardown (stop servers, destroy)", "minutes": 1.0})
    for r in rows:
        r.setdefault("usd", round(float(r["minutes"]) / 60 * rate, 3))
    return {"jobs": rows, "minutes": round(total, 1), "usd": round(total / 60 * rate, 2), "usd_per_hr": rate,
            "budget_usd": float(cfg.get("budget_usd", 8.0))}


def format_plan(p: Mapping[str, Any]) -> str:
    out = [f"{'minutes':>8} {'USD':>7}  job  [basis]"]
    for r in p["jobs"]:
        out.append(f"{r['minutes']:>8.1f} {r['usd']:>7.3f}  {r['job']}" + (f"  [{r['basis']}]" if r.get("basis") else ""))
    out.append(f"{p['minutes']:>8.1f} {p['usd']:>7.2f}  TOTAL at ${p['usd_per_hr']}/h (budget ${p['budget_usd']:.2f}; storage + bandwidth extra, cents)")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ the GPU monitor (blueprint rule 2)
def monitor_script(ports: Sequence[int]) -> str:
    ps = " ".join(str(p) for p in ports)
    return f"""if command -v nvidia-smi >/dev/null 2>&1; then echo "@@gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1)"; fi
for p in {ps}; do echo "@@tok_$p=$(curl -s --max-time 5 http://127.0.0.1:$p/metrics 2>/dev/null | awk '/^llamacpp:tokens_predicted_total/ {{print $2}}' | head -1)"; done
"""


def sample_pod(sh: Shell, ports: Sequence[int]) -> tuple[Optional[float], Optional[float]]:
    """(GPU utilisation %, total tokens predicted by the servers so far); None where the pod did not say."""
    try:
        kv = _kv(sh.run(monitor_script(ports), timeout=30, check=False)[1])
    except (subprocess.SubprocessError, OSError):
        return None, None
    util: Optional[float] = None
    try:
        util = float(kv.get("gpu", "").split(",")[0])
    except ValueError:
        pass
    toks = [float(kv[k]) for k in kv if k.startswith("tok_") and kv[k].replace(".", "", 1).replace("e+", "", 1).isdigit()]
    return util, (sum(toks) if toks else None)


@dataclasses.dataclass
class Monitor:
    """Per-minute GPU utilisation and tok/s of a job. Below `low_pct` busy = the PC side is the bottleneck (too few requests in flight): flagged
    every minute, and the job is stopped after `abort_minutes` of it in a row (0 = never) - paid GPU time is never just waited out."""
    low_pct: float = 70.0
    abort_minutes: float = 8.0
    samples: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    low_since: Optional[float] = None
    last: Optional[tuple[float, float]] = None

    def add(self, t: float, util: Optional[float], tokens: Optional[float]) -> dict[str, Any]:
        tps = None
        if tokens is not None and self.last is not None and t > self.last[0]:
            tps = round(max(0.0, tokens - self.last[1]) / (t - self.last[0]), 1)
        if tokens is not None:
            self.last = (t, tokens)
        low = util is not None and util < self.low_pct
        self.low_since = (self.low_since if self.low_since is not None else t) if low else None
        s = {"t": round(t, 1), "util": util, "tok_s": tps, "low": low}
        self.samples.append(s)
        return s

    def should_abort(self, t: float) -> bool:
        return bool(self.abort_minutes) and self.low_since is not None and t - self.low_since >= self.abort_minutes * 60

    def summary(self) -> dict[str, Any]:
        u = [float(s["util"]) for s in self.samples if s["util"] is not None]
        r = [float(s["tok_s"]) for s in self.samples if s["tok_s"] is not None]
        return {"samples": len(self.samples), "util_mean": round(sum(u) / len(u), 1) if u else None, "tok_s_mean": round(sum(r) / len(r), 1) if r else None,
                "low_minutes": sum(1 for s in self.samples if s["low"]), "flag_low": any(s["low"] for s in self.samples)}


def _log(name: str, row: Mapping[str, Any]) -> None:
    p = outside_repo(gpu_dir() / name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row)) + "\n")


# ------------------------------------------------------------------------------------------------ a direct client of a pod server (smoke, probe, traces)
class PodLLM:
    """The request creator.generator.LocalModel.chat sends, straight to a forwarded pod port; the private-marker guard runs before every send.
    Context manager and `.model` / `.pulse` like a LocalModel, so judgment.chat_text and the record markers work unchanged."""

    def __init__(self, port: int, model: str, pulse: str) -> None:
        self.port, self.model, self.pulse = int(port), model, pulse
        self._tls = threading.local()                     # one kept-alive connection per thread (h52: a new connection per request cost
                                                          # ~1 s through the SSH tunnel; the pod answers a tiny chat in 0.27 s)

    def __enter__(self) -> "PodLLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._drop()

    def _drop(self) -> None:
        c = getattr(self._tls, "conn", None)
        self._tls.conn = None
        if c is not None:
            c.close()

    def _post(self, path: str, body: bytes, timeout: float) -> bytes:
        """POST on this thread's kept-alive connection; a connection the server closed while idle is reopened once (only a REUSED connection
        is retried: a fresh one's error is the server's answer). HTTP errors raise urllib.error.HTTPError as urlopen did."""
        import http.client
        import io
        for attempt in (0, 1):
            conn = getattr(self._tls, "conn", None)
            fresh = conn is None
            if conn is None:
                conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
                self._tls.conn = conn
            conn.timeout = timeout
            if conn.sock is not None:
                conn.sock.settimeout(timeout)
            try:
                conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
                r = conn.getresponse()
                data = r.read()
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest, http.client.BadStatusLine, ConnectionResetError,
                    ConnectionAbortedError, BrokenPipeError):
                self._drop()
                if fresh or attempt:
                    raise
                continue
            except BaseException:
                self._drop()
                raise
            if r.will_close:
                self._drop()
            if r.status >= 400:
                raise urllib.error.HTTPError(f"http://127.0.0.1:{self.port}{path}", r.status, r.reason, r.headers, io.BytesIO(data))
            return data
        raise PulseError("unreachable")

    def request(self, messages: Sequence[Mapping[str, str]], max_tokens: int = 400, temperature: float = 0.2, seed: int = 0,
                timeout: float = 300.0) -> dict[str, Any]:
        outbound_ok(messages)
        body = json.dumps({"messages": [dict(m) for m in messages], "max_tokens": max_tokens, "temperature": temperature, "seed": seed}).encode()
        t0 = time.monotonic()
        d = json.loads(self._post("/v1/chat/completions", body, timeout))
        return {"text": str(d["choices"][0]["message"].get("content") or ""), "tokens": int((d.get("usage") or {}).get("completion_tokens") or 0),
                "seconds": time.monotonic() - t0}

    def chat(self, messages: Sequence[Mapping[str, str]], max_tokens: int = 1500, temperature: float = 0.2, seed: int = 0,
             timeout: float = 600.0) -> str:
        return str(self.request(messages, max_tokens, temperature, seed, timeout)["text"])


def smoke(llm: PodLLM) -> dict[str, Any]:
    """The runner self-test (about a minute): the private-marker guard refuses, one known-answer request comes back right and fast."""
    try:
        outbound_ok([{"role": "user", "content": "Masterstock"}])
        guard = False
    except PulseError:
        guard = True
    from creator import generator as G
    r = llm.request(G.prepare_messages([{"role": "user", "content": "What is 17 + 25? Reply with only the number."}], llm.model),
                    max_tokens=60, temperature=0.0)
    ans = G.THINK_BLOCK.sub("", r["text"]).strip()
    return {"ok": guard and "42" in ans, "guard_refuses_private": guard, "answer": ans[:40], "tok_s": round(r["tokens"] / max(1e-6, r["seconds"]), 1)}


PROBE_PROMPT = "Write 30 short numbered facts about rivers, one per line."


def probe(llm: PodLLM, slots: int, state: Path, levels: Sequence[int] = (1, 4, 8, 16), max_tokens: int = 256) -> dict[str, Any]:
    """Slot-throughput probe: at each concurrency (up to the server's slots) that many identical-length requests at once; aggregate tok/s =
    tokens / wall. The best level is what later plans use (gpu_probe.jsonl) and the number of requests jobs keep in flight."""
    from creator import generator as G
    lv = sorted({c for c in levels if c <= slots} | {slots})
    msgs = G.prepare_messages([{"role": "user", "content": PROBE_PROMPT}], llm.model)
    curve: dict[str, float] = {}
    for c in lv:
        res: list[dict[str, Any]] = []
        lock = threading.Lock()

        def one(i: int) -> None:
            r = llm.request(msgs, max_tokens=max_tokens, temperature=0.8, seed=i)
            with lock:
                res.append(r)
        t0 = time.monotonic()
        ts = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(c)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        curve[str(c)] = round(sum(int(r["tokens"]) for r in res) / max(1e-6, time.monotonic() - t0), 1)
    best = max(curve, key=lambda k: curve[k])
    row = {"model": Path(llm.model).name, "gpu_pulse": llm.pulse, "slots": slots, "tok_s_by_concurrency": curve, "best_concurrency": int(best),
           "best_tok_s": curve[best], "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    p = Path(state) / "thinking" / "gpu_probe.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def traces(llm: Any, state: Path, repo: Path, n: int, workers: int, deadline: float,
           questions: Optional[Callable[[], Sequence[Any]]] = None) -> dict[str, Any]:
    """Pulse 4: the strong model answers fresh reasoning questions with SHORT worked reasoning; every reply is scored against the known answer
    AFTER it exists and only correct, outcome-blind traces (the prompt carried no answer and no example) go to the bank
    (creator.reasondrills.bank_add) that the CPU model's retrieval strategies show as worked examples. Frozen-benchmark questions never enter
    (reasondrills.generate drops every collision)."""
    from creator import judgment as J
    from creator import reasondrills as R
    have = set(R.trace_bank(state))
    qs: list[Any] = []
    feed: list[Any] = []                              # questions not yet handed to a worker
    st = R.Strategy("trace", "cot", max_tokens=R.TRACE_TOKENS)
    stats = {"asked": 0, "correct": 0, "kept": 0, "errors": 0}
    lock = threading.Condition()
    flow = {"queued": 0, "done": False}

    def add(part: Sequence[Any]) -> None:
        """Questions arrive in pieces (cached repositories at once, the others as they are generated): the GPU starts on the first piece instead of
        waiting minutes for every git history. One piece = the old order (reasondrills.order) of that piece."""
        with lock:
            qs.extend(part)
            for q in part:
                if q.qid not in have and R.REVISIT not in q.qid and (not n or flow["queued"] < n):
                    feed.append(q)
                    flow["queued"] += 1
            lock.notify_all()

    def produce() -> None:
        try:
            if questions is not None:
                add(list(questions()))
            else:
                for part in R.generate_stream(R.repos(Path(repo)), Path(state)):
                    add(R.order(part))
                    if time.monotonic() >= deadline:
                        break
        finally:
            with lock:
                flow["done"] = True
                lock.notify_all()
    producer = threading.Thread(target=produce, daemon=True, name="traces-questions")
    producer.start()
    pos = {"i": 0}

    def work() -> None:
        while time.monotonic() < deadline:
            with lock:
                while pos["i"] >= len(feed) and not flow["done"] and time.monotonic() < deadline:
                    lock.wait(timeout=1.0)
                if pos["i"] >= len(feed):
                    return
                q = feed[pos["i"]]
                pos["i"] += 1
            msgs = R.build_messages(st, q)
            if any("Correct answer" in m["content"] for m in msgs):          # outcome-blind by construction; checked anyway
                continue
            try:
                reply = J.chat_text(llm, msgs, max_tokens=st.max_tokens, temperature=0.2, seed=0, timeout=300.0)
            except Exception:                                                # noqa: BLE001 - one failed call is one missing trace
                with lock:
                    stats["errors"] += 1
                continue
            pick = R.parse_choice(reply)
            with lock:
                stats["asked"] += 1
                stats["correct"] += int(pick == q.answer)
            if pick == q.answer and R.trace_ok(reply):
                R.bank_add(state, q, reply, str(Path(str(getattr(llm, "model", ""))).name), str(getattr(llm, "pulse", "") or ""))
                with lock:
                    stats["kept"] += 1
    ts = [threading.Thread(target=work, daemon=True) for _ in range(max(1, workers))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    with lock:                                        # the GPU work is over: never hold the pod for questions nobody will ask
        stats["fresh_left"] = max(0, len([q for q in qs if q.qid not in have]) - stats["asked"] - stats["errors"])
        if not flow["done"]:
            stats["fresh_left_partial"] = True            # repositories still being read: a lower bound
    return stats


def run_job_here(job: Mapping[str, Any], state: Path, repo: Path, owner_dir: Path, workers: int) -> dict[str, Any]:
    """Inside the child: the pulse is on (env), so G.thinker()/LocalModel attach to the pod. Results go to the normal state files."""
    t0 = time.monotonic()
    deadline = time.monotonic() + (float(job.get("max_minutes") or 0) * 60 or 10 * 3600)
    pf = Path(os.environ["NUPEN_GPU_PULSE"])
    t = _tunnel(pf)
    pulse = str(t.get("pulse") or os.environ.get("NUPEN_PULSE_ID") or "pulse")
    if job["kind"] == "ext":
        import importlib
        mod, _, fn = str(job["call"]).partition(":")
        ctx = {"state": Path(state), "repo": Path(repo), "owner_dir": Path(owner_dir), "workers": workers, "deadline": deadline, "pulse": pulse,
               "tunnel_file": pf, "model": job.get("model", ""), "endpoints": endpoints(pf), "args": dict(job.get("args") or {})}
        r = getattr(importlib.import_module(mod), fn)(ctx)
        return dict(r or {}, seconds=round(time.monotonic() - t0, 1))
    if job["kind"] in ("smoke", "probe", "traces"):
        got = attach(pf, job["model"])
        if got is None:
            raise PulseError(f"the pulse does not serve {job['model']}")
        llm = PodLLM(got[0], str(job["model"]), got[1])
        if job["kind"] == "smoke":
            return dict(smoke(llm), seconds=round(time.monotonic() - t0, 1))
        if job["kind"] == "probe":
            slots = int((t.get("slots") or {}).get(job["model"]) or workers) * len(t["models"][job["model"]])
            return dict(probe(llm, slots, state), seconds=round(time.monotonic() - t0, 1))
        return dict(traces(llm, state, repo, int(job.get("n") or 0), workers, deadline), seconds=round(time.monotonic() - t0, 1))
    if job["kind"] == "thinkbench":
        from creator import thinkbench as TB
        TB.WORKER_CAP = workers                           # the pod serves `workers` slots: keep that many requests in flight (CPU: 2)
        frozen = TB.freeze(state, repo, owner_dir)
        safe = dict(frozen, d=[])                         # part d carries owner directives (~/Masterstock): never sent to the pod
        res = TB.run(state, repo, owner_dir, safe, workers=workers, model="thinker")
        res["gpu_pulse"] = {"pulse": pulse, "model": job["model"], "skipped": "d_planning (owner directives stay on this PC)"}
        p = TB.save(state, res, "pulse")
        bases = sorted(TB.bench_dir(state).glob("baseline_*.json"))
        cmp = TB.compare(json.loads(bases[0].read_text(encoding="utf-8")), res) if bases else None
        return {"file": str(p), "runtime_s": res.get("runtime_s"), "parts": {k: v.get("score") for k, v in res["parts"].items()},
                "vs_baseline": {k: v.get("verdict") for k, v in (cmp or {}).get("parts", {}).items()}, "seconds": round(time.monotonic() - t0, 1)}
    if job["kind"] == "judgment":
        from creator import judgment as J
        n = _drain(J.judgment_filler(state, repo, max_servers=workers), int(job.get("n") or 0), workers, deadline)
        return {"batches": n, "seconds": round(time.monotonic() - t0, 1)}
    from creator import reasondrills as R
    n = _drain(R.reasoning_filler(state, repo, max_servers=workers), int(job.get("n") or 0), workers, deadline)
    return {"batches": n, "seconds": round(time.monotonic() - t0, 1)}


def fetch_outputs(cfg: Mapping[str, Any], sh: Shell, paths: Sequence[str], dest: Path) -> Path:
    """Copy pod files home (tar over the same SSH channel; nothing else is opened) and unpack them into `dest` with tarfile's 'data' filter
    (no absolute paths, no links out of `dest`)."""
    import io
    import tarfile
    rdir = shlex.quote(str(cfg["remote_dir"]))
    rc, data = sh.run_bytes(f"cd {rdir} && tar -czf - -- {' '.join(shlex.quote(x) for x in paths)}", timeout=float(cfg.get("fetch_timeout_s", 3600)))
    if rc != 0 or not data:
        raise PulseError(f"could not copy {list(paths)} home (rc {rc})")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest, filter="data")
    return dest


def _record(state: Path, row: Mapping[str, Any]) -> None:
    """Every job's result in Nupen's state (thinking/gpu_pulse_runs.jsonl) as soon as it finishes: a cut-off pulse loses at most the job in flight."""
    p = Path(state) / "thinking" / "gpu_pulse_runs.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row)) + "\n")


def run_jobs(cfg: Mapping[str, Any], jobs: Sequence[Any], state: Path, repo: Path, owner_dir: Path, workers: int = 0,
             shell: Optional[Shell] = None, tunnel: Optional[Callable[[Mapping[str, Sequence[int]]], Path]] = None,
             say: Callable[[str], None] = print, clock: Callable[[], float] = time.time, poll_s: float = 2.0) -> list[dict[str, Any]]:
    """Each job in order: its model served (others stopped), the tunnel opened, the job run in a child process with the pulse switched on and
    inflight_factor x the server's slots requests in flight (`workers` > 0 overrides). The budget is checked before every job and every `poll_s`
    while one runs (at the cap the child is stopped and the run ends); every `monitor_s` the pod's GPU utilisation and tok/s are logged, and a
    job that keeps the GPU under `low_util_pct` for `low_util_abort_minutes` in a row is stopped with the reason (the next job still runs)."""
    budget = budget_for(cfg, clock)
    rate, reserve = float(cfg["usd_per_hr"]), float(cfg.get("reserve_minutes", 3.0)) * 60
    pulse = budget.begin(rate, str(cfg.get("instance_id", "")))
    sh = shell or shell_for(cfg)
    parsed = [parse_job(j) for j in jobs]
    out: list[dict[str, Any]] = []
    served: Optional[str] = None
    pf: Optional[Path] = None
    plan_: dict[str, list[int]] = {}
    for j in parsed:
        try:
            budget.check(rate, reserve)
        except BudgetExceeded as e:
            say(str(e))
            out.append({"job": j["spec"], "stopped": "budget"})
            break
        if j.get("free_gpu"):                             # an external job that needs all the VRAM (training): no server, no tunnel
            sh.run(stop_servers_script(str(cfg["remote_dir"])), timeout=60, check=False)
            close_tunnel()
            served, pf, plan_ = None, None, {}
        elif j["model"] and j["model"] != served:
            try:
                plan_ = serve(cfg, [j["model"]], sh, say=say)
            except BudgetExceeded:
                raise
            except PulseError as e:                       # e.g. a tuned model whose fine-tune skipped itself: this job only, the day goes on
                say(f"job {j['spec']} skipped: {j['model']} could not be served: {e}")
                res0 = {"job": j["spec"], "skipped": f"model not served: {e}", "gpu_pulse": pulse["id"], "usd_spent_total": budget.spent()}
                out.append(res0)
                _record(state, res0)
                served, pf, plan_ = None, None, {}
                continue
            pf = (tunnel or (lambda pl: open_tunnel(cfg, pl, pulse["id"])))(plan_)
            served = j["model"]
        slots = served_slots(cfg, [j["model"]])[j["model"]][0] * max(1, int(cfg.get("instances", 1))) if j["model"] else 1
        w = workers or inflight(cfg, slots)
        env = dict(os.environ, NUPEN_PULSE_MODEL=j["model"], NUPEN_PULSE_SLOTS=str(slots), NUPEN_PULSE_ID=str(pulse["id"]))
        if pf is not None:
            env["NUPEN_GPU_PULSE"] = str(pf)
        elif j["kind"] != "ext" or j["via"] == "call":
            pf = write_tunnel({}, str(pulse["id"]), path=gpu_dir() / "tunnel_none.json")    # a job without a server still runs 'pulse on'
            env["NUPEN_GPU_PULSE"] = str(pf)
        argv = [sys.executable, "-m", "creator.gpupulse", "job", json.dumps(j), "--state", str(state), "--repo", str(repo),
                "--owner-dir", str(owner_dir), "--workers", str(w)]
        stdin_script: Optional[bytes] = None
        if j["kind"] == "ext" and j["via"] == "command":
            argv = [str(a) for a in j["command"]]
        elif j["kind"] == "ext" and j["via"] == "remote":
            outbound_ok([{"content": str(j["remote"])}])
            argv, stdin_script = list(sh.argv), f"cd {shlex.quote(str(cfg['remote_dir']))} || exit 9\n{j['remote']}\n".encode("utf-8")
        say(f"job {j['spec']} ({w} requests in flight) ...")
        t0 = clock()
        mon = Monitor(float(cfg.get("low_util_pct", 70.0)), float(j.get("low_util_abort_minutes", cfg.get("low_util_abort_minutes", 8))))
        every = float(cfg.get("monitor_s", 60.0))
        ports = [p for ps in plan_.values() for p in ps]
        next_mon = clock()
        proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.PIPE if stdin_script is not None else None)
        if stdin_script is not None and proc.stdin is not None:
            proc.stdin.write(stdin_script)
            proc.stdin.close()
        text_parts: list[bytes] = []
        reader = threading.Thread(target=lambda: text_parts.append(proc.stdout.read() if proc.stdout else b""), daemon=True)
        reader.start()                                    # drained on a thread: a chatty child never blocks on a full pipe
        stopped = ""
        while proc.poll() is None:
            now = clock()
            try:
                budget.check(rate, reserve)
            except BudgetExceeded as e:
                say(f"{e}: stopping job {j['spec']}")
                stopped = "budget"
            if not stopped and every > 0 and now >= next_mon:
                next_mon = now + every
                s = mon.add(now - t0, *sample_pod(sh, ports))
                _log("monitor.jsonl", dict(s, pulse=pulse["id"], job=j["spec"]))
                if s["util"] is not None or s["tok_s"] is not None:
                    say(f"[monitor] {j['spec']}: GPU {s['util']}% busy, {s['tok_s']} tok/s" +
                        ("  LOW: the PC side is the bottleneck (raise --workers / slots)" if s["low"] else ""))
                if mon.should_abort(now - t0):
                    say(f"stopping job {j['spec']}: GPU under {mon.low_pct:.0f}% busy for {mon.abort_minutes:.0f} min (requests in flight {w}; "
                        f"the job cannot keep the GPU fed - check the job's own pace or raise --workers)")
                    stopped = "low_gpu_utilisation"
            if stopped:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            time.sleep(poll_s)
        reader.join(timeout=30)
        text = b"".join(text_parts).decode("utf-8", "replace")
        res: dict[str, Any] = {"job": j["spec"], "rc": proc.returncode, "wall_s": round(clock() - t0, 1), "usd_spent_total": budget.spent(),
                               "gpu_pulse": pulse["id"], "workers": w, "monitor": mon.summary()}
        for ln in text.splitlines():
            if ln.startswith("@@result="):
                res["result"] = json.loads(ln[len("@@result="):])
        if stopped:
            res["stopped"] = stopped
        if proc.returncode not in (0, None) and not stopped:
            res["tail"] = text[-800:]
        if j["kind"] == "ext" and j["via"] != "call" and "result" not in res:
            res["tail"] = text[-800:]
        if j.get("outputs"):
            try:
                res["outputs_home"] = str(fetch_outputs(cfg, sh, j["outputs"], Path(str(j.get("home") or gpu_dir() / "outputs" / pulse["id"] / j["name"]))))
            except PulseError as e:
                res["outputs_error"] = str(e)
        out.append(res)
        _record(state, res)
        say(json.dumps(res)[:600])
        if stopped == "budget":
            break
    return out


# ------------------------------------------------------------------------------------------------ prepare: everything staged before renting
def prepare(cfg: Mapping[str, Any], jobs: Sequence[Any], state: Path, repo: Path,
            questions: Optional[Callable[[], Sequence[Any]]] = None) -> dict[str, Any]:
    """Everything a pulse needs that can be done on the PC is done (or checked) BEFORE renting, so no paid minute goes to it: every model has a
    pinned sha256, the frozen benchmark exists, the reasoning questions are generated from the public repositories (and scanned by the private-
    marker guard), the SSH key and the ssh client are there, the local tunnel ports are free; the plan is written beside the config."""
    import shutil
    import socket
    checks: dict[str, Any] = {}
    parsed = [parse_job(j) for j in jobs]
    cat = catalog()
    models = list(dict.fromkeys(j["model"] for j in parsed if j["model"]))
    extra = extra_models(cfg)
    later = {str(j["args"]["serve_as"]) for j in parsed if isinstance(j.get("args"), Mapping) and j["args"].get("serve_as")}
    checks["models_pinned"] = {m: bool((m in cat and cat[m].get("sha256")) or m in extra or m in later) for m in models}
    checks["models_registered_mid_run"] = sorted(m for m in models if m in later and m not in extra)
    missing_cfg = [m for m in models if m not in list(cfg.get("models") or []) and m not in extra and m not in later]
    checks["models_in_config"] = not missing_cfg
    key = Path(str(cfg.get("key_path") or "~/.ssh/nupen_vast")).expanduser()
    checks["ssh_key"] = key.is_file() and key.with_suffix(key.suffix + ".pub").is_file()
    checks["ssh_client"] = bool(shutil.which(str(cfg.get("ssh", "ssh"))))
    busy = []
    for rp in [x for ps in plan_ports(cfg, models).values() for x in ps]:
        lp = int(cfg.get("local_port_base", 18100)) + rp - int(cfg.get("remote_port_base", 18100))
        with socket.socket() as so:
            if so.connect_ex(("127.0.0.1", lp)) == 0:
                busy.append(lp)
    checks["local_ports_free"] = not busy
    if any(j["kind"] == "thinkbench" for j in parsed):
        checks["thinkbench_frozen"] = (Path(state) / "thinkbench" / "items.json").is_file()
    if any(j["kind"] in ("drills", "traces") for j in parsed):
        from creator import reasondrills as R
        qs = list(questions()) if questions is not None else R.generate(R.repos(Path(repo)), Path(state))
        bad = 0
        for q in qs:
            try:
                outbound_ok(R.build_messages(R.BY_NAME[R.CONTROL], q))
            except PulseError:
                bad += 1
        have = set(R.trace_bank(state))
        checks["reasoning_questions"] = {"total": len(qs), "not_yet_in_bank": sum(1 for q in qs if q.qid not in have), "private_refused": bad}
    if any(j["kind"] == "judgment" for j in parsed):
        from creator import judgment as J
        checks["judgment_cases"] = {t: len(J.load_cases(t, Path(state), Path(repo))) for t in J.ALL_TOPICS}
    p = plan(cfg, jobs, state)
    blockers = [m for m, ok in checks["models_pinned"].items() if not ok]
    out = {"jobs": [j["spec"] for j in parsed], "checks": checks, "plan": p, "missing_from_config": missing_cfg,
           "ready": not blockers and not missing_cfg and checks["ssh_client"] and checks["local_ports_free"] and checks.get("thinkbench_frozen", True),
           "blockers": [f"no pinned sha256: {m}" for m in blockers] + ([f"add to config 'models': {missing_cfg}"] if missing_cfg else [])
           + ([] if checks["ssh_client"] else ["no ssh client on PATH"]) + ([] if checks["local_ports_free"] else [f"local ports in use: {busy}"])
           + ([] if checks.get("thinkbench_frozen", True) else ["thinkbench not frozen (python scripts/thinkbench.py --freeze)"]),
           "owner_todo": ([] if checks["ssh_key"] else [f"ssh-keygen -t ed25519 -f {key} -C nupen-gpu-pulse  (then paste {key}.pub in the Vast.ai console)"])
           + ([] if cfg.get("host") else ["after renting: put host / port / instance_id in " + str(cfg.get("_path", config_path()))]),
           "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    _write_json(gpu_dir() / "prepared.json", out)
    return out


def _job_main(argv: Sequence[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--state", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--owner-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(list(argv))
    if not os.environ.get("NUPEN_GPU_PULSE"):
        print("job runs only with the pulse switched on (env NUPEN_GPU_PULSE)")
        return 2
    res = run_job_here(json.loads(a.job), Path(a.state), Path(a.repo), Path(a.owner_dir), a.workers)
    print("@@result=" + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        raise SystemExit(_job_main(sys.argv[2:]))
    raise SystemExit("usage: python -m creator.gpupulse job <json> --state S --repo R --owner-dir O (scripts/gpu_pulse.py is the front door)")
