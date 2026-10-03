"""What machine is Nupen running on, and what settings follow from it (owner, 2 Oct 2026: "if there is anything that is device
specific we need to figure out how to change the system to make it whatever device is being operated on").

Facts are detected once and cached (`get()`); settings are DERIVED from them (`derive(dev)`) and may be pinned by the owner in
state/creator/device_overrides.json. Detection never imports torch and never starts a model. Every reader is injectable so the
derivations are testable for machines that are not this one.

Paths: the runtime dir (models, llama.cpp, LM data) is env NUPEN_RUNTIME, else the older CREATOR_RUNTIME, else ~/creator_runtime.
The sandbox root is env NUPEN_SANDBOXES, else <repo parent>/.<repo name>_creator_sandboxes (= ~/.weekly7_creator_sandboxes here).
"""
from __future__ import annotations

import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state" / "creator"
SNAPSHOT = STATE_DIR / "device.json"
OVERRIDES = STATE_DIR / "device_overrides.json"
MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_GB = 1.1                                       # that file's size; a bigger model passes its own size to derive()

# Ratios that reproduce the settings the Creator ran with on the 16.8 GB / 8-core development machine.
SLOT_GB = 3.8                # RAM per parallel test slot (16.8 GB -> 4: the machine-wide budget of creator.testslots; one source of truth)
TEST_MB = 120.0              # per-test-process MB assumed until a real launch has been measured (creator.testslots reads this)
WORKER_PER_GB = 1.9          # governor ceiling on concurrent workers per GB of RAM (16.8 GB -> 32); the RAM floor is what really limits
LM_FREE_FRACTION = 0.10      # free RAM the LM trainer needs before it starts, as a share of total ...
LM_FREE_MIN_GB = 3.0         # ... but never less than this
GPU_HEADROOM_GB = 1.0        # VRAM that must remain after the model is loaded


@dataclasses.dataclass(frozen=True)
class Device:
    os: str                  # 'Windows' | 'Linux' | 'Darwin'
    arch: str                # normalised: 'x64' | 'arm64' | other
    cores_logical: int
    cores_physical: int
    ram_gb: float
    gpu_name: str            # '' when there is no usable NVIDIA GPU
    vram_gb: float
    python: str
    home: str
    runtime_dir: str
    sandbox_root: str
    user: str = ""


# ---------------------------------------------------------------------------------------------------------- paths

def home() -> Path:
    return Path.home()


def runtime_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    e = os.environ if env is None else env
    v = e.get("NUPEN_RUNTIME") or e.get("CREATOR_RUNTIME")
    return Path(v) if v else home() / "creator_runtime"


def sandbox_root(repo: Optional[Path] = None, env: Optional[Mapping[str, str]] = None) -> Path:
    e = os.environ if env is None else env
    v = e.get("NUPEN_SANDBOXES")
    if v:
        return Path(v)
    if repo is not None:
        return Path(repo).resolve().parent / f".{Path(repo).resolve().name}_creator_sandboxes"
    return home() / ".weekly7_creator_sandboxes"


def is_windows() -> bool:
    return sys.platform == "win32"


def exe_name(base: str) -> str:
    return base + (".exe" if is_windows() else "")


def server_exe(rt: Optional[Path] = None) -> Path:
    return (rt or runtime_dir()) / "llama" / exe_name("llama-server")


def model_path(rt: Optional[Path] = None) -> Path:
    return (rt or runtime_dir()) / "models" / MODEL_FILE


def venv_python(venv: Path, windowless: bool = False) -> Path:
    """The interpreter inside a venv, whatever the OS lays out (Scripts/python.exe, Scripts/pythonw.exe, bin/python)."""
    if is_windows():
        return venv / "Scripts" / ("pythonw.exe" if windowless else "python.exe")
    return venv / "bin" / "python"


def lm_python(rt: Optional[Path] = None) -> Path:
    return venv_python((rt or runtime_dir()) / "lmenv")


# ---------------------------------------------------------------------------------------------------------- detection

def _arch(machine: str) -> str:
    m = machine.lower()
    if m in ("amd64", "x86_64", "x64"):
        return "x64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    return m or "unknown"


def read_ram_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().total) / 1e9
    except Exception:                                    # noqa: BLE001 - psutil missing: fall back to the OS's own report
        pass
    try:
        for ln in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ln.startswith("MemTotal:"):
                return float(ln.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    try:
        if is_windows():
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("aev", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(ms)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))      # type: ignore[attr-defined]
            return float(ms.total) / 1e9
        return float(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip()) / 1e9
    except Exception:                                    # noqa: BLE001
        return 0.0


def read_physical_cores(logical: int) -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:                                    # noqa: BLE001
        pass
    return max(1, logical)


def parse_nvidia_smi(text: str) -> tuple[str, float]:
    """Best GPU from `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` (MiB). ('', 0.0) when none."""
    best, best_gb = "", 0.0
    for ln in (text or "").splitlines():
        name, _, mem = ln.rpartition(",")
        try:
            gb = float(mem.strip()) * 1048576 / 1e9
        except ValueError:
            continue
        if name.strip() and gb > best_gb:
            best, best_gb = name.strip(), gb
    return best, round(best_gb, 2)


def read_gpu() -> tuple[str, float]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", 0.0
    try:
        p = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "", 0.0
    return parse_nvidia_smi(p.stdout) if p.returncode == 0 else ("", 0.0)


def detect(*, system: Callable[[], str] = platform.system, machine: Callable[[], str] = platform.machine,
           logical: Callable[[], Optional[int]] = os.cpu_count, physical: Callable[[int], int] = read_physical_cores,
           ram_gb: Callable[[], float] = read_ram_gb, gpu: Callable[[], tuple[str, float]] = read_gpu,
           env: Optional[Mapping[str, str]] = None) -> Device:
    lg = int(logical() or 1)
    try:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    except Exception:                                    # noqa: BLE001
        user = ""
    name, vram = gpu()
    return Device(os=system(), arch=_arch(machine()), cores_logical=lg, cores_physical=max(1, int(physical(lg))),
                  ram_gb=round(float(ram_gb()), 2), gpu_name=name, vram_gb=vram, python=sys.executable, home=str(home()),
                  runtime_dir=str(runtime_dir(env)), sandbox_root=str(sandbox_root(None, env)), user=user)


_CACHE: dict[str, Device] = {}


def get(refresh: bool = False) -> Device:
    """The facts about this machine, detected once per process."""
    if refresh or "dev" not in _CACHE:
        _CACHE["dev"] = detect()
    return _CACHE["dev"]


# ---------------------------------------------------------------------------------------------------------- derivation

def lm_env_has_cuda(rt: Optional[Path] = None) -> bool:
    """Is the LM env's torch a CUDA build? Read from its version file - torch itself is never imported here (RAM)."""
    base = (rt or runtime_dir()) / "lmenv"
    for ver in list(base.glob("Lib/site-packages/torch/version.py")) + list(base.glob("lib/python*/site-packages/torch/version.py")):
        try:
            if "+cu" in ver.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def derive(dev: Device, *, model_gb: float = MODEL_GB, lm_cuda: Optional[bool] = None,
           overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Every setting that depends on the machine. `overrides` (state/creator/device_overrides.json) win over what is derived."""
    reserve = 2 if dev.cores_physical <= 8 else 1       # a small machine keeps two cores for the owner (8 cores -> 6 threads)
    gpu_ok = bool(dev.gpu_name) and dev.vram_gb >= model_gb + GPU_HEADROOM_GB
    cuda = bool(dev.gpu_name) and (lm_env_has_cuda() if lm_cuda is None else lm_cuda)
    slots = max(1, min(dev.cores_logical, int(dev.ram_gb / SLOT_GB)))
    out: dict[str, Any] = {
        "llama_threads": max(1, dev.cores_physical - reserve),
        "gpu_layers": 99 if gpu_ok else 0,
        "test_slots": slots,
        "max_workers": max(1, round(dev.ram_gb * WORKER_PER_GB)),
        "governor_floor_fraction": 0.07,                 # shares of THIS machine's RAM; the fractions are the machine-independent part
        "governor_floor_min_gb": 0.8,
        "user_aware": True,                              # keep 25% free while the owner is at the keyboard (False: never yield)
        "lm_threads": max(1, dev.cores_physical // 2),
        "lm_min_free_gb": round(max(LM_FREE_MIN_GB, LM_FREE_FRACTION * dev.ram_gb), 1),
        "torch_device": "cuda" if cuda else "cpu",
        "disk_floor_gb": 1.0,                            # never start new work with less free disk than this
        "llama_servers": 1,                              # local model servers side by side (always-on: from RAM, below)
    }
    ov = dict(load_overrides() if overrides is None else overrides)
    for k, v in ov.items():
        if k in out:
            out[k] = v
    target = ov.get("ram_use_target_gib")
    if target:
        # owner, 2 Oct 2026: "we want to use 29 GB 24/7 you as claude get to use what you need then Nupen uses the rest" -
        # GB as Task Manager shows it (GiB). Everything running (Nupen, Claude, the rest) may fill RAM up to the target; the
        # free floor is what remains of THIS machine. Free RAM is measured, so whatever Claude uses is simply not Nupen's.
        out["governor_floor_fraction"] = 0.0
        out["governor_floor_min_gb"] = max(0.5, round(dev.ram_gb - float(target) * 1.073741824, 2))
    if out["user_aware"] is False and "llama_servers" not in ov:
        # ~2 GB per server (1.1 GB model + context); up to 60% of RAM, never more servers than physical cores
        out["llama_servers"] = max(1, min(dev.cores_physical, int(dev.ram_gb * 0.6 / 2.0)))
    if out["user_aware"] is False and "test_slots" not in ov:
        # owner, 2 Oct 2026 (new PC): "use all of it except a single GB" - with no yielding, the per-slot RAM guess is not the
        # limit; every slot is still admitted only while measured free RAM minus the floor holds one more test (testslots)
        out["test_slots"] = max(1, dev.cores_logical)
    return out


def load_overrides(path: Optional[Path] = None) -> dict[str, Any]:
    try:
        v = json.loads((path or OVERRIDES).read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def settings(refresh: bool = False) -> dict[str, Any]:
    return derive(get(refresh))


def snapshot(dev: Optional[Device] = None, derived: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    d = dev or get()
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": dataclasses.asdict(d),
            "settings": dict(derived if derived is not None else derive(d)), "overrides": load_overrides()}


def write_snapshot(path: Optional[Path] = None, *, log: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Write state/creator/device.json; if the machine differs from the last snapshot, say so through `log`. Never raises."""
    snap = snapshot()
    p = path or SNAPSHOT
    try:
        old = json.loads(p.read_text(encoding="utf-8")).get("device", {})
    except (OSError, ValueError, AttributeError):
        old = {}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, indent=1, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    if log is not None:
        keys = ("os", "arch", "cores_physical", "ram_gb", "gpu_name", "vram_gb")
        changed = [k for k in keys if old and old.get(k) != snap["device"][k]]
        if changed:
            log("device changed since the last start: " + ", ".join(f"{k} {old.get(k)!r} -> {snap['device'][k]!r}" for k in changed))
        log(f"device: {snap['device']['os']}/{snap['device']['arch']} {snap['device']['cores_physical']} cores "
            f"{snap['device']['ram_gb']} GB gpu={snap['device']['gpu_name'] or 'none'}; settings {snap['settings']}")
    return snap


def idle_seconds_unix() -> float:
    """Seconds since the owner's last input on Linux (xprintidle) or macOS (ioreg). 0.0 = unknown (assume the owner is there)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True, timeout=5).stdout
            for ln in out.splitlines():
                if "HIDIdleTime" in ln:
                    return int(ln.split("=")[-1].strip()) / 1e9
            return 0.0
        exe = shutil.which("xprintidle")
        if exe:
            return int(subprocess.run([exe], capture_output=True, text=True, timeout=5).stdout.strip()) / 1000.0
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 0.0


if __name__ == "__main__":
    s = snapshot()
    print(json.dumps(s, indent=1, sort_keys=True))
