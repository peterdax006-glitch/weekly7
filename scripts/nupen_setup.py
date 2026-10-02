"""Set Nupen up on a NEW machine with one command (owner, 2 Oct 2026: the system must adapt to whatever device it runs on).

    python scripts/nupen_setup.py --dry-run            # print every step for THIS machine, change nothing
    python scripts/nupen_setup.py                      # do it; downloads are listed with sizes and need a yes (or --yes)
    python scripts/nupen_setup.py --autostart          # also enable start-at-logon (otherwise the entry is only written)

Steps: (1) .venv from requirements.txt; (2) the LM env with the right torch build (CUDA when an NVIDIA GPU is present);
(3) the matching llama.cpp release for this OS / CPU architecture / GPU; (4) the model file creator/generator.py expects;
(5) autostart (HKCU Run on Windows, systemd user unit on Linux, launchd agent on macOS) - written, enabled only with --autostart;
(6) scripts/nupen_doctor.py. It may use the network. Nothing here is run by the Creator itself; the owner runs it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creator import device as DEV  # noqa: E402

LLAMA_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/" + DEV.MODEL_FILE
MODEL_APPROX_GB = DEV.MODEL_GB
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"


@dataclasses.dataclass
class Step:
    name: str
    what: str                                   # one line a human can check
    download: str = ""                          # URL when the step fetches something big (asks for confirmation)
    run: Optional[Callable[[], None]] = None    # None = informational only


def llama_asset_patterns(os_name: str, arch: str, gpu: bool) -> list[str]:
    """Regexes (best first) for the llama.cpp release asset that fits this machine."""
    if os_name == "Windows":
        a = "arm64" if arch == "arm64" else "x64"
        pats = [rf"bin-win-cuda-[\d.]+-x64\.zip$"] if (gpu and a == "x64") else []
        return pats + [rf"bin-win-cpu-{a}\.zip$"] + ([r"bin-win-vulkan-x64\.zip$"] if gpu else [])
    if os_name == "Darwin":
        return [rf"bin-macos-{'arm64' if arch == 'arm64' else 'x64'}\.(zip|tar\.gz)$"]
    a = "arm64" if arch == "arm64" else "x64"
    return ([r"bin-ubuntu-vulkan-x64\.(zip|tar\.gz)$"] if (gpu and a == "x64") else []) + [rf"bin-ubuntu-{a}\.(zip|tar\.gz)$"]


def pick_asset(assets: Sequence[dict[str, Any]], patterns: Sequence[str]) -> Optional[dict[str, Any]]:
    for pat in patterns:
        for a in assets:
            if re.search(pat, str(a.get("name", ""))):
                return a
    return None


def torch_command(venv_py: Path, cuda: bool) -> list[str]:
    return [str(venv_py), "-m", "pip", "install", "torch", "numpy", "--index-url", TORCH_CUDA_INDEX if cuda else TORCH_CPU_INDEX]


def _sh(cmd: Sequence[str]) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}\n    -> {dest}", flush=True)
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:      # noqa: S310 - fixed https URLs above
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _install_llama(dev: DEV.Device, rt: Path) -> None:
    with urllib.request.urlopen(LLAMA_API) as r:                       # noqa: S310
        rel = json.load(r)
    asset = pick_asset(rel.get("assets", []), llama_asset_patterns(dev.os, dev.arch, bool(dev.gpu_name)))
    if asset is None:
        raise SystemExit(f"no llama.cpp asset for {dev.os}/{dev.arch}; download one by hand into {rt / 'llama'}")
    print(f"  asset {asset['name']} ({asset.get('size', 0) / 1e6:.0f} MB) {asset['browser_download_url']}")
    z = rt / ("llama" + (".zip" if asset["name"].endswith(".zip") else ".tar.gz"))
    _fetch(asset["browser_download_url"], z)
    import shutil
    shutil.unpack_archive(str(z), str(rt / "llama"))
    for exe in (rt / "llama").rglob(DEV.exe_name("llama-server")):    # releases may nest one folder: flatten to runtime/llama/
        if exe.parent != rt / "llama":
            for f in exe.parent.iterdir():
                f.replace(rt / "llama" / f.name)
        break
    (rt / "llama" / DEV.exe_name("llama-server")).chmod(0o755)


def build_plan(dev: DEV.Device, root: Path = ROOT, *, autostart: bool = False, base_python: str = sys.executable) -> list[Step]:
    rt = Path(dev.runtime_dir)
    gpu = bool(dev.gpu_name)
    venv, lmenv = root / ".venv", rt / "lmenv"
    vpy, lpy = DEV.venv_python(venv), DEV.venv_python(lmenv)
    setting = DEV.derive(dev, lm_cuda=gpu)
    plan = [
        Step("detect", f"{dev.os}/{dev.arch}, {dev.cores_physical} cores, {dev.ram_gb} GB RAM, GPU: "
                       f"{dev.gpu_name + f' ({dev.vram_gb} GB)' if gpu else 'none'}; derived: {setting}"),
        Step("venv", f"create {venv} and install requirements.txt (+ pytest, mypy, psutil)",
             run=lambda: (_sh([base_python, "-m", "venv", str(venv)]) if not vpy.exists() else None,
                          _sh([str(vpy), "-m", "pip", "install", "-r", str(root / "requirements.txt"), "pytest", "mypy", "psutil"]))[-1]),
        Step("lm-env", f"create {lmenv}; torch from {TORCH_CUDA_INDEX if gpu else TORCH_CPU_INDEX} "
                       f"({'CUDA build: NVIDIA GPU present' if gpu else 'CPU build: no NVIDIA GPU'})",
             run=lambda: (_sh([base_python, "-m", "venv", str(lmenv)]) if not lpy.exists() else None,
                          _sh(torch_command(lpy, gpu)))[-1]),
        Step("llama.cpp", f"fetch the latest release asset matching {llama_asset_patterns(dev.os, dev.arch, gpu)} from {LLAMA_API} "
                          f"into {rt / 'llama'}", download=LLAMA_API, run=lambda: _install_llama(dev, rt)),
        Step("model", f"fetch {DEV.MODEL_FILE} (~{MODEL_APPROX_GB} GB) into {rt / 'models'}", download=MODEL_URL,
             run=lambda: _fetch(MODEL_URL, DEV.model_path(rt))),
        Step("state-dirs", f"create {root / 'state' / 'creator'} and {rt}",
             run=lambda: [p.mkdir(parents=True, exist_ok=True) for p in (root / "state" / "creator", rt)] and None),
        Step("autostart", ("write AND enable" if autostart else "write only (enable later with --autostart)")
             + (" the HKCU Run entry 'Nupen'" if dev.os == "Windows" else
                " the launchd agent" if dev.os == "Darwin" else " the systemd user unit"),
             run=lambda: _sh([sys.executable, str(root / "scripts" / "nupen_autostart.py"), "install" if autostart else "write"])),
        Step("doctor", "run scripts/nupen_doctor.py (read-only health check)",
             run=lambda: subprocess.run([str(vpy if vpy.exists() else sys.executable), str(root / "scripts" / "nupen_doctor.py")])
             and None),
    ]
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print every step, do nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask before downloads")
    ap.add_argument("--autostart", action="store_true", help="also enable start-at-logon")
    a = ap.parse_args(argv)
    dev = DEV.get()
    plan = build_plan(dev, autostart=a.autostart)
    for i, s in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {s.name}: {s.what}" + (f"\n      network: {s.download}" if s.download else ""))
    if a.dry_run:
        print("dry run: nothing was changed")
        return 0
    if any(s.download for s in plan) and not a.yes:
        if input("The steps marked 'network' download several hundred MB. Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            print("cancelled")
            return 1
    for s in plan:
        if s.run is None:
            continue
        print(f"== {s.name}", flush=True)
        s.run()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
