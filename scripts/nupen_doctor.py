"""Read-only health check of this machine for Nupen: prints a PASS/FAIL table and what to run to fix each FAIL.

    python scripts/nupen_doctor.py [--json]

Changes nothing, starts no model, imports no torch (the LM env is probed by reading its files and, only for the import check,
by a short child process). Exit code 0 when everything passes, 1 otherwise. First run on a new machine: scripts/nupen_setup.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from creator import device as DEV  # noqa: E402

SETUP = "python scripts/nupen_setup.py"


class Check(NamedTuple):
    name: str
    ok: bool
    detail: str
    fix: str


def _probe(python: Path, code: str, timeout: float = 90.0) -> tuple[bool, str]:
    try:
        p = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)[:120]
    return p.returncode == 0, (p.stdout or p.stderr).strip().splitlines()[-1][:120] if (p.stdout or p.stderr).strip() else ""


def run_checks(dev: DEV.Device, root: Path = ROOT, *, probe: Callable[[Path, str], tuple[bool, str]] = _probe,
               remote_ok: Optional[Callable[[Path], tuple[bool, str]]] = None, imports: bool = True) -> list[Check]:
    rt = Path(dev.runtime_dir)
    cfg = DEV.derive(dev)
    out: list[Check] = []

    def add(name: str, ok: bool, detail: str, fix: str) -> None:
        out.append(Check(name, bool(ok), detail, "" if ok else fix))

    add("device", dev.ram_gb > 0 and dev.cores_logical > 0,
        f"{dev.os}/{dev.arch}, {dev.cores_physical} cores, {dev.ram_gb} GB, gpu={dev.gpu_name or 'none'}; settings {cfg}",
        "install psutil in the venv: pip install psutil")
    snap = root / "state" / "creator" / "device.json"
    add("device snapshot", snap.exists(), str(snap), "starts with the service; or: python -m creator.device")
    vpy = DEV.venv_python(root / ".venv")
    add(".venv python", vpy.exists(), str(vpy), SETUP)
    if vpy.exists() and imports:
        ok, d = probe(vpy, "import pandas, numpy, psutil, pytest, mypy; print('imports ok')")
        add(".venv imports", ok, d, f"{vpy} -m pip install -r requirements.txt pytest mypy psutil")
    lpy = DEV.lm_python(rt)
    add("LM env python", lpy.exists(), str(lpy), SETUP)
    if lpy.exists() and imports:
        ok, d = probe(lpy, "import torch; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')")
        add("LM env torch", ok, d, f"{SETUP}  (creates the LM env with the right torch build)")
        if dev.gpu_name:
            add("GPU usable by LM env", ok and "cuda" in d, d, f"reinstall torch with CUDA: {lpy} -m pip install torch --index-url "
                "https://download.pytorch.org/whl/cu126")
    add("GPU", True, (f"{dev.gpu_name} {dev.vram_gb} GB, llama -ngl {cfg['gpu_layers']}" if dev.gpu_name
                      else "none: llama.cpp runs on the CPU (fine, just slower)"), "")
    exe = DEV.server_exe(rt)
    add("llama-server", exe.is_file(), str(exe), SETUP)
    add("model file", DEV.model_path(rt).is_file(), str(DEV.model_path(rt)), SETUP)
    for label, p in (("runtime dir", rt), ("sandbox root parent", Path(dev.sandbox_root).parent), ("state/creator", root / "state" / "creator")):
        add(label, p.is_dir(), str(p), f"mkdir {p}")
    for tool in ("git",):
        import shutil
        add(tool, shutil.which(tool) is not None, shutil.which(tool) or "not on PATH", f"install {tool} and put it on PATH")
    if remote_ok is None:
        remote_ok = _remote
    ok, d = remote_ok(root)
    add("git remote reachable", ok, d, "check the network / git credentials: git -C . ls-remote origin")
    return out


def _remote(root: Path) -> tuple[bool, str]:
    try:
        p = subprocess.run(["git", "-C", str(root), "ls-remote", "--exit-code", "--heads", "origin"], capture_output=True, text=True,
                           timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)[:120]
    return p.returncode == 0, "origin answered" if p.returncode == 0 else (p.stderr.strip().splitlines() or ["no answer"])[-1][:120]


def render(checks: Sequence[Check]) -> str:
    w = max(len(c.name) for c in checks)
    lines = [f"{'PASS' if c.ok else 'FAIL'}  {c.name:<{w}}  {c.detail}" for c in checks]
    fails = [c for c in checks if not c.ok]
    if fails:
        lines += ["", "To fix:"] + [f"  {c.name}: {c.fix}" for c in fails]
    lines.append(f"\n{len(checks) - len(fails)}/{len(checks)} passed")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    checks = run_checks(DEV.get())
    print(json.dumps([c._asdict() for c in checks], indent=1) if a.json else render(checks))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
