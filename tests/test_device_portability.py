"""Device portability (owner, 2 Oct 2026): settings follow the machine; nothing is pinned to one user's folders."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from creator import device as DEV  # noqa: E402
from creator import generator as G  # noqa: E402

NVIDIA_12GB = "NVIDIA GeForce RTX 4070, 12282\n"


def machine(*, os_name="Windows", arch="ARM64", logical=8, physical=8, ram=16.76, gpu="") -> DEV.Device:
    return DEV.detect(system=lambda: os_name, machine=lambda: arch, logical=lambda: logical, physical=lambda n: physical,
                      ram_gb=lambda: ram, gpu=lambda: DEV.parse_nvidia_smi(gpu), env={})


def test_16gb_arm_no_gpu_reproduces_todays_constants():
    s = DEV.derive(machine(), overrides={}, lm_cuda=False)
    assert s["llama_threads"] == 6                # LocalModel(threads=6)
    assert s["gpu_layers"] == 0                   # no -ngl
    assert s["test_slots"] == 4                   # creator_swarm --test-parallel 4 (h10/ram budget)
    assert s["max_workers"] == 32                 # creator_swarm --max-workers 32
    assert (s["governor_floor_fraction"], s["governor_floor_min_gb"]) == (0.07, 0.8)
    assert s["lm_threads"] == 4                   # nupen_lm --threads 4
    assert s["lm_min_free_gb"] == 3.0             # nupen_service LM_MIN_FREE_GB
    assert s["torch_device"] == "cpu"


def test_32gb_x64_with_12gb_vram_gets_more_and_the_gpu():
    big = machine(arch="AMD64", logical=16, physical=8 * 2, ram=32.0, gpu=NVIDIA_12GB)
    assert (big.gpu_name, big.arch) == ("NVIDIA GeForce RTX 4070", "x64") and 12.0 < big.vram_gb < 13.0
    s = DEV.derive(big, overrides={}, lm_cuda=True)
    small = DEV.derive(machine(), overrides={}, lm_cuda=False)
    assert s["test_slots"] > small["test_slots"] and s["max_workers"] > small["max_workers"]
    assert s["gpu_layers"] == 99 and s["torch_device"] == "cuda"
    assert s["llama_threads"] == 15


def test_gpu_too_small_for_the_model_stays_on_cpu_and_cuda_needs_a_cuda_torch():
    tiny = machine(gpu="NVIDIA GT 710, 1024\n")
    assert DEV.derive(tiny, overrides={}, lm_cuda=True)["gpu_layers"] == 0
    gpu = machine(gpu=NVIDIA_12GB)
    assert DEV.derive(gpu, overrides={}, lm_cuda=False)["torch_device"] == "cpu"
    assert DEV.derive(machine(), overrides={}, lm_cuda=True)["torch_device"] == "cpu"      # no GPU: never cuda


def test_nvidia_smi_parsing_picks_the_biggest_and_ignores_junk():
    assert DEV.parse_nvidia_smi("") == ("", 0.0)
    assert DEV.parse_nvidia_smi("garbage") == ("", 0.0)
    name, gb = DEV.parse_nvidia_smi("A, 8192\nB, 24576\n")
    assert name == "B" and 25.0 < gb < 26.0


def test_overrides_win_and_unknown_keys_are_ignored(tmp_path):
    p = tmp_path / "device_overrides.json"
    p.write_text(json.dumps({"llama_threads": 3, "gpu_layers": 20, "nonsense": 1}), encoding="utf-8")
    ov = DEV.load_overrides(p)
    s = DEV.derive(machine(), overrides=ov, lm_cuda=False)
    assert s["llama_threads"] == 3 and s["gpu_layers"] == 20 and "nonsense" not in s
    assert DEV.load_overrides(tmp_path / "missing.json") == {}
    assert DEV.derive(machine(), overrides={}, lm_cuda=False)["user_aware"] is True              # yields to the owner by default
    s = DEV.derive(machine(), overrides={"user_aware": False, "governor_floor_fraction": 0.0, "governor_floor_min_gb": 1.0},
                   lm_cuda=False)
    assert (s["user_aware"], s["governor_floor_fraction"], s["governor_floor_min_gb"]) == (False, 0.0, 1.0)
    p.write_text("not json", encoding="utf-8")
    assert DEV.load_overrides(p) == {}


def test_env_vars_relocate_runtime_and_sandboxes(tmp_path):
    assert DEV.runtime_dir({"NUPEN_RUNTIME": str(tmp_path / "rt")}) == tmp_path / "rt"
    assert DEV.runtime_dir({"CREATOR_RUNTIME": str(tmp_path / "old")}) == tmp_path / "old"
    assert DEV.runtime_dir({"NUPEN_RUNTIME": "a", "CREATOR_RUNTIME": "b"}) == Path("a")
    assert DEV.runtime_dir({}) == Path.home() / "creator_runtime"
    assert DEV.sandbox_root(None, {"NUPEN_SANDBOXES": str(tmp_path / "sb")}) == tmp_path / "sb"
    assert DEV.sandbox_root(None, {}) == Path.home() / ".weekly7_creator_sandboxes"
    repo = tmp_path / "weekly7"
    assert DEV.sandbox_root(repo, {}) == tmp_path / ".weekly7_creator_sandboxes"           # unchanged per-repo default


def test_executable_names_follow_the_os(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert DEV.server_exe(Path("/r")) == Path("/r/llama/llama-server")
    assert DEV.venv_python(Path("/v")) == Path("/v/bin/python")
    monkeypatch.setattr(sys, "platform", "win32")
    assert DEV.server_exe(Path("/r")).name == "llama-server.exe"
    assert DEV.venv_python(Path("/v"), windowless=True) == Path("/v/Scripts/pythonw.exe")


def test_llama_command_gets_ngl_only_with_gpu_layers(tmp_path):
    cpu = G.LocalModel(model=tmp_path / "m.gguf", exe=tmp_path / "s", threads=3, gpu_layers=0, pidfile=tmp_path / "p.pid")
    assert "-ngl" not in cpu._command() and cpu._command()[cpu._command().index("-t") + 1] == "3"
    gpu = G.LocalModel(model=tmp_path / "m.gguf", exe=tmp_path / "s", threads=3, gpu_layers=99, pidfile=tmp_path / "p.pid")
    c = gpu._command()
    assert c[c.index("-ngl") + 1] == "99"


def test_lm_env_cuda_is_read_from_the_version_file_without_importing_torch(tmp_path):
    v = tmp_path / "lmenv" / "Lib" / "site-packages" / "torch"
    v.mkdir(parents=True)
    (v / "version.py").write_text("__version__ = '2.5.1+cu124'\n", encoding="utf-8")
    assert DEV.lm_env_has_cuda(tmp_path) is True
    (v / "version.py").write_text("__version__ = '2.5.1+cpu'\n", encoding="utf-8")
    assert DEV.lm_env_has_cuda(tmp_path) is False
    assert DEV.lm_env_has_cuda(tmp_path / "nowhere") is False


def test_snapshot_is_written_and_a_changed_machine_is_logged(tmp_path, monkeypatch):
    snap = tmp_path / "device.json"
    monkeypatch.setattr(DEV, "get", lambda refresh=False: machine())
    monkeypatch.setattr(DEV, "OVERRIDES", tmp_path / "none.json")
    lines: list[str] = []
    DEV.write_snapshot(snap, log=lines.append)
    assert json.loads(snap.read_text())["settings"]["llama_threads"] == 6 and not any("changed" in x for x in lines)
    monkeypatch.setattr(DEV, "get", lambda refresh=False: machine(ram=32.0, gpu=NVIDIA_12GB))
    DEV.write_snapshot(snap, log=lines.append)
    assert any("device changed" in x and "ram_gb" in x for x in lines)


def test_setup_dry_run_lists_every_step_for_both_machines(monkeypatch, capsys):
    import nupen_setup as S
    for dev, torch_url, asset in ((machine(), "whl/cpu", "win-cpu-arm64"),
                                  (machine(arch="AMD64", logical=16, physical=16, ram=32.0, gpu=NVIDIA_12GB), "whl/cu", "win-cuda")):
        monkeypatch.setattr(DEV, "get", lambda refresh=False, d=dev: d)
        assert S.main(["--dry-run"]) == 0
        out = capsys.readouterr().out
        for step in ("venv", "lm-env", "llama.cpp", "model", "autostart", "doctor"):
            assert f"] {step}:" in out
        assert torch_url in out and asset in out and "nothing was changed" in out
    assert "cu126" in " ".join(S.torch_command(Path("p"), True)) and "cpu" in " ".join(S.torch_command(Path("p"), False))


def test_llama_asset_choice_by_os_arch_gpu():
    import nupen_setup as S
    names = [{"name": n} for n in ("llama-b1-bin-win-cpu-arm64.zip", "llama-b1-bin-win-cpu-x64.zip", "llama-b1-bin-win-cuda-12.4-x64.zip",
                                   "llama-b1-bin-ubuntu-x64.tar.gz", "llama-b1-bin-macos-arm64.tar.gz")]
    pick = lambda os_name, arch, gpu: (S.pick_asset(names, S.llama_asset_patterns(os_name, arch, gpu)) or {}).get("name")   # noqa: E731
    assert pick("Windows", "arm64", False) == "llama-b1-bin-win-cpu-arm64.zip"
    assert pick("Windows", "x64", True) == "llama-b1-bin-win-cuda-12.4-x64.zip"
    assert pick("Windows", "x64", False) == "llama-b1-bin-win-cpu-x64.zip"
    assert pick("Linux", "x64", False) == "llama-b1-bin-ubuntu-x64.tar.gz"
    assert pick("Darwin", "arm64", False) == "llama-b1-bin-macos-arm64.tar.gz"


def test_autostart_writes_unit_files_for_unix(tmp_path, monkeypatch):
    import nupen_autostart as A
    assert "WantedBy=default.target" in A.systemd_unit(tmp_path) and "nupen_service.py" in A.systemd_unit(tmp_path)
    assert "RunAtLoad" in A.launchd_plist(tmp_path)
    monkeypatch.setattr(A, "SYSTEMD_UNIT", tmp_path / "u" / "nupen.service")
    monkeypatch.setattr(sys, "platform", "linux")
    assert A.main(["write"]) == 0 and (tmp_path / "u" / "nupen.service").exists()      # written, nothing enabled


def test_doctor_reports_fail_with_a_fix_and_pass_otherwise(tmp_path):
    import nupen_doctor as DOC
    dev = machine()
    checks = DOC.run_checks(dev, tmp_path, probe=lambda p, c: (True, "ok"), remote_ok=lambda r: (False, "offline"), imports=False)
    byname = {c.name: c for c in checks}
    assert not byname[".venv python"].ok and "nupen_setup" in byname[".venv python"].fix
    assert not byname["git remote reachable"].ok and byname["device"].ok
    text = DOC.render(checks)
    assert "FAIL" in text and "To fix:" in text and "passed" in text


def test_no_hard_coded_user_path_remains_in_live_code():
    pat = re.compile(r"C:\\+Users\\+Peter|C:/Users/Peter|/c/Users/Peter", re.I)
    historical = re.compile(r"^scripts/(_|legacy/|patch_)")
    bad = []
    for base in ("creator", "scripts"):
        for p in (ROOT / base).rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if p.suffix not in (".py", ".sh", ".ps1", ".json") or historical.match(rel) or "__pycache__" in rel:
                continue
            if "/packages/" in rel or rel == "creator/capabilities.json":
                continue
            for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pat.search(ln):
                    bad.append(f"{rel}:{i}")
    assert not bad, bad
