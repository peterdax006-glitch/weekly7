# Device audit (2 Oct 2026)

Owner: "if there is anything that is device specific we need to figure out how to change the system to make it whatever
device is being operated on". Machine today: Windows 11 on ARM64 running an x64 (emulated) venv, 8 cores, 16.8 GB, no usable GPU,
user `Peter`. The next one: 32 GB, bigger GPU, probably x64 + NVIDIA.

All device facts and derived settings now live in `creator/device.py` (detected once, cached; owner pins in
`state/creator/device_overrides.json`; snapshot in `state/creator/device.json` at every service / swarm start). New machine:
`python scripts/nupen_setup.py`; health: `python scripts/nupen_doctor.py`.

Status: FIXED = replaced by a `creator/device.py` lookup (behaviour here unchanged, see the test); OK = already portable;
OPEN = left, with the reason.

| File | Line (before) | Assumption | Fix / status |
|---|---|---|---|
| creator/generator.py | 42-44 | runtime dir `~/creator_runtime`, env `CREATOR_RUNTIME` only; `llama-server.exe`; model path | FIXED: `device.runtime_dir()` (`NUPEN_RUNTIME`, then `CREATOR_RUNTIME`, then default), `device.server_exe()` (no `.exe` off Windows) |
| creator/generator.py | 6 | docstring named `C:\Users\Peter\creator_runtime` | FIXED |
| creator/generator.py | 258 | llama threads hard-coded 6 | FIXED: `threads=None` -> `settings()["llama_threads"]` (physical cores minus 2 on <=8 cores, else minus 1; 8 cores -> 6) |
| creator/generator.py | 270 | no `-ngl`: the GPU is never used | FIXED: `-ngl 99` when VRAM >= model size + 1 GB (`gpu_layers`), else not passed |
| creator/generator.py | 58-250 | `ctypes.WinDLL("kernel32")` job object, pid image query, TerminateProcess; `msvcrt` byte-range lock | OK: all behind `_WIN` / `sys.platform` with `os.kill`, `fcntl` branches |
| creator/lm/paths.py | 9 | default `C:\Users\Peter\creator_runtime` | FIXED: `device.runtime_dir()` |
| creator/lm/dialogue.py | 23 | same, env `NUPEN_RUNTIME` | FIXED |
| creator/lm/train.py | whole | CPU only; checkpoints `map_location="cpu"` | FIXED: `device` argument (model and batches moved; checkpoints saved device-free); `nupen_lm.py train` picks `torch_device` |
| creator/lm/api.py | 31-33 | inference is CPU (`map_location="cpu"`) | OK: intentional, small model; revisit if the LM grows |
| creator/sandbox.py | 162, 449 | sandbox root `<repo parent>/.<repo>_creator_sandboxes` | FIXED: `device.sandbox_root(repo)`; env `NUPEN_SANDBOXES` relocates; default identical (`~/.weekly7_creator_sandboxes`) |
| creator/kernel.py | 443 | same | FIXED |
| creator/swarm.py | 265, 341 | same | FIXED |
| creator/swarm.py | 148-167 | owner idle time via `ctypes.windll` GetLastInputInfo; non-Windows returned 0 ("owner always there": the LM trainer would never run) | FIXED: `device.idle_seconds_unix()` (xprintidle on Linux, ioreg HIDIdleTime on macOS) |
| creator/swarm.py | 170-192 | governor floors | OK: already fractions of total RAM (7 %, min 0.8 GB) via `psutil`; `max_workers` default now RAM-derived in the CLI |
| scripts/creator_swarm.py | 117-120 | `--max-workers 32`, `--test-parallel 6` (tuned on 16 GB) | FIXED: defaults from `settings()["max_workers"]`, `["test_slots"]` (16.8 GB -> 32 and 6; 32 GB -> 61 and 11) |
| scripts/nupen_service.py | 43 | `LM_PYTHON` under `~/creator_runtime/lmenv` | FIXED: `device.lm_python()` |
| scripts/nupen_service.py | 45, 149 | `LM_MIN_FREE_GB = 3.0`; STOP file under `~/creator_runtime` | FIXED: `settings()["lm_min_free_gb"]` (max(3.0, 10 % of RAM)); runtime dir lookup |
| scripts/nupen_service.py | 251, 345 | `.venv/Scripts/pythonw.exe` / `python.exe` | FIXED: `device.venv_python()` |
| scripts/nupen_service.py | 61-90, 118 | ctypes kernel32, `taskkill /T` | OK: non-Windows branches (`os.kill`, `/proc/uptime`, `killpg`) |
| scripts/nupen_service.py | start | no record of which machine it ran on | FIXED: `write_snapshot(log=log)` at "supervisor up" |
| scripts/nupen_watchdog.py | 77, 87 | `taskkill`, `pythonw.exe` | taskkill OK (killpg branch); interpreter FIXED |
| scripts/nupen_autostart.py | all | Windows registry only (`import winreg` at top of `main`), `pythonw.exe` | FIXED: HKCU Run on Windows; systemd user unit (Linux) and launchd plist (macOS) written by `write`, enabled only by `install` / `nupen_setup.py --autostart` |
| scripts/nupen_lm.py | 2, 64 | docstring path; `--threads 4`; CPU | FIXED: `--threads`/`--device` default to the derived values |
| scripts/lowprio.py | 19-27 | `SetPriorityClass` | OK: `os.nice(10)` branch |
| scripts/grid_runner.py | 17 | `.venv/Scripts/python.exe` | FIXED: `device.venv_python()` |
| scripts/grid_runner.py | 22 | free RAM via PowerShell `Get-CimInstance` | OPEN: research script, Windows only; `psutil.virtual_memory().available` is the one-line fix when it is next used |
| scripts/backfill_lessons.py | 306 | default `--repo C:\Users\Peter\weekly7` | FIXED: the checkout the script lives in |
| scripts/sync_intraday.py | 9 | `GH = C:\Users\Peter\tools\gh\bin\gh.exe` | FIXED: `~/tools/gh/bin/gh.exe`, already falls back to `gh` on PATH |
| scripts/save_owner_doc.py | 8-9 | `C:\Users\Peter\.claude\projects\...` | FIXED: `Path.home()`-based |
| scripts/movers_batch.sh, publish_patterns.sh, restart_loop2.sh | 3 | `cd /c/Users/Peter/weekly7` (Git Bash path) | FIXED: `cd "$(dirname "$0")/.."` |
| scripts/relaunch_loop_after.ps1 | 4, 9 | `C:\Users\Peter\weekly7` | FIXED: `$PSScriptRoot` |
| scripts/*.sh | various | `.venv/Scripts/python` (Windows venv layout) | OPEN: Git-Bash-on-Windows launchers; on Linux use `.venv/bin/python` |
| scripts/run_parity.py | 33-56 | `ctypes.windll` psapi / SetPriorityClass | OPEN: one-off research probe, Windows only |
| scripts/leak_audit.py, miner_coverage.py, ... (docstrings) | n/a | `.venv/Scripts/python`, PowerShell `Start-Process` launch recipes | OPEN: documentation of how the owner launched them on Windows |
| creator/recon.py | 175 | process census via PowerShell `Get-CimInstance` | FIXED: psutil command lines off Windows |
| creator/agents.py | 177 | `claude` .cmd shim | OK: `shutil.which(cli)` |
| creator/build.py, testrun.py, prescreen.py, ... | n/a | interpreter | OK: `sys.executable` everywhere |
| creator/efficiency.py, kernel.py | n/a | psutil | OK: psutil is in the venv (setup installs it) |
| Git Bash location | n/a | no code refers to `C:\Program Files\Git`; git is found on PATH | OK (doctor checks `git`) |
| HKCU Run entry | n/a | Windows registry only | FIXED (see autostart) |
| llama.cpp build (this machine: Windows, x64 build emulated on ARM64) | n/a | binary lives in the runtime dir, was fetched by hand | FIXED: `nupen_setup.py` picks the release asset for OS / arch / NVIDIA (cuda build when a GPU is present) |
| torch CPU-only LM env | n/a | `lmenv` built by hand | FIXED: `nupen_setup.py` installs the CPU or CUDA wheel; `lm_env_has_cuda()` reads the version file (never imports torch) to decide `torch_device` |

## Historical, untouched

`scripts/_c34.py` ... `_c67.py`, `_chk.py`, `_memfix.py`, `_p0.py`, `_pfix.py`, `_save_*.py`, `_c40_45.py`, `_c44.py`,
`scripts/patch_*.py` and `scripts/legacy/` are one-off scripts that already ran; they keep whatever paths they had.
`creator/packages/*.md` are historical package texts. A test (`tests/test_device_portability.py`) greps every other `.py`, `.sh`,
`.ps1` and `.json` under `creator/` and `scripts/` for `C:\Users\Peter` and fails if one comes back.

## What the settings are here and on the 32 GB profile

| setting | this machine (8 cores, 16.8 GB, no GPU) | today's constant | 32 GB, 16 cores, x64, 12 GB VRAM, CUDA torch |
|---|---|---|---|
| llama_threads | 6 | `threads=6` | 15 |
| gpu_layers | 0 | none | 99 |
| test_slots | 6 | `--test-parallel 6` | 11 |
| max_workers | 32 | `--max-workers 32` | 61 |
| governor floor | 7 % of RAM, min 0.8 GB | same | 7 % = 2.2 GB, min 0.8 GB |
| lm_threads | 4 | `--threads 4` | 8 |
| lm_min_free_gb | 3.0 | `LM_MIN_FREE_GB = 3.0` | 3.2 |
| torch_device | cpu | cpu | cuda |
