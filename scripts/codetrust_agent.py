"""Agent recipes for the coding trust gate (scripts/coding_trust.py run --agent-cmd ...): Aider, Aider with a tool path (aider_t) and
OpenCode, invoked as in the agent bake-off (wt53_agents scripts/agent_bakeoff/bakeoff.py, whose metering proxy and aider_tools.py are
reused READ-ONLY), against any OpenAI-compatible endpoint. After the agent ran, its tokens (counted by the proxy) and a CONFIDENCE go to
<workdir>/.codetrust_result.json: the agent's own file if it wrote one, else one short self-rating call to the same model with the task and
the agent's diff ("self-rated after the run"; its tokens are counted too). Agent droppings (.aider*, AGENTS.md copies) are removed.

  --agent-cmd "python scripts/codetrust_agent.py aider --workdir {workdir} --task-file {task_file} --url {url}"
  (aider | aider_t | opencode; --cap seconds, default 1800)"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

AG = Path.home() / "creator_runtime" / "agents"
BAKEOFF = Path(os.environ.get("CODETRUST_BAKEOFF_DIR", str(Path.home() / "wt53_agents" / "scripts" / "agent_bakeoff")))
PROJECT_PY = Path.home() / "weekly7" / ".venv" / "Scripts" / "python.exe"
CTX = 32768
DROPPINGS = (".aider", "AGENTS.md", "opencode.json")
CONF_RE = re.compile(r"(?im)^\s*CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?|\.\d+)\s*$")


def proxy(url: str) -> Any:
    """The bake-off's metering proxy (counts prompt/completion tokens per run); None when the bake-off tree is not there."""
    try:
        sys.path.insert(0, str(BAKEOFF))
        import bakeoff                                                 # noqa: PLC0415 - optional, read-only reuse
        return bakeoff.Proxy(url.rstrip("/").removesuffix("/v1"), think="off")
    except Exception:                                                  # noqa: BLE001
        return None


def server_ctx(url: str, default: int = CTX) -> int:
    """The per-slot context of a llama-server (/props); agents told a larger one overflow and compact the task away."""
    try:
        with urllib.request.urlopen(url.rstrip("/").removesuffix("/v1") + "/props", timeout=30) as r:
            d = json.loads(r.read().decode("utf-8"))
        return int((d.get("default_generation_settings") or {}).get("n_ctx") or default)
    except Exception:                                                  # noqa: BLE001
        return default


def task_tests(task_text: str, ws: Path) -> list[str]:
    return [p for p in re.findall(r"^- (tests/\S+\.py)$", task_text, re.M) if (ws / p).is_file()]


def build(agent: str, ws: Path, base: str, task_text: str, rd: Path, ctx: int = CTX) -> list[str]:
    if agent in ("aider", "aider_t"):
        meta = rd / "aider_model_meta.json"
        meta.write_text(json.dumps({"openai/qwen": {"max_input_tokens": ctx - 4096, "max_output_tokens": min(8192, ctx // 4), "max_tokens": min(8192, ctx // 4),
                                                    "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai",
                                                    "mode": "chat"}}), encoding="utf-8")
        tests = task_tests(task_text, ws)
        argv = [str(AG / "aider_venv" / "Scripts" / "aider.exe"), "--model", "openai/qwen", "--openai-api-base", base, "--openai-api-key",
                "sk-local", "--model-metadata-file", str(meta), "--edit-format", "diff", "--yes-always", "--no-auto-commits",
                "--no-dirty-commits", "--no-check-update", "--no-analytics", "--no-pretty", "--no-fancy-input", "--no-show-model-warnings",
                "--no-gitignore", "--map-tokens", "2048"]
        if tests:
            argv += ["--auto-test", "--test-cmd", f'"{PROJECT_PY}" -m pytest -q -x -p no:cacheprovider ' + " ".join(tests)]
        argv += ["--message", task_text]
        if agent == "aider_t":
            argv = [str(AG / "aider_venv" / "Scripts" / "python.exe"), str(BAKEOFF / "aider_tools.py"), "--rounds", "6", "--", *argv[1:]]
        return argv
    if agent == "opencode":
        cfg = {"$schema": "https://opencode.ai/config.json", "autoupdate": False, "share": "disabled", "model": "nupen/qwen",
               "provider": {"nupen": {"npm": "@ai-sdk/openai-compatible", "name": "Nupen llama-server", "options": {"baseURL": base, "apiKey": "sk-local"},
                                      "models": {"qwen": {"name": "qwen", "tool_call": True, "limit": {"context": ctx, "output": min(8192, ctx // 4)}}}}},
               "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "external_directory": "deny"}}
        (rd / "opencode.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        exe = AG / "npm" / "node_modules" / "opencode-windows-x64" / "bin" / "opencode.exe"
        return [str(exe), "run", "--format", "json", "--auto", "-m", "nupen/qwen", "--dir", str(ws), task_text]
    raise SystemExit(f"unknown agent {agent}")


def env_for(agent: str, base: str, rd: Path) -> dict[str, str]:
    env = dict(os.environ)
    home = AG / "home" / agent
    home.mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), USERPROFILE=str(home), XDG_CONFIG_HOME=str(home / ".config"), XDG_DATA_HOME=str(home / ".local" / "share"),
               XDG_CACHE_HOME=str(home / ".cache"), XDG_STATE_HOME=str(home / ".local" / "state"), OPENAI_API_KEY="sk-local",
               OPENAI_BASE_URL=base, OPENAI_API_BASE=base, OPENAI_MODEL="qwen", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", NO_COLOR="1",
               TERM="dumb", CI="1")
    for k in ("ANTHROPIC_API_KEY", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "GH_TOKEN", "GITHUB_TOKEN"):
        env.pop(k, None)
    if agent == "opencode":
        env.update(OPENCODE_CONFIG=str(rd / "opencode.json"), OPENCODE_DISABLE_AUTOUPDATE="1")
    return env


def self_rate(url: str, task_text: str, diff: str) -> tuple[float | None, int, int]:
    """One short call: the model's probability that the diff passes the hidden tests (the confidence the gate calibrates)."""
    msg = (f"A coding agent was given this task:\n{task_text[:6000]}\n\nIt produced this diff:\n{diff[:20000] or '(no change)'}\n\n"
           "What is the probability (0 to 1) that this change passes the project's hidden tests for the task without breaking any other "
           "test? Reply with one line exactly: CONFIDENCE: p")
    body = {"messages": [{"role": "user", "content": msg}], "max_tokens": 30, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    try:
        req = urllib.request.Request(url.rstrip("/").removesuffix("/v1") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:                                                  # noqa: BLE001
        return None, 0, 0
    text = str(d["choices"][0]["message"].get("content") or "")
    m = CONF_RE.search(text)
    u = d.get("usage") or {}
    return (min(1.0, max(0.0, float(m.group(1)))) if m else None), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def clean(ws: Path) -> None:
    for p in ws.iterdir():
        if p.name.startswith(DROPPINGS):
            if p.is_dir():
                import shutil
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=("aider", "aider_t", "opencode"))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--cap", type=float, default=1800.0)
    a = ap.parse_args(argv)
    ws, tf = Path(a.workdir), Path(a.task_file)
    rd = tf.parent / f"{tf.stem}.{a.agent}"
    rd.mkdir(parents=True, exist_ok=True)
    task_text = tf.read_text(encoding="utf-8")
    px = proxy(a.url)
    base = px.base if px is not None else a.url.rstrip("/").removesuffix("/v1") + "/v1"
    t0 = time.monotonic()
    rc, timed_out = -1, False
    with (rd / "agent.log").open("w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(build(a.agent, ws, base, task_text, rd, server_ctx(a.url)), cwd=ws, env=env_for(a.agent, base, rd), stdout=log,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            rc = p.wait(timeout=a.cap)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True, timeout=60) if sys.platform == "win32" else p.kill()
            p.wait(timeout=60)
    meter = px.meter.as_dict() if px is not None else {}
    if px is not None:
        px.close()
    clean(ws)
    rf = ws / ".codetrust_result.json"
    own: dict[str, Any] = {}
    if rf.is_file():
        try:
            own = dict(json.loads(rf.read_text(encoding="utf-8")))
        except ValueError:
            own = {}
    conf = own.get("confidence") if isinstance(own.get("confidence"), (int, float)) else None
    rin = rout = 0
    how = "agent"
    if conf is None:
        subprocess.run(["git", "-C", str(ws), "add", "-A"], capture_output=True, timeout=120)
        diff = subprocess.run(["git", "-C", str(ws), "diff", "--cached", "HEAD", "--", ".", ":(exclude).codetrust_result.json"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120).stdout
        conf, rin, rout = self_rate(a.url, task_text, diff)
        how = "self-rated after the run"
    res = {"confidence": conf, "confidence_source": how, "tokens_in": int(meter.get("prompt_tokens") or 0) + rin,
           "tokens_out": int(meter.get("completion_tokens") or 0) + rout, "requests": meter.get("requests"), "rc": rc,
           "timed_out": timed_out, "agent_seconds": round(time.monotonic() - t0, 1)}
    rf.write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
