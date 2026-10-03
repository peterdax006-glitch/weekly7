"""PUBLIC DATA that Nupen collects itself (owner, 3 Oct 2026: "you can also use public information it can learn off of"; 08:45: Nupen must
collect its own data when it runs out).

Two network jobs, both handed out by drillsources.drill_filler(public=True) as ordinary filler jobs at BELOW_NORMAL priority, each logging its
own errors and retried later (nothing here can stop the swarm):
  refresh   `git fetch` of ONE active public clone in runtime_dir()/public_repos (each at most every FETCH_EVERY_S), then the new upstream
            commits are appended to that project's privacy-stripped cache (drillsources.refresh_extra_cache). That is what lets
            drillsources.live_pass_x predict commits whose outcome window is still open and resolve them hours later: prospective evidence.
  acquire   when every git drill search has stopped for its present data (nothing left to learn) - or at most once an hour anyway - clone ONE
            more public repository: first from the curated CANDIDATES (active, varied ecosystems, permissive licences), then from the
            unauthenticated GitHub search API (popular, recently pushed, permissive, size-capped). Shallow (the last ACQ_HISTORY_DAYS), no
            checkout, staged outside public_repos and moved in only when complete; within a disk budget. Each attempt is one row of
            state/creator/thinking/public_acquisitions.jsonl (public repository names and sizes only).
PRIVACY: nothing local is ever sent - a search query and git's own protocol only; never authenticated (no token, credential helpers disabled,
no prompt). Archived projects (~/oldpc) are never fetched. Authors and messages never reach any stored file (drillsources.stripped)."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from creator import drillsources as D

FETCH_EVERY_S = 2400.0          # a public clone is fetched at most every 40 minutes
CHECK_EVERY_S = 60.0            # how often the filler looks for a due fetch
ACQ_CHECK_EVERY_S = 300.0       # ... and whether an acquisition is due
ACQ_EVERY_S = 3600.0            # the steady cadence: at most one new repository an hour
ACQ_DRY_GAP_S = 900.0           # the drills ran dry: sooner (still never two at once - one job in flight)
ACQ_HISTORY_DAYS = 270          # shallow clone: the recent history is what the live predictions and the replay need
FETCH_TIMEOUT_S = 1800.0
CLONE_TIMEOUT_S = 3 * 3600.0    # a slow line (0.3-2 MB/s) and a few hundred MB
REPO_CAP_MB = 1500.0            # a clone bigger than this is removed again (recorded 'too_big')
BUDGET_GB = 20.0                # all public clones together
HEADROOM_GB = 20.0              # free disk kept above the device's disk floor
MAX_FAILS = 2                   # a repository that failed this often is not tried again
FAIL_RETRY_S = 86400.0
DISCOVER_LANGS = ("python", "rust", "go", "typescript", "javascript", "java", "c++", "c", "c#", "ruby", "php", "kotlin", "swift", "scala")
DISCOVER_MAX_KB = 600_000       # GitHub's size (KB, whole history) - the shallow clone is smaller
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "mpl-2.0", "psf-2.0", "unlicense", "0bsd", "zlib", "bsl-1.0"}

# active upstreams (many commits a week), varied ecosystems, permissive licences
CANDIDATES: tuple[str, ...] = (
    "pydantic/pydantic", "fastapi/fastapi", "astral-sh/ruff", "astral-sh/uv", "django/django", "numpy/numpy", "scikit-learn/scikit-learn",
    "pandas-dev/pandas", "facebook/react", "matplotlib/matplotlib", "home-assistant/core", "python/cpython", "huggingface/transformers",
    "microsoft/vscode", "golang/go", "rust-lang/rust", "sympy/sympy", "pypa/pip", "denoland/deno", "nodejs/node", "python/mypy",
    "pola-rs/polars", "duckdb/duckdb", "apache/airflow", "apache/arrow", "prometheus/prometheus", "tokio-rs/tokio", "bevyengine/bevy",
    "vitejs/vite", "sveltejs/svelte", "rails/rails", "laravel/framework", "spring-projects/spring-boot", "neovim/neovim", "godotengine/godot",
    "sqlalchemy/sqlalchemy", "celery/celery", "encode/httpx", "psf/black", "pallets/click", "ansible/ansible", "helix-editor/helix",
    "kubernetes/kubernetes", "microsoft/TypeScript", "vercel/next.js", "grafana/loki", "hashicorp/consul", "pytorch/vision",
)

UA = "nupen-public-data (unauthenticated; public metadata only)"


# ------------------------------------------------------------------------------------------------ plumbing
def runtime() -> Path:
    from creator import device as DEV
    return DEV.runtime_dir()


def staging_dir() -> Path:
    return runtime() / "public_staging"


def acquisitions_path(state: Path) -> Path:
    return Path(state) / "thinking" / "public_acquisitions.jsonl"


def fetch_state_path() -> Path:
    return runtime() / "thinking" / "public_fetch.json"


def discovery_path() -> Path:
    return runtime() / "thinking" / "public_discovery.json"


def _env() -> dict[str, str]:
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "GIT_ASKPASS": "", "SSH_ASKPASS": ""}


def _net_git(args: list[str], timeout: float, cwd: Optional[Path] = None) -> str:
    """A git command that may use the network: never authenticated, never prompts, below-normal priority."""
    p = subprocess.run(["git", "-c", "credential.helper=", "-c", "core.askPass=", *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=str(cwd) if cwd else None, env=_env(),
                       creationflags=D.IDLE_PRIORITY if os.name == "nt" else 0)
    if p.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {p.stderr.strip()[:300]}")
    return p.stdout


def _read_json(path: Path) -> dict[str, Any]:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, v: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(v, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _rmtree(path: Path) -> None:
    def onerr(func: Callable[..., Any], p: str, _exc: Any) -> None:
        try:
            os.chmod(p, stat.S_IWRITE)                                  # git marks pack files read-only on Windows
            func(p)
        except OSError:
            pass
    if path.exists():
        shutil.rmtree(path, onerror=onerr)


def dir_mb(path: Path) -> float:
    total = 0
    for d, _ds, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total / 1e6


# ------------------------------------------------------------------------------------------------ refresh (fetch + incremental cache)
def public_repos() -> list[Path]:
    return [r for r in D.extra_repos() if D.is_public(r)]


def due_fetches(now: Optional[float] = None, every: float = FETCH_EVERY_S) -> list[Path]:
    """Public clones whose last fetch attempt is older than `every`, the longest-waiting first."""
    now = time.time() if now is None else now
    st = _read_json(fetch_state_path())
    due = [(float((st.get(r.name) or {}).get("at", 0.0)), r) for r in public_repos()]
    return [r for at, r in sorted(due, key=lambda x: (x[0], x[1].name)) if now - at >= every]


def refresh_public(repo: Path, now: Optional[float] = None, timeout: float = FETCH_TIMEOUT_S) -> int:
    """`git fetch` one public clone (no merge, no checkout: its upstream branch moves), then append the new commits to its stripped cache.
    The attempt is recorded first, so a failing repository waits FETCH_EVERY_S like any other. Returns the number of new commits cached."""
    now = time.time() if now is None else now
    repo = Path(repo)
    if not D.is_public(repo):
        raise ValueError(f"{repo.name} is not a public clone: archived projects are never fetched")
    path = fetch_state_path()
    st = _read_json(path)
    st[repo.name] = {**(st.get(repo.name) or {}), "at": now}
    _write_json(path, st)
    try:
        _net_git(["-C", str(repo), "fetch", "--quiet", "--no-tags", "origin"], timeout)
        if D.upstream_ref(repo) == "HEAD":                             # origin/HEAD missing (an old clone): learn it once
            _net_git(["-C", str(repo), "remote", "set-head", "origin", "--auto"], 300.0)
        n = D.refresh_extra_cache(repo)
    except Exception as e:                                              # noqa: BLE001 - recorded, then raised to the job's error row
        st = _read_json(path)
        st[repo.name] = {**(st.get(repo.name) or {}), "at": now, "ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}
        _write_json(path, st)
        raise
    st = _read_json(path)
    st[repo.name] = {"at": now, "ok": True, "new": n, "done": time.time()}
    _write_json(path, st)
    return n


# ------------------------------------------------------------------------------------------------ acquisition
def acquisitions(state: Path) -> list[dict[str, Any]]:
    from creator import thinking as T
    return T._jsonl(acquisitions_path(state))


def _record(state: Path, row: dict[str, Any]) -> dict[str, Any]:
    p = acquisitions_path(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **row}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def drills_exhausted(state: Path) -> bool:
    """Every git drill (Nupen's own + other projects) has stopped searching for its present data: more data is the only way on."""
    srcs = [s for s, kind in D.SOURCES.items() if kind in ("git", "gitx")]
    rep = D.search_report(state, srcs)
    return bool(rep) and all(s in rep and rep[s]["stopped"] for s in srcs)


def acquisition_due(state: Path, now: Optional[float] = None) -> bool:
    now = time.time() if now is None else now
    last = max([float(r.get("t", 0.0)) for r in acquisitions(state)] or [0.0])
    if now - last < ACQ_DRY_GAP_S:
        return False
    return now - last >= ACQ_EVERY_S or drills_exhausted(state)


def disk_ok(extra_mb: float = 0.0) -> tuple[bool, str]:
    try:
        from creator import device as DEV
        floor = float(DEV.settings().get("disk_floor_gb", 1.0))
    except Exception:                                                   # noqa: BLE001
        floor = 1.0
    rt = runtime()
    rt.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(rt).free / 1e9
    if free < floor + HEADROOM_GB + extra_mb / 1e3:
        return False, f"free disk {free:.1f} GB < floor {floor} + headroom {HEADROOM_GB} GB"
    pub = D.public_dir()
    used = sum(dir_mb(r / ".git") for r in pub.iterdir() if r.is_dir()) / 1e3 if pub.is_dir() else 0.0
    if used + extra_mb / 1e3 >= BUDGET_GB:
        return False, f"public clones use {used:.1f} GB of the {BUDGET_GB} GB budget"
    return True, f"free {free:.1f} GB, public {used:.1f} GB"


def dir_name(full: str) -> str:
    owner, _, name = full.partition("/")
    pub = D.public_dir()
    plain = name or owner
    if (pub / plain / ".git").exists():
        try:
            if (pub / plain / ".git" / "nupen_source").read_text(encoding="utf-8").strip() == full:
                return plain                                            # acquired by Nupen under that name
        except OSError:
            pass
        try:
            url = _net_git(["-C", str(pub / plain), "config", "--get", "remote.origin.url"], 30.0).strip().lower().rstrip("/")
            if url.removesuffix(".git").replace("\\", "/").endswith("/" + full.lower()):
                return plain                                            # that very repository is already here
        except RuntimeError:
            pass
        return f"{owner}__{name}"
    return plain


def present(full: str) -> bool:
    return (D.public_dir() / dir_name(full) / ".git").exists()


def _github_get(url: str, timeout: float = 30.0) -> tuple[int, dict[str, Any], dict[str, str]]:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:                         # noqa: S310 - fixed https host
            return r.status, json.loads(r.read().decode("utf-8")), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, {}, dict(e.headers or {})


def github_discover(now: Optional[float] = None, get: Callable[[str], tuple[int, dict[str, Any], dict[str, str]]] = _github_get) -> list[dict[str, Any]]:
    """Popular, recently pushed, permissively licensed, size-capped repositories of the next language in DISCOVER_LANGS (one search request
    per call; rate limits honoured: a 403/429 waits until GitHub's reset time). Found ones accumulate in public_discovery.json."""
    now = time.time() if now is None else now
    path = discovery_path()
    st = _read_json(path)
    found: list[dict[str, Any]] = list(st.get("found") or [])
    if now < float(st.get("retry_at", 0.0)):
        return found
    i = int(st.get("lang_i", 0))
    lang = DISCOVER_LANGS[i % len(DISCOVER_LANGS)]
    since = (dt.datetime.fromtimestamp(now, dt.timezone.utc) - dt.timedelta(days=2)).strftime("%Y-%m-%d")
    q = f"language:{lang} stars:>3000 pushed:>{since} size:<{DISCOVER_MAX_KB} archived:false fork:false"
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": q, "sort": "stars", "order": "desc", "per_page": 50})
    try:
        code, body, headers = get(url)
    except (OSError, ValueError) as e:
        st.update(retry_at=now + 900.0, err=f"{type(e).__name__}: {str(e)[:200]}")
        _write_json(path, st)
        return found
    if code in (403, 429):
        reset = float(headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset") or 0.0)
        st.update(retry_at=max(now + 60.0, reset or now + 3600.0), err=f"rate limited ({code})")
        _write_json(path, st)
        return found
    have = {f["repo"] for f in found}
    for it in (body.get("items") or []) if code == 200 else []:
        lic = ((it.get("license") or {}).get("spdx_id") or "").lower()
        full = str(it.get("full_name") or "")
        if lic in PERMISSIVE and full and full not in have and not it.get("archived") and int(it.get("size") or 0) <= DISCOVER_MAX_KB:
            found.append({"repo": full, "lang": lang, "license": lic, "size_kb": int(it.get("size") or 0), "stars": int(it.get("stargazers_count") or 0)})
            have.add(full)
    st.update(found=found, lang_i=i + 1, retry_at=now + 60.0, err="" if code == 200 else f"http {code}")
    _write_json(path, st)
    return found


def _url(full: str) -> str:
    return f"https://github.com/{full}.git"


def choose(state: Path, now: float, candidates: Sequence[str] = CANDIDATES,
           discover: Optional[Callable[[float], list[dict[str, Any]]]] = github_discover,
           url_of: Callable[[str], str] = _url) -> Optional[tuple[str, str, str]]:
    """(repo, url, why) of the next repository to acquire, or None: curated first, then discovered; never one already here, never one that
    failed MAX_FAILS times or within FAIL_RETRY_S."""
    fails: dict[str, list[float]] = {}
    for r in acquisitions(state):
        if not r.get("ok") and r.get("repo") and r.get("why") != "disk":
            fails.setdefault(str(r["repo"]), []).append(float(r.get("t", 0.0)))

    def ok(full: str) -> bool:
        f = fails.get(full, [])
        return not present(full) and len(f) < MAX_FAILS and (not f or now - max(f) >= FAIL_RETRY_S)
    for full in candidates:
        if ok(full):
            return full, url_of(full), "curated active upstream"
    if discover is not None:
        for c in discover(now):
            if ok(str(c["repo"])):
                return str(c["repo"]), url_of(str(c["repo"])), f"github search: {c.get('lang')} {c.get('license')} {c.get('stars')} stars"
    return None


def acquire_one(state: Path, now: Optional[float] = None, candidates: Sequence[str] = CANDIDATES,
                discover: Optional[Callable[[float], list[dict[str, Any]]]] = github_discover, url_of: Callable[[str], str] = _url,
                timeout: float = CLONE_TIMEOUT_S, cap_mb: float = REPO_CAP_MB) -> Optional[dict[str, Any]]:
    """Clone ONE more public repository into public_repos (shallow, no checkout, staged then moved in whole). Every attempt - success,
    failure, too big, no disk - is one row of public_acquisitions.jsonl. Returns that row (None: nothing left to acquire)."""
    now = time.time() if now is None else now
    good, why_disk = disk_ok()
    if not good:
        return _record(state, {"t": now, "ok": False, "why": "disk", "detail": why_disk})
    pick = choose(state, now, candidates, discover, url_of)
    if pick is None:
        return None
    full, url, why = pick
    name = dir_name(full)
    stage = staging_dir() / name
    _rmtree(stage)
    stage.parent.mkdir(parents=True, exist_ok=True)
    since = (dt.datetime.fromtimestamp(now, dt.timezone.utc) - dt.timedelta(days=ACQ_HISTORY_DAYS)).strftime("%Y-%m-%d")
    t0 = time.time()
    try:
        _net_git(["clone", "--quiet", "--no-tags", "--single-branch", "--no-checkout", f"--shallow-since={since}", url, str(stage)], timeout)
        size = dir_mb(stage)
        if size > cap_mb:
            _rmtree(stage)
            return _record(state, {"t": now, "repo": full, "ok": False, "why": "too_big", "size_mb": round(size, 1)})
        commits = int(_net_git(["-C", str(stage), "rev-list", "--count", "HEAD"], 600.0).strip() or 0)
        (stage / ".git" / "nupen_source").write_text(full, encoding="utf-8")
        dest = D.public_dir() / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(dest)
    except Exception as e:                                              # noqa: BLE001 - recorded; the repository is tried again after FAIL_RETRY_S
        _rmtree(stage)
        return _record(state, {"t": now, "repo": full, "ok": False, "why": "error", "detail": f"{type(e).__name__}: {str(e)[:200]}"})
    return _record(state, {"t": now, "repo": full, "dir": name, "ok": True, "commits": commits, "size_mb": round(size, 1),
                           "seconds": round(time.time() - t0, 1), "why": why, "since": since})
