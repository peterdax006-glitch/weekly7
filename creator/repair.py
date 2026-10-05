"""R2 + R4 of the 5000-lines/h plan: Claude-like TEST-DRIVEN REPAIR that is cheap, and the guard against truncation loops.

R2 (after CODE, driven by the VISIBLE examples only, never the hidden tests):
  failing visible test -> traceback parser + creator.tools.pinpoint rank the suspect lines -> strategy "line": the model edits ONLY one
  pinpointed line (the SEARCH half of the edit and its start are prefilled, the output is 1-5 lines under a grammar, ~20 tokens);
  strategy "regen": CODE again with the failing example in the prompt (also the answer to an edit that did not apply).
  Rounds stop early when two rounds in a row bring no progress; a round that made things worse is rolled back.
R4 (wall not spent generating): looping() is the stop test of a streamed generation (a ladder `if n == 1: ... if n == 2: ...` or any
  repeated n-gram), retry_extra() is the one retry policy that does not repeat a greedy loop (repeat penalty + a tighter cap).

Everything here is pure text work (testable without a model); the orchestration lives in scripts/pc_baseline.py and creator/slotteam.py."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

NL = chr(10)
FENCE = chr(96) * 3
DIGITS = re.compile(r"\d+")
STRATEGIES = ("none", "line", "regen", "line+regen", "regen+line")
RETRY_EXTRA = {"repeat_penalty": 1.15, "temperature": 0.3, "seed": 7}
RETRY_CAP_SHARE = 0.7                       # the retry may use at most this share of the cap the looping call had
LINE_CAP = 72                               # tokens of a one-line edit (measured need ~20)
WINDOW = 20                                 # lines of code shown on each side of the suspect line


# ------------------------------------------------------------------------------------------------------------------ R4: loop guard
def _norm(line: str) -> str:
    return DIGITS.sub("0", line.strip())


def looping(text: str, min_chars: int = 6) -> bool:
    """True when the generated text so far degenerated: the last complete lines repeat with a period of 1-4 lines (digits are normalised, so
    the ladder `if n == 1: return 0` / `if n == 2: return 0` counts), or the tail repeats a 12-80 character unit four times."""
    lines = [_norm(x) for x in text.split(NL)[:-1]]
    lines = [x for x in lines if x]
    for p in (1, 2, 3, 4):
        need = 5 if p == 1 else 4
        if len(lines) >= p * need:
            tail = lines[-p * need:]
            if all(tail[i] == tail[i % p] for i in range(len(tail))) and sum(len(x) for x in tail[:p]) >= min_chars:
                return True
    t = text.rstrip()
    for unit in range(12, 81, 4):
        if len(t) >= unit * 4 and t[-unit:] == t[-2 * unit:-unit] == t[-3 * unit:-2 * unit] == t[-4 * unit:-3 * unit]:
            return True
    return False


def retry_extra(cap: int) -> tuple[dict[str, Any], int]:
    """Sampling overrides and the token cap of the ONE retry after a loop (greedy decoding would loop the same way again)."""
    return dict(RETRY_EXTRA), max(48, int(cap * RETRY_CAP_SHARE))


# ------------------------------------------------------------------------------------------------------------------ the failing example
def brief(val: str, limit: int = 300) -> str:
    """The visible failure in one or two lines: the assertion message (input -> got != expected) or the first error text."""
    for ln in val.splitlines():
        m = re.search(r"AssertionError: (.+)", ln)
        if m:
            return m.group(1)[:limit]
    m = re.search(r"\b(\w*Error: .+)", val)
    if m:
        return m.group(1)[:limit]
    txt = re.sub(r"^FAIL\s+(?:FAILED|\w+)?:?\s*", "", val.strip())
    return NL.join(txt.splitlines()[:3])[:limit]


def score(val: str) -> int:
    """Badness of a VALIDATE result: 0 pass, 1000 a patch that did not apply / static error, else the number of failing visible tests."""
    if val.startswith("PASS"):
        return 0
    if val.startswith("FAIL apply") or val.startswith("FAIL static"):
        return 1000
    return max(1, len(re.findall(r"(?:^|: )FAIL [\w.\[\]-]*\.[\w.\[\]-]*:", val, re.M)))


def signature(val: str) -> str:
    return _norm(brief(val, 160))


# ------------------------------------------------------------------------------------------------------------------ the line strategy
def candidates(pin: str, ws: Path, skip: set[tuple[str, int]]) -> list[tuple[str, int, str]]:
    """The pinpoint output ('file:line: text', best first) as (file, line, text) of lines worth editing: not tests, not blank, not a docstring,
    a bare def / raise NotImplementedError, and not tried before."""
    out: list[tuple[str, int, str]] = []
    for ln in pin.splitlines():
        m = re.match(r"^(.+?):(\d+): ?(.*)$", ln)
        if not m:
            continue
        f, n = m.group(1), int(m.group(2))
        txt = m.group(3).strip()
        low = f.replace("\\", "/")
        if low.startswith("tests/") or "/tests/" in low or low.rsplit("/", 1)[-1].startswith("test_"):
            continue
        if not txt or txt.startswith(('"""', "'''", "#", "def ", "async def ", "class ", "raise NotImplementedError", "@")) or (f, n) in skip:
            continue
        out.append((f, n, txt))
    return out


def window(ws: Path, rel: str, ln: int, ctx: int = WINDOW) -> str:
    try:
        lines = (ws / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return NL.join(lines[max(0, ln - 1 - ctx): ln + ctx])


def line_prefill(ws: Path, rel: str, ln: int, with_file: bool) -> tuple[str, str]:
    """(prefilled start of the answer, the SEARCH text). The SEARCH is the suspect line; when that line occurs more than once in the file the
    line above joins it (and is copied as the first replacement line) so the patch tool finds exactly one match."""
    lines = (ws / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    cur = lines[ln - 1]
    search, head = cur, ""
    if sum(1 for x in lines if x.strip() == cur.strip()) > 1 and ln >= 2:
        prev = lines[ln - 2]
        search, head = prev + NL + cur, prev + NL
    pre = (f"FILE: {rel}{NL}" if with_file else "") + f"<<<<<<< SEARCH{NL}{search}{NL}======={NL}{head}"
    return pre, search


def line_prompt(ws: Path, rel: str, ln: int, text: str, fail: str, fn_task: bool) -> str:
    """User prompt of a one-line repair: the code around the line, the failing example, the line to replace. Short: prompt tokens are read at
    ~100/s, so the context is the 2 x WINDOW lines around the suspect and 300 characters of failure."""
    code = window(ws, rel, ln)
    return (f"ROLE: DEBUG{NL}Current {rel}:{NL}{FENCE}python{NL}{code}{NL}{FENCE}{NL}{NL}Failing example:{NL}{brief(fail)}{NL}{NL}"
            f"Suspect line:{NL}{text}{NL}{NL}Replace only the suspect line (1-5 lines) so the example passes. Edit format: one block exactly like{NL}"
            f"{'' if fn_task else 'FILE: <path>' + NL}<<<<<<< SEARCH{NL}<the suspect line>{NL}======={NL}<new lines>{NL}>>>>>>> REPLACE")


def regen_hint(fail: str) -> str:
    """The failing example for a regenerated CODE call (inserted into the user prompt after the request)."""
    if fail.startswith("FAIL apply"):
        return f"Your previous edit did not apply ({fail[11:150].strip()}). The SEARCH text must be copied exactly from the code shown."
    return f"Your previous answer failed this example:{NL}{brief(fail)}{NL}Fix it."


def py_snapshot(ws: Path) -> dict[str, str]:
    return {p.relative_to(ws).as_posix(): p.read_text(encoding="utf-8", errors="replace") for p in ws.rglob("*.py")
            if "__pycache__" not in p.parts and not p.relative_to(ws).as_posix().startswith("tests/")}


def py_restore(ws: Path, snap: dict[str, str]) -> None:
    for rel, txt in snap.items():
        p = ws / rel
        if not p.is_file() or p.read_text(encoding="utf-8", errors="replace") != txt:
            p.write_text(txt, encoding="utf-8")


def plan_rounds(strategy: str, max_rounds: int) -> list[str]:
    """The strategy of each repair round: 'line' = pinpointed one-line edits, 'regen' = CODE again with the failing example; a combination
    runs its first strategy up to max_rounds - 1 times and its second at the end."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown repair strategy {strategy!r}")
    if strategy == "none" or max_rounds <= 0:
        return []
    if strategy in ("line", "regen"):
        return [strategy] * max_rounds
    a, b = strategy.split("+")
    return [a] * max(0, max_rounds - 1) + [b]


def stop_early(history: list[int], limit: int = 2) -> bool:
    """No progress: the last `limit` rounds all failed to lower the badness score below the best one before them."""
    if len(history) <= limit:
        return False
    best_before = min(history[:-limit])
    return all(h >= best_before for h in history[-limit:])
