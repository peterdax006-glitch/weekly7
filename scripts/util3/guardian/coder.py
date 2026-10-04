#!/usr/bin/env python3
"""GPU guardian CODING item (runs ON THE POD, stdlib + system pytest): role-tagged development experience from PUBLIC tasks only.

Owner/teacher 3-4 Oct: teach the small home models the whole pipeline, one role at a time, every row verified by a real run. Per task (staged
public rl_tasks train rows: OpenCodeInstruct + mined public repositories; never a Nupen-repository task), against a filler server whose port
the guardian tagged 'coding', with TEACHER_BRIEF.md as the system prompt:
  SPEC   -> requirements + ACCEPTANCE TESTS; verified = the tests pass on the final verified solution AND fail on the starting stub
  PLAN   -> steps / risks / test plan; verified = the trajectory ended in a passing solution
  REVIEW -> verdict on the first candidate BEFORE it is run; verified = the verdict matches the real test result
  CODE   -> SEARCH/REPLACE edits on solution.py; verified = the hidden tests pass
  DEBUG  -> CAUSE/EVIDENCE/FIX/EXPECT from the REAL failing output (up to 3 rounds); verified = the fix makes the hidden tests pass
  VALIDATE -> MEETS SPEC yes|no on the final code; verified = it matches the real outcome (hidden tests and the SPEC's own tests)
  LESSON -> one line; verified = the trajectory it summarises ended in a pass
Failures are kept (verified false). The hidden tests and any solution never enter a SPEC/PLAN/CODE prompt; DEBUG sees only the real failing
output. Tests run here on the pod CPU (python -I, temp dir, timeout, nice 10, 2 GB address space, at most 4 at once).
Rows -> out_roles/<model>.jsonl (one row per role output + one 'trajectory' row per task); the PC pulls them home."""
import ast, gzip, json, os, re, resource, subprocess, sys, tempfile, threading, time, urllib.request

D = "/root/guardian"
TASKS = f"{D}/q/coding_tasks.jsonl.gz"
BRIEF = f"{D}/q/TEACHER_BRIEF.md"
PORTS = (18350, 18351, 18352, 18353)
PER_PORT = 16                    # tasks in flight per server (each task is a chain of single calls; 2x the 8 slots)
ROUNDS = 3
TESTS = threading.Semaphore(4)   # pod CPU: test processes at once
THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
SR = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.S)
EDIT_HELP = ("Edit format: one or more blocks exactly like\n<<<<<<< SEARCH\n<exact lines of solution.py>\n=======\n<new lines>\n>>>>>>> REPLACE")

lock = threading.Lock()
done = set()                     # (model, task_id) with a trajectory row
taken = set()
live = {}                        # port -> model ('' = not a coding port / down)


def log(msg):
    with open(f"{D}/coder.log", "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def chat(port, model, system, user, max_tokens):
    body = json.dumps({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user + "\n/no_think"}],
                       "max_tokens": max_tokens, "temperature": 0.3, "seed": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    if os.path.basename(str(d.get("model") or model)) != model:
        raise RuntimeError("model changed")
    return THINK.sub("", str(d["choices"][0]["message"].get("content") or "")).strip()


def _limits():
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


def run_py(files, cmd, timeout=15):
    """Run `cmd` in a fresh temp dir holding `files`; (returncode, trimmed output)."""
    with TESTS, tempfile.TemporaryDirectory() as td:
        for name, text in files.items():
            with open(os.path.join(td, name), "w", encoding="utf-8") as f:
                f.write(text)
        try:
            p = subprocess.run(cmd, cwd=td, capture_output=True, timeout=timeout, preexec_fn=_limits)
            out = (p.stdout + p.stderr).decode("utf-8", "replace")
            return p.returncode, out.replace(td, ".")[-1500:]
        except subprocess.TimeoutExpired:
            return 124, f"TIMEOUT after {timeout}s"


HIDDEN = r'''
import ast, json, sys
sys.path.insert(0, ".")
from solution import {name} as _f
_cases = ast.literal_eval(open("cases.txt", encoding="utf-8").read())
def _n(x):
    return json.loads(json.dumps(x, default=str))
for _c in _cases:
    try:
        _got = _f(*_c["args"])
    except Exception as e:
        print("FAIL: %s(%s) raised %s: %s (expected %r)" % ("{name}", ", ".join(map(repr, _c["args"])), type(e).__name__, e, _c["expect"]))
        sys.exit(1)
    if _n(_got) != _n(_c["expect"]):
        print("FAIL: %s(%s) returned %r, expected %r" % ("{name}", ", ".join(map(repr, _c["args"])), _got, _c["expect"]))
        sys.exit(1)
print("PASS: %d cases" % len(_cases))
'''


def hidden(task, code):
    rc, out = run_py({"solution.py": code, "cases.txt": task["tests"], "check.py": HIDDEN.replace("{name}", task["name"])},
                     [sys.executable, "-I", "check.py"])
    return rc == 0 and out.strip().startswith("PASS"), out.strip()


def acceptance(code, tests):
    """The SPEC's own pytest functions against `code` (solution.py); True = all pass."""
    src = "from solution import *\n" + tests
    rc, out = run_py({"solution.py": code, "test_spec.py": src}, [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                                                  "test_spec.py"], timeout=30)
    return rc == 0, out.strip()[-600:]


def apply_edits(code, reply, name):
    """SEARCH/REPLACE blocks applied to `code`; else a ```python block defining the function replaces the file. (new code, format)."""
    blocks = SR.findall(reply)
    if blocks:
        new = code
        for s, r in blocks:
            if s not in new:
                return None, "search_miss"
            new = new.replace(s, r, 1)
        return new, "search_replace"
    cb = [b for b in CODE_BLOCK.findall(reply) if f"def {name}" in b]
    if cb:
        return max(cb, key=len), "code_block"
    return None, "no_edit"


def spec_tests(spec):
    m = re.search(r"ACCEPTANCE TESTS:\s*(.*?)(?:\nOPEN QUESTIONS:|\Z)", spec, re.S)
    if not m:
        return ""
    body = m.group(1)
    cb = CODE_BLOCK.findall(body)
    return (cb[0] if cb else body).strip()


def write(model, rows):
    with lock, open(f"{D}/out_roles/{model}.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def one_task(port, model, brief, task):
    tid, name = task["id"], task["name"]
    stub = f"def {name}(*args, **kwargs):\n    raise NotImplementedError\n"
    req = task["prompt"]
    rows, t0 = [], time.time()

    def row(role, user, out, verified, **kw):
        rows.append(dict({"role": role, "task_id": tid, "model": model, "verified": verified, "input": user, "output": out,
                          "ts": round(time.time(), 1)}, **kw))
    u_spec = f"ROLE: SPEC\nRequest:\n{req}\n\nStarting code (solution.py):\n```python\n{stub}```\nWrite the acceptance tests as pytest functions that import nothing but the function under test (it is already imported)."
    spec = chat(port, model, brief, u_spec, 700)
    u_plan = f"ROLE: PLAN\nSPEC:\n{spec}\n\nRequest:\n{req}"
    plan = chat(port, model, brief, u_plan, 400)
    code = stub
    u_code = f"ROLE: CODE\nPlan:\n{plan}\n\nRequest:\n{req}\n\nCurrent solution.py:\n```python\n{code}```\n{EDIT_HELP}"
    reply = chat(port, model, brief, u_code, 1500)
    cand, fmt = apply_edits(code, reply, name)
    u_rev = f"ROLE: REVIEW\nRequest:\n{req}\n\nCandidate solution.py:\n```python\n{cand or reply}\n```"
    review = chat(port, model, brief, u_rev, 300)
    ok, out = hidden(task, cand) if cand else (False, f"edit could not be applied ({fmt})")
    vm = re.search(r"VERDICT:\s*(correct|incorrect|unsure)", review, re.I)
    verdict = vm.group(1).lower() if vm else None
    row("CODE", u_code, reply, ok, edit_format=fmt, test_output=out)
    row("REVIEW", u_rev, review, (verdict == "correct") == ok if verdict in ("correct", "incorrect") else False, verdict=verdict, real_pass=ok)
    attempts = [{"stage": "code", "format": fmt, "pass": ok, "output": out}]
    rnd = 1
    while not ok and rnd < ROUNDS:
        rnd += 1
        u_dbg = (f"ROLE: DEBUG\nRequest (expected behaviour):\n{req}\n\nCurrent solution.py:\n```python\n{cand or code}```\n\n"
                 f"Exact failing output:\n{out}\n{EDIT_HELP}")
        dbg = chat(port, model, brief, u_dbg, 900)
        new, fmt = apply_edits(cand or code, dbg, name)
        ok2, out2 = hidden(task, new) if new else (False, f"edit could not be applied ({fmt})")
        row("DEBUG", u_dbg, dbg, ok2, edit_format=fmt, failing_output=out, test_output=out2, round=rnd)
        attempts.append({"stage": f"debug{rnd}", "format": fmt, "pass": ok2, "output": out2})
        if new:
            cand = new
        ok, out = ok2, out2
    tests = spec_tests(spec)
    spec_ok, spec_note = None, ""
    if tests:
        fails_on_stub = not acceptance(stub, tests)[0]
        if ok:
            passes, spec_note = acceptance(cand, tests)
            spec_ok = passes and fails_on_stub
        else:
            spec_ok = None if fails_on_stub else False           # unknown without a verified solution, unless it already "passes" the stub
    row("SPEC", u_spec, spec, bool(spec_ok), spec_verified=spec_ok, spec_tests_found=bool(tests), spec_note=spec_note)
    row("PLAN", u_plan, plan, ok)
    if ok:
        u_val = f"ROLE: VALIDATE\nSPEC:\n{spec}\n\nFinal solution.py:\n```python\n{cand}```"
        val = chat(port, model, brief, u_val, 250)
        mm = re.search(r"MEETS SPEC:\s*(yes|no)", val, re.I)
        truth = ok and (spec_ok is not False)
        row("VALIDATE", u_val, val, (mm.group(1).lower() == "yes") == truth if mm else False, meets=(mm.group(1).lower() if mm else None), truth=truth)
    summary = "; ".join(f"{a['stage']}: {'pass' if a['pass'] else 'fail'} ({a['output'][:160]})" for a in attempts)
    u_les = f"ROLE: LESSON\nTask: {name}\nTrajectory: {summary}"
    lesson = chat(port, model, brief, u_les, 80)
    row("LESSON", u_les, lesson, ok and "->" in lesson)
    rows.append({"role": "trajectory", "task_id": tid, "model": model, "verified": ok, "rounds": rnd, "attempts": attempts, "final_code": cand if ok else None,
                 "spec_verified": spec_ok, "seconds": round(time.time() - t0, 1), "ts": round(time.time(), 1)})
    write(model, rows)
    return ok


def worker(port, tasks, brief):
    while not os.path.exists(f"{D}/STOP_CODER"):
        model = live.get(port, "")
        if not model:
            time.sleep(3)
            continue
        task = None
        with lock:
            for t in tasks:
                k = (model, t["id"])
                if k not in done and k not in taken:
                    taken.add(k)
                    task = t
                    break
        if task is None:
            with lock:
                left = sum(1 for t in tasks if (model, t["id"]) not in done)
            if left == 0:                       # every task answered by this model: give its queue line back (guard.sh then serves the next item)
                subprocess.run(["sed", "-i", f"/ {re.escape(model)} coding/d", f"{D}/queue.txt"])
                log(f"all tasks done for {model}: coding line removed")
            time.sleep(60)
            continue
        try:
            one_task(port, model, brief, task)
            with lock:
                done.add((model, task["id"]))
        except Exception as e:
            log(f"{port} {task['id']}: {e!r}"[:300])
            time.sleep(2)
        finally:
            with lock:
                taken.discard((model, task["id"]))


def port_tag(port):
    try:
        with open(f"{D}/port_{port}.tag") as f:
            return f.read().strip()
    except OSError:
        return ""


def served(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
            return os.path.basename(str(json.loads(r.read())["data"][0]["id"]))
    except Exception:
        return ""


def main():
    os.makedirs(f"{D}/out_roles", exist_ok=True)
    while not (os.path.exists(TASKS) and os.path.exists(BRIEF)):
        time.sleep(30)
    with gzip.open(TASKS, "rt", encoding="utf-8") as f:
        tasks = [json.loads(ln) for ln in f]
    brief = open(BRIEF, encoding="utf-8").read()
    for name in os.listdir(f"{D}/out_roles"):
        with open(f"{D}/out_roles/{name}", encoding="utf-8", errors="replace") as f:
            for ln in f:
                if '"trajectory"' in ln:
                    try:
                        r = json.loads(ln)
                        done.add((r["model"], r["task_id"]))
                    except ValueError:
                        pass
    log(f"coder: {len(tasks)} public tasks, {len(done)} done")
    for p in PORTS:
        for _ in range(PER_PORT):
            threading.Thread(target=worker, args=(p, tasks, brief), daemon=True).start()
    while not os.path.exists(f"{D}/STOP_CODER"):
        for p in PORTS:
            live[p] = served(p) if port_tag(p) == "coding" else ""
        time.sleep(3)


if __name__ == "__main__":
    main()
