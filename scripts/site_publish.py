"""Bible Phase 29 - background publisher: rebuild the public pages when their sources change, audit, and commit only the site.

Replaces the ad-hoc scripts/publish_patterns.sh for the new pages. Each cycle: `site_build.check_fresh()`; if stale, build
(the build itself refuses on any secret or sealed mark), run site_audit, and only when both are clean `git add` exactly the four
generated pages and commit. It never pushes unless --push is given, never stages anything else, and skips the cycle if
the working tree has staged changes from someone else (so it cannot sweep another agent's work into its commit).
    python scripts/site_publish.py --once           one cycle then exit
    python scripts/site_publish.py --interval 300   loop (background)
Every cycle appends one line to state/research/site/publish.log with the reason, so silence is never ambiguous."""
import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import site_audit as A, site_build as B, site_safety as S

LOG = ROOT / "state" / "research" / "site" / "publish.log"
SITE_FILES = [f"docs/{n}" for n in B.PAGES]


def _log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}\n"
    with open(LOG, "ab") as fh:
        fh.write(line.encode("utf-8"))
    print(line.strip())


def _git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120)


def staged_foreign(cwd=ROOT):
    """Files already staged that are not ours; committing now would include them."""
    r = _git("diff", "--cached", "--name-only", cwd=cwd)
    return [x for x in r.stdout.splitlines() if x and x not in SITE_FILES]


def cycle(paths=None, push=False, commit=True, cwd=ROOT):
    """One publish cycle. Returns a short status string (also logged)."""
    paths = paths or B.default_paths()
    fresh, why = B.check_fresh(paths)
    if fresh:
        return _done("fresh: nothing to do")
    try:
        B.build(paths)
    except S.Leak as e:
        return _done(f"REFUSED by safety gate: {e}")
    errs = [x for x in A.audit_site(paths["out"]) if x["severity"] == "error"]
    if errs:
        return _done(f"built but audit failed ({len(errs)}): {errs[0]['page']} {errs[0]['check']}; not committed")
    if not commit:
        return _done(f"built and audited clean ({len(why)} source change(s)); commit disabled")
    foreign = staged_foreign(cwd)
    if foreign:
        return _done(f"skipped commit: {len(foreign)} unrelated staged file(s), e.g. {foreign[0]}")
    _git("add", "--", *SITE_FILES, cwd=cwd)
    c = _git("commit", "-q", "-m", "Public site: refresh generated pages\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>", "--", *SITE_FILES, cwd=cwd)
    if c.returncode != 0:
        return _done("nothing to commit" if "nothing" in (c.stdout + c.stderr).lower() else f"commit failed: {c.stderr.strip()[:120]}")
    if push:
        p = _git("push", "-q", cwd=cwd)
        return _done("committed and pushed" if p.returncode == 0 else f"committed; push failed: {p.stderr.strip()[:120]}")
    return _done("committed (not pushed)")


def _done(msg):
    _log(msg)
    return msg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args(argv)
    while True:
        cycle(push=a.push, commit=not a.no_commit)
        if a.once:
            return 0
        time.sleep(max(a.interval, 30))


if __name__ == "__main__":
    sys.exit(main())
