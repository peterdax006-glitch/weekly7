"""Bible Phase 29 - build the public explanation site (static pages for GitHub Pages) from state/ artefacts.

    python scripts/site_build.py                 # write docs/explorer.html, sensitivity2.html, checklist.html, runs.html
    python scripts/site_build.py --dry-run       # build and audit in memory, write nothing
    python scripts/site_build.py --verify        # re-audit the pages already in docs/ for secrets and sealed data

Pipeline: load sources (site_sources) -> render pages (site_pages) -> audit every page for secrets and sealed marks
(site_safety) -> only then write, atomically and byte-exact (LF, so hashes are portable). If any page fails the audit
NOTHING is written and the exit code is 2. A manifest with the SHA-256 of every page and the source files it was built
from goes to state/research/site/manifest.json so the site can be reproduced and stale pages detected (`--check`).
Deterministic: same sources and same `--built-at` give byte-identical pages."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import site_pages as P, site_safety as S, site_sources as SRC

PAGES = {"explorer.html": "explorer", "sensitivity2.html": "sensitivity2", "checklist.html": "checklist", "runs.html": "runs"}
MANIFEST_REL = Path("state") / "research" / "site" / "manifest.json"


def default_paths(root=ROOT):
    root = Path(root)
    return {"algo": root / "state" / "research" / "algorithm", "bank": root / "state" / "pattern_bank",
            "sens": root / "state" / "research" / "sensitivity.json", "sens_pub": root / "docs" / "sensitivity.json",
            "checklist": root / "state" / "CHECKLIST.md", "registry": root / "state" / "experiments.jsonl",
            "cycles": root / "state" / "livesim" / "cycles.json", "reports": root / "state" / "reports",
            "research": root / "state" / "research", "out": root / "docs"}


def _plain():
    """The repo's plain-English indicator names when available; the site still builds without the engine."""
    try:
        from engine.explain import plain
        return plain
    except Exception:
        return None


def _sha(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None


def source_hashes(paths):
    """SHA-256 prefixes of every input file the pages depend on (the sealed cycle list only by hash of the file, never content)."""
    files = [paths["sens"], paths["sens_pub"], paths["checklist"], paths["registry"], paths["cycles"]]
    files += sorted(Path(paths["algo"]).glob("patterns_*.parquet")) + sorted(Path(paths["algo"]).glob("test_*.json"))
    files += [Path(paths["algo"]) / "planted_calibration.json"]
    bank = Path(paths["bank"])
    if bank.is_dir():
        files += sorted(bank.glob("bank_v*.json"))[-1:]
    return {str(Path(f).relative_to(ROOT)) if ROOT in Path(f).parents else Path(f).name: _sha(f) for f in files if _sha(f)}


def load_all(paths):
    problems = {"patterns": [], "sens": [], "check": [], "runs": []}
    plain = _plain()
    runs = SRC.load_pattern_runs(paths["algo"], plain, problems["patterns"])
    if not runs:
        problems["patterns"].append("no miner output (patterns_*.parquet) found under state/research/algorithm")
    bank = SRC.load_bank(paths["bank"], plain, problems["patterns"])
    calib = S.safe_read_json(Path(paths["algo"]) / "planted_calibration.json", None)
    sens_raw = S.safe_read_json(paths["sens"], None) or S.safe_read_json(paths["sens_pub"], None)
    sens = SRC.build_sensitivity(sens_raw, problems["sens"])
    ck_text = S.safe_read_text(paths["checklist"]) if Path(paths["checklist"]).exists() else ""
    if not ck_text:
        problems["check"].append("state/CHECKLIST.md not found or empty")
    ck = SRC.parse_checklist(ck_text)
    registry = SRC.load_registry(paths["registry"], problems["runs"])
    cycles = SRC.load_cycles(paths["cycles"], problems["runs"])
    reports = SRC.list_reports([("reports", paths["reports"]), ("research", paths["research"]), ("algorithm", paths["algo"])])
    unproven = SRC.build_unproven(ck, registry)
    return {"runs": runs, "bank": bank, "calib": calib, "sens": sens, "ck": ck, "unproven": unproven, "registry": registry,
            "cycles": cycles, "reports": reports, "problems": problems}


def render_all(d, built_at):
    gf = lambda **kw: {"built_at": built_at, **kw}
    ex, ex_s = P.build_explorer(d["runs"], d["bank"], d["calib"], d["problems"]["patterns"],
                                gf(patterns=f'{len(d["runs"])} miner runs', bank=("v%s" % d["bank"]["version"]) if d["bank"]["available"] else "not created yet"))
    se, se_s = P.build_sensitivity2(d["sens"], d["problems"]["sens"], gf(experiments=f'{d["sens"]["n_variants"]} tested values'))
    ck, ck_s = P.build_checklist(d["ck"], d["unproven"], d["problems"]["check"], gf(checklist="state/CHECKLIST.md", registry="state/experiments.jsonl"))
    ru, ru_s = P.build_runs(d["cycles"], d["registry"], d["reports"], d["problems"]["runs"],
                            gf(cycles="revealed windows only", registry=f'{len(d["registry"])} records'))
    return {"explorer.html": ex, "sensitivity2.html": se, "checklist.html": ck, "runs.html": ru}, \
           {"explorer.html": ex_s, "sensitivity2.html": se_s, "checklist.html": ck_s, "runs.html": ru_s}


def audit_pages(pages):
    """Run the leak audit on every page; return the list of failures instead of raising, so all are reported at once."""
    bad = []
    for name, text in pages.items():
        try:
            S.audit_text(name, text)
        except S.Leak as e:
            bad.append(str(e))
    return bad


def write_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:                              # binary: no CRLF translation on Windows
        fh.write(text.encode("utf-8"))
    os.replace(tmp, path)


def build(paths=None, built_at=None, dry_run=False, manifest_path=None):
    """Build, audit, then write. Returns the manifest. Raises S.Leak (nothing written) if any page fails the audit."""
    paths = paths or default_paths()
    built_at = built_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    d = load_all(paths)
    pages, summaries = render_all(d, built_at)
    bad = audit_pages(pages)
    if bad:
        raise S.Leak("site build refused, nothing written: " + " | ".join(bad))
    manifest = {"built_at": built_at, "pages": {n: {"sha256": hashlib.sha256(t.encode("utf-8")).hexdigest(), "bytes": len(t.encode("utf-8")),
                                                     "summary": summaries[n]} for n, t in pages.items()},
                "sources": source_hashes(paths), "problems": {k: v for k, v in d["problems"].items() if v}}
    if not dry_run:
        out = Path(paths["out"])
        for n, t in pages.items():
            write_atomic(out / n, t)
        mp = Path(manifest_path) if manifest_path else Path(paths["out"]).parent / MANIFEST_REL
        write_atomic(mp, json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def check_fresh(paths=None, manifest_path=None):
    """Stale-page detector: compare the sources recorded in the last manifest with the sources on disk now.
    Returns (fresh, reasons). A missing manifest or missing page is stale by definition."""
    paths = paths or default_paths()
    mp = Path(manifest_path) if manifest_path else Path(paths["out"]).parent / MANIFEST_REL
    if not mp.exists():
        return False, ["no manifest: the site has never been built"]
    m = json.loads(mp.read_text(encoding="utf-8"))
    reasons = []
    for name, info in m.get("pages", {}).items():
        f = Path(paths["out"]) / name
        if not f.exists():
            reasons.append(f"{name} is missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != info["sha256"]:
            reasons.append(f"{name} was edited after the build")
    now = source_hashes(paths)
    for k in sorted(set(now) | set(m.get("sources", {}))):
        if now.get(k) != m.get("sources", {}).get(k):
            reasons.append(f"source changed: {k}")
    return not reasons, reasons


def verify(paths=None):
    """Audit the pages that are already on disk (what is actually published)."""
    paths = paths or default_paths()
    pages = {}
    for n in PAGES:
        f = Path(paths["out"]) / n
        if f.exists():
            pages[n] = f.read_text(encoding="utf-8", errors="replace")
    return audit_pages(pages), sorted(pages)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check", action="store_true", help="exit 1 if the site is stale relative to its sources")
    ap.add_argument("--built-at", default=None, help="fixed timestamp text for reproducible output")
    a = ap.parse_args(argv)
    if a.verify:
        bad, names = verify()
        print(f"verified {len(names)} page(s): {'CLEAN' if not bad else 'LEAK'}")
        for b in bad:
            print("  " + b)
        return 2 if bad else 0
    if a.check:
        fresh, why = check_fresh()
        print("site is fresh" if fresh else "site is STALE")
        for w in why[:20]:
            print("  " + w)
        return 0 if fresh else 1
    try:
        m = build(built_at=a.built_at, dry_run=a.dry_run)
    except S.Leak as e:
        print(f"REFUSED: {e}")
        return 2
    for n, info in m["pages"].items():
        print(f"{n}: {info['bytes']:,} bytes  {json.dumps(info['summary'], default=str)[:160]}")
    for k, v in m["problems"].items():
        for p in v:
            print(f"  problem [{k}]: {p}")
    print("dry run: nothing written" if a.dry_run else f"wrote {len(m['pages'])} pages and the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
