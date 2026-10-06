"""Compile Jelly source with the local Open-Jellycore build (WSL) and print diagnostics.

    python scripts/jelly_compile.py FILE.jelly [more.jelly | stored_shortcut.json ...] [--export OUT.shortcut] [--json]

A .json file is a stored Nupen shortcut (<runtime>/phone/shortcuts/<id>.json); its "code" field is compiled.
Exit code: 0 all compiled, 1 at least one error, 2 compiler not installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from creator import jellyc as J  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--export", help="with one input file: also write the compiled shortcut plist here")
    ap.add_argument("--json", action="store_true", help="one JSON line per file")
    ap.add_argument("--warnings", action="store_true", help="also print warnings (e.g. optional parameters left out)")
    a = ap.parse_args(argv)
    if not J.available():
        print("jelly compiler not installed: needs WSL Ubuntu + Open-Jellycore built (see phone/PHONE_CONTROL.md)", file=sys.stderr)
        return 2
    bad = 0
    for name in a.files:
        p = Path(name)
        tmp = None
        if p.suffix.lower() == ".json":
            code = str(json.loads(p.read_text(encoding="utf-8")).get("code", ""))
            tmp = Path(tempfile.mkdtemp(prefix="jellyc_")) / (p.stem + ".jelly")
            tmp.write_text(code, encoding="utf-8", newline="\n")
        r = J.compile_file(tmp or p, export=Path(a.export) if a.export and len(a.files) == 1 else None)
        if tmp:
            tmp.unlink(missing_ok=True)
            tmp.parent.rmdir()
        bad += not r["ok"]
        if a.json:
            print(json.dumps({"file": name, "ok": r["ok"], "errors": r["errors"], "warnings": r.get("warnings", [])}))
        else:
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {name}  ({len(r.get('warnings', []))} warnings)")
            for e in r["errors"]:
                print(f"     {e}")
            if a.warnings:
                for w in r.get("warnings", []):
                    print(f"     warning: {w}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
