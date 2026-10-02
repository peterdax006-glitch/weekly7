"""Pull the minute-candle archive from the intraday-YYYY-MM releases into data/intraday/<interval>/<date>.parquet.
Only downloads days that are missing locally or newer upstream (by asset size). Usage: python scripts/sync_intraday.py
[--months N]  (default: all months)."""
import argparse, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import config as K

GH = str(Path.home() / "tools" / "gh" / "bin" / "gh.exe")      # falls back to gh on PATH
OUT = K.DATA / "intraday"


def gh(*args):
    exe = GH if Path(GH).exists() else "gh"
    r = subprocess.run([exe, *args], cwd=K.ROOT, capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[:400])
    return r.stdout


def releases():
    rows = json.loads(gh("release", "list", "--limit", "200", "--json", "tagName"))
    return sorted(r["tagName"] for r in rows if r["tagName"].startswith("intraday-"))


def assets(tag):
    return json.loads(gh("release", "view", tag, "--json", "assets"))["assets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=0)
    a = ap.parse_args()
    tags = releases()
    if a.months:
        tags = tags[-a.months:]
    got = 0
    for tag in tags:
        for asset in assets(tag):
            name = asset["name"]                                  # e.g. 1m_2026-09-28.parquet
            if not name.endswith(".parquet") or "_" not in name:
                continue
            iv, day = name[:-8].split("_", 1)
            dest = OUT / iv / f"{day}.parquet"
            if dest.exists() and dest.stat().st_size >= asset.get("size", 0):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            gh("release", "download", tag, "--pattern", name, "--dir", str(dest.parent), "--clobber")
            (dest.parent / name).replace(dest)
            got += 1
    print(f"synced {got} day files from {len(tags)} monthly releases")


if __name__ == "__main__":
    main()
