"""Save each archived window's opening prices (canon C33: fills at the next open)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import livesim
n = 0
for a in sorted(livesim.DIR.iterdir()):
    if a.is_dir() and (livesim.DIR / f"sealed_{a.name}.json").exists() and not (a / "opens_v2.parquet").exists():
        f = livesim.Feed(livesim.SealedYear(a.name))
        f._stocks["Open"].loc[f.first_live:].to_parquet(a / "opens_v2.parquet"); n += 1
print("opens archived:", n)
