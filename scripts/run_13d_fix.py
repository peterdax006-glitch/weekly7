"""Resumable 13D activist attribution fix since 2001 (engine.edgar.fix_activist checkpoints per company to
data/cache/13d_fix_progress_2001-01-01.json). Launch detached so a session switch cannot kill it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from engine import config as K, edgar

edgar.fix_activist(pd.read_csv(K.CACHE / "universe.csv"), since="2001-01-01")
print("13D fix done", flush=True)
