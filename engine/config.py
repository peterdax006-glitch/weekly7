"""Central settings. Numbers here are the blueprint's numbers; the learning loop (Part M)
may only change the ones listed in TUNABLE, and only through a promoted challenger."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
STATE = ROOT / "state"
SITE = ROOT / "docs"
for p in (DATA, CACHE, STATE, SITE):
    p.mkdir(parents=True, exist_ok=True)

SEC_UA = "Weekly7 research peterdax006-glitch@users.noreply.github.com"  # SEC requires a contact User-Agent

START_CASH = 1000.0
WEEKLY_TARGET = 0.07
HISTORY_START = "2012-01-01"

# Universe (Part 4.1 / B)
MIN_PRICE = 3.0
MIN_DOLLAR_VOL = 5e6          # 20-day median dollar volume

# Portfolio construction (Part D2)
N_CANDIDATES = 40
MIN_NAMES, MAX_NAMES = 4, 10
MAX_WEIGHT = 0.25
MAX_SECTOR = 0.40
MAX_SQUEEZE = 0.10
MAX_EARNINGS_HOLDS = 2
CVAR_LIMIT = -0.12            # mean of worst 5% of simulated weeks
N_SCENARIOS = 20000
REBALANCE_BAND = 0.03

# Within-week control (Part E)
BANK_LEVEL = 0.07
BANK_EXPOSURE = 0.40
BRAKE_LEVEL = -0.08
BRAKE_EXPOSURE = 1 / 3
MAX_CATCHUP = 1.25
REGIME_GROSS = {"calm": 1.0, "choppy": 0.8, "stress": 0.4}

# Stops (Part 4.6)
STOP_ATR = 2.0
TIME_STOP_DAYS = 5
PARTIAL_TAKE = 0.12

# Costs for backtests, per side
COST_BPS_LIQUID = 5
COST_BPS_ILLIQUID = 30

TUNABLE = ["BANK_LEVEL", "BANK_EXPOSURE", "BRAKE_LEVEL", "STOP_ATR", "TIME_STOP_DAYS",
           "PARTIAL_TAKE", "REGIME_GROSS", "CVAR_LIMIT", "N_CANDIDATES"]

# Promoted challengers (Part M4) override tunables here; nothing else may change them.
import json as _json
_ov = STATE / "config_overrides.json"
if _ov.exists():
    for _k, _v in _json.loads(_ov.read_text()).items():
        if _k in TUNABLE:
            globals()[_k] = _v
