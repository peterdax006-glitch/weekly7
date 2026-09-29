"""Scheduler dispatcher. GitHub cron fires every 30 min (UTC, and sometimes late); this works
out from New York time which job is due, and uses markers so nothing runs twice a day.
Handles daylight-saving changes and market holidays without touching the cron."""
import json, sys, traceback
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config as K

ET = ZoneInfo("America/New_York")
MARK = K.STATE / "markers.json"
HEART = K.STATE / "heartbeat.json"


def market_open_today(now) -> bool:
    if now.weekday() >= 5:
        return False
    try:
        from .broker import get_broker                            # tick is live-side: it schedules the live jobs
        open_ = get_broker().is_session(now.date())
        if open_ is not None:
            return open_
        import yfinance as yf
        d = yf.download("SPY", period="1d", interval="5m", progress=False)
        return len(d) > 0 and d.index[-1].tz_convert(ET).date() == now.date()
    except Exception:
        return True


def due(now, marks):
    today = now.strftime("%Y-%m-%d")
    hm = now.hour * 60 + now.minute
    jobs = []
    if now.weekday() == 5 and marks.get("weekly") != today:
        return ["weekly"]
    if now.weekday() >= 5 or not market_open_today(now):
        return []
    if 9 * 60 + 45 <= hm < 16 * 60:
        jobs.append("risk")
    if 15 * 60 + 5 <= hm < 16 * 60 and marks.get("decide") != today:
        jobs.append("decide")
    if hm >= 16 * 60 + 5 and marks.get("close") != today:
        jobs.append("close")
    return jobs


def main(force=None):
    from . import live
    now = datetime.now(ET)
    marks = json.loads(MARK.read_text()) if MARK.exists() else {}
    jobs = [force] if force else due(now, marks)
    beat = {"t": now.isoformat(timespec="seconds"), "jobs": jobs, "errors": []}
    for j in jobs:
        try:
            {"risk": live.risk, "decide": live.decide, "close": live.close,
             "weekly": lambda: __import__("engine.improve", fromlist=["x"]).weekly()}[j]()
            if j != "risk":
                marks[j] = now.strftime("%Y-%m-%d")
        except Exception:
            beat["errors"].append({"job": j, "trace": traceback.format_exc()[-2000:]})
            traceback.print_exc()
    MARK.write_text(json.dumps(marks, indent=1))
    HEART.write_text(json.dumps(beat, indent=1))
    if beat["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
