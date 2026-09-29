# B08_exits_stops
Bible: PHASES 15 and 16 (lines 1076-1141). Estimated code: 2,000-4,000 lines.
You own: engine/exits.py, engine/stops.py, tests/test_exits.py, tests/test_stops.py

Exit learner (take-profit/time/trailing rules learned walk-forward from path data, per type) and stop/loss engine (losers never worse than -20% target, gap-risk aware since fills at next open; ATR/vol-scaled stops learned walk-forward).
