"""S17b smoke: loop2's worker path with --learner legit on ONE synthetic window (wiring proof, not a result)."""
import sys, json, tempfile, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from engine import livesim, blind_gates as BG, leak_audit
import test_learning_test_path as T
home = Path(tempfile.mkdtemp()); livesim.DIR = home
sys.argv = ["livesim_loop2.py", "--learner", "legit"]
spec = importlib.util.spec_from_file_location("loop2smoke", ROOT / "scripts" / "livesim_loop2.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
assert L.LEARNER == "legit"
L.DIR = home; L.CURATOR_ROOT = home / "curator"
rec = BG.seal_window([], 7, "2026-01-01", tag="s1"); rec.update(start=T.START, shift_days=7 * 9000); rec["digest"] = BG.seal_digest(rec)
(home / "sealed_s1.json").write_text(json.dumps(rec))
base = leak_audit.hardened_feed_class()
class F(base):
    def __init__(self, sealed, **k):
        super().__init__(sealed, warmup_years=3, data=T.make_data(T.START, n=120, years=3), use_insider=False)
livesim.blind_feed_class = lambda hardened=True: F
livesim.parity_test = lambda *a, **k: 0.0
L.livesim = livesim
L._worker("s1", L.NEUTRAL_CFG, dict(L.A.META_DEFAULT))
r = json.loads((home / "s1" / "result2.json").read_text())
print("RESULT keys ok:", "legit" in r, "mean_week", r["mean_week"])
print(json.dumps({k: r["legit"][k] for k in ("counts", "filed", "curator_years_filed", "findings", "release_date_hits")}, indent=1))
print("audit:", json.loads((home / "s1" / "blind_audit2.json").read_text()) if (home / "s1" / "blind_audit2.json").exists() else None)
