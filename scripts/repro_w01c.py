"""Bible phase 23/33: rerun window w01c exactly as the loop's worker does (fresh process, same cfg/meta) and compare."""
import json, subprocess, sys, shutil
from pathlib import Path
root = Path(__file__).resolve().parent.parent
res = json.loads((root / "state/livesim/w01c/result2.json").read_text())
bak = root / "state/livesim/_w01c_original"
if not bak.exists():
    shutil.copytree(root / "state/livesim/w01c", bak)
r = subprocess.run([sys.executable, "-u", "scripts/livesim_loop2.py", "--worker", "w01c", json.dumps(res["prior_cfg"]), json.dumps(res["meta"])],
                   cwd=root, capture_output=True, text=True)
new = json.loads((root / "state/livesim/w01c/result2.json").read_text())
print("original:", round(res["year_return"], 4), [(a["date"][:10], a.get("knob"), a["action"]) for a in res["adaptations"]])
print("rerun   :", round(new["year_return"], 4), [(a["date"][:10], a.get("knob"), a["action"]) for a in new["adaptations"]])
print("same preseason:", res["preseason"] == new["preseason"], "| same used_cfg:", res["used_cfg"] == new["used_cfg"])
print(r.stderr[-800:] if r.returncode else "worker ok")
