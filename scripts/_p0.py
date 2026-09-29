import pathlib
p = pathlib.Path("engine/improve.py"); s = p.read_text(encoding="utf-8")
old = '''def log_experiment(rec):
    rec = {"t": datetime.utcnow().isoformat(timespec="seconds"), **rec}'''
new = '''def log_experiment(rec, cfg=None, seed=None):
    """Append-only registry (Bible Phase 0): every record carries provenance; nothing is ever overwritten."""
    from .provenance import stamp
    rec = {"t": datetime.utcnow().isoformat(timespec="seconds"), **stamp(cfg, seed), **rec}
    rec.setdefault("experiment_id", f"E{abs(hash((rec['t'], rec.get('event'), rec.get('run_id')))) % 10**10:010d}")'''
assert old in s; s = s.replace(old, new); p.write_text(s, encoding="utf-8", newline="\n")
