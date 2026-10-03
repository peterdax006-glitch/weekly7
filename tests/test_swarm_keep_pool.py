"""h38 (3 Oct 2026): in THINKING focus an idle round keeps the thinking servers loaded; RAM_TIGHT and a drain still close them."""
from __future__ import annotations

import json
from pathlib import Path

from creator import swarm as W


def test_thinking_focus_keeps_the_pool_after_an_empty_round(tmp_path: Path) -> None:
    assert not W._keep_pool("NOTHING_TO_DO", tmp_path)                       # no focus file: the old behaviour (close)
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "thinking"}), encoding="utf-8")
    assert W._keep_pool("NOTHING_TO_DO", tmp_path) and W._keep_pool("AUDIT_RED", tmp_path)
    assert not W._keep_pool("RAM_TIGHT", tmp_path) and not W._keep_pool("DRAINED", tmp_path)
    (tmp_path / "focus.json").write_text(json.dumps({"focus": "development"}), encoding="utf-8")
    assert not W._keep_pool("NOTHING_TO_DO", tmp_path)
