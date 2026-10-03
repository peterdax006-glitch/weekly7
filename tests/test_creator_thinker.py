"""The 'thinker' role: a separate, bigger THINKING model for judgment/reasoning (device 'think_model'), RAM-gated, falling back to the fast model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from creator import device as DEV
from creator import generator as G
from creator import judgment as J
from creator import thinking as T


def dev(ram: float = 33.8) -> DEV.Device:
    return DEV.Device(os="Windows", arch="x64", cores_logical=14, cores_physical=14, ram_gb=ram, gpu_name="", vram_gb=0.0, python="p", home="h",
                      runtime_dir="r", sandbox_root="s")


def test_think_model_setting_resolves_names_and_paths_and_missing_means_none(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    f = tmp_path / "models" / "think.gguf"
    f.write_bytes(b"x" * 2048)
    assert DEV.think_model_path({"think_model": "think.gguf"}, tmp_path) == f
    assert DEV.think_model_path({"think_model": str(f)}, tmp_path) == f
    assert DEV.think_model_path({"think_model": "absent.gguf"}, tmp_path) is None
    assert DEV.think_model_path({"think_model": ""}, tmp_path) is None


def test_server_size_follows_the_model_file_and_think_servers_follow_ram(tmp_path: Path) -> None:
    big = tmp_path / "big.gguf"
    with big.open("wb") as fh:
        fh.truncate(int(3 * 2**30))
    assert DEV.server_gb_for(big) == 3.7 and DEV.server_gb_for(tmp_path / "missing") == DEV.SERVER_GB
    s = DEV.derive(dev(), overrides={"think_model": "t.gguf"})
    assert s["think_model"] == "t.gguf" and s["think_servers"] == 2           # 25% of 33.8 GB fits two, capped at two
    assert DEV.derive(dev(8.0), overrides={"think_model": "t.gguf"})["think_servers"] == 0     # a small machine keeps one fast model only
    assert DEV.derive(dev(), overrides={})["think_servers"] == 0              # nothing configured, nothing started
    assert DEV.derive(dev(), overrides={"think_model": "t.gguf", "think_servers": 1})["think_servers"] == 1   # the owner can pin it


def _cfg(tmp_path: Path, **kw: Any) -> tuple[dict[str, Any], Path]:
    f = tmp_path / "think.gguf"
    f.write_bytes(b"x" * 1024)
    return {"think_model": str(f), "think_servers": 2, "llama_servers": 3, "llama_threads": 6, "gpu_layers": 0, **kw}, f


def test_thinker_uses_the_thinking_model_when_ram_allows_else_the_fast_model(tmp_path: Path) -> None:
    cfg, f = _cfg(tmp_path)
    assert G.thinker(free_gb=lambda: 30.0, cfg=cfg).model == f
    assert G.thinker(free_gb=lambda: 1.0, cfg=cfg).model == G.DEFAULT_MODEL           # no room: never pulled below the floor
    assert G.thinker(free_gb=lambda: None, cfg=cfg).model == G.DEFAULT_MODEL          # unknown free RAM never opens the gate
    assert G.thinker(free_gb=lambda: 30.0, cfg=dict(cfg, think_servers=0)).model == G.DEFAULT_MODEL
    assert G.thinker(free_gb=lambda: 30.0, cfg=dict(cfg, think_model="")).model == G.DEFAULT_MODEL
    assert G.thinker(free_gb=lambda: 30.0, cfg=dict(cfg, think_model=str(tmp_path / "gone.gguf"))).model == G.DEFAULT_MODEL


def test_the_code_edit_model_is_untouched_by_the_thinker_setting(tmp_path: Path) -> None:
    assert G.LocalModel().model == G.DEFAULT_MODEL


def test_think_blocks_are_stripped_from_answers() -> None:
    assert G.THINK_BLOCK.sub("", "<think>hmm\nmore</think>\nIt is risky.\nPROBABILITY: 0.20").strip() == "It is risky.\nPROBABILITY: 0.20"

    class L:
        def chat(self, m: Any, **kw: Any) -> str:
            return "<think>\nlong scratch work\n</think>\nok.\nPROBABILITY: 0.3"
    assert J.parse(J.chat_text(L(), [])) == 0.3


class _Fake:
    def __init__(self, model: Path, log: list[Any]) -> None:
        self.model, self.log = model, log

    def __enter__(self) -> "_Fake":
        return self

    def __exit__(self, *a: Any) -> None:
        pass

    def chat(self, messages: Any, **kw: Any) -> str:
        self.log.append(messages)
        return "<think>x</think>fine.\nPROBABILITY: 0.60"


def test_records_carry_the_model_and_each_models_trust_is_scored_apart(tmp_path: Path, monkeypatch: Any) -> None:
    st = tmp_path / "st"
    st.mkdir()
    cs = [J.Case("verdict", f"p{i}", 1000.0 + i * 10, 1000.0 + i * 10 + 15, int(i % 2 == 0), "K", f"case {i}") for i in range(30)]
    monkeypatch.setattr(J, "load_cases", lambda topic, state, repo: cs if topic == "verdict" else [])
    monkeypatch.setattr(J, "active_tag", lambda: "think.gguf")
    nj = J.judgment_filler(st, tmp_path, llm_factory=lambda: _Fake(tmp_path / "think.gguf", []))
    job = nj()
    assert job is not None
    job()
    rows = T._jsonl(J.path(st))
    assert rows and all(r["model"] == "think.gguf" and r["p"] == 0.6 for r in rows)
    J._append(st, {"topic": "verdict", "subject": "old", "strategy": {"shots": 0, "hint": 0}, "created": 1.0, "resolved": 2.0, "y": 1, "p": 0.9})   # legacy row, no model
    assert [r["subject"] for r in J._mine(T._jsonl(J.path(st)), "think.gguf")] == [r["subject"] for r in rows]
    assert [r["subject"] for r in J._mine(T._jsonl(J.path(st)), J.LEGACY_TAG)] == ["old"]


def test_a_batch_waits_silently_when_the_thinking_model_has_no_ram(tmp_path: Path, monkeypatch: Any) -> None:
    st = tmp_path / "st"
    st.mkdir()
    cs = [J.Case("verdict", f"p{i}", 1000.0 + i * 10, 1000.0 + i * 10 + 15, i % 2, "K", f"case {i}") for i in range(30)]
    monkeypatch.setattr(J, "load_cases", lambda topic, state, repo: cs if topic == "verdict" else [])
    monkeypatch.setattr(J, "active_tag", lambda: "think.gguf")
    nj = J.judgment_filler(st, tmp_path, llm_factory=lambda: (_ for _ in ()).throw(J._Skip()))
    job = nj()
    assert job is not None
    job()
    assert not J.path(st).exists()                                   # no error row, no answer of the wrong model
