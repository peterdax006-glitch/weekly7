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
    assert DEV.derive(dev(), overrides={"think_model": ""})["think_servers"] == 0    # nothing configured, nothing started
    assert DEV.derive(dev(), overrides={})["think_model"] == "Qwen3-1.7B-Q4_K_M.gguf"      # the measured default
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


def test_qwen3_gets_the_no_think_switch_and_other_models_do_not() -> None:
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    assert G.prepare_messages(m, Path("x/Qwen3-1.7B-Q4_K_M.gguf"))[-1]["content"] == "q\n/no_think"
    assert G.prepare_messages(m, Path("qwen2.5-coder-1.5b-instruct-q4_k_m.gguf")) == m and m[-1]["content"] == "q"      # input never mutated


# ------------------------------------------------------------------ a bigger thinker (h26, 3 Oct 2026: Qwen3-4B / 8B candidates)

def _sparse(p: Path, gib: float) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.truncate(int(gib * 2**30))
    return p


def test_threads_follow_the_servers_that_really_run_at_once() -> None:
    cfg = {"llama_threads": 11}
    assert [DEV.server_threads(cfg, n) for n in (1, 2, 4, 7, 14)] == [11, 4, 4, 3, 2]
    assert DEV.server_threads(cfg, 0) == 11                                  # never divides by zero


def test_timeouts_grow_with_the_model_and_keep_the_fast_models_values(tmp_path: Path) -> None:
    assert DEV.call_timeout_s(tmp_path / "missing.gguf", 300.0) == 300.0
    assert DEV.call_timeout_s(_sparse(tmp_path / "small.gguf", 1.0), 300.0) == 300.0           # 1 GiB: the fast model's size
    big = _sparse(tmp_path / "big.gguf", 4.7)                                                  # an 8B Q4_K_M
    assert DEV.call_timeout_s(big, 300.0) == round(300.0 * (4.7 + DEV.THINK_OVERHEAD_GB) / DEV.SERVER_GB, 1) > 850


def test_think_servers_are_sized_by_the_thinking_models_own_file(tmp_path: Path) -> None:
    d = DEV.Device(os="Windows", arch="x64", cores_logical=14, cores_physical=12, ram_gb=33.8, gpu_name="", vram_gb=0.0, python="p", home="h",
                   runtime_dir=str(tmp_path), sandbox_root="s")
    _sparse(tmp_path / "models" / "small.gguf", 1.0)
    _sparse(tmp_path / "models" / "big.gguf", 4.7)
    assert DEV.derive(d, overrides={"think_model": "small.gguf"})["think_servers"] == 2
    assert DEV.derive(d, overrides={"think_model": "big.gguf"})["think_servers"] == 1          # 5.4 GB servers: one in 25% of 33.8 GB
    assert DEV.derive(d, overrides={"think_model": "big.gguf", "think_servers": 4})["think_servers"] == 4   # the owner's pin still wins


def test_a_cold_thinker_gets_the_threads_and_startup_of_its_size(tmp_path: Path) -> None:
    big = _sparse(tmp_path / "Qwen3-8B-Q4_K_M.gguf", 4.7)
    cfg = {"think_model": str(big), "think_servers": 2, "llama_servers": 7, "llama_threads": 11, "gpu_layers": 0}
    lm = G.thinker(free_gb=lambda: 30.0, cfg=cfg)
    assert lm.model == big and lm.threads == 4 and lm.startup_s > 300           # 2 thinkers -> 4 threads each, not the 3 of 7 slots
    assert G.thinker(free_gb=lambda: 30.0, cfg=dict(cfg, think_servers=1)).threads == 11
    assert G.thinker(free_gb=lambda: 30.0, cfg=cfg, threads=5).threads == 5      # an explicit choice wins


def test_a_loading_big_thinker_is_counted_at_its_own_size(tmp_path: Path) -> None:
    big = _sparse(tmp_path / "big.gguf", 4.7)
    lm = G.LocalModel(model=big, pidfile=tmp_path / "llama_server.pid", servers=3)
    lm.free_gb = lambda: 12.0
    assert lm._ram_allows_extra_server()                       # 12 - 5.4 leaves room
    (tmp_path / "llama_server.1.starting").write_text("x", encoding="utf-8")       # another big server is loading and has not taken its RAM
    assert not lm._ram_allows_extra_server()                   # 12 - 2 x 5.4 does not (it was counted as 1.8 before)


def test_an_unclosed_think_block_is_scratch_work_too() -> None:
    assert G.THINK_BLOCK.sub("", "<think>still thinking when max_tokens ran out").strip() == ""
    assert G.THINK_BLOCK.sub("", "a<think>x</think>b<think>y").strip() == "ab"


def test_qwen3_4b_and_8b_files_get_the_no_think_switch() -> None:
    m = [{"role": "user", "content": "q"}]
    for f in ("Qwen3-4B-Q4_K_M.gguf", "Qwen3-8B-Q4_K_M.gguf"):
        assert G.prepare_messages(m, Path(f))[-1]["content"] == "q\n/no_think"


def test_a_thinking_pool_sizes_threads_by_its_max_servers(tmp_path: Path) -> None:
    from creator import modelpool as MP
    p = MP.ModelPool(base_pidfile=tmp_path / "llama_server.pid", slots=7, max_servers=2, floor_gb=1.0, free_gb=lambda: 30.0)
    try:
        assert p.threads == DEV.server_threads(DEV.settings(), 2)
    finally:
        p.close()


def test_judgment_calls_get_a_timeout_scaled_to_the_thinking_model(tmp_path: Path, monkeypatch: Any) -> None:
    big = _sparse(tmp_path / "Qwen3-8B-Q4_K_M.gguf", 4.7)
    seen: list[float] = []

    class L(_Fake):
        def chat(self, messages: Any, **kw: Any) -> str:
            seen.append(kw["timeout"])
            return "PROBABILITY: 0.5"
    st = tmp_path / "st"
    st.mkdir()
    cs = [J.Case("verdict", f"p{i}", 1000.0 + i * 10, 1000.0 + i * 10 + 15, i % 2, "K", f"case {i}") for i in range(6)]
    J.run_batch(st, [(cs[-1], {"shots": 0, "hint": 0})], cs, lambda: L(big, []))
    J.run_batch(st, [(cs[-1], {"shots": 0, "hint": 0})], cs, lambda: L(tmp_path / "missing.gguf", []))
    assert seen == [DEV.call_timeout_s(big, 300.0), 300.0] and seen[0] > 850
