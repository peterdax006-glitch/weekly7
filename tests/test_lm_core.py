"""Nupen LM core: tokenizer, promotion rule, api fallback. Torch-free except where importorskip'd."""
from __future__ import annotations

import json

import pytest

from creator.lm import evaluate
from creator.lm.tokenizer import BPETokenizer

TEXT = ("Once upon a time there was a little girl named Lily. She loved her red ball. " * 20
        + "The dog ran after the ball. café ☃ 123 <|endoftext|> Another story begins here.\n")


def test_tokenizer_round_trip_and_determinism(tmp_path):
    a = BPETokenizer.train(TEXT, 300)
    b = BPETokenizer.train(TEXT, 300)
    assert a.merges == b.merges and len(a.merges) > 10
    for s in ("Once upon a time", "café ☃!  two  spaces\n\nnew", "", "x <|endoftext|> y"):
        assert a.decode(a.encode(s)) == s
    assert len(a.encode("Lily loved her red ball")) < len("Lily loved her red ball")
    p = tmp_path / "t.json"
    a.save(p)
    c = BPETokenizer.load(p)
    assert c.encode(TEXT) == a.encode(TEXT)


def _res(bpb, half, sha="s"):
    return {"bpb": bpb, "lo": bpb - half, "hi": bpb + half, "eval_sha256": sha}


def test_promotion_rule():
    cur = _res(2.0, 0.05)
    assert evaluate.should_promote(_res(1.5, 0.05), None)[0]
    assert evaluate.should_promote(_res(1.5, 0.05), cur)[0]
    assert not evaluate.should_promote(_res(1.95, 0.05), cur)[0]      # CIs overlap: noise
    assert not evaluate.should_promote(_res(2.5, 0.05), cur)[0]       # worse
    assert not evaluate.should_promote(_res(1.0, 0.05, sha="other"), cur)[0]   # different eval set


def test_summarize_ci_brackets_estimate():
    pytest.importorskip("numpy")
    per = [(100.0 + i, 100) for i in range(50)]
    r = evaluate.summarize(per)
    assert r["lo"] <= r["bpb"] <= r["hi"]


def test_load_current_none_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("CREATOR_RUNTIME", str(tmp_path))
    from creator.lm.api import load_current
    assert load_current() is None
    (tmp_path / "lmckpt").mkdir()
    (tmp_path / "lmckpt" / "current.json").write_text("not json")
    assert load_current() is None
    (tmp_path / "lmckpt" / "current.json").write_text(json.dumps({"weights": "missing.pt"}))
    assert load_current() is None


def test_tiny_model_generates():
    torch = pytest.importorskip("torch")
    from creator.lm.model import LMConfig, build_model
    cfg = LMConfig(vocab_size=300, n_layer=1, d_model=32, n_head=2, ctx=16)
    m = build_model(cfg)
    out = m.generate([1, 2, 3], 5, temperature=0.8)
    assert len(out) == 8 and torch is not None
