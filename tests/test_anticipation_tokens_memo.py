"""h38 (3 Oct 2026): anticipation.tokens is memoised per text; results, including set iteration order, equal the uncached tokenizer."""
from __future__ import annotations

from creator import anticipation as A


def _uncached(text: str) -> set[str]:
    out = set()
    for w in A._words(text):
        out.add(w)
    return out


def test_memo_equals_uncached_including_order_and_is_not_shared() -> None:
    A._TOKENS.clear()
    text = "Nupen should keep thinking about the thinking drills, tests and testing; reviewing reviewed reviews of the ledger"
    first = A.tokens(text)
    assert text in A._TOKENS                                              # old code: no memo
    again = A.tokens(text)
    ref = _uncached(text)
    assert first == again == ref and list(first) == list(again) == list(ref)
    first.add("mutated")
    assert "mutated" not in A.tokens(text)


def test_memo_is_bounded() -> None:
    A._TOKENS.clear()
    for i in range(20001):
        A.tokens(f"word{i} something")
    assert len(A._TOKENS) <= 20000
