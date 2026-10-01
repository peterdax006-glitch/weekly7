from app.wordcount import top_k

def test_top_k():
    assert top_k("b a b c a b", 2) == [("b", 3), ("a", 2)]

def test_ties_alphabetical():
    assert top_k("zeta alpha mid", 3) == [("alpha", 1), ("mid", 1), ("zeta", 1)]

def test_k_larger_than_vocab():
    assert top_k("one", 5) == [("one", 1)] and top_k("", 3) == []
