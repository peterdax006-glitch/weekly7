from app.solution import word_freq


def test_case_0():
    assert word_freq(*('',)) == {}
