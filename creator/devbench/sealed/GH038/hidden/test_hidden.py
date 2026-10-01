from app.solution import word_freq


def test_case_0():
    assert word_freq(*('the cat the hat',)) == {'the': 2, 'cat': 1, 'hat': 1}

def test_case_1():
    assert word_freq(*('',)) == {}

def test_case_2():
    assert word_freq(*('Hi, hi!',)) == {'hi': 2}

def test_case_3():
    assert word_freq(*('a1b',)) == {'a': 1, 'b': 1}
