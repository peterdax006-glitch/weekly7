from app.solution import dedupe


def test_case_0():
    assert dedupe(*([1, 2, 1, 3, 2],)) == [1, 2, 3]

def test_case_1():
    assert dedupe(*([],)) == []

def test_case_2():
    assert dedupe(*(['a', 'a'],)) == ['a']

def test_case_3():
    assert dedupe(*([3, 2, 1],)) == [3, 2, 1]
