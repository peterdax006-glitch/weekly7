from app.solution import flatten


def test_case_0():
    assert flatten(*([1, [2, [3, 4]], 5],)) == [1, 2, 3, 4, 5]

def test_case_1():
    assert flatten(*([],)) == []

def test_case_2():
    assert flatten(*([[[]]],)) == []

def test_case_3():
    assert flatten(*([1, 2],)) == [1, 2]
