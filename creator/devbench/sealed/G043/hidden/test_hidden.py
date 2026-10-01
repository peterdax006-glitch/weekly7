from app.solution import rotate


def test_case_0():
    assert rotate(*([1, 2, 3, 4], 1)) == [4, 1, 2, 3]

def test_case_1():
    assert rotate(*([1, 2, 3], 5)) == [2, 3, 1]

def test_case_2():
    assert rotate(*([], 3)) == []

def test_case_3():
    assert rotate(*([1, 2], 0)) == [1, 2]
