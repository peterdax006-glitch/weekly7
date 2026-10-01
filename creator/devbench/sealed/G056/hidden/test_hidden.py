from app.solution import second_largest


def test_case_0():
    assert second_largest(*([3, 1, 4, 4],)) == 3

def test_case_1():
    assert second_largest(*([5],)) == None

def test_case_2():
    assert second_largest(*([],)) == None

def test_case_3():
    assert second_largest(*([2, 2],)) == None

def test_case_4():
    assert second_largest(*([-1, -2, -3],)) == -2
