from app.solution import max_window


def test_case_0():
    assert max_window(*([1, 2, 3, 4], 2)) == 7

def test_case_1():
    assert max_window(*([5], 1)) == 5

def test_case_2():
    assert max_window(*([1, 2], 3)) == None

def test_case_3():
    assert max_window(*([-1, -2, -3], 2)) == -3

def test_case_4():
    assert max_window(*([1, 2], 0)) == None
