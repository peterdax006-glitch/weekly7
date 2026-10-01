from app.solution import binary_search


def test_case_0():
    assert binary_search(*([1, 3, 5, 7], 5)) == 2

def test_case_1():
    assert binary_search(*([1, 3, 5, 7], 4)) == -1

def test_case_2():
    assert binary_search(*([], 1)) == -1

def test_case_3():
    assert binary_search(*([2], 2)) == 0

def test_case_4():
    assert binary_search(*([1, 2, 3, 4, 5, 6], 6)) == 5
