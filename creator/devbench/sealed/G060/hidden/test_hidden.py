from app.solution import pairs_sum


def test_case_0():
    assert pairs_sum(*([1, 2, 3, 4], 5)) == 2

def test_case_1():
    assert pairs_sum(*([], 1)) == 0

def test_case_2():
    assert pairs_sum(*([1, 1, 1], 2)) == 3

def test_case_3():
    assert pairs_sum(*([5], 10)) == 0
