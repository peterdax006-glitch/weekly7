from app.solution import median_of


def test_case_0():
    assert median_of(*([3, 1, 2],)) == 2

def test_case_1():
    assert median_of(*([4, 1, 3, 2],)) == 2.5

def test_case_2():
    assert median_of(*([5],)) == 5

def test_case_3():
    assert median_of(*([1, 1, 2, 2],)) == 1.5
