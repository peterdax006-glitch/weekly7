from app.solution import prefix_sums


def test_case_0():
    assert prefix_sums(*([1, 2, 3],)) == [1, 3, 6]

def test_case_1():
    assert prefix_sums(*([],)) == []

def test_case_2():
    assert prefix_sums(*([5, -5, 5],)) == [5, 0, 5]

def test_case_3():
    assert prefix_sums(*([0],)) == [0]
