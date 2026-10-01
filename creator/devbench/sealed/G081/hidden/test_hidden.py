from app.solution import transpose


def test_case_0():
    assert transpose(*([[1, 2, 3], [4, 5, 6]],)) == [[1, 4], [2, 5], [3, 6]]

def test_case_1():
    assert transpose(*([],)) == []

def test_case_2():
    assert transpose(*([[1]],)) == [[1]]

def test_case_3():
    assert transpose(*([[1, 2]],)) == [[1], [2]]
