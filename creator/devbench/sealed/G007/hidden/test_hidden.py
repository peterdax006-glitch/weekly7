from app.solution import running_max


def test_case_0():
    assert running_max(*([3, 1, 4, 1, 5],)) == [3, 3, 4, 4, 5]

def test_case_1():
    assert running_max(*([],)) == []

def test_case_2():
    assert running_max(*([2, 2, 1],)) == [2, 2, 2]

def test_case_3():
    assert running_max(*([-1, -5, 0],)) == [-1, -1, 0]
