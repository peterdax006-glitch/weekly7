from app.solution import chunk


def test_case_0():
    assert chunk(*([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

def test_case_1():
    assert chunk(*([], 3)) == []

def test_case_2():
    assert chunk(*([1, 2, 3], 3)) == [[1, 2, 3]]

def test_case_3():
    assert chunk(*([1], 5)) == [[1]]
