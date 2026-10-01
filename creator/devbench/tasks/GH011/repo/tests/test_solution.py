from app.solution import chunk


def test_case_0():
    assert chunk(*([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
