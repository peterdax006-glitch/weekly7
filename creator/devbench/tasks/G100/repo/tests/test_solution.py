from app.solution import max_window


def test_case_0():
    assert max_window(*([1, 2, 3, 4], 2)) == 7
