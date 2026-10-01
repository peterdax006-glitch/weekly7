from app.solution import rotate


def test_case_0():
    assert rotate(*([1, 2, 3, 4], 1)) == [4, 1, 2, 3]
