from app.solution import second_largest


def test_case_0():
    assert second_largest(*([3, 1, 4, 4],)) == 3
