from app.solution import running_max


def test_case_0():
    assert running_max(*([3, 1, 4, 1, 5],)) == [3, 3, 4, 4, 5]
