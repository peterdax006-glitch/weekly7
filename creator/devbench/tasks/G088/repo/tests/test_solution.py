from app.solution import prefix_sums


def test_case_0():
    assert prefix_sums(*([1, 2, 3],)) == [1, 3, 6]
