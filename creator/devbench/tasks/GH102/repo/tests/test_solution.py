from app.solution import split_even_odd


def test_case_0():
    assert split_even_odd(*([1, 2, 3, 4],)) == ([2, 4], [1, 3])
