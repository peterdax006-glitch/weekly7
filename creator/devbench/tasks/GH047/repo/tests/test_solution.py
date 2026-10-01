from app.solution import flatten


def test_case_0():
    assert flatten(*([1, [2, [3, 4]], 5],)) == [1, 2, 3, 4, 5]
