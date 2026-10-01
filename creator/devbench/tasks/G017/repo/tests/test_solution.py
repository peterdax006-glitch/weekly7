from app.solution import dedupe


def test_case_0():
    assert dedupe(*([1, 2, 1, 3, 2],)) == [1, 2, 3]
