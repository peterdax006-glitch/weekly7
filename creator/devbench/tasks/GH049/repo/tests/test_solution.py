from app.solution import binary_search


def test_case_0():
    assert binary_search(*([1, 3, 5, 7], 5)) == 2
