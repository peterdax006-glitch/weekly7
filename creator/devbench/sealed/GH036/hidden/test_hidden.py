from app.solution import merge_sorted


def test_case_0():
    assert merge_sorted(*([1, 3, 5], [2, 4])) == [1, 2, 3, 4, 5]

def test_case_1():
    assert merge_sorted(*([], [1])) == [1]

def test_case_2():
    assert merge_sorted(*([1, 1], [1])) == [1, 1, 1]

def test_case_3():
    assert merge_sorted(*([], [])) == []
