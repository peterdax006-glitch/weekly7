from app.solution import split_even_odd


def test_case_0():
    assert split_even_odd(*([1, 2, 3, 4],)) == ([2, 4], [1, 3])

def test_case_1():
    assert split_even_odd(*([],)) == ([], [])

def test_case_2():
    assert split_even_odd(*([0, -1],)) == ([0], [-1])

def test_case_3():
    assert split_even_odd(*([7],)) == ([], [7])
