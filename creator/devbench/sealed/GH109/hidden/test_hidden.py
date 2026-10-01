from app.solution import top_n


def test_case_0():
    assert top_n(*({'a': 3, 'b': 5, 'c': 3}, 2)) == ['b', 'a']

def test_case_1():
    assert top_n(*({}, 1)) == []

def test_case_2():
    assert top_n(*({'x': 1}, 3)) == ['x']

def test_case_3():
    assert top_n(*({'b': 2, 'a': 2}, 1)) == ['a']
