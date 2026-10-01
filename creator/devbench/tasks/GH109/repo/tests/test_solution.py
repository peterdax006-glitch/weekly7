from app.solution import top_n


def test_case_0():
    assert top_n(*({'a': 3, 'b': 5, 'c': 3}, 2)) == ['b', 'a']
