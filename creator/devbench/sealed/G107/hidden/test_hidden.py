from app.solution import normalize_spaces


def test_case_0():
    assert normalize_spaces(*('  a   b  ',)) == 'a b'

def test_case_1():
    assert normalize_spaces(*('',)) == ''

def test_case_2():
    assert normalize_spaces(*('x',)) == 'x'

def test_case_3():
    assert normalize_spaces(*('a\\tb\\nc',)) == 'a\\tb\\nc'
