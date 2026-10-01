from app.solution import title_words


def test_case_0():
    assert title_words(*('hello wORLD',)) == 'Hello World'

def test_case_1():
    assert title_words(*('',)) == ''

def test_case_2():
    assert title_words(*('a b  c',)) == 'A B  C'

def test_case_3():
    assert title_words(*('x',)) == 'X'
