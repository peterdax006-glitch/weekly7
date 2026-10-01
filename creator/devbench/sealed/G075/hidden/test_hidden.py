from app.solution import compress


def test_case_0():
    assert compress(*('aaab',)) == 'a3b1'

def test_case_1():
    assert compress(*('',)) == ''

def test_case_2():
    assert compress(*('abc',)) == 'a1b1c1'

def test_case_3():
    assert compress(*('zzzz',)) == 'z4'
