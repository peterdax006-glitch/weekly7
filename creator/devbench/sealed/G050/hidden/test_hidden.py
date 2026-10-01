from app.solution import caesar


def test_case_0():
    assert caesar(*('abc', 1)) == 'bcd'

def test_case_1():
    assert caesar(*('XYZ', 3)) == 'ABC'

def test_case_2():
    assert caesar(*('Hello, World!', 13)) == 'Uryyb, Jbeyq!'

def test_case_3():
    assert caesar(*('', 4)) == ''
