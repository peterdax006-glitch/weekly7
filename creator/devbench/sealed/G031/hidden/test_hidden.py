from app.solution import gcd


def test_case_0():
    assert gcd(*(12, 18)) == 6

def test_case_1():
    assert gcd(*(0, 5)) == 5

def test_case_2():
    assert gcd(*(7, 0)) == 7

def test_case_3():
    assert gcd(*(0, 0)) == 0

def test_case_4():
    assert gcd(*(17, 13)) == 1
