from app.solution import digits_sum


def test_case_0():
    assert digits_sum(*(123,)) == 6

def test_case_1():
    assert digits_sum(*(0,)) == 0

def test_case_2():
    assert digits_sum(*(-45,)) == 9

def test_case_3():
    assert digits_sum(*(9999,)) == 36
