from app.solution import clamp


def test_case_0():
    assert clamp(*(5, 0, 10)) == 5

def test_case_1():
    assert clamp(*(-3, 0, 10)) == 0

def test_case_2():
    assert clamp(*(12, 0, 10)) == 10

def test_case_3():
    assert clamp(*(0, 0, 10)) == 0

def test_case_4():
    assert clamp(*(10, 0, 10)) == 10
