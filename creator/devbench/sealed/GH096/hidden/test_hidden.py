from app.solution import safe_div


def test_case_0():
    assert safe_div(*(6, 3)) == 2.0

def test_case_1():
    assert safe_div(*(1, 0)) == 0

def test_case_2():
    assert safe_div(*(1, 0, -1)) == -1

def test_case_3():
    assert safe_div(*(0, 5)) == 0.0
