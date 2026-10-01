from app.solution import leap_year


def test_case_0():
    assert leap_year(*(2000,)) == True

def test_case_1():
    assert leap_year(*(1900,)) == False

def test_case_2():
    assert leap_year(*(2024,)) == True

def test_case_3():
    assert leap_year(*(2023,)) == False
