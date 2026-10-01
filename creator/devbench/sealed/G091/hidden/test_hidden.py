from app.solution import roman


def test_case_0():
    assert roman(*(1,)) == 'I'

def test_case_1():
    assert roman(*(4,)) == 'IV'

def test_case_2():
    assert roman(*(1994,)) == 'MCMXCIV'

def test_case_3():
    assert roman(*(3999,)) == 'MMMCMXCIX'

def test_case_4():
    assert roman(*(58,)) == 'LVIII'
