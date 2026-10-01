from app.solution import caesar


def test_case_0():
    assert caesar(*('abc', 1)) == 'bcd'
