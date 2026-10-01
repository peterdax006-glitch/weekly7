from app.solution import compress


def test_case_0():
    assert compress(*('aaab',)) == 'a3b1'
