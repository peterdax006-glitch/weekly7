from app.solution import clamp


def test_case_0():
    assert clamp(*(5, 0, 10)) == 5
