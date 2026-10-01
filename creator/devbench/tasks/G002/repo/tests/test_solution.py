from app.solution import clamp


def test_case_0():
    assert clamp(*(-3, 0, 10)) == 0
