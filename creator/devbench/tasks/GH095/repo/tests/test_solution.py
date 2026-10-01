from app.solution import safe_div


def test_case_0():
    assert safe_div(*(1, 0)) == 0
