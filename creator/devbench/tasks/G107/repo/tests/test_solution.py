from app.solution import normalize_spaces


def test_case_0():
    assert normalize_spaces(*('  a   b  ',)) == 'a b'
