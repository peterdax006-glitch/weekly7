from app.parse import parse_int

def test_digits():
    assert parse_int("42") == 42
