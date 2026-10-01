from app.dates import parse_date

def test_parse():
    assert parse_date("2024-03-15") == (2024, 3, 15)
    assert parse_date("1999-12-01") == (1999, 12, 1)
