from app.dates import parse_date

def test_more_dates():
    assert parse_date("2000-01-31") == (2000, 1, 31) and parse_date("2023-07-04") == (2023, 7, 4)
