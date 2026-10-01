import pytest
from app.parse import parse_int

def test_signs_and_space():
    assert parse_int(" +5 ") == 5 and parse_int("-17") == -17 and parse_int("-0") == 0

@pytest.mark.parametrize("bad", ["", "  ", "+", "-", "4a", "1.5"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        parse_int(bad)
