import pytest
from app.stats import median

def test_even():
    assert median([4, 1, 3, 2]) == 2.5

def test_single_and_unsorted():
    assert median([7]) == 7 and median([10, -2, 4, 8]) == 6

def test_empty_raises():
    with pytest.raises(ValueError):
        median([])
