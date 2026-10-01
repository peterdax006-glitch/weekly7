from app.stats import median

def test_odd():
    assert median([3, 1, 2]) == 2
