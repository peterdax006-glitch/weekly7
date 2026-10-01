from app.billing import total

def test_total():
    assert total([0.1, 0.2]) == "0.30"
    assert total([1.005]) == "1.01"
