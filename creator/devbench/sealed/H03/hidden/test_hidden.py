from app.billing import total

def test_half_up_rounding():
    assert total([2.675]) == "2.68" and total([0.125, 0.0]) == "0.13" and total([]) == "0.00"
