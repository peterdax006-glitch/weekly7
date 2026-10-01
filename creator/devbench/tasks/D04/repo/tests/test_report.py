from app.report import cost_report, sales_report

def test_outputs():
    assert sales_report([("a", 1)]) == "a         |    1.00"
    assert cost_report([("b", 2.5)]) == "COSTS\nb         |    2.50"
