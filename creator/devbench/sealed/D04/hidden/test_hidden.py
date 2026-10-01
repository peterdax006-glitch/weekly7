import ast, inspect
import app.report as R

def test_helper_exists_and_formats():
    assert R.format_row("x", 3) == "x         |    3.00"

def test_both_use_the_helper():
    for fn in (R.sales_report, R.cost_report):
        tree = ast.parse(inspect.getsource(fn))
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "format_row" in calls, fn.__name__

def test_outputs_unchanged():
    rows = [("alpha", 1234.567), ("b", 0)]
    assert R.sales_report(rows) == "alpha     | 1234.57\nb         |    0.00"
    assert R.cost_report([]) == "COSTS"
