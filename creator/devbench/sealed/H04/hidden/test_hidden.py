import ast, inspect, math
import pytest
import app.shapes as S

def test_behaviour():
    assert S.area("circle", r=1) == pytest.approx(math.pi) and S.area("rect", w=2, h=5) == 10
    with pytest.raises(ValueError):
        S.area("hexagon", side=1)

def test_no_if_chain():
    tree = ast.parse(inspect.getsource(S.area))
    assert sum(isinstance(n, ast.If) for n in ast.walk(tree)) <= 1
