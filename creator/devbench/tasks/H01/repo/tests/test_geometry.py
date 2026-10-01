from app.geometry import polygon_area

def test_ccw_square():
    assert polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)]) == 1
