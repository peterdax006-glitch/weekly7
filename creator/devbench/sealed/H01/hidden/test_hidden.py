from app.geometry import polygon_area

def test_cw():
    assert polygon_area([(0, 0), (0, 2), (3, 2), (3, 0)]) == 6

def test_triangle_both_ways():
    t = [(0, 0), (4, 0), (0, 3)]
    assert polygon_area(t) == 6 and polygon_area(t[::-1]) == 6
