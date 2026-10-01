from app.shapes import area

def test_square():
    assert area("square", side=3) == 9
