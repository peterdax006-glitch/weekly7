from app.cache import Cache

def test_put_get():
    c = Cache()
    c.put("a", 1)
    assert c.get("a") == 1 and c.get("b", 0) == 0
