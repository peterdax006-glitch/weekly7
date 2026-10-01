from app.cache import Cache

def test_lru_eviction():
    c = Cache(capacity=2)
    c.put("a", 1); c.put("b", 2)
    assert c.get("a") == 1          # a is now most recent
    c.put("c", 3)                   # evicts b
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3

def test_overwrite_does_not_evict():
    c = Cache(capacity=2)
    c.put("a", 1); c.put("b", 2); c.put("a", 9)
    assert c.get("a") == 9 and c.get("b") == 2

def test_default_capacity():
    c = Cache()
    for i in range(200):
        c.put(i, i)
    assert c.get(0) is None and c.get(199) == 199
