from app.inventory import Inventory

def test_independent():
    a, b = Inventory(), Inventory()
    a.add("apple")
    assert b.items == [] and a.items == ["apple"]

def test_given_list_is_used():
    start = ["k"]
    inv = Inventory(start)
    inv.add("z")
    assert inv.items == ["k", "z"]
