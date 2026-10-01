from app.inventory import Inventory

def test_add():
    assert Inventory().add("x").items[-1] == "x"
