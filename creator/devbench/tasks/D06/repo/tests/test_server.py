from app.server import start

def test_starts():
    assert start()["status"] == "started"
