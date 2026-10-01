import importlib

def test_uses_config_port(monkeypatch):
    monkeypatch.setenv("APP_PORT", "9123")
    import app.server as S
    importlib.reload(S)
    assert S.start()["port"] == 9123

def test_default_port(monkeypatch):
    monkeypatch.delenv("APP_PORT", raising=False)
    import app.server as S
    importlib.reload(S)
    assert S.start()["port"] == 8080 and S.start()["host"] == "127.0.0.1"
