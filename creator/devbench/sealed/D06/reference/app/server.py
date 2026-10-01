from app.config import load


def start():
    cfg = load()
    return {"host": cfg["host"], "port": cfg["port"], "status": "started"}
