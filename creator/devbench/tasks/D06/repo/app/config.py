import os


def load():
    return {"port": int(os.environ.get("APP_PORT", "8080")), "host": "127.0.0.1"}
