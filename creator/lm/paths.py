"""Where the LM keeps its data and checkpoints - outside the repo (override with CREATOR_RUNTIME)."""
from __future__ import annotations

import os
from pathlib import Path


def runtime_dir() -> Path:
    return Path(os.environ.get("CREATOR_RUNTIME", r"C:\Users\Peter\creator_runtime"))


def data_dir() -> Path:
    return runtime_dir() / "lmdata"


def ckpt_dir() -> Path:
    return runtime_dir() / "lmckpt"


def current_json() -> Path:
    return ckpt_dir() / "current.json"
