"""Where the LM keeps its data and checkpoints - outside the repo (override with CREATOR_RUNTIME)."""
from __future__ import annotations

from pathlib import Path


def runtime_dir() -> Path:
    from creator import device
    return device.runtime_dir()


def data_dir() -> Path:
    return runtime_dir() / "lmdata"


def ckpt_dir() -> Path:
    return runtime_dir() / "lmckpt"


def current_json() -> Path:
    return ckpt_dir() / "current.json"
