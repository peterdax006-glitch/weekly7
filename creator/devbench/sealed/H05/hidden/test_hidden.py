import shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

def run_with(text, tmp_path):
    work = tmp_path / "w"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns("_hidden_acceptance", "__pycache__"))
    (work / "app" / "fifo.py").write_text(text, encoding="utf-8")
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=work,
                          capture_output=True, text=True).returncode

def test_real_module_passes(tmp_path):
    assert (ROOT / "tests" / "test_fifo.py").is_file()
    assert run_with((ROOT / "app" / "fifo.py").read_text(encoding="utf-8"), tmp_path) == 0

def test_catches_lifo(tmp_path):
    assert run_with((HERE / "m_lifo.txt").read_text(encoding="utf-8"), tmp_path) != 0

def test_catches_missing_empty_error(tmp_path):
    assert run_with((HERE / "m_empty.txt").read_text(encoding="utf-8"), tmp_path) != 0

def test_catches_wrong_len(tmp_path):
    assert run_with((HERE / "m_len.txt").read_text(encoding="utf-8"), tmp_path) != 0
