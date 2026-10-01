import shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

def run_with(module_text, tmp_path):
    work = tmp_path / "w"
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns("_hidden_acceptance", "__pycache__"))
    (work / "app" / "money.py").write_text(module_text, encoding="utf-8")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=work,
                       capture_output=True, text=True)
    return p.returncode

def test_tests_exist_and_pass_on_the_real_module(tmp_path):
    assert (ROOT / "tests" / "test_money.py").is_file()
    assert run_with((ROOT / "app" / "money.py").read_text(encoding="utf-8"), tmp_path) == 0

def test_catches_wrong_sum(tmp_path):
    assert run_with((HERE / "mutant_sum.txt").read_text(encoding="utf-8"), tmp_path) != 0

def test_catches_missing_negative_check(tmp_path):
    assert run_with((HERE / "mutant_neg.txt").read_text(encoding="utf-8"), tmp_path) != 0

def test_catches_missing_type_check(tmp_path):
    assert run_with((HERE / "mutant_type.txt").read_text(encoding="utf-8"), tmp_path) != 0
