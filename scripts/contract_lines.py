"""Meaningful-line counter for the Self-Learning contract (C62 section 60/83): counts code lines that are not blank, not
comments and not docstrings, so every builder is judged by one ruler. Usage: python scripts/contract_lines.py [files...]
(default: engine/learning/*.py and tests/test_learning_*.py)."""
import ast, glob, io, sys, tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def docstring_lines(src):
    lines = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return lines
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return lines


def meaningful(path):
    src = Path(path).read_text(encoding="utf-8")
    doc = docstring_lines(src)
    code = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                                tokenize.ENDMARKER):
                code.update(range(tok.start[0], tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError):
        return sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith("#"))
    return len(code - doc)


def main(argv):
    files = argv or sorted(glob.glob(str(ROOT / "engine" / "learning" / "*.py"))) + \
        sorted(glob.glob(str(ROOT / "tests" / "test_learning_*.py")))
    total = 0
    for f in files:
        n = meaningful(f)
        total += n
        print(f"{n:6d}  {Path(f).relative_to(ROOT) if Path(f).is_absolute() else f}")
    print(f"{total:6d}  TOTAL ({len(files)} files)")


if __name__ == "__main__":
    main(sys.argv[1:])
