"""Static checker for hand-written Jelly (Jellycuts) scripts such as scripts/nupen2.jelly. No Swift compiler runs on Windows, so this
checks what the Open-Jellycore compiler and the tree-sitter-jelly grammar would reject, plus the rules learned from their source:

  * structure: balanced braces/parentheses/quotes, `if(<var> <op> <value>) {`, `} else {`, `repeat(<number>) {`, `repeatEach(<var>) {`;
  * functions: only names in FUNCS (each read on docs.jellycuts.com), only their documented parameter labels, required labels present;
  * enums: only values whose Jelly spelling equals the compiler's raw value (CamelCase vs "After Pause" style differs between docs and
    compiler, so scripts must avoid the ambiguous ones);
  * variables: every ${x}, argument and assignment refers to a `var` or a `>> magic` defined earlier; magic names are unique (the compiler
    resolves a reused name to the first one); no ${} inside a JSON dictionary literal (the compiler drops it); RepeatItem only in a
    repeatEach that is not nested in another repeat (nested names are 'Repeat Item 2' in Shortcuts but the compiler emits other names);
  * placeholders: with --served, no PASTE_ placeholder may remain.

    python scripts/jelly_check.py scripts/nupen2.jelly [--served]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# name -> (required labels, optional labels). Source: https://docs.jellycuts.com/Documentation/Shortcuts/<name>.html (5 Oct 2026)
FUNCS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "downloadURL": (("url",), ("method", "headers", "requestType", "requestJSON", "requestVar")),
    "valueFor": (("key", "dictionary"), ()),
    "runShortcut": (("name",), ("input", "show")),
    "openIn": (("app",), ("input", "ask")),
    "quicklook": ((), ("input",)),
    "grabJellycut": ((), ()),
    "setName": (("input", "name"), ("dontIncludeExtension",)),
    "getDictionaryFrom": (("input",), ()),
    "text": (("text",), ()),
    "speakText": (("text",), ("wait", "rate", "pitch", "language")),
    "exit": ((), ("var",)),
    "batteryLevel": ((), ()),
    "getLocation": ((), ("input",)),
    "locationDetail": (("detail", "location"), ()),
    "encodeURL": (("url",), ()),
    "dictateText": ((), ("language", "endTrigger")),
    "calculate": (("input",), ()),
    "openURL": (("url",), ()),
    "timer": (("duration",), ()),
    "createNote": (("text",), ("show",)),
    "setFlashlight": (("state",), ("level",)),
    "setDND": (("state",), ()),
    "setVolume": (("level",), ()),
    "setBrightness": (("value",), ()),
    "setWiFi": (("state",), ()),
    "setBluetooth": (("value",), ()),
    "lowPowerMode": (("state",), ()),
    "setAppearance": (("mode",), ()),
    "setClipboard": ((), ("variable", "local", "expiration")),
    "getCurrentConditions": ((), ()),
    "conditionDetail": (("detail", "condition"), ()),
    "getLastVideo": (("count",), ()),
    "selectPhoto": ((), ("types", "multiple")),
}
# parameters whose value is an enum, and the values whose spelling is identical in docs and compiler
ENUMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("downloadURL", "method"): ("GET", "POST", "PUT", "PATCH", "DELETE"),
    ("downloadURL", "requestType"): ("Json", "Form", "File"),
    ("locationDetail", "detail"): ("City", "Name"),
    ("conditionDetail", "detail"): ("Temperature", "Condition"),
}
VAR_PARAMS = {("valueFor", "dictionary"), ("locationDetail", "location"), ("conditionDetail", "condition"), ("setClipboard", "variable"),
              ("downloadURL", "requestVar"), ("getLocation", "input"), ("exit", "var")}
DICT_PARAMS = {("downloadURL", "headers"), ("downloadURL", "requestJSON")}
BOOLS = {"true", "false"}
GLOBALS = {"ShortcutInput", "Clipboard", "CurrentDate", "Ask", "DeviceDetails"}
COLORS = {"red", "orange", "tangerine", "yellow", "green", "teal", "lightblue", "blue", "navy", "grape", "purple", "pink", "grayblue",
          "graygreen", "graybrown"}
OPS = ("==", "!=", "<=", ">=", "<", ">", "::", "!:", "$$", "$!")
IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

_CALL = re.compile(rf"^({IDENT})\((.*)\)(?:\s+>>\s+({IDENT}))?$")
_IF = re.compile(rf"^if\(({IDENT}) (==|!=|<=|>=|<|>|::|!:|\$\$|\$!) (.+)\) \{{$")
_VAR = re.compile(rf"^var ({IDENT}) = (.+)$")
_SET = re.compile(rf"^({IDENT}) = (.+)$")
_REPEAT = re.compile(r"^repeat\((\d+)\) \{$")
_EACH = re.compile(rf"^repeatEach\(({IDENT})\) \{{$")
_INTERP = re.compile(r"\$\{([^}]*)\}")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_DUR = re.compile(r"\d+(?:\.\d+)? (?:sec|min|hr)")


def _split_args(s: str) -> list[str]:
    out, cur, q, depth, i = [], "", False, 0, 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and q and i + 1 < len(s):
            cur += s[i:i + 2]
            i += 2
            continue
        if ch == '"':
            q = not q
        elif not q and ch in "({[":
            depth += 1
        elif not q and ch in ")}]":
            depth -= 1
        if ch == "," and not q and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def _string_ok(v: str) -> bool:
    if len(v) < 2 or v[0] != '"' or v[-1] != '"':
        return False
    body = v[1:-1]
    return '"' not in re.sub(r'\\"', "", body)


def check(src: str, served: bool = False) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings)."""
    errs: list[str] = []
    warns: list[str] = []
    known: set[str] = set(GLOBALS)
    magic: dict[str, int] = {}
    declared: dict[str, int] = {}
    used: set[str] = set()
    stack: list[str] = []                         # kinds of open blocks: if / else / repeat / each
    saw_import = False

    def ref(name: str, no: int, what: str) -> None:
        used.add(name)
        if name == "RepeatItem" or name == "RepeatIndex":
            loops = [b for b in stack if b in ("repeat", "each")]
            if name == "RepeatItem" and (not loops or loops[-1] != "each"):
                errs.append(f"line {no}: RepeatItem outside a repeatEach")
            elif len(loops) > 1:
                errs.append(f"line {no}: {name} inside a nested repeat (Shortcuts names it '{name[:6]} {name[6:]} 2'; the compiler does not)")
            return
        if name not in known:
            errs.append(f"line {no}: {what} '{name}' is not defined above")

    def value(v: str, no: int, fn: str = "", param: str = "") -> None:
        if v.startswith('"'):
            if not _string_ok(v):
                errs.append(f"line {no}: bad string {v[:40]}")
            for x in _INTERP.findall(v):
                if not re.fullmatch(IDENT, x):
                    errs.append(f"line {no}: bad interpolation ${{{x}}}")
                else:
                    ref(x, no, "variable")
            if (fn, param) in ENUMS:
                errs.append(f"line {no}: {fn}.{param} takes an enum, not a string")
            return
        if v.startswith("{"):
            if (fn, param) not in DICT_PARAMS:
                errs.append(f"line {no}: dictionary literal not allowed for {fn}.{param}")
            if "${" in v:
                errs.append(f"line {no}: variables inside a dictionary literal are dropped by the compiler")
            import json
            try:
                json.loads(v)
            except ValueError:
                errs.append(f"line {no}: dictionary literal is not valid JSON")
            return
        if (fn, param) in ENUMS:
            if v not in ENUMS[(fn, param)]:
                errs.append(f"line {no}: {fn}.{param}={v} is not a verified enum value {ENUMS[(fn, param)]}")
            return
        if fn == "timer" and param == "duration":
            if not _DUR.fullmatch(v):
                errs.append(f"line {no}: timer duration must be a literal like '5 min' (the compiler takes no variable here)")
            return
        if v in BOOLS or _NUM.fullmatch(v):
            if (fn, param) in VAR_PARAMS:
                errs.append(f"line {no}: {fn}.{param} needs a variable")
            return
        if re.fullmatch(IDENT, v):
            ref(v, no, "variable")
            return
        errs.append(f"line {no}: cannot read value {v[:40]}")

    lines = src.splitlines()
    for no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if "/*" in line:
            errs.append(f"line {no}: use // comments (block comments are fragile in the grammar)")
            continue
        if raw != raw.rstrip():
            warns.append(f"line {no}: trailing whitespace")
        if served and "PASTE_" in line:
            errs.append(f"line {no}: placeholder left in the served script")
        if line == "import Shortcuts":
            if stack or no != next(i for i, r in enumerate(lines, 1) if r.strip() and not r.strip().startswith("//")):
                errs.append(f"line {no}: import must be the first statement")
            saw_import = True
            continue
        if line.startswith("#"):
            for flag in [f.strip() for f in line.split(",") if f.strip()]:
                m = re.fullmatch(r"#(Color|Icon): (\w+)", flag)
                if not m:
                    errs.append(f"line {no}: bad flag {flag}")
                elif m[1] == "Color" and m[2] not in COLORS:
                    errs.append(f"line {no}: unknown colour {m[2]}")
            continue
        if not saw_import:
            errs.append(f"line {no}: statement before 'import Shortcuts'")
        if line == "}":
            if not stack:
                errs.append(f"line {no}: unbalanced }}")
            else:
                stack.pop()
            continue
        if line == "} else {":
            if not stack or stack[-1] != "if":
                errs.append(f"line {no}: else without if")
            else:
                stack[-1] = "else"
            continue
        if m := _IF.match(line):
            ref(m[1], no, "condition variable")
            sec = m[3]
            if sec != "nil" and not (_NUM.fullmatch(sec) or _string_ok(sec)):
                errs.append(f"line {no}: if compares with a literal string, number or nil only (got {sec})")
            if sec.startswith('"') and "${" in sec:
                errs.append(f"line {no}: no variables on the right of an if")
            stack.append("if")
            continue
        if line.startswith("if"):
            errs.append(f"line {no}: if must be 'if(<var> <op> <value>) {{' with spaces around the operator")
            continue
        if m := _REPEAT.match(line):
            stack.append("repeat")
            continue
        if m := _EACH.match(line):
            ref(m[1], no, "list")
            stack.append("each")
            continue
        if line.startswith("repeat"):
            errs.append(f"line {no}: bad repeat header")
            continue
        if m := _VAR.match(line):
            name, v = m[1], m[2]
            value(v, no)
            if name in magic:
                errs.append(f"line {no}: var '{name}' shadows a magic variable")
            if name in declared:
                errs.append(f"line {no}: var '{name}' declared twice (use '{name} = ...')")
            declared[name] = no
            known.add(name)
            continue
        if (m := _SET.match(line)) and "(" not in line.split("=")[0]:
            if m[1] not in declared:
                errs.append(f"line {no}: '{m[1]}' assigned before 'var {m[1]} = ...'")
            used.add(m[1])
            value(m[2], no)
            continue
        if m := _CALL.match(line):
            fn, args, mv = m[1], m[2], m[3]
            if fn not in FUNCS:
                errs.append(f"line {no}: unknown or unverified function {fn}")
            else:
                req, opt = FUNCS[fn]
                seen = set()
                for a in _split_args(args):
                    am = re.match(r"^(\w+): (.+)$", a, re.S)
                    if not am:
                        errs.append(f"line {no}: {fn}: unnamed or badly spaced parameter '{a[:30]}'")
                        continue
                    lab, v = am[1], am[2]
                    if lab not in req + opt:
                        errs.append(f"line {no}: {fn} has no parameter '{lab}'")
                    if lab in seen:
                        errs.append(f"line {no}: {fn}: '{lab}' twice")
                    seen.add(lab)
                    value(v, no, fn, lab)
                for lab in req:
                    if lab not in seen:
                        errs.append(f"line {no}: {fn} is missing '{lab}'")
            if mv:
                if mv in magic or mv in declared:
                    errs.append(f"line {no}: magic name '{mv}' already used on line {magic.get(mv) or declared.get(mv)}")
                magic[mv] = no
                known.add(mv)
            continue
        errs.append(f"line {no}: not a Jelly statement: {line[:60]}")
    if stack:
        errs.append(f"end of file: {len(stack)} block(s) not closed ({', '.join(stack)})")
    for name, no in declared.items():
        if name not in used:
            warns.append(f"line {no}: var '{name}' is never used")
    for name, no in magic.items():
        if name not in used:
            warns.append(f"line {no}: magic variable '{name}' is never used")
    return errs, warns


def main(argv: list[str]) -> int:
    path = Path(argv[0])
    errs, warns = check(path.read_text(encoding="utf-8"), served="--served" in argv)
    for w in warns:
        print("warning:", w)
    for e in errs:
        print("ERROR:", e)
    print(f"{path.name}: {len(errs)} errors, {len(warns)} warnings")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
