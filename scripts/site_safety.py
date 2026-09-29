"""Bible Phase 29 (public explanation system), canon C19/C11: what may NEVER reach a public page.

Three independent guards, each fail-closed:
  1. path guard   - only ``state/livesim/cycles.json`` may be read from the sealed tree, and only its revealed cycles;
  2. secret scan  - API keys, tokens, private keys and .env style assignments in any string that would be published;
  3. sealed-date scan - the sealed windows use disguised calendar years (2100-2299). A disguised date or a disguised
     stock code (S0123) in a page means a sealed window leaked, so the build stops instead of publishing.
`Leak` is raised with the offending page and a short excerpt; nothing is written when any page fails."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEALED_DIR = (ROOT / "state" / "livesim").resolve()
ALLOWED_SEALED = {(SEALED_DIR / "cycles.json").resolve()}
FORBIDDEN_DIRS = [SEALED_DIR, (ROOT / "data" / "cache").resolve()]
PRIVATE_NAMES = re.compile(r"(^|[\\/])(\.env[^\\/]*|.*secret.*|.*credential.*|id_rsa.*|.*\.pem|.*\.key)$", re.I)


class Leak(RuntimeError):
    pass


class SealedAccess(Leak):
    pass


def guard_path(p):
    """Return the resolved path if it is readable for the site, else raise SealedAccess."""
    rp = Path(p).resolve()
    if rp in ALLOWED_SEALED:
        return rp
    for d in FORBIDDEN_DIRS:
        if d == rp or d in rp.parents:
            raise SealedAccess(f"site builder may not read {rp} (sealed or cache tree)")
    if PRIVATE_NAMES.search(rp.name):
        raise SealedAccess(f"site builder may not read {rp.name} (looks like a secret)")
    return rp


def safe_read_text(p, encoding="utf-8"):
    return guard_path(p).read_text(encoding=encoding, errors="replace")


def safe_read_json(p, default=None):
    """JSON file through the path guard. Missing or unparsable file returns ``default`` (the page says 'not available')."""
    rp = guard_path(p)
    if not rp.exists():
        return default
    try:
        return json.loads(rp.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        return default


# ---------------------------------------------------------------- content scans
SECRET_PATTERNS = [
    ("alpaca key id", re.compile(r"\b(PK|AK|CK)[A-Z0-9]{18}\b")),
    ("github token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("openai/anthropic style key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("assignment of a secret", re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd|"
                                          r"alpaca_secret\w*|apca_api_secret_key)\b\s*[:=]\s*[\"']?[A-Za-z0-9/+_\-]{12,}")),
    ("long base64-ish secret", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{24,}")),
]
DISGUISED_DATE = re.compile(r"\b(2[12]\d\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
DISGUISED_CODE = re.compile(r"\bS\d{4}\b")
DISGUISED_YEAR_WORD = re.compile(r"\b(21[0-9]{2}|22[0-9]{2})\b")


def find_secrets(text):
    return [(name, m.group(0)[:24]) for name, rx in SECRET_PATTERNS for m in rx.finditer(text)]


def find_sealed_marks(text):
    """Disguised dates and disguised stock codes. A bare 4-digit 21xx/22xx is only flagged inside a date-like context
    (`year`/`Year`/`window` next to it) so ordinary numbers such as 2,150 dollars do not trip the scan."""
    out = [("disguised date", m.group(0)) for m in DISGUISED_DATE.finditer(text)]
    out += [("disguised code", m.group(0)) for m in DISGUISED_CODE.finditer(text)]
    for m in DISGUISED_YEAR_WORD.finditer(text):
        ctx = text[max(0, m.start() - 12): m.end() + 6].lower()
        if "year" in ctx or "window" in ctx or "week_end" in ctx:
            out.append(("disguised year", m.group(0)))
    return out


def audit_text(name, text):
    """Raise Leak when the page contains a secret or a sealed mark; return True when clean."""
    hits = find_secrets(text) + find_sealed_marks(text)
    if hits:
        kinds = sorted({h[0] for h in hits})
        raise Leak(f"{name}: {len(hits)} forbidden item(s) [{', '.join(kinds)}] e.g. {hits[0][1]!r}")
    return True


# ---------------------------------------------------------------- data-level filters
REVEALED_KEYS = ("run_id", "revealed_year", "config", "config_version", "diagnosis")
DROP_DIAG_KEYS = {"worst_positions", "best_positions", "codes", "trades", "positions", "holdings"}


def public_cycles(cycles):
    """Only revealed windows, only headline diagnosis fields. Position level detail carries disguised codes and dates."""
    out = []
    for c in cycles or []:
        if not c.get("revealed_year"):
            continue
        d = {k: v for k, v in (c.get("diagnosis") or {}).items() if k not in DROP_DIAG_KEYS and not isinstance(v, (list, dict))}
        out.append({"run_id": str(c.get("run_id")), "year": str(c["revealed_year"]), "config": c.get("config") or {},
                    "version": c.get("config_version"), **d})
    return out


def scrub(obj, _depth=0):
    """Recursively drop dict keys that look secret and replace secret-looking strings; never mutates the input."""
    if _depth > 12:
        return None
    if isinstance(obj, dict):
        return {k: scrub(v, _depth + 1) for k, v in obj.items()
                if not re.search(r"(?i)secret|password|token|api_?key|credential", str(k))}
    if isinstance(obj, (list, tuple)):
        return [scrub(v, _depth + 1) for v in obj]
    if isinstance(obj, str):
        s = obj
        for _, rx in SECRET_PATTERNS:
            s = rx.sub("[removed]", s)
        return s
    return obj
