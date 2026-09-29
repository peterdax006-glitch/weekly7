"""Bible Phase 42 (the standard for every claim): no number reaches a report without an evidence trail.

A `Claim` is one quantitative statement plus everything Phase 42 demands behind it:
  * the ARTEFACT it came from - path and SHA-256, re-checked against the file on disk when the claim is verified, and
    (for a JSON artefact) the key path that must actually hold the claimed value;
  * the exact DEFINITION of the quantity (what is measured, on what population, over which window);
  * the COST ADJUSTMENT - applied in basis points, or explicitly "gross" with the reason (a gross number is rendered
    as GROSS, never passed off as net);
  * the STATISTICAL COMPARISON - baseline vs candidate, statistic, interval, n, p;
  * the CONTROL - a null, random or shuffled comparison and what it scored (a claim whose control matches the claim is
    refused: the control says the effect is not there).
`kind` adds the Phase 42 lists: a "pattern" claim also needs the discovery/confirmation/later samples, week-clustered
statistics, permutation evidence, FDR, P(real), effect size, incremental gain and regime distribution; a "setting" claim
needs independent windows, risk, stability and the adoption reason.
`render` raises `ClaimIncomplete` listing EVERY gap. `scan_report` is the converse: it reads a finished report and lists
each number that no claim covers. Pure functions of their inputs; no clock, no randomness."""
import hashlib
import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

KINDS = ("general", "pattern", "setting")
PATTERN_EXTRAS = ("discovery_sample", "confirmation_sample", "later_sample", "week_clustered_stat", "permutation_evidence",
                  "fdr_result", "p_real", "effect_size", "incremental_gain", "regime_distribution")
SETTING_EXTRAS = ("baseline", "candidate", "independent_windows", "risk", "stability", "adoption_reason")
EXTRAS = {"general": (), "pattern": PATTERN_EXTRAS, "setting": SETTING_EXTRAS}


class ClaimIncomplete(ValueError):
    def __init__(self, claim_id, problems):
        self.claim_id, self.problems = claim_id, list(problems)
        super().__init__(f"claim {claim_id!r} refused: " + "; ".join(self.problems))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _blank(v):
    return v is None or (isinstance(v, (str, list, tuple, dict)) and len(v) == 0) or (isinstance(v, str) and not v.strip())


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass
class Comparison:
    baseline: str = ""
    candidate: str = ""
    statistic: str = ""            # e.g. "paired bootstrap of mean weekly return difference"
    estimate: float = None         # candidate minus baseline
    ci_low: float = None
    ci_high: float = None
    n: int = None                  # independent units (weeks, windows), not rows
    p_value: float = None


@dataclass
class Control:
    name: str = ""                 # e.g. "1000 label permutations"
    result: float = None           # what the control scored on the same statistic
    note: str = ""


@dataclass
class CostAdjustment:
    applied: bool = None
    bps: float = None              # round-trip cost charged when applied
    note: str = ""                 # required when not applied: why the number is gross


@dataclass
class Claim:
    claim_id: str
    statement: str
    value: float = None
    unit: str = ""                 # "%", "x", "" ... how the value is printed
    artefact_path: str = ""
    artefact_hash: str = ""
    artefact_key: str = ""         # dotted path into a JSON artefact holding `value` (optional but checked when given)
    definition: str = ""
    cost: CostAdjustment = field(default_factory=CostAdjustment)
    comparison: Comparison = field(default_factory=Comparison)
    control: Control = field(default_factory=Control)
    kind: str = "general"
    extras: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- validation
    def problems(self, root=None, check_file=True):
        """Every reason this claim may not be rendered. `root` resolves a relative artefact path."""
        p = []
        if _blank(self.claim_id):
            p.append("no claim_id")
        if _blank(self.statement):
            p.append("no statement")
        if self.kind not in KINDS:
            p.append(f"unknown kind {self.kind!r}")
        if not _finite(self.value):
            p.append("value missing or not finite")
        # artefact: path + hash, and the hash must match the file
        if _blank(self.artefact_path):
            p.append("no artefact path")
        if _blank(self.artefact_hash):
            p.append("no artefact hash")
        elif not re.fullmatch(r"[0-9a-f]{64}", self.artefact_hash):
            p.append("artefact hash is not a SHA-256 hex digest")
        if check_file and not _blank(self.artefact_path) and re.fullmatch(r"[0-9a-f]{64}", self.artefact_hash or ""):
            ap = _resolve(self.artefact_path, root)
            if not ap.exists():
                p.append(f"artefact {self.artefact_path} does not exist")
            elif sha256_file(ap) != self.artefact_hash:
                p.append("artefact hash does not match the file on disk (the artefact changed after the claim)")
            elif not _blank(self.artefact_key):
                got = _json_lookup(ap, self.artefact_key)
                if got is _MISSING:
                    p.append(f"key {self.artefact_key!r} not found in the artefact")
                elif not _finite(got) or not _finite(self.value) or not math.isclose(got, self.value, rel_tol=1e-6, abs_tol=1e-9):
                    p.append(f"artefact holds {got!r} at {self.artefact_key!r}, not the claimed {self.value!r}")
        if _blank(self.definition):
            p.append("no exact definition")
        elif len(self.definition.split()) < 4:
            p.append("definition too thin to reproduce the number (fewer than 4 words)")
        c = self.cost
        if c.applied is None:
            p.append("cost adjustment not stated (applied or gross)")
        elif c.applied and not (_finite(c.bps) and c.bps >= 0):
            p.append("cost adjustment applied but bps missing")
        elif not c.applied and _blank(c.note):
            p.append("gross of costs with no stated reason")
        p += self._comparison_problems()
        k = self.control
        if _blank(k.name):
            p.append("no control")
        elif not _finite(k.result):
            p.append("control has no result")
        p += self._control_vs_claim()
        for name in EXTRAS.get(self.kind, ()):
            if _blank(self.extras.get(name)):
                p.append(f"{self.kind} claim missing '{name}'")
        return p

    def _comparison_problems(self):
        c, p = self.comparison, []
        for f in ("baseline", "candidate", "statistic"):
            if _blank(getattr(c, f)):
                p.append(f"comparison has no {f}")
        if not _finite(c.estimate):
            p.append("comparison has no estimate")
        if not (_finite(c.ci_low) and _finite(c.ci_high)):
            p.append("comparison has no interval")
        elif c.ci_low > c.ci_high:
            p.append("interval is reversed (low > high)")
        elif _finite(c.estimate) and not (c.ci_low - 1e-12 <= c.estimate <= c.ci_high + 1e-12):
            p.append("estimate lies outside its own interval")
        if not (isinstance(c.n, int) and not isinstance(c.n, bool) and c.n > 0):
            p.append("comparison has no sample size n")
        if not (_finite(c.p_value) and 0.0 <= c.p_value <= 1.0):
            p.append("comparison has no valid p-value in [0,1]")
        if not _blank(c.baseline) and c.baseline == c.candidate:
            p.append("baseline and candidate are the same thing")
        return p

    def _control_vs_claim(self):
        k, c = self.control, self.comparison
        if _finite(k.result) and _finite(c.estimate) and c.estimate != 0 and math.isclose(k.result, c.estimate, rel_tol=1e-9):
            return ["control scored exactly the claimed effect: it is not distinguishable from the control"]
        if _finite(k.result) and _finite(c.ci_low) and _finite(c.ci_high) and c.ci_low <= k.result <= c.ci_high \
                and _finite(c.estimate) and c.estimate != k.result:
            return [f"control result {k.result:g} lies inside the claim's own interval [{c.ci_low:g}, {c.ci_high:g}]: "
                    f"the claim does not beat its control"]
        return []

    # ---------------------------------------------------------------- output
    def covered_numbers(self):
        """Numbers a report may quote on this claim's authority: the value, the estimate, the interval, the control
        result, n, p and the cost bps."""
        c, k = self.comparison, self.control
        nums = [self.value, c.estimate, c.ci_low, c.ci_high, k.result, c.n, c.p_value, self.cost.bps]
        return [float(x) for x in nums if _finite(x)]

    def printed_value(self):
        if not _finite(self.value):
            return "n/a"
        return f"{self.value * 100:.2f}%" if self.unit == "%" else f"{self.value:.4g}{self.unit}"

    def render(self, root=None, check_file=True):
        """One paragraph of text with the whole trail. Raises ClaimIncomplete when anything is missing."""
        pr = self.problems(root, check_file)
        if pr:
            raise ClaimIncomplete(self.claim_id, pr)
        c, k, x = self.comparison, self.control, self.cost
        pct = self.unit == "%"
        f = (lambda v: f"{v * 100:.2f}%") if pct else (lambda v: f"{v:.4g}")
        cost = f"net of {x.bps:g} bps round-trip costs" if x.applied else f"GROSS of costs ({x.note})"
        lines = [f"[{self.claim_id}] {self.statement}: {self.printed_value()}",
                 f"  definition: {self.definition}",
                 f"  source: {self.artefact_path} sha256:{self.artefact_hash[:16]}" + (f" key {self.artefact_key}" if self.artefact_key else ""),
                 f"  costs: {cost}",
                 f"  comparison: {c.candidate} vs {c.baseline} by {c.statistic}: {f(c.estimate)} "
                 f"(interval {f(c.ci_low)} to {f(c.ci_high)}, n={c.n}, p={c.p_value:.4g})",
                 f"  control: {k.name} scored {f(k.result)}" + (f" ({k.note})" if k.note else "")]
        for name in EXTRAS[self.kind]:
            lines.append(f"  {name.replace('_', ' ')}: {self.extras[name]}")
        return "\n".join(lines)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(cost=CostAdjustment(**d.pop("cost", {})), comparison=Comparison(**d.pop("comparison", {})),
                   control=Control(**d.pop("control", {})), **d)


_MISSING = object()


def _resolve(path, root=None):
    p = Path(path)
    return p if p.is_absolute() or root is None else Path(root) / p


def _json_lookup(path, dotted):
    try:
        cur = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return _MISSING
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


def make_claim(claim_id, statement, value, artefact_path, definition, root=None, **kw):
    """Convenience: hashes the artefact from disk so the hash is never typed. Raises FileNotFoundError when the artefact
    is missing - a claim about a file that is not there cannot even be drafted."""
    ap = _resolve(artefact_path, root)
    return Claim(claim_id=claim_id, statement=statement, value=value, artefact_path=str(artefact_path),
                 artefact_hash=sha256_file(ap), definition=definition, **kw)


# ------------------------------------------------------------------------------------------------ claim book
class ClaimBook:
    """A set of claims for one report. Ids are unique; the book renders all or none."""

    def __init__(self, root=None):
        self.root, self.claims = root, {}

    def add(self, claim):
        if claim.claim_id in self.claims:
            raise ValueError(f"duplicate claim id {claim.claim_id!r}")
        self.claims[claim.claim_id] = claim
        return claim

    def audit(self):
        """{claim_id: [problems]} for every claim that would not render."""
        return {cid: pr for cid, c in self.claims.items() if (pr := c.problems(self.root))}

    def render_all(self):
        bad = self.audit()
        if bad:
            cid = sorted(bad)[0]
            raise ClaimIncomplete(cid, bad[cid] + ([f"(and {len(bad) - 1} other claim(s) also refused)"] if len(bad) > 1 else []))
        return "\n\n".join(self.claims[c].render(self.root) for c in sorted(self.claims))

    def covered(self):
        out = []
        for c in self.claims.values():
            out += [(c.claim_id, v) for v in c.covered_numbers()]
        return out

    def save(self, path):
        Path(path).write_text("\n".join(json.dumps(c.to_dict(), sort_keys=True) for c in self.claims.values()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, root=None):
        b = cls(root)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                b.add(Claim.from_dict(json.loads(line)))
        return b


# ------------------------------------------------------------------------------------------------ report scanner
NUM = re.compile(r"(?<![\w.#/\\-])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?\s?(?:%|x\b|bps\b|pp\b)?", re.I)
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b")
VERSION = re.compile(r"\bv?\d+\.\d+\.\d+\b")
HEXID = re.compile(r"\b[0-9a-f]{7,}\b", re.I)
LABEL = re.compile(r"\b(?:phase|section|step|item|test|table|figure|fig|canon|rule|tier|window|week|weeks|day|days|year|years|"
                   r"seed|run|top|k|n|pid|line|page|part|gate|version|v|c|p|b|a|t)\s*[-#]?\s*$", re.I)
ORDINAL = re.compile(r"^\s*(?:[-*]\s+)?\d+[.)]\s")


def _decimals(tok):
    t = re.sub(r"[%xX]|bps|pp|,|\s", "", tok, flags=re.I)
    return len(t.split(".")[1].split("e")[0]) if "." in t else 0


def _to_float(tok):
    t = re.sub(r"[%]|x\b|bps|pp|,|\s", "", tok.strip(), flags=re.I)
    try:
        return float(t)
    except ValueError:
        return None


def extract_numbers(text):
    """[(line_no, token, value, is_percent, context)] for every number that is quoted as a finding. Skips dates, times,
    version strings, hashes, list ordinals, and integers introduced by a label word ("Phase 30", "n=", "seed 7") - those are
    identifiers, not results. A bare integer <= 10 is treated as a count and skipped unless it carries a unit."""
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = DATE.sub(lambda m: " " * len(m.group(0)), raw)
        line = VERSION.sub(lambda m: " " * len(m.group(0)), line)
        line = HEXID.sub(lambda m: " " * len(m.group(0)) if re.search(r"[a-f]", m.group(0), re.I) and re.search(r"\d", m.group(0)) else m.group(0), line)
        line = re.sub(r"\b[A-Za-z_]+\d+[A-Za-z_\d]*\b", lambda m: " " * len(m.group(0)), line)   # identifiers such as B21, E1f3
        if ORDINAL.match(line):
            line = re.sub(r"^\s*(?:[-*]\s+)?\d+[.)]", lambda m: " " * len(m.group(0)), line, count=1)
        for m in NUM.finditer(line):
            tok = m.group(0).strip()
            v = _to_float(tok)
            if v is None:
                continue
            pct = tok.endswith("%")
            has_unit = pct or bool(re.search(r"(x|bps|pp)$", tok, re.I))
            before = line[:m.start()]
            if LABEL.search(before) and not has_unit and "." not in tok:
                continue
            if before.rstrip().endswith("=") and re.search(r"\b(n|k|seed|pid)\s*=\s*$", before, re.I):
                continue
            if "." not in tok and not has_unit and abs(v) <= 10:
                continue
            if "." not in tok and not has_unit and 1900 <= v <= 2100:
                continue
            out.append((i, tok, v, pct, raw.strip()[:140]))
    return out


def covers(claim_value, tok, tol_extra=0.0):
    """Does the printed token `tok` state `claim_value` at the precision it was printed at? A percent token is compared
    with value*100, a plain token with value, and both must agree to half a unit in the last printed digit."""
    v = _to_float(tok)
    if v is None or not _finite(claim_value):
        return False
    dec = _decimals(tok)
    half = 0.5 * 10 ** (-dec) + tol_extra
    if tok.strip().endswith("%"):
        return abs(v - claim_value * 100.0) <= half + 1e-9
    return abs(v - claim_value) <= half + 1e-9


def scan_report(text, book_or_claims, allow=()):
    """List every number in `text` that no claim covers. `allow` is a list of literal tokens the author vouches are not
    findings (rare; each is reported so the exemption is visible). Returns
    {'numbers': N, 'covered': [...], 'uncovered': [...], 'allowed': [...], 'ok': bool, 'coverage': share}."""
    claims = list(book_or_claims.claims.values()) if hasattr(book_or_claims, "claims") else list(book_or_claims)
    pool = [(c.claim_id, v) for c in claims for v in c.covered_numbers()]
    covered, uncovered, allowed = [], [], []
    for ln, tok, val, pct, ctx in extract_numbers(text):
        if tok in allow:
            allowed.append({"line": ln, "token": tok})
            continue
        hit = next((cid for cid, cv in pool if covers(cv, tok)), None)
        row = {"line": ln, "token": tok, "context": ctx}
        if hit:
            covered.append({**row, "claim": hit})
        else:
            uncovered.append(row)
    n = len(covered) + len(uncovered) + len(allowed)
    return {"numbers": n, "covered": covered, "uncovered": uncovered, "allowed": allowed,
            "ok": not uncovered, "coverage": (len(covered) + len(allowed)) / n if n else 1.0}


def scan_file(path, book, allow=()):
    p = Path(path)
    return scan_report(p.read_text(encoding="utf-8", errors="replace"), book, allow)


def format_scan(res, limit=25):
    L = [f"{res['numbers']} numbers, {len(res['covered'])} covered by a claim, {len(res['uncovered'])} without one"
         f" ({res['coverage']:.0%} coverage)"]
    L += [f"  line {u['line']}: {u['token']!r} in: {u['context']}" for u in res["uncovered"][:limit]]
    if len(res["uncovered"]) > limit:
        L.append(f"  ... and {len(res['uncovered']) - limit} more")
    return "\n".join(L)
