"""K07 research: questions, evidence, source evaluation, contradictions, confidence, memory, design bridge.

Pure standard library. Web/doc evidence arrives as Evidence objects supplied by
agents; repo evidence is collected here by scanning files.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

SOURCE_WEIGHTS = {"repo": 1.0, "test": 0.9, "docs": 0.7, "web": 0.4, "agent": 0.3}
NEGATIONS = ("not ", "no ", "never ", "cannot ", "without ", "does not ", "doesn't ")
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"}


@dataclass(frozen=True)
class Question:
    """A research question: text, topic, importance in [0,1] and uncertainty in [0,1]."""
    text: str
    topic: str = ""
    importance: float = 0.5
    uncertainty: float = 1.0


@dataclass(frozen=True)
class Evidence:
    """One finding: a claim about a question, from a source of a given kind, with a stance."""
    question: str
    claim: str
    source: str
    kind: str = "agent"
    supports: bool = True


def generate_questions(gaps: list[dict]) -> list[Question]:
    """Turn gap dicts (keys: id, kind, description, importance) into deduplicated questions.

    Empty input gives an empty list. Gaps lacking a description are skipped.
    """
    seen: set[str] = set()
    out: list[Question] = []
    for g in gaps or []:
        desc = str(g.get("description", "")).strip()
        if not desc:
            continue
        kind = str(g.get("kind", "")).strip() or "GENERAL"
        text = f"How should we close: {desc}?"
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        imp = min(1.0, max(0.0, float(g.get("importance", 0.5))))
        unc = 1.0 if kind.upper() == "KNOWLEDGE" else 0.6
        out.append(Question(text, kind, imp, unc))
    return out


def prioritise(questions: list[Question], budget: int | None = None) -> list[Question]:
    """Order questions by importance*uncertainty descending (ties by text); keep at most `budget`.

    A budget of 0 or less yields an empty list; None keeps all.
    """
    ranked = sorted(questions, key=lambda q: (-q.importance * q.uncertainty, q.text))
    if budget is None:
        return ranked
    return ranked[: max(0, budget)]


def collect_repo_evidence(question: str, root: Path, terms: list[str],
                          max_hits: int = 20) -> list[Evidence]:
    """Scan .py/.md files under root for lines containing any of `terms` (case-insensitive).

    Returns at most max_hits Evidence items (kind "repo"/"docs"/"test"). Missing root or no
    terms yields []. Unreadable files are skipped.
    """
    root = Path(root)
    wanted = [t.lower() for t in terms if t.strip()]
    if not wanted or not root.is_dir():
        return []
    hits: list[Evidence] = []
    for p in sorted(root.rglob("*")):
        if len(hits) >= max_hits:
            break
        if not p.is_file() or p.suffix not in (".py", ".md"):
            continue
        if _SKIP_DIRS & set(p.relative_to(root).parts):
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        rel = p.relative_to(root).as_posix()
        kind = "docs" if p.suffix == ".md" else ("test" if p.name.startswith("test_") else "repo")
        for n, line in enumerate(lines, 1):
            if any(t in line.lower() for t in wanted):
                hits.append(Evidence(question, line.strip(), f"{rel}:{n}", kind))
                if len(hits) >= max_hits:
                    break
    return hits


def source_weight(kind: str) -> float:
    """Reliability weight in [0,1] for a source kind; unknown kinds get 0.1."""
    return SOURCE_WEIGHTS.get(kind, 0.1)


def _norm(claim: str) -> str:
    return re.sub(r"\s+", " ", claim.lower().strip().rstrip("."))


def _negated(claim: str) -> bool:
    c = " " + _norm(claim) + " "
    return any(" " + n in c for n in NEGATIONS)


def _core(claim: str) -> str:
    c = " " + _norm(claim) + " "
    for n in NEGATIONS:
        c = c.replace(" " + n, " ")
    return re.sub(r"\s+", " ", c).strip()


def detect_contradictions(evidence: list[Evidence]) -> list[tuple[Evidence, Evidence]]:
    """Pairs of evidence on the same question that disagree.

    Disagreement: opposite `supports` stance on the same normalised claim, or the same claim
    with differing negation. Each pair is reported once, in input order.
    """
    out: list[tuple[Evidence, Evidence]] = []
    for i, a in enumerate(evidence):
        for b in evidence[i + 1:]:
            if a.question != b.question or _core(a.claim) != _core(b.claim):
                continue
            pol_a = a.supports != _negated(a.claim)
            pol_b = b.supports != _negated(b.claim)
            if pol_a != pol_b:
                out.append((a, b))
    return out


def confidence(evidence: list[Evidence]) -> float:
    """Confidence in [0,1] that the evidence supports an answer.

    Weighted share of supporting evidence, scaled by sample size (saturating at 4 items) and
    reduced by contradictions. No evidence gives 0.0.
    """
    if not evidence:
        return 0.0
    total = sum(source_weight(e.kind) for e in evidence)
    support = sum(source_weight(e.kind) for e in evidence if e.supports)
    share = support / total if total else 0.0
    size = min(1.0, len(evidence) / 4)
    penalty = 1.0 / (1.0 + len(detect_contradictions(evidence)))
    return round(max(0.0, min(1.0, share * size * penalty)), 6)


class ResearchMemory:
    """Append-only research memory persisted as JSON lines; never hides contradicting evidence."""

    def __init__(self, path: Path | None = None) -> None:
        """Create a memory; with a path, load any existing records from it."""
        self.path = Path(path) if path else None
        self.items: list[Evidence] = []
        if self.path and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.items.append(Evidence(**json.loads(line)))
                    except (ValueError, TypeError):
                        continue

    def add(self, ev: Evidence) -> None:
        """Record evidence in memory and, if a path is set, append it to the file."""
        self.items.append(ev)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), sort_keys=True) + "\n")

    def recall(self, question: str) -> list[Evidence]:
        """All evidence recorded for exactly this question ([] if never seen)."""
        return [e for e in self.items if e.question == question]


@dataclass
class DesignInput:
    """Bridge object handed to design: a question, its confidence, findings and open issues."""
    question: str
    confidence: float
    findings: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    open: bool = True


def to_design_input(question: str, evidence: list[Evidence],
                    threshold: float = 0.5) -> DesignInput:
    """Summarise evidence for a question as design input.

    `open` is True when confidence is below threshold or contradictions exist, so design treats
    the question as unresolved. Findings list supporting claims, strongest sources first.
    """
    ev = [e for e in evidence if e.question == question]
    conflicts = detect_contradictions(ev)
    conf = confidence(ev)
    ordered = sorted((e for e in ev if e.supports), key=lambda e: -source_weight(e.kind))
    return DesignInput(
        question=question,
        confidence=conf,
        findings=[f"{e.claim} [{e.source}]" for e in ordered],
        contradictions=[f"{a.source} vs {b.source}: {a.claim}" for a, b in conflicts],
        open=conf < threshold or bool(conflicts),
    )
