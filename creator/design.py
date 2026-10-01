"""K08 design: N alternative designs per decision, assumptions, failure modes, cost/risk,
adversarial critique, explicit selection criteria, rejected alternatives kept.

Pure standard library.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

DEFAULT_CRITERIA = {"cost": 0.3, "risk": 0.4, "benefit": 0.3}


@dataclass(frozen=True)
class DesignOption:
    """One alternative design. cost, risk and benefit are in [0,1] (lower cost/risk is better)."""
    name: str
    summary: str = ""
    assumptions: tuple = ()
    failure_modes: tuple = ()
    cost: float = 0.5
    risk: float = 0.5
    benefit: float = 0.5


@dataclass
class Decision:
    """The outcome of choosing among options: the selected one, rejected ones with reasons, criteria."""
    question: str
    selected: DesignOption | None
    scores: dict = field(default_factory=dict)
    rejected: list = field(default_factory=list)  # (DesignOption, reason) pairs
    criteria: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Plain-dict form (JSON-serialisable) of the decision, including rejected alternatives."""
        return {
            "question": self.question,
            "selected": asdict(self.selected) if self.selected else None,
            "scores": dict(self.scores),
            "rejected": [{"option": asdict(o), "reason": r} for o, r in self.rejected],
            "criteria": dict(self.criteria),
        }


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def generate_options(question: str, candidates: list[dict], n: int = 3) -> list[DesignOption]:
    """Build up to n distinct DesignOptions from candidate dicts (keys: name, summary,
    assumptions, failure_modes, cost, risk, benefit).

    Candidates without a name, or with a duplicate name (case-insensitive), are skipped;
    numeric fields are clamped to [0,1]. Empty candidates or n <= 0 yields [].
    """
    out: list[DesignOption] = []
    seen: set[str] = set()
    for c in candidates or []:
        if len(out) >= max(0, n):
            break
        name = str(c.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(DesignOption(
            name=name,
            summary=str(c.get("summary", "")),
            assumptions=tuple(c.get("assumptions", ()) or ()),
            failure_modes=tuple(c.get("failure_modes", ()) or ()),
            cost=_clamp(c.get("cost", 0.5)),
            risk=_clamp(c.get("risk", 0.5)),
            benefit=_clamp(c.get("benefit", 0.5)),
        ))
    return out


def critique(option: DesignOption) -> list[str]:
    """Adversarial critique: a list of concrete objections to the option (empty means none found).

    Flags missing assumptions, missing failure modes, high risk, high cost and low benefit.
    """
    issues: list[str] = []
    if not option.assumptions:
        issues.append("no assumptions stated")
    if not option.failure_modes:
        issues.append("no failure modes considered")
    if option.risk > 0.7:
        issues.append("risk is high")
    if option.cost > 0.7:
        issues.append("cost is high")
    if option.benefit < 0.3:
        issues.append("benefit is low")
    return issues


def score(option: DesignOption, criteria: dict | None = None) -> float:
    """Weighted score in [0,1], higher is better: cost and risk count as (1 - value).

    Unknown criteria keys are ignored; non-positive total weight scores 0.0. Each critique
    issue subtracts a 0.02 penalty (floored at 0).
    """
    crit = DEFAULT_CRITERIA if criteria is None else criteria
    total = 0.0
    weight = 0.0
    for key, w in crit.items():
        if key not in ("cost", "risk", "benefit") or w <= 0:
            continue
        v = getattr(option, key)
        total += w * (v if key == "benefit" else 1.0 - v)
        weight += w
    if weight <= 0:
        return 0.0
    return max(0.0, total / weight - 0.02 * len(critique(option)))


def select(question: str, options: list[DesignOption], criteria: dict | None = None) -> Decision:
    """Pick the highest-scoring option (ties broken by name); keep every other option as rejected
    with the reason. No options yields a Decision with selected=None.
    """
    crit = dict(DEFAULT_CRITERIA if criteria is None else criteria)
    if not options:
        return Decision(question, None, {}, [], crit)
    each = [(score(o, crit), o) for o in options]          # per option, not per name: same-named options stay distinct
    each.sort(key=lambda t: (-t[0], t[1].name))
    scores: dict = {}
    for s, o in each:
        scores.setdefault(o.name, s)                       # the best-scoring option of a name represents it
    top, best = each[0]
    rejected = [(o, f"scored {s:.3f} < {top:.3f} ({best.name})") for s, o in each[1:]]
    return Decision(question, best, scores, rejected, crit)
