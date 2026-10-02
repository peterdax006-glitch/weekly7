"""K13 meta: development strategies as explicit parameterised policies, outcome
tracking per problem class, strategy selection, and process-improvement proposals.

Pure standard library.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace

TEST_MODES = ("test_first", "code_first")

# Allowed (inclusive) integer ranges for numeric policy parameters.
BOUNDS = {
    "research_budget": (0, 20),
    "design_breadth": (1, 6),
    "reviewer_depth": (0, 5),
    "max_retries": (0, 10),
}


@dataclass(frozen=True)
class Strategy:
    """A named, parameterised development policy.

    Fields: research_budget, design_breadth, test_mode (one of TEST_MODES),
    reviewer_depth, agent (agent choice) and max_retries. Use validate() to check it.
    """
    name: str
    research_budget: int = 3
    design_breadth: int = 2
    test_mode: str = "test_first"
    reviewer_depth: int = 1
    agent: str = "default"
    max_retries: int = 2

    def to_dict(self) -> dict:
        """Plain-dict (JSON-serialisable) form of the strategy."""
        return asdict(self)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems ([] when the strategy is valid)."""
        problems = []
        if not str(self.name).strip():
            problems.append("name must be non-empty")
        for key, (lo, hi) in BOUNDS.items():
            v = getattr(self, key)
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                problems.append(f"{key} must be an int in [{lo}, {hi}], got {v!r}")
        if self.test_mode not in TEST_MODES:
            problems.append(f"test_mode must be one of {TEST_MODES}, got {self.test_mode!r}")
        if not str(self.agent).strip():
            problems.append("agent must be non-empty")
        return problems


@dataclass
class Outcome:
    """Result of applying one strategy to one problem class: success flag and cost."""
    strategy: str
    problem_class: str
    success: bool
    cost: float = 0.0


@dataclass
class Proposal:
    """A process-improvement proposal: a target policy/code area, a change and the evidence."""
    target: str
    change: str
    rationale: str
    evidence: dict = field(default_factory=dict)


@dataclass
class MetaLearner:
    """Registry of strategies plus outcome tracking and selection per problem class."""
    strategies: dict = field(default_factory=dict)
    outcomes: list = field(default_factory=list)

    def register(self, strategy: Strategy) -> Strategy:
        """Add a strategy and return it.

        Raises ValueError if it is invalid or its name is already registered.
        """
        problems = strategy.validate()
        if problems:
            raise ValueError("; ".join(problems))
        if strategy.name in self.strategies:
            raise ValueError(f"strategy already registered: {strategy.name!r}")
        self.strategies[strategy.name] = strategy
        return strategy

    def record(self, strategy: str, problem_class: str, success: bool, cost: float = 0.0) -> Outcome:
        """Record an outcome for a registered strategy and return it.

        Raises ValueError for an unknown strategy, blank problem class or negative cost.
        """
        if strategy not in self.strategies:
            raise ValueError(f"unknown strategy: {strategy!r}")
        if not str(problem_class).strip():
            raise ValueError("problem_class must be non-empty")
        if not (math.isfinite(cost) and cost >= 0):
            raise ValueError("cost must be a finite number >= 0")
        out = Outcome(strategy, str(problem_class).strip(), bool(success), float(cost))
        self.outcomes.append(out)
        return out

    def stats(self, strategy: str, problem_class: str | None = None) -> dict:
        """Counts for a strategy (optionally one problem class): n, successes, success_rate
        (None when n == 0) and mean_cost (None when n == 0)."""
        rows = [o for o in self.outcomes if o.strategy == strategy
                and (problem_class is None or o.problem_class == problem_class)]
        n = len(rows)
        wins = sum(1 for o in rows if o.success)
        return {"n": n, "successes": wins,
                "success_rate": wins / n if n else None,
                "mean_cost": sum(o.cost for o in rows) / n if n else None}

    def select(self, problem_class: str, explore: float = 1.0) -> Strategy:
        """Pick a strategy for a problem class with a UCB1 rule over recorded outcomes.

        Untried strategies are chosen first (in registration order); otherwise the highest
        success_rate + explore * sqrt(2 ln N / n) wins, ties going to the earlier
        registration. Raises ValueError when no strategy is registered.
        """
        if not self.strategies:
            raise ValueError("no strategies registered")
        st = {name: self.stats(name, problem_class) for name in self.strategies}
        for name, s in st.items():
            if s["n"] == 0:
                return self.strategies[name]
        total = sum(s["n"] for s in st.values())
        best, best_score = None, -math.inf
        for name, s in st.items():
            score = s["success_rate"] + explore * math.sqrt(2 * math.log(total) / s["n"])
            if score > best_score:
                best, best_score = name, score
        return self.strategies[best]

    def propose_improvements(self, min_n: int = 5, floor: float = 0.5) -> list[Proposal]:
        """Propose changes to the policies themselves from outcome evidence.

        For each (strategy, problem class) with at least min_n outcomes and success rate
        below floor, propose raising reviewer_depth/retries (or switching to test_first);
        strategies with no outcomes yield no proposals. Returns [] with no evidence.
        """
        groups: dict = {}
        for o in self.outcomes:
            groups.setdefault((o.strategy, o.problem_class), []).append(o)
        proposals = []
        for (name, pc), rows in sorted(groups.items()):
            n = len(rows)
            rate = sum(1 for o in rows if o.success) / n
            if n < min_n or rate >= floor:
                continue
            strat = self.strategies[name]
            if strat.test_mode == "code_first":
                change = "test_mode: code_first -> test_first"
            else:
                change = f"reviewer_depth: {strat.reviewer_depth} -> {min(strat.reviewer_depth + 1, BOUNDS['reviewer_depth'][1])}"
            proposals.append(Proposal(
                target=f"strategy:{name}", change=change,
                rationale=f"success rate {rate:.2f} < {floor:.2f} on {pc!r} over {n} outcomes",
                evidence={"problem_class": pc, "n": n, "success_rate": rate}))
        return proposals

    def apply(self, proposal_target: str, **changes) -> Strategy:
        """Register a variant of an existing strategy with changed parameters and return it.

        The variant is named '<base>+v<k>'. Raises ValueError for an unknown base or an
        invalid resulting strategy.
        """
        base = proposal_target.removeprefix("strategy:")
        if base not in self.strategies:
            raise ValueError(f"unknown strategy: {base!r}")
        k = 1
        while f"{base}+v{k}" in self.strategies:
            k += 1
        return self.register(replace(self.strategies[base], name=f"{base}+v{k}", **changes))
