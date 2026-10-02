"""Creator K24b - sample-size arithmetic for the recursion's A/B tests (C77 CR191-198: a process decision needs power).

creator.model.improvement_verdict compares the pooled primary rates with z = 2 and an UNPAIRED standard error
sqrt(se_base^2 + se_cand^2). So the number of tasks per arm the VERDICT needs is `required_n` below, not the (smaller) paired one;
`paired_required_n` is reported next to it because the solvers are deterministic and the same tasks run in both arms.
Everything is stdlib arithmetic on counts, so a pilot's numbers can be re-derived by hand."""
from __future__ import annotations

import math
from typing import Optional, Sequence

Z_ALPHA = 2.0        # the verdict's z
Z_POWER = 0.84       # 80% power


def unpaired_se(p_base: float, p_cand: float, n: int) -> float:
    """The standard error improvement_verdict uses for n tasks in each arm."""
    return math.sqrt((p_base * (1 - p_base) + p_cand * (1 - p_cand)) / n)


def required_n(p_base: float, effect: float, z_alpha: float = Z_ALPHA, z_power: float = Z_POWER) -> int:
    """Tasks per arm so a true gain of `effect` clears the verdict's interval with 80% power."""
    p_cand = min(max(p_base + effect, 0.0), 1.0)
    var = p_base * (1 - p_base) + p_cand * (1 - p_cand)
    if effect <= 0 or var <= 0:
        raise ValueError("effect must be positive and the rates not both 0 or 1")
    return math.ceil((z_alpha + z_power) ** 2 * var / effect ** 2)


def detectable_effect(p_base: float, n: int, z_alpha: float = Z_ALPHA, z_power: float = Z_POWER) -> float:
    """The smallest true gain the verdict detects with 80% power at n tasks per arm (fixed point: the candidate's variance depends on it)."""
    eff = 0.1
    for _ in range(50):
        p_cand = min(p_base + eff, 1.0)
        eff = (z_alpha + z_power) * math.sqrt((p_base * (1 - p_base) + p_cand * (1 - p_cand)) / n)
    return eff


def paired_required_n(discordant: float, effect: float, z_alpha: float = Z_ALPHA, z_power: float = Z_POWER) -> int:
    """Tasks needed when the same tasks run in both arms: the gain's variance is the discordant-pair rate / n."""
    if effect <= 0 or discordant < effect:
        raise ValueError("need 0 < effect <= discordant rate")
    return math.ceil((z_alpha + z_power) ** 2 * discordant / effect ** 2)


def pilot(base: Sequence[bool], cand: Sequence[bool]) -> dict[str, Optional[float]]:
    """Noise and effect of one pilot: per-task solved flags of both arms on the SAME tasks."""
    if len(base) != len(cand) or not base:
        raise ValueError("a pilot needs the same non-empty task list in both arms")
    n = len(base)
    pb, pc = sum(base) / n, sum(cand) / n
    disc = sum(1 for b, c in zip(base, cand) if b != c) / n
    paired_var = max(disc - (pc - pb) ** 2, 0.0) / n
    gain = pc - pb
    return {"n": n, "p_base": pb, "p_cand": pc, "gain": gain, "discordant": disc,
            "unpaired_se": unpaired_se(pb, pc, n), "paired_se": math.sqrt(paired_var),
            "detectable_unpaired": detectable_effect(min(max(pb, 0.05), 0.95), n),
            "n_for_observed_gain_unpaired": required_n(pb, gain) if gain > 0 and 0 < pb < 1 else None,
            "n_for_observed_gain_paired": paired_required_n(disc, gain) if 0 < gain <= disc else None}
