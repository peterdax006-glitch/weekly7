"""Canon C55 / blind reruns: the Test path's stock choice must not depend on what the tickers are CALLED. Scores are
averaged percentile ranks, so exact ties are common; with a content tie-break (volatility) a random rename picks the
same stocks. Live's default (no tie-break) is unchanged, and without the tie-break the old name dependence is still
visible - so this test can fail."""
import numpy as np
import pandas as pd

from engine import policy

CFG = {"k": 3, "exit_q": 0.8, "max_per_sector": None, "pick": "top", "pool_q": 0.7, "stress_thr": None, "stress_k": 2,
       "trend_filter": None, "trend_gross": 0.0}


def market(seed=0, n=40):
    rng = np.random.default_rng(seed)
    names = [f"N{i:02d}" for i in range(n)]
    score = pd.Series(np.round(rng.uniform(0, 1, n) * 4) / 4, index=names)      # heavy ties: only 5 distinct values
    vol = pd.Series(rng.uniform(0.01, 0.05, n), index=names)
    return score, vol


def rename(score, vol, seed):
    rng = np.random.default_rng(seed)
    new = [f"Z{x:03d}" for x in rng.permutation(len(score))]                   # random codes: order scrambled
    mp = dict(zip(score.index, new))
    # the feed orders rows by (code) name, so a rename also REORDERS rows - which is what reaches tie-breaks
    return score.rename(mp).sort_index(), vol.rename(mp).sort_index(), {v: k for k, v in mp.items()}


def test_test_path_is_rename_invariant():
    for seed in range(5):
        score, vol = market(seed)
        base = set(policy.regime_targets(score, [], CFG, vol, {}, {}).index)
        for r in range(4):
            s2, v2, back = rename(score, vol, 100 + r)
            got = {back[t] for t in policy.regime_targets(s2, [], CFG, v2, {}, {}).index}
            assert got == base, (seed, r)


def test_hivol_mode_is_rename_invariant():
    cfg = {**CFG, "pick": "hivol"}
    score, vol = market(7)
    vol = vol.round(2)                                                          # ties in volatility too
    base = set(policy.regime_targets(score, [], cfg, vol, {}, {}).index)
    s2, v2, back = rename(score, vol, 3)
    assert {back[t] for t in policy.regime_targets(s2, [], cfg, v2, {}, {}).index} == base


def test_without_tiebreak_names_can_decide():
    """Control: the old path (no tie-break, as Live uses) DOES change with a rename on tied scores."""
    changed = 0
    for seed in range(10):
        score, vol = market(seed)
        base = set(policy.topk_targets(score, [], k=3, exit_q=0.8).index)
        s2, v2, back = rename(score, vol, 5)
        changed += {back[t] for t in policy.topk_targets(s2, [], k=3, exit_q=0.8).index} != base
    assert changed > 0


def test_live_default_behaviour_unchanged():
    score, vol = market(1)
    old = list(score.sort_values(ascending=False).index)
    fill = [t for t in old][:3]
    assert list(policy.topk_targets(score, [], k=3, exit_q=0.8).index) == fill
