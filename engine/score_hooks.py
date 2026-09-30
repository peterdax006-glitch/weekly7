"""Fitting the B07 score hooks (engine.backtest.ScoreHooks). Kept out of engine.backtest because backtest is on the trader's
import closure and engine.trust is research-only (leak audit channel 8d, 29 Sep): the trader path may carry a ScoreHooks, but
only this research-side module builds the TrustTable / DirectionEngine inside it."""
import pandas as pd

from .backtest import ScoreHooks


def fit_score_hooks(X, y, now, trust=False, info=None, indicators=None, direction=False, direction_inputs=None, up=None,
                    movers=None, trust_weight=0.5, trust_kw=None, direction_kw=None) -> ScoreHooks:
    """Fit what the flags switch on, strictly as of `now` (both modules drop rows whose forward window had not closed)."""
    h = ScoreHooks(trust_weight=trust_weight)
    if trust:
        from .trust import TrustTable
        ind = list(indicators) if indicators is not None else [c for c in X.columns if not c.startswith("m_")]
        h.trust = TrustTable(**(trust_kw or {})).fit(X, y, info, now, ind)
        h.trust_on, h.trust_indicators, h.trust_info = True, tuple(ind), info
    if direction:
        from .direction import DirectionEngine
        h.direction = DirectionEngine(**(direction_kw or {})).fit(direction_inputs, up, now, movers)
        h.direction_on, h.direction_inputs, h.direction_now = True, direction_inputs, pd.Timestamp(now)
    return h.validate()
