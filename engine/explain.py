"""Plain-English names for features and market readings, and the top reasons behind a pick. Pure text helpers
shared by the live trader, research scripts and the public pages. Lives outside engine/live.py so research
code never imports the live order path (Bible Phase 28 / quality gate import boundary)."""


PLAIN = {"ear": "strong earnings reaction", "ins_buyers30": "insiders buying", "ins_value30": "large insider purchases",
         "ins_officer30": "officers buying", "ins_opportunistic30": "insider buying cluster",
         "dist_52wh": "near its 52-week high", "ind_mom60": "strong industry (3 months)", "ind_mom20": "strong industry (1 month)",
         "frog": "steady, not spiky, uptrend", "ev_activist": "activist stake filed", "overnight20": "overnight strength",
         "intraday20": "intraday strength", "r5_nonews": "quiet pullback likely to revert", "r5_news": "news-driven move",
         "mom_12_1": "12-month momentum", "vol_surge5": "rising volume", "vol_surge1": "volume spike today",
         "rel_ind60": "beating its industry", "rel_ind20": "beating its industry (1 month)", "max20": "no lottery-style spike",
         "atr_pct": "volatility profile", "vol20": "volatility profile", "vol_ratio": "volatility trend",
         "r1": "yesterday's move", "r5": "last week's move", "r20": "last month's move", "r60": "3-month trend",
         "r120": "6-month trend", "dist_ma50": "above its 50-day average", "dist_ma200": "above its 200-day average",
         "skew60": "return shape", "log_dv": "liquidity", "close_loc": "closed near the day's high",
         "range_compress": "tight trading range", "gap_today": "today's opening gap", "days_since_earn": "earnings timing",
         "days_to_earn": "earnings coming up", "earn_in_week": "earnings this week", "news5": "recent filings",
         "ev_offering": "no share offering", "ev_red_flag": "no red-flag filings", "min20": "no recent crash day",
         "vol_spread": "options: calls pricier than puts", "cp_volume": "options: call-heavy volume"}


MARKET_PLAIN = {"m_vix": "market fear (VIX level)", "m_vix_chg5": "market fear rising (VIX 5-day change)",
                "m_vix_term": "market stress (short vs 3-month VIX)", "m_spy_ma50": "market vs its 50-day average",
                "m_spy_ma200": "market vs its 200-day average", "m_spy_r5": "market's last week",
                "m_breadth": "market breadth (share of stocks above 50-day avg)",
                "m_dispersion": "how differently stocks are moving (dispersion)"}


def plain(k):
    if k.startswith("m_"):
        return MARKET_PLAIN.get(k, "market conditions")
    return PLAIN.get(k, k.replace("_", " "))


def explain(contrib_row, evid_row=None, top=3):
    s = contrib_row.sort_values(ascending=False)
    out = []
    for k in s.index:
        if s[k] <= 0 or len(out) >= top:
            break
        w = plain(k)
        if w not in out:
            out.append(w)
    return out
