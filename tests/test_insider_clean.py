"""engine.features.clean_insider: impossible Form 4 dates are dropped (trade after its own filing, trade year before
1990 or after the filing year); legitimately late filings are kept (they enter only on their filing day)."""
import pandas as pd

from engine import features as F


def rows(pairs):
    return pd.DataFrame({"filed": pd.to_datetime([f for f, _ in pairs]), "tdate": pd.to_datetime([t for _, t in pairs]),
                         "symbol": "A", "owner_cik": "1", "value": 5000.0, "relation": "", "title": ""})


def test_normal_rows_kept():
    d = rows([("2020-03-05", "2020-03-03"), ("2020-03-05", "2020-03-05")])
    assert len(F.clean_insider(d)) == 2


def test_trade_after_filing_dropped():
    d = rows([("2020-03-05", "2020-03-09"), ("2020-03-05", "2020-03-03")])
    out = F.clean_insider(d)
    assert len(out) == 1 and F.INSIDER_DROPPED["trade_after_filing"] == 1


def test_typo_years_dropped():
    d = rows([("2020-03-05", "0013-03-03"), ("2020-03-05", "1985-03-03"), ("2020-03-05", "2020-03-03")])
    out = F.clean_insider(d)
    assert len(out) == 1 and F.INSIDER_DROPPED["bad_year"] == 2


def test_late_filing_kept_and_counted():
    d = rows([("2020-03-05", "2015-06-01")])
    out = F.clean_insider(d)
    assert len(out) == 1 and F.INSIDER_DROPPED["over_3y_late"] == 1


def test_empty_and_none_pass_through():
    assert F.clean_insider(None) is None
    e = rows([]).iloc[:0]
    assert F.clean_insider(e).empty
