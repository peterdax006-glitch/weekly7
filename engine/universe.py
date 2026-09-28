"""Tradable universe: US common stocks on NYSE / Nasdaq / AMEX, no ETFs, warrants, units,
rights, preferreds or test issues. Price/liquidity filters are applied later on bar data."""
import json, re
import pandas as pd
import requests
from .config import DATA, SEC_UA

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
# Canon C11: crypto is off limits - crypto funds/trusts and companies whose business is crypto.
CRYPTO_NAME = re.compile(r"bitcoin|crypto|blockchain|ethereum|\bether\b|digital asset|\bcoin\b|\bbtc\b|solana|\bxrp\b", re.I)
CRYPTO_TICKERS = {"COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK", "BITF", "HIVE", "CIFR", "IREN", "WULF", "BTBT",
                  "CORZ", "BTDR", "GBTC", "ETHE", "BITO", "IBIT", "FBTC", "ARKB", "BITB", "HODL", "BRRR", "EZBC",
                  "GLXY", "HOOD_CRYPTO", "SMLR", "BKKT", "APLD", "CAN", "SDIG", "GREE", "ARBK", "BTCS", "SOS"}
EXCLUDE = re.compile(r"warrant|\bunits?\b|\brights?\b|preferred|depositary|notes due|debenture|"
                     r"trust pref|%|acquisition corp|\bspac\b|beneficial interest|limited partnership",
                     re.I)


def refresh_sources():
    h = {"User-Agent": SEC_UA}
    (DATA / "nasdaqtraded.txt").write_bytes(requests.get(NASDAQ_URL, headers=h, timeout=60).content)
    (DATA / "company_tickers_exchange.json").write_bytes(requests.get(SEC_TICKERS_URL, headers=h, timeout=60).content)


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(DATA / "nasdaqtraded.txt", sep="|", dtype=str).dropna(subset=["Symbol"])
    df = df[df["Symbol"] != df["Symbol"].iloc[-1]] if df["Symbol"].iloc[-1].startswith("File Creation") else df
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N") & (df["NextShares"] != "Y")]
    df = df[df["Listing Exchange"].isin(["N", "Q", "A", "P"])]           # NYSE, Nasdaq, AMEX, Arca
    df = df[~df["Security Name"].str.contains(EXCLUDE, na=False)]
    df = df[~df["Security Name"].str.contains(CRYPTO_NAME, na=False) & ~df["Symbol"].isin(CRYPTO_TICKERS)]
    df = df[df["Symbol"].str.fullmatch(r"[A-Z]{1,5}")]                   # drops class/when-issued suffixes
    sec = json.loads((DATA / "company_tickers_exchange.json").read_text())
    sec = pd.DataFrame(sec["data"], columns=sec["fields"])
    df = df.merge(sec[["ticker", "cik"]], left_on="Symbol", right_on="ticker", how="inner")  # must be an SEC filer
    out = df[["Symbol", "Security Name", "Listing Exchange", "cik"]].rename(
        columns={"Symbol": "ticker", "Security Name": "name", "Listing Exchange": "exch"})
    return out.drop_duplicates("ticker").reset_index(drop=True)


if __name__ == "__main__":
    u = load_universe()
    print(len(u)); print(u.head())
