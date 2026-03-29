"""Asset universe per brief §2.1 — 50+ instruments across 6 classes."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    FX = "fx"
    VOLATILITY = "volatility"
    CRYPTO = "crypto"


class AssetMeta(TypedDict, total=False):
    ticker: str
    asset_class: AssetClass
    risk_factors: list[str]


UNIVERSE: dict[str, AssetMeta] = {}

# Equities: 20 large-cap + 5 sector ETFs (brief)
_EQUITIES = [
    ("AAPL", ["market_beta", "earnings"]),
    ("MSFT", ["market_beta", "sector_tech"]),
    ("GOOGL", ["market_beta", "sector_tech"]),
    ("AMZN", ["market_beta", "consumer"]),
    ("NVDA", ["market_beta", "sector_tech"]),
    ("META", ["market_beta", "sector_tech"]),
    ("BRK-B", ["market_beta", "financial"]),
    ("JPM", ["market_beta", "financial"]),
    ("GS", ["market_beta", "financial"]),
    ("BAC", ["market_beta", "financial"]),
    ("XOM", ["market_beta", "energy"]),
    ("CVX", ["market_beta", "energy"]),
    ("UNH", ["market_beta", "healthcare"]),
    ("LLY", ["market_beta", "healthcare"]),
    ("SPY", ["market_beta"]),
    ("QQQ", ["market_beta", "sector_tech"]),
    ("IWM", ["market_beta", "small_cap"]),
    ("XLF", ["sector_exposure", "financial"]),
    ("XLE", ["sector_exposure", "energy"]),
    ("XLK", ["sector_exposure", "tech"]),
    ("WMT", ["market_beta", "consumer"]),
    ("JNJ", ["market_beta", "healthcare"]),
    ("PG", ["market_beta", "consumer"]),
    ("HD", ["market_beta", "consumer"]),
    ("DIS", ["market_beta", "consumer"]),
]

for t, factors in _EQUITIES:
    UNIVERSE[t] = {"ticker": t, "asset_class": AssetClass.EQUITY, "risk_factors": factors}

_FIXED = [
    ("SHY", ["duration", "rates"]),
    ("IEF", ["duration", "rates"]),
    ("TLT", ["duration", "rates"]),
    ("HYG", ["credit_spread"]),
    ("EMB", ["em_sovereign", "credit_spread"]),
]
for t, factors in _FIXED:
    UNIVERSE[t] = {"ticker": t, "asset_class": AssetClass.FIXED_INCOME, "risk_factors": factors}

_COMMODITIES = [
    ("GLD", ["commodity"]),
    ("SLV", ["commodity"]),
    ("USO", ["commodity"]),
    ("UNG", ["commodity"]),
    ("CORN", ["commodity"]),
    ("WEAT", ["commodity"]),
    ("CPER", ["commodity"]),
    ("PDBC", ["commodity"]),
]
for t, factors in _COMMODITIES:
    UNIVERSE[t] = {"ticker": t, "asset_class": AssetClass.COMMODITY, "risk_factors": factors}

_FX = [
    ("UUP", ["usd", "carry"]),
    ("FXE", ["eur"]),
    ("FXY", ["jpy"]),
    ("FXB", ["gbp"]),
    ("FXC", ["cad"]),
    ("FXA", ["aud"]),
    ("CYB", ["cny"]),
    ("EWZ", ["brl", "em_equity"]),
]
for t, factors in _FX:
    UNIVERSE[t] = {"ticker": t, "asset_class": AssetClass.FX, "risk_factors": factors}

_VOL = [
    ("^VIX", ["vol_level", "risk_off"]),
    ("UVXY", ["vol_level", "vol_of_vol"]),
    ("SVXY", ["vol_level"]),
    ("VXX", ["vol_level", "vol_of_vol"]),
]
for t, factors in _VOL:
    UNIVERSE[t] = {"ticker": t, "asset_class": AssetClass.VOLATILITY, "risk_factors": factors}

_CRYPTO = [
    ("BTC-USD", ["crypto", "liquidity"]),
    ("ETH-USD", ["crypto", "liquidity"]),
    ("SOL-USD", ["crypto", "liquidity"]),
    ("BNB-USD", ["crypto", "liquidity"]),
]
for t, factors in _CRYPTO:
    UNIVERSE[t] = {"ticker": t, "asset_class": AssetClass.CRYPTO, "risk_factors": factors}

# Vol ETPs + crypto: GARCH/DCC and MC-VaR explode; drawdown rules spam CRITICAL in simulation.
_EXCLUDE_FROM_CORE: frozenset[str] = frozenset(
    {
        "^VIX",
        "UVXY",
        "SVXY",
        "VXX",
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "BNB-USD",
    }
)


def get_tickers(profile: str = "full") -> list[str]:
    """full = all brief instruments; core = same minus vol/crypto (saner live dashboard)."""
    all_k = list(UNIVERSE.keys())
    if profile == "core":
        return [t for t in all_k if t not in _EXCLUDE_FROM_CORE]
    return all_k
