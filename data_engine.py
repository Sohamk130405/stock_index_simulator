"""
data_engine.py — Core data simulation and index logic
All processing done with Python + Pandas (no external financial APIs)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime


# ─── Company Universe ────────────────────────────────────────────────────────

COMPANY_UNIVERSE = [
    # Technology
    {"name": "AlphaTech",    "ticker": "ALPH", "sector": "Technology",   "shares": 500_000_000, "base_price": 320.0},
    {"name": "ByteCore",     "ticker": "BYTE", "sector": "Technology",   "shares": 800_000_000, "base_price": 145.0},
    {"name": "CloudNine",    "ticker": "CLN9", "sector": "Technology",   "shares": 300_000_000, "base_price": 275.0},
    {"name": "DataStream",   "ticker": "DSTR", "sector": "Technology",   "shares": 450_000_000, "base_price": 188.0},
    {"name": "EdgeSystems",  "ticker": "EDGS", "sector": "Technology",   "shares": 600_000_000, "base_price": 92.0},
    {"name": "FluxAI",       "ticker": "FLXA", "sector": "Technology",   "shares": 200_000_000, "base_price": 410.0},
    # Finance
    {"name": "GoldVault",    "ticker": "GLDV", "sector": "Finance",      "shares": 700_000_000, "base_price": 198.0},
    {"name": "HorizonBank",  "ticker": "HBNK", "sector": "Finance",      "shares": 900_000_000, "base_price": 134.0},
    {"name": "InvestCorp",   "ticker": "INVC", "sector": "Finance",      "shares": 350_000_000, "base_price": 242.0},
    {"name": "JetCapital",   "ticker": "JETC", "sector": "Finance",      "shares": 250_000_000, "base_price": 310.0},
    # Energy
    {"name": "KryptonEnergy","ticker": "KRYP", "sector": "Energy",       "shares": 1_000_000_000, "base_price": 78.0},
    {"name": "LumenPower",   "ticker": "LMNP", "sector": "Energy",       "shares": 800_000_000, "base_price": 95.0},
    {"name": "MegaWatt",     "ticker": "MGWT", "sector": "Energy",       "shares": 550_000_000, "base_price": 112.0},
    {"name": "NovaSolar",    "ticker": "NVSR", "sector": "Energy",       "shares": 400_000_000, "base_price": 156.0},
    # Healthcare
    {"name": "OmegaPharm",   "ticker": "OMPH", "sector": "Healthcare",   "shares": 300_000_000, "base_price": 285.0},
    {"name": "PulseHealth",  "ticker": "PLSH", "sector": "Healthcare",   "shares": 450_000_000, "base_price": 210.0},
    {"name": "QuantumMed",   "ticker": "QNTM", "sector": "Healthcare",   "shares": 200_000_000, "base_price": 370.0},
    {"name": "RenewBio",     "ticker": "RNWB", "sector": "Healthcare",   "shares": 350_000_000, "base_price": 165.0},
    # Consumer
    {"name": "SummitRetail", "ticker": "SMRT", "sector": "Consumer",     "shares": 600_000_000, "base_price": 118.0},
    {"name": "TitanBrands",  "ticker": "TTBR", "sector": "Consumer",     "shares": 500_000_000, "base_price": 142.0},
    {"name": "UltraGoods",   "ticker": "ULGD", "sector": "Consumer",     "shares": 400_000_000, "base_price": 178.0},
    {"name": "VaultLux",     "ticker": "VLUX", "sector": "Consumer",     "shares": 250_000_000, "base_price": 295.0},
    # Industrials
    {"name": "WarpDrive",    "ticker": "WRPD", "sector": "Industrials",  "shares": 700_000_000, "base_price": 88.0},
    {"name": "XcelMachines", "ticker": "XCLM", "sector": "Industrials",  "shares": 550_000_000, "base_price": 105.0},
    {"name": "YieldSteel",   "ticker": "YDST", "sector": "Industrials",  "shares": 800_000_000, "base_price": 72.0},
    # Telecom
    {"name": "ZeroLatency",  "ticker": "ZRLC", "sector": "Telecom",      "shares": 900_000_000, "base_price": 68.0},
    {"name": "ArcSignal",    "ticker": "ARCS", "sector": "Telecom",      "shares": 600_000_000, "base_price": 87.0},
    {"name": "BridgeNet",    "ticker": "BRDG", "sector": "Telecom",      "shares": 450_000_000, "base_price": 115.0},
    # Real Estate
    {"name": "CrestRealty",  "ticker": "CRST", "sector": "Real Estate",  "shares": 300_000_000, "base_price": 195.0},
    {"name": "DomeProperties","ticker": "DOME","sector": "Real Estate",  "shares": 250_000_000, "base_price": 225.0},
]


def initialize_dataframe() -> pd.DataFrame:
    """
    Build the master DataFrame from the company universe.
    Returns a fully computed Pandas DataFrame with market cap and metadata.
    """
    df = pd.DataFrame(COMPANY_UNIVERSE)

    # Add slight randomization to initial prices
    df["price"] = df["base_price"] * np.random.uniform(0.97, 1.03, size=len(df))
    df["price"] = df["price"].round(2)

    # Initial previous price = current price
    df["prev_price"] = df["price"].copy()

    # Compute derived columns
    df = _compute_market_cap(df)
    df["pct_change"] = 0.0
    df["status"] = "neutral"  # 'in_index', 'dropped', 'added', 'neutral'

    return df.reset_index(drop=True)


def _compute_market_cap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market cap column from price × shares_outstanding."""
    df["market_cap"] = (df["price"] * df["shares"]).round(0)
    return df


# ─── Price Simulation ─────────────────────────────────────────────────────────

def simulate_price_tick(df: pd.DataFrame, volatility: float = 1.0) -> pd.DataFrame:
    """
    Apply random price changes to all companies each tick.
    volatility: multiplier on the random walk step (0.5 = calm, 2.0 = chaotic)

    Uses numpy random walk logic; stores previous price for % change display.
    """
    df = df.copy()
    df["prev_price"] = df["price"].copy()

    # Percentage-based random walk: each company has its own drift
    n = len(df)
    base_sigma = 0.008 * volatility  # ~0.8% std dev per tick by default
    pct_moves = np.random.normal(loc=0.0, scale=base_sigma, size=n)

    df["price"] = (df["price"] * (1 + pct_moves)).round(2)

    # Floor prices at $1 to avoid negative prices
    df["price"] = df["price"].clip(lower=1.0)

    # Recompute pct_change and market cap
    df["pct_change"] = ((df["price"] - df["prev_price"]) / df["prev_price"] * 100).round(3)
    df = _compute_market_cap(df)

    return df


# ─── Index Construction ───────────────────────────────────────────────────────

def construct_index(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Select top N companies by market cap (the 'index constituents').
    Assign weights proportional to market cap (market-cap weighting).

    Returns a filtered DataFrame of index members with weight column.
    """
    # Sort descending by market cap (Pandas sort + head)
    sorted_df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    index_df = sorted_df.head(top_n).copy()

    total_mcap = index_df["market_cap"].sum()
    index_df["weight"] = (index_df["market_cap"] / total_mcap * 100).round(3)

    return index_df


def compute_index_value(index_df: pd.DataFrame, base_value: float = 1000.0,
                        base_mcap: float = None) -> float:
    """
    Compute current index value using total market cap of constituents,
    normalized to a base value of 1000 at inception.
    """
    current_total = index_df["market_cap"].sum()
    if base_mcap is None or base_mcap == 0:
        return base_value
    return round(base_value * (current_total / base_mcap), 2)


# ─── Rebalancing Logic ────────────────────────────────────────────────────────

def rebalance(df: pd.DataFrame, current_index_tickers: set,
              top_n: int = 10) -> tuple[pd.DataFrame, list, list]:
    """
    Determine which companies stay in, get dropped, or get added to the index.

    Returns:
        new_index_df: DataFrame of new index constituents
        added: list of tickers newly added
        removed: list of tickers removed from index
    """
    new_index_df = construct_index(df, top_n=top_n)
    new_tickers = set(new_index_df["ticker"].tolist())

    added   = list(new_tickers - current_index_tickers)
    removed = list(current_index_tickers - new_tickers)

    return new_index_df, added, removed


# ─── Sector Aggregation ───────────────────────────────────────────────────────

def sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by sector, aggregate market cap and count.
    Demonstrates Pandas groupby + aggregation.
    """
    summary = (
        df.groupby("sector")
          .agg(
              total_market_cap=("market_cap", "sum"),
              avg_price=("price", "mean"),
              num_companies=("name", "count"),
          )
          .reset_index()
          .sort_values("total_market_cap", ascending=False)
    )
    summary["total_market_cap_B"] = (summary["total_market_cap"] / 1e9).round(2)
    return summary


def filter_by_sector(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Filter DataFrame rows for a given sector."""
    if sector == "All":
        return df
    return df[df["sector"] == sector].copy()


def top_movers(df: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return top N gainers and losers by pct_change using Pandas sort."""
    sorted_df = df.sort_values("pct_change", ascending=False)
    gainers = sorted_df.head(n)[["ticker", "name", "price", "pct_change", "sector"]]
    losers  = sorted_df.tail(n)[["ticker", "name", "price", "pct_change", "sector"]]
    return gainers, losers
