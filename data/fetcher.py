"""Data fetcher using yfinance for both US and Taiwan stocks."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    Index is DatetimeIndex.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # yfinance may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Ensure standard column names
        df = df.rename(columns={
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        st.warning(f"Failed to fetch price data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_info(ticker: str) -> dict:
    """Fetch fundamental info from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    """Fetch financial statements (income, balance, cashflow)."""
    try:
        t = yf.Ticker(ticker)
        result = {
            "income_stmt": t.income_stmt if hasattr(t, "income_stmt") else pd.DataFrame(),
            "balance_sheet": t.balance_sheet if hasattr(t, "balance_sheet") else pd.DataFrame(),
            "cashflow": t.cashflow if hasattr(t, "cashflow") else pd.DataFrame(),
            "quarterly_income": t.quarterly_income_stmt if hasattr(t, "quarterly_income_stmt") else pd.DataFrame(),
            "quarterly_balance": t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else pd.DataFrame(),
            "quarterly_cashflow": t.quarterly_cashflow if hasattr(t, "quarterly_cashflow") else pd.DataFrame(),
        }
        return result
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dividends(ticker: str) -> pd.DataFrame:
    """Fetch dividend history."""
    try:
        t = yf.Ticker(ticker)
        divs = t.dividends
        if divs is not None and not divs.empty:
            return divs.to_frame(name="Dividend")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_multiple_prices(
    tickers: list,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch Close prices for multiple tickers. Returns DataFrame with ticker columns."""
    try:
        df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"]
        else:
            # Single ticker
            return df[["Close"]].rename(columns={"Close": tickers[0]})
    except Exception:
        return pd.DataFrame()


def get_fundamental_metrics(info: dict) -> dict:
    """Extract key fundamental metrics from yfinance info dict."""
    def safe_get(key, default=None):
        v = info.get(key, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v

    return {
        "PE": safe_get("trailingPE"),
        "Forward_PE": safe_get("forwardPE"),
        "PB": safe_get("priceToBook"),
        "EPS": safe_get("trailingEps"),
        "Forward_EPS": safe_get("forwardEps"),
        "ROE": safe_get("returnOnEquity"),
        "ROA": safe_get("returnOnAssets"),
        "Dividend_Yield": safe_get("dividendYield"),
        "Revenue_Growth": safe_get("revenueGrowth"),
        "DE_Ratio": safe_get("debtToEquity"),
        "Operating_Margin": safe_get("operatingMargins"),
        "Profit_Margin": safe_get("profitMargins"),
        "Current_Ratio": safe_get("currentRatio"),
        "Quick_Ratio": safe_get("quickRatio"),
        "FCF": safe_get("freeCashflow"),
        "Market_Cap": safe_get("marketCap"),
        "52W_High": safe_get("fiftyTwoWeekHigh"),
        "52W_Low": safe_get("fiftyTwoWeekLow"),
        "Avg_Volume": safe_get("averageVolume"),
        "Beta": safe_get("beta"),
        "Short_Name": safe_get("shortName", ""),
        "Sector": safe_get("sector", ""),
        "Industry": safe_get("industry", ""),
    }
