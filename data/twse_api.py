"""TWSE / TPEx direct API wrapper for Taiwan stock-specific data.

Provides: 三大法人買賣超, 融資融券, 月營收, P/E/P/B/殖利率
All endpoints are free and public. Rate limit: ~3 req / 5 seconds.
"""

import requests
import pandas as pd
import time
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TWSE_BASE_URL, TWSE_OPENAPI_URL, TPEX_OPENAPI_URL, TWSE_REQUEST_DELAY


_last_request_time = 0.0


def _rate_limit():
    """Ensure minimum delay between TWSE requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < TWSE_REQUEST_DELAY:
        time.sleep(TWSE_REQUEST_DELAY - elapsed)
    _last_request_time = time.time()


def _twse_get(path: str, params: dict = None) -> dict:
    """Make a GET request to TWSE legacy API with rate limiting."""
    _rate_limit()
    url = f"{TWSE_BASE_URL}{path}"
    if params is None:
        params = {}
    params["response"] = "json"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("stat") in ("OK", "ok", "正常"):
            return data
        return {}
    except Exception:
        return {}


def _openapi_get(base_url: str, path: str) -> list:
    """Make a GET request to TWSE/TPEx OpenAPI (latest day only)."""
    _rate_limit()
    url = f"{base_url}{path}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


# ------------------------------------------------------------------
# 三大法人買賣超 (Institutional Investors Buy/Sell)
# ------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def fetch_institutional_trading(stock_id: str, date_str: str) -> dict:
    """Fetch institutional buy/sell for a specific stock on a given date.

    Args:
        stock_id: e.g. "2330" (without .TW suffix)
        date_str: "YYYYMMDD" format

    Returns:
        dict with keys: foreign_buy, foreign_sell, foreign_net,
                        trust_buy, trust_sell, trust_net,
                        dealer_buy, dealer_sell, dealer_net,
                        total_net
    """
    data = _twse_get("/fund/T86", {"date": date_str, "selectType": "ALLBUT0999"})
    if not data or "data" not in data:
        return {}

    for row in data["data"]:
        # row[0] is stock code (may have leading/trailing spaces)
        code = row[0].strip()
        if code == stock_id:
            def parse_int(s):
                try:
                    return int(str(s).replace(",", "").strip())
                except (ValueError, TypeError):
                    return 0
            return {
                "stock_id": code,
                "stock_name": row[1].strip() if len(row) > 1 else "",
                "foreign_buy": parse_int(row[2]),
                "foreign_sell": parse_int(row[3]),
                "foreign_net": parse_int(row[4]),
                "trust_buy": parse_int(row[5]),
                "trust_sell": parse_int(row[6]),
                "trust_net": parse_int(row[7]),
                "dealer_net": parse_int(row[8]) if len(row) > 8 else 0,
                "total_net": parse_int(row[-1]) if len(row) > 10 else 0,
            }
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_institutional_trading_range(
    stock_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch institutional trading data for a date range.

    This makes one API call per trading day, so keep ranges small (≤ 30 days).
    Returns DataFrame with date index and institutional columns.
    """
    rows = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    progress_bar = st.progress(0, text="Fetching institutional data...")
    total_days = (end - current).days + 1
    day_count = 0

    while current <= end:
        day_count += 1
        progress_bar.progress(
            min(day_count / max(total_days, 1), 1.0),
            text=f"Fetching institutional data... {current.strftime('%Y-%m-%d')}"
        )
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y%m%d")
        result = fetch_institutional_trading(stock_id, date_str)
        if result:
            result["date"] = current.strftime("%Y-%m-%d")
            rows.append(result)
        current += timedelta(days=1)

    progress_bar.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ------------------------------------------------------------------
# 融資融券 (Margin Trading)
# ------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def fetch_margin_trading(stock_id: str, date_str: str) -> dict:
    """Fetch margin trading data for a specific stock on a given date.

    Returns dict with margin_buy, margin_sell, margin_balance,
    short_buy, short_sell, short_balance, etc.
    """
    data = _twse_get("/exchangeReport/MI_MARGN", {
        "date": date_str,
        "selectType": "ALL",
    })
    if not data:
        return {}

    rows = data.get("data", [])
    for row in rows:
        code = row[0].strip()
        if code == stock_id:
            def parse_int(s):
                try:
                    return int(str(s).replace(",", "").strip())
                except (ValueError, TypeError):
                    return 0
            return {
                "stock_id": code,
                "margin_buy": parse_int(row[2]) if len(row) > 2 else 0,
                "margin_sell": parse_int(row[3]) if len(row) > 3 else 0,
                "margin_cash_repay": parse_int(row[4]) if len(row) > 4 else 0,
                "margin_balance": parse_int(row[5]) if len(row) > 5 else 0,
                "short_sell": parse_int(row[8]) if len(row) > 8 else 0,
                "short_buy": parse_int(row[9]) if len(row) > 9 else 0,
                "short_balance": parse_int(row[11]) if len(row) > 11 else 0,
            }
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_margin_trading_range(
    stock_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch margin trading data for a date range."""
    rows = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    progress_bar = st.progress(0, text="Fetching margin data...")
    total_days = (end - current).days + 1
    day_count = 0

    while current <= end:
        day_count += 1
        progress_bar.progress(
            min(day_count / max(total_days, 1), 1.0),
            text=f"Fetching margin data... {current.strftime('%Y-%m-%d')}"
        )
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y%m%d")
        result = fetch_margin_trading(stock_id, date_str)
        if result:
            result["date"] = current.strftime("%Y-%m-%d")
            rows.append(result)
        current += timedelta(days=1)

    progress_bar.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ------------------------------------------------------------------
# OpenAPI endpoints (latest day only)
# ------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_twse_pe_pb_all() -> pd.DataFrame:
    """Fetch P/E, Dividend Yield, P/B for all TWSE stocks (latest day)."""
    data = _openapi_get(TWSE_OPENAPI_URL, "/exchangeReport/BWIBBU_ALL")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Columns: Code, Name, PEratio, DividendYield, PBratio
    rename_map = {
        "Code": "stock_id",
        "Name": "stock_name",
        "PEratio": "PE",
        "DividendYield": "Dividend_Yield",
        "PBratio": "PB",
    }
    df = df.rename(columns=rename_map)
    for col in ["PE", "Dividend_Yield", "PB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_twse_monthly_revenue() -> pd.DataFrame:
    """Fetch monthly revenue for all TWSE stocks (latest month)."""
    data = _openapi_get(TWSE_OPENAPI_URL, "/opendata/t187ap05_P")
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


# ------------------------------------------------------------------
# Helper: extract stock_id from ticker
# ------------------------------------------------------------------

def ticker_to_stock_id(ticker: str) -> str:
    """Convert '2330.TW' or '6510.TWO' to '2330' or '6510'."""
    return ticker.split(".")[0]
