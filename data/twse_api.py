"""TWSE / TPEx direct API wrapper for Taiwan stock-specific data.

Provides: 三大法人買賣超, 融資融券, 月營收, P/E/P/B/殖利率
All endpoints are free and public. Rate limit: ~3 req / 5 seconds.

OPTIMIZATION: T86 and MI_MARGN return ALL stocks per call. We cache the
full response per date, so multiple stocks on the same date don't trigger
extra API calls. We also limit range fetches to ~20 trading days max and
reduce sleep to 2 seconds.
"""

import requests
import pandas as pd
import time
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TWSE_BASE_URL, TWSE_OPENAPI_URL, TPEX_OPENAPI_URL

# Reduced delay — 2s is safe in practice; TWSE bans at ~5+ req/5s
_REQUEST_DELAY = 2.0
_last_request_time = 0.0

# In-memory cache for full-day API responses (avoids re-fetching same date)
_inst_day_cache: Dict[str, list] = {}  # date_str -> raw data rows
_margin_day_cache: Dict[str, list] = {}  # date_str -> raw data rows


def _rate_limit():
    """Ensure minimum delay between TWSE requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _REQUEST_DELAY:
        time.sleep(_REQUEST_DELAY - elapsed)
    _last_request_time = time.time()


def _twse_get(path: str, params: dict = None) -> dict:
    """Make a GET request to TWSE legacy API with rate limiting."""
    _rate_limit()
    url = f"{TWSE_BASE_URL}{path}"
    if params is None:
        params = {}
    params["response"] = "json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysis/1.0",
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


def _parse_int(s) -> int:
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0


# ------------------------------------------------------------------
# Trading dates helper
# ------------------------------------------------------------------

def _get_trading_dates(start_date: str, end_date: str, max_days: int = 20) -> List[str]:
    """Generate trading dates (weekdays only), limited to most recent max_days.

    Args:
        start_date, end_date: YYYYMMDD strings
        max_days: max number of trading days to fetch (from end backwards)

    Returns:
        List of YYYYMMDD strings, most recent first
    """
    end = datetime.strptime(end_date, "%Y%m%d")
    start = datetime.strptime(start_date, "%Y%m%d")

    dates = []
    current = end
    while current >= start and len(dates) < max_days:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current.strftime("%Y%m%d"))
        current -= timedelta(days=1)

    return dates  # most recent first


# ------------------------------------------------------------------
# 三大法人買賣超 (Institutional Investors Buy/Sell)
# ------------------------------------------------------------------

def _fetch_inst_day_all(date_str: str) -> list:
    """Fetch T86 data for ALL stocks on a given date. Cached per date."""
    if date_str in _inst_day_cache:
        return _inst_day_cache[date_str]

    data = _twse_get("/fund/T86", {"date": date_str, "selectType": "ALLBUT0999"})
    rows = data.get("data", []) if data else []
    _inst_day_cache[date_str] = rows
    return rows


def _extract_inst_stock(rows: list, stock_id: str) -> dict:
    """Extract a single stock from T86 all-stock response."""
    for row in rows:
        code = row[0].strip()
        if code == stock_id:
            return {
                "stock_id": code,
                "stock_name": row[1].strip() if len(row) > 1 else "",
                "foreign_buy": _parse_int(row[2]),
                "foreign_sell": _parse_int(row[3]),
                "foreign_net": _parse_int(row[4]),
                "trust_buy": _parse_int(row[5]),
                "trust_sell": _parse_int(row[6]),
                "trust_net": _parse_int(row[7]),
                "dealer_net": _parse_int(row[8]) if len(row) > 8 else 0,
                "total_net": _parse_int(row[-1]) if len(row) > 10 else 0,
            }
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_institutional_trading(stock_id: str, date_str: str) -> dict:
    """Fetch institutional buy/sell for a specific stock on a given date."""
    rows = _fetch_inst_day_all(date_str)
    return _extract_inst_stock(rows, stock_id)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_institutional_trading_range(
    stock_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch institutional trading data for a date range.

    OPTIMIZED: fetches only last 20 trading days, caches full-day responses.
    ~20 API calls × 2s = ~40 seconds (vs old: 65 calls × 3.5s = 228s).
    """
    dates = _get_trading_dates(start_date, end_date, max_days=20)
    if not dates:
        return pd.DataFrame()

    rows = []
    total = len(dates)
    progress_bar = st.progress(0, text="Fetching institutional data...")

    for i, date_str in enumerate(dates):
        progress_bar.progress(
            min((i + 1) / total, 1.0),
            text=f"Fetching institutional data... {date_str[:4]}-{date_str[4:6]}-{date_str[6:]} ({i+1}/{total})"
        )
        result = fetch_institutional_trading(stock_id, date_str)
        if result:
            result["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            rows.append(result)

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

def _fetch_margin_day_all(date_str: str) -> list:
    """Fetch MI_MARGN data for ALL stocks on a given date. Cached per date."""
    if date_str in _margin_day_cache:
        return _margin_day_cache[date_str]

    data = _twse_get("/exchangeReport/MI_MARGN", {
        "date": date_str,
        "selectType": "ALL",
    })
    rows = data.get("data", []) if data else []
    _margin_day_cache[date_str] = rows
    return rows


def _extract_margin_stock(rows: list, stock_id: str) -> dict:
    """Extract a single stock from MI_MARGN all-stock response."""
    for row in rows:
        code = row[0].strip()
        if code == stock_id:
            return {
                "stock_id": code,
                "margin_buy": _parse_int(row[2]) if len(row) > 2 else 0,
                "margin_sell": _parse_int(row[3]) if len(row) > 3 else 0,
                "margin_cash_repay": _parse_int(row[4]) if len(row) > 4 else 0,
                "margin_balance": _parse_int(row[5]) if len(row) > 5 else 0,
                "short_sell": _parse_int(row[8]) if len(row) > 8 else 0,
                "short_buy": _parse_int(row[9]) if len(row) > 9 else 0,
                "short_balance": _parse_int(row[11]) if len(row) > 11 else 0,
            }
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_margin_trading(stock_id: str, date_str: str) -> dict:
    """Fetch margin trading data for a specific stock on a given date."""
    rows = _fetch_margin_day_all(date_str)
    return _extract_margin_stock(rows, stock_id)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_margin_trading_range(
    stock_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch margin trading data for a date range.

    OPTIMIZED: fetches only last 20 trading days, reuses institutional
    day-cache dates where possible.
    """
    dates = _get_trading_dates(start_date, end_date, max_days=20)
    if not dates:
        return pd.DataFrame()

    rows = []
    total = len(dates)
    progress_bar = st.progress(0, text="Fetching margin data...")

    for i, date_str in enumerate(dates):
        progress_bar.progress(
            min((i + 1) / total, 1.0),
            text=f"Fetching margin data... {date_str[:4]}-{date_str[4:6]}-{date_str[6:]} ({i+1}/{total})"
        )
        result = fetch_margin_trading(stock_id, date_str)
        if result:
            result["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            rows.append(result)

    progress_bar.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ------------------------------------------------------------------
# Batch fetch: institutional + margin in ONE pass over dates
# ------------------------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_chip_data_combined(
    stock_id: str, start_date: str, end_date: str
) -> Dict[str, pd.DataFrame]:
    """Fetch BOTH institutional + margin in a single date loop.

    This halves the API calls because T86 and MI_MARGN share the same
    date iteration. Each date requires 2 calls (T86 + MI_MARGN) with
    2s delay each = 4s/date × 20 dates = ~80s total.

    But if the date is already cached (from a prior stock on same session),
    it's instant — the full-day response is reused.

    Returns:
        {"institutional": DataFrame, "margin": DataFrame}
    """
    dates = _get_trading_dates(start_date, end_date, max_days=20)
    if not dates:
        return {"institutional": pd.DataFrame(), "margin": pd.DataFrame()}

    inst_rows = []
    margin_rows = []
    total = len(dates)
    progress_bar = st.progress(0, text="Fetching chip data...")

    for i, date_str in enumerate(dates):
        date_display = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        progress_bar.progress(
            min((i + 1) / total, 1.0),
            text=f"Fetching chip data... {date_display} ({i+1}/{total})"
        )

        # Institutional (T86) — uses day cache
        inst = fetch_institutional_trading(stock_id, date_str)
        if inst:
            inst["date"] = date_display
            inst_rows.append(inst)

        # Margin (MI_MARGN) — uses day cache
        margin = fetch_margin_trading(stock_id, date_str)
        if margin:
            margin["date"] = date_display
            margin_rows.append(margin)

    progress_bar.empty()

    def to_df(rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    return {
        "institutional": to_df(inst_rows),
        "margin": to_df(margin_rows),
    }


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
