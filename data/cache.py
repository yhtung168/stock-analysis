"""Simple caching utilities for data fetching."""

import streamlit as st
import pandas as pd
from typing import Optional
from . import fetcher, twse_api
from config import get_market


class DataStore:
    """Central data store that caches fetched data in session state."""

    @staticmethod
    def get_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
        """Get price data with caching."""
        key = f"price_{ticker}_{start}_{end}"
        if key not in st.session_state:
            st.session_state[key] = fetcher.fetch_price_data(ticker, start, end)
        return st.session_state[key]

    @staticmethod
    def get_stock_info(ticker: str) -> dict:
        """Get stock info with caching."""
        key = f"info_{ticker}"
        if key not in st.session_state:
            st.session_state[key] = fetcher.fetch_stock_info(ticker)
        return st.session_state[key]

    @staticmethod
    def get_fundamental_metrics(ticker: str) -> dict:
        """Get fundamental metrics."""
        info = DataStore.get_stock_info(ticker)
        return fetcher.get_fundamental_metrics(info)

    @staticmethod
    def get_financials(ticker: str) -> dict:
        """Get financial statements."""
        key = f"financials_{ticker}"
        if key not in st.session_state:
            st.session_state[key] = fetcher.fetch_financials(ticker)
        return st.session_state[key]

    @staticmethod
    def get_dividends(ticker: str) -> pd.DataFrame:
        """Get dividend history."""
        key = f"dividends_{ticker}"
        if key not in st.session_state:
            st.session_state[key] = fetcher.fetch_dividends(ticker)
        return st.session_state[key]

    @staticmethod
    def _ensure_chip_data(ticker: str, start: str, end: str):
        """Fetch both institutional + margin in a single pass (TW only)."""
        market = get_market(ticker)
        if market not in ("TW", "TWO"):
            return
        stock_id = twse_api.ticker_to_stock_id(ticker)
        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "")
        key = f"chip_{stock_id}_{start_fmt}_{end_fmt}"
        if key not in st.session_state:
            st.session_state[key] = twse_api.fetch_chip_data_combined(
                stock_id, start_fmt, end_fmt
            )

    @staticmethod
    def get_institutional_data(
        ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Get institutional trading data (TW only)."""
        market = get_market(ticker)
        if market not in ("TW", "TWO"):
            return pd.DataFrame()
        DataStore._ensure_chip_data(ticker, start, end)
        stock_id = twse_api.ticker_to_stock_id(ticker)
        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "")
        key = f"chip_{stock_id}_{start_fmt}_{end_fmt}"
        chip = st.session_state.get(key, {})
        return chip.get("institutional", pd.DataFrame())

    @staticmethod
    def get_margin_data(
        ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Get margin trading data (TW only)."""
        market = get_market(ticker)
        if market not in ("TW", "TWO"):
            return pd.DataFrame()
        DataStore._ensure_chip_data(ticker, start, end)
        stock_id = twse_api.ticker_to_stock_id(ticker)
        start_fmt = start.replace("-", "")
        end_fmt = end.replace("-", "")
        key = f"chip_{stock_id}_{start_fmt}_{end_fmt}"
        chip = st.session_state.get(key, {})
        return chip.get("margin", pd.DataFrame())

    @staticmethod
    def get_multiple_prices(
        tickers: list, start: str, end: str
    ) -> pd.DataFrame:
        """Get close prices for multiple tickers."""
        key = f"multi_{'_'.join(sorted(tickers))}_{start}_{end}"
        if key not in st.session_state:
            st.session_state[key] = fetcher.fetch_multiple_prices(
                tickers, start, end
            )
        return st.session_state[key]

    @staticmethod
    def clear_all():
        """Clear all cached data from session state."""
        keys_to_remove = [
            k for k in st.session_state
            if k.startswith(("price_", "info_", "financials_", "dividends_",
                            "inst_", "margin_", "multi_"))
        ]
        for k in keys_to_remove:
            del st.session_state[k]
