"""Momentum screening: find strong/weak stocks over various periods."""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional
import io

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS


# Default stock universes
TW_TOP50 = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2382.TW",
    "2303.TW", "2412.TW", "2881.TW", "2882.TW", "2886.TW",
    "2891.TW", "2884.TW", "2885.TW", "3711.TW", "2357.TW",
    "1301.TW", "1303.TW", "1326.TW", "2002.TW", "1216.TW",
    "2912.TW", "5880.TW", "2892.TW", "3008.TW", "2207.TW",
    "6669.TW", "2603.TW", "5871.TW", "2880.TW", "3045.TW",
    "2801.TW", "4904.TW", "9910.TW", "2301.TW", "4938.TW",
    "2345.TW", "3034.TW", "2379.TW", "6505.TW", "1101.TW",
    "2395.TW", "8046.TW", "3231.TW", "2327.TW", "5876.TW",
    "3037.TW", "2883.TW", "2887.TW", "6415.TW", "2105.TW",
]

US_TOP50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "XOM", "PG", "MA",
    "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "AVGO", "TMO",
    "MCD", "WMT", "CSCO", "ACN", "ABT",
    "DHR", "CRM", "ADBE", "TXN", "NEE",
    "NFLX", "AMD", "INTC", "QCOM", "AMGN",
    "BA", "CAT", "GS", "MS", "LOW",
    "SBUX", "PYPL", "UBER", "SQ", "COIN",
]


def screen_momentum(
    tickers: List[str],
    period: str = "1mo",
    progress_callback=None,
) -> pd.DataFrame:
    """Screen stocks by momentum over a given period.

    Args:
        tickers: list of ticker symbols
        period: "1d", "5d", "1mo", "3mo", "6mo", "1y"
        progress_callback: fn(current, total, ticker)

    Returns:
        DataFrame sorted by return% descending with columns:
        Ticker, Name, Price, Return%, Volume_Change%, RSI, Relative_Strength, Rank
    """
    total = len(tickers)
    results = []

    # Download all at once for speed
    try:
        period_map = {
            "1d": 5, "5d": 10, "1mo": 35, "3mo": 100, "6mo": 190, "1y": 370,
        }
        days = period_map.get(period, 35)
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        all_data = yf.download(tickers, start=start_date, end=end_date,
                               progress=False, auto_adjust=True, threads=True)
    except Exception:
        all_data = pd.DataFrame()

    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx, total, ticker)

        try:
            row = _compute_momentum_single(ticker, all_data, period)
            if row:
                results.append(row)
        except Exception:
            continue

    if progress_callback:
        progress_callback(total, total, "Done")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Calculate percentile rank
    if "Return_Pct" in df.columns and len(df) > 1:
        df["Rank"] = df["Return_Pct"].rank(ascending=False, method="min").astype(int)
        df["Percentile"] = (df["Return_Pct"].rank(pct=True) * 100).round(0).astype(int)
        df = df.sort_values("Rank").reset_index(drop=True)

    # Classify strength
    if len(df) > 0:
        q75 = df["Return_Pct"].quantile(0.75)
        q25 = df["Return_Pct"].quantile(0.25)
        df["Strength"] = df["Return_Pct"].apply(
            lambda x: "Strong" if x >= q75 else "Weak" if x <= q25 else "Neutral"
        )

    return df


def _compute_momentum_single(ticker: str, all_data: pd.DataFrame, period: str) -> dict:
    """Compute momentum metrics for a single stock from bulk download."""
    try:
        if isinstance(all_data.columns, pd.MultiIndex):
            if ticker not in all_data["Close"].columns:
                return None
            close = all_data["Close"][ticker].dropna()
            volume = all_data["Volume"][ticker].dropna() if "Volume" in all_data.columns.get_level_values(0) else pd.Series()
        else:
            close = all_data["Close"].dropna()
            volume = all_data.get("Volume", pd.Series()).dropna()
    except Exception:
        return None

    if close.empty or len(close) < 2:
        return None

    # Period slicing
    period_days = {"1d": 1, "5d": 5, "1mo": 22, "3mo": 66, "6mo": 132, "1y": 252}
    n = period_days.get(period, 22)
    if len(close) <= n:
        n = len(close) - 1
    if n < 1:
        return None

    latest = close.iloc[-1]
    past = close.iloc[-(n + 1)] if len(close) > n else close.iloc[0]
    return_pct = (latest - past) / past * 100

    # Volume change
    vol_change = None
    if not volume.empty and len(volume) > n:
        recent_vol = volume.tail(min(5, n)).mean()
        prev_vol = volume.iloc[-(n + 1):-(n + 1) + 5].mean() if len(volume) > n + 5 else volume.iloc[:5].mean()
        if prev_vol > 0:
            vol_change = (recent_vol - prev_vol) / prev_vol * 100

    # Simple RSI (14)
    rsi = None
    if len(close) >= 15:
        delta = close.diff().tail(15)
        gain = delta.where(delta > 0, 0).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        if loss > 0:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        elif gain > 0:
            rsi = 100

    # Name from ticker
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName", ticker) if info else ticker
    except Exception:
        name = ticker

    # 52-week position
    high_52w = close.max()
    low_52w = close.min()
    pct_from_high = (latest - high_52w) / high_52w * 100 if high_52w > 0 else 0

    return {
        "Ticker": ticker,
        "Name": name[:20] if name else ticker,
        "Price": round(latest, 2),
        "Return_Pct": round(return_pct, 2),
        "Vol_Change_Pct": round(vol_change, 1) if vol_change is not None else None,
        "RSI": round(rsi, 1) if rsi is not None else None,
        "Pct_From_High": round(pct_from_high, 1),
    }


def export_momentum_excel(df: pd.DataFrame, period: str) -> bytes:
    """Export momentum screening to styled Excel."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=f"Momentum_{period}", index=False)
        wb = writer.book
        ws = writer.sheets[f"Momentum_{period}"]

        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#1a1a2e", "font_color": "#00d4aa",
            "border": 1, "text_wrap": True,
        })
        strong_fmt = wb.add_format({"bg_color": "#1b5e20", "font_color": "white"})
        weak_fmt = wb.add_format({"bg_color": "#b71c1c", "font_color": "white"})

        for col_num, value in enumerate(df.columns):
            ws.write(0, col_num, value, header_fmt)

        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max() if len(df) > 0 else 0, len(col))
            ws.set_column(i, i, min(max_len + 2, 25))

        if "Strength" in df.columns:
            s_col = df.columns.get_loc("Strength")
            for row_idx in range(len(df)):
                val = df.iloc[row_idx]["Strength"]
                if val == "Strong":
                    ws.write(row_idx + 1, s_col, val, strong_fmt)
                elif val == "Weak":
                    ws.write(row_idx + 1, s_col, val, weak_fmt)

        ws.freeze_panes(1, 2)
        ws.autofilter(0, 0, len(df), len(df.columns) - 1)

    return output.getvalue()
