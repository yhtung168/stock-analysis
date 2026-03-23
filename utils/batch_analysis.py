"""Batch analysis: score multiple stocks and export to Excel."""

import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from typing import List, Optional

import yfinance as yf

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_market, get_default_index, score_label, DEFAULT_WEIGHTS
from data.fetcher import get_fundamental_metrics
from indicators.technical import add_all_technical_indicators, get_latest_signals
from indicators.fundamental import analyze_fundamentals
from indicators.chip import calculate_quantitative_metrics
from scoring.composite import compute_all_scores, compute_composite, generate_highlights_risks


def batch_analyze(
    tickers: List[str],
    start: str,
    end: str,
    weights: dict = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Analyze multiple stocks and return a summary DataFrame.

    Args:
        tickers: list of ticker symbols
        start, end: date strings
        weights: scoring weights dict
        progress_callback: fn(current, total, ticker) for progress updates

    Returns:
        DataFrame with one row per stock, columns = all metrics + scores + signal
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    rows = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx, total, ticker)

        try:
            row = _analyze_single(ticker, start, end, weights)
            if row:
                rows.append(row)
        except Exception as e:
            rows.append({"Ticker": ticker, "Name": f"ERROR: {e}", "Signal": "N/A"})

    if progress_callback:
        progress_callback(total, total, "Done")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by composite score descending
    if "Composite_Score" in df.columns:
        df = df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)

    return df


def _analyze_single(ticker: str, start: str, end: str, weights: dict) -> dict:
    """Analyze a single stock and return a dict of all metrics."""
    # Fetch price
    price_df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if price_df.empty:
        return {"Ticker": ticker, "Name": "No data", "Signal": "N/A"}

    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)

    # Fetch info
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    fund_raw = get_fundamental_metrics(info)

    # Technical indicators
    df_ind = add_all_technical_indicators(price_df)
    tech_signals = get_latest_signals(df_ind)
    fund_analysis = analyze_fundamentals(fund_raw)

    # Quantitative
    bench_ticker = get_default_index(ticker)
    try:
        bench_df = yf.download(bench_ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = bench_df.columns.get_level_values(0)
    except Exception:
        bench_df = pd.DataFrame()
    quant_data = calculate_quantitative_metrics(price_df, bench_df)

    # Scoring
    all_scores = compute_all_scores(tech_signals, fund_analysis, {}, {}, quant_data)
    composite = compute_composite(all_scores, weights)
    label, _ = score_label(composite)
    highlights, risks = generate_highlights_risks(all_scores)

    # Price info
    latest_close = price_df["Close"].iloc[-1]
    prev_close = price_df["Close"].iloc[-2] if len(price_df) > 1 else latest_close
    change_pct = (latest_close - prev_close) / prev_close * 100

    # Build result row
    row = {
        "Ticker": ticker,
        "Name": fund_raw.get("Short_Name", ""),
        "Sector": fund_raw.get("Sector", ""),
        "Price": round(latest_close, 2),
        "Change%": round(change_pct, 2),
        # Composite
        "Composite_Score": round(composite, 2),
        "Signal": label,
        "Tech_Avg": round(all_scores["technical_avg"], 2) if all_scores["technical_avg"] else None,
        "Fund_Avg": round(all_scores["fundamental_avg"], 2) if all_scores["fundamental_avg"] else None,
        "Chip_Avg": round(all_scores["chip_avg"], 2) if all_scores["chip_avg"] else None,
    }

    # Individual technical scores
    for s in all_scores["technical"]:
        row[f"T_{s['name']}"] = s["score"]

    # Individual fundamental scores
    for s in all_scores["fundamental"]:
        row[f"F_{s['name']}"] = s["score"]

    # Raw fundamental values
    for key in ["PE", "PB", "EPS", "ROE", "ROA", "Dividend_Yield",
                "Operating_Margin", "Current_Ratio", "DE_Ratio"]:
        val = fund_raw.get(key)
        if val is not None:
            if key in ("ROE", "ROA", "Dividend_Yield", "Operating_Margin") and abs(val) < 1:
                val = val * 100
            if key == "DE_Ratio" and val > 10:
                val = val / 100
            row[f"Raw_{key}"] = round(val, 2) if isinstance(val, float) else val

    # Quantitative
    for key, data in quant_data.items():
        row[f"Q_{key}"] = round(data["value"], 2) if isinstance(data.get("value"), (int, float)) else None

    # Key technical values
    row["RSI_14"] = round(tech_signals.get("RSI", 0), 1) if tech_signals.get("RSI") else None
    row["MACD_Cross"] = tech_signals.get("MACD_Cross", "")
    row["KD_Cross"] = tech_signals.get("KD_Cross", "")

    # Highlights / risks summary
    row["Highlights"] = " | ".join(highlights[:3]) if highlights else ""
    row["Risks"] = " | ".join(risks[:3]) if risks else ""

    return row


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Export batch analysis DataFrame to styled Excel bytes."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Stock Analysis", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Stock Analysis"]

        # Formats
        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#1a1a2e", "font_color": "#00d4aa",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        })
        buy_fmt = workbook.add_format({"bg_color": "#1b5e20", "font_color": "white"})
        sell_fmt = workbook.add_format({"bg_color": "#b71c1c", "font_color": "white"})
        neutral_fmt = workbook.add_format({"bg_color": "#f57f17", "font_color": "white"})
        num_fmt = workbook.add_format({"num_format": "0.00"})
        pct_fmt = workbook.add_format({"num_format": "0.00%"})

        # Write headers
        for col_num, value in enumerate(df.columns):
            worksheet.write(0, col_num, value, header_fmt)

        # Column widths
        for i, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max() if len(df) > 0 else 0, len(col))
            worksheet.set_column(i, i, min(max_len + 2, 30))

        # Conditional format for Signal column
        if "Signal" in df.columns:
            sig_col = df.columns.get_loc("Signal")
            for row_idx in range(len(df)):
                sig = df.iloc[row_idx]["Signal"]
                fmt = neutral_fmt
                if "買進" in str(sig):
                    fmt = buy_fmt
                elif "賣出" in str(sig):
                    fmt = sell_fmt
                worksheet.write(row_idx + 1, sig_col, sig, fmt)

        # Conditional format for score column
        if "Composite_Score" in df.columns:
            score_col = df.columns.get_loc("Composite_Score")
            for row_idx in range(len(df)):
                score = df.iloc[row_idx]["Composite_Score"]
                if pd.notna(score):
                    fmt = buy_fmt if score >= 7 else sell_fmt if score <= 3.5 else neutral_fmt
                    worksheet.write(row_idx + 1, score_col, score, fmt)

        # Freeze top row
        worksheet.freeze_panes(1, 2)

        # Auto-filter
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    return output.getvalue()
