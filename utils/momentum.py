"""Momentum screening: find strong/weak stocks over various periods.

Features:
  - Dynamic stock lists by market cap rank (e.g., rank 1-50, 50-100)
  - Sector/industry classification for both TW and US
  - Sector-level momentum analysis (which sectors are hot?)
  - Stock-level momentum within a sector
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import io
import time
import requests

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS

# TWSE sector code to Chinese name mapping
TW_SECTOR_MAP = {
    "01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織",
    "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙",
    "10": "鋼鐵", "11": "橡膠", "12": "汽車", "13": "電子",
    "14": "建材營造", "15": "航運", "16": "觀光", "17": "金融保險",
    "18": "貿易百貨", "19": "綜合", "20": "其他", "21": "化學工業",
    "22": "生技醫療", "23": "油電燃氣", "24": "半導體", "25": "電腦及週邊設備",
    "26": "光電", "27": "通信網路", "28": "電子零組件", "29": "電子通路",
    "30": "資訊服務", "31": "其他電子", "32": "文化創意", "33": "農業科技",
    "34": "電子商務", "35": "綠能環保", "36": "數位雲端",
    "37": "運動休閒", "38": "居家生活",
    "39": "其他類", "91": "存託憑證(TDR)",
}


# ============================================================
# Static fallback lists (used if dynamic fetch fails)
# ============================================================
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


# ============================================================
# Taiwan stock universe fetcher (TWSE API)
# ============================================================
_tw_stock_cache: Optional[pd.DataFrame] = None
_tw_cache_time: Optional[datetime] = None


def fetch_tw_stock_universe(progress_callback=None) -> pd.DataFrame:
    """Fetch all TWSE-listed stocks with market cap and sector.

    Returns DataFrame with columns:
        Code, Name, Sector, Shares, Price, MarketCap, MarketCap_Rank
    """
    global _tw_stock_cache, _tw_cache_time

    # Cache for 1 hour
    if _tw_stock_cache is not None and _tw_cache_time:
        if (datetime.now() - _tw_cache_time).seconds < 3600:
            return _tw_stock_cache.copy()

    if progress_callback:
        progress_callback(0, 3, "Fetching TW stock list...")

    try:
        # Step 1: Get company info (shares outstanding + sector)
        url_info = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysisPlatform/1.0"}
        resp = requests.get(url_info, timeout=30, headers=headers)
        resp.raise_for_status()
        info_data = resp.json()
        df_info = pd.DataFrame(info_data)

        # Key columns: 公司代號, 公司簡稱, 產業別, 已發行普通股數或TDR原股發行股數
        df_info = df_info.rename(columns={
            "公司代號": "Code",
            "公司簡稱": "Name",
            "產業別": "Sector",
            "已發行普通股數或TDR原股發行股數": "Shares_Raw",
        })
        df_info = df_info[["Code", "Name", "Sector", "Shares_Raw"]].copy()
        df_info["Code"] = df_info["Code"].astype(str).str.strip()
        # Parse shares - remove commas
        df_info["Shares"] = pd.to_numeric(
            df_info["Shares_Raw"].astype(str).str.replace(",", ""), errors="coerce"
        )

        if progress_callback:
            progress_callback(1, 3, "Fetching TW prices...")

        # Step 2: Get today's closing prices
        time.sleep(0.5)
        url_price = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        resp2 = requests.get(url_price, timeout=30, headers=headers)
        resp2.raise_for_status()
        price_data = resp2.json()
        df_price = pd.DataFrame(price_data)

        df_price = df_price.rename(columns={
            "Code": "Code",
            "Name": "Name_P",
            "ClosingPrice": "Price_Raw",
        })
        df_price["Code"] = df_price["Code"].astype(str).str.strip()
        df_price["Price"] = pd.to_numeric(
            df_price["Price_Raw"].astype(str).str.replace(",", ""), errors="coerce"
        )

        if progress_callback:
            progress_callback(2, 3, "Computing market caps...")

        # Step 3: Merge and compute market cap
        merged = df_info.merge(df_price[["Code", "Price"]], on="Code", how="inner")
        merged = merged.dropna(subset=["Shares", "Price"])
        merged = merged[merged["Price"] > 0]
        merged["MarketCap"] = merged["Shares"] * merged["Price"]
        merged = merged.sort_values("MarketCap", ascending=False).reset_index(drop=True)
        merged["MarketCap_Rank"] = range(1, len(merged) + 1)
        merged["Ticker"] = merged["Code"].apply(lambda c: f"{c}.TW")

        # Map sector codes to names
        merged["Sector"] = merged["Sector"].astype(str).str.strip().map(TW_SECTOR_MAP).fillna(merged["Sector"])

        # Format market cap for display
        merged["MarketCap_B"] = (merged["MarketCap"] / 1e8).round(1)  # 億元

        _tw_stock_cache = merged
        _tw_cache_time = datetime.now()

        if progress_callback:
            progress_callback(3, 3, "Done")

        return merged.copy()

    except Exception as e:
        print(f"[WARN] Failed to fetch TW universe from TWSE API: {e}")
        print("[INFO] Using static TW Top 50 with hardcoded sector info")
        # Fallback: static list WITH sector classification
        return _tw_static_fallback()


def _tw_static_fallback() -> pd.DataFrame:
    """Static fallback with sector info for when TWSE API is unreachable (e.g., Streamlit Cloud overseas IP)."""
    _tw_static = [
        ("2330", "台積電", "半導體"), ("2317", "鴻海", "其他電子"), ("2454", "聯發科", "半導體"),
        ("2308", "台達電", "電子零組件"), ("2382", "廣達", "電腦及週邊設備"),
        ("2303", "聯電", "半導體"), ("2412", "中華電", "通信網路"),
        ("2881", "富邦金", "金融保險"), ("2882", "國泰金", "金融保險"), ("2886", "兆豐金", "金融保險"),
        ("2891", "中信金", "金融保險"), ("2884", "玉山金", "金融保險"), ("2885", "元大金", "金融保險"),
        ("3711", "日月光投控", "半導體"), ("2357", "華碩", "電腦及週邊設備"),
        ("1301", "台塑", "塑膠"), ("1303", "南亞", "塑膠"), ("1326", "台化", "塑膠"),
        ("2002", "中鋼", "鋼鐵"), ("1216", "統一", "食品"),
        ("2912", "統一超", "貿易百貨"), ("5880", "合庫金", "金融保險"),
        ("2892", "第一金", "金融保險"), ("3008", "大立光", "光電"), ("2207", "和泰車", "汽車"),
        ("6669", "緯穎", "電腦及週邊設備"), ("2603", "長榮", "航運"),
        ("5871", "中租-KY", "金融保險"), ("2880", "華南金", "金融保險"), ("3045", "台灣大", "通信網路"),
        ("2801", "彰銀", "金融保險"), ("4904", "遠傳", "通信網路"),
        ("9910", "豐泰", "其他電子"), ("2301", "光寶科", "電腦及週邊設備"), ("4938", "和碩", "電腦及週邊設備"),
        ("2345", "智邦", "通信網路"), ("3034", "聯詠", "半導體"),
        ("2379", "瑞昱", "半導體"), ("6505", "台塑化", "塑膠"), ("1101", "台泥", "水泥"),
        ("2395", "研華", "電腦及週邊設備"), ("8046", "南電", "電子零組件"),
        ("3231", "緯創", "電腦及週邊設備"), ("2327", "國巨", "電子零組件"),
        ("5876", "上海商銀", "金融保險"), ("3037", "欣興", "電子零組件"),
        ("2883", "凱基金", "金融保險"), ("2887", "台新金", "金融保險"),
        ("6415", "矽力-KY", "半導體"), ("2105", "正新", "橡膠"),
    ]
    df = pd.DataFrame(_tw_static, columns=["Code", "Name", "Sector"])
    df["Ticker"] = df["Code"] + ".TW"
    df["MarketCap_Rank"] = range(1, len(df) + 1)
    df["Shares"] = 0
    df["Price"] = 0
    df["MarketCap"] = 0
    df["MarketCap_B"] = 0
    return df


def get_tw_sectors() -> List[str]:
    """Get list of available TW sectors."""
    df = fetch_tw_stock_universe()
    if "Sector" in df.columns:
        return sorted(df["Sector"].dropna().unique().tolist())
    return []


def get_tw_stocks_by_rank(rank_start: int = 1, rank_end: int = 50) -> List[str]:
    """Get TW stock tickers by market cap rank range."""
    df = fetch_tw_stock_universe()
    filtered = df[(df["MarketCap_Rank"] >= rank_start) & (df["MarketCap_Rank"] <= rank_end)]
    return filtered["Ticker"].tolist()


def get_tw_stocks_by_sector(sectors: List[str]) -> List[str]:
    """Get TW stock tickers by sector names."""
    df = fetch_tw_stock_universe()
    filtered = df[df["Sector"].isin(sectors)]
    return filtered["Ticker"].tolist()


# ============================================================
# US stock universe fetcher (Wikipedia S&P 500)
# ============================================================
_us_stock_cache: Optional[pd.DataFrame] = None
_us_cache_time: Optional[datetime] = None


def fetch_us_stock_universe(progress_callback=None) -> pd.DataFrame:
    """Fetch S&P 500 components with sector from Wikipedia + market cap from yfinance.

    Returns DataFrame with columns:
        Ticker, Name, Sector, SubIndustry, MarketCap, MarketCap_Rank
    """
    global _us_stock_cache, _us_cache_time

    if _us_stock_cache is not None and _us_cache_time:
        if (datetime.now() - _us_cache_time).seconds < 3600:
            return _us_stock_cache.copy()

    if progress_callback:
        progress_callback(0, 2, "Fetching S&P 500 list...")

    try:
        # Step 1: Get S&P 500 list from Wikipedia (with User-Agent to avoid 403)
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) StockAnalysisPlatform/1.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df_sp = tables[0]

        df_sp = df_sp.rename(columns={
            "Symbol": "Ticker",
            "Security": "Name",
            "GICS Sector": "Sector",
            "GICS Sub-Industry": "SubIndustry",
        })
        df_sp = df_sp[["Ticker", "Name", "Sector", "SubIndustry"]].copy()
        # Fix tickers (Wikipedia uses BRK.B, yfinance uses BRK-B)
        df_sp["Ticker"] = df_sp["Ticker"].str.replace(".", "-", regex=False)

        if progress_callback:
            progress_callback(1, 2, "Sorting by market cap...")

        # Step 2: Use a known approximate market cap ranking
        # (fetching 500 market caps from yfinance is too slow ~10min)
        # Instead, we use Wikipedia's table order which is roughly by market cap
        # and supplement with a quick batch download of latest prices
        # The S&P 500 Wikipedia table lists companies alphabetically, so we use
        # a known top-by-market-cap ordering for the top ~100
        _top_by_mcap = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
            "AVGO", "TSLA", "WMT", "JPM", "V", "UNH", "XOM", "MA", "ORCL",
            "COST", "PG", "JNJ", "HD", "NFLX", "ABBV", "BAC", "CRM", "KO",
            "CVX", "MRK", "AMD", "SAP", "PEP", "TMO", "ADBE", "CSCO", "ACN",
            "LIN", "WFC", "MCD", "ABT", "TXN", "IBM", "PM", "GE", "DHR",
            "ISRG", "QCOM", "AMGN", "INTU", "CAT", "VZ", "AMAT", "NOW",
            "GS", "BKNG", "T", "SPGI", "AXP", "BA", "NEE", "MS", "LOW",
            "PFE", "RTX", "BLK", "TJX", "UNP", "DE", "UBER", "SYK", "SCHW",
        ]
        # Assign rank: known top companies first, rest alphabetically after
        rank_map = {t: i + 1 for i, t in enumerate(_top_by_mcap)}
        next_rank = len(_top_by_mcap) + 1
        for t in df_sp["Ticker"]:
            if t not in rank_map:
                rank_map[t] = next_rank
                next_rank += 1

        df_sp["MarketCap_Rank"] = df_sp["Ticker"].map(rank_map)
        df_sp = df_sp.sort_values("MarketCap_Rank").reset_index(drop=True)
        df_sp["MarketCap_Rank"] = range(1, len(df_sp) + 1)
        df_sp["MarketCap"] = 0  # placeholder
        df_sp["MarketCap_B"] = 0

        _us_stock_cache = df_sp
        _us_cache_time = datetime.now()

        if progress_callback:
            progress_callback(2, 2, "Done")

        return df_sp.copy()

    except Exception as e:
        print(f"[WARN] Failed to fetch US universe: {e}")
        # Fallback
        return pd.DataFrame({
            "Ticker": US_TOP50,
            "Name": US_TOP50,
            "Sector": "Unknown",
            "MarketCap_Rank": range(1, len(US_TOP50) + 1),
        })


def get_us_sectors() -> List[str]:
    """Get list of available US GICS sectors."""
    df = fetch_us_stock_universe()
    if "Sector" in df.columns:
        return sorted(df["Sector"].dropna().unique().tolist())
    return []


def get_us_stocks_by_rank(rank_start: int = 1, rank_end: int = 50) -> List[str]:
    """Get US stock tickers by market cap rank range."""
    df = fetch_us_stock_universe()
    filtered = df[(df["MarketCap_Rank"] >= rank_start) & (df["MarketCap_Rank"] <= rank_end)]
    return filtered["Ticker"].tolist()


def get_us_stocks_by_sector(sectors: List[str]) -> List[str]:
    """Get US stock tickers by sector names."""
    df = fetch_us_stock_universe()
    filtered = df[df["Sector"].isin(sectors)]
    return filtered["Ticker"].tolist()


# ============================================================
# Sector momentum analysis
# ============================================================
def analyze_sector_momentum(
    market: str = "TW",
    period: str = "1mo",
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze momentum at sector level, then stock level within each sector.

    Returns:
        sector_df: DataFrame with sector-level momentum (Sector, Avg_Return%, Stocks, Top_Stock)
        stock_df: DataFrame with all stocks including Sector column
    """
    # Get universe
    if market == "TW":
        universe = fetch_tw_stock_universe()
        ticker_col = "Ticker"
    else:
        universe = fetch_us_stock_universe()
        ticker_col = "Ticker"

    if universe.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_tickers = universe[ticker_col].tolist()

    # Screen momentum for all stocks
    stock_df = screen_momentum(all_tickers, period=period, progress_callback=progress_callback)

    if stock_df.empty:
        return pd.DataFrame(), stock_df

    # Merge sector info
    if market == "TW":
        sector_map = dict(zip(universe["Ticker"], universe["Sector"]))
    else:
        sector_map = dict(zip(universe["Ticker"], universe["Sector"]))

    stock_df["Sector"] = stock_df["Ticker"].map(sector_map).fillna("Other")

    # Sector-level aggregation
    sector_agg = stock_df.groupby("Sector").agg(
        Avg_Return=("Return_Pct", "mean"),
        Median_Return=("Return_Pct", "median"),
        Stock_Count=("Ticker", "count"),
        Strong_Count=("Strength", lambda x: (x == "Strong").sum()),
        Weak_Count=("Strength", lambda x: (x == "Weak").sum()),
        Best_Return=("Return_Pct", "max"),
        Worst_Return=("Return_Pct", "min"),
    ).reset_index()

    # Find top stock per sector
    top_per_sector = stock_df.loc[stock_df.groupby("Sector")["Return_Pct"].idxmax()]
    top_map = dict(zip(top_per_sector["Sector"],
                       top_per_sector["Ticker"] + " (" + top_per_sector["Return_Pct"].round(1).astype(str) + "%)"))
    sector_agg["Top_Stock"] = sector_agg["Sector"].map(top_map)

    # Round and sort
    sector_agg["Avg_Return"] = sector_agg["Avg_Return"].round(2)
    sector_agg["Median_Return"] = sector_agg["Median_Return"].round(2)
    sector_agg["Best_Return"] = sector_agg["Best_Return"].round(2)
    sector_agg["Worst_Return"] = sector_agg["Worst_Return"].round(2)
    sector_agg = sector_agg.sort_values("Avg_Return", ascending=False).reset_index(drop=True)

    # Sector strength
    q75 = sector_agg["Avg_Return"].quantile(0.75) if len(sector_agg) > 3 else 0
    q25 = sector_agg["Avg_Return"].quantile(0.25) if len(sector_agg) > 3 else 0
    sector_agg["Strength"] = sector_agg["Avg_Return"].apply(
        lambda x: "Strong" if x >= q75 else "Weak" if x <= q25 else "Neutral"
    )
    sector_agg["Rank"] = range(1, len(sector_agg) + 1)

    return sector_agg, stock_df


# ============================================================
# Core momentum screening (unchanged interface, added Sector)
# ============================================================
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
        DataFrame sorted by return% descending
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

    # Name
    name = ticker
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName", ticker) if info else ticker
    except Exception:
        pass

    # Position from period high
    high_period = close.max()
    pct_from_high = (latest - high_period) / high_period * 100 if high_period > 0 else 0

    return {
        "Ticker": ticker,
        "Name": name[:25] if name else ticker,
        "Price": round(latest, 2),
        "Return_Pct": round(return_pct, 2),
        "Vol_Change_Pct": round(vol_change, 1) if vol_change is not None else None,
        "RSI": round(rsi, 1) if rsi is not None else None,
        "Pct_From_High": round(pct_from_high, 1),
    }


# ============================================================
# Excel export
# ============================================================
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


def export_sector_momentum_excel(sector_df: pd.DataFrame, stock_df: pd.DataFrame, period: str) -> bytes:
    """Export sector + stock momentum to multi-sheet Excel."""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        sector_df.to_excel(writer, sheet_name="Sector_Momentum", index=False)
        stock_df.to_excel(writer, sheet_name="Stock_Detail", index=False)

        wb = writer.book
        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#1a1a2e", "font_color": "#00d4aa",
            "border": 1, "text_wrap": True,
        })
        strong_fmt = wb.add_format({"bg_color": "#1b5e20", "font_color": "white"})
        weak_fmt = wb.add_format({"bg_color": "#b71c1c", "font_color": "white"})

        for sheet_name in ["Sector_Momentum", "Stock_Detail"]:
            ws = writer.sheets[sheet_name]
            df = sector_df if sheet_name == "Sector_Momentum" else stock_df
            for col_num, value in enumerate(df.columns):
                ws.write(0, col_num, value, header_fmt)
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max() if len(df) > 0 else 0, len(col))
                ws.set_column(i, i, min(max_len + 2, 30))
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
