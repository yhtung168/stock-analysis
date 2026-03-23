"""Market Sentiment Analysis - Fear & Greed Gauge.

Combines multiple indicators to produce a composite fear/greed score (0-100).
0 = Extreme Fear, 50 = Neutral, 100 = Extreme Greed.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import io
import time


# ============================================================
# Individual Sentiment Indicators
# ============================================================

def get_vix_sentiment() -> Dict:
    """VIX Fear Index. Low VIX = greed, High VIX = fear."""
    try:
        vix = yf.download("^VIX", period="3mo", progress=False, auto_adjust=True)
        if vix.empty:
            return {"name": "VIX", "value": None, "score": None, "status": "N/A"}

        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        current = float(vix["Close"].iloc[-1])
        avg_50d = float(vix["Close"].tail(50).mean()) if len(vix) >= 50 else current

        # Score: VIX < 12 = 95 (extreme greed), > 40 = 5 (extreme fear)
        if current <= 12:
            score = 95
        elif current <= 15:
            score = 80
        elif current <= 20:
            score = 60
        elif current <= 25:
            score = 40
        elif current <= 30:
            score = 25
        elif current <= 40:
            score = 10
        else:
            score = 5

        # Adjust based on trend (rising VIX = more fear)
        if current > avg_50d * 1.1:
            score = max(score - 10, 0)
        elif current < avg_50d * 0.9:
            score = min(score + 10, 100)

        trend = "Rising ↑" if current > avg_50d else "Falling ↓" if current < avg_50d else "Stable →"

        return {
            "name": "VIX 恐慌指數",
            "value": round(current, 2),
            "score": score,
            "detail": f"VIX={current:.1f}, 50日均={avg_50d:.1f}, {trend}",
            "interpretation": _vix_interpret(current),
            "category": "volatility",
        }
    except Exception as e:
        return {"name": "VIX 恐慌指數", "value": None, "score": None, "detail": str(e), "category": "volatility"}


def _vix_interpret(vix):
    if vix < 12: return "極度樂觀 (Extreme Complacency)"
    if vix < 15: return "低波動 (Low Volatility)"
    if vix < 20: return "正常 (Normal)"
    if vix < 25: return "不安 (Elevated)"
    if vix < 30: return "恐慌 (Fear)"
    if vix < 40: return "高度恐慌 (High Fear)"
    return "極度恐慌 (Extreme Fear) — 歷史上常為底部"


def get_sp500_momentum() -> Dict:
    """S&P 500 vs 125-day Moving Average."""
    try:
        spy = yf.download("^GSPC", period="1y", progress=False, auto_adjust=True)
        if spy.empty:
            return {"name": "S&P 500 Momentum", "value": None, "score": None}

        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        current = float(spy["Close"].iloc[-1])
        ma125 = float(spy["Close"].tail(125).mean()) if len(spy) >= 125 else current

        pct_above = (current - ma125) / ma125 * 100

        # Score: > +10% above MA = extreme greed, < -10% = extreme fear
        score = np.clip(50 + pct_above * 5, 0, 100)

        return {
            "name": "S&P 500 Momentum",
            "value": round(current, 2),
            "score": round(score),
            "detail": f"S&P={current:.0f}, 125MA={ma125:.0f}, {pct_above:+.1f}% above MA",
            "interpretation": "Above 125-day MA = Bullish" if pct_above > 0 else "Below 125-day MA = Bearish",
            "category": "momentum",
        }
    except Exception as e:
        return {"name": "S&P 500 Momentum", "value": None, "score": None, "detail": str(e), "category": "momentum"}


def get_put_call_ratio() -> Dict:
    """CBOE Put/Call Ratio (equity). High = fear, Low = greed."""
    try:
        url = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
        df = pd.read_csv(url, skiprows=2)
        df.columns = [c.strip() for c in df.columns]

        if "P/C Ratio" in df.columns:
            ratio_col = "P/C Ratio"
        elif "RATIO" in df.columns:
            ratio_col = "RATIO"
        else:
            # Try last column
            ratio_col = df.columns[-1]

        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df = df.dropna(subset=[ratio_col])

        if df.empty:
            return {"name": "Put/Call Ratio", "value": None, "score": None, "category": "options"}

        current = float(df[ratio_col].iloc[-1])
        avg_5d = float(df[ratio_col].tail(5).mean())

        # Score: P/C < 0.5 = extreme greed (100), > 1.2 = extreme fear (0)
        score = np.clip(100 - (avg_5d - 0.5) / 0.7 * 100, 0, 100)

        return {
            "name": "Put/Call Ratio",
            "value": round(avg_5d, 3),
            "score": round(score),
            "detail": f"5日均 P/C={avg_5d:.3f}, 最新={current:.3f}",
            "interpretation": "P/C > 1.0 = 偏恐懼 (more puts)" if avg_5d > 1.0 else "P/C < 0.7 = 偏貪婪 (more calls)" if avg_5d < 0.7 else "中性",
            "category": "options",
        }
    except Exception as e:
        return {"name": "Put/Call Ratio", "value": None, "score": None, "detail": str(e), "category": "options"}


def get_safe_haven_demand() -> Dict:
    """Stock vs Bond returns (20-day). Stocks outperform = greed, bonds outperform = fear."""
    try:
        data = yf.download(["^GSPC", "^TYX"], period="3mo", progress=False, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            sp_close = data["Close"]["^GSPC"].dropna()
            bond_close = data["Close"]["^TYX"].dropna()
        else:
            return {"name": "Safe Haven Demand", "value": None, "score": None, "category": "safe_haven"}

        if len(sp_close) < 20 or len(bond_close) < 20:
            return {"name": "Safe Haven Demand", "value": None, "score": None, "category": "safe_haven"}

        stock_ret = (sp_close.iloc[-1] / sp_close.iloc[-20] - 1) * 100
        bond_ret = (bond_close.iloc[-1] / bond_close.iloc[-20] - 1) * 100

        # When stocks outperform bonds = greed; when bonds outperform = fear
        diff = stock_ret - bond_ret

        score = np.clip(50 + diff * 5, 0, 100)

        return {
            "name": "Safe Haven Demand 避險需求",
            "value": round(diff, 2),
            "score": round(score),
            "detail": f"S&P 20日報酬={stock_ret:.1f}%, 30Y Bond yield change={bond_ret:.1f}%",
            "interpretation": "資金流入股市 (Risk-On)" if diff > 0 else "資金流入債市避險 (Risk-Off)",
            "category": "safe_haven",
        }
    except Exception as e:
        return {"name": "Safe Haven Demand 避險需求", "value": None, "score": None, "detail": str(e), "category": "safe_haven"}


def get_market_breadth() -> Dict:
    """Approximate market breadth using SPY sector ETFs performance."""
    try:
        sector_etfs = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
        data = yf.download(sector_etfs, period="1mo", progress=False, auto_adjust=True)

        if data.empty:
            return {"name": "Market Breadth", "value": None, "score": None, "category": "breadth"}

        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]

        # Count sectors with positive monthly returns
        returns = (close.iloc[-1] / close.iloc[0] - 1) * 100
        advancing = (returns > 0).sum()
        total = len(returns.dropna())

        if total == 0:
            return {"name": "Market Breadth", "value": None, "score": None, "category": "breadth"}

        breadth_pct = advancing / total * 100
        score = round(breadth_pct)

        return {
            "name": "Market Breadth 市場廣度",
            "value": round(breadth_pct, 1),
            "score": score,
            "detail": f"{advancing}/{total} sectors positive ({breadth_pct:.0f}%)",
            "interpretation": f"{'廣泛上漲 (Broad Rally)' if breadth_pct > 70 else '少數領漲 (Narrow)' if breadth_pct < 40 else '分歧 (Mixed)'}",
            "category": "breadth",
        }
    except Exception as e:
        return {"name": "Market Breadth 市場廣度", "value": None, "score": None, "detail": str(e), "category": "breadth"}


def get_yield_curve() -> Dict:
    """10Y-2Y Treasury yield spread. Inverted = fear/recession signal."""
    try:
        # Use yfinance for 10Y and 5Y (2Y not available on yfinance)
        data = yf.download(["^TNX", "^FVX"], period="6mo", progress=False, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            tnx = data["Close"]["^TNX"].dropna()
            fvx = data["Close"]["^FVX"].dropna()
        else:
            return {"name": "Yield Curve", "value": None, "score": None, "category": "bonds"}

        if tnx.empty or fvx.empty:
            return {"name": "Yield Curve", "value": None, "score": None, "category": "bonds"}

        spread_10y5y = float(tnx.iloc[-1]) - float(fvx.iloc[-1])

        # Positive spread = normal = greed; negative = inverted = fear
        score = np.clip(50 + spread_10y5y * 50, 0, 100)

        return {
            "name": "Yield Curve 殖利率曲線",
            "value": round(spread_10y5y, 3),
            "score": round(score),
            "detail": f"10Y={float(tnx.iloc[-1]):.2f}%, 5Y={float(fvx.iloc[-1]):.2f}%, Spread={spread_10y5y:.3f}%",
            "interpretation": "正常 (Normal)" if spread_10y5y > 0.2 else "持平 (Flat)" if spread_10y5y > -0.1 else "倒掛 (Inverted) — 衰退訊號!",
            "category": "bonds",
        }
    except Exception as e:
        return {"name": "Yield Curve 殖利率曲線", "value": None, "score": None, "detail": str(e), "category": "bonds"}


def get_gold_signal() -> Dict:
    """Gold price trend. Rising gold = fear/safe haven demand."""
    try:
        gold = yf.download("GC=F", period="3mo", progress=False, auto_adjust=True)
        if gold.empty:
            return {"name": "Gold", "value": None, "score": None, "category": "safe_haven"}

        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)

        current = float(gold["Close"].iloc[-1])
        ma50 = float(gold["Close"].tail(50).mean()) if len(gold) >= 50 else current
        ret_1m = (current / float(gold["Close"].iloc[-22]) - 1) * 100 if len(gold) >= 22 else 0

        # Rising gold = fear (inverse relationship)
        score = np.clip(50 - ret_1m * 3, 0, 100)

        return {
            "name": "Gold 黃金 (避險指標)",
            "value": round(current, 2),
            "score": round(score),
            "detail": f"Gold=${current:.0f}, 月漲跌={ret_1m:+.1f}%, 50MA=${ma50:.0f}",
            "interpretation": "黃金上漲=避險需求增加 (Fear)" if ret_1m > 3 else "黃金下跌=風險偏好上升 (Greed)" if ret_1m < -3 else "穩定",
            "category": "safe_haven",
        }
    except Exception as e:
        return {"name": "Gold 黃金", "value": None, "score": None, "detail": str(e), "category": "safe_haven"}


def get_usd_strength() -> Dict:
    """USD Index (DXY). Strong USD = risk-off/fear."""
    try:
        dxy = yf.download("DX-Y.NYB", period="3mo", progress=False, auto_adjust=True)
        if dxy.empty:
            return {"name": "USD (DXY)", "value": None, "score": None, "category": "currency"}

        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = dxy.columns.get_level_values(0)

        current = float(dxy["Close"].iloc[-1])
        ma50 = float(dxy["Close"].tail(50).mean()) if len(dxy) >= 50 else current
        ret_1m = (current / float(dxy["Close"].iloc[-22]) - 1) * 100 if len(dxy) >= 22 else 0

        # Strong USD = risk-off = fear (inverse)
        score = np.clip(50 - ret_1m * 5, 0, 100)

        return {
            "name": "USD Strength 美元強度",
            "value": round(current, 2),
            "score": round(score),
            "detail": f"DXY={current:.1f}, 月漲跌={ret_1m:+.1f}%",
            "interpretation": "美元走強=Risk-Off避險" if ret_1m > 2 else "美元走弱=Risk-On偏多" if ret_1m < -2 else "穩定",
            "category": "currency",
        }
    except Exception as e:
        return {"name": "USD Strength 美元強度", "value": None, "score": None, "detail": str(e), "category": "currency"}


def get_taiex_sentiment() -> Dict:
    """TAIEX vs moving averages + TWSE margin data for TW market sentiment."""
    try:
        tw = yf.download("^TWII", period="1y", progress=False, auto_adjust=True)
        if tw.empty:
            return {"name": "TAIEX Sentiment", "value": None, "score": None, "category": "tw_market"}

        if isinstance(tw.columns, pd.MultiIndex):
            tw.columns = tw.columns.get_level_values(0)

        current = float(tw["Close"].iloc[-1])
        ma20 = float(tw["Close"].tail(20).mean())
        ma60 = float(tw["Close"].tail(60).mean()) if len(tw) >= 60 else current
        ma120 = float(tw["Close"].tail(120).mean()) if len(tw) >= 120 else current

        above_count = sum([current > ma20, current > ma60, current > ma120])
        pct_above_60 = (current - ma60) / ma60 * 100

        score = np.clip(above_count / 3 * 60 + pct_above_60 * 2 + 20, 0, 100)

        ma_status = []
        if current > ma20: ma_status.append("站上月線")
        else: ma_status.append("跌破月線")
        if current > ma60: ma_status.append("站上季線")
        else: ma_status.append("跌破季線")
        if current > ma120: ma_status.append("站上半年線")
        else: ma_status.append("跌破半年線")

        return {
            "name": "TAIEX 台股情緒",
            "value": round(current, 0),
            "score": round(score),
            "detail": f"加權={current:.0f}, 月線={ma20:.0f}, 季線={ma60:.0f}, 半年線={ma120:.0f}",
            "interpretation": " / ".join(ma_status),
            "category": "tw_market",
        }
    except Exception as e:
        return {"name": "TAIEX 台股情緒", "value": None, "score": None, "detail": str(e), "category": "tw_market"}


def get_crypto_fear_greed() -> Dict:
    """Crypto Fear & Greed Index from alternative.me (free, no key)."""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()["data"][0]

        value = int(data["value"])
        classification = data["value_classification"]

        return {
            "name": "Crypto Fear & Greed",
            "value": value,
            "score": value,  # already 0-100
            "detail": f"Score={value}, Class={classification}",
            "interpretation": classification,
            "category": "crypto",
        }
    except Exception as e:
        return {"name": "Crypto Fear & Greed", "value": None, "score": None, "detail": str(e), "category": "crypto"}


# ============================================================
# Composite Sentiment Score
# ============================================================

INDICATOR_WEIGHTS = {
    "VIX 恐慌指數": 20,
    "S&P 500 Momentum": 15,
    "Put/Call Ratio": 15,
    "Safe Haven Demand 避險需求": 10,
    "Market Breadth 市場廣度": 10,
    "Yield Curve 殖利率曲線": 10,
    "Gold 黃金 (避險指標)": 5,
    "USD Strength 美元強度": 5,
    "TAIEX 台股情緒": 5,
    "Crypto Fear & Greed": 5,
}


def get_all_sentiment_indicators(progress_callback=None) -> Tuple[float, list]:
    """Fetch all sentiment indicators and compute composite score.

    Returns:
        (composite_score, list_of_indicator_dicts)
    """
    fetchers = [
        get_vix_sentiment,
        get_sp500_momentum,
        get_put_call_ratio,
        get_safe_haven_demand,
        get_market_breadth,
        get_yield_curve,
        get_gold_signal,
        get_usd_strength,
        get_taiex_sentiment,
        get_crypto_fear_greed,
    ]

    indicators = []
    total = len(fetchers)

    for i, fn in enumerate(fetchers):
        if progress_callback:
            progress_callback(i, total, fn.__name__.replace("get_", ""))
        try:
            result = fn()
            indicators.append(result)
        except Exception as e:
            indicators.append({"name": fn.__name__, "value": None, "score": None, "detail": str(e)})

    if progress_callback:
        progress_callback(total, total, "Done")

    # Compute weighted composite
    weighted_sum = 0
    weight_total = 0
    for ind in indicators:
        if ind.get("score") is not None:
            w = INDICATOR_WEIGHTS.get(ind["name"], 5)
            weighted_sum += ind["score"] * w
            weight_total += w

    composite = weighted_sum / weight_total if weight_total > 0 else 50

    return round(composite, 1), indicators


def sentiment_label(score: float) -> Tuple[str, str]:
    """Convert score to label and color."""
    if score >= 80: return "Extreme Greed 極度貪婪", "#ff1744"
    if score >= 65: return "Greed 貪婪", "#ff9100"
    if score >= 55: return "Slightly Greedy 偏貪婪", "#ffd740"
    if score >= 45: return "Neutral 中性", "#e0e0e0"
    if score >= 35: return "Slightly Fearful 偏恐懼", "#69f0ae"
    if score >= 20: return "Fear 恐懼", "#00c853"
    return "Extreme Fear 極度恐懼", "#00e676"
