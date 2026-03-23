"""Individual indicator scoring engine.

Each indicator gets a score from 1 to 10:
  9-10: Strong Buy
  7-8:  Buy
  5-6:  Neutral
  3-4:  Sell
  1-2:  Strong Sell
"""

import numpy as np
import pandas as pd
from typing import Optional


# ============================================================
# Technical Indicator Scoring
# ============================================================

def score_macd(signals: dict) -> dict:
    """Score MACD indicator."""
    dif = signals.get("MACD_DIF")
    sig = signals.get("MACD_Signal")
    hist = signals.get("MACD_Histogram")
    hist_prev = signals.get("MACD_Histogram_Prev")
    cross = signals.get("MACD_Cross", "none")

    if dif is None or pd.isna(dif):
        return _empty_score("MACD")

    score = 5  # neutral

    # Cross signals
    if cross == "golden":
        score = 8
        if dif > 0:
            score = 10  # golden cross above zero
    elif cross == "death":
        score = 3
        if dif < 0:
            score = 1  # death cross below zero
    else:
        # No cross, judge by DIF position and histogram
        if dif > 0:
            score = 7 if (hist and hist > 0) else 6
            if hist and hist_prev and hist > hist_prev:
                score = min(score + 1, 10)  # histogram expanding
        else:
            score = 4 if (hist and hist < 0) else 5
            if hist and hist_prev and hist < hist_prev:
                score = max(score - 1, 1)  # histogram expanding downward

    desc = _macd_description(dif, cross, hist)
    return _make_score("MACD", score, desc, _scoring_rule_macd())


def score_rsi(signals: dict) -> dict:
    """Score RSI indicator."""
    rsi = signals.get("RSI")
    if rsi is None or pd.isna(rsi):
        return _empty_score("RSI")

    if rsi < 20:
        score = 10
    elif rsi < 30:
        score = 8
    elif rsi < 40:
        score = 7
    elif rsi < 50:
        score = 5
    elif rsi < 60:
        score = 5
    elif rsi < 70:
        score = 4
    elif rsi < 80:
        score = 3
    else:
        score = 1

    desc = f"RSI(14) = {rsi:.1f}"
    if rsi < 30:
        desc += " — 超賣區，買進機會"
    elif rsi > 70:
        desc += " — 超買區，賣出警示"
    else:
        desc += " — 中性區間"

    return _make_score("RSI", score, desc, _scoring_rule_rsi())


def score_kd(signals: dict) -> dict:
    """Score KD / Stochastic indicator."""
    k = signals.get("K")
    d = signals.get("D")
    cross = signals.get("KD_Cross", "none")

    if k is None or pd.isna(k):
        return _empty_score("KD")

    score = 5
    if cross == "golden" and k < 30:
        score = 10
    elif cross == "golden":
        score = 8
    elif cross == "death" and k > 70:
        score = 1
    elif cross == "death":
        score = 3
    elif k < 20:
        score = 8
    elif k > 80:
        score = 2
    elif k > d:
        score = 6
    else:
        score = 4

    desc = f"K={k:.1f}, D={d:.1f}"
    if cross != "none":
        desc += f" — {'黃金交叉' if cross == 'golden' else '死亡交叉'}"

    return _make_score("KD", score, desc, _scoring_rule_kd())


def score_bollinger(signals: dict) -> dict:
    """Score Bollinger Bands."""
    close = signals.get("Close")
    upper = signals.get("BB_Upper")
    lower = signals.get("BB_Lower")
    pct_b = signals.get("BB_Percent")

    if close is None or upper is None or pd.isna(close) or pd.isna(upper):
        return _empty_score("Bollinger")

    score = 5
    if pct_b is not None and not pd.isna(pct_b):
        if pct_b < 0:
            score = 9  # below lower band
        elif pct_b < 0.2:
            score = 8
        elif pct_b < 0.4:
            score = 6
        elif pct_b < 0.6:
            score = 5
        elif pct_b < 0.8:
            score = 4
        elif pct_b < 1.0:
            score = 3
        else:
            score = 2  # above upper band
    else:
        if close <= lower:
            score = 9
        elif close >= upper:
            score = 2

    desc = f"股價位於布林通道 {pct_b*100:.0f}% 位置" if pct_b and not pd.isna(pct_b) else ""
    return _make_score("Bollinger", score, desc, _scoring_rule_bollinger())


def score_ma(signals: dict) -> dict:
    """Score Moving Average alignment."""
    close = signals.get("Close")
    sma20 = signals.get("SMA_20")
    sma60 = signals.get("SMA_60")
    bullish = signals.get("MA_Bullish_Align", False)
    bearish = signals.get("MA_Bearish_Align", False)

    if close is None or sma20 is None or pd.isna(close) or pd.isna(sma20):
        return _empty_score("MA")

    score = 5
    if bullish:
        score = 9
    elif bearish:
        score = 2
    elif close > sma20:
        score = 7
        if sma60 and not pd.isna(sma60) and close > sma60:
            score = 8
    elif close < sma20:
        score = 4
        if sma60 and not pd.isna(sma60) and close < sma60:
            score = 3

    desc_parts = []
    if close > sma20:
        desc_parts.append("站上 SMA20")
    else:
        desc_parts.append("跌破 SMA20")
    if bullish:
        desc_parts.append("多頭排列")
    elif bearish:
        desc_parts.append("空頭排列")

    return _make_score("MA (均線)", score, "，".join(desc_parts), _scoring_rule_ma())


def score_volume(signals: dict) -> dict:
    """Score Volume indicator."""
    ratio = signals.get("Volume_Ratio")
    obv_trend = signals.get("OBV_Trend", "unknown")

    if ratio is None or pd.isna(ratio):
        return _empty_score("Volume")

    score = 5
    if ratio > 2.0 and obv_trend == "up":
        score = 9
    elif ratio > 1.5 and obv_trend == "up":
        score = 8
    elif ratio > 1.0 and obv_trend == "up":
        score = 7
    elif ratio < 0.5:
        score = 4
    elif obv_trend == "down":
        score = 3

    desc = f"量比 {ratio:.2f}x, OBV {'上升' if obv_trend == 'up' else '下降'}"
    return _make_score("Volume", score, desc, "量比 > 1.5x + OBV上升 = 放量突破(8-9分)")


def score_williams_r(signals: dict) -> dict:
    """Score Williams %R."""
    wr = signals.get("Williams_R")
    if wr is None or pd.isna(wr):
        return _empty_score("Williams %R")

    if wr < -80:
        score = 9
    elif wr < -60:
        score = 7
    elif wr < -40:
        score = 5
    elif wr < -20:
        score = 3
    else:
        score = 1

    desc = f"%R = {wr:.1f}"
    return _make_score("Williams %R", score, desc, "%R < -80 超賣(9分), > -20 超買(1分)")


def score_cci(signals: dict) -> dict:
    """Score CCI."""
    cci = signals.get("CCI")
    if cci is None or pd.isna(cci):
        return _empty_score("CCI")

    if cci < -200:
        score = 10
    elif cci < -100:
        score = 8
    elif cci < 0:
        score = 5
    elif cci < 100:
        score = 5
    elif cci < 200:
        score = 3
    else:
        score = 1

    desc = f"CCI(20) = {cci:.0f}"
    return _make_score("CCI", score, desc, "CCI < -100 超賣(8分), > 100 超買(3分)")


def score_adx(signals: dict) -> dict:
    """Score ADX with DI crossover."""
    adx = signals.get("ADX")
    di_plus = signals.get("DI_Plus")
    di_minus = signals.get("DI_Minus")

    if adx is None or pd.isna(adx):
        return _empty_score("ADX")

    score = 5
    trending = adx > 25
    if trending:
        if di_plus and di_minus and not pd.isna(di_plus) and not pd.isna(di_minus):
            if di_plus > di_minus:
                score = 8 if adx > 40 else 7
            else:
                score = 3 if adx > 40 else 4
        else:
            score = 6
    else:
        score = 5  # no trend

    desc = f"ADX={adx:.0f}, +DI={di_plus:.0f}, -DI={di_minus:.0f}" if di_plus and di_minus else f"ADX={adx:.0f}"
    return _make_score("ADX", score, desc, "ADX > 25 有趨勢; +DI > -DI 偏多")


def score_psar(signals: dict) -> dict:
    """Score Parabolic SAR."""
    direction = signals.get("PSAR_Direction", "unknown")
    if direction == "unknown":
        return _empty_score("Parabolic SAR")

    score = 8 if direction == "bullish" else 3
    desc = "SAR 在股價下方（多頭）" if direction == "bullish" else "SAR 在股價上方（空頭）"
    return _make_score("Parabolic SAR", score, desc, "SAR下方=多頭(8分), SAR上方=空頭(3分)")


# ============================================================
# Fundamental Indicator Scoring
# ============================================================

def score_pe(fundamentals: dict) -> dict:
    """Score P/E ratio."""
    data = fundamentals.get("PE", {})
    pe = data.get("value")
    if pe is None:
        return _empty_score("P/E")

    if pe < 0:
        score = 3  # losing money
    elif pe < 10:
        score = 9
    elif pe < 15:
        score = 8
    elif pe < 20:
        score = 6
    elif pe < 25:
        score = 4
    elif pe < 30:
        score = 3
    else:
        score = 2

    return _make_score("P/E", score, data.get("description", ""), "P/E < 10 = 9分, 15-20 = 6分, > 30 = 2分")


def score_eps(fundamentals: dict) -> dict:
    """Score EPS."""
    data = fundamentals.get("EPS", {})
    growth = data.get("growth_pct")
    eps = data.get("value")

    if eps is None:
        return _empty_score("EPS")

    if growth is not None:
        if growth > 30:
            score = 10
        elif growth > 20:
            score = 9
        elif growth > 10:
            score = 7
        elif growth > 0:
            score = 6
        elif growth > -10:
            score = 4
        else:
            score = 2
    else:
        score = 6 if eps > 0 else 3

    return _make_score("EPS", score, data.get("description", ""), "EPS YoY > 20% = 9分, 衰退 = 2-4分")


def score_roe(fundamentals: dict) -> dict:
    """Score ROE."""
    data = fundamentals.get("ROE", {})
    roe = data.get("value")
    if roe is None:
        return _empty_score("ROE")

    if roe > 25:
        score = 10
    elif roe > 20:
        score = 9
    elif roe > 15:
        score = 7
    elif roe > 10:
        score = 5
    elif roe > 5:
        score = 4
    elif roe > 0:
        score = 3
    else:
        score = 1

    return _make_score("ROE", score, data.get("description", ""), "ROE > 20% = 9分, < 10% = 5分以下")


def score_dividend_yield(fundamentals: dict) -> dict:
    """Score Dividend Yield."""
    data = fundamentals.get("Dividend_Yield", {})
    dy = data.get("value")
    if dy is None:
        return _empty_score("Dividend Yield")

    if dy > 7:
        score = 9  # might be a trap though
    elif dy > 5:
        score = 8
    elif dy > 4:
        score = 7
    elif dy > 3:
        score = 6
    elif dy > 2:
        score = 5
    elif dy > 1:
        score = 4
    else:
        score = 3

    return _make_score("Dividend Yield", score, data.get("description", ""), "> 5% = 8分, < 2% = 4分")


def score_de_ratio(fundamentals: dict) -> dict:
    """Score Debt-to-Equity."""
    data = fundamentals.get("DE_Ratio", {})
    de = data.get("value")
    if de is None:
        return _empty_score("D/E Ratio")

    if de < 0.3:
        score = 9
    elif de < 0.5:
        score = 8
    elif de < 1.0:
        score = 6
    elif de < 1.5:
        score = 4
    elif de < 2.0:
        score = 3
    else:
        score = 2

    return _make_score("D/E Ratio", score, data.get("description", ""), "D/E < 0.5 = 8分, > 2.0 = 2分")


def score_operating_margin(fundamentals: dict) -> dict:
    """Score Operating Margin."""
    data = fundamentals.get("Operating_Margin", {})
    om = data.get("value")
    if om is None:
        return _empty_score("Operating Margin")

    if om > 30:
        score = 10
    elif om > 20:
        score = 8
    elif om > 15:
        score = 7
    elif om > 10:
        score = 6
    elif om > 5:
        score = 4
    else:
        score = 2

    return _make_score("Operating Margin", score, data.get("description", ""), "> 20% = 8分, < 5% = 2分")


def score_current_ratio(fundamentals: dict) -> dict:
    """Score Current Ratio."""
    data = fundamentals.get("Current_Ratio", {})
    cr = data.get("value")
    if cr is None:
        return _empty_score("Current Ratio")

    if cr > 3:
        score = 9
    elif cr > 2:
        score = 8
    elif cr > 1.5:
        score = 7
    elif cr > 1.0:
        score = 5
    else:
        score = 2

    return _make_score("Current Ratio", score, data.get("description", ""), "> 2.0 = 8分, < 1.0 = 2分")


def score_fcf(fundamentals: dict) -> dict:
    """Score Free Cash Flow."""
    data = fundamentals.get("FCF", {})
    fcf = data.get("value")
    fcf_yield = data.get("fcf_yield")
    if fcf is None:
        return _empty_score("FCF")

    if fcf > 0:
        if fcf_yield and fcf_yield > 8:
            score = 10
        elif fcf_yield and fcf_yield > 5:
            score = 8
        else:
            score = 7
    else:
        score = 3

    return _make_score("FCF", score, data.get("description", ""), "FCF正 + Yield > 5% = 8分")


def score_revenue_growth(fundamentals: dict) -> dict:
    """Score Revenue Growth."""
    data = fundamentals.get("Revenue_Growth", {})
    rg = data.get("value")
    if rg is None:
        return _empty_score("Revenue Growth")

    if rg > 30:
        score = 10
    elif rg > 20:
        score = 8
    elif rg > 10:
        score = 7
    elif rg > 0:
        score = 6
    elif rg > -10:
        score = 4
    else:
        score = 2

    return _make_score("Revenue Growth", score, data.get("description", ""), "YoY > 20% = 8分, 負成長 = 2-4分")


def score_pb(fundamentals: dict) -> dict:
    """Score P/B ratio."""
    data = fundamentals.get("PB", {})
    pb = data.get("value")
    if pb is None:
        return _empty_score("P/B")

    if pb < 0.8:
        score = 9
    elif pb < 1.0:
        score = 8
    elif pb < 1.5:
        score = 7
    elif pb < 2.0:
        score = 6
    elif pb < 3.0:
        score = 4
    else:
        score = 2

    return _make_score("P/B", score, data.get("description", ""), "P/B < 1.0 = 8分, > 3.0 = 2分")


def score_roa(fundamentals: dict) -> dict:
    """Score ROA."""
    data = fundamentals.get("ROA", {})
    roa = data.get("value")
    if roa is None:
        return _empty_score("ROA")

    if roa > 15:
        score = 10
    elif roa > 10:
        score = 8
    elif roa > 5:
        score = 6
    elif roa > 0:
        score = 4
    else:
        score = 2

    return _make_score("ROA", score, data.get("description", ""), "ROA > 10% = 8分, < 5% = 4分")


# ============================================================
# Chip / Institutional Scoring
# ============================================================

def score_institutional(inst_results: dict, key: str = "Foreign_Inv") -> dict:
    """Score institutional trading indicator."""
    data = inst_results.get(key, {})
    if not data:
        return _empty_score(key)

    value = data.get("value", 0)
    consecutive = data.get("consecutive_days", 0)

    # Scoring based on consecutive days and magnitude
    if consecutive >= 5:
        score = 9
    elif consecutive >= 3:
        score = 8
    elif value > 0:
        score = 6
    elif consecutive <= -5:
        score = 2
    elif consecutive <= -3:
        score = 3
    elif value < 0:
        score = 4
    else:
        score = 5

    return _make_score(
        data.get("label", key), score, data.get("description", ""),
        "連續5日買超 = 9分, 連續5日賣超 = 2分"
    )


def score_margin(margin_results: dict) -> dict:
    """Score margin balance (contrarian indicator)."""
    data = margin_results.get("Margin_Balance", {})
    if not data:
        return _empty_score("融資餘額")

    trend = data.get("trend", "")
    # Contrarian: margin increasing = bearish, decreasing = bullish
    if trend == "decreasing":
        score = 7  # bullish (chip consolidation)
    elif trend == "increasing":
        score = 4  # bearish (retail chasing)
    else:
        score = 5

    return _make_score("融資餘額", score, data.get("description", ""), "融資減少=籌碼沈澱(7分), 融資增加=偏空(4分)")


def score_short_margin_ratio(margin_results: dict) -> dict:
    """Score short/margin ratio."""
    data = margin_results.get("Short_Margin_Ratio", {})
    if not data:
        return _empty_score("券資比")

    ratio = data.get("value", 0)
    if ratio > 30:
        score = 8  # high short squeeze potential
    elif ratio > 20:
        score = 7
    elif ratio > 10:
        score = 6
    else:
        score = 5

    return _make_score("券資比", score, data.get("description", ""), "券資比 > 30% = 軋空機會(8分)")


def score_quantitative(quant_results: dict, key: str) -> dict:
    """Score quantitative metrics (Beta, Sharpe, Alpha)."""
    data = quant_results.get(key, {})
    if not data:
        return _empty_score(key)

    value = data.get("value", 0)

    if key == "Beta":
        # Neutral scoring - beta itself isn't buy/sell
        if 0.8 <= value <= 1.2:
            score = 6
        elif value < 0.8:
            score = 7  # defensive
        else:
            score = 4  # high risk
    elif key == "Sharpe_Ratio":
        if value > 2:
            score = 10
        elif value > 1:
            score = 8
        elif value > 0.5:
            score = 6
        elif value > 0:
            score = 4
        else:
            score = 2
    elif key == "Alpha":
        if value > 0.1:
            score = 9
        elif value > 0.05:
            score = 8
        elif value > 0:
            score = 7
        elif value > -0.05:
            score = 4
        else:
            score = 2
    else:
        score = 5

    return _make_score(data.get("label", key), score, data.get("description", ""), "")


# ============================================================
# Helpers
# ============================================================

def _make_score(name: str, score: int, description: str, scoring_rule: str) -> dict:
    """Create a standardized score dict."""
    score = max(1, min(10, score))
    return {
        "name": name,
        "score": score,
        "description": description,
        "scoring_rule": scoring_rule,
    }


def _empty_score(name: str) -> dict:
    return {"name": name, "score": None, "description": "資料不足", "scoring_rule": ""}


def _macd_description(dif, cross, hist):
    parts = []
    if cross == "golden":
        parts.append("金叉")
    elif cross == "death":
        parts.append("死叉")
    if dif > 0:
        parts.append("DIF > 0 (多方)")
    else:
        parts.append("DIF < 0 (空方)")
    if hist and hist > 0:
        parts.append("柱狀體正值")
    elif hist and hist < 0:
        parts.append("柱狀體負值")
    return "，".join(parts)


def _scoring_rule_macd():
    return "零軸上方金叉=10分, 金叉=8分, DIF>0且上升=7分, 死叉=3分, 零軸下方死叉=1分"

def _scoring_rule_rsi():
    return "RSI < 20 = 10分, 20-30 = 8分, 30-40 = 7分, 40-60 = 5分, 60-70 = 4分, 70-80 = 3分, > 80 = 1分"

def _scoring_rule_kd():
    return "低檔黃金交叉(K<30) = 10分, 黃金交叉 = 8分, K>D = 6分, 死叉 = 3分, 高檔死叉(K>70) = 1分"

def _scoring_rule_bollinger():
    return "%B < 0(跌破下軌) = 9分, 0-20% = 8分, 40-60% = 5分, > 100%(突破上軌) = 2分"

def _scoring_rule_ma():
    return "多頭排列 = 9分, 站上SMA20+60 = 8分, 站上SMA20 = 7分, 跌破SMA20 = 4分, 空頭排列 = 2分"
