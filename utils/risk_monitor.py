"""Global Risk Monitor - Black Swan / Gray Rhino Radar.

Monitors financial stress, geopolitical risk, and news-based risk signals.
Produces a composite risk level: LOW / MODERATE / ELEVATED / HIGH / EXTREME.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
import re


# ============================================================
# Risk keyword scoring
# ============================================================
RISK_KEYWORDS = {
    # Severity 5 - Extreme
    "financial crisis": 5, "bank run": 5, "sovereign default": 5,
    "nuclear": 5, "world war": 5, "market crash": 5, "systemic risk": 5,
    "lehman": 5, "contagion": 5, "meltdown": 5, "collapse": 5,
    # Severity 4 - High
    "recession": 4, "black swan": 4, "credit crunch": 4,
    "bank failure": 4, "debt crisis": 4, "currency crisis": 4,
    "trade war": 4, "sanctions": 4, "invasion": 4, "pandemic": 4,
    "hyperinflation": 4, "stagflation": 4, "liquidity crisis": 4,
    "margin call": 4, "flash crash": 4,
    # Severity 3 - Elevated
    "tariff": 3, "geopolitical": 3, "inflation": 3, "rate hike": 3,
    "downgrade": 3, "layoffs": 3, "bankruptcy": 3, "bear market": 3,
    "sell-off": 3, "selloff": 3, "volatility spike": 3, "bubble": 3,
    "gray rhino": 3, "灰犀牛": 3, "黑天鵝": 3,
    "debt ceiling": 3, "government shutdown": 3, "default risk": 3,
    # Severity 2 - Moderate
    "uncertainty": 2, "risk": 2, "warning": 2, "concern": 2,
    "slowdown": 2, "decline": 2, "correction": 2, "weak": 2,
    "tension": 2, "conflict": 2, "volatility": 2, "fear": 2,
    "supply chain": 2, "rate cut": 2,
    # Severity 1 - Low
    "caution": 1, "headwind": 1, "challenge": 1, "pressure": 1,
}


def scan_news_risk(max_articles: int = 30) -> Dict:
    """Scan Google News RSS for financial risk keywords.

    Returns dict with risk_score, headlines, and top_risks.
    """
    feeds = [
        "https://news.google.com/rss/search?q=financial+market+risk&hl=en&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=stock+market+crash+recession&hl=en&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=black+swan+financial+crisis&hl=en&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=geopolitical+risk+trade+war&hl=en&gl=US&ceid=US:en",
    ]

    all_headlines = []
    risk_hits = []
    seen_titles = set()

    headers = {"User-Agent": "Mozilla/5.0 StockAnalysisPlatform/1.0"}

    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=10, headers=headers)
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")

            for item in items[:max_articles]:
                title = item.find("title")
                pub_date = item.find("pubDate")
                link = item.find("link")

                if title is None:
                    continue
                title_text = title.text or ""

                # Deduplicate
                title_lower = title_text.lower().strip()
                if title_lower in seen_titles:
                    continue
                seen_titles.add(title_lower)

                # Score headline
                headline_score = 0
                matched_keywords = []
                for keyword, severity in RISK_KEYWORDS.items():
                    if keyword.lower() in title_lower:
                        headline_score = max(headline_score, severity)
                        matched_keywords.append((keyword, severity))

                date_str = pub_date.text if pub_date is not None else ""
                link_str = link.text if link is not None else ""

                entry = {
                    "title": title_text,
                    "date": date_str,
                    "link": link_str,
                    "score": headline_score,
                    "keywords": matched_keywords,
                }
                all_headlines.append(entry)
                if headline_score >= 3:
                    risk_hits.append(entry)

        except Exception:
            continue

    # Compute overall news risk score (0-10)
    if not all_headlines:
        news_score = 5  # neutral if no data
    else:
        scores = [h["score"] for h in all_headlines]
        high_risk_count = sum(1 for s in scores if s >= 4)
        med_risk_count = sum(1 for s in scores if s >= 3)
        avg_score = np.mean(scores)

        # Weighted: high-severity headlines matter more
        news_score = min(avg_score * 1.5 + high_risk_count * 0.5, 10)

    # Sort by severity
    risk_hits.sort(key=lambda x: x["score"], reverse=True)

    return {
        "news_risk_score": round(news_score, 1),
        "total_headlines": len(all_headlines),
        "high_risk_headlines": len(risk_hits),
        "top_risks": risk_hits[:15],
        "all_headlines": all_headlines[:50],
    }


# ============================================================
# Financial Stress Indicators (yfinance-based, no FRED key needed)
# ============================================================

def get_yield_curve_risk() -> Dict:
    """10Y-5Y yield spread as recession indicator."""
    try:
        data = yf.download(["^TNX", "^FVX"], period="1y", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            tnx = data["Close"]["^TNX"].dropna()
            fvx = data["Close"]["^FVX"].dropna()
        else:
            return _empty_risk("Yield Curve")

        if tnx.empty or fvx.empty:
            return _empty_risk("Yield Curve")

        spread = float(tnx.iloc[-1]) - float(fvx.iloc[-1])
        spread_3m_ago = float(tnx.iloc[-60]) - float(fvx.iloc[-60]) if len(tnx) >= 60 else spread

        # Risk: inverted = high risk
        if spread < -0.5:
            risk = 9
        elif spread < -0.2:
            risk = 7
        elif spread < 0:
            risk = 6
        elif spread < 0.2:
            risk = 4
        else:
            risk = 2

        trend = "Steepening ↑" if spread > spread_3m_ago else "Flattening ↓"

        return {
            "name": "Yield Curve 殖利率曲線",
            "value": round(spread, 3),
            "risk_score": risk,
            "detail": f"10Y-5Y Spread = {spread:.3f}%, {trend}",
            "interpretation": "倒掛 = 衰退訊號!" if spread < 0 else "正常殖利率曲線",
            "icon": "📉" if spread < 0 else "📊",
        }
    except Exception as e:
        return _empty_risk("Yield Curve", str(e))


def get_vix_risk() -> Dict:
    """VIX as market fear gauge."""
    try:
        vix = yf.download("^VIX", period="6mo", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        current = float(vix["Close"].iloc[-1])
        avg = float(vix["Close"].mean())
        high = float(vix["Close"].max())

        if current > 40: risk = 10
        elif current > 30: risk = 8
        elif current > 25: risk = 6
        elif current > 20: risk = 4
        elif current > 15: risk = 2
        else: risk = 1

        return {
            "name": "VIX 市場波動",
            "value": round(current, 2),
            "risk_score": risk,
            "detail": f"VIX={current:.1f}, 6M Avg={avg:.1f}, 6M High={high:.1f}",
            "interpretation": _vix_risk_interpret(current),
            "icon": "🔴" if current > 30 else "🟡" if current > 20 else "🟢",
        }
    except Exception as e:
        return _empty_risk("VIX", str(e))


def _vix_risk_interpret(v):
    if v > 40: return "極度恐慌 — 市場可能已接近底部"
    if v > 30: return "高度恐慌 — 注意系統性風險"
    if v > 25: return "市場不安 — 保持警戒"
    if v > 20: return "偏高波動 — 正常偏高"
    return "平靜 — 低風險"


def get_credit_risk() -> Dict:
    """High yield bond ETF (HYG) vs investment grade (LQD) spread."""
    try:
        data = yf.download(["HYG", "LQD"], period="6mo", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            hyg = data["Close"]["HYG"].dropna()
            lqd = data["Close"]["LQD"].dropna()
        else:
            return _empty_risk("Credit Risk")

        # HYG/LQD ratio: declining = credit stress
        ratio_now = float(hyg.iloc[-1]) / float(lqd.iloc[-1])
        ratio_3m = float(hyg.iloc[-60]) / float(lqd.iloc[-60]) if len(hyg) >= 60 else ratio_now

        change = (ratio_now / ratio_3m - 1) * 100

        # Declining ratio = high yield underperforming = credit stress
        if change < -5: risk = 9
        elif change < -3: risk = 7
        elif change < -1: risk = 5
        elif change < 1: risk = 3
        else: risk = 1

        return {
            "name": "Credit Spread 信用風險",
            "value": round(change, 2),
            "risk_score": risk,
            "detail": f"HYG/LQD ratio change: {change:+.2f}% (3M)",
            "interpretation": "High Yield 表現弱 = 信用風險上升" if change < -2 else "信用市場穩定",
            "icon": "🔴" if risk >= 7 else "🟡" if risk >= 4 else "🟢",
        }
    except Exception as e:
        return _empty_risk("Credit Risk", str(e))


def get_gold_risk() -> Dict:
    """Gold surge = risk-off demand."""
    try:
        gold = yf.download("GC=F", period="6mo", progress=False, auto_adjust=True)
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)

        current = float(gold["Close"].iloc[-1])
        low_6m = float(gold["Close"].min())
        ret_3m = (current / float(gold["Close"].iloc[-60]) - 1) * 100 if len(gold) >= 60 else 0

        # Rapid gold rise = fear
        if ret_3m > 15: risk = 8
        elif ret_3m > 10: risk = 6
        elif ret_3m > 5: risk = 4
        elif ret_3m > 0: risk = 3
        else: risk = 1

        return {
            "name": "Gold 黃金避險",
            "value": round(current, 0),
            "risk_score": risk,
            "detail": f"Gold=${current:.0f}, 3M Return={ret_3m:+.1f}%",
            "interpretation": "黃金急漲 = 避險需求強" if ret_3m > 10 else "黃金平穩" if ret_3m < 5 else "黃金溫和上漲",
            "icon": "🥇",
        }
    except Exception as e:
        return _empty_risk("Gold", str(e))


def get_oil_risk() -> Dict:
    """Oil price spike = inflation / geopolitical risk."""
    try:
        oil = yf.download("CL=F", period="6mo", progress=False, auto_adjust=True)
        if isinstance(oil.columns, pd.MultiIndex):
            oil.columns = oil.columns.get_level_values(0)

        current = float(oil["Close"].iloc[-1])
        ret_1m = (current / float(oil["Close"].iloc[-22]) - 1) * 100 if len(oil) >= 22 else 0

        if ret_1m > 20: risk = 9
        elif ret_1m > 10: risk = 7
        elif ret_1m > 5: risk = 4
        elif ret_1m > -5: risk = 2
        else: risk = 1  # falling oil = deflation risk but less acute

        return {
            "name": "Oil 原油 (地緣風險)",
            "value": round(current, 2),
            "risk_score": risk,
            "detail": f"WTI=${current:.1f}, 月漲跌={ret_1m:+.1f}%",
            "interpretation": "油價急漲 = 地緣/通膨風險" if ret_1m > 10 else "油價穩定",
            "icon": "🛢️",
        }
    except Exception as e:
        return _empty_risk("Oil", str(e))


def get_usd_risk() -> Dict:
    """Strong USD = emerging market stress, risk-off."""
    try:
        dxy = yf.download("DX-Y.NYB", period="6mo", progress=False, auto_adjust=True)
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy.columns = dxy.columns.get_level_values(0)

        current = float(dxy["Close"].iloc[-1])
        ret_3m = (current / float(dxy["Close"].iloc[-60]) - 1) * 100 if len(dxy) >= 60 else 0

        if ret_3m > 8: risk = 8
        elif ret_3m > 5: risk = 6
        elif ret_3m > 2: risk = 4
        else: risk = 2

        return {
            "name": "USD 美元 (Risk-Off)",
            "value": round(current, 2),
            "risk_score": risk,
            "detail": f"DXY={current:.1f}, 3M Change={ret_3m:+.1f}%",
            "interpretation": "美元急升 = 全球Risk-Off" if ret_3m > 5 else "美元穩定",
            "icon": "💵",
        }
    except Exception as e:
        return _empty_risk("USD", str(e))


def _empty_risk(name, error="No data"):
    return {"name": name, "value": None, "risk_score": 5, "detail": error, "interpretation": "N/A", "icon": "❓"}


# ============================================================
# Composite Risk Dashboard
# ============================================================

RISK_WEIGHTS = {
    "VIX 市場波動": 20,
    "Yield Curve 殖利率曲線": 15,
    "Credit Spread 信用風險": 15,
    "News Risk 新聞風險": 15,
    "Gold 黃金避險": 10,
    "Oil 原油 (地緣風險)": 10,
    "USD 美元 (Risk-Off)": 10,
}


def get_full_risk_assessment(progress_callback=None) -> Dict:
    """Run complete risk assessment.

    Returns dict with:
        composite_risk: 0-10 score
        risk_level: LOW/MODERATE/ELEVATED/HIGH/EXTREME
        indicators: list of indicator dicts
        news: news scan results
    """
    if progress_callback:
        progress_callback(0, 8, "VIX...")

    indicators = []
    fetchers = [
        ("VIX...", get_vix_risk),
        ("Yield Curve...", get_yield_curve_risk),
        ("Credit Spread...", get_credit_risk),
        ("Gold...", get_gold_risk),
        ("Oil...", get_oil_risk),
        ("USD...", get_usd_risk),
    ]

    for i, (label, fn) in enumerate(fetchers):
        if progress_callback:
            progress_callback(i, 8, label)
        try:
            indicators.append(fn())
        except Exception as e:
            indicators.append(_empty_risk(label, str(e)))

    # News scan
    if progress_callback:
        progress_callback(6, 8, "Scanning news...")
    news = scan_news_risk()

    news_indicator = {
        "name": "News Risk 新聞風險",
        "value": news["news_risk_score"],
        "risk_score": min(round(news["news_risk_score"]), 10),
        "detail": f"{news['high_risk_headlines']} high-risk headlines / {news['total_headlines']} total",
        "interpretation": f"High-risk keywords detected in {news['high_risk_headlines']} articles",
        "icon": "📰",
    }
    indicators.append(news_indicator)

    if progress_callback:
        progress_callback(7, 8, "Computing composite...")

    # Composite weighted score
    weighted_sum = 0
    weight_total = 0
    for ind in indicators:
        rs = ind.get("risk_score")
        if rs is not None:
            w = RISK_WEIGHTS.get(ind["name"], 5)
            weighted_sum += rs * w
            weight_total += w

    composite = weighted_sum / weight_total if weight_total > 0 else 5

    # Risk level
    if composite >= 8: level, color = "EXTREME 極端風險", "#ff1744"
    elif composite >= 6: level, color = "HIGH 高風險", "#ff9100"
    elif composite >= 4: level, color = "ELEVATED 風險升高", "#ffd740"
    elif composite >= 2.5: level, color = "MODERATE 中度風險", "#69f0ae"
    else: level, color = "LOW 低風險", "#00c853"

    if progress_callback:
        progress_callback(8, 8, "Done")

    return {
        "composite_risk": round(composite, 1),
        "risk_level": level,
        "risk_color": color,
        "indicators": indicators,
        "news": news,
    }
