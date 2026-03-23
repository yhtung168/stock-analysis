"""Chip / Institutional / Quantitative indicator analysis."""

import pandas as pd
import numpy as np
from typing import Optional


def analyze_institutional(inst_df: pd.DataFrame, lookback_days: int = 20) -> dict:
    """Analyze institutional trading patterns.

    Args:
        inst_df: DataFrame from twse_api with columns like foreign_net, trust_net, etc.
        lookback_days: number of recent days to analyze

    Returns:
        dict of indicator results
    """
    if inst_df.empty:
        return {}

    recent = inst_df.tail(lookback_days)
    results = {}

    # --- Foreign Investors ---
    if "foreign_net" in recent.columns:
        foreign_net_total = recent["foreign_net"].sum()
        foreign_net_5d = recent["foreign_net"].tail(5).sum()
        foreign_consecutive = _consecutive_direction(recent["foreign_net"])

        results["Foreign_Inv"] = {
            "value": foreign_net_total,
            "label": "外資買賣超",
            "display": _format_shares(foreign_net_total),
            "net_5d": foreign_net_5d,
            "net_5d_display": _format_shares(foreign_net_5d),
            "consecutive_days": foreign_consecutive,
            "description": _institutional_description("外資", foreign_net_total, foreign_consecutive),
        }

    # --- Investment Trust ---
    if "trust_net" in recent.columns:
        trust_net_total = recent["trust_net"].sum()
        trust_net_5d = recent["trust_net"].tail(5).sum()
        trust_consecutive = _consecutive_direction(recent["trust_net"])

        results["Investment_Trust"] = {
            "value": trust_net_total,
            "label": "投信買賣超",
            "display": _format_shares(trust_net_total),
            "net_5d": trust_net_5d,
            "consecutive_days": trust_consecutive,
            "description": _institutional_description("投信", trust_net_total, trust_consecutive),
        }

    # --- Dealers ---
    if "dealer_net" in recent.columns:
        dealer_net_total = recent["dealer_net"].sum()
        results["Dealers"] = {
            "value": dealer_net_total,
            "label": "自營商買賣超",
            "display": _format_shares(dealer_net_total),
            "description": _institutional_description("自營商", dealer_net_total, 0),
        }

    # --- Total Institutional ---
    if "total_net" in recent.columns:
        total_net = recent["total_net"].sum()
        total_consecutive = _consecutive_direction(recent["total_net"])
        results["Institutional_Total"] = {
            "value": total_net,
            "label": "三大法人合計",
            "display": _format_shares(total_net),
            "consecutive_days": total_consecutive,
            "description": _institutional_description("三大法人", total_net, total_consecutive),
        }

    return results


def analyze_margin(margin_df: pd.DataFrame, lookback_days: int = 20) -> dict:
    """Analyze margin trading patterns.

    Args:
        margin_df: DataFrame from twse_api with margin_balance, short_balance, etc.

    Returns:
        dict of indicator results
    """
    if margin_df.empty:
        return {}

    recent = margin_df.tail(lookback_days)
    results = {}

    # --- Margin Balance ---
    if "margin_balance" in recent.columns:
        latest_margin = recent["margin_balance"].iloc[-1]
        margin_change = recent["margin_balance"].diff().tail(5).sum()
        margin_trend = "increasing" if margin_change > 0 else "decreasing"

        results["Margin_Balance"] = {
            "value": latest_margin,
            "label": "融資餘額",
            "display": f"{latest_margin:,.0f} 張",
            "change_5d": margin_change,
            "trend": margin_trend,
            "description": _margin_description(margin_change, margin_trend),
        }

    # --- Short Balance ---
    if "short_balance" in recent.columns:
        latest_short = recent["short_balance"].iloc[-1]
        short_change = recent["short_balance"].diff().tail(5).sum()

        results["Short_Balance"] = {
            "value": latest_short,
            "label": "融券餘額",
            "display": f"{latest_short:,.0f} 張",
            "change_5d": short_change,
            "description": _short_description(short_change),
        }

    # --- Short/Margin Ratio ---
    if "margin_balance" in recent.columns and "short_balance" in recent.columns:
        margin_bal = recent["margin_balance"].iloc[-1]
        short_bal = recent["short_balance"].iloc[-1]
        ratio = (short_bal / margin_bal * 100) if margin_bal > 0 else 0

        results["Short_Margin_Ratio"] = {
            "value": ratio,
            "label": "券資比",
            "display": f"{ratio:.1f}%",
            "description": _short_margin_ratio_description(ratio),
        }

    return results


def calculate_quantitative_metrics(
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame = None,
    risk_free_rate: float = 0.02,
) -> dict:
    """Calculate quantitative metrics (Beta, Sharpe, Alpha).

    Args:
        price_df: stock price DataFrame with 'Close' column
        benchmark_df: benchmark price DataFrame with 'Close' column
        risk_free_rate: annual risk-free rate (default 2%)
    """
    results = {}

    if price_df.empty or "Close" not in price_df.columns:
        return results

    returns = price_df["Close"].pct_change().dropna()

    # --- Sharpe Ratio ---
    if len(returns) >= 20:
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        results["Sharpe_Ratio"] = {
            "value": sharpe,
            "label": "Sharpe Ratio",
            "display": f"{sharpe:.2f}",
            "description": _sharpe_description(sharpe),
        }

    # --- Beta & Alpha ---
    if benchmark_df is not None and not benchmark_df.empty and "Close" in benchmark_df.columns:
        bench_returns = benchmark_df["Close"].pct_change().dropna()
        # Align dates
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) >= 20:
            r_stock = returns.loc[common_idx]
            r_bench = bench_returns.loc[common_idx]

            cov = np.cov(r_stock, r_bench)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
            results["Beta"] = {
                "value": beta,
                "label": "Beta",
                "display": f"{beta:.2f}",
                "description": _beta_description(beta),
            }

            # Jensen's Alpha (annualized)
            daily_rf = risk_free_rate / 252
            alpha = (r_stock.mean() - daily_rf - beta * (r_bench.mean() - daily_rf)) * 252
            results["Alpha"] = {
                "value": alpha,
                "label": "Alpha (Jensen's)",
                "display": f"{alpha*100:.2f}%",
                "description": _alpha_description(alpha),
            }

    return results


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _consecutive_direction(series: pd.Series) -> int:
    """Count consecutive days of same direction (buy or sell) from latest."""
    if series.empty:
        return 0
    direction = 1 if series.iloc[-1] > 0 else -1
    count = 0
    for val in reversed(series.values):
        if (val > 0 and direction > 0) or (val < 0 and direction < 0):
            count += 1
        else:
            break
    return count * direction


def _format_shares(n):
    if n is None:
        return "N/A"
    sign = "+" if n > 0 else ""
    abs_n = abs(n)
    if abs_n >= 10000:
        return f"{sign}{n/1000:.0f}K 張"
    return f"{sign}{n:,.0f} 張"


def _institutional_description(name, net_total, consecutive):
    direction = "買超" if net_total > 0 else "賣超"
    abs_consec = abs(consecutive)
    parts = [f"{name}近期累計{direction} {_format_shares(abs(net_total))}"]
    if abs_consec >= 5:
        parts.append(f"，連續 {abs_consec} 日{'買超' if consecutive > 0 else '賣超'}")
    return "".join(parts)


def _margin_description(change_5d, trend):
    if trend == "increasing":
        return f"融資餘額增加 ({change_5d:+,.0f} 張)，散戶偏多 → 逆向偏空"
    return f"融資餘額減少 ({change_5d:+,.0f} 張)，籌碼沈澱 → 偏多"


def _short_description(change_5d):
    if change_5d > 0:
        return f"融券增加 ({change_5d:+,.0f} 張)，空方力道增加"
    return f"融券減少 ({change_5d:+,.0f} 張)，空方回補"


def _short_margin_ratio_description(ratio):
    if ratio > 30:
        return f"券資比 {ratio:.1f}% 偏高，軋空行情機會"
    if ratio > 15:
        return f"券資比 {ratio:.1f}%，空方壓力中等"
    return f"券資比 {ratio:.1f}%，正常水位"


def _beta_description(beta):
    if beta > 1.5:
        return f"Beta {beta:.2f}，高波動（攻擊型），波動遠大於大盤"
    if beta > 1.0:
        return f"Beta {beta:.2f}，波動略大於大盤"
    if beta > 0.5:
        return f"Beta {beta:.2f}，波動小於大盤（防禦型）"
    return f"Beta {beta:.2f}，低波動"


def _sharpe_description(sharpe):
    if sharpe > 2:
        return f"Sharpe {sharpe:.2f}，優秀（風險調整後報酬極佳）"
    if sharpe > 1:
        return f"Sharpe {sharpe:.2f}，良好"
    if sharpe > 0.5:
        return f"Sharpe {sharpe:.2f}，普通"
    return f"Sharpe {sharpe:.2f}，偏差"


def _alpha_description(alpha):
    if alpha > 0:
        return f"Alpha {alpha*100:.2f}%，優於大盤（超額報酬為正）"
    return f"Alpha {alpha*100:.2f}%，劣於大盤"
