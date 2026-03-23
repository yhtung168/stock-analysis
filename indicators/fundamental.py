"""Fundamental indicator analysis."""

import pandas as pd
import numpy as np
from typing import Optional


def analyze_fundamentals(metrics: dict, industry_avg: dict = None) -> dict:
    """Analyze fundamental metrics and return structured results.

    Args:
        metrics: dict from fetcher.get_fundamental_metrics()
        industry_avg: optional dict of industry average values for comparison

    Returns:
        dict of {indicator_name: {value, description, signal}}
    """
    if industry_avg is None:
        industry_avg = {}

    results = {}

    # --- P/E Ratio ---
    pe = metrics.get("PE")
    if pe is not None:
        ind_pe = industry_avg.get("PE", 20)
        ratio = pe / ind_pe if ind_pe else 1
        results["PE"] = {
            "value": pe,
            "label": "P/E Ratio (本益比)",
            "display": f"{pe:.1f}x",
            "description": _pe_description(pe, ind_pe),
            "ratio_to_industry": ratio,
        }

    # --- P/B Ratio ---
    pb = metrics.get("PB")
    if pb is not None:
        results["PB"] = {
            "value": pb,
            "label": "P/B Ratio (股價淨值比)",
            "display": f"{pb:.2f}x",
            "description": _pb_description(pb),
        }

    # --- EPS ---
    eps = metrics.get("EPS")
    fwd_eps = metrics.get("Forward_EPS")
    if eps is not None:
        growth = None
        if fwd_eps and eps and eps != 0:
            growth = (fwd_eps - eps) / abs(eps) * 100
        results["EPS"] = {
            "value": eps,
            "label": "EPS (每股盈餘)",
            "display": f"${eps:.2f}",
            "forward": fwd_eps,
            "growth_pct": growth,
            "description": _eps_description(eps, growth),
        }

    # --- ROE ---
    roe = metrics.get("ROE")
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) < 1 else roe
        results["ROE"] = {
            "value": roe_pct,
            "label": "ROE (股東權益報酬率)",
            "display": f"{roe_pct:.1f}%",
            "description": _roe_description(roe_pct),
        }

    # --- ROA ---
    roa = metrics.get("ROA")
    if roa is not None:
        roa_pct = roa * 100 if abs(roa) < 1 else roa
        results["ROA"] = {
            "value": roa_pct,
            "label": "ROA (資產報酬率)",
            "display": f"{roa_pct:.1f}%",
            "description": _roa_description(roa_pct),
        }

    # --- Dividend Yield ---
    div_yield = metrics.get("Dividend_Yield")
    if div_yield is not None:
        dy_pct = div_yield * 100 if div_yield < 1 else div_yield
        results["Dividend_Yield"] = {
            "value": dy_pct,
            "label": "Dividend Yield (殖利率)",
            "display": f"{dy_pct:.2f}%",
            "description": _dividend_description(dy_pct),
        }

    # --- Revenue Growth ---
    rev_growth = metrics.get("Revenue_Growth")
    if rev_growth is not None:
        rg_pct = rev_growth * 100 if abs(rev_growth) < 5 else rev_growth
        results["Revenue_Growth"] = {
            "value": rg_pct,
            "label": "Revenue Growth (營收成長率)",
            "display": f"{rg_pct:.1f}%",
            "description": _revenue_growth_description(rg_pct),
        }

    # --- Debt-to-Equity ---
    de = metrics.get("DE_Ratio")
    if de is not None:
        # yfinance returns D/E as percentage sometimes
        de_val = de / 100 if de > 10 else de
        results["DE_Ratio"] = {
            "value": de_val,
            "label": "Debt-to-Equity (負債比)",
            "display": f"{de_val:.2f}",
            "description": _de_description(de_val),
        }

    # --- Operating Margin ---
    op_margin = metrics.get("Operating_Margin")
    if op_margin is not None:
        om_pct = op_margin * 100 if abs(op_margin) < 1 else op_margin
        results["Operating_Margin"] = {
            "value": om_pct,
            "label": "Operating Margin (營業利益率)",
            "display": f"{om_pct:.1f}%",
            "description": _operating_margin_description(om_pct),
        }

    # --- Current Ratio ---
    cr = metrics.get("Current_Ratio")
    if cr is not None:
        results["Current_Ratio"] = {
            "value": cr,
            "label": "Current Ratio (流動比率)",
            "display": f"{cr:.2f}",
            "description": _current_ratio_description(cr),
        }

    # --- Free Cash Flow ---
    fcf = metrics.get("FCF")
    mkt_cap = metrics.get("Market_Cap")
    if fcf is not None:
        fcf_yield = (fcf / mkt_cap * 100) if mkt_cap and mkt_cap > 0 else None
        results["FCF"] = {
            "value": fcf,
            "label": "Free Cash Flow (自由現金流)",
            "display": _format_large_number(fcf),
            "fcf_yield": fcf_yield,
            "description": _fcf_description(fcf, fcf_yield),
        }

    return results


def _format_large_number(n: float) -> str:
    """Format large numbers for display."""
    if n is None:
        return "N/A"
    abs_n = abs(n)
    sign = "-" if n < 0 else ""
    if abs_n >= 1e12:
        return f"{sign}${abs_n/1e12:.1f}T"
    elif abs_n >= 1e9:
        return f"{sign}${abs_n/1e9:.1f}B"
    elif abs_n >= 1e6:
        return f"{sign}${abs_n/1e6:.1f}M"
    return f"{sign}${abs_n:,.0f}"


def _pe_description(pe, industry_pe):
    if pe < 0:
        return "P/E 為負值，表示公司虧損"
    if pe < 10:
        return f"P/E {pe:.1f}x，非常便宜（需確認是否為價值陷阱）"
    if pe < 15:
        return f"P/E {pe:.1f}x，估值偏低，具吸引力"
    if pe < 20:
        return f"P/E {pe:.1f}x，估值合理"
    if pe < 30:
        return f"P/E {pe:.1f}x，估值偏高，需高成長支撐"
    return f"P/E {pe:.1f}x，估值昂貴"


def _pb_description(pb):
    if pb < 1:
        return f"P/B {pb:.2f}x < 1，股價低於淨值，可能被低估"
    if pb < 1.5:
        return f"P/B {pb:.2f}x，估值合理偏低"
    if pb < 3:
        return f"P/B {pb:.2f}x，估值合理"
    return f"P/B {pb:.2f}x，估值偏高"


def _eps_description(eps, growth):
    parts = [f"EPS ${eps:.2f}"]
    if growth is not None:
        if growth > 20:
            parts.append(f"，預估成長 {growth:.0f}%（高成長）")
        elif growth > 0:
            parts.append(f"，預估成長 {growth:.0f}%")
        else:
            parts.append(f"，預估衰退 {growth:.0f}%")
    return "".join(parts)


def _roe_description(roe_pct):
    if roe_pct > 20:
        return f"ROE {roe_pct:.1f}%，優秀（> 20%）"
    if roe_pct > 15:
        return f"ROE {roe_pct:.1f}%，良好（15-20%）"
    if roe_pct > 10:
        return f"ROE {roe_pct:.1f}%，普通（10-15%）"
    if roe_pct > 0:
        return f"ROE {roe_pct:.1f}%，偏低（< 10%）"
    return f"ROE {roe_pct:.1f}%，虧損"


def _roa_description(roa_pct):
    if roa_pct > 10:
        return f"ROA {roa_pct:.1f}%，優秀"
    if roa_pct > 5:
        return f"ROA {roa_pct:.1f}%，良好"
    return f"ROA {roa_pct:.1f}%，偏低"


def _dividend_description(dy_pct):
    if dy_pct > 6:
        return f"殖利率 {dy_pct:.2f}%，高殖利率（注意是否為殖利率陷阱）"
    if dy_pct > 4:
        return f"殖利率 {dy_pct:.2f}%，良好"
    if dy_pct > 2:
        return f"殖利率 {dy_pct:.2f}%，一般"
    return f"殖利率 {dy_pct:.2f}%，偏低"


def _revenue_growth_description(rg_pct):
    if rg_pct > 20:
        return f"營收 YoY +{rg_pct:.1f}%，高速成長"
    if rg_pct > 0:
        return f"營收 YoY +{rg_pct:.1f}%，正成長"
    return f"營收 YoY {rg_pct:.1f}%，衰退"


def _de_description(de_val):
    if de_val < 0.5:
        return f"D/E {de_val:.2f}，低槓桿，保守安全"
    if de_val < 1.0:
        return f"D/E {de_val:.2f}，適中"
    if de_val < 2.0:
        return f"D/E {de_val:.2f}，偏高"
    return f"D/E {de_val:.2f}，高槓桿，風險較高"


def _operating_margin_description(om_pct):
    if om_pct > 20:
        return f"營益率 {om_pct:.1f}%，高利潤"
    if om_pct > 10:
        return f"營益率 {om_pct:.1f}%，良好"
    if om_pct > 5:
        return f"營益率 {om_pct:.1f}%，一般"
    return f"營益率 {om_pct:.1f}%，低利潤"


def _current_ratio_description(cr):
    if cr > 2.0:
        return f"流動比率 {cr:.2f}，安全"
    if cr > 1.5:
        return f"流動比率 {cr:.2f}，良好"
    if cr > 1.0:
        return f"流動比率 {cr:.2f}，尚可"
    return f"流動比率 {cr:.2f}，短期償債能力不足"


def _fcf_description(fcf, fcf_yield):
    parts = [f"FCF {_format_large_number(fcf)}"]
    if fcf_yield is not None:
        parts.append(f"，FCF Yield {fcf_yield:.1f}%")
        if fcf_yield > 5:
            parts.append("（偏便宜）")
    if fcf and fcf > 0:
        parts.append("，正向現金流")
    elif fcf and fcf < 0:
        parts.append("，負現金流（需關注原因）")
    return "".join(parts)
