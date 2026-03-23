"""Composite scoring engine that combines all individual scores."""

from typing import Optional
from . import scorer


def compute_all_scores(
    tech_signals: dict,
    fundamentals: dict,
    institutional: dict,
    margin: dict,
    quantitative: dict,
    enabled_indicators: list = None,
) -> dict:
    """Compute all individual scores and composite score.

    Returns:
        {
            "technical": [list of score dicts],
            "fundamental": [list of score dicts],
            "chip": [list of score dicts],
            "technical_avg": float,
            "fundamental_avg": float,
            "chip_avg": float,
            "composite": float,  (computed by caller with weights)
        }
    """
    # --- Technical scores ---
    tech_scores = []
    tech_scorers = {
        "MACD": lambda: scorer.score_macd(tech_signals),
        "RSI": lambda: scorer.score_rsi(tech_signals),
        "KD": lambda: scorer.score_kd(tech_signals),
        "Bollinger": lambda: scorer.score_bollinger(tech_signals),
        "MA": lambda: scorer.score_ma(tech_signals),
        "Volume": lambda: scorer.score_volume(tech_signals),
        "Williams_R": lambda: scorer.score_williams_r(tech_signals),
        "CCI": lambda: scorer.score_cci(tech_signals),
        "ADX": lambda: scorer.score_adx(tech_signals),
        "Parabolic_SAR": lambda: scorer.score_psar(tech_signals),
    }
    for name, fn in tech_scorers.items():
        if enabled_indicators is None or name in enabled_indicators:
            result = fn()
            if result["score"] is not None:
                tech_scores.append(result)

    # --- Fundamental scores ---
    fund_scores = []
    fund_scorers = {
        "PE": lambda: scorer.score_pe(fundamentals),
        "PB": lambda: scorer.score_pb(fundamentals),
        "EPS": lambda: scorer.score_eps(fundamentals),
        "ROE": lambda: scorer.score_roe(fundamentals),
        "ROA": lambda: scorer.score_roa(fundamentals),
        "Dividend_Yield": lambda: scorer.score_dividend_yield(fundamentals),
        "Revenue_Growth": lambda: scorer.score_revenue_growth(fundamentals),
        "DE_Ratio": lambda: scorer.score_de_ratio(fundamentals),
        "FCF": lambda: scorer.score_fcf(fundamentals),
        "Operating_Margin": lambda: scorer.score_operating_margin(fundamentals),
        "Current_Ratio": lambda: scorer.score_current_ratio(fundamentals),
    }
    for name, fn in fund_scorers.items():
        if enabled_indicators is None or name in enabled_indicators:
            result = fn()
            if result["score"] is not None:
                fund_scores.append(result)

    # --- Chip / Institutional scores ---
    chip_scores = []
    # Institutional
    for key in ["Foreign_Inv", "Investment_Trust", "Dealers", "Institutional_Total"]:
        if enabled_indicators is None or key in enabled_indicators:
            result = scorer.score_institutional(institutional, key)
            if result["score"] is not None:
                chip_scores.append(result)

    # Margin
    if enabled_indicators is None or "Margin_Balance" in enabled_indicators:
        result = scorer.score_margin(margin)
        if result["score"] is not None:
            chip_scores.append(result)

    if enabled_indicators is None or "Short_Margin_Ratio" in enabled_indicators:
        result = scorer.score_short_margin_ratio(margin)
        if result["score"] is not None:
            chip_scores.append(result)

    # Quantitative
    for key in ["Beta", "Sharpe_Ratio", "Alpha"]:
        if enabled_indicators is None or key in enabled_indicators:
            result = scorer.score_quantitative(quantitative, key)
            if result["score"] is not None:
                chip_scores.append(result)

    # --- Averages ---
    tech_avg = _avg_scores(tech_scores)
    fund_avg = _avg_scores(fund_scores)
    chip_avg = _avg_scores(chip_scores)

    return {
        "technical": tech_scores,
        "fundamental": fund_scores,
        "chip": chip_scores,
        "technical_avg": tech_avg,
        "fundamental_avg": fund_avg,
        "chip_avg": chip_avg,
    }


def compute_composite(
    scores: dict,
    weights: dict,
) -> float:
    """Compute the weighted composite score.

    Args:
        scores: output from compute_all_scores
        weights: {"technical": 0.4, "fundamental": 0.35, "chip": 0.25}
    """
    tech = scores.get("technical_avg")
    fund = scores.get("fundamental_avg")
    chip = scores.get("chip_avg")

    w_tech = weights.get("technical", 0.4)
    w_fund = weights.get("fundamental", 0.35)
    w_chip = weights.get("chip", 0.25)

    # Only use categories that have scores
    total_weight = 0
    weighted_sum = 0

    if tech is not None:
        weighted_sum += tech * w_tech
        total_weight += w_tech
    if fund is not None:
        weighted_sum += fund * w_fund
        total_weight += w_fund
    if chip is not None:
        weighted_sum += chip * w_chip
        total_weight += w_chip

    if total_weight == 0:
        return 5.0  # neutral default

    return weighted_sum / total_weight


def generate_highlights_risks(scores: dict) -> tuple:
    """Generate highlights (positive) and risks lists from scores.

    Returns:
        (highlights: list[str], risks: list[str])
    """
    highlights = []
    risks = []

    all_scores = scores["technical"] + scores["fundamental"] + scores["chip"]

    for s in sorted(all_scores, key=lambda x: x["score"], reverse=True):
        if s["score"] >= 8 and s["description"]:
            highlights.append(f"**{s['name']}** ({s['score']}/10): {s['description']}")
        elif s["score"] <= 3 and s["description"]:
            risks.append(f"**{s['name']}** ({s['score']}/10): {s['description']}")

    return highlights[:6], risks[:6]


def _avg_scores(scores: list) -> Optional[float]:
    """Calculate average of score dicts."""
    valid = [s["score"] for s in scores if s["score"] is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)
