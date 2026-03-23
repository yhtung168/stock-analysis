"""Fundamental analysis chart builders."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, PLOTLY_LEGEND


def _dark_layout(fig, title="", height=300):
    """Apply dark theme."""
    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"],
        plot_bgcolor=COLORS["bg_primary"],
        font=dict(color=COLORS["text_primary"], family="Segoe UI, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=40),
        height=height,
        title=dict(text=title, font=dict(color=COLORS["accent"])) if title else None,
        legend=dict(**PLOTLY_LEGEND),
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])


def create_fundamental_radar(scores: list, title: str = "Fundamental Score Radar") -> go.Figure:
    """Create a radar chart of fundamental scores."""
    if not scores:
        return go.Figure()

    names = [s["name"] for s in scores]
    values = [s["score"] for s in scores]
    names.append(names[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=names,
        fill="toself",
        fillcolor="rgba(0,212,170,0.15)",
        line=dict(color=COLORS["accent"], width=2),
        marker=dict(size=6, color=COLORS["accent"]),
        name="Score",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[5] * len(names), theta=names,
        line=dict(color=COLORS["text_secondary"], dash="dash", width=1),
        name="Neutral (5)",
        fill="none",
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"],
        plot_bgcolor=COLORS["bg_primary"],
        font=dict(color=COLORS["text_primary"], family="Segoe UI, sans-serif"),
        polar=dict(
            bgcolor=COLORS["bg_primary"],
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=COLORS["border"],
                          color=COLORS["text_secondary"]),
            angularaxis=dict(gridcolor=COLORS["border"], color=COLORS["text_primary"]),
        ),
        title=dict(text=title, font=dict(color=COLORS["accent"])),
        height=450,
        showlegend=True,
        legend=dict(**PLOTLY_LEGEND),
    )
    return fig


def create_metric_trend_chart(
    financials: dict,
    metric_name: str,
    title: str = "",
    height: int = 300,
) -> go.Figure:
    """Create a trend chart for a financial metric."""
    fig = go.Figure()

    for period_key, label, color in [
        ("quarterly_income", "Quarterly", COLORS["chart_blue"]),
        ("income_stmt", "Annual", COLORS["accent"]),
    ]:
        stmt = financials.get(period_key, pd.DataFrame())
        if stmt is not None and not stmt.empty and metric_name in stmt.index:
            row = stmt.loc[metric_name].dropna()
            if not row.empty:
                fig.add_trace(go.Bar(
                    x=[str(d.date()) if hasattr(d, 'date') else str(d) for d in row.index],
                    y=row.values,
                    name=label,
                    marker_color=color,
                    opacity=0.7,
                ))

    _dark_layout(fig, title=title or metric_name, height=height)
    fig.update_layout(barmode="group")
    return fig


def create_dividend_chart(dividends_df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Create a dividend history chart."""
    fig = go.Figure()
    if dividends_df.empty:
        return fig

    div = dividends_df.copy()
    div.index = pd.to_datetime(div.index)
    yearly = div.resample("YE").sum()

    fig.add_trace(go.Bar(
        x=[str(d.year) for d in yearly.index],
        y=yearly["Dividend"].values,
        name="Annual Dividend",
        marker_color=COLORS["accent"],
        opacity=0.7,
    ))

    _dark_layout(fig, title="Dividend History (Annual)", height=height)
    return fig


def create_fundamental_summary_chart(metrics: dict, height: int = 350) -> go.Figure:
    """Create a horizontal bar chart showing key fundamental metrics."""
    items = []
    for key, label in [
        ("PE", "P/E"), ("PB", "P/B"), ("ROE", "ROE %"), ("ROA", "ROA %"),
        ("Dividend_Yield", "Div Yield %"), ("Operating_Margin", "Op Margin %"),
        ("Current_Ratio", "Current Ratio"), ("DE_Ratio", "D/E Ratio"),
    ]:
        val = metrics.get(key)
        if val is not None:
            if key in ("ROE", "ROA", "Dividend_Yield", "Operating_Margin") and val is not None:
                display_val = val * 100 if abs(val) < 1 else val
            elif key == "DE_Ratio" and val is not None:
                display_val = val / 100 if val > 10 else val
            else:
                display_val = val
            items.append((label, display_val))

    if not items:
        return go.Figure()

    names = [i[0] for i in items]
    values = [i[1] for i in items]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=COLORS["accent"],
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text_primary"]),
    ))

    _dark_layout(fig, title="Fundamental Metrics Overview", height=height)
    return fig
