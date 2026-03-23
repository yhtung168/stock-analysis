"""Backtest chart builders."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, PLOTLY_LEGEND


def _dark_layout(fig, title="", height=400):
    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"],
        plot_bgcolor=COLORS["bg_primary"],
        font=dict(color=COLORS["text_primary"], family="Segoe UI, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=40),
        height=height,
        title=dict(text=title, font=dict(color=COLORS["accent"])) if title else None,
        legend=dict(**PLOTLY_LEGEND, orientation="h", y=1.05, x=0),
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])


def create_backtest_equity_chart(backtest_result, title="Backtest Results", height=500):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Price + Trading Signals", "Equity Curve"),
    )

    price_series = backtest_result.get("price_series", pd.Series())
    if not price_series.empty:
        fig.add_trace(go.Scatter(
            x=price_series.index, y=price_series.values,
            name="Price", line=dict(color=COLORS["text_primary"], width=1),
        ), row=1, col=1)

    buys = backtest_result.get("buy_signals", [])
    if buys:
        bd, bp = zip(*buys)
        fig.add_trace(go.Scatter(
            x=list(bd), y=list(bp), mode="markers", name="Buy",
            marker=dict(color=COLORS["positive"], size=10, symbol="triangle-up"),
        ), row=1, col=1)

    sells = backtest_result.get("sell_signals", [])
    if sells:
        sd, sp = zip(*sells)
        fig.add_trace(go.Scatter(
            x=list(sd), y=list(sp), mode="markers", name="Sell",
            marker=dict(color=COLORS["negative"], size=10, symbol="triangle-down"),
        ), row=1, col=1)

    equity = backtest_result.get("equity_curve", pd.Series())
    if not equity.empty:
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values, name="Portfolio",
            line=dict(color=COLORS["accent"], width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.05)",
        ), row=2, col=1)

    bench = backtest_result.get("benchmark_curve", pd.Series())
    if not bench.empty:
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench.values, name="Buy & Hold",
            line=dict(color=COLORS["text_secondary"], dash="dash", width=1),
        ), row=2, col=1)

    _dark_layout(fig, title=title, height=height)
    return fig


def create_score_history_chart(score_history, title="Historical Composite Score", height=300):
    if score_history.empty:
        return go.Figure()

    fig = go.Figure()
    scores = score_history["composite_score"]
    fig.add_trace(go.Scatter(
        x=score_history.index, y=scores, name="Composite Score",
        line=dict(color=COLORS["accent"], width=2),
    ))

    fig.add_hrect(y0=8.5, y1=10, fillcolor=COLORS["positive"], opacity=0.08,
                  annotation_text="Strong Buy", annotation_position="top left")
    fig.add_hrect(y0=7.0, y1=8.5, fillcolor="#66bb6a", opacity=0.05)
    fig.add_hrect(y0=5.5, y1=7.0, fillcolor=COLORS["warning"], opacity=0.05)
    fig.add_hrect(y0=3.5, y1=5.5, fillcolor="#ef5350", opacity=0.05)
    fig.add_hrect(y0=1.0, y1=3.5, fillcolor=COLORS["negative"], opacity=0.08,
                  annotation_text="Strong Sell", annotation_position="bottom left")

    fig.add_hline(y=7, line_color=COLORS["positive"], line_dash="dash", opacity=0.3)
    fig.add_hline(y=4, line_color=COLORS["negative"], line_dash="dash", opacity=0.3)

    _dark_layout(fig, title=title, height=height)
    fig.update_yaxes(range=[1, 10], title_text="Score")
    return fig


def create_drawdown_chart(equity_curve, title="Drawdown", height=200):
    if equity_curve.empty:
        return go.Figure()

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values, name="Drawdown %",
        fill="tozeroy", line=dict(color=COLORS["negative"], width=1),
        fillcolor="rgba(255,82,82,0.15)",
    ))
    _dark_layout(fig, title=title, height=height)
    fig.update_yaxes(title_text="Drawdown %")
    return fig
