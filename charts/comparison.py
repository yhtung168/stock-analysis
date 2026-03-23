"""Market comparison chart builders."""

import plotly.graph_objects as go
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
        legend=dict(**PLOTLY_LEGEND, orientation="h", y=-0.1, x=0),
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])


def create_comparison_chart(prices_df, title="Normalized Performance Comparison", height=450):
    if prices_df.empty:
        return go.Figure()

    normalized = prices_df / prices_df.iloc[0] * 100
    chart_colors = [
        COLORS["accent"], COLORS["chart_blue"], COLORS["chart_red"],
        COLORS["chart_yellow"], COLORS["chart_purple"], "#ff9800",
    ]

    fig = go.Figure()
    for i, col in enumerate(normalized.columns):
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[col],
            name=col, line=dict(color=chart_colors[i % len(chart_colors)], width=2),
        ))

    fig.add_hline(y=100, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.3)
    _dark_layout(fig, title=title, height=height)
    fig.update_yaxes(title_text="Normalized (%)")
    return fig


def create_performance_table(prices_df):
    if prices_df.empty:
        return pd.DataFrame()

    now = prices_df.index[-1]
    periods = {
        "1W": pd.Timedelta(days=7), "1M": pd.Timedelta(days=30),
        "3M": pd.Timedelta(days=90), "6M": pd.Timedelta(days=180),
        "1Y": pd.Timedelta(days=365),
    }

    results = []
    for col in prices_df.columns:
        row = {"Ticker": col}
        latest = prices_df[col].iloc[-1]
        for label, delta in periods.items():
            target = now - delta
            mask = prices_df.index >= target
            if mask.any():
                past = prices_df[col][mask].iloc[0]
                row[label] = f"{(latest - past) / past * 100:+.2f}%"
            else:
                row[label] = "N/A"
        year_start = pd.Timestamp(now.year, 1, 1)
        mask_ytd = prices_df.index >= year_start
        if mask_ytd.any():
            ytd = prices_df[col][mask_ytd].iloc[0]
            row["YTD"] = f"{(latest - ytd) / ytd * 100:+.2f}%"
        else:
            row["YTD"] = "N/A"
        results.append(row)
    return pd.DataFrame(results)


def calculate_correlation(prices_df):
    if prices_df.empty or len(prices_df.columns) < 2:
        return pd.DataFrame()
    return prices_df.pct_change().dropna().corr().round(3)


def create_correlation_heatmap(corr_df, title="Correlation Matrix", height=400):
    if corr_df.empty:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale=[[0, COLORS["negative"]], [0.5, COLORS["bg_card"]], [1, COLORS["positive"]]],
        zmid=0,
        text=corr_df.values.round(3),
        texttemplate="%{text}",
        textfont=dict(size=12, color=COLORS["text_primary"]),
    ))
    _dark_layout(fig, title=title, height=height)
    return fig
