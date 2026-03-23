"""Chip / Institutional analysis chart builders."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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


def create_institutional_chart(inst_df, title="三大法人買賣超", height=400):
    if inst_df.empty:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.6, 0.4],
                        subplot_titles=("每日買賣超", "累計買賣超"))

    if "foreign_net" in inst_df.columns:
        colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in inst_df["foreign_net"]]
        fig.add_trace(go.Bar(
            x=inst_df.index, y=inst_df["foreign_net"], name="外資",
            marker_color=colors, opacity=0.7,
        ), row=1, col=1)

    if "trust_net" in inst_df.columns:
        fig.add_trace(go.Scatter(
            x=inst_df.index, y=inst_df["trust_net"], name="投信",
            line=dict(color=COLORS["chart_yellow"], width=1.5),
        ), row=1, col=1)

    if "dealer_net" in inst_df.columns:
        fig.add_trace(go.Scatter(
            x=inst_df.index, y=inst_df["dealer_net"], name="自營商",
            line=dict(color=COLORS["chart_purple"], width=1),
        ), row=1, col=1)

    for col, name, color in [
        ("foreign_net", "外資(累計)", COLORS["chart_blue"]),
        ("trust_net", "投信(累計)", COLORS["chart_yellow"]),
        ("total_net", "合計(累計)", COLORS["accent"]),
    ]:
        if col in inst_df.columns:
            fig.add_trace(go.Scatter(
                x=inst_df.index, y=inst_df[col].cumsum(), name=name,
                line=dict(color=color, width=1.5),
            ), row=2, col=1)

    fig.add_hline(y=0, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.3, row=1, col=1)
    fig.add_hline(y=0, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.3, row=2, col=1)
    _dark_layout(fig, title=title, height=height)
    return fig


def create_margin_chart(margin_df, title="融資融券", height=350):
    if margin_df.empty:
        return go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.5, 0.5],
                        subplot_titles=("融資餘額", "融券餘額"))

    if "margin_balance" in margin_df.columns:
        fig.add_trace(go.Scatter(
            x=margin_df.index, y=margin_df["margin_balance"],
            name="融資餘額", fill="tozeroy",
            line=dict(color=COLORS["chart_red"], width=1.5),
            fillcolor="rgba(233,87,63,0.1)",
        ), row=1, col=1)

    if "short_balance" in margin_df.columns:
        fig.add_trace(go.Scatter(
            x=margin_df.index, y=margin_df["short_balance"],
            name="融券餘額", fill="tozeroy",
            line=dict(color=COLORS["chart_blue"], width=1.5),
            fillcolor="rgba(59,175,218,0.1)",
        ), row=2, col=1)

    _dark_layout(fig, title=title, height=height)
    return fig
