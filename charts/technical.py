"""Technical analysis chart builders using Plotly."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, PLOTLY_LEGEND


def _apply_dark_layout(fig, title="", height=300, yaxis_range=None, show_legend=True,
                        legend_horizontal=True):
    """Apply consistent dark theme layout to a figure."""
    layout_kwargs = dict(
        paper_bgcolor=COLORS["bg_primary"],
        plot_bgcolor=COLORS["bg_primary"],
        font=dict(color=COLORS["text_primary"], family="Segoe UI, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=40),
        height=height,
        showlegend=show_legend,
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(color=COLORS["accent"]))
    if legend_horizontal:
        layout_kwargs["legend"] = dict(**PLOTLY_LEGEND, orientation="h", y=1.08, x=0)
    else:
        layout_kwargs["legend"] = dict(**PLOTLY_LEGEND)

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])

    yaxis_kw = dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"])
    if yaxis_range:
        yaxis_kw["range"] = yaxis_range
    fig.update_yaxes(**yaxis_kw)


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "",
    show_volume: bool = True,
    show_ma: list = None,
    show_bollinger: bool = False,
    show_psar: bool = False,
    show_ichimoku: bool = False,
    height: int = 600,
) -> go.Figure:
    """Create a candlestick chart with optional overlays."""
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=COLORS["positive"],
        decreasing_line_color=COLORS["negative"],
        increasing_fillcolor=COLORS["positive"],
        decreasing_fillcolor=COLORS["negative"],
        name="Price",
    ), row=1, col=1)

    # Moving Averages
    if show_ma is None:
        show_ma = [5, 20, 60]
    ma_colors = {
        5: COLORS["chart_yellow"],
        10: COLORS["chart_blue"],
        20: COLORS["chart_blue"],
        60: COLORS["chart_red"],
        120: COLORS["chart_purple"],
        240: "#ff9800",
    }
    for period in show_ma:
        col_name = f"SMA_{period}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name],
                name=f"SMA{period}",
                line=dict(color=ma_colors.get(period, COLORS["text_secondary"]), width=1),
            ), row=1, col=1)

    # Bollinger Bands
    if show_bollinger and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color=COLORS["chart_red"], dash="dash", width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color=COLORS["chart_green"], dash="dash", width=1),
            fill="tonexty", fillcolor="rgba(59,175,218,0.05)",
        ), row=1, col=1)

    # Parabolic SAR
    if show_psar:
        if "PSAR_Long" in df.columns:
            psar_long = df["PSAR_Long"].dropna()
            if not psar_long.empty:
                fig.add_trace(go.Scatter(
                    x=psar_long.index, y=psar_long,
                    name="SAR (Long)", mode="markers",
                    marker=dict(color=COLORS["positive"], size=3),
                ), row=1, col=1)
        if "PSAR_Short" in df.columns:
            psar_short = df["PSAR_Short"].dropna()
            if not psar_short.empty:
                fig.add_trace(go.Scatter(
                    x=psar_short.index, y=psar_short,
                    name="SAR (Short)", mode="markers",
                    marker=dict(color=COLORS["negative"], size=3),
                ), row=1, col=1)

    # Ichimoku Cloud
    if show_ichimoku:
        ichi_cols = [c for c in df.columns if c.startswith("Ichimoku_")]
        span_a = next((c for c in ichi_cols if "SpanA" in c), None)
        span_b = next((c for c in ichi_cols if "SpanB" in c), None)
        tenkan = next((c for c in ichi_cols if "Conversion" in c or "Tenkan" in c), None)
        kijun = next((c for c in ichi_cols if "Base" in c or "Kijun" in c), None)

        if tenkan and tenkan in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[tenkan], name="Tenkan-sen",
                line=dict(color=COLORS["chart_blue"], width=1),
            ), row=1, col=1)
        if kijun and kijun in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[kijun], name="Kijun-sen",
                line=dict(color=COLORS["chart_red"], width=1),
            ), row=1, col=1)
        if span_a and span_b and span_a in df.columns and span_b in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[span_a], name="Span A",
                line=dict(color="rgba(0,212,170,0.5)", width=0),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df[span_b], name="Span B",
                line=dict(color="rgba(233,87,63,0.5)", width=0),
                fill="tonexty", fillcolor="rgba(0,212,170,0.08)",
            ), row=1, col=1)

    # Volume
    if show_volume and "Volume" in df.columns:
        colors = [
            COLORS["positive"] if (i > 0 and df["Close"].iloc[i] >= df["Close"].iloc[i - 1])
            else COLORS["negative"]
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.5,
        ), row=2, col=1)

    _apply_dark_layout(fig, title=title, height=height)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def create_macd_chart(df: pd.DataFrame, height: int = 250) -> go.Figure:
    """Create MACD sub-chart."""
    fig = go.Figure()
    if "MACD_DIF" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_DIF"], name="DIF",
        line=dict(color=COLORS["chart_blue"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"], name="Signal",
        line=dict(color=COLORS["chart_red"], width=1.5),
    ))

    hist = df["MACD_Histogram"]
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in hist]
    fig.add_trace(go.Bar(
        x=df.index, y=hist, name="Histogram",
        marker_color=colors, opacity=0.6,
    ))

    fig.add_hline(y=0, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.5)
    _apply_dark_layout(fig, title="MACD (12, 26, 9)", height=height)
    return fig


def create_rsi_chart(df: pd.DataFrame, height: int = 220) -> go.Figure:
    """Create RSI chart."""
    fig = go.Figure()
    if "RSI_14" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI_14"], name="RSI(14)",
        line=dict(color=COLORS["chart_blue"], width=1.5),
    ))
    fig.add_hline(y=70, line_color=COLORS["negative"], line_dash="dash", opacity=0.5)
    fig.add_hline(y=30, line_color=COLORS["positive"], line_dash="dash", opacity=0.5)
    fig.add_hline(y=50, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.3)
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["negative"], opacity=0.05)
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["positive"], opacity=0.05)

    _apply_dark_layout(fig, title="RSI (14)", height=height, yaxis_range=[0, 100])
    return fig


def create_kd_chart(df: pd.DataFrame, height: int = 220) -> go.Figure:
    """Create KD / Stochastic chart."""
    fig = go.Figure()
    if "K" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["K"], name="K",
        line=dict(color=COLORS["chart_blue"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["D"], name="D",
        line=dict(color=COLORS["chart_red"], width=1.5),
    ))
    fig.add_hline(y=80, line_color=COLORS["negative"], line_dash="dash", opacity=0.5)
    fig.add_hline(y=20, line_color=COLORS["positive"], line_dash="dash", opacity=0.5)
    fig.add_hrect(y0=80, y1=100, fillcolor=COLORS["negative"], opacity=0.05)
    fig.add_hrect(y0=0, y1=20, fillcolor=COLORS["positive"], opacity=0.05)

    _apply_dark_layout(fig, title="KD / Stochastic (9, 3, 3)", height=height, yaxis_range=[0, 100])
    return fig


def create_adx_chart(df: pd.DataFrame, height: int = 220) -> go.Figure:
    """Create ADX chart."""
    fig = go.Figure()
    if "ADX" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ADX"], name="ADX",
        line=dict(color=COLORS["accent"], width=2),
    ))
    if "DI_Plus" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["DI_Plus"], name="+DI",
            line=dict(color=COLORS["positive"], width=1),
        ))
    if "DI_Minus" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["DI_Minus"], name="-DI",
            line=dict(color=COLORS["negative"], width=1),
        ))
    fig.add_hline(y=25, line_color=COLORS["warning"], line_dash="dash", opacity=0.5,
                  annotation_text="Trend threshold (25)")

    _apply_dark_layout(fig, title="ADX (14)", height=height)
    return fig


def create_obv_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create OBV chart."""
    fig = go.Figure()
    if "OBV" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["OBV"], name="OBV",
        line=dict(color=COLORS["chart_blue"], width=1.5),
        fill="tozeroy", fillcolor="rgba(59,175,218,0.1)",
    ))
    _apply_dark_layout(fig, title="OBV (On-Balance Volume)", height=height)
    return fig


def create_williams_r_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create Williams %R chart."""
    fig = go.Figure()
    if "Williams_R" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Williams_R"], name="%R",
        line=dict(color=COLORS["chart_blue"], width=1.5),
    ))
    fig.add_hline(y=-20, line_color=COLORS["negative"], line_dash="dash", opacity=0.5)
    fig.add_hline(y=-80, line_color=COLORS["positive"], line_dash="dash", opacity=0.5)

    _apply_dark_layout(fig, title="Williams %R (14)", height=height, yaxis_range=[-100, 0])
    return fig


def create_cci_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create CCI chart."""
    fig = go.Figure()
    if "CCI_20" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["CCI_20"], name="CCI(20)",
        line=dict(color=COLORS["chart_blue"], width=1.5),
    ))
    fig.add_hline(y=100, line_color=COLORS["negative"], line_dash="dash", opacity=0.5)
    fig.add_hline(y=-100, line_color=COLORS["positive"], line_dash="dash", opacity=0.5)
    fig.add_hline(y=0, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.3)

    _apply_dark_layout(fig, title="CCI (20)", height=height)
    return fig


def create_atr_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create ATR chart."""
    fig = go.Figure()
    if "ATR_14" not in df.columns:
        return fig

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ATR_14"], name="ATR(14)",
        line=dict(color=COLORS["chart_yellow"], width=1.5),
        fill="tozeroy", fillcolor="rgba(246,187,66,0.1)",
    ))
    _apply_dark_layout(fig, title="ATR (14) - Average True Range", height=height)
    return fig
