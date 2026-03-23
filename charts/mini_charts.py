"""Mini indicator charts for score card expanders."""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COLORS, PLOTLY_LEGEND

_H = 200  # default mini chart height


def _mini_layout(fig, title="", height=_H):
    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"],
        plot_bgcolor=COLORS["bg_primary"],
        font=dict(color=COLORS["text_primary"], size=10),
        margin=dict(l=40, r=20, t=30, b=25),
        height=height,
        title=dict(text=title, font=dict(color=COLORS["accent"], size=11)) if title else None,
        legend=dict(bgcolor=PLOTLY_LEGEND["bgcolor"], bordercolor=PLOTLY_LEGEND["bordercolor"],
                    font=dict(color=COLORS["text_primary"], size=9),
                    orientation="h", y=1.15, x=0),
        showlegend=True,
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
                     tickfont=dict(size=9))
    fig.update_yaxes(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"],
                     tickfont=dict(size=9))


def mini_macd(df: pd.DataFrame) -> go.Figure:
    """Mini MACD chart with current value annotated."""
    fig = go.Figure()
    if "MACD_DIF" not in df.columns:
        return fig

    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_DIF"], name="DIF",
                             line=dict(color=COLORS["chart_blue"], width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_Signal"], name="Signal",
                             line=dict(color=COLORS["chart_red"], width=1.5)))
    hist = d["MACD_Histogram"]
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in hist]
    fig.add_trace(go.Bar(x=d.index, y=hist, name="Histogram",
                         marker_color=colors, opacity=0.5))
    fig.add_hline(y=0, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.3)

    # Annotate latest value
    latest = d["MACD_DIF"].iloc[-1]
    fig.add_annotation(x=d.index[-1], y=latest,
                       text=f"DIF: {latest:.2f}", showarrow=True,
                       arrowcolor=COLORS["accent"], font=dict(color=COLORS["accent"], size=10),
                       bgcolor=COLORS["bg_card"], bordercolor=COLORS["accent"])
    _mini_layout(fig, "MACD (近60日)")
    return fig


def mini_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI_14" not in df.columns:
        return fig

    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI_14"], name="RSI(14)",
                             line=dict(color=COLORS["chart_blue"], width=1.5)))
    fig.add_hrect(y0=70, y1=100, fillcolor=COLORS["negative"], opacity=0.08,
                  annotation_text="超買", annotation_position="top left",
                  annotation_font_size=9, annotation_font_color=COLORS["negative"])
    fig.add_hrect(y0=0, y1=30, fillcolor=COLORS["positive"], opacity=0.08,
                  annotation_text="超賣", annotation_position="bottom left",
                  annotation_font_size=9, annotation_font_color=COLORS["positive"])
    fig.add_hline(y=70, line_color=COLORS["negative"], line_dash="dash", opacity=0.4)
    fig.add_hline(y=30, line_color=COLORS["positive"], line_dash="dash", opacity=0.4)
    fig.add_hline(y=50, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.2)

    latest = d["RSI_14"].iloc[-1]
    clr = COLORS["positive"] if latest < 30 else COLORS["negative"] if latest > 70 else COLORS["accent"]
    fig.add_annotation(x=d.index[-1], y=latest,
                       text=f"RSI: {latest:.1f}", showarrow=True,
                       arrowcolor=clr, font=dict(color=clr, size=10),
                       bgcolor=COLORS["bg_card"], bordercolor=clr)
    _mini_layout(fig, "RSI (近60日)")
    fig.update_yaxes(range=[0, 100])
    return fig


def mini_kd(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "K" not in df.columns:
        return fig

    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["K"], name="K",
                             line=dict(color=COLORS["chart_blue"], width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d["D"], name="D",
                             line=dict(color=COLORS["chart_red"], width=1.5)))
    fig.add_hrect(y0=80, y1=100, fillcolor=COLORS["negative"], opacity=0.08)
    fig.add_hrect(y0=0, y1=20, fillcolor=COLORS["positive"], opacity=0.08)
    fig.add_hline(y=80, line_color=COLORS["negative"], line_dash="dash", opacity=0.4)
    fig.add_hline(y=20, line_color=COLORS["positive"], line_dash="dash", opacity=0.4)

    k_val = d["K"].iloc[-1]
    d_val = d["D"].iloc[-1]
    fig.add_annotation(x=d.index[-1], y=k_val,
                       text=f"K:{k_val:.0f} D:{d_val:.0f}", showarrow=True,
                       arrowcolor=COLORS["accent"], font=dict(color=COLORS["accent"], size=10),
                       bgcolor=COLORS["bg_card"], bordercolor=COLORS["accent"])
    _mini_layout(fig, "KD (近60日)")
    fig.update_yaxes(range=[0, 100])
    return fig


def mini_bollinger(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "BB_Upper" not in df.columns:
        return fig

    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Price",
                             line=dict(color=COLORS["text_primary"], width=1.5)))
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_Upper"], name="Upper",
                             line=dict(color=COLORS["chart_red"], dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_Mid"], name="Mid",
                             line=dict(color=COLORS["chart_blue"], dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=d.index, y=d["BB_Lower"], name="Lower",
                             line=dict(color=COLORS["chart_green"], dash="dash", width=1),
                             fill="tonexty", fillcolor="rgba(59,175,218,0.04)"))

    pct = d["BB_Percent"].iloc[-1] if "BB_Percent" in d.columns else None
    label = f"%B: {pct:.0%}" if pct is not None and not pd.isna(pct) else ""
    fig.add_annotation(x=d.index[-1], y=d["Close"].iloc[-1],
                       text=label, showarrow=True,
                       arrowcolor=COLORS["accent"], font=dict(color=COLORS["accent"], size=10),
                       bgcolor=COLORS["bg_card"], bordercolor=COLORS["accent"])
    _mini_layout(fig, "Bollinger Bands (近60日)")
    return fig


def mini_ma(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    n = min(90, len(df))
    d = df.tail(n)

    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Price",
                             line=dict(color=COLORS["text_primary"], width=1.5)))
    ma_cols = {"SMA_5": COLORS["chart_yellow"], "SMA_20": COLORS["chart_blue"],
               "SMA_60": COLORS["chart_red"]}
    for col, color in ma_cols.items():
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col.replace("_", ""),
                                     line=dict(color=color, width=1)))
    _mini_layout(fig, "Price + MA (近90日)")
    return fig


def mini_volume(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    n = min(60, len(df))
    d = df.tail(n)

    colors = [COLORS["positive"] if (i > 0 and d["Close"].iloc[i] >= d["Close"].iloc[i-1])
              else COLORS["negative"] for i in range(len(d))]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                         marker_color=colors, opacity=0.6))
    if "Volume_MA_5" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["Volume_MA_5"], name="MA5",
                                 line=dict(color=COLORS["chart_yellow"], width=1)))
    if "Volume_MA_20" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["Volume_MA_20"], name="MA20",
                                 line=dict(color=COLORS["chart_blue"], width=1)))
    _mini_layout(fig, "Volume (近60日)")
    return fig


def mini_adx(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "ADX" not in df.columns:
        return fig

    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["ADX"], name="ADX",
                             line=dict(color=COLORS["accent"], width=2)))
    if "DI_Plus" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["DI_Plus"], name="+DI",
                                 line=dict(color=COLORS["positive"], width=1)))
    if "DI_Minus" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["DI_Minus"], name="-DI",
                                 line=dict(color=COLORS["negative"], width=1)))
    fig.add_hline(y=25, line_color=COLORS["warning"], line_dash="dash", opacity=0.4,
                  annotation_text="趨勢閾值", annotation_font_size=9)

    adx_val = d["ADX"].iloc[-1]
    fig.add_annotation(x=d.index[-1], y=adx_val,
                       text=f"ADX: {adx_val:.0f}", showarrow=True,
                       arrowcolor=COLORS["accent"], font=dict(color=COLORS["accent"], size=10),
                       bgcolor=COLORS["bg_card"], bordercolor=COLORS["accent"])
    _mini_layout(fig, "ADX (近60日)")
    return fig


def mini_williams_r(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "Williams_R" not in df.columns:
        return fig

    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["Williams_R"], name="%R",
                             line=dict(color=COLORS["chart_blue"], width=1.5)))
    fig.add_hline(y=-20, line_color=COLORS["negative"], line_dash="dash", opacity=0.4)
    fig.add_hline(y=-80, line_color=COLORS["positive"], line_dash="dash", opacity=0.4)
    fig.add_hrect(y0=-20, y1=0, fillcolor=COLORS["negative"], opacity=0.06)
    fig.add_hrect(y0=-100, y1=-80, fillcolor=COLORS["positive"], opacity=0.06)
    _mini_layout(fig, "Williams %R (近60日)")
    fig.update_yaxes(range=[-100, 0])
    return fig


def mini_cci(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "CCI_20" not in df.columns:
        return fig

    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["CCI_20"], name="CCI(20)",
                             line=dict(color=COLORS["chart_blue"], width=1.5)))
    fig.add_hline(y=100, line_color=COLORS["negative"], line_dash="dash", opacity=0.4)
    fig.add_hline(y=-100, line_color=COLORS["positive"], line_dash="dash", opacity=0.4)
    fig.add_hline(y=0, line_color=COLORS["text_secondary"], line_dash="dot", opacity=0.2)
    _mini_layout(fig, "CCI (近60日)")
    return fig


def mini_psar(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    n = min(60, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Price",
                             line=dict(color=COLORS["text_primary"], width=1.5)))
    if "PSAR_Long" in d.columns:
        pl = d["PSAR_Long"].dropna()
        if not pl.empty:
            fig.add_trace(go.Scatter(x=pl.index, y=pl, name="SAR(多)", mode="markers",
                                     marker=dict(color=COLORS["positive"], size=3)))
    if "PSAR_Short" in d.columns:
        ps = d["PSAR_Short"].dropna()
        if not ps.empty:
            fig.add_trace(go.Scatter(x=ps.index, y=ps, name="SAR(空)", mode="markers",
                                     marker=dict(color=COLORS["negative"], size=3)))
    _mini_layout(fig, "Parabolic SAR (近60日)")
    return fig


# --- Fundamental mini charts ---

def mini_pe_history(df: pd.DataFrame) -> go.Figure:
    """Show price trend as a proxy - P/E is point-in-time."""
    fig = go.Figure()
    n = min(120, len(df))
    d = df.tail(n)
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Price",
                             line=dict(color=COLORS["accent"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(0,212,170,0.05)"))
    _mini_layout(fig, "Price Trend (P/E 參考)")
    return fig


# --- Map indicator name to mini chart function ---

MINI_CHART_MAP = {
    "MACD": mini_macd,
    "RSI": mini_rsi,
    "KD": mini_kd,
    "Bollinger": mini_bollinger,
    "MA (均線)": mini_ma,
    "Volume": mini_volume,
    "ADX": mini_adx,
    "Williams %R": mini_williams_r,
    "CCI": mini_cci,
    "Parabolic SAR": mini_psar,
}
