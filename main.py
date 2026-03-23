"""
Stock Analysis Platform | 股票分析平台
Supports Taiwan (TWSE) and US stocks with 40+ indicators.
Run: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Ensure app directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    COLORS, MARKET_INDICES, DEFAULT_WEIGHTS,
    TECHNICAL_INDICATORS, FUNDAMENTAL_INDICATORS,
    CHIP_INDICATORS_TW, QUANTITATIVE_INDICATORS,
    INDICATOR_PRESETS, get_market, get_default_index, score_label,
)
from data.fetcher import fetch_price_data, fetch_stock_info, get_fundamental_metrics
from data.cache import DataStore
from indicators.technical import add_all_technical_indicators, get_latest_signals
from indicators.fundamental import analyze_fundamentals
from indicators.chip import analyze_institutional, analyze_margin, calculate_quantitative_metrics
from scoring.composite import compute_all_scores, compute_composite, generate_highlights_risks
from charts import technical as tech_charts
from charts import fundamental as fund_charts
from charts import chip as chip_charts
from charts import comparison as comp_charts
from charts import backtest as bt_charts
from charts.mini_charts import MINI_CHART_MAP
from utils.constants import (
    TW_STOCK_EXAMPLES, US_STOCK_EXAMPLES, INDICATOR_DISPLAY_NAMES,
)
from utils.helpers import run_backtest_simple, run_backtest_advanced, compute_daily_scores
from utils.batch_analysis import batch_analyze, export_to_excel
from utils.momentum import screen_momentum, export_momentum_excel, TW_TOP50, US_TOP50

# ============================================================
# Page Config & Custom CSS
# ============================================================
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown(f"""
<style>
    /* Main background */
    .stApp {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
    }}
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['bg_card']};
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {COLORS['text_primary']};
    }}
    /* Cards */
    .score-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        transition: border-color 0.3s;
    }}
    .score-card:hover {{
        border-color: {COLORS['accent']};
    }}
    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px;
    }}
    /* Tabs */
    button[data-baseweb="tab"] {{
        color: {COLORS['text_secondary']};
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLORS['accent']};
    }}
    /* Score bar */
    .score-bar-outer {{
        background: {COLORS['border']};
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }}
    .score-bar-inner {{
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s;
    }}
    /* Highlight boxes */
    .highlight-box {{
        background: rgba(0,212,170,0.08);
        border-left: 4px solid {COLORS['accent']};
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }}
    .risk-box {{
        background: rgba(255,82,82,0.08);
        border-left: 4px solid {COLORS['negative']};
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }}
    /* Expander */
    details {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
    }}
    /* Disclaimer */
    .disclaimer {{
        background: rgba(255,215,64,0.1);
        border: 1px solid {COLORS['warning']};
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.85em;
        color: {COLORS['warning']};
    }}
    /* Global text contrast fix */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {COLORS['text_primary']};
    }}
    .stMarkdown, .stMarkdown p {{
        color: {COLORS['text_primary']};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['text_primary']} !important;
    }}
    /* Sidebar labels */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {{
        color: {COLORS['text_primary']} !important;
    }}
    /* Metric values */
    div[data-testid="stMetric"] label {{
        color: {COLORS['text_secondary']} !important;
    }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
        color: {COLORS['text_primary']} !important;
    }}
    /* Table text */
    .stDataFrame td, .stDataFrame th {{
        color: {COLORS['text_primary']} !important;
    }}
    /* Tabs */
    div[data-testid="stTabs"] button p {{
        color: inherit !important;
    }}
    /* Input fields - dark bg + white text */
    input, textarea, select {{
        color: {COLORS['text_primary']} !important;
        background-color: {COLORS['bg_card']} !important;
        border: 1px solid {COLORS['border']} !important;
    }}
    /* Streamlit input wrapper */
    div[data-baseweb="input"] {{
        background-color: {COLORS['bg_card']} !important;
    }}
    div[data-baseweb="input"] input {{
        color: {COLORS['text_primary']} !important;
        -webkit-text-fill-color: {COLORS['text_primary']} !important;
    }}
    /* Select / Dropdown */
    div[data-baseweb="select"] {{
        background-color: {COLORS['bg_card']} !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {COLORS['bg_card']} !important;
        color: {COLORS['text_primary']} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {COLORS['text_primary']} !important;
    }}
    /* Dropdown menu */
    ul[role="listbox"] {{
        background-color: {COLORS['bg_card']} !important;
    }}
    ul[role="listbox"] li {{
        color: {COLORS['text_primary']} !important;
    }}
    ul[role="listbox"] li:hover {{
        background-color: {COLORS['bg_card_hover']} !important;
    }}
    /* Number input */
    div[data-baseweb="input"] input[type="number"] {{
        color: {COLORS['text_primary']} !important;
        -webkit-text-fill-color: {COLORS['text_primary']} !important;
    }}
    /* Date input */
    div[data-testid="stDateInput"] input {{
        color: {COLORS['text_primary']} !important;
        -webkit-text-fill-color: {COLORS['text_primary']} !important;
        background-color: {COLORS['bg_card']} !important;
    }}
    /* Button text fix */
    button[kind="primary"] {{
        color: {COLORS['bg_primary']} !important;
    }}
    button {{
        color: {COLORS['text_primary']} !important;
    }}
    /* Multiselect tags */
    span[data-baseweb="tag"] {{
        background-color: {COLORS['accent']} !important;
        color: {COLORS['bg_primary']} !important;
    }}
    span[data-baseweb="tag"] span {{
        color: {COLORS['bg_primary']} !important;
    }}
    /* Checkbox label */
    div[data-testid="stCheckbox"] label span {{
        color: {COLORS['text_primary']} !important;
    }}
    /* Indicator description card */
    .indicator-desc {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 12px 0;
        color: {COLORS['text_primary']};
        font-size: 0.9em;
        line-height: 1.6;
    }}
    .indicator-desc strong {{
        color: {COLORS['accent']};
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(f"## 📊 Stock Analysis Platform")
    st.markdown("---")

    # Market selection
    market_choice = st.radio("Market 市場", ["台股 (TW)", "美股 (US)"], horizontal=True)
    is_tw = market_choice.startswith("台股")

    # Ticker input
    examples = TW_STOCK_EXAMPLES if is_tw else US_STOCK_EXAMPLES
    example_str = ", ".join(list(examples.keys())[:5])
    ticker_input = st.text_input(
        "Stock Ticker 股票代號",
        value="2330.TW" if is_tw else "AAPL",
        help=f"Examples: {example_str}",
    )

    # Auto-add .TW suffix
    ticker = ticker_input.strip()
    if is_tw and not ticker.endswith((".TW", ".TWO")) and ticker.isdigit():
        ticker = f"{ticker}.TW"

    st.markdown("---")

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start 起始日", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End 結束日", value=datetime.now())

    st.markdown("---")

    # Benchmark indices
    st.markdown("**Benchmark 大盤比較**")
    default_idx = get_default_index(ticker)
    selected_indices = {}
    for name, idx_ticker in MARKET_INDICES.items():
        default_on = (idx_ticker == default_idx)
        selected_indices[name] = st.checkbox(name, value=default_on)

    st.markdown("---")

    # Indicator presets
    st.markdown("**Indicators 指標選擇**")
    preset = st.selectbox("Preset 預設組合", list(INDICATOR_PRESETS.keys()))

    if preset == "自訂":
        st.markdown("*Technical 技術面:*")
        enabled_tech = []
        for ind in TECHNICAL_INDICATORS:
            if st.checkbox(INDICATOR_DISPLAY_NAMES.get(ind, ind), value=True, key=f"tech_{ind}"):
                enabled_tech.append(ind)

        st.markdown("*Fundamental 基本面:*")
        enabled_fund = []
        for ind in FUNDAMENTAL_INDICATORS:
            if st.checkbox(INDICATOR_DISPLAY_NAMES.get(ind, ind), value=True, key=f"fund_{ind}"):
                enabled_fund.append(ind)

        st.markdown("*Chip 籌碼面:*")
        chip_list = CHIP_INDICATORS_TW if is_tw else QUANTITATIVE_INDICATORS
        enabled_chip = []
        for ind in chip_list:
            if st.checkbox(INDICATOR_DISPLAY_NAMES.get(ind, ind), value=True, key=f"chip_{ind}"):
                enabled_chip.append(ind)

        enabled_indicators = enabled_tech + enabled_fund + enabled_chip
    elif preset == "全部":
        enabled_indicators = None  # means all
    else:
        enabled_indicators = INDICATOR_PRESETS.get(preset)

    st.markdown("---")

    # Scoring weights
    st.markdown("**Scoring Weights 評分權重**")
    w_tech = st.slider("Technical 技術面 %", 0, 100, 40, key="w_tech")
    w_fund = st.slider("Fundamental 基本面 %", 0, 100, 35, key="w_fund")
    w_chip = st.slider("Chip 籌碼面 %", 0, 100, 25, key="w_chip")
    total_w = w_tech + w_fund + w_chip
    if total_w == 0:
        total_w = 100
    weights = {
        "technical": w_tech / total_w,
        "fundamental": w_fund / total_w,
        "chip": w_chip / total_w,
    }
    st.caption(f"Normalized: Tech {weights['technical']*100:.0f}% / Fund {weights['fundamental']*100:.0f}% / Chip {weights['chip']*100:.0f}%")

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze 開始分析", use_container_width=True, type="primary")


# ============================================================
# Main Content
# ============================================================

if analyze_btn or "analyzed" in st.session_state:
    st.session_state["analyzed"] = True
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Fetch data
    with st.spinner(f"Fetching data for {ticker}..."):
        price_df = DataStore.get_price_data(ticker, start_str, end_str)
        stock_info = DataStore.get_stock_info(ticker)
        fundamentals_raw = get_fundamental_metrics(stock_info)

    if price_df.empty:
        st.error(f"❌ Unable to fetch data for {ticker}. Please check the ticker symbol.")
        st.stop()

    # Stock header
    stock_name = fundamentals_raw.get("Short_Name", ticker)
    latest_close = price_df["Close"].iloc[-1]
    prev_close = price_df["Close"].iloc[-2] if len(price_df) > 1 else latest_close
    change = latest_close - prev_close
    change_pct = change / prev_close * 100

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.markdown(f"### {stock_name} ({ticker})")
    with col2:
        st.metric("Price", f"${latest_close:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    with col3:
        st.metric("52W High", f"${fundamentals_raw.get('52W_High', 'N/A')}")
    with col4:
        st.metric("52W Low", f"${fundamentals_raw.get('52W_Low', 'N/A')}")

    # Compute indicators
    with st.spinner("Computing indicators..."):
        df_indicators = add_all_technical_indicators(price_df)
        tech_signals = get_latest_signals(df_indicators)
        fund_analysis = analyze_fundamentals(fundamentals_raw)

        # Chip data (TW only)
        inst_data = {}
        margin_data = {}
        market = get_market(ticker)
        if market in ("TW", "TWO"):
            # Only fetch last 20 trading days for chip data to avoid rate limiting
            chip_start = (end_date - timedelta(days=35)).strftime("%Y-%m-%d")
            inst_df = DataStore.get_institutional_data(ticker, chip_start, end_str)
            margin_df = DataStore.get_margin_data(ticker, chip_start, end_str)
            inst_data = analyze_institutional(inst_df)
            margin_data = analyze_margin(margin_df)
        else:
            inst_df = pd.DataFrame()
            margin_df = pd.DataFrame()

        # Quantitative metrics
        bench_ticker = get_default_index(ticker)
        bench_df = DataStore.get_price_data(bench_ticker, start_str, end_str)
        quant_data = calculate_quantitative_metrics(price_df, bench_df)

        # Compute scores
        all_scores = compute_all_scores(
            tech_signals, fund_analysis, inst_data, margin_data, quant_data,
            enabled_indicators,
        )
        composite_score = compute_composite(all_scores, weights)
        label, label_color = score_label(composite_score)
        highlights, risks = generate_highlights_risks(all_scores)

    # ============================================================
    # Backtest result renderer (shared by all modes)
    # ============================================================
    def _render_backtest_results(st_mod, bt_result, df_ind, strat_info, colors):
        """Render all backtest output: indicator chart, equity, metrics, trades."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from config import PLOTLY_LEGEND

        prices = bt_result.get("price_series", pd.Series())
        buys = bt_result.get("buy_signals", [])
        sells = bt_result.get("sell_signals", [])
        equity = bt_result.get("equity_curve", pd.Series())

        if prices.empty and equity.empty:
            st_mod.warning("No results to display.")
            return

        # --- 1. Price + Indicator + Buy/Sell overlay chart ---
        overlay_cols = strat_info.get("overlay_indicators", []) if strat_info else []
        sub_cols = strat_info.get("sub_indicators", []) if strat_info else []
        has_sub = bool(sub_cols)

        n_rows = 2 + (1 if has_sub else 0)
        heights = [0.5, 0.25, 0.25] if has_sub else [0.65, 0.35]
        subtitles = ["Price + Signals"]
        if has_sub:
            sub_label = ", ".join([c.replace("MACD_", "").replace("_14", "") for c in sub_cols[:3]])
            subtitles.append(sub_label)
        subtitles.append("Equity Curve vs Buy & Hold")

        fig_combo = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                                   vertical_spacing=0.04, row_heights=heights,
                                   subplot_titles=subtitles)

        # Price
        n = min(len(df_ind), len(prices))
        d = df_ind.tail(n) if len(df_ind) >= n else df_ind
        fig_combo.add_trace(go.Scatter(
            x=d.index, y=d["Close"], name="Price",
            line=dict(color=colors["text_primary"], width=1.5),
        ), row=1, col=1)

        # Overlay indicators (SMA, Bollinger, Donchian...)
        overlay_colors = [colors["chart_blue"], colors["chart_red"],
                         colors["chart_yellow"], colors["chart_green"]]
        for j, col in enumerate(overlay_cols):
            if col in d.columns:
                fig_combo.add_trace(go.Scatter(
                    x=d.index, y=d[col], name=col.replace("_", " "),
                    line=dict(color=overlay_colors[j % len(overlay_colors)], width=1, dash="dash"),
                ), row=1, col=1)

        # Buy/Sell markers
        if buys:
            bd, bp = zip(*buys)
            fig_combo.add_trace(go.Scatter(
                x=list(bd), y=list(bp), mode="markers", name="BUY",
                marker=dict(color=colors["positive"], size=12, symbol="triangle-up",
                           line=dict(width=1, color="white")),
            ), row=1, col=1)
        if sells:
            sd, sp = zip(*sells)
            fig_combo.add_trace(go.Scatter(
                x=list(sd), y=list(sp), mode="markers", name="SELL",
                marker=dict(color=colors["negative"], size=12, symbol="triangle-down",
                           line=dict(width=1, color="white")),
            ), row=1, col=1)

        # Sub indicators (MACD, RSI, KD...)
        sub_row = 2 if has_sub else None
        if has_sub:
            sub_chart_colors = [colors["chart_blue"], colors["chart_red"],
                               colors["accent"], colors["chart_yellow"]]
            for j, col in enumerate(sub_cols):
                if col in d.columns:
                    if "Histogram" in col:
                        hist_vals = d[col]
                        h_colors = [colors["positive"] if v >= 0 else colors["negative"] for v in hist_vals]
                        fig_combo.add_trace(go.Bar(
                            x=d.index, y=hist_vals, name=col, marker_color=h_colors, opacity=0.5,
                        ), row=sub_row, col=1)
                    else:
                        fig_combo.add_trace(go.Scatter(
                            x=d.index, y=d[col], name=col.replace("MACD_", "").replace("_14", ""),
                            line=dict(color=sub_chart_colors[j % len(sub_chart_colors)], width=1.5),
                        ), row=sub_row, col=1)
            # Add reference lines for RSI/KD
            if any("RSI" in c for c in sub_cols):
                fig_combo.add_hline(y=70, line_dash="dash", line_color=colors["negative"],
                                   opacity=0.3, row=sub_row, col=1)
                fig_combo.add_hline(y=30, line_dash="dash", line_color=colors["positive"],
                                   opacity=0.3, row=sub_row, col=1)
            if any(c in ("K", "D") for c in sub_cols):
                fig_combo.add_hline(y=80, line_dash="dash", line_color=colors["negative"],
                                   opacity=0.3, row=sub_row, col=1)
                fig_combo.add_hline(y=20, line_dash="dash", line_color=colors["positive"],
                                   opacity=0.3, row=sub_row, col=1)
            if any("MACD" in c for c in sub_cols):
                fig_combo.add_hline(y=0, line_dash="dot", line_color=colors["text_secondary"],
                                   opacity=0.3, row=sub_row, col=1)

        # Equity curve
        eq_row = n_rows
        if not equity.empty:
            fig_combo.add_trace(go.Scatter(
                x=equity.index, y=equity.values, name="Portfolio",
                line=dict(color=colors["accent"], width=2),
            ), row=eq_row, col=1)
        bench = bt_result.get("benchmark_curve", pd.Series())
        if not bench.empty:
            fig_combo.add_trace(go.Scatter(
                x=bench.index, y=bench.values, name="Buy & Hold",
                line=dict(color=colors["text_secondary"], dash="dash", width=1),
            ), row=eq_row, col=1)

        fig_combo.update_layout(
            paper_bgcolor=colors["bg_primary"], plot_bgcolor=colors["bg_primary"],
            font=dict(color=colors["text_primary"]),
            height=250 * n_rows,
            legend=dict(**PLOTLY_LEGEND, orientation="h", y=1.02, x=0),
            margin=dict(l=60, r=30, t=60, b=30),
            xaxis_rangeslider_visible=False,
        )
        fig_combo.update_xaxes(gridcolor=colors["border"], zerolinecolor=colors["border"])
        fig_combo.update_yaxes(gridcolor=colors["border"], zerolinecolor=colors["border"])
        st_mod.plotly_chart(fig_combo, use_container_width=True)

        # --- 2. Drawdown chart ---
        if not equity.empty:
            fig_dd = bt_charts.create_drawdown_chart(equity)
            st_mod.plotly_chart(fig_dd, use_container_width=True)

        # --- 3. Performance Metrics ---
        metrics = bt_result.get("metrics", {})
        if metrics:
            st_mod.markdown("#### Performance Metrics 績效指標")
            m1, m2, m3, m4 = st_mod.columns(4)
            with m1: st_mod.metric("Total Return 總報酬", f"{metrics.get('total_return', 0)}%")
            with m2: st_mod.metric("CAGR 年化報酬", f"{metrics.get('cagr', 0)}%")
            with m3: st_mod.metric("Max Drawdown 最大回撤", f"{metrics.get('max_drawdown', 0)}%")
            with m4: st_mod.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0)}")

            m5, m6, m7, m8 = st_mod.columns(4)
            with m5: st_mod.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0)}")
            with m6: st_mod.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0)}")
            with m7: st_mod.metric("Win Rate 勝率", f"{metrics.get('win_rate', 0)}%")
            with m8: st_mod.metric("Profit Factor", f"{metrics.get('profit_factor', 0)}")

            m9, m10, m11, m12 = st_mod.columns(4)
            with m9: st_mod.metric("Avg Win 平均獲利", f"{metrics.get('avg_win', 0)}%")
            with m10: st_mod.metric("Avg Loss 平均虧損", f"-{metrics.get('avg_loss', 0)}%")
            with m11: st_mod.metric("Expectancy 期望值", f"{metrics.get('expectancy', 0)}%")
            with m12: st_mod.metric("Trades 交易次數", f"{metrics.get('total_trades', 0)}")

            mx1, mx2, mx3, _ = st_mod.columns(4)
            with mx1: st_mod.metric("Benchmark Return", f"{metrics.get('benchmark_return', 0)}%")
            with mx2: st_mod.metric("Exposure 持倉率", f"{metrics.get('exposure_pct', 0)}%")
            with mx3: st_mod.metric("Final Value 最終資金", f"${metrics.get('final_value', 0):,.0f}")

            # Result interpretation
            with st_mod.expander("📊 Result Interpretation 結果解讀"):
                interp = []
                tr = metrics.get("total_return", 0)
                br = metrics.get("benchmark_return", 0)
                if tr > br:
                    interp.append(f"✅ 策略報酬 ({tr}%) **優於** Buy & Hold ({br}%)，策略有附加價值")
                else:
                    interp.append(f"❌ 策略報酬 ({tr}%) **劣於** Buy & Hold ({br}%)，直接持有更好")

                sr = metrics.get("sharpe_ratio", 0)
                if sr > 1.5: interp.append(f"✅ Sharpe {sr} > 1.5，風險調整後報酬**優秀**")
                elif sr > 1.0: interp.append(f"✅ Sharpe {sr} > 1.0，風險調整後報酬**良好**")
                elif sr > 0.5: interp.append(f"⚠️ Sharpe {sr}，報酬勉強補償風險")
                else: interp.append(f"❌ Sharpe {sr} < 0.5，報酬**不足以補償風險**")

                md = metrics.get("max_drawdown", 0)
                if md > -15: interp.append(f"✅ 最大回撤 {md}%，風險控制良好")
                elif md > -30: interp.append(f"⚠️ 最大回撤 {md}%，中等風險")
                else: interp.append(f"❌ 最大回撤 {md}%，**風險很高**")

                nt = metrics.get("total_trades", 0)
                if nt < 5: interp.append(f"⚠️ 僅 {nt} 筆交易，統計意義不足，結果可能不可靠")
                elif nt < 30: interp.append(f"⚠️ {nt} 筆交易，樣本偏少")
                else: interp.append(f"✅ {nt} 筆交易，樣本數足夠")

                pf = metrics.get("profit_factor", 0)
                if isinstance(pf, (int, float)) and pf > 1.5:
                    interp.append(f"✅ Profit Factor {pf}，獲利是虧損的 {pf} 倍")

                for line in interp:
                    st_mod.markdown(line)

        # --- 4. Trade Log ---
        if not bt_result["trades"].empty:
            st_mod.markdown("#### Trade Log 交易紀錄")
            st_mod.dataframe(bt_result["trades"], use_container_width=True, hide_index=True)

    # ============================================================
    # Tabs
    # ============================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Dashboard", "📈 Technical", "📋 Fundamental",
        "🏦 Chip Analysis", "📉 Market Compare", "🔄 Backtest",
        "📋 Batch Analysis", "🚀 Momentum",
    ])

    # ==================== TAB 1: DASHBOARD ====================
    with tab1:
        # Composite Score
        st.markdown("### Composite Score 綜合評分")

        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.markdown(f"""
            <div style="text-align:center; padding:20px;">
                <div style="font-size:4em; font-weight:bold; color:{label_color};">
                    {composite_score:.1f}
                </div>
                <div style="font-size:1.2em; color:{label_color};">/ 10 — {label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Score bar
            pct = composite_score / 10 * 100
            st.markdown(f"""
            <div class="score-bar-outer">
                <div class="score-bar-inner" style="width:{pct}%; background:{label_color};"></div>
            </div>
            """, unsafe_allow_html=True)

        with score_col2:
            # Category averages
            categories = [
                ("Technical 技術面", all_scores["technical_avg"], weights["technical"]),
                ("Fundamental 基本面", all_scores["fundamental_avg"], weights["fundamental"]),
                ("Chip 籌碼面", all_scores["chip_avg"], weights["chip"]),
            ]
            for cat_name, avg, weight in categories:
                if avg is not None:
                    cat_label, cat_color = score_label(avg)
                    w_pct = weight * 100
                    bar_w = avg / 10 * 100
                    st.markdown(f"""
                    <div style="margin:8px 0;">
                        <span style="color:{COLORS['text_primary']}">{cat_name}</span>
                        <span style="color:{cat_color}; float:right;">{avg:.1f}/10 ({w_pct:.0f}% weight)</span>
                        <div class="score-bar-outer" style="margin-top:4px;">
                            <div class="score-bar-inner" style="width:{bar_w}%; background:{cat_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Scoring formula explanation
        with st.expander("📖 Scoring Formula 評分公式"):
            st.markdown(f"""
            **綜合評分 = Technical Avg × {weights['technical']*100:.0f}% + Fundamental Avg × {weights['fundamental']*100:.0f}% + Chip Avg × {weights['chip']*100:.0f}%**

            = {all_scores.get('technical_avg', 0) or 0:.2f} × {weights['technical']:.2f} + {all_scores.get('fundamental_avg', 0) or 0:.2f} × {weights['fundamental']:.2f} + {all_scores.get('chip_avg', 0) or 0:.2f} × {weights['chip']:.2f} = **{composite_score:.2f}**

            | Score Range | Signal | Action |
            |-------------|--------|--------|
            | 8.5 - 10 | 強烈買進 Strong Buy | 積極建倉 |
            | 7.0 - 8.4 | 偏多買進 Buy | 分批買進 |
            | 5.5 - 6.9 | 中性觀望 Neutral | 持有觀望 |
            | 3.5 - 5.4 | 偏空賣出 Sell | 減碼 |
            | 1.0 - 3.4 | 強烈賣出 Strong Sell | 清倉 |
            """)

        st.markdown("---")

        # Individual score cards
        st.markdown("### Indicator Scores 各指標評分")

        def render_score_cards(scores_list, section_name):
            if not scores_list:
                st.caption(f"No {section_name} data available")
                return
            cols = st.columns(min(len(scores_list), 4))
            for i, s in enumerate(scores_list):
                with cols[i % len(cols)]:
                    sc = s["score"]
                    _, sc_color = score_label(sc)
                    st.markdown(f"""
                    <div class="score-card">
                        <div style="display:flex; justify-content:space-between;">
                            <b>{s['name']}</b>
                            <span style="color:{sc_color}; font-weight:bold;">{sc}/10</span>
                        </div>
                        <div style="color:{COLORS['text_secondary']}; font-size:0.85em; margin-top:4px;">
                            {s['description']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if s.get("scoring_rule"):
                        with st.expander(f"ℹ {s['name']} detail"):
                            st.caption(s["scoring_rule"])
                            # Show mini chart if available
                            chart_fn = MINI_CHART_MAP.get(s["name"])
                            if chart_fn:
                                mini_fig = chart_fn(df_indicators)
                                if mini_fig and mini_fig.data:
                                    st.plotly_chart(mini_fig, use_container_width=True, key=f"mini_{section_name}_{s['name']}_{i}")

        st.markdown(f"**Technical 技術面** (avg: {all_scores['technical_avg']:.1f})" if all_scores['technical_avg'] else "**Technical 技術面**")
        render_score_cards(all_scores["technical"], "technical")

        st.markdown(f"**Fundamental 基本面** (avg: {all_scores['fundamental_avg']:.1f})" if all_scores['fundamental_avg'] else "**Fundamental 基本面**")
        render_score_cards(all_scores["fundamental"], "fundamental")

        st.markdown(f"**Chip 籌碼面** (avg: {all_scores['chip_avg']:.1f})" if all_scores['chip_avg'] else "**Chip 籌碼面**")
        render_score_cards(all_scores["chip"], "chip")

        st.markdown("---")

        # Highlights & Risks
        col_h, col_r = st.columns(2)
        with col_h:
            st.markdown("### 📈 Highlights 正面信號")
            if highlights:
                for h in highlights:
                    st.markdown(f'<div class="highlight-box">{h}</div>', unsafe_allow_html=True)
            else:
                st.caption("No strong positive signals")

        with col_r:
            st.markdown("### ⚠ Risks 風險警示")
            if risks:
                for r in risks:
                    st.markdown(f'<div class="risk-box">{r}</div>', unsafe_allow_html=True)
            else:
                st.caption("No strong risk signals")

        # Disclaimer
        st.markdown(f'<div class="disclaimer">⚠ 免責聲明: 本評分僅供研究參考，不構成任何投資建議。投資有風險，請自行評估。</div>', unsafe_allow_html=True)

    # ==================== TAB 2: TECHNICAL ====================
    with tab2:
        st.markdown("### Technical Analysis 技術分析")

        # Main candlestick chart
        show_ma_periods = st.multiselect(
            "Moving Averages", [5, 10, 20, 60, 120, 240],
            default=[5, 20, 60], key="ma_select"
        )
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            show_bb = st.checkbox("Bollinger Bands", value=True)
        with tc2:
            show_psar = st.checkbox("Parabolic SAR")
        with tc3:
            show_ichi = st.checkbox("Ichimoku Cloud")

        fig_main = tech_charts.create_candlestick_chart(
            df_indicators, title=f"{stock_name} ({ticker})",
            show_ma=show_ma_periods, show_bollinger=show_bb,
            show_psar=show_psar, show_ichimoku=show_ichi,
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # Chart explanation
        with st.expander("📖 K 線圖與均線說明"):
            st.markdown(f"""
<div class="indicator-desc">
<strong>K 線 (Candlestick)</strong>: 每根 K 棒顯示開盤、最高、最低、收盤四個價格。綠色 = 上漲，紅色 = 下跌。<br>
<strong>SMA (簡單移動平均)</strong>: 過去 N 日收盤價的平均值。短天期上穿長天期 = <span style="color:{COLORS['positive']}">黃金交叉 (買進)</span>，反之 = <span style="color:{COLORS['negative']}">死亡交叉 (賣出)</span>。<br>
<strong>Bollinger Bands (布林通道)</strong>: 中軌 = SMA20，上下軌 = ±2 標準差。股價觸及下軌可能反彈，觸及上軌可能回落。帶寬收窄代表即將突破。<br>
<strong>Parabolic SAR</strong>: 綠點在股價下方 = 多頭，紅點在上方 = 空頭。SAR 翻轉是趨勢反轉信號。<br>
<strong>Ichimoku Cloud</strong>: 雲帶之上 = 多頭，雲帶之下 = 空頭。轉換線穿越基準線是買賣信號。
</div>
            """, unsafe_allow_html=True)

        # Sub-charts with descriptions
        INDICATOR_DESCRIPTIONS = {
            "MACD": "**MACD (12, 26, 9)**: DIF 上穿 Signal = 金叉(買進)，下穿 = 死叉(賣出)。零軸上方金叉更強烈。柱狀體 (Histogram) 由負轉正 = 多方動能增強。背離 (股價新高但 MACD 未新高) 是強反轉信號。",
            "RSI": "**RSI (14)**: 衡量漲跌動能。RSI > 70 = 超買(可能回落)，RSI < 30 = 超賣(可能反彈)。50 為多空分界。RSI 背離是重要的反轉信號。",
            "KD": "**KD / Stochastic (9, 3, 3)**: K 線上穿 D 線 = 黃金交叉，低檔 (<20) 交叉更強。高檔 (>80) 死叉 = 賣出。注意：強趨勢中 KD 容易鈍化。",
            "ADX": "**ADX (14)**: 衡量趨勢強度 (不分方向)。ADX > 25 = 有趨勢，< 20 = 盤整。+DI > -DI 且 ADX 上升 = 多頭趨勢確認。",
            "OBV": "**OBV (On-Balance Volume)**: 量能潮指標。OBV 上升 + 股價上升 = 量價配合。OBV 上升但股價下跌 = 底背離(買進信號)。",
            "Williams_R": "**Williams %R (14)**: 與 RSI 類似的超買超賣指標。%R < -80 = 超賣(買進)，%R > -20 = 超買(賣出)。",
            "CCI": "**CCI (20)**: 順勢指標。CCI > +100 = 超買區，< -100 = 超賣區。從 -100 下方回升到 -100 以上 = 買進信號。",
            "ATR": "**ATR (14)**: 平均真實波幅，衡量波動度。ATR 越高波動越大。常用於設定停損距離 (1.5-2 × ATR)。不產生買賣信號。",
        }

        sub_charts = {
            "MACD": ("MACD", lambda: tech_charts.create_macd_chart(df_indicators)),
            "RSI": ("RSI", lambda: tech_charts.create_rsi_chart(df_indicators)),
            "KD": ("KD / Stochastic", lambda: tech_charts.create_kd_chart(df_indicators)),
            "ADX": ("ADX", lambda: tech_charts.create_adx_chart(df_indicators)),
            "OBV": ("OBV", lambda: tech_charts.create_obv_chart(df_indicators)),
            "Williams_R": ("Williams %R", lambda: tech_charts.create_williams_r_chart(df_indicators)),
            "CCI": ("CCI", lambda: tech_charts.create_cci_chart(df_indicators)),
            "ATR": ("ATR", lambda: tech_charts.create_atr_chart(df_indicators)),
        }

        selected_sub = st.multiselect(
            "Select Sub-Charts 選擇副圖",
            list(sub_charts.keys()),
            default=["MACD", "RSI", "KD"],
            format_func=lambda x: sub_charts[x][0],
        )

        for key in selected_sub:
            fig = sub_charts[key][1]()
            st.plotly_chart(fig, use_container_width=True)
            # Show indicator description below each chart
            desc = INDICATOR_DESCRIPTIONS.get(key, "")
            if desc:
                st.markdown(f'<div class="indicator-desc">{desc}</div>', unsafe_allow_html=True)

    # ==================== TAB 3: FUNDAMENTAL ====================
    with tab3:
        st.markdown("### Fundamental Analysis 基本面分析")

        with st.expander("📖 基本面指標說明"):
            st.markdown(f"""
<div class="indicator-desc">
<strong>P/E (本益比)</strong>: 股價 / EPS。越低代表越便宜，但需與同產業比較。P/E < 15 偏低，> 25 偏高。<br>
<strong>P/B (股價淨值比)</strong>: 股價 / 每股淨值。P/B < 1 代表股價低於帳面價值。<br>
<strong>EPS (每股盈餘)</strong>: 公司每股賺多少錢。持續成長是正面信號，YoY > 20% = 高成長。<br>
<strong>ROE (股東權益報酬率)</strong>: 巴菲特最看重的指標。ROE > 15% = 良好，> 20% = 優秀。<br>
<strong>ROA (資產報酬率)</strong>: 衡量資產運用效率。ROA 與 ROE 差距大 = 高槓桿風險。<br>
<strong>殖利率 (Dividend Yield)</strong>: 每股股利 / 股價。> 4% = 良好，但過高可能是股價暴跌造成的陷阱。<br>
<strong>D/E Ratio (負債比)</strong>: 總負債 / 股東權益。< 0.5 = 保守安全，> 2.0 = 高風險。<br>
<strong>FCF (自由現金流)</strong>: 營業現金流 - 資本支出。正值且成長 = 公司真正在賺錢。<br>
<strong>Operating Margin (營益率)</strong>: 營業利益 / 營收。越高代表本業賺錢能力越強。<br>
<strong>Current Ratio (流動比率)</strong>: 流動資產 / 流動負債。> 2.0 = 安全，< 1.0 = 短期償債能力不足。
</div>
            """, unsafe_allow_html=True)

        # Radar chart
        if all_scores["fundamental"]:
            fig_radar = fund_charts.create_fundamental_radar(all_scores["fundamental"])
            st.plotly_chart(fig_radar, use_container_width=True)

        # Metrics table
        st.markdown("#### Key Metrics 關鍵指標")
        metrics_data = []
        for key, analysis in fund_analysis.items():
            metrics_data.append({
                "Indicator": analysis.get("label", key),
                "Value": analysis.get("display", "N/A"),
                "Description": analysis.get("description", ""),
            })
        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        # Financial statements (if available)
        financials = DataStore.get_financials(ticker)
        if financials:
            st.markdown("#### Financial Trends 財務趨勢")
            metric_options = {
                "Total Revenue": "Total Revenue 總營收",
                "Net Income": "Net Income 淨利",
                "Operating Income": "Operating Income 營業利益",
                "Gross Profit": "Gross Profit 毛利",
            }
            selected_metric = st.selectbox(
                "Select metric", list(metric_options.keys()),
                format_func=lambda x: metric_options[x],
            )
            fig_trend = fund_charts.create_metric_trend_chart(
                financials, selected_metric, title=metric_options[selected_metric]
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Dividends
        div_df = DataStore.get_dividends(ticker)
        if not div_df.empty:
            st.markdown("#### Dividend History 歷年股利")
            fig_div = fund_charts.create_dividend_chart(div_df)
            st.plotly_chart(fig_div, use_container_width=True)

    # ==================== TAB 4: CHIP ANALYSIS ====================
    with tab4:
        st.markdown("### Chip Analysis 籌碼面分析")

        with st.expander("📖 籌碼面指標說明"):
            st.markdown(f"""
<div class="indicator-desc">
<strong>三大法人</strong>: 外資 (影響力最大，佔台股成交量 ~30%)、投信 (偏好中小型成長股)、自營商 (短線操作)。三大法人同步買超 > 5 日 = 強烈正面信號。<br>
<strong>融資 (Margin Buy)</strong>: 散戶借錢買股，代表散戶看多。<span style="color:{COLORS['warning']}">逆向指標</span>：融資大增 = 散戶搶進 → 可能是高點。融資減少 = 籌碼沈澱 → 偏多。<br>
<strong>融券 (Short Selling)</strong>: 散戶借券賣出，代表散戶看空。<br>
<strong>券資比</strong>: 融券餘額 / 融資餘額。券資比 > 30% = 軋空行情機會。<br>
<strong>Beta</strong>: 相對大盤的波動度。Beta > 1 = 波動大於大盤 (攻擊型)，< 1 = 防禦型。<br>
<strong>Sharpe Ratio</strong>: 風險調整後報酬。> 2 優秀，1-2 良好，< 0.5 差。<br>
<strong>Alpha</strong>: 超額報酬。Alpha > 0 = 優於大盤。
</div>
            """, unsafe_allow_html=True)

        if market in ("TW", "TWO"):
            if not inst_df.empty:
                fig_inst = chip_charts.create_institutional_chart(inst_df)
                st.plotly_chart(fig_inst, use_container_width=True)

                # Summary metrics
                st.markdown("#### Institutional Summary 法人摘要")
                inst_cols = st.columns(4)
                for i, (key, data) in enumerate(inst_data.items()):
                    with inst_cols[i % 4]:
                        st.metric(
                            data.get("label", key),
                            data.get("display", "N/A"),
                        )
            else:
                st.info("⏳ Institutional data requires TWSE API calls with rate limiting. Data may take a moment to load for the selected date range.")

            if not margin_df.empty:
                fig_margin = chip_charts.create_margin_chart(margin_df)
                st.plotly_chart(fig_margin, use_container_width=True)

                # Margin metrics
                st.markdown("#### Margin Summary 融資融券摘要")
                m_cols = st.columns(3)
                for i, (key, data) in enumerate(margin_data.items()):
                    with m_cols[i % 3]:
                        st.metric(data.get("label", key), data.get("display", "N/A"))
        else:
            st.info("Chip analysis is primarily available for Taiwan stocks. For US stocks, quantitative metrics (Beta, Sharpe, Alpha) are shown below.")

        # Quantitative metrics (all markets)
        st.markdown("#### Quantitative Metrics 量化指標")
        q_cols = st.columns(3)
        for i, (key, data) in enumerate(quant_data.items()):
            with q_cols[i % 3]:
                st.metric(data.get("label", key), data.get("display", "N/A"))
                st.caption(data.get("description", ""))

    # ==================== TAB 5: MARKET COMPARE ====================
    with tab5:
        st.markdown("### Market Comparison 大盤比較")

        # Collect selected benchmarks
        bench_tickers = [ticker]
        bench_names = [stock_name]
        for name, idx_ticker in MARKET_INDICES.items():
            if selected_indices.get(name, False):
                bench_tickers.append(idx_ticker)
                bench_names.append(name)

        if len(bench_tickers) > 1:
            multi_prices = DataStore.get_multiple_prices(bench_tickers, start_str, end_str)
            if not multi_prices.empty:
                # Rename columns for display
                rename_map = dict(zip(bench_tickers, bench_names))
                multi_prices = multi_prices.rename(columns=rename_map)

                fig_compare = comp_charts.create_comparison_chart(
                    multi_prices,
                    title=f"Performance Comparison (Base = 100%)",
                )
                st.plotly_chart(fig_compare, use_container_width=True)

                # Performance table
                st.markdown("#### Period Returns 區間績效")
                perf_table = comp_charts.create_performance_table(multi_prices)
                st.dataframe(perf_table, use_container_width=True, hide_index=True)

                # Correlation
                if len(bench_tickers) >= 2:
                    st.markdown("#### Correlation 相關性")
                    corr_df = comp_charts.calculate_correlation(multi_prices)
                    fig_corr = comp_charts.create_correlation_heatmap(corr_df)
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Please select at least one benchmark index from the sidebar.")

    # ==================== TAB 6: BACKTEST ====================
    with tab6:
        st.markdown("### Backtest 歷史回測")

        from utils.strategies import PRESET_STRATEGIES
        from utils.helpers import run_strategy_backtest

        # How to read results guide
        with st.expander("📖 回測使用說明 & 如何看結果"):
            st.markdown(f"""
<div class="indicator-desc">
<strong>三種回測模式:</strong><br>
1. <strong>Strategy 策略模式</strong> — 選擇預設策略 (Golden Cross, RSI, MACD...)，一鍵回測<br>
2. <strong>Score 評分模式</strong> — 用綜合評分閾值 (評分 >= X 買, <= Y 賣)，需計算每日評分(較慢)<br>
3. <strong>Advanced 進階模式</strong> — 自訂指標條件組合 (AND/OR 邏輯)

<br><br><strong>風險管理選項:</strong><br>
- <strong>Stop-Loss 停損</strong>: 虧損達 X% 強制賣出 (例: 5% = 虧損 5% 就停損)<br>
- <strong>Take-Profit 停利</strong>: 獲利達 X% 強制賣出<br>
- <strong>Trailing Stop 移動停損</strong>: 從最高點回落 X% 就賣出 (鎖住獲利)<br>
- <strong>Position Size 部位大小</strong>: 每次用多少比例的資金 (100% = all-in)

<br><br><strong>如何解讀結果:</strong><br>
- <strong>資金曲線 (Equity Curve)</strong>: 綠線應在灰色虛線 (Buy & Hold) 之上。平滑上升 = 好策略<br>
- <strong>回撤圖 (Drawdown)</strong>: 越淺越好。最大回撤 > -30% 代表風險很高<br>
- <strong>買賣信號</strong>: 綠色 ▲ = 買進, 紅色 ▼ = 賣出。觀察信號位置是否合理<br>
- <strong>Sharpe > 1.0</strong> = 良好, <strong>Win Rate > 50%</strong> 且 <strong>Profit Factor > 1.5</strong> = 策略有效<br>
- <strong>策略 vs Benchmark</strong>: 策略報酬必須 > Buy & Hold 才有附加價值
</div>
            """, unsafe_allow_html=True)

        bt_mode = st.radio("Mode 模式", [
            "Strategy 策略模式", "Score 評分模式", "Advanced 進階模式"
        ], horizontal=True)

        # --- Shared risk management controls ---
        st.markdown("---")
        st.markdown("**Risk Management 風險管理**")
        rm1, rm2, rm3, rm4 = st.columns(4)
        with rm1:
            stop_loss = st.number_input("Stop-Loss %", 0.0, 50.0, 0.0, 1.0, key="bt_sl",
                                        help="0 = disabled. E.g. 5 = sell if loss reaches -5%")
        with rm2:
            take_profit = st.number_input("Take-Profit %", 0.0, 100.0, 0.0, 1.0, key="bt_tp",
                                          help="0 = disabled. E.g. 15 = sell if gain reaches +15%")
        with rm3:
            trailing = st.number_input("Trailing Stop %", 0.0, 50.0, 0.0, 1.0, key="bt_ts",
                                       help="0 = disabled. E.g. 5 = sell if price drops 5% from peak")
        with rm4:
            pos_size = st.number_input("Position Size %", 10.0, 100.0, 100.0, 10.0, key="bt_ps",
                                       help="% of capital per trade. 100 = all-in")

        cap1, _ = st.columns([1, 2])
        with cap1:
            initial_capital = st.number_input("Initial Capital 初始資金", value=1_000_000, step=100_000, key="bt_cap")

        st.markdown("---")

        # ========== MODE 1: Strategy ==========
        if bt_mode.startswith("Strategy"):
            strat_keys = list(PRESET_STRATEGIES.keys())
            strat_names = [PRESET_STRATEGIES[k]["name"] for k in strat_keys]
            selected_idx = st.selectbox("Select Strategy 選擇策略", range(len(strat_keys)),
                                        format_func=lambda i: strat_names[i])
            strat_key = strat_keys[selected_idx]
            strat = PRESET_STRATEGIES[strat_key]

            # Show strategy description
            st.markdown(strat["description"])

            if st.button("🔄 Run Strategy Backtest", key="bt_strat", type="primary"):
                with st.spinner("Running strategy backtest..."):
                    bt_result = run_strategy_backtest(
                        df_indicators, strat_key,
                        initial_capital=initial_capital, ticker=ticker,
                        stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                        trailing_stop_pct=trailing, position_size_pct=pos_size,
                    )
                    _render_backtest_results(st, bt_result, df_indicators, strat, COLORS)

        # ========== MODE 2: Score ==========
        elif bt_mode.startswith("Score"):
            st.markdown("**Score Mode**: Buy when composite score >= threshold, sell when <= threshold.")
            sc1, sc2 = st.columns(2)
            with sc1:
                buy_threshold = st.slider("Buy Threshold 買進閾值", 1.0, 10.0, 7.0, 0.5)
            with sc2:
                sell_threshold = st.slider("Sell Threshold 賣出閾值", 1.0, 10.0, 4.0, 0.5)

            if st.button("🔄 Run Score Backtest", key="bt_score", type="primary"):
                with st.spinner("Computing daily scores... (this may take a moment)"):
                    score_history = compute_daily_scores(
                        price_df, fundamentals_raw, inst_data, margin_data,
                        quant_data, weights, enabled_indicators,
                    )
                    if not score_history.empty:
                        bt_result = run_backtest_simple(
                            price_df, score_history,
                            buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                            initial_capital=initial_capital, ticker=ticker,
                            stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                            trailing_stop_pct=trailing,
                        )
                        fig_score_hist = bt_charts.create_score_history_chart(score_history)
                        st.plotly_chart(fig_score_hist, use_container_width=True)
                        _render_backtest_results(st, bt_result, df_indicators, None, COLORS)
                    else:
                        st.warning("Not enough data. Try a longer date range (>= 6 months).")

        # ========== MODE 3: Advanced ==========
        else:
            st.markdown("**Advanced Mode**: Define custom conditions with AND/OR logic.")
            logic = st.radio("Logic 邏輯", ["AND", "OR"], horizontal=True)

            available_cols = sorted([c for c in df_indicators.columns
                                    if c not in ("Open", "High", "Low", "Close", "Volume")])

            st.markdown("**Buy Conditions 買進條件:**")
            buy_conditions = []
            n_buy = st.number_input("Buy conditions count", 1, 10, 2, key="n_buy")
            for i in range(int(n_buy)):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    ind = st.selectbox(f"Buy Indicator {i+1}", available_cols, key=f"buy_ind_{i}")
                with c2:
                    op = st.selectbox(f"Op {i+1}", ["<", ">", "<=", ">="], key=f"buy_op_{i}")
                with c3:
                    val = st.number_input(f"Val {i+1}", value=30.0, key=f"buy_val_{i}")
                buy_conditions.append({"indicator": ind, "operator": op, "value": val, "signal": "buy"})

            st.markdown("**Sell Conditions 賣出條件:**")
            sell_conditions = []
            n_sell = st.number_input("Sell conditions count", 1, 10, 2, key="n_sell")
            for i in range(int(n_sell)):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    ind = st.selectbox(f"Sell Indicator {i+1}", available_cols, key=f"sell_ind_{i}")
                with c2:
                    op = st.selectbox(f"Op {i+1}", ["<", ">", "<=", ">="], key=f"sell_op_{i}")
                with c3:
                    val = st.number_input(f"Val {i+1}", value=70.0, key=f"sell_val_{i}")
                sell_conditions.append({"indicator": ind, "operator": op, "value": val, "signal": "sell"})

            if st.button("🔄 Run Advanced Backtest", key="bt_adv", type="primary"):
                with st.spinner("Running advanced backtest..."):
                    bt_result = run_backtest_advanced(
                        price_df, df_indicators, buy_conditions + sell_conditions, logic,
                        initial_capital=initial_capital, ticker=ticker,
                        stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                        trailing_stop_pct=trailing,
                    )
                    _render_backtest_results(st, bt_result, df_indicators, None, COLORS)

        st.markdown(f'<div class="disclaimer">⚠ 回測結果基於歷史數據，不保證未來表現。過去績效不代表未來收益。</div>', unsafe_allow_html=True)

    # ==================== TAB 7: BATCH ANALYSIS ====================
    with tab7:
        st.markdown("### Batch Analysis 批次分析")
        st.markdown("""
        輸入多檔股票代號，一次計算所有指標與綜合評分，匯出 Excel 供選股參考。
        """)

        with st.expander("📖 使用說明"):
            st.markdown(f"""
<div class="indicator-desc">
<strong>如何使用:</strong><br>
1. 在下方文字框輸入股票代號，每行一個或用逗號分隔 (例: 2330.TW, 2317.TW, 2454.TW)<br>
2. 也可以點擊「Load 台股 Top 50」或「Load 美股 Top 50」快速載入清單<br>
3. 點擊「Run Batch Analysis」開始分析<br>
4. 分析完成後可以直接在網頁上查看，也可以下載 Excel 檔案<br><br>
<strong>Excel 檔案包含:</strong><br>
- 每檔股票的綜合評分 (1-10) 與買賣信號<br>
- 技術面各指標分數 (MACD, RSI, KD, Bollinger...)<br>
- 基本面各指標分數與原始數值 (P/E, EPS, ROE...)<br>
- 量化指標 (Beta, Sharpe, Alpha)<br>
- RSI 值、MACD/KD 交叉狀態<br>
- 正面信號 & 風險警示摘要<br>
- 已按綜合評分排序，高分在前
</div>
            """, unsafe_allow_html=True)

        # Quick-load buttons
        bl1, bl2, bl3 = st.columns(3)
        with bl1:
            if st.button("Load 台股 Top 50", key="load_tw50"):
                st.session_state["batch_tickers"] = "\n".join(TW_TOP50)
        with bl2:
            if st.button("Load 美股 Top 50", key="load_us50"):
                st.session_state["batch_tickers"] = "\n".join(US_TOP50)
        with bl3:
            if st.button("Clear 清除", key="clear_batch"):
                st.session_state["batch_tickers"] = ""

        default_tickers = st.session_state.get("batch_tickers", "2330.TW\n2317.TW\n2454.TW\nAAPL\nNVDA")
        batch_input = st.text_area(
            "Stock Tickers (one per line or comma-separated)",
            value=default_tickers,
            height=150,
            key="batch_text",
        )

        if st.button("🔍 Run Batch Analysis", key="run_batch", type="primary"):
            # Parse tickers
            raw = batch_input.replace(",", "\n").replace(" ", "\n")
            tickers_list = [t.strip().upper() for t in raw.split("\n") if t.strip()]
            # Auto-add .TW for pure numbers
            tickers_list = [f"{t}.TW" if t.isdigit() else t for t in tickers_list]
            tickers_list = list(dict.fromkeys(tickers_list))  # dedupe

            if not tickers_list:
                st.warning("Please enter at least one ticker.")
            else:
                st.info(f"Analyzing {len(tickers_list)} stocks...")
                progress = st.progress(0, text="Starting...")

                def update_progress(curr, total, ticker):
                    if total > 0:
                        progress.progress(min(curr / total, 1.0),
                                         text=f"Analyzing {ticker}... ({curr}/{total})")

                batch_df = batch_analyze(
                    tickers_list, start_str, end_str,
                    weights=weights,
                    progress_callback=update_progress,
                )
                progress.empty()

                if not batch_df.empty:
                    st.success(f"Analysis complete! {len(batch_df)} stocks analyzed.")

                    # Summary stats
                    if "Composite_Score" in batch_df.columns:
                        buy_count = len(batch_df[batch_df["Signal"].str.contains("買進", na=False)])
                        sell_count = len(batch_df[batch_df["Signal"].str.contains("賣出", na=False)])
                        neutral_count = len(batch_df) - buy_count - sell_count
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        with sc1: st.metric("Total Stocks", len(batch_df))
                        with sc2: st.metric("Buy Signal 買進", buy_count)
                        with sc3: st.metric("Neutral 中性", neutral_count)
                        with sc4: st.metric("Sell Signal 賣出", sell_count)

                    # Display table
                    display_cols = [c for c in batch_df.columns if c not in ("Highlights", "Risks")]
                    st.dataframe(
                        batch_df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        height=min(len(batch_df) * 38 + 40, 600),
                    )

                    # Download Excel
                    excel_bytes = export_to_excel(batch_df)
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_bytes,
                        file_name=f"stock_analysis_{ts}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    # Top picks
                    if "Composite_Score" in batch_df.columns and len(batch_df) >= 3:
                        st.markdown("#### Top Picks 推薦標的 (Score >= 7)")
                        top = batch_df[batch_df["Composite_Score"] >= 7].head(10)
                        if not top.empty:
                            for _, row in top.iterrows():
                                hl = row.get("Highlights", "")
                                st.markdown(f"""
                                <div class="highlight-box">
                                    <strong>{row['Ticker']}</strong> {row.get('Name', '')} —
                                    Score: <strong>{row['Composite_Score']}</strong> ({row['Signal']})
                                    | Price: ${row.get('Price', 'N/A')}
                                    {f'<br><small>{hl}</small>' if hl else ''}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.caption("No stocks scored >= 7 in this batch.")
                else:
                    st.error("No results returned. Check ticker symbols.")

    # ==================== TAB 8: MOMENTUM SCREENING ====================
    with tab8:
        st.markdown("### Momentum Screening 動能篩選")
        st.markdown("""
        分析一組股票在不同時間區間的漲跌幅，找出**強勢股**與**弱勢股**，
        方便動能交易 (Momentum Trading) 選股。
        """)

        with st.expander("📖 使用說明"):
            st.markdown(f"""
<div class="indicator-desc">
<strong>動能交易 (Momentum Trading) 原理:</strong><br>
近期表現強勢的股票傾向繼續走強，弱勢的傾向繼續走弱。做多強勢股、避開弱勢股。<br><br>
<strong>如何使用:</strong><br>
1. 選擇股票清單 (台股/美股 Top 50，或自訂)<br>
2. 選擇分析期間 (今日/本週/本月/3個月/6個月/1年)<br>
3. 結果按漲幅排序，Top 25% = <span style="color:{COLORS['positive']}">Strong 強勢</span>，
   Bottom 25% = <span style="color:{COLORS['negative']}">Weak 弱勢</span><br><br>
<strong>各欄位說明:</strong><br>
- <strong>Return%</strong>: 該期間的漲跌幅<br>
- <strong>Vol Change%</strong>: 成交量相對變化 (正 = 放量，負 = 縮量)<br>
- <strong>RSI</strong>: 近期相對強弱<br>
- <strong>% From High</strong>: 距離期間最高點的跌幅 (0% = 在高點)<br>
- <strong>Percentile</strong>: 在所有股票中的百分位排名 (100 = 最強)<br>
- <strong>Strength</strong>: Strong (Top 25%) / Neutral / Weak (Bottom 25%)
</div>
            """, unsafe_allow_html=True)

        # Stock universe selection
        mu1, mu2 = st.columns(2)
        with mu1:
            universe = st.radio("Stock Universe 股票清單", [
                "台股 Top 50", "美股 Top 50", "Custom 自訂"
            ], horizontal=True, key="mom_universe")
        with mu2:
            mom_period = st.selectbox("Period 分析期間", [
                "1d (Today 今日)", "5d (This Week 本週)", "1mo (This Month 本月)",
                "3mo (3 Months)", "6mo (6 Months)", "1y (1 Year)",
            ], index=2, key="mom_period")

        period_key = mom_period.split(" ")[0]

        if universe == "Custom 自訂":
            custom_mom = st.text_area(
                "Custom tickers (comma or newline separated)",
                value="2330.TW, 2317.TW, 2454.TW, AAPL, NVDA, TSLA",
                key="mom_custom",
            )
            raw = custom_mom.replace(",", "\n").replace(" ", "\n")
            mom_tickers = [t.strip().upper() for t in raw.split("\n") if t.strip()]
            mom_tickers = [f"{t}.TW" if t.isdigit() else t for t in mom_tickers]
        elif universe.startswith("台股"):
            mom_tickers = TW_TOP50
        else:
            mom_tickers = US_TOP50

        if st.button("🚀 Run Momentum Screening", key="run_mom", type="primary"):
            st.info(f"Screening {len(mom_tickers)} stocks ({period_key})...")
            progress = st.progress(0, text="Downloading data...")

            def mom_progress(curr, total, ticker):
                if total > 0:
                    progress.progress(min(curr / total, 1.0),
                                     text=f"Analyzing {ticker}... ({curr}/{total})")

            mom_df = screen_momentum(mom_tickers, period=period_key, progress_callback=mom_progress)
            progress.empty()

            if not mom_df.empty:
                st.success(f"Screening complete! {len(mom_df)} stocks.")

                # Summary
                strong = mom_df[mom_df["Strength"] == "Strong"]
                weak = mom_df[mom_df["Strength"] == "Weak"]
                avg_ret = mom_df["Return_Pct"].mean()

                ms1, ms2, ms3, ms4 = st.columns(4)
                with ms1: st.metric("Avg Return", f"{avg_ret:.2f}%")
                with ms2: st.metric("Strong Stocks", len(strong))
                with ms3: st.metric("Weak Stocks", len(weak))
                with ms4: st.metric("Total", len(mom_df))

                # Chart: horizontal bar of returns
                import plotly.graph_objects as go
                from config import PLOTLY_LEGEND
                top_n = min(30, len(mom_df))
                chart_df = mom_df.head(top_n)
                bar_colors = [COLORS["positive"] if r >= 0 else COLORS["negative"]
                              for r in chart_df["Return_Pct"]]

                fig_mom = go.Figure(go.Bar(
                    x=chart_df["Return_Pct"],
                    y=chart_df["Ticker"],
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{r:+.1f}%" for r in chart_df["Return_Pct"]],
                    textposition="outside",
                    textfont=dict(color=COLORS["text_primary"], size=10),
                ))
                fig_mom.update_layout(
                    paper_bgcolor=COLORS["bg_primary"],
                    plot_bgcolor=COLORS["bg_primary"],
                    font=dict(color=COLORS["text_primary"]),
                    title=dict(text=f"Momentum Ranking ({period_key})",
                              font=dict(color=COLORS["accent"])),
                    height=max(top_n * 25, 400),
                    margin=dict(l=80, r=60, t=50, b=30),
                    yaxis=dict(autorange="reversed", gridcolor=COLORS["border"]),
                    xaxis=dict(title="Return %", gridcolor=COLORS["border"]),
                )
                st.plotly_chart(fig_mom, use_container_width=True)

                # Full table
                st.markdown("#### Full Results 完整結果")
                st.dataframe(mom_df, use_container_width=True, hide_index=True,
                            height=min(len(mom_df) * 38 + 40, 500))

                # Download Excel
                excel_bytes = export_momentum_excel(mom_df, period_key)
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_bytes,
                    file_name=f"momentum_{period_key}_{ts}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_mom",
                )

                # Top strong & weak picks
                col_s, col_w = st.columns(2)
                with col_s:
                    st.markdown(f"#### 🟢 Strong Stocks 強勢股 (Top 25%)")
                    if not strong.empty:
                        for _, row in strong.head(10).iterrows():
                            st.markdown(f"""
                            <div class="highlight-box">
                                <strong>{row['Ticker']}</strong> {row.get('Name', '')} —
                                <span style="color:{COLORS['positive']}">{row['Return_Pct']:+.2f}%</span>
                                | RSI: {row.get('RSI', 'N/A')} | Rank #{row.get('Rank', '')}
                            </div>
                            """, unsafe_allow_html=True)
                with col_w:
                    st.markdown(f"#### 🔴 Weak Stocks 弱勢股 (Bottom 25%)")
                    if not weak.empty:
                        for _, row in weak.tail(10).iterrows():
                            st.markdown(f"""
                            <div class="risk-box">
                                <strong>{row['Ticker']}</strong> {row.get('Name', '')} —
                                <span style="color:{COLORS['negative']}">{row['Return_Pct']:+.2f}%</span>
                                | RSI: {row.get('RSI', 'N/A')} | Rank #{row.get('Rank', '')}
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("No results. Check ticker symbols or try again.")

else:
    # Landing page
    st.markdown(f"""
    # 📊 Stock Analysis Platform
    ### 股票分析平台

    支援 **台股 (TWSE)** 與 **美股** 的完整分析工具

    ---

    #### Features 功能:
    - **40+ Indicators** — 技術面 / 基本面 / 籌碼面指標
    - **Composite Scoring** — 綜合買賣評分系統 (可自訂權重)
    - **Interactive Charts** — Plotly 互動圖表
    - **Market Comparison** — 與大盤指數比較
    - **Backtesting** — 歷史回測 (Simple + Advanced 模式)

    ---

    #### Quick Start 快速開始:
    1. 在左側欄輸入股票代號 (e.g. `2330.TW` 或 `AAPL`)
    2. 選擇日期區間
    3. 點擊 **🔍 Analyze** 開始分析

    #### Taiwan Stock Format 台股格式:
    - 上市: `2330.TW` (台積電), `2317.TW` (鴻海)
    - 上櫃: `6510.TWO`

    #### US Stock Format 美股格式:
    - `AAPL` (Apple), `NVDA` (NVIDIA), `TSLA` (Tesla)
    """)
