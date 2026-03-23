"""Backtest engine and utility helpers."""

import pandas as pd
import numpy as np
from typing import Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_COSTS, get_market
from utils.strategies import PRESET_STRATEGIES, evaluate_strategy


# ==================================================================
# Strategy-based backtest (new main engine)
# ==================================================================

def run_strategy_backtest(
    df_indicators: pd.DataFrame,
    strategy_key: str,
    initial_capital: float = 1_000_000,
    ticker: str = "",
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    trailing_stop_pct: float = 0.0,
    position_size_pct: float = 100.0,
) -> dict:
    """Run backtest using a preset strategy with risk management.

    Args:
        df_indicators: DataFrame with OHLCV + all indicators
        strategy_key: key from PRESET_STRATEGIES
        stop_loss_pct: fixed stop-loss % (0 = disabled), e.g. 5.0 = -5%
        take_profit_pct: take-profit % (0 = disabled), e.g. 10.0 = +10%
        trailing_stop_pct: trailing stop % from peak (0 = disabled)
        position_size_pct: % of capital to use per trade (1-100)
    """
    market = get_market(ticker) if ticker else "US"
    costs = TRADING_COSTS.get(market, TRADING_COSTS["US"])
    commission = costs["commission_rate"]
    tax = costs["tax_rate"]

    if df_indicators.empty or len(df_indicators) < 60:
        return _empty_backtest_result(df_indicators)

    # Add Donchian channels if needed for turtle
    if strategy_key == "turtle_breakout":
        df_indicators = df_indicators.copy()
        df_indicators["_donchian_upper"] = df_indicators["High"].rolling(20).max()
        df_indicators["_donchian_lower"] = df_indicators["Low"].rolling(10).min()

    capital = initial_capital
    shares = 0
    entry_price = 0.0
    peak_price = 0.0
    equity = []
    buy_signals = []
    sell_signals = []
    trades = []

    for i in range(len(df_indicators)):
        date = df_indicators.index[i]
        price = df_indicators.iloc[i]["Close"]
        if pd.isna(price):
            continue

        # Check strategy signals
        buy_sig, sell_sig = evaluate_strategy(strategy_key, df_indicators, i)

        # Check stop-loss / take-profit / trailing stop
        sl_triggered = False
        tp_triggered = False
        ts_triggered = False
        if shares > 0:
            pnl_pct = (price - entry_price) / entry_price * 100
            if stop_loss_pct > 0 and pnl_pct <= -stop_loss_pct:
                sl_triggered = True
            if take_profit_pct > 0 and pnl_pct >= take_profit_pct:
                tp_triggered = True
            if trailing_stop_pct > 0:
                peak_price = max(peak_price, price)
                drawdown_from_peak = (price - peak_price) / peak_price * 100
                if drawdown_from_peak <= -trailing_stop_pct:
                    ts_triggered = True

        # Execute trades
        if shares == 0 and buy_sig:
            alloc = capital * (position_size_pct / 100)
            buy_cost = price * (1 + commission)
            can_buy = int(alloc / buy_cost)
            if can_buy > 0:
                shares = can_buy
                capital -= shares * buy_cost
                entry_price = price
                peak_price = price
                buy_signals.append((date, price))
                trades.append({
                    "date": date, "action": "BUY", "price": price,
                    "shares": shares, "reason": "Signal",
                })

        elif shares > 0 and (sell_sig or sl_triggered or tp_triggered or ts_triggered):
            sell_revenue = shares * price * (1 - commission - tax)
            capital += sell_revenue
            pnl_pct = (price - entry_price) / entry_price * 100
            pnl_amount = (price - entry_price) * shares

            reason = "Signal"
            if sl_triggered:
                reason = f"Stop-Loss ({-stop_loss_pct:.1f}%)"
            elif tp_triggered:
                reason = f"Take-Profit (+{take_profit_pct:.1f}%)"
            elif ts_triggered:
                reason = f"Trailing-Stop ({-trailing_stop_pct:.1f}%)"

            sell_signals.append((date, price))
            trades.append({
                "date": date, "action": "SELL", "price": price,
                "shares": shares, "reason": reason,
                "pnl_pct": round(pnl_pct, 2), "pnl_amount": round(pnl_amount, 2),
            })
            shares = 0
            entry_price = 0
            peak_price = 0

        portfolio_value = capital + shares * price
        equity.append((date, portfolio_value))

    if not equity:
        return _empty_backtest_result(df_indicators)

    equity_curve = pd.Series([e[1] for e in equity], index=[e[0] for e in equity])

    # Benchmark
    prices = df_indicators["Close"].dropna()
    bench_shares = int(initial_capital / (prices.iloc[0] * (1 + commission)))
    bench_curve = bench_shares * prices + (initial_capital - bench_shares * prices.iloc[0] * (1 + commission))

    metrics = _calculate_backtest_metrics(equity_curve, bench_curve, initial_capital, trades)

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": bench_curve,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "price_series": prices,
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "metrics": metrics,
        "indicator_df": df_indicators,
    }


# ==================================================================
# Score-based simple backtest (kept for backward compat)
# ==================================================================

def run_backtest_simple(
    price_df: pd.DataFrame,
    score_history: pd.DataFrame,
    buy_threshold: float = 7.0,
    sell_threshold: float = 4.0,
    initial_capital: float = 1_000_000,
    ticker: str = "",
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    trailing_stop_pct: float = 0.0,
) -> dict:
    """Run a simple backtest based on composite score thresholds."""
    market = get_market(ticker) if ticker else "US"
    costs = TRADING_COSTS.get(market, TRADING_COSTS["US"])
    commission = costs["commission_rate"]
    tax = costs["tax_rate"]

    common_dates = price_df.index.intersection(score_history.index)
    if len(common_dates) < 10:
        return _empty_backtest_result(price_df)

    prices = price_df.loc[common_dates, "Close"]
    scores = score_history.loc[common_dates, "composite_score"]

    capital = initial_capital
    shares = 0
    entry_price = 0.0
    peak_price = 0.0
    equity = []
    buy_signals = []
    sell_signals = []
    trades = []

    for date in common_dates:
        price = prices[date]
        score = scores[date]

        sl_hit = tp_hit = ts_hit = False
        if shares > 0:
            pnl = (price - entry_price) / entry_price * 100
            if stop_loss_pct > 0 and pnl <= -stop_loss_pct:
                sl_hit = True
            if take_profit_pct > 0 and pnl >= take_profit_pct:
                tp_hit = True
            if trailing_stop_pct > 0:
                peak_price = max(peak_price, price)
                dd = (price - peak_price) / peak_price * 100
                if dd <= -trailing_stop_pct:
                    ts_hit = True

        if shares == 0 and score >= buy_threshold:
            buy_cost = price * (1 + commission)
            can_buy = int(capital / buy_cost)
            if can_buy > 0:
                shares = can_buy
                capital -= shares * buy_cost
                entry_price = price
                peak_price = price
                buy_signals.append((date, price))
                trades.append({"date": date, "action": "BUY", "price": price,
                              "shares": shares, "reason": "Score", "score": round(score, 2)})

        elif shares > 0 and (score <= sell_threshold or sl_hit or tp_hit or ts_hit):
            sell_revenue = shares * price * (1 - commission - tax)
            capital += sell_revenue
            pnl_pct = (price - entry_price) / entry_price * 100
            reason = "Score"
            if sl_hit:
                reason = "Stop-Loss"
            elif tp_hit:
                reason = "Take-Profit"
            elif ts_hit:
                reason = "Trailing-Stop"
            sell_signals.append((date, price))
            trades.append({"date": date, "action": "SELL", "price": price,
                          "shares": shares, "reason": reason, "score": round(score, 2),
                          "pnl_pct": round(pnl_pct, 2)})
            shares = 0
            entry_price = peak_price = 0

        equity.append((date, capital + shares * price))

    equity_curve = pd.Series([e[1] for e in equity], index=[e[0] for e in equity])
    bench_shares = int(initial_capital / (prices.iloc[0] * (1 + commission)))
    bench_curve = bench_shares * prices + (initial_capital - bench_shares * prices.iloc[0] * (1 + commission))
    metrics = _calculate_backtest_metrics(equity_curve, bench_curve, initial_capital, trades)

    return {
        "equity_curve": equity_curve,
        "benchmark_curve": bench_curve,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "price_series": prices,
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "metrics": metrics,
    }


# ==================================================================
# Advanced condition-based backtest
# ==================================================================

def run_backtest_advanced(
    price_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    conditions: list,
    logic: str = "AND",
    initial_capital: float = 1_000_000,
    ticker: str = "",
    stop_loss_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    trailing_stop_pct: float = 0.0,
) -> dict:
    """Run advanced backtest based on individual indicator conditions."""
    market = get_market(ticker) if ticker else "US"
    costs = TRADING_COSTS.get(market, TRADING_COSTS["US"])
    commission = costs["commission_rate"]
    tax = costs["tax_rate"]

    if indicator_df.empty:
        return _empty_backtest_result(price_df)

    capital = initial_capital
    shares = 0
    entry_price = peak_price = 0.0
    equity = []
    buy_signals = []
    sell_signals = []
    trades = []

    buy_conditions = [c for c in conditions if c.get("signal") == "buy"]
    sell_conditions = [c for c in conditions if c.get("signal") == "sell"]

    for i in range(len(indicator_df)):
        date = indicator_df.index[i]
        if date not in price_df.index:
            continue
        price = price_df.loc[date, "Close"]
        row = indicator_df.iloc[i]

        buy_met = _evaluate_conditions(row, buy_conditions, logic)
        sell_met = _evaluate_conditions(row, sell_conditions, logic)

        sl_hit = tp_hit = ts_hit = False
        if shares > 0:
            pnl = (price - entry_price) / entry_price * 100
            if stop_loss_pct > 0 and pnl <= -stop_loss_pct:
                sl_hit = True
            if take_profit_pct > 0 and pnl >= take_profit_pct:
                tp_hit = True
            if trailing_stop_pct > 0:
                peak_price = max(peak_price, price)
                if (price - peak_price) / peak_price * 100 <= -trailing_stop_pct:
                    ts_hit = True

        if shares == 0 and buy_met:
            buy_cost = price * (1 + commission)
            can_buy = int(capital / buy_cost)
            if can_buy > 0:
                shares = can_buy
                capital -= shares * buy_cost
                entry_price = price
                peak_price = price
                buy_signals.append((date, price))
                trades.append({"date": date, "action": "BUY", "price": price,
                              "shares": shares, "reason": "Signal"})

        elif shares > 0 and (sell_met or sl_hit or tp_hit or ts_hit):
            sell_revenue = shares * price * (1 - commission - tax)
            capital += sell_revenue
            pnl_pct = (price - entry_price) / entry_price * 100
            reason = "Signal"
            if sl_hit: reason = "Stop-Loss"
            elif tp_hit: reason = "Take-Profit"
            elif ts_hit: reason = "Trailing-Stop"
            sell_signals.append((date, price))
            trades.append({"date": date, "action": "SELL", "price": price,
                          "shares": shares, "reason": reason, "pnl_pct": round(pnl_pct, 2)})
            shares = 0
            entry_price = peak_price = 0

        equity.append((date, capital + shares * price))

    if not equity:
        return _empty_backtest_result(price_df)

    equity_curve = pd.Series([e[1] for e in equity], index=[e[0] for e in equity])
    prices = price_df.loc[indicator_df.index.intersection(price_df.index), "Close"]
    bench_shares = int(initial_capital / (prices.iloc[0] * (1 + commission))) if len(prices) > 0 else 0
    bench_curve = bench_shares * prices + (initial_capital - bench_shares * prices.iloc[0] * (1 + commission)) if bench_shares > 0 else prices * 0
    metrics = _calculate_backtest_metrics(equity_curve, bench_curve, initial_capital, trades)

    return {
        "equity_curve": equity_curve, "benchmark_curve": bench_curve,
        "buy_signals": buy_signals, "sell_signals": sell_signals,
        "price_series": prices,
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
        "metrics": metrics,
    }


# ==================================================================
# Daily score computation
# ==================================================================

def compute_daily_scores(
    price_df, fundamentals, institutional, margin, quantitative, weights,
    enabled_indicators=None, window=60,
):
    from indicators.technical import add_all_technical_indicators, get_latest_signals
    from indicators.fundamental import analyze_fundamentals
    from scoring.composite import compute_all_scores, compute_composite

    if price_df.empty or len(price_df) < window:
        return pd.DataFrame()

    df_with_indicators = add_all_technical_indicators(price_df)
    scores_list = []
    start_idx = max(window, 60)

    for i in range(start_idx, len(df_with_indicators)):
        date = df_with_indicators.index[i]
        subset = df_with_indicators.iloc[:i + 1]
        tech_signals = get_latest_signals(subset)
        fund_analysis = analyze_fundamentals(fundamentals)
        all_scores = compute_all_scores(
            tech_signals, fund_analysis, institutional, margin, quantitative,
            enabled_indicators,
        )
        composite = compute_composite(all_scores, weights)
        scores_list.append({"date": date, "composite_score": composite})

    if not scores_list:
        return pd.DataFrame()
    result = pd.DataFrame(scores_list)
    result["date"] = pd.to_datetime(result["date"])
    return result.set_index("date")


# ==================================================================
# Helpers
# ==================================================================

def _evaluate_conditions(row, conditions, logic):
    if not conditions:
        return False
    results = []
    for cond in conditions:
        indicator = cond.get("indicator", "")
        operator = cond.get("operator", "")
        target = cond.get("value")
        if indicator not in row.index:
            results.append(False)
            continue
        val = row[indicator]
        if pd.isna(val):
            results.append(False)
            continue
        try:
            if operator == ">": results.append(val > target)
            elif operator == "<": results.append(val < target)
            elif operator == ">=": results.append(val >= target)
            elif operator == "<=": results.append(val <= target)
            elif operator == "==": results.append(val == target)
            else: results.append(False)
        except (TypeError, ValueError):
            results.append(False)
    return all(results) if logic == "AND" else any(results)


def _calculate_backtest_metrics(equity_curve, benchmark_curve, initial_capital, trades):
    if equity_curve.empty:
        return {}

    final_value = equity_curve.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    cagr = ((final_value / initial_capital) ** (365 / max(days, 1)) - 1) * 100 if days > 0 else 0

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # Sharpe & Sortino
    daily_returns = equity_curve.pct_change().dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    downside = daily_returns[daily_returns < 0]
    sortino = np.sqrt(252) * daily_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0

    # Calmar
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Trade-level metrics
    buy_sell_pairs = []
    wins = []
    losses = []
    for t in trades:
        if t.get("action") == "SELL" and "pnl_pct" in t:
            pnl = t["pnl_pct"]
            buy_sell_pairs.append(pnl > 0)
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(abs(pnl))

    n_trades = len([t for t in trades if t.get("action") == "SELL"])
    win_rate = sum(buy_sell_pairs) / len(buy_sell_pairs) * 100 if buy_sell_pairs else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float("inf") if wins else 0
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0
    expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

    # Exposure
    holding_days = sum(1 for i in range(len(equity_curve)) if i > 0 and equity_curve.iloc[i] != equity_curve.iloc[i-1])
    exposure = holding_days / len(equity_curve) * 100 if len(equity_curve) > 0 else 0

    # Benchmark
    bench_return = 0
    if not benchmark_curve.empty:
        bench_return = (benchmark_curve.iloc[-1] - initial_capital) / initial_capital * 100

    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "Inf",
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(win_loss_ratio, 2) if win_loss_ratio != float("inf") else "Inf",
        "expectancy": round(expectancy, 2),
        "total_trades": n_trades,
        "exposure_pct": round(exposure, 1),
        "benchmark_return": round(bench_return, 2),
        "final_value": round(final_value, 0),
    }


def _empty_backtest_result(price_df):
    return {
        "equity_curve": pd.Series(dtype=float),
        "benchmark_curve": pd.Series(dtype=float),
        "buy_signals": [], "sell_signals": [],
        "price_series": price_df["Close"] if "Close" in price_df.columns else pd.Series(dtype=float),
        "trades": pd.DataFrame(), "metrics": {},
    }
