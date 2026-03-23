"""Technical indicator calculations using the 'ta' library."""

import pandas as pd
import numpy as np
from typing import Optional
import ta as ta_lib
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator, PSARIndicator,
    CCIIndicator,
)
from ta.momentum import RSIIndicator, StochRSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a price DataFrame.

    Expects columns: Open, High, Low, Close, Volume
    Returns the same DataFrame with additional indicator columns.
    """
    if df.empty or len(df) < 20:
        return df

    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]  # noqa
    v = df["Volume"]

    # --- Moving Averages ---
    for period in [5, 10, 20, 60, 120, 240]:
        if len(df) >= period:
            df[f"SMA_{period}"] = SMAIndicator(c, window=period).sma_indicator()

    for period in [12, 26]:
        df[f"EMA_{period}"] = EMAIndicator(c, window=period).ema_indicator()

    # WMA (manual calculation)
    if len(df) >= 20:
        weights = np.arange(1, 21)
        df["WMA_20"] = c.rolling(20).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    # --- MACD ---
    macd = MACD(c, window_slow=26, window_fast=12, window_sign=9)
    df["MACD_DIF"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()

    # --- RSI ---
    df["RSI_14"] = RSIIndicator(c, window=14).rsi()
    df["RSI_6"] = RSIIndicator(c, window=6).rsi()

    # --- Stochastic / KD ---
    stoch = StochasticOscillator(h, l, c, window=9, smooth_window=3)
    df["K"] = stoch.stoch()
    df["D"] = stoch.stoch_signal()

    # --- Bollinger Bands ---
    bb = BollingerBands(c, window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Bandwidth"] = bb.bollinger_wband()
    df["BB_Percent"] = bb.bollinger_pband()

    # --- OBV ---
    df["OBV"] = OnBalanceVolumeIndicator(c, v).on_balance_volume()

    # --- Volume MA ---
    df["Volume_MA_5"] = SMAIndicator(v, window=5).sma_indicator()
    df["Volume_MA_20"] = SMAIndicator(v, window=20).sma_indicator()
    df["Volume_Ratio"] = v / df["Volume_MA_5"]

    # --- ATR ---
    df["ATR_14"] = AverageTrueRange(h, l, c, window=14).average_true_range()

    # --- Ichimoku ---
    try:
        ichi = IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
        df["Ichimoku_Conversion"] = ichi.ichimoku_conversion_line()
        df["Ichimoku_Base"] = ichi.ichimoku_base_line()
        df["Ichimoku_SpanA"] = ichi.ichimoku_a()
        df["Ichimoku_SpanB"] = ichi.ichimoku_b()
    except Exception:
        pass

    # --- Williams %R ---
    df["Williams_R"] = WilliamsRIndicator(h, l, c, lbp=14).williams_r()

    # --- CCI ---
    df["CCI_20"] = CCIIndicator(h, l, c, window=20).cci()

    # --- ADX ---
    adx = ADXIndicator(h, l, c, window=14)
    df["ADX"] = adx.adx()
    df["DI_Plus"] = adx.adx_pos()
    df["DI_Minus"] = adx.adx_neg()

    # --- Parabolic SAR ---
    try:
        psar = PSARIndicator(h, l, c, step=0.02, max_step=0.2)
        df["PSAR_Up"] = psar.psar_up()
        df["PSAR_Down"] = psar.psar_down()
        # Map to Long/Short convention
        df["PSAR_Long"] = psar.psar_up()
        df["PSAR_Short"] = psar.psar_down()
    except Exception:
        pass

    return df


def get_latest_signals(df: pd.DataFrame) -> dict:
    """Extract the latest signal values for all technical indicators.

    Returns a dict suitable for the scoring engine.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    signals = {}

    # MACD
    if "MACD_DIF" in df.columns:
        signals["MACD_DIF"] = latest.get("MACD_DIF")
        signals["MACD_Signal"] = latest.get("MACD_Signal")
        signals["MACD_Histogram"] = latest.get("MACD_Histogram")
        signals["MACD_Histogram_Prev"] = prev.get("MACD_Histogram")
        if "MACD_DIF" in prev.index and "MACD_Signal" in prev.index:
            try:
                prev_dif = float(prev["MACD_DIF"])
                prev_sig = float(prev["MACD_Signal"])
                curr_dif = float(latest["MACD_DIF"])
                curr_sig = float(latest["MACD_Signal"])
                if not (pd.isna(prev_dif) or pd.isna(prev_sig) or pd.isna(curr_dif) or pd.isna(curr_sig)):
                    signals["MACD_Cross"] = (
                        "golden" if (prev_dif <= prev_sig and curr_dif > curr_sig)
                        else "death" if (prev_dif >= prev_sig and curr_dif < curr_sig)
                        else "none"
                    )
                else:
                    signals["MACD_Cross"] = "none"
            except (TypeError, ValueError):
                signals["MACD_Cross"] = "none"

    # RSI
    if "RSI_14" in df.columns:
        signals["RSI"] = latest.get("RSI_14")

    # KD
    if "K" in df.columns and "D" in df.columns:
        signals["K"] = latest.get("K")
        signals["D"] = latest.get("D")
        try:
            prev_k = float(prev.get("K", np.nan))
            prev_d = float(prev.get("D", np.nan))
            curr_k = float(latest.get("K", np.nan))
            curr_d = float(latest.get("D", np.nan))
            if not (pd.isna(prev_k) or pd.isna(prev_d) or pd.isna(curr_k) or pd.isna(curr_d)):
                signals["KD_Cross"] = (
                    "golden" if (prev_k <= prev_d and curr_k > curr_d)
                    else "death" if (prev_k >= prev_d and curr_k < curr_d)
                    else "none"
                )
            else:
                signals["KD_Cross"] = "none"
        except (TypeError, ValueError):
            signals["KD_Cross"] = "none"

    # Bollinger
    if "BB_Upper" in df.columns:
        signals["BB_Upper"] = latest.get("BB_Upper")
        signals["BB_Lower"] = latest.get("BB_Lower")
        signals["BB_Mid"] = latest.get("BB_Mid")
        signals["BB_Percent"] = latest.get("BB_Percent")
        signals["BB_Bandwidth"] = latest.get("BB_Bandwidth")

    # Moving Averages
    close = latest.get("Close")
    signals["Close"] = close
    for period in [5, 10, 20, 60, 120, 240]:
        key = f"SMA_{period}"
        if key in df.columns:
            signals[key] = latest.get(key)

    # MA alignment check
    ma_values = []
    for period in [5, 10, 20, 60]:
        key = f"SMA_{period}"
        if key in signals and signals[key] is not None and not pd.isna(signals[key]):
            ma_values.append(float(signals[key]))
    if len(ma_values) >= 3:
        signals["MA_Bullish_Align"] = all(
            ma_values[i] >= ma_values[i + 1] for i in range(len(ma_values) - 1)
        )
        signals["MA_Bearish_Align"] = all(
            ma_values[i] <= ma_values[i + 1] for i in range(len(ma_values) - 1)
        )

    # Volume
    if "Volume_Ratio" in df.columns:
        signals["Volume_Ratio"] = latest.get("Volume_Ratio")

    # OBV trend (last 5 days)
    if "OBV" in df.columns and len(df) >= 5:
        try:
            obv_vals = df["OBV"].iloc[-5:].dropna().values
            if len(obv_vals) >= 3:
                obv_slope = np.polyfit(range(len(obv_vals)), obv_vals, 1)[0]
                signals["OBV_Trend"] = "up" if obv_slope > 0 else "down"
        except Exception:
            signals["OBV_Trend"] = "unknown"

    # ATR
    if "ATR_14" in df.columns:
        signals["ATR"] = latest.get("ATR_14")

    # Williams %R
    if "Williams_R" in df.columns:
        signals["Williams_R"] = latest.get("Williams_R")

    # CCI
    if "CCI_20" in df.columns:
        signals["CCI"] = latest.get("CCI_20")

    # ADX
    if "ADX" in df.columns:
        signals["ADX"] = latest.get("ADX")
        signals["DI_Plus"] = latest.get("DI_Plus")
        signals["DI_Minus"] = latest.get("DI_Minus")

    # Parabolic SAR
    if "PSAR_Long" in df.columns:
        psar_long = latest.get("PSAR_Long")
        psar_short = latest.get("PSAR_Short")
        if pd.notna(psar_long) and pd.isna(psar_short):
            signals["PSAR_Direction"] = "bullish"
        elif pd.isna(psar_long) and pd.notna(psar_short):
            signals["PSAR_Direction"] = "bearish"
        else:
            signals["PSAR_Direction"] = "unknown"

    # Ichimoku
    ichi_cols = [c for c in df.columns if c.startswith("Ichimoku_")]
    for col in ichi_cols:
        signals[col] = latest.get(col)

    return signals
