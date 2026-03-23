"""Preset backtesting strategy definitions."""

# Each strategy has: name, description, buy/sell conditions for the engine,
# and indicator columns it needs.

PRESET_STRATEGIES = {
    "golden_cross": {
        "name": "Golden Cross (SMA 20/60)",
        "description": (
            "**Golden Cross**: SMA(20) 上穿 SMA(60) 時買進，SMA(20) 下穿 SMA(60) 時賣出。\n\n"
            "經典的趨勢跟隨策略，信號較少但可靠。適合波段操作。\n\n"
            "- 買進: SMA(20) > SMA(60) 且前一日 SMA(20) <= SMA(60)\n"
            "- 賣出: SMA(20) < SMA(60) 且前一日 SMA(20) >= SMA(60)\n"
            "- 歷史勝率: ~70-79%"
        ),
        "type": "crossover",
        "buy_fast": "SMA_20",
        "buy_slow": "SMA_60",
        "sell_fast": "SMA_20",
        "sell_slow": "SMA_60",
        "required_cols": ["SMA_20", "SMA_60"],
        "overlay_indicators": ["SMA_20", "SMA_60"],
        "sub_indicators": [],
    },
    "rsi_reversion": {
        "name": "RSI Mean Reversion",
        "description": (
            "**RSI 均值回歸**: RSI(14) < 30 超賣時買進，RSI(14) > 70 超買時賣出。\n\n"
            "利用價格傾向回歸均值的特性。在盤整市場表現良好。\n\n"
            "- 買進: RSI(14) < 30\n"
            "- 賣出: RSI(14) > 70\n"
            "- 歷史勝率: ~73-91%\n"
            "- 注意: 在強趨勢下跌中 RSI 可能長期低於 30"
        ),
        "type": "threshold",
        "buy_conditions": [{"col": "RSI_14", "op": "<", "val": 30}],
        "sell_conditions": [{"col": "RSI_14", "op": ">", "val": 70}],
        "required_cols": ["RSI_14"],
        "overlay_indicators": [],
        "sub_indicators": ["RSI_14"],
    },
    "macd_cross": {
        "name": "MACD Crossover",
        "description": (
            "**MACD 交叉**: DIF 上穿 Signal (金叉) 買進，下穿 (死叉) 賣出。\n\n"
            "- 買進: MACD_DIF > MACD_Signal 且前一日 DIF <= Signal\n"
            "- 賣出: MACD_DIF < MACD_Signal 且前一日 DIF >= Signal\n"
            "- 歷史勝率: ~65-73%"
        ),
        "type": "crossover",
        "buy_fast": "MACD_DIF",
        "buy_slow": "MACD_Signal",
        "sell_fast": "MACD_DIF",
        "sell_slow": "MACD_Signal",
        "required_cols": ["MACD_DIF", "MACD_Signal", "MACD_Histogram"],
        "overlay_indicators": [],
        "sub_indicators": ["MACD_DIF", "MACD_Signal", "MACD_Histogram"],
    },
    "macd_rsi_combo": {
        "name": "MACD + RSI Combo",
        "description": (
            "**MACD + RSI 組合**: 結合兩個指標過濾假信號。\n\n"
            "- 買進: MACD 金叉 AND RSI < 45 (不在超買區)\n"
            "- 賣出: MACD 死叉 OR RSI > 70\n"
            "- 歷史勝率: ~73%\n"
            "- 特點: 交易次數較少，但每筆品質較高"
        ),
        "type": "custom",
        "func": "_macd_rsi_combo",
        "required_cols": ["MACD_DIF", "MACD_Signal", "RSI_14"],
        "overlay_indicators": [],
        "sub_indicators": ["MACD_DIF", "MACD_Signal", "MACD_Histogram", "RSI_14"],
    },
    "bollinger_bounce": {
        "name": "Bollinger Band Bounce",
        "description": (
            "**布林通道反彈**: 股價觸及下軌時買進，回到中軌或上軌時賣出。\n\n"
            "- 買進: Close <= BB_Lower (觸及下軌)\n"
            "- 賣出: Close >= BB_Mid (回到中軌)\n"
            "- 歷史勝率: ~70-75%\n"
            "- 適合盤整市場，趨勢下跌中可能連續觸下軌"
        ),
        "type": "threshold",
        "buy_conditions": [{"col": "BB_Percent", "op": "<=", "val": 0.0}],
        "sell_conditions": [{"col": "BB_Percent", "op": ">=", "val": 0.5}],
        "required_cols": ["BB_Upper", "BB_Mid", "BB_Lower", "BB_Percent"],
        "overlay_indicators": ["BB_Upper", "BB_Mid", "BB_Lower"],
        "sub_indicators": [],
    },
    "kd_cross": {
        "name": "KD Stochastic Crossover",
        "description": (
            "**KD 交叉策略**: K 線在低檔穿越 D 線時買進，高檔死叉時賣出。\n\n"
            "- 買進: K 上穿 D 且 K < 30 (低檔黃金交叉)\n"
            "- 賣出: K 下穿 D 且 K > 70 (高檔死亡交叉)\n"
            "- 歷史勝率: ~60-78%\n"
            "- 注意 KD 鈍化問題"
        ),
        "type": "custom",
        "func": "_kd_cross",
        "required_cols": ["K", "D"],
        "overlay_indicators": [],
        "sub_indicators": ["K", "D"],
    },
    "turtle_breakout": {
        "name": "Turtle Breakout (20-day)",
        "description": (
            "**海龜突破**: 價格突破 20 日最高價買進，跌破 10 日最低價賣出。\n\n"
            "經典趨勢跟隨策略。Richard Dennis 的海龜交易法。\n\n"
            "- 買進: Close > 20 日最高價 (突破)\n"
            "- 賣出: Close < 10 日最低價 (跌破)\n"
            "- 特點: 追漲殺跌，趨勢行情獲利大，盤整行情虧損多"
        ),
        "type": "custom",
        "func": "_turtle_breakout",
        "required_cols": ["High", "Low", "Close"],
        "overlay_indicators": ["_donchian_upper", "_donchian_lower"],
        "sub_indicators": [],
    },
    "volume_breakout": {
        "name": "Volume-Price Breakout",
        "description": (
            "**量價突破**: 股價突破 SMA(20) 且成交量放大時買進。\n\n"
            "- 買進: Close > SMA(20) AND Volume > 1.5 × Volume_MA(20)\n"
            "- 賣出: Close < SMA(20)\n"
            "- 量能確認讓突破更有效，可過濾假突破"
        ),
        "type": "custom",
        "func": "_volume_breakout",
        "required_cols": ["SMA_20", "Volume", "Volume_MA_20"],
        "overlay_indicators": ["SMA_20"],
        "sub_indicators": ["Volume"],
    },
}


def evaluate_strategy(strategy_key: str, df, i: int) -> tuple:
    """Evaluate strategy at index i. Returns (buy_signal: bool, sell_signal: bool)."""
    strat = PRESET_STRATEGIES[strategy_key]
    stype = strat["type"]

    if i < 1:
        return False, False

    curr = df.iloc[i]
    prev = df.iloc[i - 1]

    if stype == "crossover":
        fast_col = strat["buy_fast"]
        slow_col = strat["buy_slow"]
        if fast_col not in df.columns or slow_col not in df.columns:
            return False, False
        import pandas as pd
        cf = curr.get(fast_col)
        cs = curr.get(slow_col)
        pf = prev.get(fast_col)
        ps = prev.get(slow_col)
        if pd.isna(cf) or pd.isna(cs) or pd.isna(pf) or pd.isna(ps):
            return False, False
        buy = (pf <= ps) and (cf > cs)
        sell = (pf >= ps) and (cf < cs)
        return buy, sell

    elif stype == "threshold":
        import pandas as pd
        buy = True
        for cond in strat["buy_conditions"]:
            v = curr.get(cond["col"])
            if pd.isna(v):
                buy = False
                break
            if cond["op"] == "<" and not (v < cond["val"]):
                buy = False
            elif cond["op"] == "<=" and not (v <= cond["val"]):
                buy = False
            elif cond["op"] == ">" and not (v > cond["val"]):
                buy = False
            elif cond["op"] == ">=" and not (v >= cond["val"]):
                buy = False

        sell = True
        for cond in strat["sell_conditions"]:
            v = curr.get(cond["col"])
            if pd.isna(v):
                sell = False
                break
            if cond["op"] == "<" and not (v < cond["val"]):
                sell = False
            elif cond["op"] == "<=" and not (v <= cond["val"]):
                sell = False
            elif cond["op"] == ">" and not (v > cond["val"]):
                sell = False
            elif cond["op"] == ">=" and not (v >= cond["val"]):
                sell = False

        return buy, sell

    elif stype == "custom":
        fn_name = strat["func"]
        if fn_name == "_macd_rsi_combo":
            return _macd_rsi_combo(df, i)
        elif fn_name == "_kd_cross":
            return _kd_cross(df, i)
        elif fn_name == "_turtle_breakout":
            return _turtle_breakout(df, i)
        elif fn_name == "_volume_breakout":
            return _volume_breakout(df, i)

    return False, False


def _macd_rsi_combo(df, i):
    import pandas as pd
    curr, prev = df.iloc[i], df.iloc[i - 1]
    dif_c = curr.get("MACD_DIF")
    sig_c = curr.get("MACD_Signal")
    dif_p = prev.get("MACD_DIF")
    sig_p = prev.get("MACD_Signal")
    rsi = curr.get("RSI_14")
    if any(pd.isna(x) for x in [dif_c, sig_c, dif_p, sig_p, rsi]):
        return False, False
    golden = (dif_p <= sig_p) and (dif_c > sig_c)
    death = (dif_p >= sig_p) and (dif_c < sig_c)
    buy = golden and rsi < 45
    sell = death or rsi > 70
    return buy, sell


def _kd_cross(df, i):
    import pandas as pd
    curr, prev = df.iloc[i], df.iloc[i - 1]
    kc, dc = curr.get("K"), curr.get("D")
    kp, dp = prev.get("K"), prev.get("D")
    if any(pd.isna(x) for x in [kc, dc, kp, dp]):
        return False, False
    buy = (kp <= dp) and (kc > dc) and (kc < 30)
    sell = (kp >= dp) and (kc < dc) and (kc > 70)
    return buy, sell


def _turtle_breakout(df, i):
    import pandas as pd
    if i < 20:
        return False, False
    close = df.iloc[i]["Close"]
    high_20 = df["High"].iloc[i - 20:i].max()
    low_10 = df["Low"].iloc[max(0, i - 10):i].min()
    if pd.isna(close) or pd.isna(high_20) or pd.isna(low_10):
        return False, False
    buy = close > high_20
    sell = close < low_10
    return buy, sell


def _volume_breakout(df, i):
    import pandas as pd
    curr = df.iloc[i]
    close = curr.get("Close")
    sma20 = curr.get("SMA_20")
    vol = curr.get("Volume")
    vol_ma = curr.get("Volume_MA_20")
    if any(pd.isna(x) for x in [close, sma20, vol, vol_ma]) or vol_ma == 0:
        return False, False
    buy = (close > sma20) and (vol > 1.5 * vol_ma)
    sell = close < sma20
    return buy, sell
