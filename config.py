"""Global configuration for the Stock Analysis Platform."""

# ============================================================
# Color Theme (Dark)
# ============================================================
COLORS = {
    "bg_primary": "#0f0f1a",
    "bg_card": "#1a1a2e",
    "bg_card_hover": "#16213e",
    "accent": "#00d4aa",
    "positive": "#00c853",
    "negative": "#ff5252",
    "warning": "#ffd740",
    "text_primary": "#ffffff",
    "text_secondary": "#b0b0b0",
    "chart_blue": "#3bafda",
    "chart_red": "#e9573f",
    "chart_yellow": "#f6bb42",
    "chart_green": "#00c853",
    "chart_purple": "#ab47bc",
    "border": "#2a2a4a",
}

# ============================================================
# Plotly dark layout template
# ============================================================
PLOTLY_LAYOUT = {
    "paper_bgcolor": COLORS["bg_primary"],
    "plot_bgcolor": COLORS["bg_primary"],
    "font": {"color": COLORS["text_primary"], "family": "Segoe UI, sans-serif"},
    "xaxis": {
        "gridcolor": COLORS["border"],
        "zerolinecolor": COLORS["border"],
    },
    "yaxis": {
        "gridcolor": COLORS["border"],
        "zerolinecolor": COLORS["border"],
    },
    "margin": {"l": 60, "r": 30, "t": 50, "b": 40},
}

# Default legend style (apply separately, not in PLOTLY_LAYOUT to avoid conflicts)
PLOTLY_LEGEND = {
    "bgcolor": "rgba(26,26,46,0.8)",
    "bordercolor": COLORS["border"],
    "font": {"color": COLORS["text_primary"]},
}

# ============================================================
# Market / Ticker Configuration
# ============================================================
MARKET_INDICES = {
    "TAIEX (加權指數)": "^TWII",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "PHLX Semiconductor (費半)": "^SOX",
}

# Detect market from ticker
def get_market(ticker: str) -> str:
    """Return 'TW', 'TWO', or 'US' based on ticker suffix."""
    t = ticker.upper()
    if t.endswith(".TW"):
        return "TW"
    elif t.endswith(".TWO"):
        return "TWO"
    else:
        return "US"

def get_default_index(ticker: str) -> str:
    """Return the default benchmark index ticker for the given stock."""
    market = get_market(ticker)
    if market in ("TW", "TWO"):
        return "^TWII"
    return "^GSPC"

# ============================================================
# Scoring defaults
# ============================================================
DEFAULT_WEIGHTS = {
    "technical": 0.40,
    "fundamental": 0.35,
    "chip": 0.25,
}

SCORE_LABELS = {
    (8.5, 10.0): ("強烈買進", COLORS["positive"]),
    (7.0, 8.49): ("偏多買進", "#66bb6a"),
    (5.5, 6.99): ("中性觀望", COLORS["warning"]),
    (3.5, 5.49): ("偏空賣出", "#ef5350"),
    (1.0, 3.49): ("強烈賣出", COLORS["negative"]),
}

def score_label(score: float) -> tuple:
    """Return (label_text, color) for a given score."""
    for (lo, hi), (label, color) in SCORE_LABELS.items():
        if lo <= score <= hi:
            return label, color
    return "N/A", COLORS["text_secondary"]

# ============================================================
# Trading costs
# ============================================================
TRADING_COSTS = {
    "TW": {"commission_rate": 0.001425, "tax_rate": 0.003},
    "TWO": {"commission_rate": 0.001425, "tax_rate": 0.003},
    "US": {"commission_rate": 0.0, "tax_rate": 0.0},
}

# ============================================================
# Indicator presets
# ============================================================
INDICATOR_PRESETS = {
    "全部": None,  # means all
    "短線交易": ["MACD", "RSI", "KD", "Bollinger", "Volume_MA", "ATR"],
    "波段操作": ["SMA", "MACD", "ADX", "Ichimoku", "OBV", "Parabolic_SAR"],
    "存股/價值": ["PE", "EPS", "ROE", "Dividend_Yield", "DE_Ratio", "FCF"],
    "自訂": None,  # user picks manually
}

# All available technical indicators
TECHNICAL_INDICATORS = [
    "SMA", "EMA", "WMA", "MACD", "RSI", "KD", "Bollinger",
    "OBV", "Volume_MA", "ATR", "Ichimoku", "Williams_R",
    "CCI", "ADX", "Parabolic_SAR",
]

FUNDAMENTAL_INDICATORS = [
    "PE", "PB", "EPS", "ROE", "ROA", "Dividend_Yield",
    "Revenue_Growth", "DE_Ratio", "FCF", "Operating_Margin",
    "Current_Ratio",
]

CHIP_INDICATORS_TW = [
    "Foreign_Inv", "Investment_Trust", "Dealers", "Institutional_Total",
    "Margin_Balance", "Short_Balance", "Short_Margin_Ratio",
]

CHIP_INDICATORS_US = [
    "Beta", "Sharpe_Ratio", "Alpha", "VIX", "Put_Call_Ratio",
]

QUANTITATIVE_INDICATORS = [
    "Beta", "Sharpe_Ratio", "Alpha",
]

# ============================================================
# TWSE API Configuration
# ============================================================
TWSE_BASE_URL = "https://www.twse.com.tw"
TWSE_OPENAPI_URL = "https://openapi.twse.com.tw/v1"
TPEX_OPENAPI_URL = "https://www.tpex.org.tw/openapi/v1"
TWSE_REQUEST_DELAY = 2.0  # seconds between requests (2s is safe, 3.5s was overly conservative)
