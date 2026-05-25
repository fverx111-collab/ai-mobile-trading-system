from __future__ import annotations

import hmac
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(page_title="AI 交易參考系統", layout="wide", initial_sidebar_state="expanded")


US_UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "TSM", "ARM", "SMCI", "PLTR", "MU", "QCOM", "INTC", "CRM", "ADBE", "ORCL",
    "LLY", "JPM", "V", "MA", "COST", "WMT", "UNH", "NVO", "ASML", "MELI",
    "QQQ", "SPY", "DIA", "IWM", "VTI", "SOXX", "XLK", "XLF", "XLE", "GLD",
]

TW_UNIVERSE = [
    "2330.TW", "2317.TW", "2454.TW", "2303.TW", "2382.TW", "2412.TW", "2881.TW",
    "2891.TW", "2882.TW", "3711.TW", "2357.TW", "2308.TW", "2603.TW", "2618.TW",
    "3008.TW", "3034.TW", "2379.TW", "3661.TW", "3231.TW", "0050.TW", "006208.TW",
]

PERIOD_OPTIONS = {"3 個月": "3mo", "6 個月": "6mo", "1 年": "1y", "2 年": "2y", "5 年": "5y"}

RECOMMENDATION_MAP = {
    "strong_buy": "強力買進",
    "buy": "買進",
    "hold": "持有",
    "sell": "賣出",
    "strong_sell": "強力賣出",
    "none": "無資料",
}


def inject_mobile_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
        }
        div[data-testid="stMetricValue"] { font-size: 1.45rem; }
        div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
        @media (max-width: 720px) {
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
            div[data-testid="stMetric"] { padding: 10px 11px; }
            div[data-testid="stMetricValue"] { font-size: 1.15rem; }
            h1 { font-size: 1.65rem !important; }
            h2 { font-size: 1.35rem !important; }
            h3 { font-size: 1.12rem !important; }
            h4 { font-size: 1rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["NVDA", "MSFT", "TSM", "QQQ", "2330.TW"]


def get_app_password() -> str | None:
    password = None
    try:
        password = st.secrets.get("APP_PASSWORD")
    except Exception:
        password = None
    return password or os.environ.get("APP_PASSWORD")


def require_password_if_configured() -> None:
    password = get_app_password()
    if not password or st.session_state.get("authenticated"):
        return

    st.title("AI 交易參考系統")
    st.caption("請輸入存取密碼。")
    with st.form("password_form"):
        entered = st.text_input("密碼", type="password")
        submitted = st.form_submit_button("進入系統")

    if submitted:
        if hmac.compare_digest(entered, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密碼不正確。")
    st.stop()


def normalize_symbol(raw_symbol: str, market: str, tw_suffix: str) -> str:
    symbol = raw_symbol.strip().upper()
    if market == "台股" and symbol and "." not in symbol:
        return f"{symbol}.{tw_suffix}"
    return symbol


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df.loc[:, ~df.columns.duplicated()].copy()


@st.cache_data(ttl=300, show_spinner=False)
def load_price_data(symbol: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = flatten_columns(df)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in df.columns for column in required):
        return pd.DataFrame()

    df = df[required].copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"])


@st.cache_data(ttl=1800, show_spinner=False)
def load_analyst_snapshot(symbol: str) -> dict[str, Any]:
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}

    recommendation_key = str(info.get("recommendationKey") or "none").lower()
    current_price = safe_float(info.get("currentPrice", info.get("regularMarketPrice")))
    target_mean = safe_float(info.get("targetMeanPrice"))
    upside = np.nan
    if np.isfinite(current_price) and current_price > 0 and np.isfinite(target_mean):
        upside = target_mean / current_price - 1

    return {
        "recommendation_key": recommendation_key,
        "recommendation": RECOMMENDATION_MAP.get(recommendation_key, recommendation_key or "無資料"),
        "mean_rating": safe_float(info.get("recommendationMean")),
        "analyst_count": info.get("numberOfAnalystOpinions") or "-",
        "current_price": current_price,
        "target_mean": target_mean,
        "target_median": safe_float(info.get("targetMedianPrice")),
        "target_high": safe_float(info.get("targetHighPrice")),
        "target_low": safe_float(info.get("targetLowPrice")),
        "upside": upside,
    }


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def money(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:,.2f}"


def compact_money(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_0000_0000_0000:
        return f"{sign}{value / 1_0000_0000_0000:.2f} 兆"
    if value >= 1_0000_0000:
        return f"{sign}{value / 1_0000_0000:.2f} 億"
    if value >= 1_0000:
        return f"{sign}{value / 1_0000:.2f} 萬"
    return f"{sign}{value:,.0f}"


def pct(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2%}"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["Close"]
    high = result["High"]
    low = result["Low"]
    volume = result["Volume"].fillna(0)
    typical_price = (high + low + close) / 3

    for window in [5, 20, 60, 120]:
        result[f"MA{window}"] = close.rolling(window).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_SIGNAL"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD_HIST"] = result["MACD"] - result["MACD_SIGNAL"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    result["ATR14"] = true_range.rolling(14).mean()
    result["VOL_MA20"] = volume.rolling(20).mean()
    result["RETURN_1D"] = close.pct_change()
    result["RETURN_20D"] = close.pct_change(20)
    result["HIGH_20D"] = high.rolling(20).max()
    result["LOW_20D"] = low.rolling(20).min()

    result["TYPICAL_PRICE"] = typical_price
    result["RAW_MONEY_FLOW"] = typical_price * volume
    flow_sign = np.sign(close.diff()).replace(0, np.nan).ffill().fillna(0)
    result["SIGNED_MONEY_FLOW"] = result["RAW_MONEY_FLOW"] * flow_sign
    result["NET_FLOW_5D"] = result["SIGNED_MONEY_FLOW"].rolling(5).sum()
    result["NET_FLOW_20D"] = result["SIGNED_MONEY_FLOW"].rolling(20).sum()
    result["FLOW_RATIO_20D"] = result["NET_FLOW_20D"] / result["RAW_MONEY_FLOW"].rolling(20).sum().replace(0, np.nan)
    result["VWAP20"] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    result["VWAP60"] = (typical_price * volume).rolling(60).sum() / volume.rolling(60).sum().replace(0, np.nan)

    positive_flow = result["RAW_MONEY_FLOW"].where(typical_price > typical_price.shift(1), 0)
    negative_flow = result["RAW_MONEY_FLOW"].where(typical_price < typical_price.shift(1), 0)
    money_flow_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum().replace(0, np.nan)
    result["MFI14"] = 100 - (100 / (1 + money_flow_ratio))
    result["OBV"] = (flow_sign * volume).cumsum()
    return result


def score_stock(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or len(df) < 25:
        return {"score": 50, "rating": "資料不足", "bias": "觀望", "color": "#64748b", "reasons": ["資料筆數不足。"]}

    data = compute_indicators(df)
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    close = safe_float(latest["Close"])
    ma5 = safe_float(latest["MA5"])
    ma20 = safe_float(latest["MA20"])
    ma60 = safe_float(latest["MA60"])
    rsi = safe_float(latest["RSI14"])
    macd = safe_float(latest["MACD"])
    macd_signal = safe_float(latest["MACD_SIGNAL"])
    volume = safe_float(latest["Volume"], 0)
    volume_ma20 = safe_float(latest["VOL_MA20"], 0)
    return_20d = safe_float(latest["RETURN_20D"], 0)
    high_20d = safe_float(latest["HIGH_20D"])
    low_20d = safe_float(latest["LOW_20D"])
    previous_close = safe_float(previous["Close"])
    flow_ratio = safe_float(latest["FLOW_RATIO_20D"], 0)

    score = 50
    reasons: list[str] = []
    if close > ma20:
        score += 10
        reasons.append("價格站上 20 日均線，短中期趨勢偏正向。")
    else:
        score -= 10
        reasons.append("價格低於 20 日均線，短中期動能需要保守看待。")

    if ma5 > ma20:
        score += 8
        reasons.append("5 日均線高於 20 日均線，短線買盤較有延續性。")
    else:
        score -= 5

    if ma20 > ma60:
        score += 12
        reasons.append("20 日均線高於 60 日均線，中期趨勢仍在多方結構。")
    else:
        score -= 8
        reasons.append("20 日均線低於 60 日均線，中期結構尚未轉強。")

    if macd > macd_signal:
        score += 8
        reasons.append("MACD 在訊號線上方，動能指標偏多。")
    else:
        score -= 6

    if 45 <= rsi <= 68:
        score += 8
        reasons.append("RSI 位於健康區間，尚未明顯過熱。")
    elif rsi > 75:
        score -= 12
        reasons.append("RSI 偏高，追價風險上升。")
    elif rsi < 35:
        score -= 4
        reasons.append("RSI 偏弱，反彈前需要確認買盤回來。")

    if volume_ma20 > 0:
        volume_ratio = volume / volume_ma20
        if volume_ratio >= 1.25 and close >= previous_close:
            score += 8
            reasons.append("量能放大且收高，買盤參與度提升。")
        elif volume_ratio >= 1.25 and close < previous_close:
            score -= 8
            reasons.append("量能放大但價格收低，可能有賣壓釋放。")

    if flow_ratio > 0.08:
        score += 7
        reasons.append("20 日資金流向偏流入，籌碼面加分。")
    elif flow_ratio < -0.08:
        score -= 7
        reasons.append("20 日資金流向偏流出，籌碼面保守。")

    if return_20d > 0.15:
        score -= 5
        reasons.append("近 20 日漲幅較大，短線容易震盪。")
    elif return_20d < -0.12:
        score -= 5
        reasons.append("近 20 日跌幅較深，需等待止跌訊號。")

    if np.isfinite(high_20d) and np.isfinite(low_20d) and high_20d > low_20d:
        range_position = (close - low_20d) / (high_20d - low_20d)
        if range_position > 0.85:
            score += 3
        elif range_position < 0.2:
            score -= 6
            reasons.append("價格接近 20 日區間低位，防守優先。")

    score = int(max(0, min(100, round(score))))
    if score >= 80:
        return {"score": score, "rating": "強勢偏多", "bias": "可列入優先觀察", "color": "#16a34a", "reasons": reasons[:6]}
    if score >= 65:
        return {"score": score, "rating": "偏多觀察", "bias": "等待合理進場點", "color": "#65a30d", "reasons": reasons[:6]}
    if score >= 45:
        return {"score": score, "rating": "中性震盪", "bias": "先看區間，不急追價", "color": "#ca8a04", "reasons": reasons[:6]}
    return {"score": score, "rating": "偏弱防守", "bias": "降低部位或等待轉強", "color": "#dc2626", "reasons": reasons[:6]}


def build_trade_plan(df: pd.DataFrame, capital: float, risk_percent: float) -> dict[str, Any]:
    data = compute_indicators(df)
    latest = data.iloc[-1]
    close = safe_float(latest["Close"])
    atr = safe_float(latest["ATR14"])
    ma20 = safe_float(latest["MA20"])
    high_20d = safe_float(latest["HIGH_20D"])
    if not np.isfinite(atr) or atr <= 0:
        atr = close * 0.03

    pullback_entry = close - atr * 0.45
    buy_low = max(0, pullback_entry - atr * 0.4)
    buy_high = pullback_entry + atr * 0.25
    breakout_price = high_20d if np.isfinite(high_20d) else close + atr
    base_stop = min(close - atr * 1.5, ma20 - atr * 0.8 if np.isfinite(ma20) else close - atr * 1.5)
    stop_loss = max(0, base_stop)
    risk_per_share = max(pullback_entry - stop_loss, close * 0.005)
    target_1 = pullback_entry + risk_per_share * 1.5
    target_2 = pullback_entry + risk_per_share * 2.5
    risk_budget = capital * (risk_percent / 100)
    raw_quantity = math.floor(risk_budget / risk_per_share) if risk_per_share > 0 else 0
    capital_limited_quantity = math.floor(capital / max(pullback_entry, 0.01))

    return {
        "buy_low": buy_low,
        "buy_high": buy_high,
        "breakout_price": breakout_price,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": risk_per_share,
        "risk_budget": risk_budget,
        "quantity": max(0, min(raw_quantity, capital_limited_quantity)),
    }


def describe_trend(df: pd.DataFrame) -> str:
    data = compute_indicators(df)
    latest = data.iloc[-1]
    close = safe_float(latest["Close"])
    ma5 = safe_float(latest["MA5"])
    ma20 = safe_float(latest["MA20"])
    ma60 = safe_float(latest["MA60"])
    if close > ma5 > ma20 > ma60:
        return "多頭排列"
    if close < ma5 < ma20 < ma60:
        return "空頭排列"
    if close > ma20 and ma20 > ma60:
        return "中期偏多"
    if close < ma20 and ma20 < ma60:
        return "中期偏弱"
    return "區間整理"


def analyze_money_flow(df: pd.DataFrame) -> dict[str, Any]:
    data = compute_indicators(df)
    latest = data.iloc[-1]
    net_5d = safe_float(latest["NET_FLOW_5D"], 0)
    net_20d = safe_float(latest["NET_FLOW_20D"], 0)
    ratio_20d = safe_float(latest["FLOW_RATIO_20D"], 0)
    mfi = safe_float(latest["MFI14"])

    if net_5d > 0 and net_20d > 0:
        status, color = "資金持續流入", "#16a34a"
    elif net_5d > 0 and net_20d <= 0:
        status, color = "短線資金回補", "#65a30d"
    elif net_5d < 0 and net_20d < 0:
        status, color = "資金持續流出", "#dc2626"
    else:
        status, color = "資金動能轉弱", "#ca8a04"

    if abs(ratio_20d) >= 0.18:
        strength = "強"
    elif abs(ratio_20d) >= 0.08:
        strength = "中"
    else:
        strength = "弱"

    return {
        "status": status,
        "color": color,
        "strength": strength,
        "net_5d": net_5d,
        "net_20d": net_20d,
        "ratio_20d": ratio_20d,
        "mfi": mfi,
    }


def describe_flow(df: pd.DataFrame) -> str:
    return analyze_money_flow(df)["status"]


def compute_chip_distribution(df: pd.DataFrame, lookback: int = 60) -> dict[str, Any]:
    data = compute_indicators(df)
    recent = data.tail(lookback).dropna(subset=["Close", "Volume"]).copy()
    latest = data.iloc[-1]
    close = safe_float(latest["Close"])
    atr = safe_float(latest["ATR14"], close * 0.03)
    vwap20 = safe_float(latest["VWAP20"])
    vwap60 = safe_float(latest["VWAP60"])

    if recent.empty:
        return {
            "lookback": lookback,
            "peak_zone": "-",
            "peak_mid": np.nan,
            "chip_bias": "資料不足",
            "below_pct": np.nan,
            "near_pct": np.nan,
            "above_pct": np.nan,
            "accumulation_days": 0,
            "distribution_days": 0,
            "vwap20": vwap20,
            "vwap60": vwap60,
            "atr": atr,
            "profile": pd.DataFrame(),
        }

    low = safe_float(recent["Low"].min())
    high = safe_float(recent["High"].max())
    bins = np.linspace(low, high, 9)
    if len(np.unique(bins)) < 2:
        bins = np.linspace(close * 0.95, close * 1.05, 9)

    bucket = pd.cut(recent["Close"], bins=bins, include_lowest=True)
    profile = (
        recent.groupby(bucket, observed=False)["Volume"]
        .sum()
        .reset_index()
        .rename(columns={"Close": "價格區間", "Volume": "成交量"})
    )
    total_volume = safe_float(profile["成交量"].sum(), 0)
    profile["佔比%"] = profile["成交量"] / total_volume * 100 if total_volume > 0 else 0
    top_row = profile.sort_values("成交量", ascending=False).iloc[0]
    peak_interval = top_row["價格區間"]
    peak_mid = safe_float(getattr(peak_interval, "mid", np.nan))
    peak_zone = f"{money(getattr(peak_interval, 'left', np.nan))} - {money(getattr(peak_interval, 'right', np.nan))}"

    below = safe_float(recent.loc[recent["Close"] < close * 0.98, "Volume"].sum(), 0)
    near = safe_float(recent.loc[recent["Close"].between(close * 0.98, close * 1.02), "Volume"].sum(), 0)
    above = safe_float(recent.loc[recent["Close"] > close * 1.02, "Volume"].sum(), 0)
    denom = below + near + above
    accumulation_days = int(((recent["Close"] > recent["Close"].shift(1)) & (recent["Volume"] > recent["VOL_MA20"])).sum())
    distribution_days = int(((recent["Close"] < recent["Close"].shift(1)) & (recent["Volume"] > recent["VOL_MA20"])).sum())
    top_two_pct = safe_float(profile.sort_values("成交量", ascending=False).head(2)["佔比%"].sum(), 0)

    if top_two_pct >= 48 and accumulation_days >= distribution_days:
        chip_bias = "籌碼偏集中，且放量上漲天數較多"
    elif distribution_days > accumulation_days + 2:
        chip_bias = "籌碼有鬆動跡象，放量下跌天數偏多"
    else:
        chip_bias = "籌碼分佈中性，仍需等待方向"

    profile["價格區間"] = profile["價格區間"].astype(str)
    profile["成交量"] = profile["成交量"].round(0).astype("Int64")
    profile["佔比%"] = profile["佔比%"].round(2)

    return {
        "lookback": lookback,
        "peak_zone": peak_zone,
        "peak_mid": peak_mid,
        "chip_bias": chip_bias,
        "below_pct": below / denom if denom > 0 else np.nan,
        "near_pct": near / denom if denom > 0 else np.nan,
        "above_pct": above / denom if denom > 0 else np.nan,
        "accumulation_days": accumulation_days,
        "distribution_days": distribution_days,
        "vwap20": vwap20,
        "vwap60": vwap60,
        "atr": atr,
        "profile": profile,
    }


def estimate_institutional_entry(df: pd.DataFrame) -> dict[str, Any]:
    data = compute_indicators(df)
    recent = data.tail(80).copy()
    latest = data.iloc[-1]
    close = safe_float(latest["Close"])
    atr = safe_float(latest["ATR14"], close * 0.03)
    ma20 = safe_float(latest["MA20"])
    high_20d = safe_float(latest["HIGH_20D"])
    vwap20 = safe_float(latest["VWAP20"])

    signal_days = recent[
        (recent["Volume"] >= recent["VOL_MA20"] * 1.35)
        & (recent["Close"] > recent["Open"])
        & (recent["Close"] >= recent["MA20"])
    ].dropna(subset=["TYPICAL_PRICE", "Volume"])

    if not signal_days.empty and safe_float(signal_days["Volume"].sum(), 0) > 0:
        entry_price = safe_float(np.average(signal_days["TYPICAL_PRICE"], weights=signal_days["Volume"]))
        active_days = len(signal_days)
        avg_volume_ratio = safe_float((signal_days["Volume"] / signal_days["VOL_MA20"]).mean(), 0)
        confidence = min(95, max(45, round(45 + active_days * 5 + avg_volume_ratio * 10)))
        source = "近 80 日放量上漲日加權均價"
    else:
        entry_price = vwap20 if np.isfinite(vwap20) else close
        active_days = 0
        confidence = 40
        source = "近 20 日 VWAP 估算"

    zone_low = max(0, entry_price - atr * 0.55)
    zone_high = entry_price + atr * 0.55
    stop_reference = max(0, min(zone_low - atr * 0.45, ma20 - atr * 0.7 if np.isfinite(ma20) else zone_low - atr * 0.45))
    return {
        "entry_price": entry_price,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "breakout_reference": high_20d if np.isfinite(high_20d) else close + atr,
        "stop_reference": stop_reference,
        "confidence": confidence,
        "active_days": active_days,
        "source": source,
    }


def build_entry_signal(df: pd.DataFrame, score: dict[str, Any], flow: dict[str, Any], plan: dict[str, Any], institution: dict[str, Any]) -> dict[str, Any]:
    data = compute_indicators(df)
    latest = data.iloc[-1]
    close = safe_float(latest["Close"])
    ma20 = safe_float(latest["MA20"])
    ma60 = safe_float(latest["MA60"])
    rsi = safe_float(latest["RSI14"])
    macd = safe_float(latest["MACD"])
    macd_signal = safe_float(latest["MACD_SIGNAL"])
    high_20d = safe_float(latest["HIGH_20D"])

    trend_ok = close > ma20 and ma20 >= ma60
    momentum_ok = macd > macd_signal
    flow_ok = flow["net_5d"] > 0 and flow["ratio_20d"] > -0.03
    rsi_ok = 42 <= rsi <= 70
    near_plan = plan["buy_low"] <= close <= plan["buy_high"] * 1.025
    near_institution = institution["zone_low"] <= close <= institution["zone_high"] * 1.035
    breakout_ok = np.isfinite(high_20d) and close >= high_20d * 0.995
    too_hot = (np.isfinite(ma20) and close > ma20 * 1.1) or rsi > 75

    reasons = []
    if trend_ok:
        reasons.append("趨勢站在 20/60 日均線上方")
    if momentum_ok:
        reasons.append("MACD 動能偏多")
    if flow_ok:
        reasons.append("短線資金流向支持")
    if near_plan:
        reasons.append("價格接近拉回買進區")
    if near_institution:
        reasons.append("價格接近機構進場位推估區")
    if too_hot:
        reasons.append("短線漲幅或 RSI 偏熱")

    if score["score"] >= 75 and trend_ok and momentum_ok and flow_ok and rsi_ok and not too_hot and (near_plan or near_institution):
        return {"signal": "可分批進場", "action": "用 2 到 3 批建立部位，停損放在計畫停損下方。", "color": "#16a34a", "reasons": reasons[:5]}
    if score["score"] >= 72 and trend_ok and momentum_ok and flow_ok and breakout_ok and not too_hot:
        return {"signal": "突破可追蹤", "action": "等收盤站穩突破價，隔日不破可小部位試單。", "color": "#65a30d", "reasons": reasons[:5]}
    if score["score"] >= 60 and trend_ok:
        return {"signal": "等待拉回", "action": "不急追價，等回到買進區或機構成本區附近再觀察。", "color": "#ca8a04", "reasons": reasons[:5]}
    if score["score"] >= 45:
        return {"signal": "觀望整理", "action": "先看區間，等待量能與均線重新轉強。", "color": "#64748b", "reasons": reasons[:5]}
    return {"signal": "暫不進場", "action": "偏弱時先保留現金，等趨勢轉強再評估。", "color": "#dc2626", "reasons": reasons[:5]}


def scan_symbol(symbol: str, period: str, include_analyst: bool) -> dict[str, Any] | None:
    df = load_price_data(symbol, period)
    if df.empty or len(df) < 25:
        return None
    data = compute_indicators(df)
    latest = data.iloc[-1]
    score = score_stock(df)
    flow = analyze_money_flow(df)
    plan = build_trade_plan(df, 100_000, 1)
    institution = estimate_institutional_entry(df)
    entry_signal = build_entry_signal(df, score, flow, plan, institution)
    close = safe_float(latest["Close"])
    previous_close = safe_float(data.iloc[-2]["Close"])
    day_change = (close / previous_close - 1) if previous_close else np.nan
    rsi = safe_float(latest["RSI14"])
    row: dict[str, Any] = {
        "代號": symbol,
        "現價": round(close, 2),
        "日漲跌%": round(day_change * 100, 2) if np.isfinite(day_change) else np.nan,
        "AI分數": score["score"],
        "可進場訊號": entry_signal["signal"],
        "趨勢": describe_trend(df),
        "資金流向": flow["status"],
        "5日淨流": compact_money(flow["net_5d"]),
        "機構進場位": round(institution["entry_price"], 2),
        "RSI": round(rsi, 1) if np.isfinite(rsi) else np.nan,
    }
    if include_analyst:
        analyst = load_analyst_snapshot(symbol)
        row["分析師評級"] = analyst["recommendation"]
        row["目標價空間%"] = round(analyst["upside"] * 100, 2) if np.isfinite(analyst["upside"]) else np.nan
    return row


def build_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    data = compute_indicators(df)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.54, 0.17, 0.15, 0.14],
        vertical_spacing=0.025,
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="K 線",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )
    for name, color in [("MA5", "#f59e0b"), ("MA20", "#2563eb"), ("MA60", "#7c3aed"), ("VWAP20", "#0f766e")]:
        fig.add_trace(go.Scatter(x=data.index, y=data[name], mode="lines", line={"width": 1.6, "color": color}, name=name), row=1, col=1)

    volume_colors = np.where(data["Close"] >= data["Open"], "#16a34a", "#dc2626")
    fig.add_trace(go.Bar(x=data.index, y=data["Volume"], marker_color=volume_colors, name="成交量"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["VOL_MA20"], mode="lines", line={"width": 1.4, "color": "#475569"}, name="量均線"), row=2, col=1)
    fig.add_trace(
        go.Bar(x=data.index, y=data["NET_FLOW_5D"], marker_color=np.where(data["NET_FLOW_5D"] >= 0, "#16a34a", "#dc2626"), name="5日資金淨流"),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8", row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI14"], mode="lines", line={"width": 1.5, "color": "#0891b2"}, name="RSI14"), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MFI14"], mode="lines", line={"width": 1.2, "color": "#9333ea"}, name="MFI14"), row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16a34a", row=4, col=1)
    fig.update_layout(
        title=f"{symbol} 技術與資金走勢",
        height=820,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        template="plotly_white",
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="量", row=2, col=1)
    fig.update_yaxes(title_text="資金", row=3, col=1)
    fig.update_yaxes(title_text="RSI/MFI", range=[0, 100], row=4, col=1)
    return fig


def metric_delta(value: float) -> str | None:
    if not np.isfinite(value):
        return None
    return f"{value:.2%}"


def render_badge(title: str, value: str, detail: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="border:1px solid #e2e8f0;border-left:6px solid {color};
                    padding:14px 16px;border-radius:8px;background:#ffffff;margin-bottom:12px;">
            <div style="font-size:13px;color:#64748b;">{title}</div>
            <div style="font-size:24px;font-weight:700;color:{color};line-height:1.18;">{value}</div>
            <div style="font-size:14px;color:#475569;margin-top:4px;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analyst_section(symbol: str, close: float) -> dict[str, Any]:
    analyst = load_analyst_snapshot(symbol)
    st.markdown("#### 分析師評級")
    if analyst["recommendation_key"] == "none" and not np.isfinite(analyst["target_mean"]):
        st.info("Yahoo Finance 目前沒有提供這個代號的分析師評級。台股與 ETF 較常出現缺資料。")
        return analyst
    cols = st.columns(4)
    cols[0].metric("評級", analyst["recommendation"])
    cols[1].metric("分析師數", analyst["analyst_count"])
    cols[2].metric("平均目標價", money(analyst["target_mean"]))
    cols[3].metric("目標價空間", pct(analyst["upside"]))
    table = pd.DataFrame(
        [
            ["目前價格", money(close)],
            ["平均目標價", money(analyst["target_mean"])],
            ["中位目標價", money(analyst["target_median"])],
            ["最高目標價", money(analyst["target_high"])],
            ["最低目標價", money(analyst["target_low"])],
            ["評級均值", money(analyst["mean_rating"])],
        ],
        columns=["項目", "數值"],
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("分析師資料來自 yfinance 可取得欄位；不同市場與商品可能不完整。")
    return analyst


def render_chip_and_flow(chip: dict[str, Any], flow: dict[str, Any], institution: dict[str, Any]) -> None:
    st.markdown("#### 籌碼分佈與資金流")
    cols = st.columns(4)
    cols[0].metric("60日最大量區", chip["peak_zone"])
    cols[1].metric("20日VWAP", money(chip["vwap20"]))
    cols[2].metric("5日淨流", compact_money(flow["net_5d"]))
    cols[3].metric("20日淨流", compact_money(flow["net_20d"]), pct(flow["ratio_20d"]))
    st.write(chip["chip_bias"])
    dist = pd.DataFrame(
        [
            ["現價下方籌碼", pct(chip["below_pct"])],
            ["現價附近籌碼", pct(chip["near_pct"])],
            ["現價上方壓力", pct(chip["above_pct"])],
            ["放量上漲天數", chip["accumulation_days"]],
            ["放量下跌天數", chip["distribution_days"]],
            ["MFI 資金指標", money(flow["mfi"])],
        ],
        columns=["項目", "數值"],
    )
    st.dataframe(dist, hide_index=True, use_container_width=True)
    st.markdown("#### 機構進場位推估")
    inst_table = pd.DataFrame(
        [
            ["推估成本", money(institution["entry_price"])],
            ["進場觀察區", f"{money(institution['zone_low'])} - {money(institution['zone_high'])}"],
            ["突破確認價", money(institution["breakout_reference"])],
            ["防守參考", money(institution["stop_reference"])],
            ["信心分數", f"{institution['confidence']} / 100"],
            ["放量上漲樣本", institution["active_days"]],
            ["估算來源", institution["source"]],
        ],
        columns=["項目", "數值"],
    )
    st.dataframe(inst_table, hide_index=True, use_container_width=True)
    st.caption("機構進場位是用放量上漲日與 VWAP 推估，不等於真實法人持倉成本。")
    with st.expander("查看成交量價格分佈"):
        st.dataframe(chip["profile"], hide_index=True, use_container_width=True)


def render_single_stock(symbol: str, period: str, capital: float, risk_percent: float) -> None:
    df = load_price_data(symbol, period)
    if df.empty:
        st.error("查不到資料，請確認代號是否正確，或稍後再試。")
        return

    data = compute_indicators(df)
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) >= 2 else latest
    close = safe_float(latest["Close"])
    previous_close = safe_float(previous["Close"])
    day_change = (close / previous_close - 1) if previous_close else np.nan
    rsi = safe_float(latest["RSI14"])
    atr = safe_float(latest["ATR14"])
    score = score_stock(df)
    flow = analyze_money_flow(df)
    plan = build_trade_plan(df, capital, risk_percent)
    chip = compute_chip_distribution(df)
    institution = estimate_institutional_entry(df)
    entry_signal = build_entry_signal(df, score, flow, plan, institution)

    header_cols = st.columns(5)
    header_cols[0].metric("現價", money(close), metric_delta(day_change))
    header_cols[1].metric("趨勢", describe_trend(df))
    header_cols[2].metric("RSI 14", f"{rsi:.1f}" if np.isfinite(rsi) else "-")
    header_cols[3].metric("ATR 14", money(atr))
    header_cols[4].metric("資金流向", flow["status"])

    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        render_badge("AI 綜合評分", f"{score['score']} / 100", f"{score['rating']}，{score['bias']}", score["color"])
        render_badge("可進場訊號", entry_signal["signal"], entry_signal["action"], entry_signal["color"])
        st.markdown("#### 交易計畫參考")
        st.dataframe(
            pd.DataFrame(
                [
                    ["拉回買進區", f"{money(plan['buy_low'])} - {money(plan['buy_high'])}"],
                    ["突破觀察價", money(plan["breakout_price"])],
                    ["停損參考", money(plan["stop_loss"])],
                    ["目標一", money(plan["target_1"])],
                    ["目標二", money(plan["target_2"])],
                    ["單位風險", money(plan["risk_per_share"])],
                    ["建議股數", f"{plan['quantity']:,}"],
                ],
                columns=["項目", "數值"],
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("#### 判斷依據")
        for reason in score["reasons"]:
            st.write(f"- {reason}")
        for reason in entry_signal["reasons"]:
            st.write(f"- {reason}")

    with right:
        st.plotly_chart(build_chart(df, symbol), use_container_width=True)

    detail_tabs = st.tabs(["分析師評級", "籌碼資金", "最近資料"])
    with detail_tabs[0]:
        render_analyst_section(symbol, close)
    with detail_tabs[1]:
        render_chip_and_flow(chip, flow, institution)
    with detail_tabs[2]:
        export = data.tail(120).copy()
        export.index = export.index.strftime("%Y-%m-%d")
        st.dataframe(export.round(2), use_container_width=True)
        st.download_button(
            "下載最近 120 筆資料 CSV",
            data=export.round(4).to_csv(index=True).encode("utf-8-sig"),
            file_name=f"{symbol}_price_data.csv",
            mime="text/csv",
            key=f"download_price_{symbol}",
        )


def render_scan(symbols: list[str], period: str, title: str, include_analyst: bool) -> None:
    st.subheader(title)
    if not symbols:
        st.info("目前沒有代號可以掃描。")
        return
    progress = st.progress(0)
    rows = []
    for index, symbol in enumerate(symbols, start=1):
        row = scan_symbol(symbol, period, include_analyst)
        if row:
            rows.append(row)
        progress.progress(index / len(symbols))
    progress.empty()
    if not rows:
        st.warning("目前沒有可用資料。")
        return
    table = pd.DataFrame(rows).sort_values(["AI分數", "日漲跌%"], ascending=[False, False])
    config: dict[str, Any] = {
        "現價": st.column_config.NumberColumn(format="%.2f"),
        "日漲跌%": st.column_config.NumberColumn(format="%.2f%%"),
        "AI分數": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
        "機構進場位": st.column_config.NumberColumn(format="%.2f"),
        "RSI": st.column_config.NumberColumn(format="%.1f"),
    }
    if include_analyst:
        config["目標價空間%"] = st.column_config.NumberColumn(format="%.2f%%")
    st.dataframe(table, hide_index=True, use_container_width=True, column_config=config)
    st.download_button(
        "下載掃描結果 CSV",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name="ai_market_scan.csv",
        mime="text/csv",
        key=f"download_scan_{title}_{include_analyst}",
    )


def render_mobile_guide() -> None:
    st.subheader("手機上可以使用嗎")
    st.write("可以。這套系統是網頁工具，只要電腦在執行，手機就能用瀏覽器開啟。")
    st.write("同一個 Wi-Fi 下最簡單：先在電腦點 `run.bat`，再用手機開電腦的區網網址。")
    st.markdown(
        """
        1. 電腦先執行 `run.bat`。
        2. 電腦查自己的 IPv4 位址，例如 `192.168.1.23`。
        3. 手機連同一個 Wi-Fi。
        4. 手機瀏覽器輸入 `http://192.168.1.23:8501`。
        5. 如果 Windows 防火牆詢問，允許私人網路連線。
        """
    )
    st.info("如果要在外面也能用，建議部署到 Streamlit Community Cloud、VPS，或使用安全的內網穿透服務。")


def main() -> None:
    init_state()
    inject_mobile_style()
    require_password_if_configured()
    st.title("AI 交易參考系統")
    st.caption("以價格、均線、動能、量能、籌碼推估、資金流與風險控管做綜合評估；結果僅供研究與紀錄，不是投資建議。")

    with st.sidebar:
        st.header("設定")
        market = st.radio("市場", ["美股/ETF", "台股"], horizontal=True)
        tw_suffix = st.selectbox("台股類型", ["TW", "TWO"], index=0, disabled=market != "台股")
        default_symbol = "NVDA" if market == "美股/ETF" else "2330"
        raw_symbol = st.text_input("股票代號", default_symbol)
        symbol = normalize_symbol(raw_symbol, market, tw_suffix)
        period_label = st.selectbox("資料區間", list(PERIOD_OPTIONS.keys()), index=2)
        period = PERIOD_OPTIONS[period_label]
        include_analyst_scan = st.checkbox("掃描加入分析師評級（較慢）", value=False)

        st.divider()
        st.subheader("風險控管")
        capital = st.number_input("可投入資金", min_value=1000.0, max_value=100_000_000.0, value=100_000.0, step=5000.0)
        risk_percent = st.slider("單筆最大風險 (%)", 0.5, 5.0, 1.0, 0.5)

        st.divider()
        watch_cols = st.columns(2)
        if watch_cols[0].button("加入觀察", use_container_width=True) and symbol:
            if symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(symbol)
                st.toast(f"已加入 {symbol}")
        if watch_cols[1].button("更新資料", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if st.session_state.watchlist:
            remove_symbol = st.selectbox("移除觀察", ["不移除"] + st.session_state.watchlist)
            if st.button("移除選取代號", use_container_width=True) and remove_symbol != "不移除":
                st.session_state.watchlist.remove(remove_symbol)
                st.rerun()

    tabs = st.tabs(["個股分析", "觀察清單", "市場掃描", "使用說明"])
    with tabs[0]:
        if symbol:
            render_single_stock(symbol, period, capital, risk_percent)
        else:
            st.info("請輸入股票代號。")
    with tabs[1]:
        render_scan(st.session_state.watchlist, period, "我的觀察清單", include_analyst_scan)
    with tabs[2]:
        universe = US_UNIVERSE if market == "美股/ETF" else TW_UNIVERSE
        render_scan(universe, period, f"{market} 參考清單掃描", include_analyst_scan)
    with tabs[3]:
        st.subheader("這個工具怎麼看")
        st.write(
            "AI 分數是規則式量化評分，會綜合均線結構、MACD、RSI、量能、資金流與區間位置。"
            "分數越高代表技術面條件越完整，但仍需要搭配基本面、消息面與個人風險承受度。"
        )
        st.write(
            "籌碼分佈、資金淨流出入與機構進場位是由成交量、價格位置、VWAP 與放量上漲日推估。"
            "這些不是交易所公布的法人明細，但可以用來判斷市場是否有資金正在靠近。"
        )
        st.write(
            "可進場訊號會同時檢查趨勢、動能、資金流、RSI 熱度、拉回買進區與機構成本區。"
            "若訊號顯示等待，通常代表條件還不完整，不代表標的不好。"
        )
        st.info("提醒：任何模型都可能失準，請把它當成整理資訊與建立紀律的參考工具，而不是自動下單訊號。")
        render_mobile_guide()


if __name__ == "__main__":
    main()
