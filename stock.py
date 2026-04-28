import os
import asyncio
import json
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

load_dotenv()

# ================================
# LOAD KEY
# ================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="NSE Trading AI", layout="wide", page_icon="📈")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');

.metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6c757d;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 20px;
    font-weight: 600;
    color: #212529;
    font-family: 'DM Mono', monospace;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-bullish  { background: #d4edda; color: #155724; }
.badge-bearish  { background: #f8d7da; color: #721c24; }
.badge-neutral  { background: #fff3cd; color: #856404; }
.badge-buy      { background: #d4edda; color: #155724; }
.badge-sell     { background: #f8d7da; color: #721c24; }
.badge-hold     { background: #fff3cd; color: #856404; }
.badge-avoid    { background: #e2e3e5; color: #383d41; }
.badge-high     { background: #d4edda; color: #155724; }
.badge-medium   { background: #fff3cd; color: #856404; }
.badge-low      { background: #f8d7da; color: #721c24; }
.section-header {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6c757d;
    font-weight: 600;
    margin: 16px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e9ecef;
}
.level-resistance { color: #dc3545; font-family: 'DM Mono', monospace; font-size: 13px; }
.level-support    { color: #28a745; font-family: 'DM Mono', monospace; font-size: 13px; }
.headline-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 13px;
}
.rec-action {
    font-size: 36px;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    padding: 8px 20px;
    border-radius: 10px;
    display: inline-block;
    margin-bottom: 12px;
}
.disclaimer {
    font-size: 11px;
    color: #999;
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #eee;
}
.price-big {
    font-size: 32px;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    color: #212529;
}
</style>
""", unsafe_allow_html=True)


# ================================
# STATE
# ================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    stock_symbol: str
    stock_data: str
    news_data: str
    market_analysis: str
    final_recommendation: str


# ================================
# SYMBOL EXTRACTOR
# ================================
def extract_symbol(query: str) -> str:
    known = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "SBIN", "WIPRO", "AXISBANK", "BAJFINANCE", "HINDUNILVR",
        "ITC", "KOTAKBANK", "LT", "MARUTI", "NTPC", "ONGC",
        "POWERGRID", "SUNPHARMA", "TATAMOTORS", "TATASTEEL",
        "TECHM", "TITAN", "ULTRACEMCO", "ADANIENT", "ADANIPORTS",
        "APOLLOHOSP", "ASIANPAINT", "BAJAJFINSV", "BPCL", "BRITANNIA",
        "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
        "GRASIM", "HCLTECH", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
        "INDUSINDBK", "JSWSTEEL", "NESTLEIND", "SBILIFE",
        "SHREECEM", "TATACONSUM", "UPL", "VEDL", "ZOMATO", "PAYTM",
    ]
    upper = query.upper()
    for sym in known:
        if sym in upper:
            return sym
    words = query.split()
    for w in reversed(words):
        if w.isupper() and len(w) >= 2:
            return w
    return words[-1].upper() if words else "RELIANCE"


# ================================
# INDICATORS
# ================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()


# ================================
# TOOLS
# ================================
@tool
def get_intraday_ohlcv(symbol: str) -> str:
    """Fetch 5-min intraday OHLCV with RSI and VWAP for an NSE stock. ~15min delayed."""
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.NS")
        df = ticker.history(period="1d", interval="5m")
        if df.empty:
            return f"No intraday data found for {symbol}."
        df["RSI"] = compute_rsi(df["Close"])
        df["VWAP"] = compute_vwap(df)
        df = df.tail(20)[["Open", "High", "Low", "Close", "Volume", "RSI", "VWAP"]].round(2)
        current_price = df["Close"].iloc[-1]
        rsi_val = df["RSI"].iloc[-1]
        vwap_val = df["VWAP"].iloc[-1]
        vol = df["Volume"].iloc[-1]
        avg_vol = df["Volume"].mean()
        return (
            f"Current Price: {current_price}\n"
            f"RSI(14): {rsi_val:.2f} ({'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'})\n"
            f"VWAP: {vwap_val:.2f} ({'Above' if current_price > vwap_val else 'Below'} VWAP)\n"
            f"Volume: {int(vol):,} ({'High' if vol > avg_vol * 1.5 else 'Normal'} vs avg {int(avg_vol):,})\n"
            f"Last 20 Candles:\n{df.to_string()}"
        )
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_support_resistance(symbol: str) -> str:
    """Calculate S/R levels for an NSE stock using 1hr candles over 5 days."""
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.NS")
        df = ticker.history(period="5d", interval="1h")
        if df.empty:
            return f"No data for {symbol}."
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        resistance = sorted(set([round(highs.max(), 2), round(np.percentile(highs, 75), 2), round(np.percentile(highs, 60), 2)]), reverse=True)
        support = sorted(set([round(lows.min(), 2), round(np.percentile(lows, 25), 2), round(np.percentile(lows, 40), 2)]))
        current = round(closes[-1], 2)
        sma_20 = round(pd.Series(closes).rolling(20).mean().iloc[-1], 2)
        trend = "Uptrend" if current > sma_20 else "Downtrend"
        today_mask = df.index.date == df.index[-1].date()
        day_high = round(df.loc[today_mask, "High"].max(), 2)
        day_low = round(df.loc[today_mask, "Low"].min(), 2)
        return f"Current: {current} | Day H/L: {day_high}/{day_low} | Trend: {trend} (SMA20={sma_20})\nResistance: {resistance}\nSupport: {support}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_stock_news(symbol: str) -> str:
    """Fetch latest NSE stock news from Google News RSS."""
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        if not feed.entries:
            return f"No news for {symbol}."
        bullish_kw = ["surge", "jump", "gain", "rally", "buy", "upgrade", "profit", "growth", "record", "beat", "strong", "positive", "up", "rise"]
        bearish_kw = ["fall", "drop", "decline", "sell", "downgrade", "loss", "weak", "miss", "negative", "down", "crash", "concern", "risk", "cut"]
        results = []
        for entry in feed.entries[:8]:
            title = entry.title
            lower = title.lower()
            if any(k in lower for k in bullish_kw):
                s = "BULLISH"
            elif any(k in lower for k in bearish_kw):
                s = "BEARISH"
            else:
                s = "NEUTRAL"
            results.append({"sentiment": s, "title": title})
        return json.dumps(results)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_fundamentals(symbol: str) -> str:
    """Get PE, market cap, 52-week range, beta for an NSE stock."""
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.NS")
        info = ticker.info
        return (
            f"Company: {info.get('longName', 'N/A')} | Sector: {info.get('sector', 'N/A')}\n"
            f"Market Cap: ₹{info.get('marketCap', 0) / 1e7:.0f} Cr | PE: {info.get('trailingPE', 'N/A')}\n"
            f"52W High: {info.get('fiftyTwoWeekHigh', 'N/A')} | 52W Low: {info.get('fiftyTwoWeekLow', 'N/A')}\n"
            f"Avg Volume: {info.get('averageVolume', 0):,} | Beta: {info.get('beta', 'N/A')}"
        )
    except Exception as e:
        return f"Error: {str(e)}"


# ================================
# MODEL
# ================================
@st.cache_resource
def get_model():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0)


# ================================
# TOOL AGENT
# ================================
def run_tool_agent(system_prompt: str, user_prompt: str, tools_list: list) -> str:
    model = get_model()
    model_with_tools = model.bind_tools(tools_list)
    tool_map = {t.name: t for t in tools_list}
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    for _ in range(6):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            if tc["name"] in tool_map:
                try:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {str(e)}"
            else:
                result = f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "{}"


def parse_json(text: str) -> dict:
    """Safely extract JSON from model output."""
    try:
        clean = text.strip()
        if "```" in clean:
            parts = clean.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                try:
                    return json.loads(p)
                except Exception:
                    continue
        return json.loads(clean)
    except Exception:
        return {"raw": text}


# ================================
# GRAPH NODES
# ================================
async def stock_finder_node(state: AgentState) -> dict:
    symbol = state["stock_symbol"]
    output = await asyncio.get_event_loop().run_in_executor(None, lambda: run_tool_agent(
        system_prompt=f"""You are an NSE intraday trading expert. Analyze {symbol}.
Return ONLY a valid JSON object with these exact keys:
- current_price (number)
- trend (string: "Uptrend" or "Downtrend")
- rsi (number)
- rsi_label (string: "Overbought", "Oversold", or "Neutral")
- vwap (number)
- vwap_signal (string: "Above VWAP" or "Below VWAP")
- volume_context (string)
- resistance_levels (list of 3 numbers)
- support_levels (list of 3 numbers)
- day_high (number)
- day_low (number)
- sma20 (number)
- company_name (string)
- sector (string)
- market_cap_cr (number)
- pe_ratio (number or null)
- week52_high (number)
- week52_low (number)
- beta (number or null)
- outlook (string, 2-3 sentences)
No extra text, just JSON.""",
        user_prompt=f"Analyze {symbol} for intraday trading today.",
        tools_list=[get_intraday_ohlcv, get_support_resistance, get_fundamentals],
    ))
    return {"messages": [AIMessage(content=output)], "stock_data": output}


async def market_data_node(state: AgentState) -> dict:
    symbol = state["stock_symbol"]
    output = await asyncio.get_event_loop().run_in_executor(None, lambda: run_tool_agent(
        system_prompt=f"""You are a technical analyst for NSE. Analyze {symbol}.
Return ONLY a valid JSON object with these exact keys:
- verdict (string: "BULLISH", "BEARISH", or "NEUTRAL")
- rsi_signal (string)
- vwap_signal (string)
- volume_signal (string)
- momentum (string)
- trend_strength (string: "Strong", "Moderate", or "Weak")
- key_observations (list of 4 strings, each a concise observation)
No extra text, just JSON.""",
        user_prompt=f"Full technical analysis for {symbol} intraday.",
        tools_list=[get_intraday_ohlcv, get_support_resistance],
    ))
    return {"messages": [AIMessage(content=output)], "market_analysis": output}


async def news_node(state: AgentState) -> dict:
    symbol = state["stock_symbol"]
    output = await asyncio.get_event_loop().run_in_executor(None, lambda: run_tool_agent(
        system_prompt=f"""You are a financial news analyst. Analyze {symbol} news.
Return ONLY a valid JSON object with these exact keys:
- overall_sentiment (string: "BULLISH", "BEARISH", or "NEUTRAL")
- bullish_count (number)
- bearish_count (number)
- neutral_count (number)
- headlines (list of objects, each with "title" and "sentiment" keys)
- summary (string, 2 sentences max)
No extra text, just JSON.""",
        user_prompt=f"Get and analyze latest news for {symbol}.",
        tools_list=[get_stock_news],
    ))
    return {"messages": [AIMessage(content=output)], "news_data": output}


async def recommendation_node(state: AgentState) -> dict:
    symbol = state["stock_symbol"]
    model = get_model()
    prompt = f"""You are a senior NSE intraday trading advisor. Based on this data for {symbol}:

STOCK DATA: {state.get("stock_data", "")}
TECHNICAL: {state.get("market_analysis", "")}
NEWS: {state.get("news_data", "")}

Return ONLY a valid JSON object with these exact keys:
- action (string: "BUY", "SELL", "HOLD", or "AVOID")
- entry_price (number)
- entry_range (string like "1355 - 1360")
- target_price (number)
- stop_loss (number)
- risk_reward (string like "1:1.5")
- confidence (string: "High", "Medium", or "Low")
- reasoning (string, 3-4 sentences)
- disclaimer (string: "For educational purposes only. Not SEBI-registered financial advice.")
No extra text, just JSON."""
    result = await asyncio.get_event_loop().run_in_executor(None, lambda: model.invoke(prompt))
    return {"messages": [AIMessage(content=result.content)], "final_recommendation": result.content}


# ================================
# GRAPH
# ================================
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("stock_finder", stock_finder_node)
    builder.add_node("market_data", market_data_node)
    builder.add_node("news", news_node)
    builder.add_node("recommendation", recommendation_node)
    builder.set_entry_point("stock_finder")
    builder.add_edge("stock_finder", "market_data")
    builder.add_edge("market_data", "news")
    builder.add_edge("news", "recommendation")
    builder.add_edge("recommendation", END)
    return builder.compile()


# ================================
# RENDER HELPERS
# ================================
def badge(label: str, kind: str) -> str:
    return f'<span class="badge badge-{kind.lower()}">{label}</span>'


def render_stock_overview(data: dict):
    price = data.get("current_price", "—")
    trend = data.get("trend", "—")
    trend_kind = "bullish" if "Up" in str(trend) else "bearish"

    st.markdown(f"""
    <div style="display:flex; align-items:baseline; gap:12px; margin-bottom:4px;">
        <span class="price-big">₹{price}</span>
        {badge(trend, trend_kind)}
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    rsi = data.get("rsi", "—")
    rsi_label = data.get("rsi_label", "")
    rsi_kind = "bearish" if rsi_label == "Overbought" else "bullish" if rsi_label == "Oversold" else "neutral"
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">RSI (14)</div><div class="metric-value">{rsi:.1f} if isinstance(rsi, float) else rsi</div></div>', unsafe_allow_html=True)
        st.markdown(badge(rsi_label, rsi_kind), unsafe_allow_html=True)
    with c2:
        vwap = data.get("vwap", "—")
        vsig = data.get("vwap_signal", "")
        vsig_kind = "bullish" if "Above" in str(vsig) else "bearish"
        st.markdown(f'<div class="metric-card"><div class="metric-label">VWAP</div><div class="metric-value">₹{vwap}</div></div>', unsafe_allow_html=True)
        st.markdown(badge(vsig, vsig_kind), unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Day High</div><div class="metric-value">₹{data.get("day_high","—")}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-label">Day Low</div><div class="metric-value">₹{data.get("day_low","—")}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">SMA 20</div><div class="metric-value">₹{data.get("sma20","—")}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><div class="metric-label">Volume</div><div class="metric-value">{data.get("volume_context","—")}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Support & Resistance</div>', unsafe_allow_html=True)
    r_col, s_col = st.columns(2)
    with r_col:
        st.markdown("**Resistance**")
        for r in data.get("resistance_levels", []):
            st.markdown(f'<div class="level-resistance">↑ ₹{r}</div>', unsafe_allow_html=True)
    with s_col:
        st.markdown("**Support**")
        for s in data.get("support_levels", []):
            st.markdown(f'<div class="level-support">↓ ₹{s}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Fundamentals</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Market Cap</div><div class="metric-value" style="font-size:16px">₹{data.get("market_cap_cr","—")} Cr</div></div>', unsafe_allow_html=True)
    with fc2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">P/E Ratio</div><div class="metric-value" style="font-size:16px">{data.get("pe_ratio","—")}</div></div>', unsafe_allow_html=True)
    with fc3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">52W High</div><div class="metric-value" style="font-size:16px">₹{data.get("week52_high","—")}</div></div>', unsafe_allow_html=True)
    with fc4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">52W Low</div><div class="metric-value" style="font-size:16px">₹{data.get("week52_low","—")}</div></div>', unsafe_allow_html=True)

    if data.get("outlook"):
        st.info(data["outlook"])


def render_technical(data: dict):
    verdict = data.get("verdict", "NEUTRAL")
    strength = data.get("trend_strength", "")
    v_kind = verdict.lower()
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f'<div style="text-align:center; padding:16px;">{badge(verdict, v_kind)}<br/><small style="color:#666">{strength}</small></div>', unsafe_allow_html=True)
    with col2:
        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, (label, key) in zip([mc1, mc2, mc3, mc4], [
            ("RSI", "rsi_signal"), ("VWAP", "vwap_signal"),
            ("Volume", "volume_signal"), ("Momentum", "momentum")
        ]):
            with col:
                val = data.get(key, "—")
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:13px; font-weight:500">{val}</div></div>', unsafe_allow_html=True)

    observations = data.get("key_observations", [])
    if observations:
        st.markdown('<div class="section-header">Key Observations</div>', unsafe_allow_html=True)
        for obs in observations:
            st.markdown(f"— {obs}")


def render_news(data: dict):
    overall = data.get("overall_sentiment", "NEUTRAL")
    o_kind = overall.lower()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card" style="text-align:center"><div class="metric-label">Overall</div>{badge(overall, o_kind)}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="background:#d4edda"><div class="metric-label">Bullish</div><div class="metric-value">{data.get("bullish_count",0)}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="background:#fff3cd"><div class="metric-label">Neutral</div><div class="metric-value">{data.get("neutral_count",0)}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card" style="background:#f8d7da"><div class="metric-label">Bearish</div><div class="metric-value">{data.get("bearish_count",0)}</div></div>', unsafe_allow_html=True)

    headlines = data.get("headlines", [])
    if headlines:
        st.markdown('<div class="section-header">Headlines</div>', unsafe_allow_html=True)
        for h in headlines:
            s = h.get("sentiment", "NEUTRAL")
            s_kind = s.lower()
            st.markdown(f'<div class="headline-row">{badge(s, s_kind)}<span>{h.get("title","")}</span></div>', unsafe_allow_html=True)

    if data.get("summary"):
        st.caption(data["summary"])


def render_recommendation(data: dict, symbol: str):
    action = data.get("action", "HOLD").upper()
    confidence = data.get("confidence", "Medium")
    action_colors = {
        "BUY":   ("#d4edda", "#155724"),
        "SELL":  ("#f8d7da", "#721c24"),
        "HOLD":  ("#fff3cd", "#856404"),
        "AVOID": ("#e2e3e5", "#383d41"),
    }
    bg, fg = action_colors.get(action, action_colors["HOLD"])

    left, right = st.columns([2, 3])
    with left:
        st.markdown(f"""
        <div style="background:{bg}; border-radius:12px; padding:20px; text-align:center">
            <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:{fg}; opacity:0.7; margin-bottom:6px">{symbol}</div>
            <div style="font-size:48px; font-weight:700; font-family:'DM Mono',monospace; color:{fg}">{action}</div>
            <div style="margin-top:8px">{badge(confidence + " Confidence", confidence.lower())}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        r1, r2 = st.columns(2)
        entry = data.get("entry_range") or (f"₹{data.get('entry_price','—')}")
        for col, (label, val) in zip([r1, r2, r1, r2], [
            ("Entry Range",   f"₹{entry}"),
            ("Target",        f"₹{data.get('target_price','—')}"),
            ("Stop Loss",     f"₹{data.get('stop_loss','—')}"),
            ("Risk / Reward", data.get("risk_reward", "—")),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{val}</div></div>', unsafe_allow_html=True)

    if data.get("reasoning"):
        st.markdown('<div class="section-header">Reasoning</div>', unsafe_allow_html=True)
        st.markdown(data["reasoning"])

    st.markdown(f'<div class="disclaimer">⚠ {data.get("disclaimer","For educational purposes only. Not SEBI-registered financial advice.")}</div>', unsafe_allow_html=True)


# ================================
# MAIN UI
# ================================
st.markdown("## 📈 NSE Intraday Trading AI")
st.caption("Multi-agent analysis · yfinance data (~15 min delayed) · Not SEBI-registered advice")
st.divider()

query = st.text_input(
    "Stock query",
    placeholder="e.g. RELIANCE  or  Give me intraday analysis for TCS",
    label_visibility="collapsed"
)
run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=False)

if run_btn:
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()
    if not query.strip():
        st.warning("Please enter a stock symbol or query.")
        st.stop()

    symbol = extract_symbol(query)
    st.markdown(f"Analyzing **{symbol}** &nbsp; `NSE:{symbol}`", unsafe_allow_html=True)
    st.divider()

    # Placeholders for streaming sections
    ph_overview  = st.empty()
    ph_technical = st.empty()
    ph_news      = st.empty()
    ph_rec       = st.empty()

    async def run_stream():
        graph = build_graph()
        init_state: AgentState = {
            "messages":             [HumanMessage(content=query)],
            "stock_symbol":         symbol,
            "stock_data":           "",
            "news_data":            "",
            "market_analysis":      "",
            "final_recommendation": "",
        }

        node_map = {
            "stock_finder":   ("📊 Stock Overview",      ph_overview),
            "market_data":    ("📉 Technical Analysis",  ph_technical),
            "news":           ("📰 News Sentiment",      ph_news),
            "recommendation": ("🎯 Recommendation",      ph_rec),
        }

        async for chunk in graph.astream(init_state):
            for node_name, node_output in chunk.items():
                if not node_output or not node_output.get("messages"):
                    continue

                label, placeholder = node_map.get(node_name, (node_name, st.empty()))
                last_msg = node_output["messages"][-1]
                content  = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                data     = parse_json(content)

                with placeholder.container():
                    st.markdown(f"### {label}")
                    if "raw" in data:
                        st.markdown(data["raw"])
                    elif node_name == "stock_finder":
                        render_stock_overview(data)
                    elif node_name == "market_data":
                        render_technical(data)
                    elif node_name == "news":
                        render_news(data)
                    elif node_name == "recommendation":
                        render_recommendation(data, symbol)
                    st.divider()

    asyncio.run(run_stream())