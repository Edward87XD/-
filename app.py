from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import warnings
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser
from urllib.parse import quote_plus
import os
import json

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ------------------------------
# App setup
# ------------------------------
st.set_page_config(
    page_title="台股 AI 分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = Path(__file__).with_name("stock_analysis_history.csv")
MARKETS = ["上市（TWSE）", "上櫃（TPEx）"]
PERIOD_OPTIONS = {
    "近 3 個月": "3mo",
    "近 6 個月": "6mo",
    "近 1 年": "1y",
    "近 2 年": "2y",
    "近 5 年": "5y",
}

st.title("📈 台股 AI 分析中心")
st.caption("Taiwan Stock Analytics · 技術分析、策略回測、風險管理與情境預測")

with st.sidebar:
    st.header("⚙️ 分析設定")
    st.toggle(
        "🌱 新手模式",
        key="beginner_mode_global",
        help="開啟後，全站會把技術指標、新聞、預測與風險用白話解釋。",
    )
    investment_style = st.radio(
        "🎯 主要投資方式",
        options=["當沖", "短線", "中長線"],
        index=1,
        key="investment_style_global",
        format_func=lambda value: {
            "當沖": "⚡ 當沖｜當天買進賣出",
            "短線": "📈 短線｜持有幾天到幾週",
            "中長線": "🌱 中長線｜持有幾個月以上",
        }[value],
        help="切換後會調整預測觀察期間與相關提示。當沖分析需要盤中行情；目前資料來源主要是日線資料。",
    )
    st.caption("行情來源可能延遲；預測是統計情境，不是保證。")
    with st.expander("🤖 新聞分析方式設定（方案 A＋C＋D）", expanded=True):
        news_analysis_provider = st.selectbox(
            "選擇新聞分析方式",
            options=["免費關鍵字分析（方案 A）", "本機 Ollama AI（方案 C）", "OpenAI API（選用）"],
            index=0,
            key="news_analysis_provider",
            help="建議先用免費關鍵字分析；安裝並啟動 Ollama 後可改用本機 AI。官方公告核對（方案 D）會一併保留。",
        )
        st.session_state["news_analysis_provider_current"] = news_analysis_provider
        if news_analysis_provider == "本機 Ollama AI（方案 C）":
            st.text_input(
                "Ollama 模型名稱",
                value="qwen2.5:7b",
                key="ollama_model_input",
                help="模型必須先在電腦安裝下載。常見例子 qwen2.5:7b；實際名稱以 `ollama list` 顯示為準。",
            )
            st.caption("Ollama 在本機執行，不需要 OpenAI API Key。請先安裝 Ollama，並在終端機執行：ollama pull qwen2.5:7b。若未啟動或模型不存在，系統會明確提示並退回免費關鍵字分析。")
        elif news_analysis_provider == "OpenAI API（選用）":
            _saved_api_key = os.getenv("OPENAI_API_KEY", "")
            try:
                _saved_api_key = _saved_api_key or str(st.secrets.get("OPENAI_API_KEY", ""))
            except Exception:
                pass
            st.text_input(
                "OpenAI API Key（選填）",
                type="password",
                key="openai_api_key_input",
                value=_saved_api_key,
                help="OpenAI API 使用可能產生費用。不要把金鑰貼在公開程式碼或分享截圖。",
            )
        else:
            st.caption("免費模式不需要 API Key 或額外 AI 安裝。它會用明確的標題關鍵字做初步分類，並顯示官方公告供交叉核對。")
        st.caption("方案 D：官方重大訊息會與媒體新聞分開顯示；官方公告是已發布資訊，不代表股價一定上漲或下跌。")
    if st.button("🔄 清除行情快取", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Streamlit config.toml is used for the overall dark theme; these styles refine the page.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {background: rgba(100,116,139,.10); border: 1px solid rgba(148,163,184,.18); padding: 12px 14px; border-radius: 12px;}
      div[data-testid="stAlert"] {border-radius: 10px;}
      h1, h2, h3 {letter-spacing: -.02em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helpers
# ------------------------------
def to_yahoo_ticker(stock_code: str, market: str) -> str:
    suffix = ".TW" if market == MARKETS[0] else ".TWO"
    return f"{stock_code.strip()}{suffix}"


@st.cache_data(ttl=300, show_spinner=False)
def download_history(ticker_code: str, period: str) -> pd.DataFrame:
    """Download daily adjusted/unadjusted market history from Yahoo Finance."""
    data = yf.Ticker(ticker_code).history(
        period=period, interval="1d", auto_adjust=False, actions=False
    )
    if data is None or data.empty:
        return pd.DataFrame()
    data = data.copy()
    data.columns = [str(c) for c in data.columns]
    data = data[~data.index.duplicated(keep="last")].sort_index()
    if "Close" in data:
        data = data.dropna(subset=["Close"])
    return data


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    close = pd.to_numeric(data["Close"], errors="coerce")
    volume = pd.to_numeric(data.get("Volume", pd.Series(index=data.index, dtype=float)), errors="coerce")

    data["MA5"] = close.rolling(5, min_periods=5).mean()
    data["MA20"] = close.rolling(20, min_periods=20).mean()
    data["MA60"] = close.rolling(60, min_periods=60).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI14"] = 100 - (100 / (1 + rs))
    data.loc[(avg_loss == 0) & (avg_gain > 0), "RSI14"] = 100
    data.loc[(avg_loss == 0) & (avg_gain == 0), "RSI14"] = 50

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
    data["Volume_MA5"] = volume.rolling(5, min_periods=5).mean()
    data["Volume_Change_Pct"] = volume.pct_change() * 100
    data["Daily_Return"] = close.pct_change().replace([np.inf, -np.inf], np.nan)
    return data


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_num(value, decimals=2, suffix=""):
    value = safe_float(value)
    return f"{value:,.{decimals}f}{suffix}" if value is not None else "資料不足"


def get_strategy(data: pd.DataFrame, ma_short: int = 20, ma_long: int = 60) -> pd.DataFrame:
    """A transparent long/flat trend strategy; signal is shifted to next bar."""
    out = data.copy()
    short_col, long_col = f"MA{ma_short}", f"MA{ma_long}"
    if short_col not in out:
        out[short_col] = out["Close"].rolling(ma_short, min_periods=ma_short).mean()
    if long_col not in out:
        out[long_col] = out["Close"].rolling(ma_long, min_periods=ma_long).mean()
    condition = ((out["Close"] > out[short_col]) & (out[short_col] > out[long_col])).fillna(False)
    out["Position"] = condition.shift(1).fillna(False).astype(int)
    out["Signal"] = out["Position"].diff().fillna(out["Position"]).map({1: "買進", -1: "賣出", 0: ""})
    out["Strategy_Return_Gross"] = out["Daily_Return"].fillna(0) * out["Position"]
    # Conservative Taiwan cash-equity transaction cost estimate: fee each side + sell tax.
    fee_rate = 0.001425 * 0.28
    sell_tax = 0.003
    turnover = out["Position"].diff().abs().fillna(out["Position"])
    sell_turnover = ((out["Position"].shift(1).fillna(0) == 1) & (out["Position"] == 0)).astype(float)
    out["Trading_Cost"] = turnover * fee_rate + sell_turnover * sell_tax
    out["Strategy_Return"] = out["Strategy_Return_Gross"] - out["Trading_Cost"]
    out["Strategy_Equity"] = (1 + out["Strategy_Return"].fillna(0)).cumprod()
    out["BuyHold_Equity"] = (1 + out["Daily_Return"].fillna(0)).cumprod()
    return out


def performance_metrics(returns: pd.Series, equity: pd.Series | None = None) -> dict:
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"total_return": np.nan, "annual_return": np.nan, "volatility": np.nan,
                "sharpe": np.nan, "max_drawdown": np.nan, "win_rate": np.nan, "days": 0}
    eq = (1 + r).cumprod() if equity is None else pd.to_numeric(equity, errors="coerce").dropna()
    total = float(eq.iloc[-1] - 1) if not eq.empty else np.nan
    years = max(len(r) / 252, 1 / 252)
    annual = float((1 + total) ** (1 / years) - 1) if total > -1 and np.isfinite(total) else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() * 252) / vol) if vol and np.isfinite(vol) and vol != 0 else np.nan
    running_max = eq.cummax() if not eq.empty else eq
    dd = eq / running_max - 1 if not eq.empty else pd.Series(dtype=float)
    return {
        "total_return": total,
        "annual_return": annual,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if not dd.empty else np.nan,
        "win_rate": float((r > 0).mean()),
        "days": int(len(r)),
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fundamentals(ticker_code: str) -> dict:
    """Fetch basic valuation/fundamental fields when Yahoo Finance provides them."""
    info = yf.Ticker(ticker_code).info or {}
    return {
        "shortName": info.get("shortName"),
        "marketCap": info.get("marketCap"),
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "dividendYield": info.get("dividendYield"),
        "trailingEps": info.get("trailingEps"),
        "forwardEps": info.get("forwardEps"),
        "currency": info.get("currency", "TWD"),
    }


def fetch_twse_monthly_revenue(stock_code: str) -> dict | None:
    """Try the TWSE listed-company monthly revenue OpenAPI."""
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
    response = requests.get(url, timeout=12)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        return None
    frame = pd.DataFrame(rows)
    code_col = next((c for c in frame.columns if "公司代號" in str(c)), None)
    if not code_col:
        return None
    found = frame[frame[code_col].astype(str).str.strip().eq(stock_code)]
    if found.empty:
        return None
    row = found.iloc[0]
    revenue_col = next((c for c in frame.columns if "當月營收" in str(c) and not any(x in str(c) for x in ["去年", "上月", "累計"])), None)
    yoy_col = next((c for c in frame.columns if "去年同月增減" in str(c) or "營收年增率" in str(c)), None)
    def parse_number(v):
        if v is None or pd.isna(v):
            return None
        s = str(v).replace(",", "").replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return {
        "name": row.get("公司名稱", "未提供"),
        "period": row.get("資料年月", "未提供"),
        "revenue": parse_number(row.get(revenue_col)) if revenue_col else None,
        "yoy": parse_number(row.get(yoy_col)) if yoy_col else None,
        "source": "臺灣證券交易所 OpenAPI",
    }


@st.cache_data(ttl=600, show_spinner=False)
def fetch_company_news(stock_code: str, company_name: str = "") -> list[dict]:
    """Fetch recent company-related headlines from Google News RSS. Headlines are not verified investment signals."""
    query_parts = [f'"{stock_code}" 台股']
    if company_name and company_name.strip() and company_name.strip() != stock_code:
        query_parts.insert(0, f'"{company_name.strip()}"')
    query = " OR ".join(query_parts)
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TaiwanStockAnalytics/1.0)"}
    response = requests.get(url, headers=headers, timeout=12)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items = []
    for item in root.findall("./channel/item")[:12]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        description = strip_html_text(item.findtext("description") or "")[:1600]
        source_node = item.find("source")
        source = (source_node.text or "新聞來源") if source_node is not None else "新聞來源"
        if title and link:
            items.append({"title": title, "link": link, "date": pub_date, "source": source, "description": description})
    return items


class _PlainTextHTML(HTMLParser):
    """Small standard-library HTML text extractor for RSS descriptions."""
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        cleaned = " ".join((data or "").split())
        if cleaned:
            self.parts.append(cleaned)


def strip_html_text(value: str) -> str:
    parser = _PlainTextHTML()
    try:
        parser.feed(value or "")
        return unescape(" ".join(parser.parts)).strip()
    except Exception:
        return unescape(str(value or ""))[:1200]


def run_ai_deep_news_analysis(news_items: list[dict], company_name: str, stock_code: str, investment_style: str, api_key: str, official_items: list[dict] | None = None) -> dict:
    """Use OpenAI chat completions to analyze available RSS snippets and official announcement titles."""
    if not api_key or not api_key.strip():
        raise ValueError("請先在左側「AI 深度新聞分析設定」填入 OpenAI API Key。")
    selected_news = []
    for i, item in enumerate(news_items[:10], start=1):
        selected_news.append({
            "id": i,
            "title": str(item.get("title", ""))[:500],
            "source": str(item.get("source", ""))[:120],
            "published_at": str(item.get("date", ""))[:100],
            "available_snippet": str(item.get("description", ""))[:1600],
            "url": str(item.get("link", ""))[:500],
        })
    official = [{"date": str(x.get("公告日期", ""))[:80], "title": str(x.get("公告主旨", ""))[:500], "source": str(x.get("來源", "官方公告"))[:100]} for x in (official_items or [])[:8]]
    payload = {
        "company": company_name or stock_code,
        "stock_code": stock_code,
        "investment_style": investment_style,
        "news": selected_news,
        "official_announcements": official,
    }
    system_prompt = """你是謹慎的台股新聞研究助理。只能根據提供的新聞標題、RSS 摘要和公告標題分析，不可聲稱已閱讀未提供的全文，不可編造事實或股價。請以繁體中文回覆有效 JSON，格式為：{"overall_summary":"...","overall_bias":"可能偏多/可能偏空/多空並存/資訊不足","key_points":["..."],"items":[{"id":1,"classification":"可能偏利多/可能偏利空/多空並存/中性或資訊不足","importance":"高/中/低","reason":"...","possible_impact":"...","time_horizon":"當日/數天至數週/數月以上/難以判定","fact_vs_inference":"已提供資料可支持的內容與推測要分開說明","verification_steps":["..."],"risks":["..."]}],"counter_evidence":["..."],"caution":"..."}。區分已確認資料與推測；媒體報導不是官方確認，傳聞要標示；不能從新聞直接推斷股價因果或保證漲跌。要考慮消息是否已被市場預期、影響是否一次性、來源與時間、利多利空是否同時存在。投資方式只用來設定觀察時間範圍，不可輸出確定買賣指令。"""
    user_prompt = "請分析以下資料，並嚴格只回傳 JSON：\n" + json.dumps(payload, ensure_ascii=False)
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60,
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("error", {}).get("message", response.text[:300])
        except Exception:
            detail = response.text[:300]
        raise RuntimeError(f"OpenAI API 回應錯誤（HTTP {response.status_code}）：{detail}")
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    result = json.loads(content)
    result["model"] = body.get("model", "gpt-4o-mini")
    result["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


def run_ollama_deep_news_analysis(news_items: list[dict], company_name: str, stock_code: str, investment_style: str, model: str, official_items: list[dict] | None = None) -> dict:
    """Analyze news through a local Ollama server; no external AI API key is required."""
    selected_news = [{
        "id": i,
        "title": str(item.get("title", ""))[:500],
        "source": str(item.get("source", ""))[:120],
        "published_at": str(item.get("date", ""))[:100],
        "available_snippet": str(item.get("description", ""))[:1200],
        "url": str(item.get("link", ""))[:500],
    } for i, item in enumerate(news_items[:10], start=1)]
    official = [{"date": str(x.get("公告日期", ""))[:80], "title": str(x.get("公告主旨", ""))[:500], "source": str(x.get("來源", "官方公告"))[:100]} for x in (official_items or [])[:8]]
    payload = {"company": company_name or stock_code, "stock_code": stock_code, "investment_style": investment_style, "news": selected_news, "official_announcements": official}
    system_prompt = """你是謹慎的台股新聞研究助理。只能根據提供的新聞標題、RSS 摘要和官方公告標題分析，不可聲稱已閱讀未提供的全文，不可編造事實或股價。請以繁體中文回覆有效 JSON，格式為：{"overall_summary":"...","overall_bias":"可能偏多/可能偏空/多空並存/資訊不足","key_points":["..."],"items":[{"id":1,"classification":"可能偏利多/可能偏利空/多空並存/中性或資訊不足","importance":"高/中/低","reason":"...","possible_impact":"...","time_horizon":"當日/數天至數週/數月以上/難以判定","fact_vs_inference":"已提供資料可支持的內容與推測要分開說明","verification_steps":["..."],"risks":["..."]}],"counter_evidence":["..."],"caution":"..."}。區分已確認資料與推測；媒體報導不是官方確認，傳聞要標示；不能從新聞直接推斷股價因果或保證漲跌。投資方式只用來設定觀察時間範圍，不可輸出確定買賣指令。官方公告與媒體報導如有差異，要指出並建議核對原文。"""
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": (model or "qwen2.5:7b").strip(), "stream": False,
                  "format": "json", "options": {"temperature": 0.2},
                  "messages": [{"role": "system", "content": system_prompt},
                               {"role": "user", "content": "請分析以下資料，並嚴格只回傳 JSON：\n" + json.dumps(payload, ensure_ascii=False)}]},
            timeout=180,
        )
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("連不到 Ollama。本機 AI 可能尚未安裝或啟動；請啟動 Ollama 後重試。") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Ollama 回應逾時。可改用較小模型，或稍後再試。") from exc
    if response.status_code >= 400:
        detail = response.text[:350]
        if response.status_code == 404:
            detail = "找不到模型或 Ollama API。請在終端機執行 `ollama list` 確認模型名稱，必要時執行 `ollama pull qwen2.5:7b`。"
        raise RuntimeError(f"Ollama 回應錯誤（HTTP {response.status_code}）：{detail}")
    body = response.json()
    content = body.get("message", {}).get("content", "")
    try:
        result = json.loads(content)
    except Exception as exc:
        raise RuntimeError("本機模型沒有回傳有效 JSON，請重試或更換模型。") from exc
    result["model"] = f"Ollama：{body.get('model', model)}"
    result["analyzed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


def analyze_news_headline(title: str) -> dict:
    """Transparent keyword-based headline triage; not an LLM and not a causal price forecast."""
    text = str(title or "").strip().lower()
    positive_terms = {
        "營收創高": ["營收創高", "營收創新高", "營收新高", "創歷史新高"],
        "成長／接單": ["成長", "年增", "月增", "接單", "訂單", "擴產", "出貨增", "需求增", "獲利增加", "轉盈", "上修"],
        "合作／產品": ["合作", "取得認證", "量產", "新產品", "新客戶", "入列", "獲選"],
        "股利／回購": ["配息", "股利", "現金股利", "庫藏股", "買回"],
    }
    negative_terms = {
        "營運下滑": ["衰退", "年減", "月減", "營收下滑", "獲利下滑", "轉虧", "虧損擴大", "下修"],
        "風險／調查": ["違約", "遭罰", "裁罰", "調查", "訴訟", "下市", "違法", "停工", "停產", "資安事件"],
        "成本／需求壓力": ["砍單", "需求疲弱", "庫存增加", "成本上升", "缺料", "延遲出貨", "裁員", "減產"],
    }
    uncertainty_terms = ["傳聞", "據傳", "市場傳出", "不排除", "可能", "傳出", "法人估", "預期", "外資喊", "目標價"]
    positive_hits = [(group, term) for group, terms in positive_terms.items() for term in terms if term in text]
    negative_hits = [(group, term) for group, terms in negative_terms.items() for term in terms if term in text]
    uncertain = any(term in text for term in uncertainty_terms)
    # When both directions appear, keep it mixed rather than forcing a bullish/bearish label.
    if positive_hits and negative_hits:
        label = "多空並存／需查全文"
        impact = "標題同時出現正負面線索，需閱讀內文與公司公告才能判斷淨影響。"
        confidence = "低"
    elif positive_hits:
        label = "可能偏利多"
        impact = "標題含有成長、接單或營運改善等線索；仍須核對實際數字、持續性及市場是否已反映。"
        confidence = "中低" if uncertain else "中"
    elif negative_hits:
        label = "可能偏利空"
        impact = "標題含有營運壓力或風險線索；需確認事件規模、公司回應及是否已反映在股價。"
        confidence = "中低" if uncertain else "中"
    else:
        label = "中性／資訊不足"
        impact = "標題沒有明確的內建利多或利空關鍵字，不能據此判斷股價方向。"
        confidence = "低"
    if uncertain:
        impact += " 標題含推測或未確認措辭，請特別核對原文。"
    matched = [term for _, term in positive_hits[:4]] + [term for _, term in negative_hits[:4]]
    return {"新聞判讀": label, "可能影響": impact, "判讀線索": "、".join(matched) if matched else "未匹配明確關鍵字", "判讀信心": confidence}


def build_keyword_news_report(news_items: list[dict], company_name: str, stock_code: str, official_items: list[dict] | None = None) -> dict:
    """Build a consistent report schema using transparent free keyword rules, not generative AI."""
    report_items = []
    labels = []
    for i, item in enumerate(news_items[:10], start=1):
        triage = analyze_news_headline(item.get("title", ""))
        label = triage["新聞判讀"]
        labels.append(label)
        report_items.append({
            "id": i, "classification": label,
            "importance": "中" if triage["判讀信心"] in ("中", "中低") else "低",
            "reason": f"標題關鍵字線索：{triage['判讀線索']}。這是規則式初步分類，並非全文理解。",
            "possible_impact": triage["可能影響"],
            "time_horizon": "依投資方式與事件性質判斷；目前無法單靠標題確定",
            "fact_vs_inference": f"已取得的資料：新聞標題與 RSS 可用摘要。推測：{label}；尚未核實全文與事件實際影響。",
            "verification_steps": ["開啟新聞原文確認全文、發布時間與引用來源", "查核公司官方重大訊息、月營收或財報"],
            "risks": ["標題關鍵字可能誤判語境", "消息可能已反映在股價，且新聞與漲跌不代表因果關係"],
        })
    positive = labels.count("可能偏利多")
    negative = labels.count("可能偏利空")
    mixed = labels.count("多空並存／需查全文")
    if not labels:
        bias = "資訊不足"
    elif positive > negative and positive > mixed:
        bias = "可能偏多"
    elif negative > positive and negative > mixed:
        bias = "可能偏空"
    elif positive or negative or mixed:
        bias = "多空並存"
    else:
        bias = "資訊不足"
    official_items = official_items or []
    key_points = [f"標題規則分類：可能偏利多 {positive} 則、可能偏利空 {negative} 則、多空並存 {mixed} 則。",
                  f"官方重大訊息：目前取得 {len(official_items)} 則；請以公告原文及日期確認事件內容。"]
    if not official_items:
        key_points.append("本次沒有取得官方重大訊息，不代表公司沒有公告；可能是來源暫時無法讀取或該資料集未涵蓋。")
    return {"overall_summary": f"{company_name or stock_code}：目前以免費標題關鍵字規則整理新聞，不能視為完整 AI 語意分析或買賣訊號。",
            "overall_bias": bias, "key_points": key_points, "items": report_items,
            "counter_evidence": ["標題可能缺少上下文；需查看新聞全文與公司公告。", "即使公告屬實，也不代表股價必然朝相同方向變動。"],
            "caution": "免費關鍵字模式不是生成式 AI。請分開看待媒體線索與官方公告，核對時間、數字及公司說法；此內容不構成投資建議。",
            "model": "免費關鍵字規則（方案 A）", "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_official_announcements(stock_code: str) -> list[dict]:
    """Fetch latest TWSE listed-company material announcements; official text is fact, not price impact."""
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
    response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        return []
    frame = pd.DataFrame(rows)
    code_col = next((c for c in frame.columns if "公司代號" in str(c) or "股票代號" in str(c)), None)
    if not code_col:
        return []
    matched = frame[frame[code_col].astype(str).str.strip().eq(str(stock_code).strip())]
    out = []
    for _, row in matched.head(8).iterrows():
        def pick(keys):
            col = next((c for c in frame.columns if any(k in str(c) for k in keys)), None)
            value = row.get(col) if col else None
            return str(value).strip() if value is not None and not pd.isna(value) else ""
        title = pick(["主旨", "事由", "摘要"])
        date = pick(["發言日期", "公告日期", "日期"])
        if title:
            out.append({"公告日期": date or "未提供", "公告主旨": title, "來源": "臺灣證券交易所每日重大訊息"})
    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_related_stock_metrics(stock_code: str) -> dict:
    """Fetch recent daily prices; returns observed price/volume changes, not causal impact."""
    last_error = None
    data = pd.DataFrame()
    used_ticker = ""
    for suffix in (".TW", ".TWO"):
        ticker = f"{str(stock_code).strip()}{suffix}"
        try:
            candidate = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=False, actions=False)
            if candidate is not None and not candidate.empty and "Close" in candidate.columns:
                data, used_ticker = candidate.dropna(subset=["Close"]).sort_index(), ticker
                break
        except Exception as exc:
            last_error = str(exc)
    if data.empty:
        return {"行情狀態": "無可用資料", "5日漲跌": np.nan, "20日漲跌": np.nan, "最新收盤": np.nan, "最新成交量": np.nan, "量比(對前5日均量)": np.nan, "資料日期": "", "來源代碼": used_ticker or (last_error or "")[:60]}
    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(data.get("Volume", pd.Series(index=data.index, dtype=float)), errors="coerce").dropna()
    latest_close = float(close.iloc[-1]) if len(close) else np.nan
    ret5 = (latest_close / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 and close.iloc[-6] else np.nan
    ret20 = (latest_close / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 and close.iloc[-21] else np.nan
    vol_ratio = np.nan
    if len(volume) >= 6 and float(volume.iloc[-6:-1].mean()) > 0:
        vol_ratio = float(volume.iloc[-1] / volume.iloc[-6:-1].mean())
    return {"行情狀態": "已取得行情", "5日漲跌": ret5, "20日漲跌": ret20, "最新收盤": latest_close, "最新成交量": float(volume.iloc[-1]) if len(volume) else np.nan, "量比(對前5日均量)": vol_ratio, "資料日期": data.index[-1].strftime("%Y-%m-%d"), "來源代碼": used_ticker}


# This is a transparent, keyword-based thematic map, not a claim that a stock will move.
RELATED_STOCK_THEMES = [
    (("台積電", "晶圓代工", "先進製程", "半導體", "晶片", "封裝", "CoWoS"), [("2330", "台積電"), ("2454", "聯發科"), ("3711", "日月光投控"), ("3034", "聯詠"), ("3443", "創意")]),
    (("AI伺服器", "AI server", "伺服器", "資料中心", "GPU", "輝達", "NVIDIA"), [("2317", "鴻海"), ("2382", "廣達"), ("3231", "緯創"), ("6669", "緯穎"), ("2376", "技嘉")]),
    (("電源", "電力", "變壓器", "重電", "電網", "能源建設"), [("1513", "中興電"), ("1519", "華城"), ("1503", "士電"), ("2308", "台達電")]),
    (("面板", "顯示器", "OLED", "LCD"), [("2409", "友達"), ("3481", "群創"), ("6116", "彩晶")]),
    (("航運", "貨櫃", "運價", "散裝", "海運"), [("2603", "長榮"), ("2609", "陽明"), ("2615", "萬海"), ("2610", "華航")]),
    (("電動車", "車用電子", "電池", "充電樁"), [("2308", "台達電"), ("2317", "鴻海"), ("1504", "東元"), ("1513", "中興電")]),
    (("金融", "銀行", "利率", "壽險", "金控"), [("2881", "富邦金"), ("2882", "國泰金"), ("2891", "中信金"), ("2886", "兆豐金")]),
    (("鋼鐵", "鋼價", "不鏽鋼"), [("2002", "中鋼"), ("2014", "中鴻"), ("2027", "大成鋼")]),
]

def infer_related_stocks(news_items: list[dict], current_code: str) -> list[dict]:
    text = " ".join(item.get("title", "") for item in news_items).lower()
    found = {}
    for keywords, stocks in RELATED_STOCK_THEMES:
        matched = [word for word in keywords if word.lower() in text]
        if matched:
            for code, name in stocks:
                if code != current_code:
                    found[code] = {"代號": code, "股票": name, "關聯線索": "、".join(matched[:3])}
    return list(found.values())[:15]


def fetch_mis_quote(stock_code: str, market: str) -> dict | None:
    """Query TWSE MIS quote endpoint; this can be delayed/unavailable."""
    exchange = "tse" if market == MARKETS[0] else "otc"
    url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
    params = {"ex_ch": f"{exchange}_{stock_code}.tw", "json": "1", "delay": "0", "_": int(datetime.now().timestamp() * 1000)}
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://mis.twse.com.tw/stock/index.jsp"}
    response = requests.get(url, params=params, headers=headers, timeout=8)
    response.raise_for_status()
    items = response.json().get("msgArray", [])
    if not items:
        return None
    item = items[0]
    price = item.get("z")
    if price in (None, "", "-"):
        price = item.get("y")
    try:
        price = float(price)
    except (TypeError, ValueError):
        price = None
    return {"name": item.get("n", stock_code), "price": price, "time": item.get("t", ""), "date": item.get("d", "")}


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            return pd.DataFrame()


def append_history(record: dict) -> None:
    frame = pd.DataFrame([record])
    exists = HISTORY_FILE.exists()
    frame.to_csv(HISTORY_FILE, mode="a", header=not exists, index=False, encoding="utf-8-sig")


def rule_score(latest: pd.Series) -> tuple[int, list[str]]:
    score = 50
    reasons = []
    close, ma20, ma60 = (safe_float(latest.get(k)) for k in ["Close", "MA20", "MA60"])
    rsi = safe_float(latest.get("RSI14"))
    macd, macd_signal = (safe_float(latest.get(k)) for k in ["MACD", "MACD_Signal"])
    vol, vol_ma = (safe_float(latest.get(k)) for k in ["Volume", "Volume_MA5"])
    if close is not None and ma20 is not None:
        if close > ma20:
            score += 10; reasons.append("收盤價高於 MA20")
        else:
            score -= 10; reasons.append("收盤價低於 MA20")
    if ma20 is not None and ma60 is not None:
        if ma20 > ma60:
            score += 10; reasons.append("MA20 高於 MA60")
        else:
            score -= 10; reasons.append("MA20 未高於 MA60")
    if rsi is not None:
        if 50 <= rsi < 70:
            score += 10; reasons.append("RSI 位於偏強區間")
        elif rsi >= 70:
            score -= 5; reasons.append("RSI 偏高，留意過熱")
        elif rsi < 30:
            score += 3; reasons.append("RSI 偏低，仍需其他訊號確認")
        else:
            score -= 3; reasons.append("RSI 尚未站上 50")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            score += 10; reasons.append("MACD 高於訊號線")
        else:
            score -= 5; reasons.append("MACD 未高於訊號線")
    if vol is not None and vol_ma and vol_ma > 0:
        if vol > vol_ma:
            score += 5; reasons.append("成交量高於 5 日均量")
        else:
            reasons.append("成交量未高於 5 日均量")
    return max(0, min(100, int(score))), reasons


def forecast_scenarios(data: pd.DataFrame, horizons=(5, 10, 20), simulations=5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap scenario ranges from recent daily returns; descriptive, not a guaranteed prediction."""
    close = pd.to_numeric(data["Close"], errors="coerce").dropna()
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(120)
    if len(close) < 30 or len(returns) < 20:
        return pd.DataFrame(), pd.DataFrame()
    current = float(close.iloc[-1])
    # Recent drift is damped to reduce unstable extrapolation from a short window.
    recent_drift = float(returns.tail(60).mean())
    drift = float(np.clip(recent_drift, -0.01, 0.01) * 0.35)
    vol = float(returns.tail(60).std(ddof=1))
    if not np.isfinite(vol):
        return pd.DataFrame(), pd.DataFrame()
    vol = max(vol, 0.0001)
    rng = np.random.default_rng(42)
    max_h = max(horizons)
    # Historical-return bootstrap keeps the scenarios tied to observed daily moves.
    sampled = rng.choice(returns.to_numpy(dtype=float), size=(simulations, max_h), replace=True)
    sampled = np.clip(sampled, -0.15, 0.15)
    sampled = sampled - float(np.mean(sampled)) + drift
    paths = current * np.cumprod(1 + sampled, axis=1)
    rows = []
    path_rows = [{"交易日": 0, "情境中位數": current, "10%情境": current, "90%情境": current}]
    for h in horizons:
        vals = paths[:, h - 1]
        rows.append({
            "預測交易日": h,
            "情境中位數": float(np.percentile(vals, 50)),
            "偏弱情境（10%）": float(np.percentile(vals, 10)),
            "偏強情境（90%）": float(np.percentile(vals, 90)),
            "高於目前價格機率": float(np.mean(vals > current) * 100),
            "低於目前價格機率": float(np.mean(vals < current) * 100),
        })
        path_rows.append({"交易日": h, "情境中位數": float(np.percentile(vals, 50)), "10%情境": float(np.percentile(vals, 10)), "90%情境": float(np.percentile(vals, 90))})
    return pd.DataFrame(rows), pd.DataFrame(path_rows)


def chart_price(data: pd.DataFrame) -> alt.Chart:
    chart = data[["Close", "MA5", "MA20", "MA60"]].copy().reset_index()
    chart = chart.rename(columns={chart.columns[0]: "日期", "Close": "收盤價"})
    lines = alt.Chart(chart).transform_fold(["收盤價", "MA5", "MA20", "MA60"], as_=["指標", "價格"]).mark_line().encode(
        x=alt.X("日期:T", title="日期"),
        y=alt.Y("價格:Q", title="價格（元）"),
        color=alt.Color("指標:N", title="線條"),
        tooltip=[alt.Tooltip("日期:T"), alt.Tooltip("指標:N"), alt.Tooltip("價格:Q", format=".2f")],
    )
    strat = get_strategy(data)
    signals = strat[["Close", "Signal"]].copy().reset_index().rename(columns={strat.index.name or "Date": "日期", "Close": "收盤價"})
    if "日期" not in signals:
        signals = signals.rename(columns={signals.columns[0]: "日期"})
    buys = alt.Chart(signals[signals["Signal"] == "買進"]).mark_point(shape="triangle-up", size=130, filled=True, color="#22c55e").encode(
        x="日期:T", y="收盤價:Q", tooltip=["日期:T", "Signal:N", "收盤價:Q"]
    )
    sells = alt.Chart(signals[signals["Signal"] == "賣出"]).mark_point(shape="triangle-down", size=130, filled=True, color="#ef4444").encode(
        x="日期:T", y="收盤價:Q", tooltip=["日期:T", "Signal:N", "收盤價:Q"]
    )
    return alt.layer(lines, buys, sells).properties(height=430).interactive()


def chart_candlestick(data: pd.DataFrame) -> alt.VConcatChart:
    """Interactive candlestick chart with moving averages and volume."""
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "MA5", "MA20", "MA60"] if c in data.columns]
    frame = data[cols].copy().reset_index()
    frame = frame.rename(columns={frame.columns[0]: "日期"})
    frame["日期"] = pd.to_datetime(frame["日期"])
    frame = frame.dropna(subset=[c for c in ["Open", "High", "Low", "Close"] if c in frame.columns])
    if frame.empty or not all(c in frame.columns for c in ["Open", "High", "Low", "Close"]):
        raise ValueError("缺少繪製 K 線所需的開盤、最高、最低或收盤價資料。")

    direction_color = alt.condition(
        alt.datum.Close >= alt.datum.Open,
        alt.value("#ef4444"),  # Taiwan market convention: red for up
        alt.value("#22c55e"),  # green for down
    )
    base = alt.Chart(frame).encode(
        x=alt.X("日期:T", title="日期", axis=alt.Axis(format="%Y-%m-%d", labelAngle=-35)),
    )
    wicks = base.mark_rule().encode(
        y=alt.Y("Low:Q", title="股價（元）", scale=alt.Scale(zero=False)),
        y2=alt.Y2("High:Q"),
        color=direction_color,
        tooltip=[alt.Tooltip("日期:T", title="日期"), alt.Tooltip("Open:Q", title="開盤", format=".2f"),
                 alt.Tooltip("High:Q", title="最高", format=".2f"), alt.Tooltip("Low:Q", title="最低", format=".2f"),
                 alt.Tooltip("Close:Q", title="收盤", format=".2f")],
    )
    bodies = base.mark_bar(size=7).encode(
        y=alt.Y("Open:Q", title="股價（元）", scale=alt.Scale(zero=False)),
        y2=alt.Y2("Close:Q"),
        color=direction_color,
    )
    candles = alt.layer(wicks, bodies)
    moving_cols = [c for c in ["MA5", "MA20", "MA60"] if c in frame.columns]
    if moving_cols:
        averages = alt.Chart(frame).transform_fold(moving_cols, as_=["均線", "價格"]).mark_line(strokeWidth=1.5).encode(
            x=alt.X("日期:T", title="日期"),
            y=alt.Y("價格:Q", title="股價（元）", scale=alt.Scale(zero=False)),
            color=alt.Color("均線:N", scale=alt.Scale(domain=["MA5", "MA20", "MA60"], range=["#facc15", "#38bdf8", "#c084fc"]), title="移動平均線"),
            tooltip=[alt.Tooltip("日期:T", title="日期"), alt.Tooltip("均線:N", title="均線"), alt.Tooltip("價格:Q", title="價格", format=".2f")],
        )
        price = alt.layer(candles, averages)
    else:
        price = candles
    price = price.properties(height=360, title="K 線與 MA5／MA20／MA60 均線").interactive()

    if "Volume" in frame.columns:
        volume = alt.Chart(frame).mark_bar().encode(
            x=alt.X("日期:T", title="日期", axis=alt.Axis(format="%Y-%m-%d", labelAngle=-35)),
            y=alt.Y("Volume:Q", title="成交量"),
            color=direction_color,
            tooltip=[alt.Tooltip("日期:T", title="日期"), alt.Tooltip("Volume:Q", title="成交量", format=",")],
        ).properties(height=120, title="成交量").interactive()
        return alt.vconcat(price, volume, spacing=12).resolve_scale(x="shared").properties(
            usermeta={"embedOptions": {"actions": False}}
        )
    return alt.vconcat(price).properties(usermeta={"embedOptions": {"actions": False}})

# ------------------------------
# Tabs and inputs
# ------------------------------
tab_overview, tab_technical, tab_backtest, tab_risk, tab_history, tab_news, tab_forecast = st.tabs([
    "📊 行情總覽", "📈 技術分析", "🧪 回測績效", "🛡️ 風險分析", "🗂️ 歷史紀錄", "📰 公司情報", "🔮 未來走向"
])

with tab_overview:
    st.subheader("🔎 股票查詢")
    left, right = st.columns([1, 1])
    with left:
        stock_code = st.text_input("股票代號（Stock Code）", value="2330", max_chars=6).strip()
        market = st.radio("股票市場（Market）", MARKETS, horizontal=True)
    with right:
        period_label = st.selectbox("查詢期間（Period）", list(PERIOD_OPTIONS.keys()), index=2)
        watchlist_input = st.text_input("自選股清單（Watchlist）", value="2330, 2317, 2454", help="輸入股票代號，以逗號分隔")
    run_analysis = st.button("🔍 查詢與分析", type="primary", use_container_width=True)
    beginner_mode = st.session_state.get("beginner_mode_global", False)
    st.info(f"目前投資方式：**{st.session_state.get('investment_style_global', '短線')}**。選擇會影響未來走向的觀察期間與各分頁風險提示。")
    if st.session_state.get("investment_style_global", "短線") == "當沖":
        st.warning("當沖模式提醒：目前主要資料是日線，尚未接入盤中分鐘行情；請勿把日線分析當作即時當沖訊號。")
    if beginner_mode:
        st.info("🌱 新手模式已開啟：各分頁會加入白話說明。請把歷史數據當作參考，不要只靠單一指標決定買賣。")
    if run_analysis:
        st.session_state["run_analysis"] = True
        st.session_state["analysis_code"] = stock_code
        st.session_state["analysis_market"] = market
        st.session_state["analysis_period"] = PERIOD_OPTIONS[period_label]
        st.session_state["analysis_period_label"] = period_label
        st.session_state["analysis_watchlist"] = watchlist_input

# State is retained across tab clicks and Streamlit reruns.
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

if st.session_state["run_analysis"]:
    stock_code = str(st.session_state.get("analysis_code", "2330")).strip()
    market = st.session_state.get("analysis_market", MARKETS[0])
    period = st.session_state.get("analysis_period", "1y")
    watchlist_input = st.session_state.get("analysis_watchlist", "2330, 2317, 2454")

    if not stock_code.isdigit() or not (2 <= len(stock_code) <= 6):
        st.error("請輸入有效的股票代號，例如 2330。")
        st.stop()

    ticker_code = to_yahoo_ticker(stock_code, market)
    try:
        with st.spinner(f"正在取得 {ticker_code} 的歷史行情…"):
            raw_data = download_history(ticker_code, period)
    except Exception as exc:
        st.error("下載行情失敗，請稍後重試或檢查網路連線。")
        st.code(str(exc))
        st.stop()

    if raw_data.empty or "Close" not in raw_data.columns:
        st.error("沒有取得行情。請確認股票代號、市場是否正確，或稍後再試。")
        st.stop()

    data = calculate_indicators(raw_data)
    latest = data.iloc[-1]
    latest_date = data.index[-1].strftime("%Y-%m-%d")
    score, score_reasons = rule_score(latest)
    metrics = performance_metrics(data["Daily_Return"])

    # News and impact tab
    with tab_news:
        # 新手模式總開關已移到「查詢與分析」旁；此處讀取同一個全站設定。
        beginner_mode = st.session_state.get("beginner_mode_global", False)
        st.subheader("📰 公司情報｜近期動態與可能影響")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話看公司情報：官方公告是公司或交易所發布的內容；新聞是媒體報導，需點開原文核對；利多／利空只是可能的影響方向，不代表股價一定漲跌。關聯股是依關鍵字推測出的可能相關公司。")
        st.caption("新聞以 Google News RSS 搜尋結果為主；關聯股依標題關鍵字推測產業關聯，不代表股價必然同向，也不構成投資建議。")
        if beginner_mode:
            st.info(
                "📘 新手閱讀指南：\n\n"
                "• **已確認的官方公告**：代表公司或交易所發布的公告內容；公告本身不等於股價一定上漲或下跌。\n\n"
                "• **近期新聞**：是媒體報導或搜尋到的標題，尚需點開原文核對，不能直接當成公司已證實的事實。\n\n"
                "• **關聯股**：依新聞關鍵字找出的可能相關公司，不代表兩家公司一定有直接供應鏈關係，也不代表股價會一起漲跌。\n\n"
                "• **5 日／20 日漲跌**：比較最近 5／20 個交易日的價格變化，只反映過去，不是未來預測。\n\n"
                "• **成交量與量比**：成交量是交易活躍程度；量比高於 1 代表高於前 5 日平均量，不能單靠它判斷買進或賣出。"
            )
        refresh_news = st.button("🔄 更新公司新聞與關聯股", key="refresh_company_news", type="primary")
        should_fetch_news = (
            refresh_news
            or st.session_state.get("company_news_code") != stock_code
            or st.session_state.get("company_news_items") is None
        )
        if should_fetch_news:
            try:
                display_name = ""
                try:
                    display_name = str((yf.Ticker(ticker_code).info or {}).get("shortName") or "")
                except Exception:
                    pass
                if not display_name:
                    try:
                        quote_for_name = fetch_mis_quote(stock_code, market)
                        display_name = quote_for_name.get("name", "") if quote_for_name else ""
                    except Exception:
                        display_name = ""
                news_items = fetch_company_news(stock_code, display_name)
                # 取得新聞後立即自動做透明的關鍵字判讀，不需要額外 API 金鑰。
                for news_item in news_items:
                    news_item.update(analyze_news_headline(news_item.get("title", "")))
                st.session_state["company_news_items"] = news_items
                st.session_state["company_news_code"] = stock_code
                st.session_state["company_news_name"] = display_name or stock_code
                st.session_state["company_news_related"] = infer_related_stocks(news_items, stock_code)
                try:
                    st.session_state["company_official_announcements"] = fetch_official_announcements(stock_code)
                except Exception as announcement_exc:
                    st.session_state["company_official_announcements"] = []
                    st.session_state["company_official_error"] = str(announcement_exc)[:180]
            except Exception as exc:
                st.warning(f"暫時無法取得新聞，請稍後再試。原因：{exc}")

        if st.session_state.get("company_news_code") == stock_code and st.session_state.get("company_news_items") is not None:
            news_items = st.session_state.get("company_news_items", [])
            if not news_items:
                st.info("目前沒有找到可顯示的近期新聞。可稍後重試，或直接查看官方重大訊息。")
            else:
                st.markdown(f"**查詢公司：** {st.session_state.get('company_news_name', stock_code)}（{stock_code}）")
                st.subheader("✅ 官方公告核對（方案 D）")
                st.caption("官方公告與媒體新聞分開列出。公告可用來確認公司已發布的資訊，但不能單獨證明股價影響；若沒有抓到公告，也不代表公司一定沒有公告。")
                official_items = st.session_state.get("company_official_announcements", [])
                if official_items:
                    for item in official_items:
                        st.markdown(f"- **{item['公告日期']}｜{item['公告主旨']}**")
                        st.caption(f"來源：{item['來源']}（官方公告內容；對股價的影響仍需分析）")
                else:
                    st.info("目前沒有取得此代號的官方重大訊息，或官方資料暫時無法讀取。可到公開資訊觀測站再次核對。")
                st.subheader("📰 近期新聞與自動判讀（標題關鍵字分析）")
                # 相容先前已存入 session_state、尚未附有判讀結果的新聞。
                for item in news_items:
                    if "新聞判讀" not in item:
                        item.update(analyze_news_headline(item.get("title", "")))
                positive_count = sum(1 for item in news_items if item.get("新聞判讀") == "可能偏利多")
                negative_count = sum(1 for item in news_items if item.get("新聞判讀") == "可能偏利空")
                mixed_count = sum(1 for item in news_items if item.get("新聞判讀") == "多空並存／需查全文")
                neutral_count = len(news_items) - positive_count - negative_count - mixed_count
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("可能偏利多", positive_count)
                c2.metric("可能偏利空", negative_count)
                c3.metric("多空並存", mixed_count)
                c4.metric("中性／不足", neutral_count)
                st.divider()
                st.subheader("🧠 AI 深度新聞分析")
                st.caption("AI 會分析目前取得的新聞標題、RSS 可用摘要及官方公告標題。這不保證能讀到新聞全文；付費牆或摘要不足時，結論會受限。")
                run_deep_analysis = st.button("🧠 分析這檔股票的新聞與風險", key="run_ai_deep_news", type="primary")
                if run_deep_analysis:
                    provider = st.session_state.get("news_analysis_provider_current", "免費關鍵字分析（方案 A）")
                    official_for_analysis = st.session_state.get("company_official_announcements", [])
                    company_for_analysis = st.session_state.get("company_news_name", stock_code)
                    style_for_analysis = st.session_state.get("investment_style_global", "短線")
                    if provider == "免費關鍵字分析（方案 A）":
                        deep_result = build_keyword_news_report(news_items, company_for_analysis, stock_code, official_for_analysis)
                        st.session_state["company_news_deep_analysis"] = deep_result
                        st.session_state["company_news_deep_code"] = stock_code
                        st.info("已使用免費關鍵字分析。它不會呼叫外部 AI；請搭配官方公告與新聞原文核對。")
                    elif provider == "本機 Ollama AI（方案 C）":
                        model_name = st.session_state.get("ollama_model_input", "qwen2.5:7b")
                        try:
                            with st.spinner("本機 Ollama 正在整理新聞、官方公告與反向風險…第一次執行可能需要較久。"):
                                deep_result = run_ollama_deep_news_analysis(news_items, company_for_analysis, stock_code, style_for_analysis, model_name, official_for_analysis)
                            st.session_state["company_news_deep_analysis"] = deep_result
                            st.session_state["company_news_deep_code"] = stock_code
                        except Exception as deep_exc:
                            fallback_result = build_keyword_news_report(news_items, company_for_analysis, stock_code, official_for_analysis)
                            fallback_result["caution"] = "Ollama 本機 AI 未能完成，因此本次已退回免費關鍵字分析。原因：" + str(deep_exc) + "。" + fallback_result["caution"]
                            st.session_state["company_news_deep_analysis"] = fallback_result
                            st.session_state["company_news_deep_code"] = stock_code
                            st.warning(f"本機 AI 暫時無法使用，已退回免費關鍵字分析。原因：{deep_exc}")
                    else:
                        api_key = st.session_state.get("openai_api_key_input", "") or os.getenv("OPENAI_API_KEY", "")
                        try:
                            with st.spinner("OpenAI 正在整理新聞線索、可能影響與反向風險…"):
                                deep_result = run_ai_deep_news_analysis(news_items, company_for_analysis, stock_code, style_for_analysis, api_key, official_for_analysis)
                            st.session_state["company_news_deep_analysis"] = deep_result
                            st.session_state["company_news_deep_code"] = stock_code
                        except Exception as deep_exc:
                            fallback_result = build_keyword_news_report(news_items, company_for_analysis, stock_code, official_for_analysis)
                            fallback_result["caution"] = "OpenAI 分析未能完成，因此本次已退回免費關鍵字分析。原因：" + str(deep_exc) + "。" + fallback_result["caution"]
                            st.session_state["company_news_deep_analysis"] = fallback_result
                            st.session_state["company_news_deep_code"] = stock_code
                            st.warning(f"OpenAI 分析未完成，已退回免費關鍵字分析。原因：{deep_exc}")
                deep_result = st.session_state.get("company_news_deep_analysis") if st.session_state.get("company_news_deep_code") == stock_code else None
                if deep_result:
                    st.caption(f"分析時間：{deep_result.get('analyzed_at', '未提供')}｜模型：{deep_result.get('model', '未提供')}｜模式：{st.session_state.get('investment_style_global', '短線')}")
                    st.markdown(f"**整體判讀：** {deep_result.get('overall_bias', '資訊不足')}")
                    st.write(deep_result.get("overall_summary", "沒有整體摘要。"))
                    if deep_result.get("key_points"):
                        st.markdown("**關鍵觀察**")
                        for point in deep_result["key_points"]:
                            st.markdown(f"- {point}")
                    ai_items = {str(x.get("id")): x for x in deep_result.get("items", []) if isinstance(x, dict)}
                    for i, news_item in enumerate(news_items[:10], start=1):
                        ai_item = ai_items.get(str(i))
                        if not ai_item:
                            continue
                        with st.expander(f"新聞 {i}｜{ai_item.get('classification', '未判定')}｜重要性：{ai_item.get('importance', '未判定')}"):
                            st.markdown(f"**[{news_item.get('title', '新聞標題')}]({news_item.get('link', '#')})**")
                            st.write(ai_item.get("reason", "未提供判讀理由。"))
                            st.markdown(f"**可能影響：** {ai_item.get('possible_impact', '資訊不足')}")
                            st.markdown(f"**觀察期間：** {ai_item.get('time_horizon', '難以判定')}")
                            st.markdown(f"**已知資料與推測：** {ai_item.get('fact_vs_inference', '需核對原文')}")
                            for list_key, label in [("verification_steps", "建議核對"), ("risks", "風險")]:
                                vals = ai_item.get(list_key, [])
                                if vals:
                                    st.markdown(f"**{label}**")
                                    for val in vals:
                                        st.markdown(f"- {val}")
                    if deep_result.get("counter_evidence"):
                        st.markdown("**反向證據／不能忽略的風險**")
                        for point in deep_result["counter_evidence"]:
                            st.markdown(f"- {point}")
                    st.warning(deep_result.get("caution", "AI 分析只提供研究線索，不構成買賣建議。請核對新聞全文與官方公告。"))
                for item in news_items:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    st.caption(f"來源：{item['source']}｜發布時間：{item['date'] or '新聞來源未提供時間'}")
                    st.markdown(f"**自動判讀：** {item.get('新聞判讀', '中性／資訊不足')}　｜　**判讀信心：** {item.get('判讀信心', '低')}")
                    st.write(item.get("可能影響", "請開啟原文核對，暫無判讀。"))
                    st.caption(f"判讀線索：{item.get('判讀線索', '未匹配明確關鍵字')}")
                    st.divider()
                st.caption("上方關鍵字判讀屬規則分類；AI 深度分析則只依目前取得的標題、RSS 摘要與公告標題推論，並不保證取得全文。兩者都可能誤判，請核對原文與官方公告，不能直接當成買賣訊號。")
                related = st.session_state.get("company_news_related", [])
                st.divider()
                st.subheader("🔗 關聯股：近期漲跌幅與成交量比較")
                if related:
                    with st.spinner("正在比對候選關聯股的近期行情…"):
                        metric_rows = []
                        for candidate in related[:10]:
                            metric = fetch_related_stock_metrics(candidate["代號"])
                            metric_rows.append({**candidate, **metric})
                    compare = pd.DataFrame(metric_rows)
                    if not compare.empty:
                        compare = compare.rename(columns={"代號": "股票代號", "股票": "股票名稱"})
                        for col in ["5日漲跌", "20日漲跌"]:
                            if col in compare:
                                compare[col] = compare[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "資料不足")
                        if "最新收盤" in compare:
                            compare["最新收盤"] = compare["最新收盤"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "資料不足")
                        if "最新成交量" in compare:
                            compare["最新成交量"] = compare["最新成交量"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "資料不足")
                        if "量比(對前5日均量)" in compare:
                            compare["量比(對前5日均量)"] = compare["量比(對前5日均量)"].map(lambda x: f"{x:.2f} 倍" if pd.notna(x) else "資料不足")
                        st.dataframe(compare, use_container_width=True, hide_index=True)
                        st.caption("5日／20日漲跌以最近可取得的日收盤資料計算；量比＝最新一日成交量 ÷ 前5個交易日平均成交量。行情可能有延遲或缺漏。")
                        if beginner_mode:
                            st.info(
                                "看表小技巧：5 日漲跌看短期、20 日漲跌看稍長一點的走勢；兩者都為正代表這段期間價格上漲，"
                                "兩者都為負代表這段期間價格下跌。量比 1.5 倍代表最新成交量約是前 5 日平均量的 1.5 倍。"
                                "這些只是歷史現象，還要一起看公司公告、估值與風險。"
                            )
                    st.subheader("🧭 利多／利空方向：分析推測，不是已確認結論")
                    if beginner_mode:
                        st.caption("白話說明：『偏強／偏弱』是根據近期價格變化做的簡單描述，不代表新聞造成漲跌，也不是買賣建議。")
                    for candidate in metric_rows:
                        r5 = candidate.get("5日漲跌", np.nan)
                        r20 = candidate.get("20日漲跌", np.nan)
                        vr = candidate.get("量比(對前5日均量)", np.nan)
                        if pd.notna(r5) and pd.notna(r20):
                            if r5 > 0 and r20 > 0:
                                direction = "短中期價格偏強"
                            elif r5 < 0 and r20 < 0:
                                direction = "短中期價格偏弱"
                            else:
                                direction = "短中期方向分歧／可能轉折"
                            if pd.notna(vr) and vr >= 1.5:
                                direction += "，且最新成交量高於前5日均量 1.5 倍"
                            st.markdown(f"- **{candidate['股票']}（{candidate['代號']}）**：{direction}。題材關聯為「{candidate['關聯線索']}」；這只描述行情現象，不能證明由該新聞造成。")
                        else:
                            st.markdown(f"- **{candidate['股票']}（{candidate['代號']}）**：近期行情資料不足，暫不判定方向。")
                    st.caption("關聯股由新聞關鍵字推定，價格強弱只代表近期市場反應；要判斷真正因果，仍須檢查公司公告、供應鏈關係、營收及同業表現。")
                else:
                    st.info("新聞標題沒有匹配到目前內建的產業關鍵字，因此未列出關聯股；這不代表沒有關聯公司。")
        else:
            st.info("先在「行情總覽」完成股票查詢，再按「更新公司新聞與關聯股」取得近期情報。")
        st.markdown("**官方資訊核對：** [公開資訊觀測站（重大訊息、公告、月營收）](https://mops.twse.com.tw/) ｜ [臺灣證券交易所](https://www.twse.com.tw/zh/)")

    # Overview tab
    with tab_overview:
        st.divider()
        st.subheader("📊 股票分析儀表板")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話看行情：最新收盤價是資料期間最後一個交易日的價格；漲跌幅是相對前一筆收盤價的變化；MA20 是近 20 個交易日平均價；技術評分是規則式摘要，不是股價預測。")
        previous_close = data["Close"].iloc[-2] if len(data) >= 2 else latest["Close"]
        change_pct = ((latest["Close"] / previous_close) - 1) * 100 if previous_close else np.nan
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盤價", fmt_num(latest["Close"], suffix=" 元"))
        c2.metric("相對前一筆漲跌", fmt_num(change_pct, suffix="%"), delta=fmt_num(change_pct, suffix="%"))
        c3.metric("MA20", fmt_num(latest.get("MA20"), suffix=" 元"))
        c4.metric("技術評分（規則式）", f"{score}/100")
        c5, c6, c7 = st.columns(3)
        c5.metric("RSI（14 日）", fmt_num(latest.get("RSI14")))
        macd_state = "資料不足"
        if pd.notna(latest.get("MACD")) and pd.notna(latest.get("MACD_Signal")):
            macd_state = "偏多" if latest["MACD"] > latest["MACD_Signal"] else "偏弱"
        c6.metric("MACD 狀態", macd_state)
        c7.metric("歷史波動度（年化）", fmt_num(metrics["volatility"] * 100, suffix="%"))
        st.caption(f"資料來源：Yahoo Finance｜最後一筆交易日期：{latest_date}。歷史行情不保證為即時報價。評分為規則式技術摘要，不是機器學習預測，也不是買賣指令。")

        with st.expander("📌 技術評分依據", expanded=False):
            for reason in score_reasons:
                st.write(f"• {reason}")

        st.subheader("📡 即時行情參考（MIS）")
        quote_col, quote_note_col = st.columns([1, 2])
        with quote_col:
            if st.button("🔄 更新即時行情", key="refresh_quote", use_container_width=True):
                try:
                    st.session_state["latest_mis_quote"] = fetch_mis_quote(stock_code, market)
                    st.session_state["latest_mis_quote_code"] = stock_code
                except Exception as exc:
                    st.session_state["latest_mis_quote_error"] = str(exc)[:200]
        with quote_note_col:
            st.caption("來源：TWSE MIS 報價端點；可能延遲、盤中無成交或服務暫時不可用，並非保證即時。")
        quote = st.session_state.get("latest_mis_quote") if st.session_state.get("latest_mis_quote_code") == stock_code else None
        if quote and quote.get("price") is not None:
            st.success(f"{quote.get('name', stock_code)}｜參考成交價 {quote['price']:.2f} 元｜報價時間 {quote.get('date', '')} {quote.get('time', '')}")
        elif st.session_state.get("latest_mis_quote_error"):
            st.warning(f"目前無法取得 MIS 報價：{st.session_state['latest_mis_quote_error']}")
        else:
            st.info("尚未查詢盤中報價；按上方按鈕更新。休市時可能只回傳前收盤價或沒有成交價。")

        st.subheader("🧾 官方月營收（上市公司）")
        if st.session_state.get("beginner_mode_global", False):
            st.caption("新手解讀：營收是公司銷售商品或服務的收入，不等於淨利；年增率是和去年同月份比較。營收增加仍需搭配成本、獲利與公告一起看。")
        if market == MARKETS[0]:
            if st.button("查詢官方月營收", key="fetch_revenue"):
                try:
                    rev = fetch_twse_monthly_revenue(stock_code)
                    if rev is None:
                        st.info("官方資料中暫時找不到這個代號的月營收欄位。")
                    else:
                        st.write(f"公司名稱：{rev['name']}｜資料年月：{rev['period']}")
                        r1, r2 = st.columns(2)
                        r1.metric("當月營收（官方原始數值）", f"{rev['revenue']:,.0f}" if rev["revenue"] is not None else "暫無資料")
                        r2.metric("年增率（YoY）", f"{rev['yoy']:+.2f}%" if rev["yoy"] is not None else "暫無資料")
                        st.caption("來源：臺灣證券交易所 OpenAPI。營收數值單位請以官方欄位說明核對。")
                except Exception as exc:
                    st.warning(f"目前無法取得官方月營收：{exc}")
        else:
            st.info("目前這個版本的官方月營收查詢先支援上市公司；上櫃公司請以櫃買中心公告為準。")

        st.subheader("🏢 估值與 EPS 基本面")
        if st.session_state.get("beginner_mode_global", False):
            st.caption("新手解讀：EPS 是每股盈餘；本益比通常是股價相對每股盈餘的倍數。估值指標會受產業、景氣與資料品質影響，不能單看數字高低決定買賣。")
        if st.button("查詢估值與 EPS", key="fetch_fundamentals"):
            try:
                fundamentals = fetch_fundamentals(ticker_code)
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("本益比（Trailing P/E）", fmt_num(fundamentals.get("trailingPE")))
                f2.metric("預估本益比（Forward P/E）", fmt_num(fundamentals.get("forwardPE")))
                f3.metric("股價淨值比（P/B）", fmt_num(fundamentals.get("priceToBook")))
                f4.metric("股息殖利率", fmt_num((fundamentals.get("dividendYield") * 100) if fundamentals.get("dividendYield") is not None else None, suffix="%"))
                f5, f6, f7 = st.columns(3)
                f5.metric("近四季 EPS", fmt_num(fundamentals.get("trailingEps")))
                f6.metric("預估 EPS", fmt_num(fundamentals.get("forwardEps")))
                f7.metric("市值", f"{fundamentals['marketCap']:,.0f}" if isinstance(fundamentals.get("marketCap"), (int, float)) else "資料不足")
                st.caption("來源：Yahoo Finance 基本面欄位，可能缺漏、延遲或與官方公告口徑不同；請交叉核對公開資訊觀測站與交易所資料。")
            except Exception as exc:
                st.warning(f"目前無法取得估值與 EPS：{exc}")

        st.subheader("👀 自選股比較")
        if st.session_state.get("beginner_mode_global", False):
            st.caption("新手解讀：自選股表格只是比較所選期間的歷史表現；期間報酬較高不代表未來一定繼續上漲，也要比較風險與公司基本面。")
        codes = list(dict.fromkeys(x.strip() for x in watchlist_input.split(",") if x.strip().isdigit()))
        if codes:
            rows = []
            with st.spinner("正在比較自選股近期表現…"):
                for code in codes[:10]:
                    try:
                        hist = download_history(to_yahoo_ticker(code, market), "3mo")
                        if hist.empty or "Close" not in hist:
                            continue
                        close = hist["Close"].dropna()
                        if close.empty:
                            continue
                        chg = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 and close.iloc[0] else np.nan
                        rows.append({"股票代號": code, "最近收盤價": round(float(close.iloc[-1]), 2), "近 3 個月區間報酬": chg, "資料日期": hist.index[-1].strftime("%Y-%m-%d")})
                    except Exception:
                        continue
            if rows:
                compare = pd.DataFrame(rows).sort_values("近 3 個月區間報酬", ascending=False, na_position="last")
                st.dataframe(compare.style.format({"最近收盤價": "{:.2f}", "近 3 個月區間報酬": "{:+.2f}%"}), use_container_width=True, hide_index=True)
            else:
                st.info("自選股目前沒有取得可比較的資料。")

        report_text = (
            f"台股 AI 分析報告\n"
            f"股票代號：{stock_code}\n市場：{market}\n資料期間：{st.session_state.get('analysis_period_label', period)}\n"
            f"資料日期：{latest_date}\n最新收盤價：{float(latest['Close']):.2f}\n"
            f"技術評分（規則式）：{score}/100\nRSI14：{fmt_num(latest.get('RSI14'))}\n"
            f"MACD：{fmt_num(latest.get('MACD'))}\n歷史區間報酬：{fmt_num(metrics['total_return'] * 100, suffix='%')}\n"
            f"歷史最大回撤：{fmt_num(metrics['max_drawdown'] * 100, suffix='%')}\n\n技術條件摘要：\n"
            + "\n".join(f"- {reason}" for reason in score_reasons)
            + "\n\n注意：此報告僅供學習與研究，不是投資建議。歷史資料與回測不代表未來績效。\n"
        )
        st.download_button("📄 下載本次分析報告 TXT", data=report_text.encode("utf-8-sig"), file_name=f"{stock_code}_analysis_report.txt", mime="text/plain")

    # Technical tab
    with tab_technical:
        st.subheader("🕯️ K 線、成交量與移動平均線")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話看 K 線：每根 K 線代表一個交易日的開盤、最高、最低、收盤價格；成交量表示交易股數；MA5／MA20／MA60 是近 5／20／60 個交易日平均價格。它們只能協助觀察，不會告訴你未來一定怎麼走。")
        st.caption("台股慣例：紅 K 代表收盤高於或等於開盤，綠 K 代表收盤低於開盤。可用滑鼠查看日期與價格，並縮放／平移圖表。MA5、MA20、MA60 分別代表 5、20、60 個交易日均線。")
        try:
            st.altair_chart(chart_candlestick(data), use_container_width=True)
        except Exception as exc:
            st.warning(f"K 線圖繪製失敗：{exc}")
        with st.expander("📉 原本的均線與策略訊號圖", expanded=False):
            try:
                st.altair_chart(chart_price(data), use_container_width=True)
            except Exception as exc:
                st.warning(f"圖表繪製失敗：{exc}")
        st.subheader("技術指標數值")
        indicator_cols = [c for c in ["Close", "MA5", "MA20", "MA60", "RSI14", "MACD", "MACD_Signal", "MACD_Hist", "Volume", "Volume_MA5", "Volume_Change_Pct"] if c in data.columns]
        st.dataframe(data[indicator_cols].tail(30).rename(columns={"Close":"收盤價", "Volume":"成交量", "Volume_MA5":"5 日均量", "Volume_Change_Pct":"成交量變化%", "RSI14":"RSI14", "MACD_Signal":"MACD 訊號線", "MACD_Hist":"MACD 柱狀值"}).sort_index(ascending=False), use_container_width=True)
        st.subheader("技術面摘要")
        t1, t2, t3 = st.columns(3)
        rsi_value = safe_float(latest.get("RSI14"))
        rsi_text = "資料不足" if rsi_value is None else ("偏高，留意過熱" if rsi_value >= 70 else "偏低，留意弱勢或反彈" if rsi_value <= 30 else "中性區間")
        t1.metric("RSI 判讀", rsi_text)
        t2.metric("均線排列", "MA20 > MA60" if pd.notna(latest.get("MA20")) and pd.notna(latest.get("MA60")) and latest["MA20"] > latest["MA60"] else "尚未呈現多頭排列")
        t3.metric("MACD 判讀", macd_state)

    # Backtest tab
    with tab_backtest:
        st.subheader("🧪 趨勢策略歷史回測")
        if st.session_state.get("investment_style_global", "短線") == "當沖":
            st.warning("目前回測使用日線資料，並非盤中分鐘線，因此不能用來驗證真正的當沖策略。當沖模式需要接入可靠的盤中行情並重新設計進出場與交易成本模型。")
        elif st.session_state.get("investment_style_global", "短線") == "中長線":
            st.info("中長線模式：目前回測仍是示範性的均線趨勢策略。評估持有數月以上的策略時，建議使用更長歷史區間並納入股利、除權息與交易成本。")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話看回測：回測是把規則套用到過去行情，看看過去可能的結果。累積報酬是期間總變化；最大回撤是資產曲線從高點跌下來的最大幅度；勝率不代表每次交易都會賺，也不保證未來績效。")
        st.caption("預設規則：收盤價高於 MA20，且 MA20 高於 MA60 時持有；訊號延後一個交易日，並估算交易成本。這是示範性規則，不構成投資建議。")
        bt = get_strategy(data)
        strat_metrics = performance_metrics(bt["Strategy_Return"], bt["Strategy_Equity"])
        bh_metrics = performance_metrics(bt["Daily_Return"], bt["BuyHold_Equity"])
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("策略累積報酬", fmt_num(strat_metrics["total_return"] * 100, suffix="%"))
        b2.metric("買進持有報酬", fmt_num(bh_metrics["total_return"] * 100, suffix="%"))
        b3.metric("策略最大回撤", fmt_num(strat_metrics["max_drawdown"] * 100, suffix="%"))
        b4.metric("策略勝率（日報酬）", fmt_num(strat_metrics["win_rate"] * 100, suffix="%"))
        equity_frame = pd.DataFrame({"日期": bt.index, "策略資金曲線": bt["Strategy_Equity"].values, "買進持有資金曲線": bt["BuyHold_Equity"].values})
        eq_chart = alt.Chart(equity_frame).transform_fold(["策略資金曲線", "買進持有資金曲線"], as_=["策略", "資金倍數"]).mark_line().encode(
            x=alt.X("日期:T", title="日期"), y=alt.Y("資金倍數:Q", title="起始資金 = 1"), color=alt.Color("策略:N"), tooltip=["日期:T", "策略:N", alt.Tooltip("資金倍數:Q", format=".3f")]
        ).properties(height=330).interactive()
        st.altair_chart(eq_chart, use_container_width=True)
        starting_capital = st.number_input("模擬起始資金（元）", min_value=1000, max_value=100000000, value=100000, step=10000)
        st.metric("策略期末模擬資金", f"{starting_capital * (1 + strat_metrics['total_return']):,.0f} 元" if np.isfinite(strat_metrics["total_return"]) else "資料不足")
        trade_rows = bt[bt["Signal"].isin(["買進", "賣出"])].copy()
        trade_rows["日期"] = trade_rows.index.strftime("%Y-%m-%d")
        st.subheader("策略訊號紀錄")
        if trade_rows.empty:
            st.info("此期間沒有策略進出場訊號。")
        else:
            st.dataframe(trade_rows[["日期", "Close", "Signal", "Position", "Strategy_Return"]].rename(columns={"Close":"收盤價", "Signal":"訊號", "Position":"持倉狀態", "Strategy_Return":"策略日報酬"}).sort_index(ascending=False), use_container_width=True, hide_index=True)
        st.subheader("分段驗證（前 70% / 後 30%）")
        split = int(len(bt) * 0.7)
        if split >= 20 and len(bt) - split >= 10:
            train = performance_metrics(bt["Strategy_Return"].iloc[:split])
            test = performance_metrics(bt["Strategy_Return"].iloc[split:])
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("**前 70%（樣本內）**")
                st.write(f"累積報酬：{train['total_return']:.2%}｜最大回撤：{train['max_drawdown']:.2%}｜年化波動：{train['volatility']:.2%}" if np.isfinite(train["total_return"]) else "資料不足")
            with v2:
                st.markdown("**後 30%（樣本外）**")
                st.write(f"累積報酬：{test['total_return']:.2%}｜最大回撤：{test['max_drawdown']:.2%}｜年化波動：{test['volatility']:.2%}" if np.isfinite(test["total_return"]) else "資料不足")
            st.caption("樣本外結果仍受市場環境、交易成本假設、資料長度影響；不是未來績效保證。")
        else:
            st.info("資料筆數較少，暫時無法可靠地切分前 70% 與後 30%。")
        st.download_button("下載回測明細 CSV", data=bt.to_csv(index=True, encoding="utf-8-sig").encode("utf-8-sig"), file_name=f"{stock_code}_backtest.csv", mime="text/csv")

    # Risk tab
    with tab_risk:
        st.subheader("🛡️ 風險分析")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話看風險：波動度越高，價格起伏通常越大；最大回撤代表歷史上從高點到低點的最大跌幅；參考停損／停利價只是依你設定的百分比計算，不會預測實際走勢，也不會自動下單。")
        daily_returns = data["Daily_Return"].dropna()
        risk_metrics = performance_metrics(daily_returns)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("年化波動度", fmt_num(risk_metrics["volatility"] * 100, suffix="%"))
        r2.metric("夏普比率（近似）", fmt_num(risk_metrics["sharpe"]))
        r3.metric("歷史最大回撤", fmt_num(risk_metrics["max_drawdown"] * 100, suffix="%"))
        r4.metric("歷史區間報酬", fmt_num(risk_metrics["total_return"] * 100, suffix="%"))
        close = float(latest["Close"])
        stop_pct = st.slider("停損幅度參考（%）", min_value=2.0, max_value=20.0, value=8.0, step=0.5)
        target_pct = st.slider("停利幅度參考（%）", min_value=3.0, max_value=40.0, value=15.0, step=0.5)
        rr = target_pct / stop_pct if stop_pct else np.nan
        s1, s2, s3 = st.columns(3)
        s1.metric("參考停損價", f"{close * (1 - stop_pct / 100):.2f} 元")
        s2.metric("參考停利價", f"{close * (1 + target_pct / 100):.2f} 元")
        s3.metric("潛在報酬／風險比", f"{rr:.2f} : 1")
        st.caption("停損／停利價格僅按你選擇的百分比計算，不考慮跳空、流動性、手續費、個人部位或承受能力，不是個人化投資建議。")
        dd = data[["Close"]].copy()
        dd["資產曲線"] = (1 + data["Daily_Return"].fillna(0)).cumprod()
        dd["歷史回撤"] = dd["資產曲線"] / dd["資產曲線"].cummax() - 1
        drawdown_chart = alt.Chart(dd.reset_index().rename(columns={dd.index.name or "Date":"日期"})).mark_area(opacity=0.45).encode(x=alt.X("日期:T"), y=alt.Y("歷史回撤:Q", axis=alt.Axis(format="%")), tooltip=["日期:T", alt.Tooltip("歷史回撤:Q", format=".2%")]).properties(height=260)
        st.subheader("歷史回撤曲線")
        st.altair_chart(drawdown_chart, use_container_width=True)
        st.subheader("簡易參數穩定性檢視")
        stability_rows = []
        for short_n, long_n in [(5, 20), (10, 30), (20, 60), (20, 120)]:
            if len(data) < long_n + 5:
                continue
            candidate = data.copy()
            candidate[f"MA{short_n}"] = candidate["Close"].rolling(short_n, min_periods=short_n).mean()
            candidate[f"MA{long_n}"] = candidate["Close"].rolling(long_n, min_periods=long_n).mean()
            candidate["Daily_Return"] = candidate["Close"].pct_change()
            candidate = get_strategy(candidate, short_n, long_n)
            pm = performance_metrics(candidate["Strategy_Return"], candidate["Strategy_Equity"])
            stability_rows.append({"短均線": short_n, "長均線": long_n, "累積報酬": pm["total_return"], "最大回撤": pm["max_drawdown"], "夏普比率（近似）": pm["sharpe"]})
        if stability_rows:
            stability = pd.DataFrame(stability_rows)
            st.dataframe(stability.style.format({"累積報酬":"{:.2%}", "最大回撤":"{:.2%}", "夏普比率（近似）":"{:.2f}"}), use_container_width=True, hide_index=True)
        else:
            st.info("資料筆數不足，無法比較不同均線組合。")

    # History tab
    with tab_history:
        st.subheader("🗂️ 歷史分析紀錄")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話看歷史紀錄：這裡保存每次分析時的部分數值，方便比較不同時間的結果。過去分數或報酬較好，不代表下一次也會一樣。")
        if st.button("將本次分析存入歷史紀錄", type="primary", key="save_history"):
            record = {
                "分析時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "股票代號": stock_code,
                "市場": market,
                "期間": st.session_state.get("analysis_period_label", period),
                "最新收盤價": safe_float(latest["Close"]),
                "RSI14": safe_float(latest.get("RSI14")),
                "MACD": safe_float(latest.get("MACD")),
                "技術評分": score,
                "歷史區間報酬%": safe_float(metrics["total_return"] * 100),
                "歷史最大回撤%": safe_float(metrics["max_drawdown"] * 100),
            }
            try:
                append_history(record)
                st.success(f"已儲存至 {HISTORY_FILE.name}")
            except Exception as exc:
                st.error(f"儲存失敗：{exc}")
        history = load_history()
        if history.empty:
            st.info("目前還沒有歷史紀錄。請先在此分頁按「將本次分析存入歷史紀錄」。")
        else:
            st.dataframe(history.sort_values("分析時間", ascending=False) if "分析時間" in history else history, use_container_width=True, hide_index=True)
            if {"分析時間", "技術評分"}.issubset(history.columns):
                hist_chart = alt.Chart(history).mark_line(point=True).encode(x=alt.X("分析時間:T", title="分析時間"), y=alt.Y("技術評分:Q", scale=alt.Scale(domain=[0, 100])), color=alt.Color("股票代號:N"), tooltip=["分析時間:T", "股票代號:N", "技術評分:Q"]).properties(height=280).interactive()
                st.subheader("歷史技術評分變化")
                st.altair_chart(hist_chart, use_container_width=True)
            if {"分析時間", "最新收盤價"}.issubset(history.columns):
                price_chart = alt.Chart(history).mark_line(point=True).encode(x=alt.X("分析時間:T", title="分析時間"), y=alt.Y("最新收盤價:Q", title="收盤價"), color=alt.Color("股票代號:N"), tooltip=["分析時間:T", "股票代號:N", alt.Tooltip("最新收盤價:Q", format=".2f")]).properties(height=280).interactive()
                st.subheader("歷史收盤價變化")
                st.altair_chart(price_chart, use_container_width=True)
            st.download_button("下載全部歷史紀錄 CSV", data=history.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), file_name="stock_analysis_history.csv", mime="text/csv")

    # Forecast tab: probabilistic scenarios based on historical daily returns
    with tab_forecast:
        st.subheader("🔮 未來走向｜統計情境預測")
        selected_style = st.session_state.get("investment_style_global", "短線")
        forecast_horizons = {"當沖": (1,), "短線": (5, 10, 20), "中長線": (20, 60, 120)}[selected_style]
        if selected_style == "當沖":
            st.warning("你目前選擇「當沖」。目前程式的行情主要是日線資料，以下只能展示未來 1 個交易日的歷史報酬情境，不能代替盤中分時圖、即時成交量或真正的當沖預測。")
        elif selected_style == "短線":
            st.caption("短線模式：使用最近最多 120 個交易日的歷史報酬率，展示未來 5／10／20 個交易日的情境區間。這不是買賣保證。")
        else:
            st.caption("中長線模式：使用最近最多 120 個交易日的歷史報酬率，展示未來 20／60／120 個交易日的情境區間。現有模型仍屬簡化模擬，長期預測不確定性更高。")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話解讀：中位數是模擬結果的中間位置；偏弱／偏強情境是模擬分布的第 10／90 百分位，不是最低價、最高價，也不是保證會落在這個範圍。『高於目前價格機率』是模型模擬的比例，不是真實世界已驗證的勝率。")
        forecast_table, forecast_path = forecast_scenarios(data, horizons=forecast_horizons)
        current_close = float(data["Close"].iloc[-1])
        if forecast_table.empty:
            st.warning("歷史資料不足，至少需要約 30 個交易日才能產生情境分析。")
        else:
            st.metric("模型基準價格（最近日收盤）", f"{current_close:,.2f} 元", help=f"歷史資料最後日期：{latest_date}；不一定是盤中即時價。")
            chart = alt.Chart(forecast_path).mark_area(opacity=0.22).encode(
                x=alt.X("交易日:Q", title="未來交易日"),
                y=alt.Y("10%情境:Q", title="情境價格（元）", scale=alt.Scale(zero=False)),
                y2="90%情境:Q",
                tooltip=[alt.Tooltip("交易日:Q"), alt.Tooltip("10%情境:Q", format=".2f"), alt.Tooltip("情境中位數:Q", format=".2f"), alt.Tooltip("90%情境:Q", format=".2f")],
            )
            median_line = alt.Chart(forecast_path).mark_line(point=True, strokeWidth=3, color="#38bdf8").encode(
                x=alt.X("交易日:Q", title="未來交易日"), y=alt.Y("情境中位數:Q", title="情境價格（元）", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("交易日:Q"), alt.Tooltip("情境中位數:Q", format=".2f")]
            )
            st.altair_chart((chart + median_line).properties(height=330).interactive(), use_container_width=True)
            display_forecast = forecast_table.copy()
            for col in ["情境中位數", "偏弱情境（10%）", "偏強情境（90%）"]:
                display_forecast[col] = display_forecast[col].map(lambda x: f"{x:,.2f} 元")
            for col in ["高於目前價格機率", "低於目前價格機率"]:
                display_forecast[col] = display_forecast[col].map(lambda x: f"{x:.1f}%")
            st.dataframe(display_forecast, use_container_width=True, hide_index=True)
            st.subheader("走向判讀（依目前歷史數據推測）")
            last = data.iloc[-1]
            ma20 = safe_float(last.get("MA20")); ma60 = safe_float(last.get("MA60")); rsi = safe_float(last.get("RSI14"))
            notes=[]
            if ma20 is not None and ma60 is not None:
                notes.append("中期均線偏多（MA20 高於 MA60）" if ma20 > ma60 else "中期均線偏弱（MA20 未高於 MA60）")
            if rsi is not None:
                notes.append("RSI 偏高，需留意短線過熱" if rsi >= 70 else "RSI 偏低，可能仍弱勢或有反彈波動" if rsi < 30 else "RSI 未顯示極端超買／超賣")
            if len(data) >= 6:
                five_day = (float(data["Close"].iloc[-1]) / float(data["Close"].iloc[-6]) - 1) * 100
                notes.append(f"近 5 個交易日價格變化 {five_day:+.2f}%")
            for note in notes:
                st.markdown(f"- {note}")
            st.warning("重要限制：預測會受突發新聞、財報、法說、政策、流動性與大盤影響。模擬機率未經前瞻驗證，請勿把它當成實際獲利機率；若要使用於交易，應先做滾動樣本外測試並納入交易成本。")
            st.download_button("下載預測情境 CSV", data=forecast_table.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), file_name=f"{stock_code}_forecast_scenarios.csv", mime="text/csv")

else:
    with tab_technical:
        st.info("請先到「行情總覽」輸入股票代號並按下「查詢與分析」，即可查看技術分析。")
        if st.session_state.get("beginner_mode_global", False):
            st.caption("新手提示：技術分析是用過去價格與成交量觀察走勢，不是預測未來的保證工具。")
    with tab_backtest:
        st.info("完成股票查詢後，這裡會顯示策略回測與資金曲線。")
        if st.session_state.get("beginner_mode_global", False):
            st.caption("新手提示：回測只呈現規則套用到過去資料的結果，不代表實際交易一定能複製。")
    with tab_risk:
        st.info("完成股票查詢後，這裡會顯示風險指標與停損停利參考。")
        if st.session_state.get("beginner_mode_global", False):
            st.caption("新手提示：風險指標幫助理解可能的損失，不會消除投資風險。")
    with tab_history:
        st.subheader("🗂️ 歷史分析紀錄")
        if st.session_state.get("beginner_mode_global", False):
            st.info("📘 白話看歷史紀錄：這裡列出先前保存的分析結果，適合比較，不代表未來績效。")
        history = load_history()
        if history.empty:
            st.info("目前還沒有歷史紀錄。先完成一次股票分析，再到此分頁儲存。")
        else:
            st.dataframe(history, use_container_width=True, hide_index=True)
            st.download_button("下載全部歷史紀錄 CSV", data=history.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), file_name="stock_analysis_history.csv", mime="text/csv")

    with tab_news:
        st.info("請先到「行情總覽」完成股票查詢，再按「更新公司新聞與關聯股」查看公司近期動態。")
        if st.session_state.get("beginner_mode_global", False):
            st.caption("新手提示：官方公告與媒體新聞要分開看；標題可能不完整，請開啟原文確認。")
    with tab_forecast:
        st.info("請先到「行情總覽」輸入股票代號並按下「查詢與分析」，再查看未來走向情境。")

st.divider()
st.caption("風險提醒：本工具僅供學習與研究。資料可能延遲、缺漏或有調整；技術指標與歷史回測不代表未來表現，請勿僅依本工具做投資決策。")
