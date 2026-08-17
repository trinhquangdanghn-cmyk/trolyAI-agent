import os
import json
import time
import requests
import functools
import threading
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import pandas_datareader.data as web
import pytz
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from supabase import Client, create_client

# ==========================================
# 1. CẤU HÌNH & BIẾN MÔI TRƯỜNG (CONFIG)
# ==========================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "FREE_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

CHAT_SESSIONS_FILE = "chat_sessions.json"
ALERT_COOLDOWNS_FILE = "alert_cooldowns.json"

ALERT_COOLDOWN_HOURS = 4
MAX_CHAT_HISTORY = 20         
MAX_CONTEXT_HISTORY = 10      
MAX_PROFILE_LENGTH = 1000     

STRICT_SYSTEM_PROMPT = """
Bạn là QuangAnh Investment Agent - Cố vấn tài chính định lượng cao cấp.

[QUY TẮC CHÀO HỎI & CẤU TRÚC TRẢ LỜI]:
- Chỉ chào hỏi khi đây là tin nhắn đầu tiên của một cuộc trò chuyện mới.
- Nếu cuộc trò chuyện đã có lịch sử, KHÔNG lặp lại lời chào.
- Trả lời trực tiếp vào nội dung người dùng hỏi.

[ƯU TIÊN PHÂN TÍCH]:
1. Kết luận trước.
2. Dữ liệu sau.
3. Giải thích cuối cùng.

[QUY TẮC BẮT BUỘC 100% VỀ PI NETWORK (NẾU ĐỒNG COIN ĐANG CHỌN LÀ PI)]:
1. Pi Network (PI) ĐÃ CHÍNH THỨC RA MẮT OPEN MAINNET THỰC TẾ và giao dịch trực tiếp trên OKX (PI/USDT).
2. TUYỆT ĐỐI KHÔNG sử dụng các từ: "IOU", "Futures", "Hợp đồng tương lai", "Chưa niêm yết".
3. Mọi phân tích kỹ thuật và dòng tiền của PI dựa hoàn toàn trên dữ liệu Mainnet thực tế từ OKX.

[QUY TẮC PHÂN TÍCH GIÁ THỰC TẾ & SO SÁNH TƯƠNG QUAN]:
1. COIN TRỌNG TÂM: Mọi phân tích kỹ thuật, chỉ báo (RSI, ADX, ATR, Market Structure BOS/CHOCH, Volume Profile, Open Interest Change) và mốc giao dịch (Entry, SL, TP) PHẢI lấy coin đang chọn trong Context làm trọng tâm chính.
2. SO SÁNH LINH HOẠT: Khi cố vấn yêu cầu so sánh hoặc đối chiếu (ví dụ: so sánh với BTC, ETH hay DXY), bạn ĐƯỢC PHÉP sử dụng dữ liệu vĩ mô, vốn hóa TOTAL/TOTAL2/TOTAL3, Stablecoin Supply, Exchange Reserve, DefiLlama TVL và On-chain để phân tích tương quan dòng tiền.
3. MINH BẠCH DỮ LIỆU: TUYỆT ĐỐI KHÔNG tự bịa giá. Nếu giá hoặc chỉ báo của coin trả về là "N/A", phải thông báo rõ ràng dữ liệu real-time chưa sẵn sàng.
"""

# ==========================================
# RETRY DECORATOR (XỬ LÝ LỖI MẠNG / RETRY API)
# ==========================================
def retry_on_failure(retries=3, delay=1, backoff=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"[API Error - Exceeded Retries]: {func.__name__} failed with {e}")
                        raise e
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

# ==========================================
# 2. SCHEMA TRÍCH XUẤT BỘ NHỚ CẤU TRÚC
# ==========================================
class MemoryExtractionSchema(BaseModel):
    user_interests: List[str] = Field(description="Các đồng coin, chỉ báo hoặc chủ đề giao dịch cố vấn quan tâm")
    trading_intent: str = Field(description="Ý định/mục tiêu chính trong tin nhắn")
    user_sentiment: str = Field(description="Tâm lý/Thái độ của nhà đầu tư")
    key_facts: List[str] = Field(description="Các thông tin/sự thật cá nhân mới về nhà đầu tư")

# ==========================================
# 3. TẦNG DỮ LIỆU & PERSISTENCE (DATABASE)
# ==========================================
@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception:
            return None
    return None

def load_user_profile() -> str:
    supabase = get_supabase_client()
    if not supabase:
        return "Chưa kết nối Supabase."
    try:
        res = supabase.table("user_profile").select("*").execute()
        if not res.data:
            return "Hồ sơ trống."
        profile_data = {item['key']: item['value'] for item in res.data}
        return (
            f"- Gu rủi ro: {profile_data.get('risk_profile', 'N/A')}\n"
            f"- Danh mục ưu tiên: {profile_data.get('preferred_coins', 'N/A')}\n"
            f"- Ghi chú & Quy tắc cá nhân: {profile_data.get('trading_notes', 'N/A')}"
        )
    except Exception as e:
        return f"Lỗi đọc Hồ sơ: {e}"

def save_memory_log(coin: str, action: str, details: str) -> bool:
    supabase = get_supabase_client()
    if not supabase:
        return False
    try:
        supabase.table("memory_logs").insert({
            "coin": coin, "action": action, "details": details
        }).execute()
        return True
    except Exception:
        return False

def get_recent_memory_logs(limit: int = 5) -> str:
    supabase = get_supabase_client()
    if not supabase:
        return "Chưa có nhật ký."
    try:
        res = supabase.table("memory_logs").select("*").order("created_at", desc=True).limit(limit).execute()
        if not res.data:
            return "Chưa có lưu vết nào."
        logs = [f"- [{item['created_at'][:10]}] ({item['coin']}) {item['action']}: {item['details']}" for item in res.data]
        return "\n".join(logs)
    except Exception:
        return "Không đọc được nhật ký."

def _extract_and_save_memory_worker(user_query: str, ai_response: str, selected_coin: str):
    if not GEMINI_API_KEY:
        return

    supabase = get_supabase_client()
    client = genai.Client(api_key=GEMINI_API_KEY)

    extraction_prompt = f"""
    Hãy đóng vai là một Chuyên gia phân tích dữ liệu bộ nhớ.
    Phân tích lượt trao đổi sau giữa Cố vấn và AI Agent để trích xuất dữ liệu phân tích:
    
    [CỐ VẤN]: {user_query}
    [AI AGENT]: {ai_response}
    [COIN ĐANG CHỌN]: {selected_coin}
    """

    try:
        response = client.models.generate_content(
            model='models/gemini-3.6-flash',
            contents=extraction_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MemoryExtractionSchema,
                temperature=0.1
            ),
        )
        
        extracted_data = json.loads(response.text)
        details_summary = f"Ý định: {extracted_data.get('trading_intent')} | Tâm lý: {extracted_data.get('user_sentiment')} | Quan tâm: {', '.join(extracted_data.get('user_interests', []))}"
        
        save_memory_log(coin=selected_coin, action="ANALYSIS_EXTRACTED", details=details_summary)

        key_facts = extracted_data.get("key_facts", [])
        if key_facts and supabase:
            try:
                res = supabase.table("user_profile").select("value").eq("key", "trading_notes").execute()
                existing_notes = res.data[0]['value'] if res.data and len(res.data) > 0 else ""
                
                existing_set = set([f.strip() for f in existing_notes.split(";") if f.strip()])
                for fact in key_facts:
                    if fact.strip():
                        existing_set.add(fact.strip())
                
                updated_notes = "; ".join(existing_set)
                if len(updated_notes) > MAX_PROFILE_LENGTH:
                    updated_notes = updated_notes[-MAX_PROFILE_LENGTH:]

                supabase.table("user_profile").upsert({
                    "key": "trading_notes",
                    "value": updated_notes
                }).execute()
            except Exception as e:
                print(f"[Profile Update Error]: {e}")

    except Exception as e:
        print(f"[Memory Extraction Error]: {e}")

def extract_and_save_memory_bg(user_query: str, ai_response: str, selected_coin: str):
    thread = threading.Thread(
        target=_extract_and_save_memory_worker, 
        args=(user_query, ai_response, selected_coin),
        daemon=True
    )
    thread.start()

def load_chat_sessions():
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_sessions(sessions):
    try:
        trimmed_sessions = []
        for s in sessions:
            msgs = s.get("messages", [])
            if len(msgs) > MAX_CHAT_HISTORY:
                msgs = msgs[-MAX_CHAT_HISTORY:]
            trimmed_sessions.append({
                "title": s.get("title", "Trò chuyện"),
                "messages": msgs
            })
        with open(CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed_sessions, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Lỗi lưu file lịch sử: {e}")

# ==========================================
# 4. DỊCH VỤ DỮ LIỆU THỊ TRƯỜNG (MARKET DATA & BỔ SUNG NGUỒN MỚI)
# ==========================================
def get_current_system_time():
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now = datetime.now(tz)
    return {
        "full_datetime": now.strftime("%Y-%m-%d %H:%M:%S GMT+7"),
        "date_only": now.strftime("%d/%m/%Y"),
        "time_only": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A")
    }

@retry_on_failure(retries=3, delay=1)
def fetch_coin_price_raw(coin_symbol: str):
    clean_coin = coin_symbol.upper().replace("-USDT", "").replace("USDT", "").strip()
    try:
        res = requests.get(f"https://www.okx.com/api/v5/market/ticker?instId={clean_coin}-USDT", timeout=3).json()
        if res.get("code") == "0" and res.get("data") and len(res["data"]) > 0:
            return float(res["data"][0]["last"])
    except Exception:
        pass
    try:
        res_b = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={clean_coin}USDT", timeout=3).json()
        if "price" in res_b:
            return float(res_b["price"])
    except Exception:
        pass
    return "N/A"

@st.cache_data(ttl=10)
def get_single_coin_price(coin_symbol: str):
    return fetch_coin_price_raw(coin_symbol)

@st.cache_data(ttl=3600)
def get_fear_and_greed_index():
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            value = data["data"][0]["value"]
            classification = data["data"][0]["value_classification"]
            return int(value), classification
    except Exception:
        pass
    return None, "N/A"

@st.cache_data(ttl=3600)
def get_market_dominance():
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "data" in data and "market_cap_percentage" in data["data"]:
            btc_dominance = data["data"]["market_cap_percentage"].get("btc", 0)
            return round(btc_dominance, 2)
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_market_caps():
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "data" in data:
            global_data = data["data"]
            total_mcap = global_data.get("total_market_cap", {}).get("usd", 0)
            btc_dominance = global_data.get("market_cap_percentage", {}).get("btc", 0)
            eth_dominance = global_data.get("market_cap_percentage", {}).get("eth", 0)
            usdt_dominance = global_data.get("market_cap_percentage", {}).get("usdt", 0)

            total_t = total_mcap / 1e12
            total2_t = total_t * (1 - (btc_dominance / 100))
            total3_t = total_t * (1 - ((btc_dominance + eth_dominance) / 100))

            return {
                "TOTAL": f"${total_t:.2f}T",
                "TOTAL2": f"${total2_t:.2f}T",
                "TOTAL3": f"${total3_t:.2f}T",
                "USDT_Dominance": f"{usdt_dominance:.2f}%"
            }
    except Exception:
        pass
    return {"TOTAL": "N/A", "TOTAL2": "N/A", "TOTAL3": "N/A", "USDT_Dominance": "N/A"}

@st.cache_data(ttl=3600)
def get_stablecoin_supply():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category=stablecoins&order=market_cap_desc&per_page=5&page=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            total_stable = sum([coin.get("market_cap", 0) for coin in data if coin.get("symbol", "").lower() in ["usdt", "usdc", "dai"]])
            stable_b = total_stable / 1e9
            return f"${stable_b:.2f}B"
    except Exception:
        pass
    return "$165.40B"

# ==========================================
# 4.1 BỔ SUNG API MỚI (DEFLLAMA TVL & STABLECOIN, AGGREGATED OI, LIQUIDATION ESTIMATE)
# ==========================================
@st.cache_data(ttl=3600)
def get_defillama_metrics():
    """Lấy dữ liệu tổng TVL toàn thị trường và dòng tiền Stablecoin từ DefiLlama (Miễn phí hoàn toàn)"""
    tvl_str, stable_inflow_str = "N/A", "N/A"
    try:
        # Lấy tổng TVL
        res_tvl = requests.get("https://api.llama.fi/protocols", timeout=5).json()
        if isinstance(res_tvl, list):
            total_tvl = sum([p.get("tvl", 0) for p in res_tvl if isinstance(p.get("tvl"), (int, float))])
            tvl_str = f"${total_tvl / 1e9:,.2f}B"
    except Exception:
        pass

    try:
        # Lấy dữ liệu Stablecoins từ DefiLlama
        res_stable = requests.get("https://stablecoins.defillama.com/stablecoins?includePrices=true", timeout=5).json()
        if "peggedAssets" in res_stable:
            total_stable_mcap = sum([coin.get("circulating", {}).get("peggedUSD", 0) for coin in res_stable["peggedAssets"]])
            stable_inflow_str = f"${total_stable_mcap / 1e9:,.2f}B"
    except Exception:
        pass

    return {"tvl": tvl_str, "stablecoin_inflow": stable_inflow_str}

@st.cache_data(ttl=300)
def get_aggregated_market_oi(coin="BTC"):
    """Tổng hợp Open Interest toàn thị trường từ Binance & OKX"""
    clean_coin = coin.upper().replace("-USDT", "").replace("USDT", "").strip()
    total_oi_usd = 0.0
    sources_count = 0
    try:
        # OKX OI
        okx_res = requests.get(f"https://www.okx.com/api/v5/market/open-interest?instId={clean_coin}-USDT-SWAP", timeout=3).json()
        if okx_res.get("code") == "0" and okx_res.get("data"):
            total_oi_usd += float(okx_res["data"][0]["oiCcy"]) * float(okx_res["data"][0].get("last", 1)) # quy đổi tương đương nếu cần hoặc lấy thẳng oiccy
            sources_count += 1
    except Exception:
        pass

    try:
        # Binance OI (Public API)
        binance_res = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={clean_coin}USDT", timeout=3).json()
        if "openInterest" in binance_res:
            oi_amt = float(binance_res["openInterest"])
            price_b = fetch_coin_price_raw(clean_coin)
            if isinstance(price_b, (int, float)):
                total_oi_usd += oi_amt * price_b
                sources_count += 1
    except Exception:
        pass

    if sources_count > 0:
        return f"${total_oi_usd / 1e6:,.2f}M USDT (Multi-Exchange)"
    return "N/A"

@st.cache_data(ttl=600)
def get_exchange_reserve_and_onchain(coin="BTC"):
    clean_coin = coin.upper().replace("-USDT", "").replace("USDT", "").strip()
    try:
        url = f"https://www.okx.com/api/v5/rubik/stat/market/exchange-reserve?ccy={clean_coin}"
        res = requests.get(url, timeout=4).json()
        
        if res.get("code") == "0" and res.get("data") and isinstance(res["data"], list) and len(res["data"]) > 0:
            latest_res = res["data"][0]
            reserve_val = float(latest_res[1]) if len(latest_res) > 1 else 0.0
            
            flow_text = "Cân bằng ⚖️"
            if len(res["data"]) >= 2:
                prev_reserve = float(res["data"][1][1])
                delta = reserve_val - prev_reserve
                if delta < 0:
                    flow_text = f"Rút ròng ({delta:,.2f} {clean_coin}) 🟢 (Tích lũy)"
                elif delta > 0:
                    flow_text = f"Nạp ròng (+{delta:,.2f} {clean_coin}) 🔴 (Cảnh báo xả)"

            return {
                "exchange_reserve": f"{reserve_val:,.2f} {clean_coin}",
                "onchain_flow": flow_text,
                "whale_status": "Theo dõi biến động dòng tiền thực tế"
            }
    except Exception as e:
        print(f"[On-chain API Error for {clean_coin}]: {e}")
    
    return {
        "exchange_reserve": "N/A", 
        "onchain_flow": "Dữ liệu thời gian thực chưa sẵn sàng ⚠️", 
        "whale_status": "N/A"
    }

@st.cache_data(ttl=600)
def get_latest_crypto_news(coin_symbol: str):
    clean_coin = coin_symbol.upper().replace("-USDT", "").replace("USDT", "").strip()
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTOPANIC_API_KEY}&currencies={clean_coin}&public=true"
    try:
        res = requests.get(url, timeout=4).json()
        if isinstance(res, dict) and "results" in res and isinstance(res["results"], list):
            posts = []
            for item in res["results"][:5]:
                published_raw = item.get('published_at', '')
                try:
                    dt_obj = datetime.fromisoformat(published_raw.replace('Z', '+00:00'))
                    pub_time_formatted = dt_obj.strftime("%H:%M %d/%m/%Y")
                except Exception:
                    pub_time_formatted = published_raw[:16]
                posts.append(f"- [{pub_time_formatted}] {item.get('title')}")
            return "\n".join(posts) if posts else "Không có tin tức mới."
    except Exception:
        pass
    return "Không thể tải tin tức thời gian thực lúc này."

@st.cache_data(ttl=3600)
def get_usdvnd_rate() -> float:
    try:
        df = yf.Ticker("USDVND=X").history(period="1d")
        if not df.empty:
            return float(df['Close'].iloc[-1])
    except Exception:
        pass
    return 25400.0

@st.cache_data(ttl=43200)
def get_fed_funds_rate() -> str:
    try:
        fed_data = yf.Ticker("FEDFUNDS").history(period="1mo")
        if not fed_data.empty:
            return f"{fed_data['Close'].iloc[-1]:.2f}%"
    except Exception:
        pass
    if FRED_API_KEY:
        try:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit=1"
            res = requests.get(url, timeout=3).json()
            if "observations" in res and res["observations"]:
                return f"{float(res['observations'][0]['value']):.2f}%"
        except Exception:
            pass
    return "5.25% - 5.50%"

@st.cache_data(ttl=3600)
def get_macro_data():
    try:
        dxy_data = None
        for ticker in ["DX-Y.NYB", "DX=F"]:
            df = yf.Ticker(ticker).history(period="7d")
            if not df.empty:
                dxy_data = df
                break
        
        us10y_data = yf.Ticker("^TNX").history(period="7d")
        fed_rate = get_fed_funds_rate()

        if dxy_data is not None and not dxy_data.empty:
            dxy_latest = round(dxy_data['Close'].iloc[-1], 2)
            dxy_change = round(((dxy_data['Close'].iloc[-1] - dxy_data['Close'].iloc[-2]) / dxy_data['Close'].iloc[-2]) * 100, 2) if len(dxy_data) >= 2 else 0.0
            dxy_status = "TĂNG 📈" if dxy_change > 0 else "GIẢM 📉"
        else:
            dxy_latest, dxy_change, dxy_status = "103.50", 0.0, "N/A"

        us10y_latest = f"{round(us10y_data['Close'].iloc[-1], 2)}%" if not us10y_data.empty else "3.90%"

        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        tickers = {
            'WALCL': 'FED_Assets',
            'WTREGEN': 'TGA',
            'RRPONTSYD': 'RRP',
            'WM2NS': 'US_M2'
        }
        
        us_liquidity, rrp_val, us_m2_val = "N/A", "N/A", "N/A"
        m2_update_date = datetime.now().strftime("%d/%m/%Y")
        try:
            fred_df = web.DataReader(list(tickers.keys()), 'fred', start_date, end_date).ffill().bfill()
            if not fred_df.empty:
                latest_row = fred_df.iloc[-1]
                net_liq = (latest_row['WALCL'] / 1000) - latest_row['WTREGEN'] - latest_row['RRPONTSYD']
                us_liquidity = f"${net_liq:,.2f}B"
                rrp_val = f"${latest_row['RRPONTSYD']:,.2f}B"
                us_m2_val = f"${latest_row['WM2NS']:,.2f}B"
                m2_update_date = fred_df.index[-1].strftime("%d/%m/%Y")
        except Exception:
            pass

        return {
            "DXY": dxy_latest, "DXY_Change_%": dxy_change, "DXY_Status": dxy_status,
            "US10Y": us10y_latest, "FED_Rate": fed_rate,
            "US_Liquidity": us_liquidity, "Reverse_Repo": rrp_val, "US_M2": us_m2_val,
            "M2_Update_Date": m2_update_date
        }
    except Exception as e:
        return {
            "DXY": "103.50", "US10Y": "3.90%", "FED_Rate": "5.25% - 5.50%", 
            "DXY_Change_%": 0, "DXY_Status": "N/A", "US_Liquidity": "N/A", 
            "Reverse_Repo": "N/A", "US_M2": "N/A", "M2_Update_Date": datetime.now().strftime("%d/%m/%Y"), "Error": str(e)
        }

@retry_on_failure(retries=3, delay=1)
def fetch_okx_candlesticks_raw(coin_symbol: str, bar="1D", limit=100) -> pd.DataFrame:
    clean_coin = coin_symbol.upper().replace("-USDT", "").replace("USDT", "").strip()
    url = f"https://www.okx.com/api/v5/market/candles?instId={clean_coin}-USDT&bar={bar}&limit={limit}"
    try:
        data = requests.get(url, timeout=5).json()
        if data.get("code") == "0" and data.get("data") and isinstance(data["data"], list) and len(data["data"]) > 0:
            df = pd.DataFrame(data["data"], columns=["timestamp", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
            for col in ["open", "high", "low", "close", "vol"]:
                df[col] = df[col].astype(float)
            return df.iloc[::-1].reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60)
def get_okx_candlesticks(coin_symbol: str, bar="1D", limit=100) -> pd.DataFrame:
    return fetch_okx_candlesticks_raw(coin_symbol, bar, limit)

@st.cache_data(ttl=60)
def get_okx_taker_volume(coin="BTC"):
    clean_coin = coin.upper().replace("-USDT", "").replace("USDT", "").strip()
    try:
        res = requests.get(f"https://www.okx.com/api/v5/rubik/stat/taker-volume?ccy={clean_coin}&contractType=SWAP", timeout=3).json()
        if res.get("code") == "0" and res.get("data") and isinstance(res["data"], list) and len(res["data"]) > 0:
            latest = res["data"][0]
            buy_vol, sell_vol = float(latest[1]), float(latest[2])
            ratio = round(buy_vol / sell_vol, 2) if sell_vol > 0 else 1.0
            flow_status = f"🟢 MUA CHỦ ĐỘNG MẠNH (Ratio: {ratio})" if ratio > 1.2 else (f"🔴 BÁN CHỦ ĐỘNG MẠNH (Ratio: {ratio})" if ratio < 0.8 else f"⚖️ CÂN BẰNG MUA/BÁN (Ratio: {ratio})")
            return {"taker_ratio": ratio, "buy_vol": round(buy_vol, 2), "sell_vol": round(sell_vol, 2), "flow_status": flow_status}
    except Exception:
        pass
    return {"flow_status": "N/A", "taker_ratio": 1.0}

@st.cache_data(ttl=300)
def get_okx_derivatives_data(coin="BTC"):
    clean_coin = coin.upper().replace("-USDT", "").replace("USDT", "").strip()
    inst_id = f"{clean_coin}-USDT-SWAP"
    funding_rate, open_interest, oi_change = "N/A", "N/A", "N/A"
    try:
        res_f = requests.get(f"https://www.okx.com/api/v5/public/funding-rate?instId={inst_id}", timeout=3).json()
        if res_f.get("code") == "0" and res_f.get("data") and isinstance(res_f["data"], list) and len(res_f["data"]) > 0:
            funding_rate = f"{float(res_f['data'][0]['fundingRate']) * 100:+.4f}%"

        open_interest = get_aggregated_market_oi(clean_coin)

        res_oi_hist = requests.get(f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history?ccy={clean_coin}&period=1H&limit=2", timeout=3).json()
        if res_oi_hist.get("code") == "0" and res_oi_hist.get("data") and isinstance(res_oi_hist["data"], list) and len(res_oi_hist["data"]) >= 2:
            current_oi, prev_oi = float(res_oi_hist["data"][0][1]), float(res_oi_hist["data"][1][1])
            delta_oi = current_oi - prev_oi
            pct_oi = (delta_oi / prev_oi) * 100 if prev_oi > 0 else 0.0
            oi_change = f"{'📈 +' if delta_oi > 0 else '📉 '}${delta_oi / 1e6:,.2f}M ({pct_oi:+.2f}%)"
    except Exception:
        pass
    return {"funding_rate": funding_rate, "open_interest": open_interest, "oi_change": oi_change}

def send_telegram_alert(message: str):
    if TELEGRAM_BOT_TOKEN in ["YOUR_BOT_TOKEN", ""]:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except Exception:
        pass

def format_currency(value_usd, currency="USD", usdvnd_rate=25400.0) -> str:
    if not isinstance(value_usd, (int, float)):
        return "N/A"
    if currency == "VND":
        return f"{value_usd * usdvnd_rate:,.0f} ₫"
    return f"${value_usd:,.4f}" if value_usd < 1 else f"${value_usd:,.2f}"

# ==========================================
# 5. ĐỘNG CƠ ĐỊNH LƯỢNG (QUANT ENGINE)
# ==========================================
def detect_market_structure(df: pd.DataFrame, window=5):
    if df.empty or len(df) < (window * 2 + 1):
        return [], "N/A (Thiếu dữ liệu nến)", []

    df = df.copy()
    df['swing_high'] = False
    df['swing_low'] = False

    for i in range(window, len(df) - window):
        if df['high'].iloc[i] == df['high'].iloc[i - window : i + window + 1].max():
            df.at[df.index[i], 'swing_high'] = True
        if df['low'].iloc[i] == df['low'].iloc[i - window : i + window + 1].min():
            df.at[df.index[i], 'swing_low'] = True

    pivots = []
    for idx, row in df.iterrows():
        if row['swing_high']:
            pivots.append(('HIGH', row['high'], idx))
        elif row['swing_low']:
            pivots.append(('LOW', row['low'], idx))

    structure_labels, last_high, last_low = [], None, None
    for p_type, val, idx in pivots:
        label = ""
        if p_type == 'HIGH':
            label = "HIGH" if last_high is None else ("HH" if val > last_high else "LH")
            last_high = val
        elif p_type == 'LOW':
            label = "LOW" if last_low is None else ("HL" if val > last_low else "LL")
            last_low = val
        structure_labels.append((idx, p_type, val, label))

    structure_events = []
    if len(pivots) >= 2:
        current_trend = None
        for i in range(1, len(structure_labels)):
            idx, p_type, val, label = structure_labels[i]
            if current_trend is None:
                if label == "HH": current_trend = 'BULLISH'
                elif label == "LL": current_trend = 'BEARISH'
                continue

            if current_trend == 'BULLISH':
                if label == "HH": structure_events.append((idx, "BOS_BULLISH", val, f"BOS Tăng (Phá đỉnh HH tại ${val:.2f})"))
                elif label == "LL":
                    current_trend = 'BEARISH'
                    structure_events.append((idx, "CHOCH_BEARISH", val, f"CHOCH Đảo chiều Giảm (Phá đáy HL tại ${val:.2f})"))
            elif current_trend == 'BEARISH':
                if label == "LL": structure_events.append((idx, "BOS_BEARISH", val, f"BOS Giảm (Phá đáy LL tại ${val:.2f})"))
                elif label == "HH":
                    current_trend = 'BULLISH'
                    structure_events.append((idx, "CHOCH_BULLISH", val, f"CHOCH Đảo chiều Tăng (Phá đỉnh LH tại ${val:.2f})"))

    recent_labels = [item[3] for item in structure_labels[-4:]]
    recent_events = [event[1] for event in structure_events[-2:]] if structure_events else []

    if "CHOCH_BULLISH" in recent_events: trend = "BULLISH 🟢 (Đảo chiều Tăng - CHOCH)"
    elif "CHOCH_BEARISH" in recent_events: trend = "BEARISH 🔴 (Đảo chiều Giảm - CHOCH)"
    elif "HH" in recent_labels and "HL" in recent_labels: trend = "BULLISH 🟢 (Uptrend - Cấu trúc Tăng tiếp diễn BOS)"
    elif "LH" in recent_labels and "LL" in recent_labels: trend = "BEARISH 🔴 (Downtrend - Cấu trúc Giảm tiếp diễn BOS)"
    else: trend = "SIDEWAY / CHOCH 🟡 (Cấu trúc Tích lũy / Biến động)"

    return structure_labels, trend, structure_events

def calculate_volume_profile(df: pd.DataFrame, num_bins=30, va_percentage=0.70):
    if df.empty or len(df) < 20:
        return {"poc": "N/A", "vah": "N/A", "val": "N/A"}

    price_min, price_max = df["low"].min(), df["high"].max()
    if price_min == price_max:
        return {"poc": price_min, "vah": price_min, "val": price_min}

    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    volume_profile = np.zeros(num_bins)

    for _, row in df.iterrows():
        mask = (bin_centers >= row["low"]) & (bin_centers <= row["high"])
        if mask.any():
            volume_profile[mask] += row["vol"] / mask.sum()
        else:
            volume_profile[np.argmin(np.abs(bin_centers - row["close"]))] += row["vol"]

    poc_index = np.argmax(volume_profile)
    poc_price = round(bin_centers[poc_index], 4)
    target_volume = volume_profile.sum() * va_percentage

    va_indices = {poc_index}
    current_volume = volume_profile[poc_index]
    left_ptr, right_ptr = poc_index - 1, poc_index + 1

    while current_volume < target_volume:
        vol_left = volume_profile[left_ptr] if left_ptr >= 0 else -1
        vol_right = volume_profile[right_ptr] if right_ptr < num_bins else -1
        if vol_left == -1 and vol_right == -1: break

        if vol_left >= vol_right:
            current_volume += vol_left
            va_indices.add(left_ptr)
            left_ptr -= 1
        else:
            current_volume += vol_right
            va_indices.add(right_ptr)
            right_ptr += 1

    va_bins = bin_centers[list(va_indices)]
    return {"poc": poc_price, "vah": round(va_bins.max(), 4), "val": round(va_bins.min(), 4)}

def calculate_quant_indicators(df: pd.DataFrame):
    if df.empty or len(df) < 30: return {}

    close, high, low, vol = df["close"], df["high"], df["low"], df["vol"]
    close_prev, high_prev, low_prev = close.shift(1), high.shift(1), low.shift(1)
    current_price = close.iloc[-1]

    ma20, ma50 = close_prev.rolling(20).mean().iloc[-1], close_prev.rolling(50).mean().iloc[-1]
    trend_simple = "Tăng" if current_price > ma20 > ma50 else ("Giảm" if current_price < ma20 < ma50 else "Đi ngang")

    rsi_series = ta.rsi(close_prev, length=14)
    rsi = rsi_series.iloc[-1] if rsi_series is not None and not rsi_series.empty else 50

    vol_usd = (vol * close).shift(1)
    z_score = ((vol * close).iloc[-1] - vol_usd.tail(20).mean()) / (vol_usd.tail(20).std() + 1e-10)

    adx_df = ta.adx(high_prev, low_prev, close_prev, length=14)
    adx_val = round(adx_df['ADX_14'].iloc[-1], 2) if adx_df is not None and not adx_df.empty else 0.0

    vp = calculate_volume_profile(df)
    support_30, resistance_30 = low_prev.tail(30).min(), high_prev.tail(30).max()
    
    tr = np.maximum(high_prev - low_prev, np.maximum(abs(high_prev - close_prev.shift(1)), abs(low_prev - close_prev.shift(1))))
    atr14 = tr.rolling(14).mean().iloc[-1]

    return {
        "rsi": round(rsi, 2),
        "rsi_status": "Quá mua (>70)" if rsi > 70 else ("Quá bán (<30)" if rsi < 30 else "Trung tính"),
        "z_score": round(z_score, 2), "adx": adx_val,
        "trend_simple": trend_simple,
        "poc": vp.get("poc", "N/A"), "vah": vp.get("vah", "N/A"), "val": vp.get("val", "N/A"),
        "support": round(support_30, 4), "resistance": round(resistance_30, 4), "atr": round(atr14, 4)
    }

def calculate_volume_metrics(df_1d: pd.DataFrame):
    if df_1d.empty or len(df_1d) < 60: return {}
    vol_usd = df_1d["vol"] * df_1d["close"]
    z_score = (vol_usd.iloc[-1] - vol_usd.shift(1).tail(20).mean()) / (vol_usd.shift(1).tail(20).std() + 1e-10)
    obv = (np.sign(df_1d["close"].diff()) * df_1d["vol"]).fillna(0).cumsum()
    return {
        "vol_usd_mil": round(vol_usd.iloc[-1] / 1e6, 2),
        "z_score": round(z_score, 2),
        "vol_status": f"🔥 THỊ TRƯỜNG MẠNH (+{z_score:.2f}σ)" if z_score >= 2.0 else (f"❄️ CẠN KIỆT THANH KHOẢN ({z_score:.2f}σ)" if z_score <= -1.5 else f"⚖️ BÌNH THƯỜNG ({z_score:.2f}σ)"),
        "obv_trend": "Dòng tiền vào (OBV Tăng)" if obv.iloc[-1] > obv.tail(30).mean() else "Dòng tiền rút (OBV Giảm)"
    }

def analyze_signals(coin, price, tf_data, vol_metrics, taker_info):
    signals, status = [], "NEUTRAL"
    d1, h4, h1 = tf_data.get('1D', {}), tf_data.get('4H', {}), tf_data.get('1H', {})
    
    rsi_1d, rsi_4h, rsi_1h = d1.get('rsi', 50), h4.get('rsi', 50), h1.get('rsi', 50)
    val_1h, vah_1h, atr_1h = h1.get('val', 'N/A'), h1.get('vah', 'N/A'), h1.get('atr', 0)
    poc_1h, support_1h = h1.get('poc', 'N/A'), h1.get('support', 'N/A')
    taker_ratio = taker_info.get('taker_ratio', 1.0)

    if rsi_1d > 70 and rsi_4h > 70:
        signals.append("⚠️ CẢNH BÁO QUÁ MUA ĐA KHUNG: RSI 1D & 4H > 70.")
        status = "BEARISH_WARNING"
    elif rsi_1d < 30 and rsi_4h < 30:
        signals.append("🚀 QUÁ BÁN ĐA KHUNG: RSI 1D & 4H < 30.")
        status = "BULLISH_SIGNAL"

    if isinstance(price, (int, float)) and isinstance(poc_1h, (int, float)) and isinstance(val_1h, (int, float)):
        if price < val_1h and rsi_1h < 30 and taker_ratio > 1.1:
            signals.append(f"🎯 ĐIỂM MUA GIÁ TRỊ: Giá dưới VAL 1H (${val_1h}) + RSI 1H Quá bán.")
            status = "VALUE_BUY"
        elif isinstance(vah_1h, (int, float)) and price > vah_1h and rsi_1h > 70 and taker_ratio < 0.9:
            signals.append(f"🚨 CẢNH BÁO PHÁT TÁN: Giá trên VAH 1H (${vah_1h}) + RSI 1H Quá mua.")
            status = "DISTRIBUTION_WARNING"

    entry_target = poc_1h if isinstance(poc_1h, (int, float)) else support_1h
    stop_loss = round(price - (2 * atr_1h), 4) if isinstance(price, (int, float)) and atr_1h else "N/A"
    tp1 = round(price + (2 * atr_1h), 4) if isinstance(price, (int, float)) and atr_1h else "N/A"

    return {
        "status": status, "signals": signals,
        "entry_zone_usd": entry_target, "stop_loss_usd": stop_loss, "take_profit_1_usd": tp1
    }

# ==========================================
# 6. SCHEDULER CHẠY NGẦM & COOLDOWN (SINGLETON)
# ==========================================
def load_alert_cooldowns() -> dict:
    if os.path.exists(ALERT_COOLDOWNS_FILE):
        try:
            with open(ALERT_COOLDOWNS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_alert_cooldowns(cooldowns: dict):
    try:
        with open(ALERT_COOLDOWNS_FILE, "w", encoding="utf-8") as f:
            json.dump(cooldowns, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[Cooldown Save Error]: {e}")

def is_spam(coin: str, alert_type: str) -> bool:
    alert_key = f"{coin}_{alert_type}"
    now = datetime.now()
    cooldowns = load_alert_cooldowns()

    if alert_key in cooldowns:
        try:
            last_sent = datetime.fromisoformat(cooldowns[alert_key])
            if now - last_sent < pd.Timedelta(hours=ALERT_COOLDOWN_HOURS):
                return True
        except Exception:
            pass

    cooldowns[alert_key] = now.isoformat()
    save_alert_cooldowns(cooldowns)
    return False

def background_market_scanner():
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    now_str = datetime.now(tz).strftime("%H:%M %d/%m/%Y")
    watch_list = ["BTC", "ETH", "PI"]
    
    for coin in watch_list:
        price = fetch_coin_price_raw(coin)
        if price == "N/A": continue
            
        tf_data = {}
        for tf in ["1D", "4H", "1H"]:
            df = fetch_okx_candlesticks_raw(coin, bar=tf, limit=100)
            tf_data[tf] = calculate_quant_indicators(df)
            
        df_1d = fetch_okx_candlesticks_raw(coin, bar="1D", limit=100)
        vol_metrics = calculate_volume_metrics(df_1d)
        taker_info = get_okx_taker_volume(coin)
        
        analysis = analyze_signals(coin, price, tf_data, vol_metrics, taker_info)
        
        if (
            analysis["signals"] 
            and analysis["status"] != "NEUTRAL" 
            and not is_spam(coin, analysis["status"])
        ):
            alert_msg = f"🔔 **[CẢNH BÁO ĐA KHUNG THỜI GIAN - {now_str}]**\n"
            alert_msg += f"Coin: **{coin}** | Giá: `${price}`\n"
            for sig in analysis["signals"]:
                alert_msg += f"- {sig}\n"
            alert_msg += f"👉 Entry Gợi ý: ${analysis['entry_zone_usd']}\n"
            alert_msg += f"🛑 Stoploss (1H ATR): ${analysis['stop_loss_usd']}"
            send_telegram_alert(alert_msg)

@st.cache_resource
def get_global_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")
    scheduler.start()
    return scheduler

def init_background_scheduler():
    scheduler = get_global_scheduler()
    if not scheduler.get_job('market_scan_job'):
        scheduler.add_job(
            background_market_scanner, 
            'interval', 
            minutes=15, 
            id='market_scan_job', 
            replace_existing=True
        )

if "scheduler_started" not in st.session_state:
    init_background_scheduler()
    st.session_state.scheduler_started = True

# ==========================================
# 7. GIAO DIỆN VÀ ĐIỀU KHIỂN (STREAMLIT UI)
# ==========================================
st.set_page_config(page_title="QuangAnh Crypto Investment Agent", page_icon="📈", layout="wide")
st.title("📈 AI Agent - Phân Tích & Theo Dõi Thị Trường Crypto")
st.caption("Hệ thống phân tích định lượng Đa Khung Thời Gian (1D, 4H, 1H), Market Structure (BOS/CHOCH), Volume Profile & AI Cố vấn")

macro_info = get_macro_data()
usdvnd_rate = get_usdvnd_rate()

# Dữ liệu Thị trường tổng quan & DefiLlama
fng_val, fng_class = get_fear_and_greed_index()
btc_dom = get_market_dominance()
mcap_data = get_market_caps()
stablecoin_sup = get_stablecoin_supply()
defillama_data = get_defillama_metrics()

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Tổng Quan Thị Trường & Vĩ Mô")

# CSS tinh chỉnh kích thước thu nhỏ các chỉ số metric trên sidebar
st.sidebar.markdown("""
<style>
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Bố cục 3 cột nhỏ gọn tối ưu không gian sidebar
col_f1, col_f2, col_f3 = st.sidebar.columns(3)
col_f1.metric("F&G", f"{fng_val}" if fng_val else "N/A")
col_f2.metric("BTC.D", f"{btc_dom}%" if btc_dom else "N/A")
col_f3.metric("DXY", f"{macro_info.get('DXY')}")

col_t1, col_t2, col_t3 = st.sidebar.columns(3)
col_t1.metric("TOTAL", mcap_data.get("TOTAL", "N/A"))
col_t2.metric("TOTAL2", mcap_data.get("TOTAL2", "N/A"))
col_t3.metric("TOTAL3", mcap_data.get("TOTAL3", "N/A"))

col_m1, col_m2, col_m3 = st.sidebar.columns(3)
col_m1.metric("US10Y", f"{macro_info.get('US10Y')}")
col_m2.metric("FED", macro_info.get("FED_Rate", "5.5%").split(" ")[0])
col_m3.metric("Stable", stablecoin_sup)

with st.sidebar.expander("💧 Xem chi tiết Thanh khoản, TVL & Vĩ mô"):
    st.caption(f"• **DeFi TVL (DefiLlama):** {defillama_data.get('tvl', 'N/A')}")
    st.caption(f"• **Stablecoin Inflow (DefiLlama):** {defillama_data.get('stablecoin_inflow', 'N/A')}")
    st.caption(f"• **US Net Liquidity:** {macro_info.get('US_Liquidity', 'N/A')}")
    st.caption(f"• **Reverse Repo (RRP):** {macro_info.get('Reverse_Repo', 'N/A')}")
    st.caption(f"• **US M2 ({macro_info.get('M2_Update_Date', 'Cập nhật')}):** {macro_info.get('US_M2', 'N/A')}")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions() or [{"title": "Cuộc trò chuyện mới", "messages": []}]

if "active_index" not in st.session_state:
    st.session_state.active_index = 0

st.sidebar.markdown("---")
st.sidebar.subheader("💱 Cấu Hình Tiền Tệ")
selected_currency = st.sidebar.radio("Đơn vị hiển thị:", options=["USD ($)", "VND (₫)"], index=0, horizontal=True)
currency_code = "VND" if "VND" in selected_currency else "USD"

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Quản lý Lịch sử Chat")
if st.sidebar.button("➕ Tạo cuộc trò chuyện mới"):
    st.session_state.chat_sessions.append({"title": f"Trò chuyện {len(st.session_state.chat_sessions)+1}", "messages": []})
    st.session_state.active_index = len(st.session_state.chat_sessions) - 1
    save_chat_sessions(st.session_state.chat_sessions)
    st.rerun()

session_titles = [f"{i+1}. {s.get('title', 'Trò chuyện')}" for i, s in enumerate(st.session_state.chat_sessions)]
st.session_state.active_index = max(0, min(st.session_state.active_index, len(session_titles) - 1))
selected_session = st.sidebar.selectbox("Chọn cuộc trò chuyện:", options=range(len(session_titles)), format_func=lambda x: session_titles[x], index=st.session_state.active_index)
st.session_state.active_index = selected_session

col_del1, col_del2 = st.sidebar.columns(2)
with col_del1:
    if st.sidebar.button("🗑️ Xóa hội thoại"):
        if len(st.session_state.chat_sessions) > 1:
            st.session_state.chat_sessions.pop(st.session_state.active_index)
            st.session_state.active_index = max(0, st.session_state.active_index - 1)
        else:
            st.session_state.chat_sessions = [{"title": "Cuộc trò chuyện mới", "messages": []}]
            st.session_state.active_index = 0
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

with col_del2:
    if st.sidebar.button("⚠️ Xóa tất cả"):
        st.session_state.chat_sessions = [{"title": "Cuộc trò chuyện mới", "messages": []}]
        st.session_state.active_index = 0
        save_chat_sessions(st.session_state.chat_sessions)
        st.rerun()

st.sidebar.markdown("---")
selected_coin = st.sidebar.selectbox("Chọn hoặc gõ tên đồng Coin:", options=["BTC", "ETH", "SOL", "PI", "BNB", "XRP", "DOGE", "ADA", "NEAR", "SUI", "LINK", "AVAX"], index=0)

coin_price = get_single_coin_price(selected_coin)
price_display = format_currency(coin_price, currency_code, usdvnd_rate)

df_1d = get_okx_candlesticks(selected_coin, bar="1D", limit=100)
tf_data = {tf: calculate_quant_indicators(df_1d if tf == "1D" else get_okx_candlesticks(selected_coin, bar=tf, limit=100)) for tf in ["1D", "4H", "1H"]}

vol_metrics = calculate_volume_metrics(df_1d)
taker_info = get_okx_taker_volume(selected_coin)
deriv_info = get_okx_derivatives_data(selected_coin)
onchain_info = get_exchange_reserve_and_onchain(selected_coin)

structure_labels, ms_trend, structure_events = detect_market_structure(df_1d)
latest_event_str = structure_events[-1][3] if structure_events else "Chưa có tín hiệu BOS/CHOCH mới"
rule_analysis = analyze_signals(selected_coin, coin_price, tf_data, vol_metrics, taker_info)

st.sidebar.markdown(f"### 📌 **{selected_coin.upper()}**: `{price_display}`")

if coin_price != "N/A":
    tab1, tab2, tab3 = st.sidebar.tabs(["Khung 1D", "Khung 4H", "Khung 1H"])
    for tab, tf in zip([tab1, tab2, tab3], ["1D", "4H", "1H"]):
        with tab:
            st.caption(f"• **Xu hướng:** {tf_data[tf].get('trend_simple', 'N/A')}")
            st.caption(f"• **RSI (14):** {tf_data[tf].get('rsi', 'N/A')} ({tf_data[tf].get('rsi_status', '')})")
            st.caption(f"• **Z-Score Vol:** `{tf_data[tf].get('z_score', 'N/A')}σ`")
            st.caption(f"• **POC:** `{format_currency(tf_data[tf].get('poc'), currency_code, usdvnd_rate)}`")
            st.caption(f"• **VAL - VAH:** `{format_currency(tf_data[tf].get('val'), currency_code, usdvnd_rate)}` - `{format_currency(tf_data[tf].get('vah'), currency_code, usdvnd_rate)}`")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"• **Market Structure:** {ms_trend}")
    st.sidebar.caption(f"• **Cấu trúc mới:** {latest_event_str}")
    st.sidebar.caption(f"• **Funding Rate:** `{deriv_info.get('funding_rate')}`")
    st.sidebar.caption(f"• **Aggregated OI:** `{deriv_info.get('open_interest')}`")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"📦 **Exchange Reserve ({selected_coin.upper()}):**\n`{onchain_info.get('exchange_reserve')}`")
    st.sidebar.markdown(f"🌊 **On-chain Flow:**\n{onchain_info.get('onchain_flow')}")
    st.sidebar.markdown(f"🐋 **Whale Behavior:**\n`{onchain_info.get('whale_status')}`")

    if rule_analysis["signals"]:
        for sig in rule_analysis["signals"]:
            st.sidebar.warning(sig)

# --- GIAO DIỆN CHAT ---
current_session = st.session_state.chat_sessions[st.session_state.active_index]
messages = current_session["messages"]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi chiến lược đa khung thời gian, BOS/CHOCH, Stablecoin, DefiLlama TVL, On-chain..."):
    if not messages:
        current_session["title"] = prompt[:20] + ("..." if len(prompt) > 20 else "")

    messages.append({"role": "user", "content": prompt})
    save_chat_sessions(st.session_state.chat_sessions)

    with st.chat_message("user"):
        st.markdown(prompt)

    time_info = get_current_system_time()
    news_info = get_latest_crypto_news(selected_coin)
    user_profile_context = load_user_profile()
    recent_memories = get_recent_memory_logs()

    system_instruction_text = f"""
{STRICT_SYSTEM_PROMPT}

[HỒ SƠ TÀI KHẢO & NGUYÊN TẮC CÁ NHÂN (HỒ SƠ DÀI HẠN)]
{user_profile_context}

[NHẬT KÝ TƯƠNG TÁC GẦN ĐÂY (LỊCH SỬ NGẮN HẠN)]
{recent_memories}

[THỜI GIAN THỰC TẾ]: {time_info['full_datetime']}
[DỮ LIỆU VĨ MÔ & DÒNG TIỀN TOÀN CẦU]:
- Fear & Greed Index: {fng_val}/100 ({fng_class}) | BTC Dominance: {btc_dom}%
- Vốn hóa toàn thị trường: TOTAL: {mcap_data.get('TOTAL')} | TOTAL2: {mcap_data.get('TOTAL2')} | TOTAL3: {mcap_data.get('TOTAL3')}
- Stablecoin Supply & Inflow (DefiLlama): {defillama_data.get('stablecoin_inflow')} (Market Cap: {stablecoin_sup})
- DeFi Total Value Locked (TVL - DefiLlama): {defillama_data.get('tvl')}
- Lãi suất FED: {macro_info.get('FED_Rate')} | DXY: {macro_info.get('DXY')} | US10Y: {macro_info.get('US10Y')}
- US Net Liquidity: {macro_info.get('US_Liquidity')}
- Reverse Repo (RRP): {macro_info.get('Reverse_Repo')}
- US M2 Money Supply ({macro_info.get('M2_Update_Date')}): {macro_info.get('US_M2')}

[THÔNG TIN COIN TRỌNG TÂM: {selected_coin.upper()}]
- Giá Spot: {price_display} (${coin_price if isinstance(coin_price, (int, float)) else 'N/A'})
- Market Structure (1D): {ms_trend} | Tín hiệu: {latest_event_str}
- Taker Flow: {taker_info.get('flow_status')} | Funding Rate: {deriv_info.get('funding_rate')} | Aggregated OI: {deriv_info.get('open_interest')}
- Exchange Reserve: {onchain_info.get('exchange_reserve')} | On-chain Flow: {onchain_info.get('onchain_flow')} | Whale: {onchain_info.get('whale_status')}
- Gợi ý Entry: {format_currency(rule_analysis['entry_zone_usd'], currency_code, usdvnd_rate)} | SL: {format_currency(rule_analysis['stop_loss_usd'], currency_code, usdvnd_rate)}
[TIN TỨC]
{news_info}
"""

    if not GEMINI_API_KEY:
        with st.chat_message("assistant"):
            st.error("❌ Thiếu `GEMINI_API_KEY`. Vui lòng cấu hình file `.env`.")
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                history_subset = messages[:-1][-MAX_CONTEXT_HISTORY:]
                
                formatted_contents = [{"role": "user", "parts": [{"text": system_instruction_text}]}]
                for msg in history_subset:
                    role = "user" if msg["role"] == "user" else "model"
                    formatted_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
                formatted_contents.append({"role": "user", "parts": [{"text": prompt}]})

                response = client.models.generate_content_stream(
                    model="models/gemini-3.6-flash",
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(temperature=0.3)
                )

                for chunk in response:
                    if chunk.text:
                        cleaned_chunk = chunk.text.replace("\\n", "\n")
                        full_response += cleaned_chunk
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                messages.append({"role": "assistant", "content": full_response})
                save_chat_sessions(st.session_state.chat_sessions)

                extract_and_save_memory_bg(prompt, full_response, selected_coin)

            except Exception as e:
                message_placeholder.markdown(f"❌ Có lỗi xảy ra khi kết nối AI: {e}")
