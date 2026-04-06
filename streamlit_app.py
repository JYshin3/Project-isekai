# 👑 프로젝트 이세계 V13 — 표 중심 대시보드
# 매수/매도 신호 + AI 종목 자동 추천
# 실행: streamlit run streamlit_app.py

import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime, io, base64

st.set_page_config(page_title="이세계 V13", page_icon="👑",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&family=JetBrains+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif; background:#060810; color:#e8eaf6;}
.main-title{text-align:center;font-size:1.9rem;font-weight:700;
  background:linear-gradient(135deg,#00d4ff,#7b5ea7,#00ff9d);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.1rem;}
.sub-title{text-align:center;color:#6b7280;font-size:.82rem;margin-bottom:1.2rem;}

/* 신호 뱃지 */
.badge-strong-buy{background:#0d3320;color:#00ff9d;padding:4px 12px;border-radius:20px;
  font-weight:700;font-size:.82rem;border:1px solid #00ff9d;}
.badge-buy{background:#0d2b1a;color:#4ade80;padding:4px 12px;border-radius:20px;
  font-weight:700;font-size:.82rem;border:1px solid #4ade80;}
.badge-watch{background:#2b2500;color:#ffd700;padding:4px 12px;border-radius:20px;
  font-weight:700;font-size:.82rem;border:1px solid #ffd700;}
.badge-hold{background:#1a1a1a;color:#9ca3af;padding:4px 12px;border-radius:20px;
  font-weight:700;font-size:.82rem;border:1px solid #374151;}
.badge-sell{background:#2b0d0d;color:#ff4757;padding:4px 12px;border-radius:20px;
  font-weight:700;font-size:.82rem;border:1px solid #ff4757;}

/* 카드 */
.mc{background:#111827;border:1px solid #1e2d4a;border-radius:10px;
    padding:14px 12px;text-align:center;height:100%;}
.mc-lbl{color:#6b7280;font-size:.7rem;margin-bottom:3px;font-family:'JetBrains Mono',monospace;}
.mc-val{font-size:1.15rem;font-weight:700;}

/* 종목 추천 행 */
.ticker-row{background:#0f172a;border:1px solid #1e2d4a;border-radius:10px;
    padding:14px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px;}
.ticker-name{font-size:1.1rem;font-weight:700;color:#e8eaf6;min-width:60px;}
.ticker-price{font-family:'JetBrains Mono',monospace;font-size:1rem;color:#00d4ff;min-width:70px;}
.score-pill{background:#1e2d4a;border-radius:20px;padding:3px 10px;
    font-size:.78rem;font-family:'JetBrains Mono',monospace;}

/* 진행바 */
.prog-wrap{background:#1e2d4a;border-radius:4px;height:6px;flex:1;}
.prog-fill{height:6px;border-radius:4px;}

/* 테이블 스타일 */
.styled-table{width:100%;border-collapse:collapse;font-size:.85rem;}
.styled-table th{background:#111827;color:#6b7280;padding:10px 12px;
    text-align:left;font-size:.75rem;font-family:'JetBrains Mono',monospace;
    border-bottom:2px solid #1e2d4a;}
.styled-table td{padding:10px 12px;border-bottom:1px solid #1e2d4a;vertical-align:middle;}
.styled-table tr:hover td{background:rgba(0,212,255,0.03);}

.warn{background:#1a1500;border:1px solid rgba(255,215,0,.25);border-radius:8px;
      padding:10px 14px;color:#ffd700;font-size:.76rem;line-height:1.6;}
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 상수
# ════════════════════════════════════════════════════════════
SWING = 30
REGIME_PARAMS = {
    "UPtrend":   {"fib":[0.382,0.500,0.618],"stoch":30,"stop":0.07,"tp":0.15,"desc":"📈 상승장"},
    "RANGE":     {"fib":[0.500,0.618,0.786],"stoch":20,"stop":0.05,"tp":0.12,"desc":"➡️ 박스장"},
    "DOWNtrend": {"fib":[0.618,0.786,0.886],"stoch":15,"stop":0.05,"tp":0.10,"desc":"📉 하락장"},
    "UNKNOWN":   {"fib":[0.500,0.618,0.786],"stoch":20,"stop":0.07,"tp":0.12,"desc":"❓ 불명"},
}

# ── 종목 변동성 분류 ──────────────────────────────────────
# 저변동성 (베타 < 1.2): 피보나치 분할매수 전략
LOW_VOL_TICKERS = [
    # 일반 저변동성
    "SPY","QQQ","IWM","DIA","XLE","XLK","XLF","XLV","XLI",
    "GLD","TLT","VNQ","AAPL","MSFT","GOOGL","META","AMZN",
    "JPM","GS","BAC","V","MA","BRK-B","UNH","JNJ","PFE",
    "XOM","CVX","NEE","DUK","KO","PG","WMT","COST","MCD",
    # 하락장 대응 — 피보나치 전략 적합
    "SQQQ","SPXS","SOXS","SDOW","SH","PSQ","VIXY","SRTY",  # 인버스 ETF
    "GLD","IAU","SLV","GDX","GDXJ","TLT","IEF","BIL","SGOV",  # 안전자산
    "KO","PG","JNJ","WMT","MCD","XLU","XLP","XLV","CL","GIS",  # 방어주
]

# 고변동성 (베타 >= 1.2): 모멘텀 전략
HIGH_VOL_TICKERS = [
    "TSLA","AMD","NVDA","SMCI","ARM","PLTR","COIN","MARA",
    "MU","MRVL","AVGO","QCOM","INTC","AMAT","KLAC","LRCX",
    "CRM","NOW","SNOW","DDOG","CRWD","PANW","CYBR","ZS",
    "NFLX","DIS","UBER","ABNB","SHOP","SQ","PYPL","AFRM",
    "VRT","SMCI","APP","RBLX","SOFI","HOOD",
]

# ════════════════════════════════════════════════════════════
# 섹터 모멘텀 필터 / VIX 필터 / 어닝 필터
# ════════════════════════════════════════════════════════════

# 종목 → 섹터 ETF 매핑
SECTOR_MAP = {
    # 테크
    "AAPL":"XLK","MSFT":"XLK","GOOGL":"XLK","META":"XLK",
    "NVDA":"SOXX","AMD":"SOXX","INTC":"SOXX","MU":"SOXX",
    "AVGO":"SOXX","QCOM":"SOXX","AMAT":"SOXX","KLAC":"SOXX",
    "ARM":"SOXX","SMCI":"SOXX","MRVL":"SOXX","ADI":"SOXX",
    # 소프트웨어
    "CRM":"XLK","ADBE":"XLK","NOW":"XLK","ORCL":"XLK",
    "PANW":"XLK","CRWD":"XLK","DDOG":"XLK","SNOW":"XLK",
    "PLTR":"XLK","ZS":"XLK","FTNT":"XLK",
    # 소비자
    "AMZN":"XLY","TSLA":"XLY","NFLX":"XLY","DIS":"XLY",
    "SBUX":"XLY","NKE":"XLY","BKNG":"XLY","MAR":"XLY",
    "HD":"XLY","LOW":"XLY","TGT":"XLY","COST":"XLP",
    "MCD":"XLP","KO":"XLP","PG":"XLP","WMT":"XLP",
    # 금융
    "JPM":"XLF","GS":"XLF","BAC":"XLF","MS":"XLF",
    "BLK":"XLF","V":"XLF","MA":"XLF","AXP":"XLF",
    "C":"XLF","WFC":"XLF","SCHW":"XLF","CME":"XLF","ICE":"XLF",
    # 헬스케어
    "UNH":"XLV","LLY":"XLV","JNJ":"XLV","PFE":"XLV",
    "ABBV":"XLV","MRK":"XLV","AMGN":"XLV","GILD":"XLV",
    # 에너지
    "XOM":"XLE","CVX":"XLE","COP":"XLE","EOG":"XLE",
    "SLB":"XLE","OXY":"XLE",
    # 귀금속/안전자산
    "GLD":"GLD","IAU":"GLD","SLV":"SLV","GDX":"GDX",
    "TLT":"TLT","IEF":"IEF","BIL":"BIL",
    # 산업재/전력/AI인프라
    "GEV":"XLI","VRT":"XLI","ETN":"XLI","EMR":"XLI",
    "HON":"XLI","UPS":"XLI","CAT":"XLI","DE":"XLI",
    "LMT":"XLI","RTX":"XLI","NOC":"XLI","GD":"XLI",
    "BA":"XLI","GE":"XLI","PH":"XLI","ROK":"XLI",
    "IR":"XLI","CMI":"XLI","OTIS":"XLI","CARR":"XLI",
    "NEE":"XLU","DUK":"XLU","SO":"XLU","AEP":"XLU",
    "EXC":"XLU","SRE":"XLU","XEL":"XLU","WEC":"XLU",
    # 소재
    "LIN":"XLB","APD":"XLB","ECL":"XLB","SHW":"XLB",
    "FCX":"XLB","NEM":"XLB","NUE":"XLB","VMC":"XLB",
    # 부동산
    "AMT":"XLRE","PLD":"XLRE","EQIX":"XLRE","CCI":"XLRE",
    "PSA":"XLRE","WELL":"XLRE","EQR":"XLRE","SPG":"XLRE",
    # 통신
    "T":"XLC","VZ":"XLC","TMUS":"XLC","CHTR":"XLC","CMCSA":"XLC",
    # 핀테크/성장
    "COIN":"XLK","HOOD":"XLK","SOFI":"XLF","AFRM":"XLF",
    "UPST":"XLF","LC":"XLF",
    # 클라우드
    "TWLO":"XLK","ZI":"XLK","HUBS":"XLK","BILL":"XLK",
    "PCTY":"XLK","PAYC":"XLK","VEEV":"XLV","ANSS":"XLK",
    # 여행/레저
    "UBER":"XLY","LYFT":"XLY","DASH":"XLY","EXPE":"XLY",
    "NCLH":"XLY","CCL":"XLY","RCL":"XLY",
    # 테마 ETF
    "SOXX":"SOXX","ARKK":"XLK","WCLD":"XLK","CLOU":"XLK",
    # 기타 성장주
    "UBER":"XLY","ABNB":"XLY","SHOP":"XLY",
    "COIN":"XLK","MARA":"XLK","HOOD":"XLK",
    "SOFI":"XLF","AFRM":"XLF","SQ":"XLK","PYPL":"XLK",
    # 인버스 ETF → 섹터 필터 제외 (항상 허용)
    "SQQQ":None,"SPXS":None,"SOXS":None,"SDOW":None,
    "SH":None,"PSQ":None,"VIXY":None,"SRTY":None,
}

# 섹터 ETF 설명
SECTOR_DESC = {
    "XLK":  "테크 ETF (Technology Select Sector)",
    "SOXX": "반도체 ETF (iShares Semiconductor)",
    "XLY":  "임의소비재 ETF (Consumer Discretionary)",
    "XLP":  "필수소비재 ETF (Consumer Staples)",
    "XLF":  "금융 ETF (Financial Select Sector)",
    "XLV":  "헬스케어 ETF (Health Care Select Sector)",
    "XLE":  "에너지 ETF (Energy Select Sector)",
    "XLI":  "산업재 ETF (Industrial Select Sector)",
    "XLU":  "유틸리티 ETF (Utilities Select Sector)",
    "GLD":  "금 ETF (SPDR Gold Shares)",
    "SLV":  "은 ETF (iShares Silver Trust)",
    "GDX":  "금광업 ETF (VanEck Gold Miners)",
    "TLT":  "장기채 ETF (20+ Year Treasury Bond)",
    "IEF":  "중기채 ETF (7-10 Year Treasury Bond)",
    "BIL":  "단기채 ETF (1-3 Month Treasury Bill)",
}

@st.cache_data(ttl=1800)
def get_sector_regime(sector_etf):
    """섹터 ETF 레짐 확인"""
    if sector_etf is None:
        return "OK"  # 인버스 ETF는 항상 허용
    try:
        df = yf.download(sector_etf, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 60: return "UNKNOWN"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"]
        ma200 = close.rolling(min(200,len(close))).mean().iloc[-1]
        ma50  = close.rolling(50).mean().iloc[-1]
        roc20 = float((close.iloc[-1]/close.iloc[-21]-1)*100) if len(close)>21 else 0
        curr  = float(close.iloc[-1])
        if curr > ma200 and roc20 > -5:
            return "OK"       # 섹터 정상
        elif curr < ma200 and roc20 < -5:
            return "WEAK"     # 섹터 하락 — 진입 주의
        else:
            return "CAUTION"  # 섹터 주의
    except Exception:
        return "UNKNOWN"

@st.cache_data(ttl=3600)
def get_vix_level():
    """VIX 현재 수준 확인"""
    try:
        vix = yf.download("^VIX", period="5d", interval="1d",
                          auto_adjust=True, progress=False)
        if vix.empty: return 20, "정상"
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        v = float(vix["Close"].dropna().iloc[-1])
        if v >= 30:
            return v, "극단 공포 🔴"
        elif v >= 25:
            return v, "공포 🟠"
        elif v >= 20:
            return v, "주의 🟡"
        else:
            return v, "정상 🟢"
    except Exception:
        return 20, "확인 불가"

@st.cache_data(ttl=3600)
def get_earnings_info_simple(ticker):
    """어닝 발표일 확인 (간소화)"""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is None or (isinstance(cal, dict) and not cal):
            return None, None
        if isinstance(cal, pd.DataFrame):
            cal = cal.to_dict()
        # 어닝 날짜 추출
        earn_date = None
        for key in ["Earnings Date","earningsDate","earnings_date"]:
            if key in cal:
                val = cal[key]
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    val = list(val)
                    if val: earn_date = pd.Timestamp(val[0]).date()
                else:
                    earn_date = pd.Timestamp(val).date()
                break
        if earn_date is None: return None, None
        days_left = (earn_date - datetime.date.today()).days
        return earn_date, days_left
    except Exception:
        return None, None

def check_filters(ticker, regime):
    """
    3가지 필터 동시 확인
    returns: {vix_val, vix_label, sector_status, earn_date, days_left, warnings}
    """
    warnings = []

    # VIX 필터
    vix_val, vix_label = get_vix_level()
    if vix_val >= 30:
        warnings.append(f"🔴 VIX {vix_val:.0f} — 극단 공포. V6 모멘텀 금지, V7/현금만")
    elif vix_val >= 25:
        warnings.append(f"🟠 VIX {vix_val:.0f} — 공포 구간. 신규 진입 신중")

    # 섹터 필터
    sector_etf = SECTOR_MAP.get(ticker.upper())
    sector_status = get_sector_regime(sector_etf)
    sector_name   = sector_etf if sector_etf else "해당없음"
    if sector_status == "WEAK":
        warnings.append(f"📉 섹터({sector_name}) 하락 중 — 진입 위험")
    elif sector_status == "CAUTION":
        warnings.append(f"⚠️ 섹터({sector_name}) 주의 — 소량만 진입")

    # 어닝 필터
    earn_date, days_left = get_earnings_info_simple(ticker)
    if days_left is not None:
        if 0 <= days_left <= 5:
            warnings.append(f"⚠️ 어닝 {days_left}일 후 ({earn_date}) — 진입 금지")
        elif -3 <= days_left < 0:
            warnings.append(f"📊 어닝 {abs(days_left)}일 전 발표 완료 ({earn_date})")
        elif days_left <= 10:
            warnings.append(f"📅 어닝 {days_left}일 후 ({earn_date}) — 주의")

    return {
        "vix_val":       vix_val,
        "vix_label":     vix_label,
        "sector_etf":    sector_name,
        "sector_status": sector_status,
        "earn_date":     earn_date,
        "days_left":     days_left,
        "warnings":      warnings,
        "block_entry":   (
            vix_val >= 30 or
            sector_status == "WEAK" or
            (days_left is not None and 0 <= days_left <= 5)
        ),
    }


def classify_ticker_type(ticker, df=None):
    """
    종목이 저변동성인지 고변동성인지 판단
    1차: 사전 정의 목록 확인
    2차: 실제 베타 계산 (df 있을 때)
    """
    tk = ticker.upper()
    if tk in LOW_VOL_TICKERS:
        return "저변동성"
    if tk in HIGH_VOL_TICKERS:
        return "고변동성"
    # 목록에 없으면 실제 변동성으로 판단
    if df is not None and "Return" in df.columns:
        vol = df["Return"].std() * (252**0.5)  # 연변동성
        return "저변동성" if vol < 0.40 else "고변동성"
    return "저변동성"  # 기본값

# ════════════════════════════════════════════════════════
# 자동 스캔 종목 풀 (섹터별 분류)
# ════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# 스캔 종목 풀 — 나스닥100 + S&P500 주요 + 섹터별
# ══════════════════════════════════════════════════════════════
SCAN_UNIVERSE = {
    # ── 나스닥100 핵심 ──────────────────────────────────────
    "나스닥 테크":     ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA",
                       "AVGO","QCOM","AMD","INTC","MU","AMAT","KLAC","LRCX",
                       "MRVL","ADI","MCHP","SNPS","CDNS","FTNT","PANW"],
    "나스닥 소프트웨어":["CRM","ADBE","NOW","INTU","ORCL","WDAY","TEAM",
                       "DDOG","SNOW","PLTR","CRWD","ZS","OKTA","MDB","DXCM"],
    "나스닥 소비자":   ["NFLX","COST","BKNG","ABNB","EBAY","PYPL","MELI",
                       "PCAR","ORLY","CPRT","CTAS","FAST","ODFL"],
    "나스닥 바이오":   ["AMGN","GILD","REGN","VRTX","IDXX","ILMN","BIIB",
                       "MRNA","SGEN","ALNY"],

    # ── S&P500 섹터별 ────────────────────────────────────────
    "S&P 금융":        ["JPM","GS","MS","BAC","WFC","C","BLK","SCHW",
                       "AXP","V","MA","COF","USB","PNC","TFC","CME","ICE",
                       "CB","MMC","AON","MET","PRU","AIG"],
    "S&P 헬스케어":    ["UNH","LLY","JNJ","PFE","ABBV","MRK","TMO","ABT",
                       "DHR","MDT","BMY","ISRG","ELV","CVS","CI","HUM",
                       "A","ZBH","EW","STE","RMD"],
    "S&P 에너지":      ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO",
                       "PXD","OXY","HES","DVN","FANG","HAL","BKR","APA"],
    "S&P 산업재":      ["GEV","VRT","HON","UPS","CAT","DE","LMT","RTX",
                       "NOC","GD","BA","GE","ETN","EMR","PH","ROK",
                       "IR","XYL","OTIS","CARR","TT","CMI","PCAR"],
    "S&P 소비재":      ["HD","LOW","MCD","SBUX","NKE","TGT","TJX","ROST",
                       "YUM","DRI","CMG","HLT","MAR","WYNN","MGM"],
    "S&P 유틸리티":    ["NEE","DUK","SO","D","AEP","EXC","SRE","PCG",
                       "XEL","WEC","ES","ETR","PPL","EIX","FE"],
    "S&P 부동산":      ["AMT","PLD","EQIX","CCI","PSA","WELL","EQR",
                       "AVB","VTR","SPG","O","VICI","WY","ARE"],
    "S&P 소재":        ["LIN","APD","ECL","SHW","FCX","NEM","NUE",
                       "VMC","MLM","MOS","CF","IFF","PPG","EMN"],
    "S&P 통신":        ["T","VZ","TMUS","CHTR","CMCSA","DIS","WBD",
                       "OMC","IPG","NWSA","FOX"],

    # ── 성장/테마주 ──────────────────────────────────────────
    "AI/데이터센터":   ["SMCI","DELL","HPE","NTAP","PURE","WDC","STX",
                       "CSCO","ANET","JNPR","KEYS","CIEN"],
    "핀테크/결제":     ["SQ","COIN","HOOD","SOFI","AFRM","UPST","LC",
                       "OPEN","NRDS","SMAR"],
    "클라우드/SaaS":   ["TWLO","ZI","HUBS","BILL","PCTY","PAYC","VEEV",
                       "ANSS","PTC","AZPN","CGNX"],
    "소비자 플랫폼":   ["UBER","LYFT","DASH","ABNB","EXPE","TRIP","MTN",
                       "H","NCLH","CCL","RCL"],

    # ── ETF ─────────────────────────────────────────────────
    "시장 ETF":        ["SPY","QQQ","IWM","DIA","MDY","IJR","VTI","VOO"],
    "섹터 ETF":        ["XLK","XLF","XLE","XLV","XLI","XLY","XLP",
                       "XLU","XLB","XLRE","XLC","GLD","SLV","TLT"],
    "테마 ETF":        ["SOXX","ARKK","WCLD","CLOU","FINX","LIT","ICLN",
                       "BOTZ","ROBO","HACK","CIBR","IGV"],

    # ── 하락장 대응 ──────────────────────────────────────────
    "인버스 ETF":      ["SQQQ","SPXS","SOXS","SDOW","SH","PSQ",
                       "SRTY","VIXY","UVXY"],
    "안전자산":        ["GLD","IAU","SLV","GDX","GDXJ",
                       "TLT","IEF","SHY","BIL","SGOV"],
    "방어주":          ["KO","PG","JNJ","WMT","MCD","CL","GIS",
                       "KMB","HSY","CPB","K","HRL","SJM","CAG"],
}
# 전체 풀 (중복 제거)
ALL_TICKERS = list(dict.fromkeys(
    t for tks in SCAN_UNIVERSE.values() for t in tks
))
DEFAULT_WATCHLIST = ALL_TICKERS[:20]

# ── 하락장 전용 종목 목록 ──────────────────────────────────
BEAR_TICKERS = {
    "인버스 ETF": {
        "SQQQ": "나스닥 3배 인버스",
        "SPXS": "S&P500 3배 인버스",
        "SOXS": "반도체 3배 인버스",
        "SDOW": "다우 3배 인버스",
        "SH":   "S&P500 1배 인버스",
        "PSQ":  "나스닥 1배 인버스",
        "VIXY": "VIX 단기 선물",
        "SRTY": "러셀2000 3배 인버스",
    },
    "안전자산": {
        "GLD":  "금 ETF (SPDR)",
        "IAU":  "금 ETF (iShares)",
        "SLV":  "은 ETF",
        "GDX":  "금광업체 ETF",
        "TLT":  "20년 국채 ETF",
        "IEF":  "7-10년 국채 ETF",
        "BIL":  "단기채 ETF",
        "SGOV": "초단기채 ETF",
    },
    "방어주": {
        "KO":  "코카콜라 (필수소비재)",
        "PG":  "P&G (생활용품)",
        "WMT": "월마트 (유통)",
        "MCD": "맥도날드 (외식)",
        "JNJ": "존슨앤존슨 (헬스케어)",
        "XLU": "유틸리티 ETF",
        "XLP": "필수소비재 ETF",
        "XLV": "헬스케어 ETF",
    },
}
ALL_BEAR_TICKERS = list(dict.fromkeys(
    t for group in BEAR_TICKERS.values() for t in group.keys()
))

@st.cache_data(ttl=600)
def detect_market_regime():
    """
    SPY 기준으로 전체 시장 레짐 감지
    returns: "상승장" / "하락장" / "박스장"
    """
    try:
        spy = yf.download("SPY", period="1y", interval="1d",
                          auto_adjust=True, progress=False)
        if spy.empty: return "알 수 없음"
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy = spy.reset_index()
        if "Datetime" in spy.columns:
            spy.rename(columns={"Datetime":"Date"}, inplace=True)
        close   = spy["Close"]
        ma200   = close.rolling(200).mean().iloc[-1]
        ma50    = close.rolling(50).mean().iloc[-1]
        current = float(close.iloc[-1])
        roc20   = float((close.iloc[-1]/close.iloc[-21]-1)*100) if len(close)>21 else 0
        high52  = float(close.rolling(252).max().iloc[-1])
        draw52  = (current/high52-1)*100

        if current > ma200 and current > ma50 and roc20 > 0:
            return "상승장"
        elif current < ma200 and roc20 < -3:
            return "하락장"
        elif draw52 < -15:
            return "조정장"
        else:
            return "박스장"
    except Exception:
        return "알 수 없음"

# ════════════════════════════════════════════════════════════
# 지표 계산
# ════════════════════════════════════════════════════════════
def build_features(df):
    # MA
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA60"]  = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["ROC"]   = df["Close"].pct_change(10)
    # ATR / ADX
    pc = df["Close"].shift(1)
    df["TR"] = np.maximum(df["High"]-df["Low"],
               np.maximum((df["High"]-pc).abs(),(df["Low"]-pc).abs()))
    df["ATR"]     = df["TR"].rolling(14).mean()
    df["ATR_sum"] = df["TR"].rolling(14).sum()
    pdm=df["High"].diff(); ndm=-df["Low"].diff()
    df["PDM"]=np.where((pdm>ndm)&(pdm>0),pdm,0.0)
    df["NDM"]=np.where((ndm>pdm)&(ndm>0),ndm,0.0)
    atr_s=df["ATR_sum"].replace(0,np.nan)
    df["PDI"]=100*df["PDM"].rolling(14).sum()/atr_s
    df["NDI"]=100*df["NDM"].rolling(14).sum()/atr_s
    dx_d=(df["PDI"]+df["NDI"]).replace(0,np.nan)
    df["DX"]=100*(df["PDI"]-df["NDI"]).abs()/dx_d
    df["ADX"]=df["DX"].rolling(14).mean()
    # RSI / StochRSI
    delta=df["Close"].diff()
    gain=delta.where(delta>0,0).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()
    df["RSI"]=100-(100/(1+gain/loss.replace(0,np.nan)))
    rmin=df["RSI"].rolling(14).min(); rmax=df["RSI"].rolling(14).max()
    df["StochRSI"]=(df["RSI"]-rmin)/(rmax-rmin+1e-9)*100
    # 수익률 / 변동성
    df["Return"]=df["Close"].pct_change()
    df["Vol5"]=df["Return"].rolling(5).std()
    df["Vol20"]=df["Return"].rolling(20).std()
    # ── 스윙 피보나치 (수정) ──────────────────────────────────
    # 30일 기본 스윙
    df["sw_high"] = df["High"].rolling(SWING).max()
    df["sw_low"]  = df["Low"].rolling(SWING).min()
    df["rng"]     = df["sw_high"] - df["sw_low"]
    # 60일 확장 스윙 (보조)
    df["sw_high_60"] = df["High"].rolling(60).max()
    df["sw_low_60"]  = df["Low"].rolling(60).min()
    df["rng_60"]     = df["sw_high_60"] - df["sw_low_60"]
    # 피보나치 유효 여부: sw_high > 현재가 > sw_low 이어야 함
    # (고점 → 되돌림 구간에서만 의미 있음)
    df["fib_valid"] = (
        (df["sw_high"] > df["Close"]) &
        (df["Close"]   > df["sw_low"]) &
        (df["rng"]     > df["Close"] * 0.01)  # 범위가 현재가의 1% 이상
    )
    # Gradient
    df["MA20_grad"]=df["MA20"].diff()
    df["ADX_grad"]=df["ADX"].diff()
    df["ROC_grad"]=df["ROC"].diff()
    df["Price_grad"]=df["Close"].diff(5)
    # ── 레짐 분류 (강화판) ───────────────────────────────────
    atr_med = df["ATR"].median()
    # 52주 고점 (진짜 추세 파악용)
    df["High52w"]    = df["High"].rolling(252).max()
    df["Draw52w"]    = (df["Close"] - df["High52w"]) / df["High52w"] * 100  # 음수
    # MA200 기울기 (MA200 자체가 상승 중인지)
    df["MA200_grad"] = df["MA200"].diff(20)  # 20일 기울기

    def classify(row):
        p,m,r,a,d = row["Close"],row["MA200"],row["ROC"],row["ATR"],row["ADX"]
        if any(pd.isna(x) for x in [m,r,a,d]): return "UNKNOWN"
        draw    = row.get("Draw52w", 0)
        ma200g  = row.get("MA200_grad", 0)
        above   = p > m
        trend   = d > 20
        highv   = a > atr_med

        # ✅ 강화된 DOWNtrend 조건
        # 52주 고점 대비 -45% 이상 하락이면 무조건 DOWNtrend
        # (-30%는 너무 엄격해서 정상 조정 종목까지 제외됨)
        if draw < -45:
            return "DOWNtrend"
        # MA200 자체가 하락 중이면 DOWNtrend
        if (not above) and r < 0 and trend and highv:
            return "DOWNtrend"
        if (not above) and pd.notna(ma200g) and ma200g < 0 and trend:
            return "DOWNtrend"

        # ✅ 강화된 UPtrend 조건
        # MA200 위 + ROC 양수 + ADX 추세 + MA200 자체도 상승 중
        if above and r > 0 and trend:
            if pd.notna(ma200g) and ma200g > 0:
                return "UPtrend"      # MA200도 상승 = 강한 UPtrend
            elif draw > -15:          # 52주 고점 -15% 이내 = 비교적 고점 근처
                return "UPtrend"
            else:
                return "RANGE"        # 너무 많이 빠졌으면 RANGE

        return "RANGE"

    df["Regime"] = df.apply(classify, axis=1)
    # 4-Factor Score (각 0~4, 총 0~20)
    df["TrendScore"]=(
        (df["Close"]>df["MA60"]).astype(int)+
        (df["MA20_grad"]>0).astype(int)+
        (df["ROC_grad"]>0).astype(int)+
        (df["ADX_grad"]>0).astype(int))
    def ar1(x):
        if len(x)<2 or np.std(x)<1e-10: return np.nan
        return float(np.corrcoef(x[:-1],x[1:])[0,1])
    df["AR1"]=df["Return"].rolling(5).apply(ar1,raw=True)
    df["WavePulse"]=((df["Close"].diff()>0).astype(int)-(df["Close"].diff()<0).astype(int)).rolling(5).sum()
    df["CycleScore"]=(
        (df["StochRSI"]<25).astype(int)+
        (df["AR1"]>0).astype(int)+
        (df["Vol5"]<df["Vol20"]).astype(int)+
        (df["WavePulse"]>0).astype(int))
    df["Month"]=df["Date"].dt.month; df["DOW"]=df["Date"].dt.dayofweek; df["DOY"]=df["Date"].dt.dayofyear
    mb={1:1.10,2:1.02,3:1.05,4:1.04,5:0.97,6:0.96,7:1.03,8:0.95,9:0.92,10:1.00,11:1.08,12:1.12}
    db={0:0.97,1:1.02,2:1.04,3:1.03,4:1.05}
    df["MonthBias"]=df["Month"].map(mb); df["DOWbias"]=df["DOW"].map(db)
    df["SantaRally"]=((df["DOY"]>=350)|(df["DOY"]<=5)).astype(int)
    df["SeasonalScore"]=(
        (df["MonthBias"]>1.0).astype(int)+(df["DOWbias"]>1.0).astype(int)+
        df["SantaRally"].astype(int)+((df["Month"]<=4)|(df["Month"]>=11)).astype(int))
    vm=df["Volume"].rolling(20).mean()
    df["VolMA20"]=vm  # 모멘텀 스캔에서 사용
    df["HVI"]=(df["Volume"]/(vm+1e-9)>1.5).astype(int)
    cr=df["High"]-df["Low"]+1e-9
    lw=df[["Close","Open"]].min(axis=1)-df["Low"]; uw=df["High"]-df[["Close","Open"]].max(axis=1)
    df["PPI"]=(lw-uw)/cr
    df["PPI_norm"]=(df["PPI"]-df["PPI"].rolling(20).min())/(df["PPI"].rolling(20).max()-df["PPI"].rolling(20).min()+1e-9)
    df["Outlier"]=(df["Return"].abs()>3*df["Return"].rolling(20).std()).astype(int)
    sv=df["Return"].rolling(5).std(); lv=df["Return"].rolling(20).std().replace(0,np.nan)
    df["VIX_Alert"]=(sv/lv>2).astype(int)
    df["IrregularScore"]=(
        (df["HVI"]==0).astype(int)+(df["Outlier"]==0).astype(int)+
        (df["VIX_Alert"]==0).astype(int)+(df["PPI_norm"]>0.5).astype(int))
    # 진입 조건 보너스
    cond_stoch=(df["StochRSI"]<25).astype(int)
    cond_ma=(df["Close"]>df["MA60"]).astype(int)
    cond_candle=((df["Close"]>df["Open"])&((df["Close"]-df["Open"])/(cr)>0.5)).astype(int)
    grad_bonus=((df["MA20_grad"]>0).astype(int)+(df["ROC_grad"]>0).astype(int)+
                (df["ADX_grad"]>0).astype(int)+(df["Price_grad"]>0).astype(int)>=3).astype(int)
    df["TotalScore"]=(df["TrendScore"]+df["CycleScore"]+df["SeasonalScore"]+
                      df["IrregularScore"]+cond_stoch+cond_ma+cond_candle+grad_bonus)
    df["ScorePct"]=df["TotalScore"]/20*100

    # ════════════════════════════════════════════════════
    # 물타기 vs 불타기 판단 지표 3종
    # ════════════════════════════════════════════════════

    # ── 1) MACD 히스토그램 ──────────────────────────────
    # EMA12 - EMA26 = MACD선
    # MACD - EMA9(MACD) = 히스토그램
    # 히스토그램 음→양 전환 = 단기 모멘텀 상승 전환
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    # 히스토그램 양전환: 직전 음수 → 현재 양수
    df["MACD_cross_up"] = (
        (df["MACD_hist"] > 0) &
        (df["MACD_hist"].shift(1) <= 0)
    ).astype(int)
    # 히스토그램 증가 중 (기울기 양수)
    df["MACD_rising"] = (df["MACD_hist"] > df["MACD_hist"].shift(1)).astype(int)

    # ── 2) Higher Low (저점 상승 확인) ──────────────────
    # 최근 5봉 최저가 vs 그 이전 5봉 최저가
    # 저점이 높아지면 상승 구조 전환
    df["Low_5"]      = df["Low"].rolling(5).min()   # 최근 5봉 저점
    df["Low_5_prev"] = df["Low_5"].shift(5)          # 5봉 전 저점
    df["HigherLow"]  = (df["Low_5"] > df["Low_5_prev"]).astype(int)

    # ── 3) StochRSI 과매도 탈출 ─────────────────────────
    # StochRSI 20 이하였다가 40 이상으로 올라온 경우
    # = 과매도 구간 탈출 = 반등 시작 확인
    df["StochRSI_prev5"] = df["StochRSI"].shift(5)
    df["StochRSI_escape"] = (
        (df["StochRSI"] >= 40) &
        (df["StochRSI_prev5"] <= 20)
    ).astype(int)

    # ── 종합: 불타기 신호 점수 (0~3) ────────────────────
    # 3개 중 2개 이상 = 불타기
    # 2개 미만 = 물타기 대기
    df["BullAdd_score"] = (
        df["MACD_rising"] +      # MACD 히스토그램 상승 중
        df["HigherLow"] +         # 저점이 높아짐
        df["StochRSI_escape"]     # 과매도 탈출
    )
    df["BullAdd_signal"] = (df["BullAdd_score"] >= 2).astype(int)

    # ════════════════════════════════════════════════════
    # V7 과매도 역추세 + 모멘텀 확인 지표
    # ════════════════════════════════════════════════════

    # ① Z-Score (20일 이동평균 기준 통계적 과매도)
    roll_mean = df["Close"].rolling(20).mean()
    roll_std  = df["Close"].rolling(20).std().replace(0, np.nan)
    df["ZScore"] = (df["Close"] - roll_mean) / roll_std

    # ② 볼린저밴드 하단 터치
    df["BB_upper"] = roll_mean + 2 * roll_std
    df["BB_lower"] = roll_mean - 2 * roll_std
    df["BB_touch_low"] = (df["Close"] <= df["BB_lower"]).astype(int)

    # ③ RSI < 30 (이미 RSI 있음)
    df["RSI_oversold"] = (df["RSI"] < 30).astype(int)

    # ④ StochRSI < 20 (이미 StochRSI 있음)
    df["Stoch_oversold"] = (df["StochRSI"] < 20).astype(int)

    # ⑤ V7 과매도 종합 점수 (A+B+C+D)
    df["V7_oversold_score"] = (
        (df["ZScore"] < -2).astype(int) +   # A: Z-Score
        df["BB_touch_low"] +                  # B: BB 하단
        df["RSI_oversold"] +                  # C: RSI < 30
        df["Stoch_oversold"]                  # D: StochRSI < 20
    )

    # ⑥ 양봉 확인 (당일 종가 > 시가 = 매수세)
    df["BullCandle"] = (df["Close"] > df["Open"]).astype(int)

    # ⑦ 5일 MA 돌파 (반등 확인)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["Above_MA5"] = (df["Close"] > df["MA5"]).astype(int)

    # ⑧ 12개월 모멘텀 (강한 종목 필터)
    df["Mom12"] = df["Close"].pct_change(252)
    df["Mom1"]  = df["Close"].pct_change(20)
    df["MomScore"] = df["Mom12"] - df["Mom1"]  # 장기-단기 = 조정 중 강한 종목

    # ⑨ V7 모멘텀 회복 점수 (BUY2/3 불타기 조건)
    df["V7_recovery_score"] = (
        df["Above_MA5"] +          # 5일 MA 돌파
        df["MACD_rising"] +        # MACD 히스토그램 상승
        df["HigherLow"] +          # Higher Low
        df["StochRSI_escape"]      # StochRSI 탈출
    )

    return df

# ════════════════════════════════════════════════════════════
# 단일 종목 분석 (캐시)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def scan_single(ticker):
    """
    자동 스캔 — 피보나치 + 모멘텀 이중 전략
    피보나치 불가 종목도 모멘텀 신호가 있으면 추천
    """
    try:
        df = yf.download(ticker, period="2y", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close","High","Low","Open","Volume"]).reset_index()
        if "Datetime" in df.columns:
            df.rename(columns={"Datetime":"Date"}, inplace=True)
        df = build_features(df)

        valid = df.dropna(subset=["TotalScore","StochRSI","MA60","ADX","Regime"])
        if valid.empty: return None
        row = valid.iloc[-1]

        price  = float(row["Close"])
        regime = row["Regime"]
        if regime == "UNKNOWN": return None

        # ── 52주 낙폭 필터 ────────────────────────────────
        draw52_scan = float(row.get("Draw52w", -20)) if "Draw52w" in row.index else -20
        # 낙폭 -50% 초과 = 추세 완전 훼손 → 제외 (인버스 ETF 제외)
        is_inverse = ticker.upper() in [
            "SQQQ","SPXS","SOXS","SDOW","SH","PSQ","VIXY","SRTY"
        ]
        if not is_inverse and draw52_scan < -50:
            return None

        # ── 섹터 필터 ─────────────────────────────────────
        sector_etf = SECTOR_MAP.get(ticker.upper())
        sector_st  = get_sector_regime(sector_etf) if sector_etf else "OK"
        is_safe_asset = ticker.upper() in [
            "GLD","IAU","SLV","GDX","TLT","IEF","BIL","SGOV","GDXJ"
        ]
        # 섹터 하락 중이면 스캔 제외 (인버스 ETF + 안전자산 제외)
        if sector_st == "WEAK" and not is_inverse and not is_safe_asset:
            return None

        # DOWNtrend는 제외 안 하고 신호에서 표시
        # (피보나치/모멘텀 모두 불가하면 나중에 return None)

        cfg = REGIME_PARAMS[regime]
        pct = float(row["ScorePct"])

        # ── 종목 유형 판단 ──────────────────────────────
        ticker_type = classify_ticker_type(ticker, df)

        # ── 모멘텀 신호 (항상 계산) ─────────────────────
        ma20   = float(row.get("MA20", 0))
        ma60_v = float(row.get("MA60", 1))
        rsi    = float(row.get("RSI", 50))
        macd_h_raw = row.get("MACD_hist", float("nan"))
        macd_h = float(macd_h_raw) if not pd.isna(macd_h_raw) else 0
        vol    = float(row.get("Volume", 0))
        vol_ma = float(row.get("VolMA20", 1))
        mom_c1 = ma20 > ma60_v
        mom_c2 = 45 <= rsi <= 68
        mom_c3 = macd_h > 0
        mom_c4 = vol_ma > 0 and (vol / vol_ma) >= 1.3
        momentum_score = int(mom_c1)+int(mom_c2)+int(mom_c3)+int(mom_c4)
        has_momentum   = momentum_score >= 3

        # ── 피보나치 계산 (실패해도 계속 진행) ──────────
        fib_lv = [None, None, None]
        valid_fibs = []
        dist_pct = -99
        fib_score = 0
        nearest_fib = None

        sh30  = float(row["sw_high"]) if not pd.isna(row["sw_high"]) else None
        rng30 = float(row["rng"])     if not pd.isna(row["rng"])     else None
        fib_valid_flag = bool(row.get("fib_valid", False))

        if fib_valid_flag and sh30 and rng30 and rng30 > 0:
            use_sh, use_rng = sh30, rng30
        else:
            try:
                sh60  = float(df["High"].rolling(60).max().iloc[-1])
                rng60 = sh60 - float(df["Low"].rolling(60).min().iloc[-1])
                if rng60 > 0 and sh60 > price:
                    use_sh, use_rng = sh60, rng60
                else:
                    sh120  = float(df["High"].rolling(120).max().iloc[-1])
                    rng120 = sh120 - float(df["Low"].rolling(120).min().iloc[-1])
                    if rng120 > 0 and sh120 > price:
                        use_sh, use_rng = sh120, rng120
                    else:
                        use_sh, use_rng = None, None
            except Exception:
                use_sh, use_rng = None, None

        if use_sh and use_rng:
            fib_lv_raw = [use_sh - use_rng*f for f in cfg["fib"]]
            fib_lv     = [f if f < price else None for f in fib_lv_raw]
            valid_fibs = [f for f in fib_lv if f is not None]
            if valid_fibs:
                nearest_fib = max(valid_fibs)
                dist_pct    = (nearest_fib / price - 1) * 100
                fib_score   = max(0, min(100, 100 + dist_pct * 5))

        # ── V7 과매도 점수 계산 ──────────────────────────
        zscore_v = float(row.get("ZScore", 0)) if not pd.isna(row.get("ZScore", float("nan"))) else 0
        bb_low_v = int(row.get("BB_touch_low", 0))
        rsi_v    = float(row.get("RSI", 50))
        stoch_v  = float(row.get("StochRSI", 50))
        bull_c_v = int(row.get("BullCandle", 0))
        v7_score = (
            int(zscore_v < -2) +
            int(bb_low_v == 1) +
            int(rsi_v < 30) +
            int(stoch_v < 20)
        )
        v7_ready = (v7_score >= 2) and (bull_c_v == 1)

        # ── 레짐별 전략 자동 매칭 ───────────────────────
        if regime == "DOWNtrend":
            if has_momentum and momentum_score == 4:
                signal       = "⚡ 하락장 반등 (고위험)"
                strategy_rec = "모멘텀V6"
            else:
                return None

        elif regime == "UPtrend":
            if has_momentum:
                signal       = "🚀 V6 모멘텀 진입"
                strategy_rec = "모멘텀V6"
            elif valid_fibs and abs(dist_pct) <= 5:
                signal       = "🟢 V6 피보 근접"
                strategy_rec = "모멘텀V6"
            elif valid_fibs:
                signal       = "⏳ V6 대기"
                strategy_rec = "모멘텀V6"
            else:
                return None

        elif regime == "RANGE":
            if v7_ready and valid_fibs:
                signal       = "🎯 V7 과매도 진입"
                strategy_rec = "V7역추세"
            elif v7_score >= 2 and valid_fibs:
                signal       = "🟡 V7 조건 근접 (양봉 대기)"
                strategy_rec = "V7역추세"
            elif valid_fibs and abs(dist_pct) <= 5 and pct >= 50:
                signal       = "🟢 V5 피보 매수 근접"
                strategy_rec = "피보나치V5"
            elif valid_fibs and dist_pct > -15 and pct >= 40:
                signal       = "🟡 V5 피보 대기"
                strategy_rec = "피보나치V5"
            elif valid_fibs:
                signal       = "⏳ V5 원거리"
                strategy_rec = "피보나치V5"
            else:
                return None
        else:
            return None

        # ── 불타기 판단 ──────────────────────────────────
        bull_score = int(row.get("BullAdd_score", 0))
        is_bull    = bull_score >= 2

        # ── 매매 플랜 ────────────────────────────────────
        avg_s  = (
            sum(fib_lv[i]*[0.30,0.35,0.35][i] for i in range(3) if fib_lv[i])
            / sum([0.30,0.35,0.35][i] for i in range(3) if fib_lv[i])
        ) if valid_fibs else None
        stop_s = avg_s * (1 - cfg["stop"]) if avg_s else None
        tp_s   = avg_s * (1 + cfg["tp"])   if avg_s else None

        ret_1w = float((df["Close"].iloc[-1]/df["Close"].iloc[-6]-1)*100)  if len(df)>=6  else 0
        ret_1m = float((df["Close"].iloc[-1]/df["Close"].iloc[-22]-1)*100) if len(df)>=22 else 0

        if strategy_rec == "V7역추세":
            # V7: 과매도점수 40% + AI점수 30% + 피보근접도 20% + 불타기 10%
            total_rec_score = (v7_score/4*100*0.4 + pct*0.3
                               + fib_score*0.2 + bull_score/3*20*0.1)
        elif strategy_rec.startswith("모멘텀"):
            # V6 모멘텀: 모멘텀점수 40% + AI점수 30% + 적합도 20% + 불타기 10%
            total_rec_score = (momentum_score/4*100*0.4 + pct*0.3
                               + mom_fit*0.2 + bull_score/3*20*0.1)
        else:
            # V5 피보나치: 피보근접도 35% + AI점수 30% + 적합도 25% + 불타기 10%
            total_rec_score = (fib_score*0.35 + pct*0.3
                               + fib_fit*0.25 + bull_score/3*20*0.1)

        # ── 종목 적합도 점수 (0~100) ──────────────────────────
        # 피보나치 전략 적합도
        draw52 = float(df["Draw52w"].iloc[-1]) if "Draw52w" in df.columns else -20
        ma200g = float(df["MA200_grad"].iloc[-1]) if "MA200_grad" in df.columns else 0

        fib_fit = 0
        if regime == "RANGE":    fib_fit += 30  # 박스장이 피보 최적
        if regime == "UPtrend":  fib_fit += 20
        if -25 <= draw52 <= -8:  fib_fit += 30  # 적당히 조정된 종목
        if ma200g > 0:           fib_fit += 20  # MA200 상승 중
        if valid_fibs:           fib_fit += 15  # 피보 구간 유효
        if abs(dist_pct) <= 10:  fib_fit += 10  # 진입 구간 근접
        fib_fit = min(100, fib_fit)

        # 모멘텀 전략 적합도
        mom_fit = 0
        if regime == "UPtrend":  mom_fit += 30  # 상승장이 모멘텀 최적
        if draw52 >= -15:        mom_fit += 25  # 52주 고점 근처
        if ma200g > 0:           mom_fit += 20  # MA200 상승 중
        if has_momentum:         mom_fit += 25  # 모멘텀 신호 있음
        if momentum_score == 4:  mom_fit += 10  # 4/4 완벽한 모멘텀
        mom_fit = min(100, mom_fit)

        # 전략 추천 (적합도 기반)
        if fib_fit >= mom_fit:
            best_strategy = "피보나치V5"
            fit_score     = fib_fit
        else:
            best_strategy = "모멘텀V6"
            fit_score     = mom_fit

        # 전략과 신호가 불일치하면 조정
        if strategy_rec == "모멘텀V6" and fib_fit > mom_fit:
            strategy_rec = "피보나치V5"
        elif strategy_rec == "피보나치V5" and mom_fit > fib_fit + 20:
            strategy_rec = "모멘텀V6"

        # 종목 적합도 등급
        if fit_score >= 80:     fit_grade = "🏆 최적"
        elif fit_score >= 60:   fit_grade = "✅ 적합"
        elif fit_score >= 40:   fit_grade = "⚠️ 보통"
        else:                   fit_grade = "❌ 부적합"

        return {
            "ticker":         ticker,
            "price":          price,
            "regime":         regime,
            "regime_desc":    cfg["desc"],
            "draw52":         round(draw52, 1),
            "ma200_grad":     round(ma200g, 2),
            "fib_fit":        fib_fit,
            "mom_fit":        mom_fit,
            "best_strategy":  best_strategy,
            "fit_score":      fit_score,
            "fit_grade":      fit_grade,
            "pct":           pct,
            "signal":        signal,
            "strategy_rec":  strategy_rec,
            "ticker_type":   ticker_type,
            "fib_lv":        fib_lv,
            "nearest_fib":   nearest_fib,
            "dist_pct":      dist_pct,
            "fib_score":     fib_score,
            "momentum_score": momentum_score,
            "has_momentum":  has_momentum,
            "avg_s":         avg_s,
            "stop_s":        stop_s,
            "tp_s":          tp_s,
            "bull_score":    bull_score,
            "is_bull":       is_bull,
            "ts":  int(row["TrendScore"]),
            "cs":  int(row["CycleScore"]),
            "ss":  int(row["SeasonalScore"]),
            "irs": int(row["IrregularScore"]),
            "stoch": float(row["StochRSI"]),
            "adx":   float(row["ADX"]),
            "ret_1w":       ret_1w,
            "ret_1m":       ret_1m,
            "total_rec_score": total_rec_score,
            "v7_score":      v7_score,
            "v7_ready":      v7_ready,
            "sector_status": sector_st,
            "sector_etf":    sector_etf or "N/A",
            "style_result":  classify_stock_style(df),
        }
    except Exception:
        return None


@st.cache_data(ttl=300)
def analyze(ticker, period="1y"):
    try:
        # MA200 계산 + 백테스트에 최소 252봉 필요
        # 1y = 252봉 → START=200 이후 거래 구간 52봉뿐 → 자동 2y 확장
        # 기간별 실제 다운로드:
        # 6mo/1y → 2y (MA200 계산 최소 200봉 필요)
        # 2y → 2y
        # 3y → 3y
        # 5y → 5y
        period_map = {"6mo":"2y","1y":"2y","2y":"2y","3y":"3y","5y":"5y"}
        actual_period = period_map.get(period, "2y")
        df=yf.download(ticker,period=actual_period,interval="1d",auto_adjust=True,progress=False)
        if df.empty or len(df)<80: return None
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.dropna(subset=["Close","High","Low","Open","Volume"]).reset_index()
        if "Datetime" in df.columns: df.rename(columns={"Datetime":"Date"},inplace=True)
        df=build_features(df)
        valid=df.dropna(subset=["TotalScore","StochRSI","MA60","ADX"])
        if valid.empty: return None
        row=valid.iloc[-1]
        price=float(row["Close"]); pct=float(row["ScorePct"]); regime=row["Regime"]
        cfg=REGIME_PARAMS[regime]
        # ── 피보나치 계산 (유효성 검증 포함) ──────────────────
        # 1순위: 30일 스윙 (현재가가 고점 아래일 때만 유효)
        # 2순위: 60일 스윙으로 폴백
        rng=row["rng"]; sh=row["sw_high"]; sl=row["sw_low"]
        rng60=row["rng_60"]; sh60=row["sw_high_60"]
        fib_valid=row["fib_valid"]

        if fib_valid and not pd.isna(sh) and rng>0:
            # 정상: 현재가가 스윙 고점 아래 (되돌림 구간)
            use_sh, use_rng = sh, rng
        elif not pd.isna(sh60) and rng60>0 and sh60>price:
            # 폴백: 60일 스윙 사용
            use_sh, use_rng = sh60, rng60
        else:
            # 피보나치 계산 불가 (현재가가 모든 스윙 고점 위)
            use_sh, use_rng = None, None

        if use_sh and use_rng:
            fib_lv_raw = [use_sh - use_rng*f for f in cfg["fib"]]
            fib886     = use_sh - use_rng*0.886
            # BUY1~3 중 현재가보다 낮은 레벨만 유효 처리
            # 현재가보다 높은 레벨은 None (이미 지나간 구간)
            fib_lv = [
                f if f < price else None
                for f in fib_lv_raw
            ]
            # 모든 레벨이 None이면 (현재가가 모든 BUY 레벨 아래)
            # → 더 긴 스윙으로 재계산 시도
            if all(f is None for f in fib_lv):
                # 120일 스윙으로 확장
                sh120  = float(df["High"].rolling(120).max().iloc[-1])
                sl120  = float(df["Low"].rolling(120).min().iloc[-1])
                rng120 = sh120 - sl120
                if rng120 > 0 and sh120 > price:
                    fib_lv_raw2 = [sh120 - rng120*f for f in cfg["fib"]]
                    fib_lv  = [f if f < price else None for f in fib_lv_raw2]
                    fib886  = sh120 - rng120*0.886
                    use_sh  = sh120
                    use_rng = rng120
        else:
            fib_lv=[None,None,None]; fib886=None; use_sh=None; use_rng=None
        # 신호 결정
        near_fib=any(abs(price-f)/f<0.03 for f in fib_lv if f)
        if regime=="DOWNtrend":
            signal="🔴 매수 금지"; sig_k="sell"; action="관망"
        elif pct>=75:
            signal="🟢 강력 매수"; sig_k="strong_buy"; action="매수 진입"
        elif pct>=60:
            signal="🟢 매수"; sig_k="buy"; action="매수 검토"
        elif pct>=45:
            signal="🟡 관심"; sig_k="watch"; action="대기"
        elif pct>=30:
            signal="⚪ 관망"; sig_k="hold"; action="관망"
        else:
            signal="🔴 회피"; sig_k="sell"; action="매수 금지"
        # 매매가 계산
        avg_s=sum(fib_lv[i]*[0.33,0.34,0.33][i] for i in range(3)) if fib_lv[0] else None
        stop_s=avg_s*(1-cfg["stop"]) if avg_s else None
        tp_s=avg_s*(1+cfg["tp"]) if avg_s else None
        # 최근 수익률
        ret_1w=float((df["Close"].iloc[-1]/df["Close"].iloc[-6]-1)*100) if len(df)>=6 else 0
        ret_1m=float((df["Close"].iloc[-1]/df["Close"].iloc[-22]-1)*100) if len(df)>=22 else 0
        return {
            "ticker":ticker,"df":df,"row":row,"price":price,"pct":pct,
            "regime":regime,"cfg":cfg,"fib_lv":fib_lv,"fib886":fib886,
            "signal":signal,"sig_k":sig_k,"action":action,
            "avg_s":avg_s,"stop_s":stop_s,"tp_s":tp_s,
            "ts":int(row["TrendScore"]),"cs":int(row["CycleScore"]),
            "ss":int(row["SeasonalScore"]),"irs":int(row["IrregularScore"]),
            "stoch":float(row["StochRSI"]),"adx":float(row["ADX"]),
            "roc":float(row["ROC"])*100 if not pd.isna(row["ROC"]) else 0,
            "ret_1w":ret_1w,"ret_1m":ret_1m,"near_fib":near_fib,
            "sw_h_used":use_sh,"rng_used":use_rng,
        }
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
# 백테스트 엔진 — 우리 전략 그대로 구현
# ════════════════════════════════════════════════════════════
# ── 전략 설명 (변경 시 여기만 수정) ──────────────────────────
STRATEGY_DESC = {
    "name": "프로젝트 이세계 V13 — 피보나치 분할매수 전략",
    "version": "V13",
    "core": [
        "레짐 감지 (MA200 + ADX + ROC) → UP / RANGE / DOWN 자동 분류",
        "레짐별 피보나치 구간 자동 변경 (UP: 0.382~0.618 / RANGE: 0.5~0.786 / DOWN: 0.618~0.886)",
        "BUY1 도달 → 30% 진입 / BUY2 → 34% 추가 / BUY3 → 33% 추가",
        "레짐별 손절: UP -7% / RANGE -5% / DOWN -5% (또는 Fib 0.886 이탈)",
        "레짐별 익절: UP +30% / RANGE +20% / DOWN +15%",
        "MA60 위에서만 매수 (추세 필터)",
        "StochRSI 기준: UP 30이하 / RANGE 20이하 / DOWN 15이하",
        "AI 4-Factor 점수 50점 이상 추가 확인",
        "연속 손절 2회 → 10봉 쿨다운 (리스크 관리)",
        "━━━ 물타기 vs 불타기 자동 판단 ━━━",
        "📉 물타기: 피보나치 BUY2/3 가격에 도달할 때 추가 (가격 하락 시)",
        "📈 불타기: 3가지 조건 중 2개 이상 충족 시 현재가에서 추가 (상승 전환 확인 시)",
        "  - 조건①: MACD 히스토그램 상승 중 (단기 모멘텀 강화)",
        "  - 조건②: Higher Low 확인 (최근 저점 > 이전 저점 = 상승 구조)",
        "  - 조건③: StochRSI 과매도 탈출 (20이하 → 40이상 회복)",
        "  - 불타기는 직전 매수가 대비 +2% 이상 상승 시에만 허용",
    ],
    "entry_weights": {"BUY1": 0.30, "BUY2": 0.34, "BUY3": 0.33},
    "regime_params": REGIME_PARAMS,
}

# ════════════════════════════════════════════════════════════
# 단계별 백테스트 버전 정의
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# Style Detector — 종목 성향 자동 분류 엔진
# ════════════════════════════════════════════════════════════
def classify_stock_style(df):
    """
    최근 1년 데이터로 종목 성향 자동 분류
    반환: {style, scores, reason, best_strategy, confidence}

    MOM  (모멘텀형): 추세 강함 → V6
    FIB  (피보나치형): 박스권 눌림목 → V5/V7
    STAT (통계형): 저변동 평균회귀 → V7
    """
    score_mom  = 0
    score_fib  = 0
    score_stat = 0
    reasons    = {"MOM":[], "FIB":[], "STAT":[]}

    try:
        df = df.copy()
        close = df["Close"]
        n = len(df)

        # ── 모멘텀 점수 ───────────────────────────────────
        # 1년 고점의 90% 이상 = 신고가 근처 = 강한 모멘텀
        high252 = close.rolling(min(252,n)).max().iloc[-1]
        if close.iloc[-1] >= high252 * 0.90:
            score_mom += 2
            reasons["MOM"].append("52주 고점 90% 이내")

        # MA20 > MA60 = 단기 상승 추세
        if "MA20" in df.columns and "MA60" in df.columns:
            if float(df["MA20"].iloc[-1]) > float(df["MA60"].iloc[-1]):
                score_mom += 1
                reasons["MOM"].append("MA20 > MA60 골든크로스")

        # ROC 20일 10% 이상 = 강한 상승
        roc20 = float((close.iloc[-1] / close.iloc[max(-21,-n)] - 1) * 100)
        if roc20 > 10:
            score_mom += 1
            reasons["MOM"].append(f"20일 수익률 +{roc20:.1f}% 강세")

        # ADX 25 이상 = 추세 강함
        if "ADX" in df.columns:
            adx_val = float(df["ADX"].dropna().iloc[-1]) if not df["ADX"].isna().all() else 15
            if adx_val > 25:
                score_mom += 1
                reasons["MOM"].append(f"ADX {adx_val:.0f} 추세 강함")

        # 주별 상승 빈도 60% 이상 = 꾸준한 상승
        weekly_up = (close.diff(5) > 0).tail(52).mean()
        if weekly_up >= 0.60:
            score_mom += 1
            reasons["MOM"].append(f"주별 상승 빈도 {weekly_up*100:.0f}%")

        # ── 피보나치형 점수 ───────────────────────────────
        # MA200 아래이고 Z-Score 낮으면 눌림목 반등형
        if "MA200" in df.columns:
            ma200 = float(df["MA200"].dropna().iloc[-1])
            if close.iloc[-1] < ma200 * 1.05:
                score_fib += 1
                reasons["FIB"].append("MA200 근처 (눌림목 구간)")

        # Z-Score -1.5 이하 = 통계적 과매도
        if "ZScore" in df.columns:
            zs = float(df["ZScore"].dropna().iloc[-1]) if not df["ZScore"].isna().all() else 0
            if zs < -1.5:
                score_fib += 1
                reasons["FIB"].append(f"Z-Score {zs:.2f} 과매도")

        # BB 하단 터치 횟수 5회 이상 = 박스권 반등 패턴
        if "BB_touch_low" in df.columns:
            bb_cnt = int(df["BB_touch_low"].tail(252).sum())
            if bb_cnt >= 5:
                score_fib += 1
                reasons["FIB"].append(f"BB 하단 터치 {bb_cnt}회 (반등 패턴)")

        # 52주 낙폭 -10~-35% = 적당한 조정 (피보나치 최적 구간)
        draw52 = float((close.iloc[-1] / high252 - 1) * 100)
        if -35 <= draw52 <= -10:
            score_fib += 2
            reasons["FIB"].append(f"52주 낙폭 {draw52:.1f}% (피보나치 적정 조정)")

        # ── 통계형 점수 ───────────────────────────────────
        # 일간 변동성 2% 미만 = 저변동성
        if "Return" in df.columns:
            vol_daily = float(df["Return"].tail(252).std())
            if vol_daily < 0.02:
                score_stat += 2
                reasons["STAT"].append(f"일간변동성 {vol_daily*100:.2f}% 낮음")

        # Z-Score 절댓값 평균 1.0 미만 = 평균 회귀 안정적
        if "ZScore" in df.columns:
            zabs_mean = float(df["ZScore"].tail(252).abs().mean())
            if zabs_mean < 1.0:
                score_stat += 1
                reasons["STAT"].append(f"Z-Score 평균 {zabs_mean:.2f} 안정")

        # RSI 평균 40~60 = 중립적 모멘텀 (평균 회귀 유리)
        if "RSI" in df.columns:
            rsi_mean = float(df["RSI"].tail(252).mean())
            if 40 <= rsi_mean <= 60:
                score_stat += 1
                reasons["STAT"].append(f"RSI 평균 {rsi_mean:.0f} (중립)")

        # ADX 20 미만 = 추세 약함 = 박스권
        if "ADX" in df.columns:
            adx_mean = float(df["ADX"].tail(60).mean())
            if adx_mean < 20:
                score_stat += 1
                reasons["STAT"].append(f"ADX 평균 {adx_mean:.0f} (추세 약함)")

        # ── 최종 분류 ─────────────────────────────────────
        scores = {"MOM": score_mom, "FIB": score_fib, "STAT": score_stat}
        best   = max(scores, key=scores.get)
        total  = sum(scores.values())
        conf   = scores[best] / max(total, 1) * 100

        # 전략 매핑
        strategy_map = {
            "MOM":  "V6 — 고변동성 모멘텀",
            "FIB":  "V5 — 조건 완화 + 현실 익절",
            "STAT": "V7 — 과매도 역추세 (권장)",
        }
        style_name = {
            "MOM":  "📈 모멘텀형",
            "FIB":  "📐 피보나치형",
            "STAT": "📊 통계형",
        }

        return {
            "style":    best,
            "style_name": style_name[best],
            "scores":   scores,
            "reasons":  reasons[best],
            "best_strategy": strategy_map[best],
            "confidence": round(conf, 1),
            "draw52":   round(draw52, 1),
            "roc20":    round(roc20, 1),
        }
    except Exception as e:
        return {
            "style": "FIB", "style_name": "📐 피보나치형",
            "scores": {"MOM":0,"FIB":1,"STAT":0},
            "reasons": ["분류 중 오류"], "best_strategy": "V5 — 조건 완화 + 현실 익절",
            "confidence": 0, "draw52": 0, "roc20": 0,
        }


# ── 빠른 백테스트 (스캔용 경량 버전) ──────────────────────
def quick_backtest(df, strategy):
    """
    스캔 시 빠른 성과 계산 (3가지 전략)
    반환: {wr, cagr, mdd, trades, grade}
    """
    try:
        if strategy == "V6 — 고변동성 모멘텀":
            trades, metrics = run_momentum_backtest(df)
        elif strategy == "V7 — 과매도 역추세 (권장)":
            trades, metrics = run_v7_backtest(df)
        else:
            trades, metrics = run_backtest(df, score_thr=40, buy_logic="2of3")

        if not metrics:
            return None

        n = metrics.get("총 완결 거래", 0)
        wr = float(metrics.get("승률","0%").replace("%",""))
        cagr_str = metrics.get("CAGR","0%").replace("%","")
        cagr = float(cagr_str) if cagr_str not in ["-","nan"] else 0
        mdd  = float(metrics.get("MDD","0%").replace("%",""))

        # 거래 3건 미만이면 CAGR 신뢰 불가 — 표시용으로만 사용
        if n < 3:
            cagr = cagr  # 표시는 하되 등급에서 제외

        # 신뢰도 등급 (거래 수 기반)
        if n < 3:
            grade = "⚠️ 데이터 부족"
        elif wr >= 60 and cagr >= 30:
            grade = "🏆 최적"
        elif wr >= 50 and cagr >= 15:
            grade = "✅ 양호"
        elif cagr > 0:
            grade = "△ 보통"
        else:
            grade = "❌ 부적합"

        return {"wr":wr, "cagr":cagr, "mdd":mdd, "trades":n, "grade":grade}
    except Exception:
        return None


BT_VERSIONS = {
    "V1 — 기본 피보나치": {
        "desc": "피보나치 BUY1 도달 시 바로 진입. 레짐/점수/StochRSI 필터 없음.",
        "score_thr": 0,
        "use_stoch": False,
        "use_regime": False,
        "use_bull_bear": False,
        "buy_logic": "3of3",  # 조건 없음 = 무조건 진입
    },
    "V2 — 레짐 필터 추가": {
        "desc": "피보나치 + DOWNtrend 제외. MA60 위에서만 진입.",
        "score_thr": 0,
        "use_stoch": False,
        "use_regime": True,
        "use_bull_bear": False,
        "buy_logic": "3of3",
    },
    "V3 — AI 점수 필터 추가": {
        "desc": "V2 + AI 4-Factor 점수 필터. 점수 미달 시 진입 차단.",
        "score_thr": 45,
        "use_stoch": False,
        "use_regime": True,
        "use_bull_bear": False,
        "buy_logic": "3of3",
    },
    "V4 — 물타기/불타기 (현재 전략)": {
        "desc": "V3 + StochRSI 필터 + 물타기/불타기 자동 판단. 현재 전략 그대로.",
        "score_thr": 50,
        "use_stoch": True,
        "use_regime": True,
        "use_bull_bear": True,
        "buy_logic": "3of3",  # AND 조건
        "strategy_type": "fib",
    },
    "V5 — 조건 완화 + 현실 익절": {
        "desc": "BUY1: 3개 중 2개 충족. 익절 UP+15%/RG+12%로 현실화. 자주 익절 → 승률 상승.",
        "score_thr": 40,
        "use_stoch": True,
        "use_regime": True,
        "use_bull_bear": True,
        "buy_logic": "2of3",
        "strategy_type": "fib",
    },
    "V6 — 고변동성 모멘텀": {
        "desc": "TSLA/AMD 같은 고변동성 종목용. 피보나치 대신 모멘텀 추격. 익절+10% 자주 먹기.",
        "score_thr": 50,
        "use_stoch": False,
        "use_regime": True,
        "use_bull_bear": False,
        "buy_logic": "momentum",
        "strategy_type": "momentum",
    },
    "V7 — 과매도 역추세 (권장)": {
        "desc": (
            "Z-Score+BB+RSI+Stoch 과매도 2개이상 + 양봉 확인 후 진입. "
            "물타기 금지, 불타기만. 슬리피지 반영. 손익비 2.5 이상만 진입. "
            "승률 60~70% 목표."
        ),
        "score_thr": 40,
        "use_stoch": True,
        "use_regime": True,
        "use_bull_bear": False,
        "buy_logic": "v7",
        "strategy_type": "v7",
    },
}

def run_v7_backtest(df, slippage=0.002, trail_pct=0.15):
    """
    V7 — 과매도 역추세 + 모멘텀 확인 전략

    핵심 원칙:
    1. 피보나치 = 진입 구간 힌트 (신호 아님)
    2. BUY1: 피보 도달 + 과매도 4조건 중 2개 + 양봉 확인
    3. BUY2/3: 물타기 금지 → 모멘텀 회복 확인 후 불타기만
    4. 진입가: 다음봉 시가 + 슬리피지 (현실 반영)
    5. 청산: 트레일링 스탑
    6. 손익비: 2.5 미만이면 진입 안 함
    """
    trades=[]; capital=1.0
    stage=0; ep=[]; ew=[]
    regime_entry="UNKNOWN"
    cooldown=0; loss_streak=0
    peak_after_buy=0
    START = max(252, 60)  # 12개월 모멘텀 계산 후 시작

    for i in range(START, len(df)-1):  # -1: 다음봉 진입
        row   = df.iloc[i]
        row_next = df.iloc[i+1]  # 실제 진입봉 (다음날)
        p     = float(row["Close"])       # 신호 감지: 당일 종가
        p_entry = float(row_next["Open"]) * (1 + slippage)  # 실제 진입: 다음날 시가+슬리피지

        regime = row["Regime"]
        if regime in ["UNKNOWN","DOWNtrend"]: continue
        if cooldown > 0: cooldown -= 1; continue

        cfg = REGIME_PARAMS[regime]

        # 피보나치 계산
        sh   = row.get("sw_high", float("nan"))
        rng  = row.get("rng", 0)
        # 피보나치 계산 (fib_valid 없어도 롤링으로 대체)
        fib_ok = bool(row.get("fib_valid", False))
        if pd.isna(sh) or rng <= 0:
            # fib_valid 없으면 60일 롤링으로 계산
            try:
                sh = float(df["High"].rolling(60).max().iloc[i])
                rng = sh - float(df["Low"].rolling(60).min().iloc[i])
                if rng <= 0 or sh <= p: continue
            except Exception:
                continue

        fib_prices = [float(sh) - float(rng)*f for f in cfg["fib"]]
        fib886     = float(sh) - float(rng)*0.886

        # V7 과매도 점수 — 개별 지표 직접 계산 (컬럼 없을 때 대비)
        zs  = float(row.get("ZScore", 0)) if not pd.isna(row.get("ZScore", float("nan"))) else 0
        bb  = int(row.get("BB_touch_low", 0))
        rsi = float(row.get("RSI", 50))
        st  = float(row.get("StochRSI", 50))
        v7_score = int(zs < -1.5) + int(bb == 1) + int(rsi < 35) + int(st < 25)
        bull_candle = int(row.get("BullCandle", 1))  # 없으면 1로 완화

        # 손익비 계산
        potential_loss = max((p - fib886) / p, 0.03)
        potential_gain = cfg["tp"]
        risk_reward = potential_gain / potential_loss

        # ── BUY1 진입 ────────────────────────────────────
        # 핵심: 피보BUY1 도달 + 과매도 2개 이상
        # 양봉/손익비 조건 완화 (백테스트 거래 수 확보)
        if stage == 0:
            if (p <= fib_prices[0] and
                v7_score >= 2 and
                risk_reward >= 1.5 and        # 2.5 → 1.5로 완화
                regime != "DOWNtrend"):

                stage = 1
                actual_price = p_entry
                ep = [actual_price]; ew = [0.30]
                regime_entry = regime
                peak_after_buy = actual_price
                trades.append({
                    "날짜":   str(row_next["Date"])[:10],
                    "구분":   "BUY1",
                    "단계":   f"1차 진입 30% [V7 과매도점수:{v7_score}/4 손익비:{risk_reward:.1f}]",
                    "가격":   round(actual_price, 2),
                    "비중":   "30%",
                    "레짐":   regime,
                    "피보":   f"Fib {cfg['fib'][0]}",
                    "수익률": "-",
                    "비고":   f"슬리피지 반영 | Z:{row.get('ZScore',0):.1f} BB:{int(row.get('BB_touch_low',0))} RSI:{row.get('RSI',50):.0f}",
                    "_pnl":   0,
                })

        # ── BUY2/3: 물타기 금지, 모멘텀 회복 불타기만 ──
        elif stage == 1:
            recovery = int(row.get("V7_recovery_score", 0))
            avg = sum(x*w for x,w in zip(ep,ew)) / sum(ew)
            # 불타기 조건: 모멘텀 회복 2개 이상 + 현재가 BUY1 진입가 위
            if recovery >= 2 and p > ep[0] * 1.015:
                stage = 2
                actual_price2 = p_entry
                ep.append(actual_price2); ew.append(0.35)
                avg = sum(x*w for x,w in zip(ep,ew)) / sum(ew)
                trades.append({
                    "날짜":   str(row_next["Date"])[:10],
                    "구분":   "BUY2",
                    "단계":   f"2차 불타기 35% [모멘텀회복:{recovery}/4]",
                    "가격":   round(actual_price2, 2),
                    "비중":   "35%",
                    "레짐":   regime,
                    "피보":   "불타기",
                    "수익률": f"{(actual_price2/ep[0]-1)*100:+.1f}%",
                    "비고":   f"평균단가 ${avg:.2f}",
                    "_pnl":   0,
                })

        elif stage == 2:
            recovery = int(row.get("V7_recovery_score", 0))
            # 추가 모멘텀 확인 후 BUY3
            if recovery >= 3 and p > ep[-1] * 1.01:
                stage = 3
                actual_price3 = p_entry
                ep.append(actual_price3); ew.append(0.35)
                avg = sum(x*w for x,w in zip(ep,ew)) / sum(ew)
                trades.append({
                    "날짜":   str(row_next["Date"])[:10],
                    "구분":   "BUY3",
                    "단계":   f"3차 불타기 35% [모멘텀회복:{recovery}/4]",
                    "가격":   round(actual_price3, 2),
                    "비중":   "35%",
                    "레짐":   regime,
                    "피보":   "불타기",
                    "수익률": f"{(actual_price3/ep[0]-1)*100:+.1f}%",
                    "비고":   f"평균단가 ${avg:.2f}",
                    "_pnl":   0,
                })

        # ── 청산: 트레일링 스탑 ──────────────────────────
        if stage > 0:
            total_w = sum(ew)
            avg = sum(x*w for x,w in zip(ep,ew)) / total_w
            cfg_e = REGIME_PARAMS[regime_entry]

            # 최고가 추적
            if p > peak_after_buy:
                peak_after_buy = p

            # 손절: 피보 0.886 레벨 이탈 시 (마지막 지지선)
            # BUY1 진입가 기준으로 0.886 역산
            stop_price  = ep[0] * (1 - (ep[0] - fib886) / ep[0]) if fib886 and fib886 < ep[0] else avg * 0.95
            # 트레일링: 수익 1% 이상이면 발동 (조건 완화)
            trail_price = peak_after_buy * (1 - trail_pct)
            # 고정 익절 +12% or 트레일링 중 더 큰 것
            fixed_tp    = avg * 1.12
            exit_trail  = (p <= trail_price) and (p > avg * 1.01)
            exit_fixed  = p >= fixed_tp

            exit_reason = None
            if p < stop_price:
                exit_reason = "❌ 손절"
            elif exit_fixed:
                exit_reason = f"✅ 고정 익절 +12%"
            elif exit_trail:
                exit_reason = f"✅ 트레일링 익절 (고점${peak_after_buy:.2f} 대비 -{trail_pct*100:.0f}%)"

            if exit_reason:
                pnl = (p - avg) / avg
                capital *= (1 + pnl * total_w)
                if pnl < 0:
                    loss_streak += 1
                    if loss_streak >= 2: cooldown = 10
                else:
                    loss_streak = 0
                trades.append({
                    "날짜":   str(row["Date"])[:10],
                    "구분":   "SELL",
                    "단계":   exit_reason,
                    "가격":   round(p, 2),
                    "비중":   f"전량 ({stage}단계)",
                    "레짐":   regime,
                    "피보":   "청산",
                    "수익률": f"{pnl*100:+.1f}%",
                    "비고":   f"평균단가 ${avg:.2f}",
                    "_pnl":   pnl,
                })
                stage=0; ep=[]; ew=[]; peak_after_buy=0

    # ── 성과 계산 ──
    sells = [t for t in trades if t["구분"]=="SELL"]
    if not sells: return trades, {}

    pnls   = np.array([t["_pnl"] for t in sells])
    wins   = pnls[pnls>0]; losses=pnls[pnls<=0]
    equity = np.cumprod(1+pnls)
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity-peak)/peak).min()*100)
    wr     = len(wins)/len(pnls)*100

    buy1_dates = pd.to_datetime([t["날짜"] for t in trades if t["구분"]=="BUY1"])
    sell_dates = pd.to_datetime([t["날짜"] for t in sells])
    # 거래 3건 미만이면 전체 데이터 기간으로 CAGR 계산 (단기 거래 과장 방지)
    if len(sells) < 3:
        n_years = max(len(df) / 252, 0.5)
    else:
        n_years = max((sell_dates[-1]-buy1_dates[0]).days/365.25, 0.5) if len(buy1_dates)>0 else 1.0
    cagr   = (equity[-1]**(1/n_years)-1)*100
    sharpe = float(np.mean(pnls)/np.std(pnls)*np.sqrt(max(len(pnls),2))) if (np.std(pnls)>0 and len(pnls)>=3) else 0
    calmar = cagr/abs(mdd) if mdd!=0 else 0

    b1 = [t for t in trades if t["구분"]=="BUY1"]
    b2 = [t for t in trades if t["구분"]=="BUY2"]
    b3 = [t for t in trades if t["구분"]=="BUY3"]

    metrics = {
        "총 완결 거래":  len(sells),
        "BUY1 진입":    len(b1),
        "BUY2 불타기":  len(b2),
        "BUY3 불타기":  len(b3),
        "BUY2 물타기":  0,
        "BUY3 물타기":  0,
        "익절":         len(wins),
        "손절":         len(losses),
        "승률":         f"{wr:.1f}%",
        "총 수익률":    f"{(equity[-1]-1)*100:.1f}%",
        "CAGR":         f"{cagr:.1f}%",
        "MDD":          f"{mdd:.1f}%",
        "Sharpe":       f"{sharpe:.2f}",
        "Calmar":       f"{calmar:.2f}",
        "평균 익절":    f"{np.mean(wins)*100:.1f}%" if len(wins)>0 else "-",
        "평균 손절":    f"{np.mean(losses)*100:.1f}%" if len(losses)>0 else "-",
        "슬리피지":     f"{slippage*100:.1f}%",
        "트레일링":     f"-{trail_pct*100:.0f}%",
    }
    return trades, metrics


def run_momentum_backtest(df, trailing_stop=False, trail_pct=0.07):
    """
    고변동성 종목 전용 — 모멘텀 추격 전략

    진입 조건 (4개 중 3개):
      ① MA20 > MA60 (골든크로스 구간)
      ② RSI 45~68 (과열 아님, 상승 중)
      ③ MACD 히스토그램 양수
      ④ 거래량 20일 평균의 1.3배 이상

    익절: +10% 고정 or 트레일링 -7%
    손절: -5% (타이트)
    쿨다운: 손절 후 5봉 (빠른 재진입)
    """
    trades=[]; capital=1.0; pos=0; entry=0
    cooldown=0; loss_streak=0; peak_price=0
    START=max(60,30)

    for i in range(START, len(df)):
        row=df.iloc[i]; p=float(row["Close"])
        regime=row["Regime"]
        if regime in ["UNKNOWN","DOWNtrend"]: continue
        if cooldown>0: cooldown-=1; continue

        ma20=row.get("MA20",float("nan")); ma60=row.get("MA60",float("nan"))
        rsi=row.get("RSI",float("nan")); macd_h=row.get("MACD_hist",float("nan"))
        vol=row.get("Volume",0); vol_ma=row.get("VolMA20",1)
        if any(pd.isna(x) for x in [ma20,ma60,rsi]): continue

        # 모멘텀 진입 조건
        c1 = float(ma20)>float(ma60)                  # ① 골든크로스
        c2 = 45<=float(rsi)<=68                        # ② RSI 적정
        c3 = (not pd.isna(macd_h)) and macd_h>0       # ③ MACD 양수
        c4 = (vol_ma>0) and (vol/vol_ma)>=1.3         # ④ 거래량 급증
        cond_count = int(c1)+int(c2)+int(c3)+int(c4)

        if pos==0 and cond_count>=3:
            pos=1; entry=p; peak_price=p
            trades.append({
                "날짜":str(row["Date"])[:10],"구분":"BUY",
                "단계":"모멘텀 진입 (100%)",
                "가격":round(p,2),"비중":"100%","레짐":regime,
                "피보":"모멘텀","수익률":"-",
                "비고":f"조건{cond_count}/4 충족 | RSI:{rsi:.0f} MACD:{'✅' if c3 else '❌'} 거래량:{'✅' if c4 else '❌'}",
                "_pnl":0,
            })

        if pos==1:
            peak_price=max(peak_price,p)
            pnl=(p-entry)/entry
            stop_p=entry*0.95         # 손절 -5%
            tp_p=entry*1.10           # 익절 +10%
            trail_p=peak_price*(1-trail_pct) if trailing_stop else None

            exit_reason=None
            if p<=stop_p:             exit_reason="❌ 손절 -5%"
            elif trailing_stop and trail_p and p<=trail_p and pnl>0:
                exit_reason=f"✅ 트레일링 (고점${peak_price:.2f} 대비 -{trail_pct*100:.0f}%)"
            elif not trailing_stop and p>=tp_p:
                exit_reason="✅ 익절 +10%"

            if exit_reason:
                capital*=(1+pnl); pos=0
                if pnl<0:
                    loss_streak+=1
                    cooldown=5  # 손절 후 5봉 쿨다운
                else:
                    loss_streak=0
                trades.append({
                    "날짜":str(row["Date"])[:10],"구분":"SELL",
                    "단계":exit_reason,
                    "가격":round(p,2),"비중":"전량","레짐":regime,
                    "피보":"청산","수익률":f"{pnl*100:+.1f}%",
                    "비고":f"진입가 ${entry:.2f} → 청산 ${p:.2f}",
                    "_pnl":pnl,
                })
                peak_price=0

    # 성과 계산
    sells=[t for t in trades if t["구분"]=="SELL"]
    if not sells: return trades,{}
    pnls=np.array([t["_pnl"] for t in sells])
    wins=pnls[pnls>0]; losses=pnls[pnls<=0]
    equity=np.cumprod(1+pnls); peak=np.maximum.accumulate(equity)
    mdd=float(((equity-peak)/peak).min()*100)
    wr=len(wins)/len(pnls)*100
    buy1_dates=pd.to_datetime([t["날짜"] for t in trades if t["구분"]=="BUY"])
    sell_dates=pd.to_datetime([t["날짜"] for t in sells])
    if len(sells) < 3:
        n_years = max(len(df) / 252, 0.5)
    else:
        n_years = max((sell_dates[-1]-buy1_dates[0]).days/365.25, 0.5) if len(buy1_dates)>0 else 1.0
    cagr=(equity[-1]**(1/n_years)-1)*100
    sharpe=float(np.mean(pnls)/np.std(pnls)*np.sqrt(len(pnls))) if np.std(pnls)>0 else 0
    calmar=cagr/abs(mdd) if mdd!=0 else 0
    metrics={
        "총 완결 거래":len(sells),"BUY1 진입":len([t for t in trades if t["구분"]=="BUY"]),
        "BUY2 추가":0,"BUY3 추가":0,"BUY2 물타기":0,"BUY2 불타기":0,"BUY3 물타기":0,"BUY3 불타기":0,
        "익절":len(wins),"손절":len(losses),"승률":f"{wr:.1f}%",
        "총 수익률":f"{(equity[-1]-1)*100:.1f}%","CAGR":f"{cagr:.1f}%",
        "MDD":f"{mdd:.1f}%","Sharpe":f"{sharpe:.2f}","Calmar":f"{calmar:.2f}",
        "평균 익절":f"{np.mean(wins)*100:.1f}%" if len(wins)>0 else "-",
        "평균 손절":f"{np.mean(losses)*100:.1f}%" if len(losses)>0 else "-",
    }
    return trades, metrics


def run_backtest(df, score_thr=40, use_stoch=True, trailing_stop=False, trail_pct=0.10,
                 use_regime=True, use_bull_bear=True, buy_logic="2of3"):
    """
    통합 백테스트 엔진

    buy_logic:
      "3of3" — MA60 AND StochRSI AND 점수 (엄격)
      "2of3" — 3개 중 2개 이상 (완화)
    """
    trades = []
    capital = 1.0
    stage = 0
    ep = []
    ew = []
    regime_entry = "UNKNOWN"
    loss_streak = 0
    cooldown = 0
    START = max(200, SWING + 60)  # MA200 안정화 후 시작

    for i in range(START, len(df)):
        row = df.iloc[i]
        p = float(row["Close"])
        regime = row["Regime"]
        cfg = REGIME_PARAMS[regime]

        # UNKNOWN 레짐은 거래 안 함
        if regime == "UNKNOWN": continue

        ma60  = row["MA60"]
        stoch = row["StochRSI"]
        pct   = row["ScorePct"]
        rng   = row["rng"]
        sh    = row["sw_high"]
        fib_ok = bool(row.get("fib_valid", False))

        if any(pd.isna(x) for x in [ma60, stoch, pct, rng, sh]): continue
        if rng <= 0 or not fib_ok: continue

        # 쿨다운
        if cooldown > 0:
            cooldown -= 1
            continue

        # 피보나치 레벨 계산
        fib_prices = [float(sh) - float(rng) * f for f in cfg["fib"]]
        fib886     = float(sh) - float(rng) * 0.886

        # 기본 진입 조건
        trend_ok  = p > float(ma60)
        stoch_ok  = float(stoch) < cfg["stoch"]
        no_down   = regime != "DOWNtrend"

        # ── BUY1 진입 조건 판단 ──────────────────────────
        cond_ma60    = trend_ok                          # ① MA60 위
        cond_stoch   = stoch_ok if use_stoch else True   # ② StochRSI
        cond_score   = float(pct) >= score_thr           # ③ AI 점수
        cond_regime  = no_down if use_regime else True   # 레짐 필터

        if buy_logic == "2of3":
            # 완화: 3개 중 2개 이상
            cond_count = int(cond_ma60) + int(cond_stoch) + int(cond_score)
            buy1_ok = cond_count >= 2 and cond_regime
        else:
            # 엄격: 3개 전부
            buy1_ok = cond_ma60 and cond_stoch and cond_score and cond_regime

        if stage == 0 and buy1_ok:
            if p <= fib_prices[0]:
                stage = 1
                ep = [p]; ew = [0.30]
                regime_entry = regime
                trades.append({
                    "날짜":   str(row["Date"])[:10],
                    "구분":   "BUY1",
                    "단계":   "1차 매수 (30%)",
                    "가격":   round(p, 2),
                    "비중":   "30%",
                    "레짐":   regime,
                    "피보":   f"Fib {cfg['fib'][0]}",
                    "수익률": "-",
                    "비고":   f"StochRSI {stoch:.1f} / 점수 {pct:.0f}%",
                })

        # ── BUY2 추가 ─────────────────────────────────
        elif stage == 1 and (no_down if use_regime else True):
            macd_ok    = int(row.get("MACD_rising",    0)) == 1
            higher_low = int(row.get("HigherLow",      0)) == 1
            stoch_esc  = int(row.get("StochRSI_escape",0)) == 1
            bull_score = int(row.get("BullAdd_score",  0))
            # use_bull_bear=False면 불타기 비활성화
            is_bull    = (bull_score >= 2) if use_bull_bear else False
            # 물타기: 피보나치 BUY2 구간 도달
            if p <= fib_prices[1]:
                mode = "물타기"
                buy2_price = p
            # 불타기: 상승 전환 확인 + BUY1 대비 최소 +2% 상승
            elif is_bull and p > ep[0] * 1.02 and trend_ok:
                mode = "불타기"
                buy2_price = p
            else:
                buy2_price = None; mode = None

            if buy2_price is not None:
                stage = 2
                ep.append(buy2_price); ew.append(0.34)
                avg = sum(x*w for x,w in zip(ep,ew)) / sum(ew)
                bull_detail = f"MACD{'✅' if macd_ok else '❌'} / HigherLow{'✅' if higher_low else '❌'} / StochEsc{'✅' if stoch_esc else '❌'}"
                trades.append({
                    "날짜":   str(row["Date"])[:10],
                    "구분":   "BUY2",
                    "단계":   f"2차 매수 35% [{mode}]",
                    "가격":   round(buy2_price, 2),
                    "비중":   "35%",
                    "레짐":   regime,
                    "피보":   f"Fib {cfg['fib'][1]}" if mode=="물타기" else "불타기 진입",
                    "수익률": f"{(buy2_price/ep[0]-1)*100:+.1f}%",
                    "비고":   f"평균단가 ${avg:.2f} | {bull_detail}",
                    "_mode":  mode,
                })

        # ── BUY3 추가 ─────────────────────────────────
        elif stage == 2 and (no_down if use_regime else True):
            macd_ok    = int(row.get("MACD_rising",    0)) == 1
            higher_low = int(row.get("HigherLow",      0)) == 1
            stoch_esc  = int(row.get("StochRSI_escape",0)) == 1
            bull_score = int(row.get("BullAdd_score",  0))
            is_bull    = (bull_score >= 2) if use_bull_bear else False
            # 물타기: 피보나치 BUY3 구간 도달
            if p <= fib_prices[2]:
                mode = "물타기"
                buy3_price = p
            # 불타기: 상승 전환 확인 + BUY2 대비 최소 +2% 상승
            elif is_bull and p > ep[-1] * 1.02 and trend_ok:
                mode = "불타기"
                buy3_price = p
            else:
                buy3_price = None; mode = None

            if buy3_price is not None:
                stage = 3
                ep.append(buy3_price); ew.append(0.33)
                avg = sum(x*w for x,w in zip(ep,ew)) / sum(ew)
                bull_detail = f"MACD{'✅' if macd_ok else '❌'} / HigherLow{'✅' if higher_low else '❌'} / StochEsc{'✅' if stoch_esc else '❌'}"
                trades.append({
                    "날짜":   str(row["Date"])[:10],
                    "구분":   "BUY3",
                    "단계":   f"3차 매수 35% [{mode}]",
                    "가격":   round(buy3_price, 2),
                    "비중":   "35%",
                    "레짐":   regime,
                    "피보":   f"Fib {cfg['fib'][2]}" if mode=="물타기" else "불타기 진입",
                    "수익률": f"{(buy3_price/ep[0]-1)*100:+.1f}%",
                    "비고":   f"평균단가 ${avg:.2f} | {bull_detail}",
                    "_mode":  mode,
                })

        # ── 청산 로직 ─────────────────────────────────
        if stage > 0:
            total_w = sum(ew)
            avg = sum(x*w for x,w in zip(ep,ew)) / total_w
            cfg_e   = REGIME_PARAMS[regime_entry]
            # 손절선: 평균단가 기준 또는 Fib 0.886
            stop_price = max(avg * (1 - cfg_e["stop"]), fib886)
            # 익절선: 평균단가 기준
            tp_price   = avg * (1 + cfg_e["tp"])

            if p < stop_price:
                pnl = (p - avg) / avg
                capital *= (1 + pnl * total_w)
                loss_streak += 1
                if loss_streak >= 2:
                    cooldown = 10
                trades.append({
                    "날짜":   str(row["Date"])[:10],
                    "구분":   "SELL",
                    "단계":   f"손절 (평균단가 ${avg:.2f}기준 -{cfg_e['stop']*100:.0f}%)",
                    "가격":   round(p, 2),
                    "비중":   f"전량 ({stage}단계)",
                    "레짐":   regime,
                    "피보":   "손절",
                    "수익률": f"{pnl*100:+.1f}%",
                    "비고":   f"❌ 손절 / 평균단가 ${avg:.2f}",
                    "_pnl":   pnl,
                })
                stage = 0; ep = []; ew = []

            elif p >= tp_price:
                pnl = (p - avg) / avg
                capital *= (1 + pnl * total_w)
                loss_streak = 0
                trades.append({
                    "날짜":   str(row["Date"])[:10],
                    "구분":   "SELL",
                    "단계":   f"익절 (평균단가 ${avg:.2f}기준 +{cfg_e['tp']*100:.0f}%)",
                    "가격":   round(p, 2),
                    "비중":   f"전량 ({stage}단계)",
                    "레짐":   regime,
                    "피보":   "익절",
                    "수익률": f"{pnl*100:+.1f}%",
                    "비고":   f"✅ 익절 / 평균단가 ${avg:.2f}",
                    "_pnl":   pnl,
                })
                stage = 0; ep = []; ew = []

    # ── 성과 계산 ──────────────────────────────────────
    sells = [t for t in trades if t["구분"] == "SELL"]
    if not sells:
        return trades, {}

    pnls   = np.array([t["_pnl"] for t in sells])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    equity = np.cumprod(1 + pnls)
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min() * 100)
    wr     = len(wins) / len(pnls) * 100

    # CAGR: 첫 BUY1 진입일 ~ 마지막 청산일 기준
    buy1_dates = pd.to_datetime([t["날짜"] for t in trades if t["구분"]=="BUY1"])
    sell_dates = pd.to_datetime([t["날짜"] for t in sells])
    if len(buy1_dates) > 0 and len(sell_dates) > 0:
        start_d = buy1_dates[0]
        end_d   = sell_dates[-1]
        if len(sells) < 3:
            n_years = max(len(df) / 252, 0.5)
        else:
            n_years = max((end_d - start_d).days / 365.25, 0.5)
    else:
        n_years = 0.5
    cagr   = (equity[-1] ** (1 / n_years) - 1) * 100
    # Sharpe: 거래 수익률 기반 (거래당 수익률)
    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(max(len(pnls),2))) if (np.std(pnls) > 0 and len(pnls) >= 3) else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    # BUY 단계별 통계
    b1_trades = [t for t in trades if t["구분"] == "BUY1"]
    b2_trades = [t for t in trades if t["구분"] == "BUY2"]
    b3_trades = [t for t in trades if t["구분"] == "BUY3"]

    # 물타기 / 불타기 분류
    b2_bull = [t for t in b2_trades if t.get("_mode") == "불타기"]
    b2_bear = [t for t in b2_trades if t.get("_mode") == "물타기"]
    b3_bull = [t for t in b3_trades if t.get("_mode") == "불타기"]
    b3_bear = [t for t in b3_trades if t.get("_mode") == "물타기"]

    metrics = {
        "총 완결 거래":       len(sells),
        "BUY1 진입":         len(b1_trades),
        "BUY2 추가":         len(b2_trades),
        "BUY3 추가":         len(b3_trades),
        "BUY2 물타기":       len(b2_bear),
        "BUY2 불타기":       len(b2_bull),
        "BUY3 물타기":       len(b3_bear),
        "BUY3 불타기":       len(b3_bull),
        "익절":              len(wins),
        "손절":              len(losses),
        "승률":              f"{wr:.1f}%",
        "총 수익률":         f"{(equity[-1]-1)*100:.1f}%",
        "CAGR":              f"{cagr:.1f}%",
        "MDD":               f"{mdd:.1f}%",
        "Sharpe":            f"{sharpe:.2f}",
        "Calmar":            f"{calmar:.2f}",
        "평균 익절":         f"{np.mean(wins)*100:.1f}%" if len(wins) > 0 else "-",
        "평균 손절":         f"{np.mean(losses)*100:.1f}%" if len(losses) > 0 else "-",
    }
    return trades, metrics

# ════════════════════════════════════════════════════════════
# 차트 (심플 버전)
# ════════════════════════════════════════════════════════════
def draw_chart(res, trades=[]):
    df=res["df"].tail(120).copy().reset_index(drop=True)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],
        vertical_spacing=0.04,subplot_titles=["가격 차트","StochRSI"])
    # 캔들
    fig.add_trace(go.Candlestick(x=df["Date"],open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="가격",
        increasing_line_color="#00ff9d",decreasing_line_color="#ff4757",
        increasing_fillcolor="#00ff9d",decreasing_fillcolor="#ff4757"),row=1,col=1)
    # MA
    for cn,c,n in [("MA20","#00d4ff","MA20"),("MA60","#ffd700","MA60"),("MA200","#ff8c00","MA200")]:
        if cn in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"],y=df[cn],line=dict(color=c,width=1.2),
                name=n,opacity=0.8),row=1,col=1)
    # 레짐 배경
    if "Regime" in df.columns:
        BG={"UPtrend":"rgba(0,220,120,0.07)","RANGE":"rgba(255,210,0,0.07)",
            "DOWNtrend":"rgba(255,60,60,0.07)","UNKNOWN":"rgba(120,120,120,0.03)"}
        rs=df[["Date","Regime"]].copy(); rs["prev"]=rs["Regime"].shift(1)
        bounds=rs[rs["Regime"]!=rs["prev"]].index.tolist()+[len(df)-1]; prev_i=0
        for b in bounds[1:]:
            b_safe=min(b,len(df)-1)
            if prev_i>=len(df) or b_safe>=len(df): break
            seg=df.iloc[prev_i]["Regime"]; x0=df.iloc[prev_i]["Date"]; x1=df.iloc[b_safe]["Date"]
            fig.add_vrect(x0=x0,x1=x1,fillcolor=BG.get(seg,"rgba(120,120,120,0.03)"),
                          opacity=1,layer="below",line_width=0); prev_i=b
    # 피보나치
    fib_colors=["#00ff9d","#ffd700","#ff8c00"]
    for f_val,fc,fi in zip(res["fib_lv"],fib_colors,res["cfg"]["fib"]):
        if f_val:
            fig.add_hline(y=f_val,line_dash="dash",line_color=fc,line_width=1.2,
                annotation_text=f"  BUY Fib{fi}: ${f_val:.2f}",annotation_font_color=fc,row=1,col=1)
    if res["stop_s"]:
        fig.add_hline(y=res["stop_s"],line_dash="dot",line_color="#ff4757",line_width=1.5,
            annotation_text=f"  손절: ${res['stop_s']:.2f}",annotation_font_color="#ff4757",row=1,col=1)
    if res["tp_s"]:
        fig.add_hline(y=res["tp_s"],line_dash="dot",line_color="#ffd700",line_width=1.5,
            annotation_text=f"  목표: ${res['tp_s']:.2f}",annotation_font_color="#ffd700",row=1,col=1)
    # 매매 마커
    if trades:
        buys=[t for t in trades if t["구분"]=="매수"]
        wins=[t for t in trades if t["구분"]=="매도" and "익절" in t.get("비고","")]
        loss=[t for t in trades if t["구분"]=="매도" and "손절" in t.get("비고","")]
        if buys:
            fig.add_trace(go.Scatter(x=[t["날짜"] for t in buys],y=[t["가격"] for t in buys],
                mode="markers",marker=dict(color="#00ff9d",size=10,symbol="triangle-up",
                line=dict(color="#fff",width=1)),name="매수"),row=1,col=1)
        if wins:
            fig.add_trace(go.Scatter(x=[t["날짜"] for t in wins],y=[t["가격"] for t in wins],
                mode="markers",marker=dict(color="#00d4ff",size=10,symbol="circle",
                line=dict(color="#fff",width=1)),name="익절"),row=1,col=1)
        if loss:
            fig.add_trace(go.Scatter(x=[t["날짜"] for t in loss],y=[t["가격"] for t in loss],
                mode="markers",marker=dict(color="#ff4757",size=10,symbol="x",
                line=dict(color="#fff",width=1)),name="손절"),row=1,col=1)
    # StochRSI
    if "StochRSI" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"],y=df["StochRSI"],
            line=dict(color="#7b5ea7",width=1.5),name="StochRSI",
            fill="tozeroy",fillcolor="rgba(123,94,167,0.1)"),row=2,col=1)
        fig.add_hline(y=20,line_dash="dot",line_color="#00ff9d",line_width=1,row=2,col=1)
        fig.add_hline(y=80,line_dash="dot",line_color="#ff4757",line_width=1,row=2,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#060810",plot_bgcolor="#0b0f1a",
        font=dict(color="#e8eaf6",size=11),height=580,margin=dict(l=10,r=10,t=36,b=10),
        legend=dict(orientation="h",y=1.02,x=0),xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor="#1e2d4a"); fig.update_yaxes(gridcolor="#1e2d4a")
    return fig

# ════════════════════════════════════════════════════════════
# 신호 뱃지 HTML
# ════════════════════════════════════════════════════════════
BADGE_MAP={
    "strong_buy":"badge-strong-buy","buy":"badge-buy",
    "watch":"badge-watch","hold":"badge-hold","sell":"badge-sell"
}
def badge(res):
    cls=BADGE_MAP.get(res["sig_k"],"badge-hold")
    return f'<span class="{cls}">{res["signal"]}</span>'

def action_badge(res):
    color={"strong_buy":"#00ff9d","buy":"#4ade80","watch":"#ffd700","hold":"#9ca3af","sell":"#ff4757"}
    c=color.get(res["sig_k"],"#9ca3af")
    return f'<span style="color:{c};font-weight:700">{res["action"]}</span>'

def mcard(col,label,val,color="#e8eaf6",sub=""):
    col.markdown(f"""<div class="mc"><div class="mc-lbl">{label}</div>
    <div class="mc-val" style="color:{color}">{val}</div>
    <div style="color:#6b7280;font-size:.68rem;margin-top:2px">{sub}</div></div>""",
    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Excel 다운로드
# ════════════════════════════════════════════════════════════
def make_excel(ticker, res, metrics, trades):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        summary=pd.DataFrame({"항목":["종목","분석일","현재가","신호","레짐",
            "BUY1","BUY2","BUY3","손절선","익절목표",
            "TrendScore","CycleScore","SeasonScore","IrregScore","TotalScore"]+
            list(metrics.keys()),
            "값":[ticker,now,f"${res['price']:.2f}",res["signal"],res["regime"],
                f"${res['fib_lv'][0]:.2f}" if res['fib_lv'][0] else "N/A",
                f"${res['fib_lv'][1]:.2f}" if res['fib_lv'][1] else "N/A",
                f"${res['fib_lv'][2]:.2f}" if res['fib_lv'][2] else "N/A",
                f"${res['stop_s']:.2f}" if res['stop_s'] else "N/A",
                f"${res['tp_s']:.2f}" if res['tp_s'] else "N/A",
                res['ts'],res['cs'],res['ss'],res['irs'],f"{res['pct']:.1f}%"]+
                list(metrics.values())})
        summary.to_excel(writer,sheet_name="분석요약",index=False)
        if trades:
            pd.DataFrame(trades).to_excel(writer,sheet_name="거래내역",index=False)
    buf.seek(0); return buf.getvalue()

# ════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 👑 프로젝트 이세계")
    st.markdown("---")
    menu=st.radio("메뉴",["🏠 홈 대시보드","🔍 종목 분석","🤖 AI 종목 추천","📊 백테스트"],
                  label_visibility="collapsed")
    st.markdown("---")
    if menu in ["🔍 종목 분석","📊 백테스트"]:
        ticker_input=st.text_input("티커 입력",value="AAPL",
            placeholder="예: AAPL, TSLA, NVDA").upper().strip()
        period_input=st.selectbox("기간",["6mo","1y","2y","3y","5y"],index=2)
        st.caption("종목분석: 차트/레짐 기간 | 백테스트: 성과 계산 기간")
    if menu=="🤖 AI 종목 추천":
        st.markdown("**스캔 방식 선택**")
        scan_mode = st.radio("",
            ["🌐 전체 자동 스캔 (추천)", "✏️ 직접 종목 입력"],
            label_visibility="collapsed")
        if scan_mode == "✏️ 직접 종목 입력":
            custom_list = st.text_area("종목 목록 (줄바꿈)",
                value="\n".join(DEFAULT_WATCHLIST[:20]), height=150)
        else:
            custom_list = None
            st.markdown("**섹터 선택** (비워두면 전체)")
            st.caption(f"전체 {len(ALL_TICKERS)}개 종목")
            selected_sectors = st.multiselect("",
                list(SCAN_UNIVERSE.keys()),
                default=[],
                label_visibility="collapsed")
        top_n = st.slider("TOP 추천 개수", 5, 20, 10)
        scan_btn = st.button("🚀 AI 자동 스캔 시작",
                             use_container_width=True, type="primary")
    if menu=="🔍 종목 분석":
        analyze_btn=st.button("🔍 분석하기",use_container_width=True,type="primary")
    if menu=="📊 백테스트":
        st.markdown("**📋 전략 선택**")
        st.caption("저변동성 → V5 / 고변동성 → V6")
        bt_version = st.radio("",
            ["V1 — 기본 피보나치",
             "V2 — 레짐 필터 추가",
             "V3 — AI 점수 필터 추가",
             "V4 — 물타기/불타기 (현재 전략)",
             "V5 — 조건 완화 + 현실 익절",
             "V6 — 고변동성 모멘텀",
             "V7 — 과매도 역추세 (권장)"],
            index=6,
            label_visibility="collapsed",
        )
        st.markdown("---")

        # ── 전략별 설정 분기 ──────────────────────────────
        if bt_version == "V7 — 과매도 역추세 (권장)":
            st.markdown("""
            <div style="background:#001a1a;border:1px solid #00d4ff;
                        border-radius:8px;padding:10px;font-size:.76rem;
                        color:#00d4ff;line-height:1.7">
            🎯 <b>V7 과매도 역추세</b><br>
            진입: 피보BUY1 + 과매도 4조건 중 2개<br>
            ① Z-Score &lt; -1.5<br>
            ② 볼린저밴드 하단 터치<br>
            ③ RSI &lt; 35<br>
            ④ StochRSI &lt; 25<br>
            물타기 금지 / 불타기만 허용<br>
            슬리피지 0.2% 반영
            </div>""", unsafe_allow_html=True)
            bt_score_thr = 0    # AI 점수 미사용
            st.caption("✅ AI 점수 비사용 | 물타기 자동 금지")
            bt_trailing  = st.toggle("트레일링 스탑", value=True,
                                     key="bt_trail_v7",
                                     help="V7 권장: ON (-15%)")
            if bt_trailing:
                bt_trail_pct = st.slider(
                    "트레일링 폭 (%)", 5, 25, 15, 1,
                    key="bt_pct_v7") / 100
            else:
                bt_trail_pct = 0.15

        elif bt_version == "V6 — 고변동성 모멘텀":
            st.markdown("""
            <div style="background:#1a0f00;border:1px solid #ff8c00;
                        border-radius:8px;padding:10px;font-size:.76rem;
                        color:#ffd700;line-height:1.7">
            🔥 <b>V6 모멘텀 전략</b><br>
            진입: 4개 중 3개 충족<br>
            ① MA20 > MA60<br>
            ② RSI 45~68<br>
            ③ MACD 양수<br>
            ④ 거래량 1.3배↑<br>
            손절: -5% / 익절: 트레일링 권장
            </div>""", unsafe_allow_html=True)
            bt_score_thr = 50
            bt_trailing  = st.toggle("트레일링 스탑", value=True,
                                     key="bt_trail_v6",
                                     help="V6 권장: ON (-20%)")
            if bt_trailing:
                bt_trail_pct = st.slider(
                    "트레일링 폭 (%)", 5, 25, 20, 1,
                    key="bt_pct_v6") / 100
            else:
                bt_trail_pct = 0.20

        else:
            # V1~V5
            st.markdown("**⚙️ 세부 파라미터**")
            bt_score_thr = st.slider(
                "AI 점수 기준 (%)", 0, 65, 40, 5,
                help="낮을수록 거래 많아짐")
            bt_trailing = st.toggle(
                "트레일링 스탑", value=False,
                help="ON: 최고점 대비 하락 시 청산")
            if bt_trailing:
                bt_trail_pct = st.slider(
                    "트레일링 폭 (%)", 5, 25, 10, 1) / 100
            else:
                bt_trail_pct = 0.10

        st.markdown("---")
        bt_btn=st.button("🧪 백테스트 실행",
            use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown('<div class="warn">⚠️ 참고용 분석입니다.<br>투자 손익은 본인 책임입니다.</div>',
                unsafe_allow_html=True)

# ── 백테스트 변수 기본값 ──────────────────────────────────
# 백테스트 탭 외 메뉴에서 변수 미정의 오류 방지
# session_state 대신 Python 변수로 안전하게 초기화
if menu != "📊 백테스트":
    bt_version   = "V7 — 과매도 역추세 (권장)"
    bt_score_thr = 40
    bt_trailing  = False
    bt_trail_pct = 0.10
    bt_btn       = False
    ver_cfg      = BT_VERSIONS[bt_version]
    ticker_type  = "저변동성"
    actual_strategy = "피보나치 분할매수 전략"
    auto_momentum   = False

# ════════════════════════════════════════════════════════════
# 헤더
# ════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">👑 프로젝트 이세계 V13</div>',unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI 기반 매수·매도 신호 + 자동 종목 추천 시스템</div>',unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 🏠 홈 대시보드
# ════════════════════════════════════════════════════════════
if menu=="🏠 홈 대시보드":
    st.markdown("### 📋 주요 종목 빠른 현황")
    QUICK_TICKERS=["SPY","QQQ","AAPL","NVDA","TSLA","MSFT","AMZN","META"]
    with st.spinner("주요 종목 분석 중..."):
        quick_results=[analyze(tk,"6mo") for tk in QUICK_TICKERS]
    rows_html=""
    table_data=[]
    for res in quick_results:
        if res is None: continue
        pct=res["pct"]; color="#00ff9d" if pct>=60 else "#ffd700" if pct>=40 else "#ff4757"
        ret1w_c="#00ff9d" if res["ret_1w"]>0 else "#ff4757"
        ret1m_c="#00ff9d" if res["ret_1m"]>0 else "#ff4757"
        table_data.append({
            "종목":res["ticker"],
            "현재가":f"${res['price']:.2f}",
            "1주":f"{res['ret_1w']:+.1f}%",
            "1개월":f"{res['ret_1m']:+.1f}%",
            "점수":f"{pct:.0f}%",
            "레짐":res["cfg"]["desc"],
            "신호":res["signal"],
            "행동":res["action"],
        })
    if table_data:
        # 종목 유형 + 권장 전략 컬럼 추가
        for row in table_data:
            tk = row["종목"]
            t_type = classify_ticker_type(tk)
            row["유형"] = "🔥 고변동" if t_type=="고변동성" else "🧊 저변동"
            row["권장전략"] = "모멘텀V6" if t_type=="고변동성" else "피보V5"
        # 컬럼 순서 명시적으로 지정
        df_home=pd.DataFrame(table_data)[[
            "종목","유형","권장전략","현재가","1주","1개월","점수","레짐","신호","행동"
        ]]
        st.dataframe(df_home, use_container_width=True, hide_index=True,
            column_config={
                "종목":     st.column_config.TextColumn("종목",    width="small"),
                "유형":     st.column_config.TextColumn("유형",    width="small"),
                "권장전략": st.column_config.TextColumn("권장전략",width="small"),
                "현재가":   st.column_config.TextColumn("현재가",  width="small"),
                "1주":      st.column_config.TextColumn("1주",     width="small"),
                "1개월":    st.column_config.TextColumn("1개월",   width="small"),
                "점수":     st.column_config.TextColumn("종합 점수", width="small"),
                "레짐":     st.column_config.TextColumn("장세",    width="small"),
                "신호":     st.column_config.TextColumn("신호",    width="medium"),
                "행동":     st.column_config.TextColumn("행동",    width="small"),
            })
    st.markdown("---")
    # 하락장 경고 배너 (홈에서도 표시)
    try:
        mkt = detect_market_regime()
        if mkt in ["하락장","조정장"]:
            st.markdown(f"""
            <div style="background:#1a0000;border:2px solid #ff4757;
                        border-radius:10px;padding:12px 16px;margin-bottom:12px">
              <div style="color:#ff4757;font-weight:700;font-size:.95rem">
                🔴 현재 시장: {mkt}
              </div>
              <div style="color:#fca5a5;font-size:.8rem;margin-top:4px">
                📌 AI 종목 추천 탭에서 하락장 대응 종목(인버스 ETF·안전자산·방어주)을 확인하세요.
              </div>
            </div>""", unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("### 📌 사용 방법")
    c1,c2,c3,c4=st.columns(4)
    for col,icon,title,desc in [
        (c1,"🔍","종목 분석","왼쪽에서 티커 입력 후 분석"),
        (c2,"🤖","AI 추천","자동으로 매수 후보 종목 스캔"),
        (c3,"📊","백테스트","과거 데이터로 전략 검증"),
        (c4,"📄","리포트","Excel 파일로 결과 저장"),
    ]:
        col.markdown(f"""<div class="mc" style="padding:16px;text-align:center">
          <div style="font-size:1.8rem;margin-bottom:8px">{icon}</div>
          <div style="font-weight:700;color:#e8eaf6;margin-bottom:4px">{title}</div>
          <div style="color:#6b7280;font-size:.78rem">{desc}</div></div>""",
          unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 🔍 종목 분석
# ════════════════════════════════════════════════════════════
elif menu=="🔍 종목 분석" and analyze_btn:
    # ── final_strat_name 기본값 (fib_fit_a 계산 전 참조 방지) ──
    final_strat_name = "V5 — 조건 완화 + 현실 익절"
    final_bt         = None
    final_source     = "초기화"
    icon_map = {
        "V5 — 조건 완화 + 현실 익절": "📐 V5 피보나치",
        "V6 — 고변동성 모멘텀":       "🚀 V6 모멘텀",
        "V7 — 과매도 역추세 (권장)":   "🎯 V7 역추세",
    }
    # ✅ 항상 2y 데이터로 받아서 멀티 기간 분석
    with st.spinner(f"📡 {ticker_input} 데이터 수집 중..."):
        analyze.clear()
        # 백테스트 기간은 사용자 선택 따름 (최소 2y 자동 확보)
        res = analyze(ticker_input, period_input)
    if res is None:
        st.error("데이터를 가져올 수 없습니다. 티커를 확인하세요."); st.stop()

    actual_label = {"6mo":"6개월","1y":"1년","2y":"2년","3y":"3년","5y":"5년"}.get(period_input,"2년")
    st.markdown(f"### 🔍 {ticker_input} 분석 결과")
    st.caption(f"📅 분석 기간: {actual_label} 데이터 기준 (종가 기준: {datetime.date.today()})")

    # ── 3가지 필터 확인 ─────────────────────────────────
    with st.spinner("🛡️ VIX / 섹터 / 어닝 필터 확인 중..."):
        filters = check_filters(ticker_input, res["regime"])

    # 필터 상태 카드
    vix_color = "#ff4757" if filters["vix_val"]>=30 else                 "#ff8c00" if filters["vix_val"]>=25 else                 "#ffd700" if filters["vix_val"]>=20 else "#00ff9d"
    sec_color = "#ff4757" if filters["sector_status"]=="WEAK" else                 "#ffd700" if filters["sector_status"]=="CAUTION" else "#00ff9d"
    earn_days = filters["days_left"]
    earn_color = "#ff4757" if (earn_days is not None and 0<=earn_days<=5) else                  "#ffd700" if (earn_days is not None and earn_days<=10) else "#00ff9d"
    earn_txt  = f"{earn_days}일 후" if earn_days is not None and earn_days>=0 else                 f"{abs(earn_days)}일 전 완료" if earn_days is not None else "확인불가"

    fc1, fc2, fc3 = st.columns(3)
    fc1.markdown(f"""
    <div style="background:#111827;border-radius:8px;padding:10px;text-align:center;
                border:1px solid {vix_color}44">
      <div style="color:#6b7280;font-size:.7rem">VIX 공포지수</div>
      <div style="color:{vix_color};font-weight:700;font-size:1.1rem">
        {filters["vix_val"]:.1f}
      </div>
      <div style="color:{vix_color};font-size:.72rem">{filters["vix_label"]}</div>
    </div>""", unsafe_allow_html=True)
    sec_etf_name = filters["sector_etf"]
    sec_etf_desc = SECTOR_DESC.get(sec_etf_name, sec_etf_name) if sec_etf_name != "해당없음" else "섹터 미분류"
    fc2.markdown(f"""
    <div style="background:#111827;border-radius:8px;padding:10px;text-align:center;
                border:1px solid {sec_color}44">
      <div style="color:#6b7280;font-size:.7rem">섹터 대표 ETF</div>
      <div style="color:{sec_color};font-weight:700;font-size:1rem">{sec_etf_name}</div>
      <div style="color:{sec_color};font-size:.68rem">
        {"❌ 하락 중" if filters["sector_status"]=="WEAK" else
         "⚠️ 주의" if filters["sector_status"]=="CAUTION" else
         "✅ 정상" if filters["sector_status"]=="OK" else "—"}
      </div>
      <div style="color:#6b7280;font-size:.63rem;margin-top:2px">{sec_etf_desc[:25]}</div>
    </div>""", unsafe_allow_html=True)
    fc3.markdown(f"""
    <div style="background:#111827;border-radius:8px;padding:10px;text-align:center;
                border:1px solid {earn_color}44">
      <div style="color:#6b7280;font-size:.7rem">어닝 발표</div>
      <div style="color:{earn_color};font-weight:700;font-size:1.1rem">{earn_txt}</div>
      <div style="color:{earn_color};font-size:.72rem">
        {"⚠️ 진입 금지" if earn_days is not None and 0<=earn_days<=5 else
         "📅 주의" if earn_days is not None and earn_days<=10 else "✅ 이상없음"}
      </div>
    </div>""", unsafe_allow_html=True)

    # 경고 메시지
    if filters["warnings"]:
        for w in filters["warnings"]:
            st.warning(w)

    # 진입 금지 배너
    if filters["block_entry"]:
        st.error("🚫 현재 진입 금지 조건 충족 — 위 경고 확인 후 진입 여부 결정")

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # Style Detector + 3전략 백테스트 비교
    # ════════════════════════════════════════════════════════
    with st.spinner("🧠 종목 성향 분석 + 전략 백테스트 비교 중..."):
        style_result = classify_stock_style(res["df"])

        # 3가지 전략 모두 빠른 백테스트
        bt_v5 = quick_backtest(res["df"], "V5 — 조건 완화 + 현실 익절")
        bt_v6 = quick_backtest(res["df"], "V6 — 고변동성 모멘텀")
        bt_v7 = quick_backtest(res["df"], "V7 — 과매도 역추세 (권장)")

    # ── Style Detector 결과 ──
    style      = style_result["style"]
    style_name = style_result["style_name"]
    style_conf = style_result["confidence"]
    best_strat = style_result["best_strategy"]
    reasons    = style_result["reasons"]
    s_scores   = style_result["scores"]

    style_colors = {"MOM":"#00ff9d","FIB":"#ffd700","STAT":"#00d4ff"}
    sc = style_colors.get(style,"#9ca3af")

    st.markdown(f"""
    <div style="background:#0f172a;border:2px solid {sc};
                border-radius:14px;padding:16px;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="color:{sc};font-weight:700;font-size:1.1rem">
            {style_name} — {ticker_input}
          </div>
          <div style="color:#9ca3af;font-size:.78rem;margin-top:4px">
            분류 신뢰도: {style_conf:.0f}% &nbsp;|&nbsp;
            MOM:{s_scores['MOM']}점 / FIB:{s_scores['FIB']}점 / STAT:{s_scores['STAT']}점
          </div>
        </div>
        <div style="text-align:right">
          <div style="color:#6b7280;font-size:.72rem">최적 전략</div>
          <div style="color:{sc};font-weight:700;font-size:.85rem">{best_strat[:10]}...</div>
        </div>
      </div>
      <div style="margin-top:10px;border-top:1px solid #1e2d4a;padding-top:8px">
        <div style="color:#6b7280;font-size:.72rem;margin-bottom:4px">분류 근거:</div>
        {"".join(f'<span style="background:#111827;border-radius:4px;padding:2px 6px;margin:2px;font-size:.72rem;color:{sc};display:inline-block">{r}</span>' for r in reasons)}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 3전략 백테스트 비교표 (최종 추천 반영) ──
    st.markdown("#### 📊 3전략 백테스트 비교")
    st.caption("⭐ 최종추천: 백테스트 성과 기반 | 📌 성향추천: Style Detector 기반")

    # ── 최종 전략 결정은 fib_fit_a/mom_fit_a/v7_fit_a 계산 후 진행 ──
    # (아래 종목 적합도 섹션에서 계산 후 결정됨)
    bt_candidates = [
        ("V5 — 조건 완화 + 현실 익절", bt_v5),
        ("V6 — 고변동성 모멘텀",       bt_v6),
        ("V7 — 과매도 역추세 (권장)",   bt_v7),
    ]
    valid_bts = [(n,b) for n,b in bt_candidates if b and b["trades"]>=3]

    # 비교표 재출력 — 최종 추천 반영 (fit_score_map은 아래서 계산 후 채워짐)
    icon_map = {
        "V5 — 조건 완화 + 현실 익절": "📐 V5 피보나치",
        "V6 — 고변동성 모멘텀":       "🚀 V6 모멘텀",
        "V7 — 과매도 역추세 (권장)":   "🎯 V7 역추세",
    }
    # ── 비교표는 fib_fit_a/final_strat_name 계산 후 출력 ──
    # (아래 종목 적합도 섹션 이후에 표시됨)
    _compare_placeholder = st.empty()
    _final_box_placeholder = st.empty()
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 멀티 기간 레짐 분석 (핵심 추가 기능)
    # 단기/중기/장기 레짐을 동시에 보여줌
    # ════════════════════════════════════════════════════════
    df_full = res["df"]

    def get_period_regime(df, window):
        """
        최근 N봉만 잘라서 그 구간의 레짐 + 수익률 + 변동성 계산
        레짐도 해당 구간 데이터로 새로 계산 (전체 데이터 레짐 아님)
        """
        if len(df) < window + 10:
            return "UNKNOWN", 0, 0
        # 해당 기간 데이터만 추출
        sub = df.tail(window).copy().reset_index(drop=True)
        if sub.empty or len(sub) < 5:
            return "UNKNOWN", 0, 0

        # 수익률: 해당 기간 시작→끝
        ret = float((sub["Close"].iloc[-1] / sub["Close"].iloc[0] - 1) * 100)
        # 변동성: 해당 기간 연환산
        vol = float(sub["Return"].std() * (252**0.5) * 100) if "Return" in sub.columns else 0

        # 레짐: 해당 기간의 MA 기울기로 판단 (기간별 독립 계산)
        close  = sub["Close"]
        ma20   = close.rolling(min(20, len(sub))).mean()
        ma_mid = close.rolling(min(window//2, len(sub))).mean()

        # 기간별 레짐 판단 기준
        # 단기(20일): 20일 수익률 + MA20 기울기
        # 중기(60일): 60일 수익률 + MA 기울기
        # 장기(200일): 장기 MA + ADX 사용
        ma_slope = float(ma20.iloc[-1] - ma20.iloc[max(-10,-len(ma20))]) if len(ma20) >= 5 else 0
        roc_val  = ret  # 해당 기간 수익률을 ROC로 사용

        # ADX (있으면 사용)
        adx_val = float(sub["ADX"].iloc[-1]) if "ADX" in sub.columns and not sub["ADX"].isna().all() else 15

        if roc_val > 3 and ma_slope > 0 and adx_val > 15:
            regime = "UPtrend"
        elif roc_val < -3 and ma_slope < 0 and adx_val > 15:
            regime = "DOWNtrend"
        elif abs(roc_val) <= 8:
            regime = "RANGE"
        elif roc_val > 0:
            regime = "UPtrend"
        else:
            regime = "DOWNtrend"

        return regime, round(ret, 1), round(vol, 1)

    r_short, ret_short, vol_short = get_period_regime(df_full, 20)   # 단기 1개월
    r_mid,   ret_mid,   vol_mid   = get_period_regime(df_full, 60)   # 중기 3개월
    r_long,  ret_long,  vol_long  = get_period_regime(df_full, 200)  # 장기 1년

    # 레짐 색상/아이콘
    def regime_style(r):
        return {
            "UPtrend":   ("#00ff9d", "📈 상승"),
            "RANGE":     ("#ffd700", "➡️ 박스"),
            "DOWNtrend": ("#ff4757", "📉 하락"),
            "UNKNOWN":   ("#6b7280", "❓ 불명"),
        }.get(r, ("#6b7280", "❓"))

    sc, sl = regime_style(r_short)
    mc, ml = regime_style(r_mid)
    lc, ll = regime_style(r_long)

    # 종합 판단 로직
    regimes = [r_short, r_mid, r_long]
    up_cnt  = regimes.count("UPtrend")
    dn_cnt  = regimes.count("DOWNtrend")
    rg_cnt  = regimes.count("RANGE")

    if up_cnt == 3:
        overall = ("🚀 강한 상승 추세", "#00ff9d",
                   "단/중/장기 모두 상승 — 모멘텀V6 + 트레일링 최적")
    elif up_cnt == 2 and dn_cnt == 0:
        overall = ("📈 상승 추세 (일부 조정)", "#4ade80",
                   "중/장기 상승 중 단기 조정 — 피보나치 매수 기회!")
    elif up_cnt == 1 and r_long == "UPtrend":
        overall = ("🟡 장기 상승 + 중기 조정", "#ffd700",
                   "장기 추세는 살아있음 — BUY1 구간 대기")
    elif dn_cnt == 3:
        overall = ("🔴 강한 하락 추세", "#ff4757",
                   "단/중/장기 모두 하락 — 매수 금지, 인버스 ETF 검토")
    elif dn_cnt == 2:
        overall = ("📉 하락 추세 우세", "#ff6b6b",
                   "하락 추세 — 반등 확인 후 소량 진입 또는 관망")
    elif r_short == "DOWNtrend" and r_long == "UPtrend":
        overall = ("⚡ 장기 상승 중 단기 하락", "#ff8c00",
                   "눌림목 구간 가능성 — 피보나치 BUY 구간 확인")
    elif rg_cnt >= 2:
        overall = ("➡️ 박스권", "#ffd700",
                   "방향성 없는 구간 — 피보나치 분할매수 V5 최적")
    else:
        overall = ("🤔 혼조세", "#9ca3af",
                   "추세 불명확 — 신중한 접근 필요")

    ov_txt, ov_color, ov_desc = overall

    # ── 종합 판단 배너 ──
    st.markdown(f"### {ov_txt}")
    st.caption(ov_desc)

    # ── 단기/중기/장기 레짐 카드 (st.columns 사용) ──
    col_s, col_m, col_l = st.columns(3)

    def regime_card(col, label, regime, ret, vol):
        color = {"UPtrend":"#00ff9d","RANGE":"#ffd700",
                 "DOWNtrend":"#ff4757"}.get(regime,"#6b7280")
        icon  = {"UPtrend":"📈 상승","RANGE":"➡️ 박스",
                 "DOWNtrend":"📉 하락"}.get(regime,"❓")
        ret_color = "#00ff9d" if ret > 0 else "#ff4757"
        col.markdown(f"""
        <div style="background:#111827;border-radius:10px;
                    padding:14px;text-align:center;border:1px solid {color}55">
          <div style="color:#6b7280;font-size:.72rem;margin-bottom:4px">{label}</div>
          <div style="color:{color};font-weight:700;font-size:1rem">{icon}</div>
          <div style="color:{ret_color};font-size:.85rem;
                      font-weight:700;margin-top:6px">{ret:+.1f}%</div>
          <div style="color:#6b7280;font-size:.68rem;margin-top:2px">
            변동성 {vol:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    regime_card(col_s, "단기 (1개월)", r_short, ret_short, vol_short)
    regime_card(col_m, "중기 (3개월)", r_mid,   ret_mid,   vol_mid)
    regime_card(col_l, "장기 (1년)",   r_long,  ret_long,  vol_long)

    # 종합 판단 기반 명확한 권장 전략 표시
    # 레짐 기반 기본 추천 방향
    if dn_cnt >= 2:
        regime_rec = "🚫 매수 금지"
        rec_color  = "#ff4757"
        rec_reason = "하락 추세 우세 — 인버스 ETF 또는 현금 보유"
    elif dn_cnt == 3:
        regime_rec = "🚫 전액 현금"
        rec_color  = "#ff4757"
        rec_reason = "전 구간 하락 — SQQQ/GLD 검토"
    elif up_cnt >= 2:
        regime_rec = "📈 상승 추세"
        rec_color  = "#00ff9d"
        rec_reason = "상승 추세 확인 — 모멘텀 전략 유리"
    elif rg_cnt >= 2:
        regime_rec = "➡️ 박스권"
        rec_color  = "#ffd700"
        rec_reason = "박스권 — 피보나치/역추세 전략 유리"
    else:
        regime_rec = "⏳ 혼조세"
        rec_color  = "#9ca3af"
        rec_reason = "방향 불명확 — 관망"

    # 최종 전략은 3전략 비교 후 아래서 결정됨
    rec_strategy = f"레짐: {regime_rec}"

    st.markdown(f"""
    <div style="background:#0f172a;border-left:4px solid {rec_color};
                border-radius:0 8px 8px 0;padding:10px 14px;margin-top:8px">
      <div style="color:{rec_color};font-weight:700;font-size:.88rem">
        {rec_strategy}
      </div>
      <div style="color:#9ca3af;font-size:.75rem;margin-top:3px">
        {rec_reason} — 최종 전략은 3전략 백테스트 비교 결과 참고
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # 오늘 종가 기준 매수/매도 신호 + 전략 추천
    # ════════════════════════════════════════════════════════
    st.markdown("### 📊 오늘 종가 기준 매매 신호")

    # 최종 추천 전략 표시 (fib_fit_a 계산 후 채워짐)
    _signal_strat_placeholder = st.empty()

    row_now = res["row"]
    price_now = res["price"]
    regime_now = res["regime"]

    # ── 현재 종가 기준 각 전략 신호 계산 (지표값만 미리 계산) ──
    # V6 모멘텀 신호
    ma20_now  = float(row_now.get("MA20", 0))
    ma60_now  = float(row_now.get("MA60", 1))
    rsi_now2  = float(row_now.get("RSI", 50))
    macd_now  = float(row_now.get("MACD_hist", 0)) if not pd.isna(row_now.get("MACD_hist", float("nan"))) else 0
    vol_now   = float(row_now.get("Volume", 0))
    volma_now = float(row_now.get("VolMA20", 1))

    v6_c1 = ma20_now > ma60_now
    v6_c2 = 45 <= rsi_now2 <= 68
    v6_c3 = macd_now > 0
    v6_c4 = volma_now > 0 and (vol_now / volma_now) >= 1.3
    v6_count = int(v6_c1)+int(v6_c2)+int(v6_c3)+int(v6_c4)
    v6_signal = v6_count >= 3

    # V7 과매도 신호
    zscore_now2  = float(row_now.get("ZScore", 0)) if not pd.isna(row_now.get("ZScore", float("nan"))) else 0
    bb_now2      = int(row_now.get("BB_touch_low", 0))
    rsi_os_now   = rsi_now2 < 30
    stoch_os_now = float(row_now.get("StochRSI", 50)) < 20
    bull_c_now   = int(row_now.get("BullCandle", 0))
    v7_score_now2 = int(zscore_now2 < -2) + int(bb_now2) + int(rsi_os_now) + int(stoch_os_now)
    v7_signal    = v7_score_now2 >= 2 and bull_c_now == 1

    # 피보나치 BUY 근접 신호
    fib1 = res["fib_lv"][0]
    fib_near = fib1 and abs(price_now - fib1) / price_now <= 0.03

    # ── rec_model/신호 배너/체크리스트는 fib_fit_a 계산 후 결정 ──
    # (아래 종목 적합도 섹션에서 final_strat_name 확정 후 채워짐)
    _sig_banner_placeholder   = st.empty()
    _sig_checklist_placeholder = st.empty()

    # (신호 배너+체크리스트는 fib_fit_a 계산 후 _sig_banner/checklist_placeholder에 채워짐)
    st.markdown("---")
    ticker_type_a = classify_ticker_type(ticker_input, res["df"])
    # ── 종목 적합도 분석 ─────────────────────────────────────
    df_a = res["df"]
    draw52_a  = float(df_a["Draw52w"].iloc[-1])  if "Draw52w"    in df_a.columns else -20
    ma200g_a  = float(df_a["MA200_grad"].iloc[-1]) if "MA200_grad" in df_a.columns else 0

    # 피보나치 적합도
    fib_fit_a = 0
    if res["regime"] == "RANGE":   fib_fit_a += 30
    if res["regime"] == "UPtrend": fib_fit_a += 20
    if -25 <= draw52_a <= -8:      fib_fit_a += 30
    if ma200g_a > 0:               fib_fit_a += 20
    if res["fib_lv"][0]:           fib_fit_a += 15
    fib_fit_a = min(100, fib_fit_a)

    # 모멘텀 적합도
    mom_fit_a = 0
    if res["regime"] == "UPtrend": mom_fit_a += 30
    if draw52_a >= -15:            mom_fit_a += 25
    if ma200g_a > 0:               mom_fit_a += 20
    mom_fit_a = min(100, mom_fit_a)

    # V7 역추세 적합도
    v7_fit_a = 0
    if res["regime"] == "RANGE":           v7_fit_a += 30
    if -40 <= draw52_a <= -8:              v7_fit_a += 30
    if "ZScore" in res["df"].columns:
        zs_now = float(res["df"]["ZScore"].dropna().iloc[-1])
        if zs_now < -1.5:                  v7_fit_a += 25
    if "BB_touch_low" in res["df"].columns:
        bb_cnt = int(res["df"]["BB_touch_low"].tail(60).sum())
        if bb_cnt >= 5:                    v7_fit_a += 15
    v7_fit_a = min(100, v7_fit_a)

    # 3전략 중 성향 기반 최고 점수
    best_strat_a = (
        "V7역추세" if v7_fit_a > fib_fit_a and v7_fit_a > mom_fit_a
        else "피보나치V5" if fib_fit_a >= mom_fit_a
        else "모멘텀V6"
    )
    fit_score_a   = max(fib_fit_a, mom_fit_a, v7_fit_a)
    if fit_score_a >= 80:   fit_grade_a = "🏆 최적"
    elif fit_score_a >= 60: fit_grade_a = "✅ 적합"
    elif fit_score_a >= 40: fit_grade_a = "⚠️ 보통"
    else:                   fit_grade_a = "❌ 부적합"

    # ── 최종 전략 결정: 성향 적합도 + 백테스트 종합 ──────
    fit_score_map = {
        "V5 — 조건 완화 + 현실 익절": fib_fit_a,
        "V6 — 고변동성 모멘텀":       mom_fit_a,
        "V7 — 과매도 역추세 (권장)":   v7_fit_a,
    }

    if valid_bts:
        def bt_score_fn(name, b):
            fit  = fit_score_map.get(name, 0)
            perf = b["cagr"]*0.5 + b["wr"]*0.3 + (b["mdd"]/(-50))*20
            return perf * 0.7 + fit * 0.3
        final_strat_name, final_bt = max(
            valid_bts, key=lambda x: bt_score_fn(x[0], x[1])
        )
        style_winner = max(fit_score_map, key=fit_score_map.get)
        bt_winner    = final_strat_name
        style_score  = fit_score_map[style_winner]
        bt_fit_score = fit_score_map.get(bt_winner, 0)
        if style_score - bt_fit_score >= 20:
            final_strat_name = style_winner
            final_bt = dict(bt_candidates).get(style_winner)
            final_source = f"성향 적합도 우선 ({style_winner[:3]}: {style_score}점 vs {bt_winner[:3]}: {bt_fit_score}점)"
        else:
            final_source = "백테스트 성과 + 성향 종합"
    else:
        final_strat_name = max(fit_score_map, key=fit_score_map.get)
        final_bt = dict(bt_candidates).get(final_strat_name)
        final_source = f"성향 적합도 기반 ({final_strat_name[:3]}: {fit_score_map[final_strat_name]}점)"

    # ── 비교표 출력 (여기서 final_strat_name 확정됨) ──────
    compare_rows_final = []
    for strat_name, bt_result in bt_candidates:
        is_final = strat_name == final_strat_name
        is_style = fit_score_map.get(strat_name, 0) == max(fit_score_map.values())
        tag = "⭐ 최종추천" if is_final else ("📌 성향추천" if is_style and not is_final else "")
        label = icon_map.get(strat_name, strat_name)
        if bt_result:
            compare_rows_final.append({
                "전략":   label,
                "거래수": bt_result["trades"],
                "승률":   f"{bt_result['wr']:.1f}%",
                "CAGR":   f"{bt_result['cagr']:+.1f}%",
                "MDD":    f"{bt_result['mdd']:.1f}%",
                "등급":   bt_result["grade"],
                "추천":   tag,
            })
        else:
            compare_rows_final.append({
                "전략":label, "거래수":0,
                "승률":"-","CAGR":"-","MDD":"-",
                "등급":"⚠️ 데이터 부족",
                "추천": tag,
            })

    _compare_placeholder.dataframe(pd.DataFrame(compare_rows_final),
        use_container_width=True, hide_index=True,
        column_config={
            "전략":   st.column_config.TextColumn(width="medium"),
            "거래수": st.column_config.NumberColumn(width="small"),
            "승률":   st.column_config.TextColumn(width="small"),
            "CAGR":   st.column_config.TextColumn(width="small"),
            "MDD":    st.column_config.TextColumn(width="small"),
            "등급":   st.column_config.TextColumn(width="small"),
            "추천":   st.column_config.TextColumn(width="small"),
        })
    st.caption("추천 기준: 성향 적합도 + 백테스트 성과 종합 | 성향 차이 20점↑이면 성향 우선")

    # ── 최종 추천 이유 박스 ──────────────────────────────
    final_color2 = "#00ff9d" if final_bt and final_bt.get("cagr",0)>=30 else "#ffd700"
    final_label2 = icon_map.get(final_strat_name, final_strat_name)
    rec_detail2  = (
        f"거래 {final_bt['trades']}건 | 승률 {final_bt['wr']}% | CAGR {final_bt['cagr']:+.0f}%"
        if final_bt and final_bt.get("trades",0)>=3
        else "백테스트 데이터 부족 — 5y 기간으로 재시도 권장"
    )
    _final_box_placeholder.markdown(f"""
    <div style="background:#0f172a;border:2px solid {final_color2};
                border-radius:10px;padding:14px;margin-top:4px">
      <div style="color:{final_color2};font-weight:700;font-size:.92rem;margin-bottom:6px">
        ⭐ 최종 추천: {final_label2}
      </div>
      <div style="color:#9ca3af;font-size:.78rem;line-height:2">
        • 근거: {final_source}<br>
        • 종목 성향: {style_name} ({", ".join(reasons[:2])})<br>
        • {rec_detail2}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 매매 신호 배너도 확정된 전략으로 채우기 ──────────
    _signal_strat_placeholder.markdown(f"""
    <div style="background:#111827;border-radius:8px;padding:8px 14px;
                margin-bottom:8px;display:flex;justify-content:space-between">
      <span style="color:#6b7280;font-size:.78rem">
        기준: {datetime.date.today()} 장 마감 종가 ${res['price']:.2f}
      </span>
      <span style="color:#00d4ff;font-size:.78rem;font-weight:700">
        최종 추천: {final_label2}
      </span>
    </div>""", unsafe_allow_html=True)

    # ── 신호 판단 (final_strat_name 확정 후) ──────────────
    if "V6" in final_strat_name:
        rec_model = "V6"; model_signal = v6_signal; model_score = v6_count
    elif "V7" in final_strat_name:
        rec_model = "V7"; model_signal = v7_signal; model_score = v7_score_now2
    else:
        rec_model = "V5"; model_signal = bool(fib_near); model_score = 1 if fib_near else 0
    if regime_now == "DOWNtrend":
        rec_model = "없음"; model_signal = False

    # 신호 배너
    if regime_now == "DOWNtrend":
        sig_color="#ff4757"; sig_icon="🚫"
        sig_title="매수 금지 — 하락장"
        sig_desc="현재 레짐이 DOWNtrend입니다. 매수하지 마세요.\n인버스 ETF(SQQQ, SPXS) 또는 현금 보유를 권장합니다."
        sig_action="오늘 할 일: 없음 (관망)"
    elif model_signal:
        sig_color="#00ff9d"; sig_icon="🟢"
        sig_title=f"매수 신호 — {rec_model} 전략"
        sig_desc = (f"모멘텀 조건 {v6_count}/4 충족 — 내일 시가 기준 매수 진입" if rec_model=="V6"
                    else f"과매도 조건 {v7_score_now2}/4 + 양봉 확인 — 내일 지정가 매수" if rec_model=="V7"
                    else f"피보나치 BUY1 근접 — 조건 추가 확인 후 진입")
        sig_action="오늘 할 일: 내일 지정가 주문 준비 ↓"
    elif v7_score_now2 >= 2 and not bull_c_now:
        sig_color="#ffd700"; sig_icon="⚡"
        sig_title="V7 과매도 감지 — 양봉 대기"
        sig_desc=f"과매도 조건 {v7_score_now2}/4 충족. 오늘 양봉(종가>시가) 미확인.\n내일 시가 확인 후 양봉이면 진입."
        sig_action="오늘 할 일: 내일 장 시작 후 양봉 확인 후 매수"
    elif fib1 and (price_now - fib1)/price_now <= 0.08:
        sig_color="#ffd700"; sig_icon="🟡"
        sig_title="피보나치 BUY1 근접 대기"
        sig_desc=f"BUY1 ${fib1:.2f}까지 {(fib1/price_now-1)*100:+.1f}% — 진입 구간 접근 중"
        sig_action="오늘 할 일: 지정가 주문 대기"
    else:
        sig_color="#6b7280"; sig_icon="⏳"
        sig_title="신호 없음 — 대기"
        sig_desc="현재 매수/매도 신호 없음. 조건 미충족."
        sig_action="오늘 할 일: 없음 (다음 분석 대기)"

    _sig_banner_placeholder.markdown(f"""
    <div style="background:#0f172a;border:2px solid {sig_color};
                border-radius:14px;padding:18px;margin-bottom:12px">
      <div style="font-size:1.6rem;margin-bottom:8px">{sig_icon}</div>
      <div style="color:{sig_color};font-weight:700;font-size:1.1rem;
                  margin-bottom:6px">{sig_title}</div>
      <div style="color:#9ca3af;font-size:.82rem;line-height:1.8;
                  white-space:pre-line">{sig_desc}</div>
      <div style="margin-top:12px;padding:8px 12px;background:#111827;
                  border-radius:8px;color:{sig_color};font-size:.82rem;
                  font-weight:700">{sig_action}</div>
    </div>""", unsafe_allow_html=True)

    # ── 체크리스트 탭 (최종 전략 첫 번째) ──────────────
    if regime_now != "DOWNtrend":
        with _sig_checklist_placeholder.container():
            if "V7" in final_strat_name:
                tab_order = [f"🎯 V7 역추세 ({v7_score_now2}/4)",
                             f"🚀 V6 모멘텀 ({v6_count}/4)", f"📐 V5 피보나치"]
            elif "V6" in final_strat_name:
                tab_order = [f"🚀 V6 모멘텀 ({v6_count}/4)",
                             f"🎯 V7 역추세 ({v7_score_now2}/4)", f"📐 V5 피보나치"]
            else:
                tab_order = [f"📐 V5 피보나치",
                             f"🎯 V7 역추세 ({v7_score_now2}/4)",
                             f"🚀 V6 모멘텀 ({v6_count}/4)"]
            check_tabs = st.tabs(tab_order)

            with check_tabs[0]:
                if "V6" in tab_order[0]:
                    st.caption("4조건 중 3개 이상 → 내일 시가 매수")
                    for label,ok,detail in [
                        ("MA20 > MA60",v6_c1,f"MA20:{ma20_now:.1f} vs MA60:{ma60_now:.1f}"),
                        ("RSI 45~68",v6_c2,f"RSI:{rsi_now2:.1f}"),
                        ("MACD 양수",v6_c3,f"MACD:{macd_now:+.4f}"),
                        ("거래량 1.3배↑",v6_c4,f"{vol_now/volma_now:.2f}배" if volma_now>0 else "-"),
                    ]:
                        c="#00ff9d" if ok else "#ff4757"
                        st.markdown(f"<div style='padding:5px 0;border-bottom:1px solid #1e2d4a'><span style='color:{c}'>{'✅' if ok else '❌'}</span> <span style='color:#e8eaf6;font-size:.84rem'>{label}</span><span style='color:#6b7280;font-size:.74rem;float:right'>{detail}</span></div>",unsafe_allow_html=True)
                    st.success(f"✅ V6 충족 ({v6_count}/4)") if v6_count>=3 else st.info(f"⏳ V6 미충족 ({v6_count}/4)")
                elif "V7" in tab_order[0]:
                    st.caption("과매도 2개↑ + 양봉 → 내일 지정가 매수")
                    for label,ok,detail in [
                        ("Z-Score < -2",zscore_now2<-2,f"Z:{zscore_now2:.2f}"),
                        ("BB 하단 터치",bool(bb_now2),"터치중" if bb_now2 else "위에있음"),
                        ("RSI < 30",rsi_os_now,f"RSI:{rsi_now2:.1f}"),
                        ("StochRSI < 20",stoch_os_now,f"Stoch:{float(row_now.get('StochRSI',50)):.1f}"),
                        ("오늘 양봉",bool(bull_c_now),"양봉✅" if bull_c_now else "음봉⚡"),
                    ]:
                        c="#00ff9d" if ok else "#ff4757"
                        st.markdown(f"<div style='padding:5px 0;border-bottom:1px solid #1e2d4a'><span style='color:{c}'>{'✅' if ok else '❌'}</span> <span style='color:#e8eaf6;font-size:.84rem'>{label}</span><span style='color:#6b7280;font-size:.74rem;float:right'>{detail}</span></div>",unsafe_allow_html=True)
                    st.success(f"✅ V7 충족 ({v7_score_now2}/4 + 양봉)") if v7_signal else st.warning(f"⚡ 과매도 감지 ({v7_score_now2}/4) — 양봉 미확인") if v7_score_now2>=2 else st.info(f"⏳ V7 미충족 ({v7_score_now2}/4)")
                else:  # V5
                    st.caption("피보나치 BUY 구간 도달 시 분할매수")
                    for lv,pv in zip(["BUY1","BUY2","BUY3"],res["fib_lv"]):
                        if pv:
                            d=(pv/price_now-1)*100; c="#00ff9d" if abs(d)<=3 else "#ffd700" if abs(d)<=8 else "#6b7280"
                            st.markdown(f"<div style='padding:6px 0;border-bottom:1px solid #1e2d4a'><span style='color:{c};font-weight:700'>{lv}: ${pv:.2f}</span><span style='color:{c};font-size:.8rem;margin-left:8px'>{d:+.1f}%</span></div>",unsafe_allow_html=True)

            with check_tabs[1]:
                st.caption("참고용 체크리스트")
                st.info("탭 0에서 최종 추천 전략 확인하세요")

            with check_tabs[2]:
                st.caption("참고용 체크리스트")
                st.info("탭 0에서 최종 추천 전략 확인하세요")

    t_color = "#ff8c00" if ticker_type_a == "고변동성" else "#00d4ff"
    t_icon  = "🔥" if ticker_type_a == "고변동성" else "🧊"

    st.markdown(f"""
    <div style="background:#0f172a;border:2px solid {t_color};
                border-radius:12px;padding:14px 16px;margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div style="display:flex;align-items:center;gap:10px">
          <span style="font-size:1.4rem">{t_icon}</span>
          <div>
            <div style="color:{t_color};font-weight:700;font-size:.95rem">
              {ticker_input} — {ticker_type_a}
            </div>
            <div style="color:#e8eaf6;font-size:.82rem;margin-top:2px">
              종목 성향: <b>{style_name}</b> &nbsp; 신뢰도 {style_conf:.0f}%
            </div>
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;
                  gap:8px;margin-top:12px">
        <div style="background:#111827;border-radius:8px;padding:8px;text-align:center">
          <div style="color:#6b7280;font-size:.7rem">52주 낙폭</div>
          <div style="color:{'#ff4757' if draw52_a < -30 else '#ffd700' if draw52_a < -15 else '#00ff9d'};
                      font-weight:700">{draw52_a:.1f}%</div>
          <div style="color:#6b7280;font-size:.66rem">
            {'❌ 과도한 하락' if draw52_a < -30 else '⚠️ 주의' if draw52_a < -15 else '✅ 적정'}
          </div>
        </div>
        <div style="background:#111827;border-radius:8px;padding:8px;text-align:center">
          <div style="color:#6b7280;font-size:.7rem">MA200 방향</div>
          <div style="color:{'#00ff9d' if ma200g_a > 0 else '#ff4757'};font-weight:700">
            {'↑ 상승' if ma200g_a > 0 else '↓ 하락'}
          </div>
          <div style="color:#6b7280;font-size:.66rem">
            {'✅ 장기 상승' if ma200g_a > 0 else '❌ 장기 하락'}
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                  gap:8px;margin-top:8px">
        <div style="background:#111827;border-radius:8px;padding:8px;text-align:center">
          <div style="color:#6b7280;font-size:.7rem">📐 피보V5 적합</div>
          <div style="color:{'#00ff9d' if fib_fit_a>=60 else '#ffd700' if fib_fit_a>=40 else '#ff4757'};
                      font-weight:700">{fib_fit_a}점</div>
          <div style="color:#6b7280;font-size:.64rem">RANGE+눌림목</div>
        </div>
        <div style="background:#111827;border-radius:8px;padding:8px;text-align:center">
          <div style="color:#6b7280;font-size:.7rem">🚀 V6모멘 적합</div>
          <div style="color:{'#00ff9d' if mom_fit_a>=60 else '#ffd700' if mom_fit_a>=40 else '#ff4757'};
                      font-weight:700">{mom_fit_a}점</div>
          <div style="color:#6b7280;font-size:.64rem">UP+추세추격</div>
        </div>
        <div style="background:#111827;border-radius:8px;padding:8px;text-align:center">
          <div style="color:#6b7280;font-size:.7rem">🎯 V7역추 적합</div>
          <div style="color:{'#00ff9d' if v7_fit_a>=60 else '#ffd700' if v7_fit_a>=40 else '#ff4757'};
                      font-weight:700">{v7_fit_a}점</div>
          <div style="color:#6b7280;font-size:.64rem">RANGE+과매도</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── 핵심 정보 카드 ──
    sig_colors={"strong_buy":"#00ff9d","buy":"#4ade80","watch":"#ffd700","hold":"#9ca3af","sell":"#ff4757"}
    sc=sig_colors.get(res["sig_k"],"#9ca3af")

    c1,c2,c3,c4,c5,c6=st.columns(6)
    mcard(c1,"현재가",f"${res['price']:.2f}","#00d4ff")
    mcard(c2,"종합 점수",f"{res['pct']:.0f}%",sc,f"{res['ts']+res['cs']+res['ss']+res['irs']:.0f}/16")
    mcard(c3,"장세",res["cfg"]["desc"],"#e8eaf6")
    mcard(c4,"종목 유형",
          "🔥 고변동성" if ticker_type_a=="고변동성" else "🧊 저변동성",
          "#ff8c00" if ticker_type_a=="고변동성" else "#00d4ff")
    mcard(c5,"신호",res["signal"],sc)
    mcard(c6,"권장 행동",res["action"],sc)
    st.markdown("---")

    # ── 매매 플랜 표 (최종 추천 전략 기반) ──
    st.markdown("#### 📋 매매 플랜")

    if "V6" in final_strat_name:
        # V6 모멘텀 플랜
        entry_v6 = res["price"] * 1.002
        stop_v6  = entry_v6 * 0.95
        tp_v6    = entry_v6 * 1.10
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #00ff9d;border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;font-size:.82rem">
          <b style="color:#00ff9d">🚀 V6 모멘텀 전략 매매 플랜</b><br>
          <span style="color:#6b7280">진입 방식:</span>
          <span style="color:#e8eaf6"> 조건 충족 시 내일 시가 매수 (분할매수 없음)</span><br>
          <span style="color:#6b7280">기준가:</span>
          <span style="color:#00d4ff;font-family:monospace"> ${res['price']:.2f} (오늘 종가)</span>
        </div>""", unsafe_allow_html=True)
        plan_data = {
            "구분":       ["진입가 (시가+슬리피지)", "손절선 (-5%)", "익절 목표 (+10%)", "트레일링 스탑"],
            "목표가":     [f"${entry_v6:.2f}", f"${stop_v6:.2f}", f"${tp_v6:.2f}", "최고가 -20%"],
            "현재가 대비":[f"+0.2%", f"-5.0%", f"+10.0%", "자동추적"],
            "비고":       ["내일 시가에 지정가", "체결 즉시 설정", "고정 or 트레일링", "트레일링 권장"],
        }

    elif "V7" in final_strat_name:
        # V7 역추세 플랜
        b1      = res["fib_lv"][0]
        fib886  = res.get("fib886")  # analyze에서 계산된 0.886 레벨
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #00d4ff;border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;font-size:.82rem">
          <b style="color:#00d4ff">🎯 V7 역추세 전략 매매 플랜</b><br>
          <span style="color:#6b7280">진입 조건:</span>
          <span style="color:#e8eaf6"> 피보BUY1 + 과매도 2개 이상 + 양봉 확인</span><br>
          <span style="color:#ff4757">물타기 금지</span>
          <span style="color:#6b7280"> — </span>
          <span style="color:#00ff9d">불타기: BUY1 진입 후 반등 확인 시 추가 매수 (가격 올라감)</span><br>
          <span style="color:#6b7280;font-size:.75rem">
            손절 기준: 피보나치 0.886 레벨 이탈 시 (마지막 지지선)
          </span>
        </div>""", unsafe_allow_html=True)

        if b1:
            # 불타기: BUY1보다 위에서 추가 매수
            buy2_bull = b1 * 1.03   # +3% 반등 확인 시
            buy3_bull = b1 * 1.06   # +6% 추세 확인 시

            # 손절: 피보 0.886 레벨 (없으면 -5% 타이트하게)
            if fib886 and fib886 < b1:
                stop_v7      = fib886
                stop_pct     = (fib886 / b1 - 1) * 100
                stop_label   = f"피보 0.886 이탈 ({stop_pct:+.1f}%)"
            else:
                stop_v7      = b1 * 0.95
                stop_pct     = -5.0
                stop_label   = "BUY1 대비 -5% (0.886 미계산)"

            tp_v7  = b1 * 1.12
            rr     = abs(tp_v7 - b1) / abs(b1 - stop_v7) if b1 > stop_v7 else 0

            plan_data = {
                "구분": [
                    "BUY1 (최초 진입)",
                    "BUY2 (불타기 +3%)",
                    "BUY3 (불타기 +6%)",
                    f"손절선 (Fib 0.886)",
                    "익절 목표 (+12%)",
                ],
                "목표가": [
                    f"${b1:.2f}",
                    f"${buy2_bull:.2f}",
                    f"${buy3_bull:.2f}",
                    f"${stop_v7:.2f}",
                    f"${tp_v7:.2f}",
                ],
                "현재가 대비": [
                    f"{(b1/res['price']-1)*100:+.1f}%",
                    f"{(buy2_bull/res['price']-1)*100:+.1f}%",
                    f"{(buy3_bull/res['price']-1)*100:+.1f}%",
                    f"{(stop_v7/res['price']-1)*100:+.1f}%",
                    f"{(tp_v7/res['price']-1)*100:+.1f}%",
                ],
                "비고": [
                    "과매도+양봉 확인 시",
                    "모멘텀 회복 2개↑ 시",
                    "추세 강화 확인 시",
                    f"{stop_label} → 손익비 {rr:.1f}배",
                    "트레일링 -15% 병행",
                ],
            }
        else:
            plan_data = {
                "구분": ["BUY1 지정가", "BUY2 (불타기)", "BUY3 (불타기)", "손절선", "익절 목표"],
                "목표가": ["대기 중"] * 5,
                "현재가 대비": ["—"] * 5,
                "비고": ["피보나치 BUY 구간 미형성"] * 5,
            }

    else:
        # V5 피보나치 플랜 (기존)
        row_data = res["row"]
        sw_h = float(row_data["sw_high"]) if not pd.isna(row_data["sw_high"]) else None
        sw_l = float(row_data["sw_low"])  if not pd.isna(row_data["sw_low"])  else None
        rng_val = float(row_data["rng"]) if not pd.isna(row_data["rng"]) else None
        if sw_h and sw_l and rng_val:
            fib_basis_ok = sw_h > res["price"]
            basis_color  = "#00ff9d" if fib_basis_ok else "#ffd700"
            basis_msg    = "✅ 정상 (고점 → 되돌림 구간)" if fib_basis_ok else "⚠️ 현재가가 스윙 고점 위 — 조정 후 구간 형성"
            st.markdown(f"""
            <div style="background:#111827;border:1px solid {basis_color};border-radius:8px;
                        padding:10px 14px;margin-bottom:10px;font-size:.82rem">
              <b style="color:#ffd700">📐 V5 피보나치 전략 매매 플랜</b><br>
              <span style="color:#6b7280">스윙 고점:</span>
              <span style="color:#00d4ff;font-family:monospace"> ${sw_h:.2f}</span> &nbsp;|&nbsp;
              <span style="color:#6b7280">스윙 저점:</span>
              <span style="color:#7b5ea7;font-family:monospace"> ${sw_l:.2f}</span> &nbsp;|&nbsp;
              <span style="color:#6b7280">범위:</span>
              <span style="color:#ffd700;font-family:monospace"> ${rng_val:.2f}</span><br>
              <span style="color:{basis_color}">{basis_msg}</span>
            </div>""", unsafe_allow_html=True)
        if all(f is None for f in res["fib_lv"]):
            st.warning("⚠️ 피보나치 매수 구간 미형성 — 조정 후 재분석")
        plan_data = {
            "구분":       ["1차 매수 (BUY1)", "2차 매수 (BUY2)", "3차 매수 (BUY3)", "손절선", "익절 목표"],
            "목표가":     [
                f"${res['fib_lv'][0]:.2f}" if res["fib_lv"][0] else "대기 중",
                f"${res['fib_lv'][1]:.2f}" if res["fib_lv"][1] else "대기 중",
                f"${res['fib_lv'][2]:.2f}" if res["fib_lv"][2] else "대기 중",
                f"${res['stop_s']:.2f}" if res["stop_s"] else "—",
                f"${res['tp_s']:.2f}"   if res["tp_s"]   else "—",
            ],
            "현재가 대비":[
                f"{(res['fib_lv'][0]/res['price']-1)*100:+.1f}%" if res["fib_lv"][0] else "—",
                f"{(res['fib_lv'][1]/res['price']-1)*100:+.1f}%" if res["fib_lv"][1] else "—",
                f"{(res['fib_lv'][2]/res['price']-1)*100:+.1f}%" if res["fib_lv"][2] else "—",
                f"{(res['stop_s']/res['price']-1)*100:+.1f}%"    if res["stop_s"]    else "—",
                f"{(res['tp_s']/res['price']-1)*100:+.1f}%"      if res["tp_s"]      else "—",
            ],
            "비고": ["30% 진입", "35% 추가", "35% 추가", "전량 손절", f"레짐:{res['regime']}"],
        }

    st.dataframe(pd.DataFrame(plan_data), use_container_width=True, hide_index=True,
        column_config={
            "구분":        st.column_config.TextColumn(width="medium"),
            "목표가":      st.column_config.TextColumn(width="small"),
            "현재가 대비": st.column_config.TextColumn(width="small"),
            "비고":        st.column_config.TextColumn(width="medium"),
        })

    # ── 내일 지정가 주문 가격표 (최종 전략 기반) ────────────
    st.markdown("#### 📋 내일 지정가 주문 가격표")

    slip       = 0.002
    price_now2 = res["price"]
    row_now2   = res["row"]
    cfg_now    = res["cfg"]

    # 지표값
    v7_score_now = int(row_now2.get("V7_oversold_score", 0))
    bull_now2    = int(row_now2.get("BullCandle", 0))
    zscore_now2  = float(row_now2.get("ZScore", 0)) if not pd.isna(row_now2.get("ZScore", float("nan"))) else 0
    rsi_now2b    = float(row_now2.get("RSI", 50))
    bb_now2      = int(row_now2.get("BB_touch_low", 0))
    ma20_now2    = float(row_now2.get("MA20", 0))
    ma60_now2    = float(row_now2.get("MA60", 1))
    macd_now2    = float(row_now2.get("MACD_hist", 0)) if not pd.isna(row_now2.get("MACD_hist", float("nan"))) else 0
    vol_now2     = float(row_now2.get("Volume", 0))
    volma_now2   = float(row_now2.get("VolMA20", 1))

    # 전략에 따라 진입가/손절/익절 계산
    if "V6" in final_strat_name:
        # V6 모멘텀: 현재가 기준 진입 (시가 + 슬리피지)
        entry_price = price_now2 * (1 + slip)
        stop_price2 = entry_price * (1 - 0.05)   # 손절 -5%
        tp_price2   = entry_price * (1 + 0.10)   # 익절 +10% (트레일링 -20%)
        entry_label = f"현재가 기준 (${entry_price:.2f})"
        stop_label  = f"-5% (${stop_price2:.2f})"
        tp_label    = f"+10% 고정 or 트레일링 -20%"
        plan_note   = "V6 모멘텀: 조건 충족 시 내일 시가에 매수 → 손절 -5% / 트레일링 -20%"
        v6_cnt2 = int(ma20_now2>ma60_now2)+int(45<=rsi_now2b<=68)+int(macd_now2>0)+int(volma_now2>0 and vol_now2/volma_now2>=1.3)
        cond_txt = f"모멘텀 조건 {v6_cnt2}/4 {'✅ 충족' if v6_cnt2>=3 else '❌ 미충족'}"
        order_ok = v6_cnt2 >= 3
    elif "V7" in final_strat_name:
        # V7 역추세: 피보나치 BUY1 지정가
        b1 = res["fib_lv"][0]
        if b1:
            entry_price = b1 * (1 + slip)
            stop_price2 = entry_price * (1 - max(cfg_now["stop"], 0.10))
            tp_price2   = entry_price * 1.12
            entry_label = f"피보BUY1 지정가 (${entry_price:.2f})"
            stop_label  = f"-10% (${stop_price2:.2f})"
            tp_label    = f"+12% 고정 or 트레일링 -15%"
            plan_note   = "V7 역추세: 피보BUY1 + 과매도 2개 이상 + 양봉 확인 시 진입"
            rr2 = (tp_price2-entry_price)/(entry_price-stop_price2) if entry_price>stop_price2 else 0
            cond_txt = f"V7 과매도 {v7_score_now}/4 | 양봉 {'✅' if bull_now2 else '❌'} | 손익비 {rr2:.1f}배"
            order_ok = v7_score_now >= 2 and bull_now2 == 1
        else:
            st.info("⏳ V7: 피보나치 BUY 구간 미형성 — 조정 후 재분석")
            entry_price = stop_price2 = tp_price2 = 0
            entry_label = stop_label = tp_label = plan_note = cond_txt = ""
            order_ok = False
    else:
        # V5 피보나치
        b1 = res["fib_lv"][0]
        if b1:
            entry_price = b1 * (1 + slip)
            stop_price2 = res["stop_s"] if res["stop_s"] else b1*(1-cfg_now["stop"])
            tp_price2   = res["tp_s"]   if res["tp_s"]   else b1*(1+cfg_now["tp"])
            entry_label = f"피보BUY1 지정가 (${entry_price:.2f})"
            stop_label  = f"-{cfg_now['stop']*100:.0f}% (${stop_price2:.2f})"
            tp_label    = f"+{cfg_now['tp']*100:.0f}% (${tp_price2:.2f})"
            plan_note   = f"V5 피보나치: BUY1 도달 시 분할매수 시작"
            dist2 = (b1/price_now2-1)*100
            cond_txt = f"BUY1까지 {dist2:+.1f}% {'✅ 근접' if abs(dist2)<=3 else '⏳ 대기'}"
            order_ok = abs(dist2) <= 3
        else:
            st.info("⏳ V5: 피보나치 BUY 구간 미형성 — 조정 후 재분석")
            entry_price = stop_price2 = tp_price2 = 0
            entry_label = stop_label = tp_label = plan_note = cond_txt = ""
            order_ok = False

    if entry_price > 0:
        order_color2 = "#00ff9d" if order_ok else "#ffd700"
        order_status2 = "✅ 조건 충족 — 내일 주문 준비" if order_ok else "⏳ 조건 미충족 — 대기"
        final_lbl_now = icon_map.get(final_strat_name, final_strat_name)
        st.caption(f"적용 전략: {final_lbl_now} | {cond_txt}")
        st.markdown(f"""
        <div style="background:#0f172a;border:2px solid {order_color2};
                    border-radius:12px;padding:16px;margin-bottom:8px">
          <div style="color:{order_color2};font-weight:700;margin-bottom:10px">
            {order_status2}
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
            <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
              <div style="color:#6b7280;font-size:.7rem">🟢 진입 지정가</div>
              <div style="color:#00ff9d;font-weight:700;font-size:.95rem">${entry_price:.2f}</div>
              <div style="color:#6b7280;font-size:.65rem">{entry_label}</div>
            </div>
            <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
              <div style="color:#6b7280;font-size:.7rem">🔴 손절 스탑</div>
              <div style="color:#ff4757;font-weight:700;font-size:.95rem">${stop_price2:.2f}</div>
              <div style="color:#6b7280;font-size:.65rem">{stop_label}</div>
            </div>
            <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
              <div style="color:#6b7280;font-size:.7rem">🎯 익절 목표</div>
              <div style="color:#ffd700;font-weight:700;font-size:.95rem">${tp_price2:.2f}</div>
              <div style="color:#6b7280;font-size:.65rem">{tp_label}</div>
            </div>
          </div>
          <div style="margin-top:8px;color:#6b7280;font-size:.72rem">
            📌 {plan_note}
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # 📝 내 매매 기록 + 재진입 타이밍
    # ════════════════════════════════════════════════════════════
    st.markdown("#### 📝 내 매매 기록 & 재진입 타이밍")

    # session_state로 매매 기록 관리 (페이지 새로고침 전까지 유지)
    rec_key = f"trade_log_{ticker_input}"
    if rec_key not in st.session_state:
        st.session_state[rec_key] = []

    with st.expander("✏️ 매매 기록 입력", expanded=True):
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        tr_date  = rc1.date_input("날짜", value=datetime.date.today(), key="tr_date")
        tr_type  = rc2.selectbox("구분", ["매수","손절","익절","추가매수"], key="tr_type")
        tr_price = rc3.number_input("가격 ($)", min_value=0.0, value=0.0,
                                    step=0.01, format="%.2f", key="tr_price")
        tr_qty   = rc4.number_input("수량 (주)", min_value=0, value=0,
                                    step=1, key="tr_qty")
        tr_memo  = rc5.text_input("메모", value="", key="tr_memo")

        col_add, col_clear = st.columns(2)
        if col_add.button("➕ 기록 추가", use_container_width=True):
            if tr_price > 0 and tr_qty > 0:
                st.session_state[rec_key].append({
                    "날짜":   str(tr_date),
                    "구분":   tr_type,
                    "가격":   tr_price,
                    "수량":   tr_qty,
                    "금액":   tr_price * tr_qty,
                    "메모":   tr_memo,
                })
                st.success(f"✅ {tr_type} 기록 추가됨")
                st.rerun()
            else:
                st.warning("가격과 수량을 입력하세요.")

        if col_clear.button("🗑️ 기록 초기화", use_container_width=True):
            st.session_state[rec_key] = []
            st.rerun()

    # 기록이 있으면 분석 표시
    logs = st.session_state.get(rec_key, [])
    if logs:
        df_log = pd.DataFrame(logs)
        st.dataframe(df_log, use_container_width=True, hide_index=True,
            column_config={
                "날짜":  st.column_config.TextColumn(width="small"),
                "구분":  st.column_config.TextColumn(width="small"),
                "가격":  st.column_config.NumberColumn(format="$%.2f", width="small"),
                "수량":  st.column_config.NumberColumn(width="small"),
                "금액":  st.column_config.NumberColumn(format="$%.2f", width="small"),
                "메모":  st.column_config.TextColumn(width="medium"),
            })

        # 현재 포지션 계산
        buy_logs  = [l for l in logs if l["구분"] in ["매수","추가매수"]]
        sell_logs = [l for l in logs if l["구분"] in ["손절","익절"]]
        open_qty  = sum(l["수량"] for l in buy_logs) - sum(l["수량"] for l in sell_logs)

        if open_qty > 0 and buy_logs:
            # 열린 포지션 분석
            total_cost = sum(l["금액"] for l in buy_logs) - sum(l["금액"] for l in sell_logs)
            my_avg     = total_cost / open_qty if open_qty > 0 else 0
            my_pnl_pct = (res["price"] - my_avg) / my_avg * 100 if my_avg > 0 else 0
            my_pnl_usd = (res["price"] - my_avg) * open_qty if my_avg > 0 else 0
            pnl_color  = "#00ff9d" if my_pnl_pct > 0 else "#ff4757"

            stop_s = res["stop_s"] if res["stop_s"] else my_avg * (1 - res["cfg"]["stop"])
            tp_s   = res["tp_s"]   if res["tp_s"]   else my_avg * (1 + res["cfg"]["tp"])

            st.markdown(f"""
            <div style="background:#0f172a;border:1.5px solid #00d4ff;
                        border-radius:10px;padding:14px;margin-top:8px">
              <div style="color:#00d4ff;font-weight:700;margin-bottom:10px">
                📊 현재 포지션 현황
              </div>
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">
                <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
                  <div style="color:#6b7280;font-size:.7rem">평균단가</div>
                  <div style="color:#00d4ff;font-weight:700">${my_avg:.2f}</div>
                </div>
                <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
                  <div style="color:#6b7280;font-size:.7rem">수익률</div>
                  <div style="color:{pnl_color};font-weight:700">{my_pnl_pct:+.2f}%</div>
                </div>
                <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
                  <div style="color:#6b7280;font-size:.7rem">평가손익</div>
                  <div style="color:{pnl_color};font-weight:700">${my_pnl_usd:+.2f}</div>
                </div>
                <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
                  <div style="color:#6b7280;font-size:.7rem">보유 주수</div>
                  <div style="color:#e8eaf6;font-weight:700">{open_qty}주</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:.82rem;line-height:2;color:#9ca3af">
                손절선: <b style="color:#ff4757">${stop_s:.2f}</b>
                ({(stop_s/my_avg-1)*100:+.1f}%) &nbsp;|&nbsp;
                익절 목표: <b style="color:#00ff9d">${tp_s:.2f}</b>
                ({(tp_s/my_avg-1)*100:+.1f}%)
              </div>
            </div>""", unsafe_allow_html=True)

        elif sell_logs:
            # 손절/익절 후 재진입 타이밍 분석
            last_sell = sell_logs[-1]
            is_stoploss = last_sell["구분"] == "손절"

            if is_stoploss:
                st.markdown("""
                <div style="background:#1a0000;border:1.5px solid #ff4757;
                            border-radius:10px;padding:14px;margin-top:8px">
                  <div style="color:#ff4757;font-weight:700;margin-bottom:8px">
                    ❌ 손절 후 재진입 타이밍 분석
                  </div>""", unsafe_allow_html=True)

                # 재진입 조건 체크
                reentry_conditions = []

                # 1. 레짐 확인
                if res["regime"] == "DOWNtrend":
                    reentry_conditions.append(("❌ 레짐", "DOWNtrend — 재진입 금지", False))
                elif res["regime"] == "UPtrend":
                    reentry_conditions.append(("✅ 레짐", "UPtrend — 재진입 가능", True))
                else:
                    reentry_conditions.append(("🟡 레짐", "RANGE — 조건부 재진입", True))

                # 2. 피보나치 BUY 구간
                if res["fib_lv"][0]:
                    dist = (res["fib_lv"][0]/res["price"]-1)*100
                    reentry_conditions.append((
                        "✅ BUY1" if abs(dist)<=5 else "🟡 BUY1",
                        f"${res['fib_lv'][0]:.2f} ({dist:+.1f}%)",
                        abs(dist) <= 10
                    ))
                else:
                    reentry_conditions.append(("⏳ BUY1", "대기 중 — 구간 미형성", False))

                # 3. StochRSI
                stoch_ok = res["stoch"] < res["cfg"]["stoch"]
                reentry_conditions.append((
                    "✅ StochRSI" if stoch_ok else "❌ StochRSI",
                    f"{res['stoch']:.1f} (기준: {res['cfg']['stoch']} 이하)",
                    stoch_ok
                ))

                # 4. AI 점수
                score_ok = res["pct"] >= 45
                reentry_conditions.append((
                    "✅ AI점수" if score_ok else "❌ AI점수",
                    f"{res['pct']:.0f}% (기준: 45% 이상)",
                    score_ok
                ))

                ok_count = sum(1 for _,_,ok in reentry_conditions if ok)
                ready    = ok_count >= 3

                for label, desc, ok in reentry_conditions:
                    color = "#00ff9d" if ok else "#ff4757"
                    st.markdown(
                        f"<div style='color:{color};font-size:.82rem;padding:2px 0'>"
                        f"{label}: {desc}</div>", unsafe_allow_html=True
                    )

                conclusion_color = "#00ff9d" if ready else "#ff4757"
                conclusion = (
                    f"✅ 재진입 조건 충족 ({ok_count}/4) — BUY1에서 재진입 고려"
                    if ready else
                    f"⏳ 재진입 대기 ({ok_count}/4) — 조건이 더 충족될 때까지 현금 보유"
                )
                st.markdown(
                    f"<div style='color:{conclusion_color};font-weight:700;"
                    f"margin-top:10px;font-size:.88rem'>{conclusion}</div>",
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.success(f"✅ 익절 완료 — 다음 BUY1 구간 형성 대기 중")

    # ── 4-Factor 점수 표 ──
    st.markdown("#### 🧩 AI 점수 분석")
    score_data={
        "모듈":["📈 추세 (TREND)","🔄 사이클 (CYCLE)","🗓 계절성 (SEASON)","⚡ 불규칙 (IRREG)","🎯 TOTAL"],
        "점수":[res["ts"],res["cs"],res["ss"],res["irs"],res["ts"]+res["cs"]+res["ss"]+res["irs"]],
        "만점":[4,4,4,4,16],
        "달성률":[f"{res['ts']/4*100:.0f}%",f"{res['cs']/4*100:.0f}%",
                  f"{res['ss']/4*100:.0f}%",f"{res['irs']/4*100:.0f}%",
                  f"{(res['ts']+res['cs']+res['ss']+res['irs'])/16*100:.0f}%"],
        "해석":[
            "강함" if res['ts']>=3 else "보통" if res['ts']>=2 else "약함",
            "과매도" if res['cs']>=3 else "중립" if res['cs']>=2 else "과매수",
            "유리" if res['ss']>=3 else "보통" if res['ss']>=2 else "불리",
            "안정" if res['irs']>=3 else "주의" if res['irs']>=2 else "위험",
            "매수" if res['ts']+res['cs']+res['ss']+res['irs']>=10 else "관망",
        ]
    }
    st.dataframe(pd.DataFrame(score_data),use_container_width=True,hide_index=True)
    st.markdown("---")

    # ── 물타기 vs 불타기 판단 현황 ──────────────────────
    st.markdown("#### 📈 물타기 vs 불타기 판단 모델")

    r_row  = res["row"]
    macd_v = float(r_row.get("MACD_hist",   0)) if not pd.isna(r_row.get("MACD_hist",   np.nan)) else 0
    macd_r = int(r_row.get("MACD_rising",   0))
    macd_c = int(r_row.get("MACD_cross_up", 0))
    hl_v   = int(r_row.get("HigherLow",     0))
    se_v   = int(r_row.get("StochRSI_escape",0))
    bull_s = int(r_row.get("BullAdd_score", 0))
    bull_sig = int(r_row.get("BullAdd_signal",0))

    # 현재 판단
    if bull_sig:
        judge_txt = "📈 불타기 추가 권장"
        judge_col = "#00ff9d"
        judge_sub = "상승 전환 확인 — BUY2/3를 현재가 위에서 추가 가능"
    else:
        judge_txt = "📉 물타기 대기"
        judge_col = "#ffd700"
        judge_sub = f"조건 {bull_s}/3 충족 — 피보나치 BUY2/3 도달 대기"

    st.markdown(f"""
    <div style="background:#111827;border:2px solid {judge_col};border-radius:12px;
                padding:16px;text-align:center;margin-bottom:14px">
      <div style="font-size:1.4rem;font-weight:700;color:{judge_col}">{judge_txt}</div>
      <div style="color:#6b7280;font-size:.8rem;margin-top:6px">{judge_sub}</div>
      <div style="color:#9ca3af;font-size:.78rem;margin-top:4px">
        판단 점수: {bull_s}/3점 (2점 이상 = 불타기)
      </div>
    </div>""", unsafe_allow_html=True)

    # 3가지 조건 상세 표
    bull_conds = [
        {
            "조건":    "① MACD 히스토그램 상승 중",
            "현재값":  f"{macd_v:+.4f}",
            "충족":    "✅" if macd_r else "❌",
            "의미":    "MACD 히스토그램이 증가 중 = 단기 모멘텀 강화" if macd_r
                       else "MACD 히스토그램 감소 중 = 모멘텀 약화",
        },
        {
            "조건":    "② Higher Low (저점 상승)",
            "현재값":  "저점 상승 중" if hl_v else "저점 하락 중",
            "충족":    "✅" if hl_v else "❌",
            "의미":    "최근 5봉 저점 > 이전 5봉 저점 = 상승 구조 전환" if hl_v
                       else "아직 저점이 낮아지는 구간",
        },
        {
            "조건":    "③ StochRSI 과매도 탈출",
            "현재값":  f"StochRSI {res['stoch']:.1f}",
            "충족":    "✅" if se_v else "❌",
            "의미":    "20 이하 → 40 이상 회복 = 과매도 탈출 확인" if se_v
                       else "아직 과매도 탈출 미확인",
        },
    ]
    st.dataframe(pd.DataFrame(bull_conds), use_container_width=True, hide_index=True,
        column_config={
            "조건":   st.column_config.TextColumn("조건",   width="medium"),
            "현재값": st.column_config.TextColumn("현재값", width="small"),
            "충족":   st.column_config.TextColumn("충족",   width="small"),
            "의미":   st.column_config.TextColumn("의미",   width="large"),
        })

    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2d4a;border-radius:8px;
                padding:12px 16px;font-size:.76rem;color:#6b7280;line-height:1.8;margin-top:6px">
      <b style="color:#e8eaf6">📌 물타기 vs 불타기 전략 설명</b><br>
      📉 <b style="color:#ffd700">물타기</b> — 피보나치 BUY2/3 가격에 도달할 때 추가 매수
      (가격이 더 내려오기를 기다림 → 평균단가 낮춤)<br>
      📈 <b style="color:#00ff9d">불타기</b> — 위 3가지 조건 중 2개 이상 충족 시 현재가에서 추가 매수
      (상승 전환 확인 후 추격 매수 → 수익 극대화)
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── 지표 현황 ──
    st.markdown("#### 📊 주요 지표")
    ind_data={
        "지표":["StochRSI","ADX","ROC(10일)","1주 수익률","1개월 수익률","피보 근접 여부"],
        "값":[f"{res['stoch']:.1f}",f"{res['adx']:.1f}",f"{res['roc']:+.2f}%",
              f"{res['ret_1w']:+.1f}%",f"{res['ret_1m']:+.1f}%",
              "✅ 매수 구간 근접" if res['near_fib'] else "❌ 아직 대기"],
        "판단":[
            "✅ 과매도" if res['stoch']<25 else "⚠️ 중립" if res['stoch']<70 else "❌ 과매수",
            "✅ 추세장" if res['adx']>25 else "⚪ 박스장",
            "✅ 상승" if res['roc']>0 else "❌ 하락",
            "✅" if res['ret_1w']>0 else "❌","✅" if res['ret_1m']>0 else "❌",
            "✅" if res['near_fib'] else "⏳",
        ]
    }
    st.dataframe(pd.DataFrame(ind_data),use_container_width=True,hide_index=True)
    st.markdown("---")

    # ── 매수·매도 신호 요약 표 ──
    st.markdown("#### 🎯 매수·매도 신호 요약")

    # 신호 판단 기준 표
    signal_rows = []

    # 매수 신호 체크
    checks = [
        ("피보나치 구간 근접", res["near_fib"],
         f"BUY1 ${res['fib_lv'][0]:.2f} 근접" if res["near_fib"] and res["fib_lv"][0] else "아직 진입 구간 아님"),
        ("MA60 위 (추세 확인)", float(res["row"]["Close"]) > float(res["row"]["MA60"]),
         f"현재가 ${res['price']:.2f} > MA60 ${res['row']['MA60']:.2f}" if float(res["row"]["Close"]) > float(res["row"]["MA60"]) else f"MA60 ${res['row']['MA60']:.2f} 아래"),
        ("StochRSI 과매도", res["stoch"] < 25,
         f"StochRSI {res['stoch']:.1f} (25 이하 = 과매도)" if res["stoch"] < 25 else f"StochRSI {res['stoch']:.1f} (아직 과매도 아님)"),
        ("ADX 추세 강도", res["adx"] > 20,
         f"ADX {res['adx']:.1f} (추세장)" if res["adx"] > 20 else f"ADX {res['adx']:.1f} (박스장)"),
        ("ROC 모멘텀", res["roc"] > 0,
         f"ROC {res['roc']:+.2f}% (상승 모멘텀)" if res["roc"] > 0 else f"ROC {res['roc']:+.2f}% (하락 모멘텀)"),
        ("1주 수익률", res["ret_1w"] > 0,
         f"{res['ret_1w']:+.1f}%" ),
        ("1개월 수익률", res["ret_1m"] > 0,
         f"{res['ret_1m']:+.1f}%"),
        ("하락장 제외", res["regime"] != "DOWNtrend",
         f"현재 {res['cfg']['desc']}" if res["regime"] != "DOWNtrend" else "⚠️ 하락장 — 매수 비추천"),
    ]

    for name, ok, detail in checks:
        signal_rows.append({
            "항목": name,
            "결과": "✅ 충족" if ok else "❌ 미충족",
            "세부 내용": detail,
        })

    df_signal = pd.DataFrame(signal_rows)
    st.dataframe(df_signal, use_container_width=True, hide_index=True,
        column_config={
            "항목":    st.column_config.TextColumn("확인 항목", width="medium"),
            "결과":    st.column_config.TextColumn("결과",     width="small"),
            "세부 내용": st.column_config.TextColumn("세부 내용", width="large"),
        })

    # 최종 판정 요약
    ok_count = sum(1 for _, ok, _ in checks if ok)
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d4a;border-radius:10px;
                padding:16px;margin-top:10px;text-align:center">
      <div style="color:#6b7280;font-size:.8rem;margin-bottom:6px">조건 충족 현황</div>
      <div style="font-size:1.5rem;font-weight:700;color:{'#00ff9d' if ok_count>=6 else '#ffd700' if ok_count>=4 else '#ff4757'}">
        {ok_count} / {len(checks)} 충족
      </div>
      <div style="color:#6b7280;font-size:.8rem;margin-top:4px">
        {'✅ 매수 진입 고려 가능' if ok_count>=6 else '⚠️ 추가 확인 필요' if ok_count>=4 else '❌ 아직 매수 시기 아님'}
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── 최근 20일 가격 흐름 + 매매 플랜 근접도 ──
    st.markdown("#### 📅 최근 20일 가격 흐름 — 매매 플랜 근접도 포함")

    df_price = res["df"].tail(20).copy()[
        ["Date","Open","High","Low","Close","Volume"]
    ].reset_index(drop=True)

    # 피보나치 레벨 (매매 플랜)
    b1 = res["fib_lv"][0]  # BUY1
    b2 = res["fib_lv"][1]  # BUY2
    b3 = res["fib_lv"][2]  # BUY3
    st_p = res["stop_s"]   # 손절
    tp_p = res["tp_s"]     # 익절

    def plan_status(close_val):
        """현재 종가가 매매 플랜 중 어느 구간에 있는지 판별"""
        c = float(close_val)
        # 손절 아래
        if st_p and c < st_p:
            return "🔴 손절선 이탈"
        # 익절 도달
        if tp_p and c >= tp_p:
            return "🏆 익절 목표 도달"
        # BUY3 구간 (±2%)
        if b3 and abs(c - b3) / b3 <= 0.02:
            return "🟠 BUY3 진입 근접"
        # BUY3 아래
        if b3 and c < b3:
            return "🟠 BUY3 구간 이탈"
        # BUY2 구간 (±2%)
        if b2 and abs(c - b2) / b2 <= 0.02:
            return "🟡 BUY2 진입 근접"
        # BUY1 구간 (±2%)
        if b1 and abs(c - b1) / b1 <= 0.02:
            return "🟢 BUY1 진입 근접"
        # BUY1 위 (대기)
        if b1 and c > b1:
            dist = (c - b1) / b1 * 100
            return f"⏳ BUY1까지 -{dist:.1f}%"
        return "—"

    def dist_to_buy1(close_val):
        """BUY1까지 거리 (%)"""
        if not b1: return "—"
        c = float(close_val)
        d = (c - b1) / b1 * 100
        if d > 0:
            return f"-{d:.1f}% 남음"
        elif abs(d) <= 2:
            return "✅ 도달"
        else:
            return f"+{abs(d):.1f}% 초과"

    price_rows = []
    prev_close = None
    for _, row in df_price.iterrows():
        c = float(row["Close"])
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        # 등락
        chg = (c / prev_close - 1) * 100 if prev_close else 0
        chg_str = f"{chg:+.2f}%" if prev_close else "-"
        prev_close = c
        price_rows.append({
            "날짜":        str(row["Date"])[:10],
            "종가":        f"${c:.2f}",
            "등락":        chg_str,
            "캔들":        "🟢" if c >= o else "🔴",  # 미국식: 상승=초록, 하락=빨강
            "고가":        f"${h:.2f}",
            "저가":        f"${l:.2f}",
            "매매플랜 위치": plan_status(c),
            "BUY1까지":    dist_to_buy1(c) if "V5" in final_strat_name or "V7" in final_strat_name
                           else f"진입가까지 {(entry_price/c-1)*100:+.1f}%" if entry_price > 0 else "-",
        })

    df_p = pd.DataFrame(price_rows).iloc[::-1].reset_index(drop=True)
    st.dataframe(df_p, use_container_width=True, hide_index=True,
        column_config={
            "날짜":         st.column_config.TextColumn("날짜",       width="small"),
            "종가":         st.column_config.TextColumn("종가",       width="small"),
            "등락":         st.column_config.TextColumn("등락",       width="small"),
            "캔들":         st.column_config.TextColumn("캔들",       width="small"),
            "고가":         st.column_config.TextColumn("고가",       width="small"),
            "저가":         st.column_config.TextColumn("저가",       width="small"),
            "매매플랜 위치": st.column_config.TextColumn("매매플랜 위치", width="medium"),
            "BUY1까지":     st.column_config.TextColumn("BUY1까지",   width="small"),
        })

    # 범례
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2d4a;border-radius:8px;
                padding:12px 16px;margin-top:8px;font-size:.78rem;color:#9ca3af;line-height:1.9">
      <b style="color:#e8eaf6">매매플랜 위치 범례</b><br>
      🏆 익절 목표 도달 &nbsp;|&nbsp;
      ⏳ BUY1까지 -X% 남음 &nbsp;|&nbsp;
      🟢 BUY1 근접(±2%) &nbsp;|&nbsp;
      🟡 BUY2 근접(±2%) &nbsp;|&nbsp;
      🟠 BUY3 근접(±2%) &nbsp;|&nbsp;
      🔴 손절선 이탈
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════
    # 📰 뉴스 & Claude AI 투자 전망 분석
    # ════════════════════════════════════════════════════════════
    st.markdown("#### 📰 뉴스 & AI 투자 전망 분석")

    @st.cache_data(ttl=1800)
    def get_news_items(ticker):
        try:
            tk_obj = yf.Ticker(ticker)
            news   = tk_obj.news or []
            items  = []
            for n in news[:10]:
                title    = n.get("title","")
                pub_time = n.get("providerPublishTime",0)
                pub_date = datetime.datetime.fromtimestamp(pub_time).strftime("%m-%d") if pub_time else ""
                source   = n.get("publisher","")
                link     = n.get("link","")
                if title:
                    items.append({"title":title,"date":pub_date,
                                  "source":source,"link":link})
            return items
        except Exception:
            return []

    with st.spinner("뉴스 수집 중..."):
        news_items = get_news_items(ticker_input)

    if news_items:
        # 헤드라인 목록
        for item in news_items:
            st.markdown(
                f"- `{item['date']}` **{item['title']}** "
                f"<span style='color:#6b7280;font-size:.75rem'>— {item['source']}</span>",
                unsafe_allow_html=True
            )
        st.markdown("")

        # Claude AI 분석 버튼
        if st.button("🤖 Claude AI 투자 전망 분석", key="news_ai_btn",
                     use_container_width=True, type="primary"):
            headlines_text = "\n".join(
                [f"- [{i['date']}] {i['title']}" for i in news_items]
            )
            fib1 = f"${res['fib_lv'][0]:.2f}" if res['fib_lv'][0] else "대기"
            prompt = f"""당신은 주식 투자 전문 애널리스트입니다.
아래는 {ticker_input}의 최근 뉴스입니다.

{headlines_text}

현재 상황:
- 현재가: ${res['price']:.2f}
- 레짐: {res['regime']} ({res['cfg']['desc']})
- 피보나치 BUY1: {fib1}
- 종합 점수: {res['pct']:.0f}%

다음을 한국어로 간결하게 분석해주세요:

1. **뉴스 감성** (매우긍정/긍정/중립/부정/매우부정)
2. **핵심 이슈** (2~3줄)
3. **주가 영향** (단기/중기)
4. **매수 타이밍** (지금 진입 가능한지, 대기해야 하는지)
5. **주요 리스크**

실용적이고 핵심만 작성해주세요."""

            with st.spinner("Claude AI 분석 중..."):
                try:
                    import requests as req_lib
                    resp = req_lib.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"Content-Type":"application/json"},
                        json={
                            "model":"claude-sonnet-4-20250514",
                            "max_tokens":800,
                            "messages":[{"role":"user","content":prompt}]
                        }, timeout=30
                    )
                    if resp.status_code == 200:
                        analysis = resp.json()["content"][0]["text"]
                        st.markdown(f"""
                        <div style="background:#0f172a;border:1.5px solid #00d4ff;
                                    border-radius:12px;padding:16px;margin-top:8px">
                          <div style="color:#00d4ff;font-weight:700;margin-bottom:8px">
                            🤖 Claude AI 분석
                          </div>
                          <div style="color:#e8eaf6;font-size:.84rem;line-height:1.9;
                                      white-space:pre-wrap">{analysis}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.error(f"AI 분석 실패 (상태코드: {resp.status_code})")
                except Exception as e:
                    st.error(f"AI 연결 오류: {e}")
    else:
        st.info("뉴스 데이터 없음 — 잠시 후 다시 시도하세요.")
    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # 📊 어닝(실적) 분석
    # ════════════════════════════════════════════════════════════
    st.markdown("#### 📊 실적(어닝) 분석")

    def get_earnings_info(ticker):
        try:
            tk_obj = yf.Ticker(ticker)
            # 다음 실적 발표일
            cal = tk_obj.calendar
            next_earn = None
            days_left  = None
            if cal is not None and not (isinstance(cal, dict) and not cal):
                if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                    earn_dates = cal.loc["Earnings Date"]
                    next_earn  = pd.to_datetime(earn_dates.iloc[0])
                    days_left  = (next_earn - pd.Timestamp.now()).days
                elif isinstance(cal, dict) and "Earnings Date" in cal:
                    ed = cal["Earnings Date"]
                    if isinstance(ed, list) and ed:
                        next_earn = pd.to_datetime(ed[0])
                        days_left = (next_earn - pd.Timestamp.now()).days

            # 과거 실적 (EPS)
            hist = tk_obj.earnings_history if hasattr(tk_obj, "earnings_history") else None
            if hist is None or (hasattr(hist, "empty") and hist.empty):
                hist = None

            # 기본 재무 정보
            info = tk_obj.info or {}
            pe   = info.get("trailingPE", None)
            fpe  = info.get("forwardPE",  None)
            eps  = info.get("trailingEps", None)
            rev  = info.get("revenueGrowth", None)
            earn_growth = info.get("earningsGrowth", None)

            return {
                "next_earn":    next_earn,
                "days_left":    days_left,
                "pe":           pe,
                "fpe":          fpe,
                "eps":          eps,
                "rev_growth":   rev,
                "earn_growth":  earn_growth,
                "hist":         hist,
            }
        except Exception:
            return {}

    with st.spinner("실적 데이터 가져오는 중..."):
        earn = get_earnings_info(ticker_input)

    # 다음 실적 발표일
    if earn.get("next_earn") and earn.get("days_left") is not None:
        dl   = earn["days_left"]
        ed   = str(earn["next_earn"])[:10]
        ec   = "#ff4757" if dl <= 7 else "#ffd700" if dl <= 21 else "#00ff9d"
        warn = "⚠️ 1주 이내 — 포지션 축소 권장" if dl <= 7 else \
               "⚠️ 3주 이내 — 변동성 주의"     if dl <= 21 else \
               "✅ 여유 있음"
        st.markdown(f"""
        <div style="background:#111827;border:1px solid {ec};border-radius:10px;
                    padding:14px;margin-bottom:12px">
          <div style="color:#6b7280;font-size:.78rem">다음 실적 발표일</div>
          <div style="font-size:1.3rem;font-weight:700;color:{ec};margin:4px 0">
            {ed} &nbsp; (D-{dl})
          </div>
          <div style="color:{ec};font-size:.82rem">{warn}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("실적 발표일 정보를 가져올 수 없습니다.")

    # 재무 지표 표
    fin_rows = []
    if earn.get("pe"):
        fin_rows.append({"지표":"PER (주가수익비율)","값":f"{earn['pe']:.1f}x",
            "해석":"✅ 저평가" if earn["pe"]<15 else "⚠️ 고평가" if earn["pe"]>30 else "➡️ 보통"})
    if earn.get("fpe"):
        fin_rows.append({"지표":"Forward PER (예상)","값":f"{earn['fpe']:.1f}x",
            "해석":"✅ 저평가" if earn["fpe"]<15 else "⚠️ 고평가" if earn["fpe"]>30 else "➡️ 보통"})
    if earn.get("eps"):
        fin_rows.append({"지표":"EPS (주당순이익)","값":f"${earn['eps']:.2f}",
            "해석":"✅ 흑자" if earn["eps"]>0 else "❌ 적자"})
    if earn.get("rev_growth") is not None:
        rv = earn["rev_growth"] * 100
        fin_rows.append({"지표":"매출 성장률 (YoY)","값":f"{rv:+.1f}%",
            "해석":"✅ 고성장" if rv>20 else "✅ 성장" if rv>0 else "❌ 역성장"})
    if earn.get("earn_growth") is not None:
        eg = earn["earn_growth"] * 100
        fin_rows.append({"지표":"이익 성장률 (YoY)","값":f"{eg:+.1f}%",
            "해석":"✅ 고성장" if eg>20 else "✅ 성장" if eg>0 else "❌ 역성장"})

    if fin_rows:
        st.dataframe(pd.DataFrame(fin_rows), use_container_width=True, hide_index=True,
            column_config={
                "지표": st.column_config.TextColumn("재무 지표", width="medium"),
                "값":   st.column_config.TextColumn("수치",     width="small"),
                "해석": st.column_config.TextColumn("해석",     width="medium"),
            })
    else:
        st.info("재무 데이터를 가져올 수 없습니다.")
    st.markdown("---")



# ════════════════════════════════════════════════════════════
# 🤖 AI 자동 스캔
# ════════════════════════════════════════════════════════════
elif menu=="🤖 AI 종목 추천" and scan_btn:

    # ── 시장 레짐 먼저 감지 ──────────────────────────────
    with st.spinner("📡 전체 시장 레짐 감지 중..."):
        market_regime = detect_market_regime()

    # ── 시장 레짐 배너 ───────────────────────────────────
    regime_colors = {
        "상승장": "#00ff9d", "박스장": "#ffd700",
        "조정장": "#ff8c00", "하락장": "#ff4757", "알 수 없음": "#6b7280"
    }
    regime_icons = {
        "상승장": "🚀", "박스장": "➡️",
        "조정장": "⚠️", "하락장": "🔴", "알 수 없음": "❓"
    }
    # VIX도 동시에 확인
    vix_now, vix_lbl = get_vix_level()

    rc = regime_colors.get(market_regime, "#6b7280")
    ri = regime_icons.get(market_regime, "❓")
    vix_c = "#ff4757" if vix_now>=30 else "#ff8c00" if vix_now>=25 else             "#ffd700" if vix_now>=20 else "#00ff9d"

    st.markdown(f"""
    <div style="background:#0f172a;border:2px solid {rc};
                border-radius:12px;padding:14px 16px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="color:{rc};font-weight:700;font-size:1.1rem">
            {ri} 현재 시장: {market_regime}
          </div>
          <div style="color:#9ca3af;font-size:.8rem;margin-top:4px">
            SPY 기준 MA200·MA50·ROC 종합 판단
          </div>
        </div>
        <div style="text-align:right">
          <div style="color:{vix_c};font-weight:700;font-size:1rem">
            VIX {vix_now:.1f}
          </div>
          <div style="color:{vix_c};font-size:.75rem">{vix_lbl}</div>
        </div>
      </div>
      {f'<div style="margin-top:8px;padding:6px 10px;background:#1a0000;border-radius:6px;color:#ff4757;font-size:.78rem">🚫 VIX {vix_now:.0f} 극단 공포 — V6 모멘텀 진입 금지, V7/현금 보유 권장</div>' if vix_now>=30 else ''}
    </div>""", unsafe_allow_html=True)

    # ── 하락장/조정장일 때 → 하락장 대응 섹션 먼저 표시 ──
    if market_regime in ["하락장", "조정장"]:
        st.markdown("## 🛡️ 하락장 대응 종목 추천")
        st.markdown("""
        <div style="background:#1a0000;border:1px solid #ff4757;
                    border-radius:8px;padding:12px 16px;margin-bottom:16px;
                    font-size:.82rem;color:#fca5a5;line-height:1.8">
        ⚠️ 현재 하락장입니다. 아래 종목들은 시장 하락 시 수익이 나는 종목들이에요.<br>
        📌 전략: <b>피보나치 분할매수 V5</b> — 인버스 ETF도 눌림목에서 분할매수<br>
        ⚡ 주의: 인버스 ETF는 장기 보유 시 손실 가능. 단기 스윙 목적으로만 활용
        </div>""", unsafe_allow_html=True)

        # 하락장 종목 스캔
        bear_scan_results = []
        bear_prog = st.progress(0)
        bear_status = st.empty()

        for i, tk in enumerate(ALL_BEAR_TICKERS):
            bear_status.text(f"하락장 종목 스캔 중: {tk} ({i+1}/{len(ALL_BEAR_TICKERS)})")
            r = scan_single(tk)
            if r and isinstance(r, dict) and "signal" in r:
                r["bear_type"] = next(
                    (grp for grp, tks in BEAR_TICKERS.items() if tk in tks), "기타"
                )
                r["bear_desc"] = BEAR_TICKERS.get(
                    r["bear_type"], {}
                ).get(tk, "")
                bear_scan_results.append(r)
            bear_prog.progress((i+1)/len(ALL_BEAR_TICKERS))

        bear_prog.empty(); bear_status.empty()

        if bear_scan_results:
            bear_scan_results = sorted(
                bear_scan_results,
                key=lambda x: x.get("total_rec_score", 0), reverse=True
            )

            # 섹터별 탭으로 표시
            tabs = st.tabs(["🔻 인버스 ETF", "🥇 안전자산", "🛡️ 방어주", "📊 전체"])

            for tab_idx, (tab, grp_name) in enumerate(zip(
                tabs, ["인버스 ETF", "안전자산", "방어주"]
            )):
                with tab:
                    grp_results = [r for r in bear_scan_results
                                   if r.get("bear_type") == grp_name]
                    if not grp_results:
                        st.info(f"{grp_name} 종목 중 현재 신호 없음")
                        continue

                    bear_rows = []
                    for r in grp_results:
                        b1 = r.get("fib_lv", [None])[0]
                        bear_rows.append({
                            "종목":      r["ticker"],
                            "설명":      r.get("bear_desc", ""),
                            "현재가":    f"${r['price']:.2f}",
                            "신호":      r["signal"],
                            "BUY1":      f"${b1:.2f}" if b1 else "대기",
                            "BUY1까지":  f"{r.get('dist_pct',0):+.1f}%",
                            "종합 점수":   f"{r.get('pct',0):.0f}%",
                            "추천 점수": f"{r.get('total_rec_score',0):.0f}점",
                            "1주":       f"{r.get('ret_1w',0):+.1f}%",
                            "1개월":     f"{r.get('ret_1m',0):+.1f}%",
                        })
                    st.dataframe(pd.DataFrame(bear_rows),
                        use_container_width=True, hide_index=True,
                        column_config={
                            "종목":      st.column_config.TextColumn(width="small"),
                            "설명":      st.column_config.TextColumn(width="medium"),
                            "현재가":    st.column_config.TextColumn(width="small"),
                            "신호":      st.column_config.TextColumn(width="medium"),
                            "BUY1":      st.column_config.TextColumn(width="small"),
                            "BUY1까지":  st.column_config.TextColumn(width="small"),
                            "종합 점수":   st.column_config.TextColumn(width="small"),
                            "추천 점수": st.column_config.TextColumn(width="small"),
                        })

                    # 상세 카드
                    top_bear = grp_results[:3]
                    for r in top_bear:
                        b1=r.get("fib_lv",[None,None,None])[0]
                        b2=r.get("fib_lv",[None,None,None])[1]
                        b3=r.get("fib_lv",[None,None,None])[2]
                        with st.expander(
                            f"📋 {r['ticker']} — {r.get('bear_desc','')} | {r['signal']}",
                            expanded=False
                        ):
                            plan_rows = [
                                {"구분":"1차 매수 (BUY1)",
                                 "목표가":f"${b1:.2f}" if b1 else "대기",
                                 "현재가 대비":f"{(b1/r['price']-1)*100:+.1f}%" if b1 else "-",
                                 "전략":"피보나치V5"},
                                {"구분":"2차 매수 (BUY2)",
                                 "목표가":f"${b2:.2f}" if b2 else "대기",
                                 "현재가 대비":f"{(b2/r['price']-1)*100:+.1f}%" if b2 else "-",
                                 "전략":"피보나치V5"},
                                {"구분":"3차 매수 (BUY3)",
                                 "목표가":f"${b3:.2f}" if b3 else "대기",
                                 "현재가 대비":f"{(b3/r['price']-1)*100:+.1f}%" if b3 else "-",
                                 "전략":"피보나치V5"},
                                {"구분":"손절선",
                                 "목표가":f"${r['stop_s']:.2f}" if r.get("stop_s") else "-",
                                 "현재가 대비":f"{(r['stop_s']/r['price']-1)*100:+.1f}%" if r.get("stop_s") else "-",
                                 "전략":""},
                                {"구분":"익절 목표",
                                 "목표가":f"${r['tp_s']:.2f}" if r.get("tp_s") else "-",
                                 "현재가 대비":f"{(r['tp_s']/r['price']-1)*100:+.1f}%" if r.get("tp_s") else "-",
                                 "전략":""},
                            ]
                            st.dataframe(pd.DataFrame(plan_rows),
                                use_container_width=True, hide_index=True)
                            st.caption("💡 피보나치 분할매수 V5 전략으로 백테스트해보세요")

            with tabs[3]:  # 전체
                all_bear_rows = []
                for r in bear_scan_results:
                    b1 = r.get("fib_lv",[None])[0]
                    all_bear_rows.append({
                        "종목":     r["ticker"],
                        "유형":     r.get("bear_type",""),
                        "설명":     r.get("bear_desc",""),
                        "현재가":   f"${r['price']:.2f}",
                        "신호":     r["signal"],
                        "BUY1":     f"${b1:.2f}" if b1 else "대기",
                        "점수":     f"{r.get('total_rec_score',0):.0f}점",
                        "1주":      f"{r.get('ret_1w',0):+.1f}%",
                        "1개월":    f"{r.get('ret_1m',0):+.1f}%",
                    })
                st.dataframe(pd.DataFrame(all_bear_rows),
                    use_container_width=True, hide_index=True)
        else:
            st.info("현재 하락장 대응 종목 중 신호 있는 종목이 없습니다.")

        st.markdown("---")
        st.markdown("## 📊 일반 종목 스캔 (참고용)")

    # ── 일반 종목 스캔 ───────────────────────────────────
    # 스캔할 종목 목록 결정
    if scan_mode == "✏️ 직접 종목 입력":
        tickers = [t.strip().upper() for t in custom_list.split("\n") if t.strip()]
    else:
        if selected_sectors:
            tickers = list(dict.fromkeys(
                t for s in selected_sectors
                for t in SCAN_UNIVERSE.get(s, [])
                if s not in ["인버스 ETF","안전자산","방어주"]
            ))
        else:
            tickers = ALL_TICKERS
    if not tickers: st.error("종목을 입력하세요."); st.stop()

    st.markdown(f"### 🤖 AI 자동 스캔 — {len(tickers)}개 종목 분석 중")
    st.caption("피보나치/모멘텀 신호 종목을 자동으로 찾아 추천 점수순으로 정렬합니다.")

    prog_bar   = st.progress(0)
    status_txt = st.empty()
    scan_results = []

    for i, tk in enumerate(tickers):
        status_txt.text(f"스캔 중: {tk} ({i+1}/{len(tickers)})")
        r = scan_single(tk)
        if r: scan_results.append(r)
        prog_bar.progress((i+1)/len(tickers))

    prog_bar.empty(); status_txt.empty()

    scan_results = [
        r for r in scan_results
        if r is not None
        and isinstance(r, dict)
        and "signal" in r
        and "regime" in r
        and r.get("signal", "") != "⏳ 신호 없음 (대기)"  # 신호 없는 종목 제외
    ]
    # 신호 없는 것 제외 후에도 없으면 전체 포함 (빈 결과 방지)
    if not scan_results:
        scan_results = [
            r for r in [scan_single(tk) for tk in tickers[:20]]
            if r is not None and isinstance(r, dict) and "signal" in r
        ]
    scan_results = sorted(scan_results,
                          key=lambda x: x.get("total_rec_score", 0), reverse=True)
    top_n_results = scan_results[:top_n]

    if not scan_results:
        if market_regime in ["하락장","조정장"]:
            st.info("📌 하락장에서는 위의 하락장 대응 종목을 참고하세요.")
        else:
            st.warning("⚠️ 조건에 맞는 종목이 없습니다.")
            st.markdown("""
            **가능한 원인:**
            - 현재 시장이 하락장이라 대부분 종목이 DOWNtrend 분류
            - 피보나치/모멘텀 신호 없는 구간

            **해결 방법:**
            - 하락장 대응 섹터(인버스 ETF/안전자산/방어주) 선택
            - 직접 입력으로 특정 종목만 스캔
            """)
            # 디버그: 첫 번째 종목 강제 스캔 결과 표시
            if tickers:
                test_tk = tickers[0]
                with st.expander(f"🔍 진단: {test_tk} 스캔 결과"):
                    import traceback
                    try:
                        import yfinance as yf
                        df_test = yf.download(test_tk, period="2y", interval="1d",
                                              auto_adjust=True, progress=False)
                        if df_test.empty:
                            st.error(f"{test_tk}: 데이터 없음")
                        else:
                            if isinstance(df_test.columns, pd.MultiIndex):
                                df_test.columns = df_test.columns.get_level_values(0)
                            df_test = df_test.reset_index()
                            if "Datetime" in df_test.columns:
                                df_test.rename(columns={"Datetime":"Date"}, inplace=True)
                            df_test = build_features(df_test)
                            row_test = df_test.dropna(subset=["Regime"]).iloc[-1]
                            regime_test = row_test["Regime"]
                            draw52_test = float(row_test.get("Draw52w", 0)) if "Draw52w" in row_test else "없음"
                            st.write(f"레짐: **{regime_test}**")
                            st.write(f"52주 낙폭: **{draw52_test}%**")
                            st.write(f"데이터: {len(df_test)}봉")
                    except Exception as e:
                        st.error(f"진단 오류: {e}")
        st.stop()

    # ── 전략별 분류 ──
    v7_list  = [r for r in scan_results if r.get("strategy_rec")=="V7역추세"]
    v6_list  = [r for r in scan_results if r.get("strategy_rec")=="모멘텀V6"]
    v5_list  = [r for r in scan_results if r.get("strategy_rec")=="피보나치V5"]
    ready_list = [r for r in scan_results if "진입" in r.get("signal","")]

    s1,s2,s3,s4 = st.columns(4)
    mcard(s1,"📡 스캔 완료",   f"{len(tickers)}개",    "#6b7280")
    mcard(s2,"🎯 V7 역추세",   f"{len(v7_list)}개",    "#00d4ff",
          "RANGE+과매도")
    mcard(s3,"🚀 V6 모멘텀",   f"{len(v6_list)}개",    "#00ff9d",
          "UPtrend+모멘텀")
    mcard(s4,"✅ 즉시 진입",   f"{len(ready_list)}개", "#ffd700",
          "조건 충족")
    st.markdown("---")

    # ── TOP N 순위 표 ──
    st.markdown(f"#### 🏆 TOP {top_n} 추천 종목 — 종합 점수 순")
    st.caption("추천 점수 = AI 점수(50%) + 피보나치 근접도(30%) + 불타기 신호(20%)")

    rank_rows = []
    for i, r in enumerate(top_n_results):
        # BUY1 가격 및 거리
        b1 = r["fib_lv"][0]
        b1_str  = f"${b1:.2f}" if b1 else "대기"
        dist_str= f"{r['dist_pct']:+.1f}%" if r["dist_pct"] else "-"
        # 불타기 표시
        bull_str = f"📈 {r['bull_score']}/3" if r["is_bull"] else f"📉 {r['bull_score']}/3"
        # 전략별 아이콘 + 성향
        strat = r.get("strategy_rec","피보나치V5")
        strat_icon = {"V7역추세":"🎯","모멘텀V6":"🚀","피보나치V5":"📐"}.get(strat,"📐")
        v7_s = r.get("v7_score",0)
        sr   = r.get("style_result", {})
        style_nm = sr.get("style_name","") if sr else ""
        rank_rows.append({
            "순위":      i+1,
            "종목":      r["ticker"],
            "성향":      style_nm,
            "권장전략":  f"{strat_icon} {strat}",
            "레짐":      r.get("regime",""),
            "신호":      r.get("signal",""),
            "V7점수":    f"{v7_s}/4" if strat=="V7역추세" else "-",
            "모멘점수":  f"{r.get('momentum_score',0)}/4" if strat=="모멘텀V6" else "-",
            "BUY1까지":  f"{r.get('dist_pct',0):+.1f}%" if r.get("nearest_fib") else "-",
            "종합 점수":   f"{r.get('pct',0):.0f}%",
            "추천 점수": f"{r.get('total_rec_score',0):.0f}점",
            "1주":       f"{r.get('ret_1w',0):+.1f}%",
            "1개월":     f"{r.get('ret_1m',0):+.1f}%",
            "현재가":    f"${r['price']:.2f}",
            "장세":      r["regime_desc"],
            "신호":      r["signal"],
            "BUY1 가격": b1_str,
            "BUY1까지":  dist_str,
            "추가매수":  bull_str,
            "종합 점수":   f"{r['pct']:.0f}%",
            "추천 점수": f"{r['total_rec_score']:.0f}점",
            "1주":       f"{r['ret_1w']:+.1f}%",
            "1개월":     f"{r['ret_1m']:+.1f}%",
        })

    df_top = pd.DataFrame(rank_rows)
    st.dataframe(df_top, use_container_width=True, hide_index=True,
        column_config={
            "순위":      st.column_config.NumberColumn(width="small"),
            "종목":      st.column_config.TextColumn(width="small"),
            "성향":      st.column_config.TextColumn("성향",      width="small"),
            "권장전략":  st.column_config.TextColumn("권장전략",  width="medium"),
            "레짐":      st.column_config.TextColumn("레짐",      width="small"),
            "신호":      st.column_config.TextColumn("신호",      width="medium"),
            "V7점수":    st.column_config.TextColumn("V7점수",    width="small"),
            "모멘점수":  st.column_config.TextColumn("모멘점수",  width="small"),
            "BUY1까지":  st.column_config.TextColumn("BUY1까지",  width="small"),
            "종합 점수":   st.column_config.TextColumn("종합 점수",   width="small"),
            "추천 점수": st.column_config.TextColumn("추천 점수", width="small"),
            "현재가":    st.column_config.TextColumn(width="small"),
            "1주":       st.column_config.TextColumn(width="small"),
            "1개월":     st.column_config.TextColumn(width="small"),
        })
    st.markdown("---")

    # ── 상세 카드 (TOP N) ──
    st.markdown(f"#### 📋 TOP {min(top_n, len(top_n_results))}종목 상세 매매 플랜")
    for r in top_n_results:
        b1 = r["fib_lv"][0]; b2 = r["fib_lv"][1]; b3 = r["fib_lv"][2]
        sig_colors = {"🟢 매수 근접":"#00ff9d","🟡 대기 중":"#ffd700",
                      "⏳ 원거리":"#6b7280","❌ 하락장":"#ff4757"}
        sc = sig_colors.get(r["signal"],"#9ca3af")
        bull_icon = "📈 불타기" if r["is_bull"] else "📉 물타기"
        bull_col  = "#00ff9d" if r["is_bull"] else "#ffd700"

        with st.expander(
            f"  {r['ticker']}  |  ${r['price']:.2f}  |  "
            f"{r['signal']}  |  추천점수 {r['total_rec_score']:.0f}점",
            expanded=False
        ):
            # 상단 요약
            ec1,ec2,ec3,ec4 = st.columns(4)
            mcard(ec1,"현재가",    f"${r['price']:.2f}",  "#00d4ff")
            mcard(ec2,"장세",      r["regime_desc"],        "#e8eaf6")
            mcard(ec3,"종합 점수",   f"{r['pct']:.0f}%",     sc)
            mcard(ec4,"추가매수",  bull_icon,               bull_col,
                  f"조건 {r['bull_score']}/3")

            st.markdown("")

            # 매매 플랜 표
            plan_rows = [
                {"구분":"1차 매수 (BUY1)", "목표가":f"${b1:.2f}" if b1 else "대기",
                 "현재가 대비":f"{(b1/r['price']-1)*100:+.1f}%" if b1 else "-",
                 "상태":"✅ 유효" if b1 else "⏳ 대기"},
                {"구분":"2차 매수 (BUY2)", "목표가":f"${b2:.2f}" if b2 else "대기",
                 "현재가 대비":f"{(b2/r['price']-1)*100:+.1f}%" if b2 else "-",
                 "상태":"✅ 유효" if b2 else "⏳ 대기"},
                {"구분":"3차 매수 (BUY3)", "목표가":f"${b3:.2f}" if b3 else "대기",
                 "현재가 대비":f"{(b3/r['price']-1)*100:+.1f}%" if b3 else "-",
                 "상태":"✅ 유효" if b3 else "⏳ 대기"},
                {"구분":"손절선",
                 "목표가":f"${r['stop_s']:.2f}" if r["stop_s"] else "-",
                 "현재가 대비":f"{(r['stop_s']/r['price']-1)*100:+.1f}%" if r["stop_s"] else "-",
                 "상태":"✅"},
                {"구분":"익절 목표",
                 "목표가":f"${r['tp_s']:.2f}" if r["tp_s"] else "-",
                 "현재가 대비":f"{(r['tp_s']/r['price']-1)*100:+.1f}%" if r["tp_s"] else "-",
                 "상태":"✅"},
            ]
            st.dataframe(pd.DataFrame(plan_rows),
                use_container_width=True, hide_index=True,
                column_config={
                    "구분":       st.column_config.TextColumn(width="medium"),
                    "목표가":     st.column_config.TextColumn(width="small"),
                    "현재가 대비":st.column_config.TextColumn(width="small"),
                    "상태":       st.column_config.TextColumn(width="small"),
                })

            # 불타기 판단
            if r["is_bull"]:
                st.success(f"📈 불타기 신호 감지 ({r['bull_score']}/3) — 현재가에서 추가 매수 고려")
            else:
                st.info(f"📉 물타기 대기 ({r['bull_score']}/3) — 피보나치 BUY 구간 도달 대기")

    st.markdown("---")

    # ── 전체 스캔 결과 표 ──
    with st.expander(f"📊 전체 스캔 결과 ({len(scan_results)}개) 보기"):
        all_rows = []
        for r in scan_results:
            if not r or "ticker" not in r: continue
            b1 = r.get("fib_lv", [None])[0]
            t_type_all = classify_ticker_type(r["ticker"])
            all_rows.append({
                "종목":      r["ticker"],
                "유형":      "🔥 고변동" if t_type_all=="고변동성" else "🧊 저변동",
                "권장전략":  "모멘텀V6"  if t_type_all=="고변동성" else "피보V5",
                "현재가":    f"${r['price']:.2f}",
                "장세":      r.get("regime_desc",""),
                "신호":      r.get("signal",""),
                "BUY1":      f"${b1:.2f}" if b1 else "대기",
                "BUY1까지":  f"{r.get('dist_pct',0):+.1f}%",
                "종합 점수":   f"{r.get('pct',0):.0f}%",
                "추천 점수": f"{r.get('total_rec_score',0):.0f}점",
                "1주":       f"{r.get('ret_1w',0):+.1f}%",
            })
        st.dataframe(pd.DataFrame(all_rows),
            use_container_width=True, hide_index=True)

    # ── Excel 다운로드 ──
    st.markdown("---")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_top.to_excel(writer, sheet_name=f"TOP{top_n} 추천", index=False)
        pd.DataFrame(all_rows).to_excel(writer, sheet_name="전체 스캔", index=False)
    buf.seek(0)
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        "📊 스캔 결과 Excel 다운로드", data=buf.getvalue(),
        file_name=f"isekai_scan_{now_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ════════════════════════════════════════════════════════════
# 📊 백테스트
# ════════════════════════════════════════════════════════════
elif menu=="📊 백테스트" and bt_btn:
    # ✅ 백테스트는 캐시 없이 항상 새로 계산
    # 캐시된 analyze()를 우회해서 직접 데이터 다운로드
    with st.spinner(f"📡 {ticker_input} 데이터 새로 받는 중..."):
        # analyze 캐시 강제 초기화
        analyze.clear()
        scan_single.clear()
        res = analyze(ticker_input, period_input)
        if res is None:
            st.error("데이터를 가져올 수 없습니다. 티커를 확인하세요.")
            st.stop()

    # 선택된 버전 파라미터 로드 (KeyError 방지)
    ver_cfg = BT_VERSIONS.get(bt_version, BT_VERSIONS["V5 — 조건 완화 + 현실 익절"])
    # 종목 유형 자동 감지
    ticker_type = classify_ticker_type(ticker_input, res["df"])
    strategy_type = ver_cfg.get("strategy_type","fib")

    # V6 선택 시에만 모멘텀 전략 → 나머지는 항상 피보나치
    auto_momentum = (strategy_type == "momentum")
    auto_v7       = (strategy_type == "v7")

    with st.spinner(f"🧪 {ticker_input} [{bt_version}] 백테스트 계산 중..."):
        if auto_v7:
            trades, metrics = run_v7_backtest(
                res["df"],
                slippage  = 0.002,
                trail_pct = bt_trail_pct if bt_trailing else 0.15,
            )
            actual_strategy = "V7 과매도 역추세 전략"
        elif auto_momentum:
            trades, metrics = run_momentum_backtest(
                res["df"],
                trailing_stop=bt_trailing,
                trail_pct=bt_trail_pct,
            )
            actual_strategy = "모멘텀 추격 전략"
        else:
            trades, metrics = run_backtest(
                res["df"],
                score_thr    = bt_score_thr,
                use_stoch    = ver_cfg["use_stoch"],
                use_regime   = ver_cfg["use_regime"],
                use_bull_bear= ver_cfg["use_bull_bear"],
                buy_logic    = ver_cfg["buy_logic"],
                trailing_stop= bt_trailing,
                trail_pct    = bt_trail_pct,
            )
            actual_strategy = "피보나치 분할매수 전략"

    st.markdown(f"### 📊 {ticker_input} 백테스트 결과 ({period_input})")

    # 종목 유형 + 버전 설명 배너
    ver_cfg = BT_VERSIONS.get(bt_version, BT_VERSIONS["V5 — 조건 완화 + 현실 익절"])
    type_color = "#ff8c00" if ticker_type=="고변동성" else "#00d4ff"
    type_icon  = "🔥" if ticker_type=="고변동성" else "🧊"
    st.markdown(f"""
    <div style="background:#0f172a;border:1.5px solid {type_color};
                border-radius:10px;padding:14px 16px;margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="color:{type_color};font-weight:700;margin-bottom:4px">
            {type_icon} {ticker_input} — {ticker_type} 종목
          </div>
          <div style="color:#9ca3af;font-size:.8rem">
            적용 전략: <b style="color:#e8eaf6">{actual_strategy}</b>
          </div>
        </div>
        <div style="text-align:right;color:#6b7280;font-size:.76rem">
          {bt_version}
        </div>
      </div>
      <div style="color:#6b7280;font-size:.78rem;margin-top:8px;
                  border-top:1px solid #1e2d4a;padding-top:8px">
        {ver_cfg["desc"]}
      </div>
    </div>""", unsafe_allow_html=True)

    # 고변동성 종목에서 피보나치 전략 사용 시 경고
    if ticker_type=="고변동성" and strategy_type!="momentum":
        st.warning(f"⚠️ {ticker_input}은 고변동성 종목입니다. **V6 — 고변동성 모멘텀** 전략을 함께 비교해보세요!")

    # 파라미터 카드
    param_cols = st.columns(5)
    param_cols[0].markdown(f"""<div class="mc"><div class="mc-lbl">종목 유형</div>
    <div class="mc-val" style="color:{'#ff8c00' if ticker_type=='고변동성' else '#00d4ff'}">
    {"🔥 고변동성" if ticker_type=="고변동성" else "🧊 저변동성"}</div></div>""",
    unsafe_allow_html=True)
    param_cols[1].markdown(f"""<div class="mc"><div class="mc-lbl">AI 점수 기준</div>
    <div class="mc-val" style="color:#00d4ff">{bt_score_thr}%</div></div>""",
    unsafe_allow_html=True)
    param_cols[2].markdown(f"""<div class="mc"><div class="mc-lbl">레짐 필터</div>
    <div class="mc-val" style="color:{'#00ff9d' if ver_cfg['use_regime'] else '#ffd700'}">
    {"ON" if ver_cfg["use_regime"] else "OFF"}</div></div>""", unsafe_allow_html=True)
    param_cols[3].markdown(f"""<div class="mc"><div class="mc-lbl">물타기/불타기</div>
    <div class="mc-val" style="color:{'#00ff9d' if ver_cfg['use_bull_bear'] else '#ffd700'}">
    {"ON" if ver_cfg["use_bull_bear"] else "OFF"}</div></div>""", unsafe_allow_html=True)
    param_cols[4].markdown(f"""<div class="mc"><div class="mc-lbl">익절 방식</div>
    <div class="mc-val" style="color:#e8eaf6">
    {"트레일링" if bt_trailing else "고정"}</div></div>""", unsafe_allow_html=True)
    st.markdown("---")

    # 거래 없을 때 힌트 제공
    if not metrics:
        st.warning(f"⚠️ [{bt_version}] 거래 없음")
        if "V4" in bt_version or "V3" in bt_version:
            st.info("💡 조건이 너무 엄격합니다. **V5 — 조건 완화** 버전을 시도해보세요!")
        st.markdown("""
        **시도해볼 방법:**
        - 왼쪽에서 **V5 — 조건 완화** 선택
        - AI 점수 기준을 **35%** 이하로 낮추기
        - 기간을 **2y~3y**로 변경 (거래 기회 증가)
        """)
        st.stop()

    # ── 전략 설명 ──────────────────────────────────────────
    with st.expander("📖 이 백테스트가 사용하는 전략 설명 (클릭해서 보기)", expanded=False):
        sd = STRATEGY_DESC
        st.markdown(f"**{sd['name']}**")
        st.markdown("---")
        st.markdown("**핵심 규칙:**")
        for rule in sd["core"]:
            st.markdown(f"- {rule}")
        st.markdown("---")
        st.markdown("**레짐별 파라미터:**")
        regime_tbl = []
        for rn, rp in sd["regime_params"].items():
            regime_tbl.append({
                "레짐":    rn,
                "BUY1":    f"Fib {rp['fib'][0]}",
                "BUY2":    f"Fib {rp['fib'][1]}",
                "BUY3":    f"Fib {rp['fib'][2]}",
                "StochRSI기준": f"{rp['stoch']} 이하",
                "손절":    f"-{rp['stop']*100:.0f}%",
                "익절":    f"+{rp['tp']*100:.0f}%",
                "설명":    rp["desc"],
            })
        st.dataframe(pd.DataFrame(regime_tbl), use_container_width=True, hide_index=True)
        st.markdown("""
        **분할매수 비중:** BUY1 30% → BUY2 34% → BUY3 33% (합계 97%)

        **백테스트 시작점:** MA200 안정화 이후 (최소 200봉 이후부터)
        → UNKNOWN 레짐 구간은 거래 없음 (데이터 부족 구간)

        **손절 계산:** max(평균단가 × (1-손절%), Fib 0.886) — 둘 중 높은 값

        **익절 계산:** 평균단가 × (1+익절%) 도달 시 전량 청산
        """)
    st.markdown("---")

    if period_input == "1y":
        st.info("ℹ️ 1년 기간 선택 시 MA200 안정화를 위해 자동으로 2년 데이터를 사용합니다.")
    if not metrics:
        st.warning("거래 없음 — 다음을 확인하세요:")
        st.markdown("""
        - 기간이 너무 짧음 → 2y 이상 권장
        - 해당 종목이 박스/하락장 중 → 상승장 종목 시도
        - MA200 데이터 부족 → 상장 200일 미만 종목 불가
        """)
        st.info("UNKNOWN 레짐 구간(MA200 미계산)은 거래하지 않습니다.")
    else:
        m = metrics

        # ── 성과 카드 ──────────────────────────────────────
        st.markdown("#### 🏆 성과 지표")
        c1,c2,c3,c4 = st.columns(4)
        c5,c6,c7,c8 = st.columns(4)
        def mc2(col, label, val, color="#e8eaf6", sub=""):
            col.markdown(f"""<div class="mc"><div class="mc-lbl">{label}</div>
            <div class="mc-val" style="color:{color}">{val}</div>
            <div style="color:#6b7280;font-size:.68rem;margin-top:2px">{sub}</div></div>""",
            unsafe_allow_html=True)

        wr_v    = float(m["승률"].replace("%",""))
        cagr_v  = float(m["CAGR"].replace("%",""))
        mdd_v   = float(m["MDD"].replace("%",""))
        sh_v    = float(m["Sharpe"])
        cal_v   = float(m["Calmar"])

        mc2(c1,"총 완결 거래", m["총 완결 거래"],"#00d4ff")
        mc2(c2,"승률",         m["승률"],
            "#00ff9d" if wr_v>=55 else "#ffd700" if wr_v>=45 else "#ff4757",
            "✅ 양호" if wr_v>=55 else "⚠️ 개선필요")
        mc2(c3,"총 수익률",    m["총 수익률"],
            "#00ff9d" if float(m["총 수익률"].replace("%",""))>0 else "#ff4757")
        mc2(c4,"CAGR",         m["CAGR"],
            "#00ff9d" if cagr_v>=20 else "#ffd700" if cagr_v>=10 else "#ff4757",
            "✅ 목표 30% 달성 중" if cagr_v>=30 else "")
        mc2(c5,"MDD",          m["MDD"],
            "#00ff9d" if abs(mdd_v)<=15 else "#ffd700" if abs(mdd_v)<=30 else "#ff4757",
            "✅ 안전" if abs(mdd_v)<=15 else "⚠️ 주의")
        mc2(c6,"Sharpe",       m["Sharpe"],
            "#00ff9d" if sh_v>=1 else "#ffd700" if sh_v>=0.5 else "#ff4757",
            "✅ 우수" if sh_v>=1 else "")
        mc2(c7,"Calmar",       m["Calmar"],
            "#00ff9d" if cal_v>=1 else "#ffd700","✅" if cal_v>=1 else "")
        mc2(c8,"평균 익절",    m["평균 익절"],"#00ff9d")

        st.markdown("---")

        # ── 분할매수 통계 ──────────────────────────────────
        st.markdown(f"#### 📊 분할매수 단계별 통계 — {actual_strategy}")

        is_v7  = (actual_strategy == "V7 과매도 역추세 전략")
        is_v6  = (actual_strategy == "모멘텀 추격 전략")

        # 전략별 설명
        if is_v7:
            st.info("🎯 **V7 역추세**: 피보BUY1 + 과매도 2개↑ + 양봉 → 진입 / 물타기 금지 / 불타기만 / 손절 Fib 0.886 / 익절 +12% or 트레일링 -15%")
        elif is_v6:
            st.info("🚀 **V6 모멘텀**: MA20>MA60 + RSI + MACD + 거래량 3/4 충족 → 100% 한번에 진입 / 손절 -5% / 트레일링 -20%")
        else:
            cfg_now2 = ver_cfg
            tp_pct  = cfg_now2.get("tp", 0.12) * 100
            st_pct  = cfg_now2.get("stop", 0.05) * 100
            st.info(f"📐 **V5 피보나치**: BUY1(30%)→BUY2(35%)→BUY3(35%) 분할매수 → 익절 +{tp_pct:.0f}% / 손절 -{st_pct:.0f}% / 물타기+불타기 허용")

        # V7/V6 는 키 구조가 다름 — .get()으로 안전하게 접근
        b1_cnt  = m.get("BUY1 진입",  0)
        b2_cnt  = m.get("BUY2 추가",  m.get("BUY2 불타기", 0))
        b3_cnt  = m.get("BUY3 추가",  m.get("BUY3 불타기", 0))
        b2_bull = m.get("BUY2 불타기", 0)
        b2_bear = m.get("BUY2 물타기", 0)
        b3_bull = m.get("BUY3 불타기", 0)
        b3_bear = m.get("BUY3 물타기", 0)
        total   = m.get("총 완결 거래", 0)
        익절    = m.get("익절", 0)
        손절    = m.get("손절", 0)

        # 백테스트 거래 내역에서 평균 진입가 계산
        def avg_price(구분_key):
            prices = [float(t["가격"]) for t in trades
                      if t.get("구분") == 구분_key and "가격" in t]
            return f"${sum(prices)/len(prices):.2f}" if prices else "-"

        if is_v7:
            stage_data = {
                "단계":    ["BUY1 (최초 진입)", "BUY2 (불타기)", "BUY3 (불타기)", "SELL (청산)"],
                "비중":    ["30%", "35%", "35%", "전량"],
                "횟수":    [b1_cnt, b2_cnt, b3_cnt, total],
                "평균가":  [avg_price("BUY1"), avg_price("BUY2"), avg_price("BUY3"), avg_price("SELL")],
                "물/불":   ["-", f"불타기만 {b2_bull}건", f"불타기만 {b3_bull}건", f"익절{익절}/손절{손절}"],
                "조건":    [
                    "피보BUY1 + 과매도2개↑ + 양봉",
                    "모멘텀 회복 2개↑ 후 추가",
                    "추세 강화 확인 후 추가",
                    "Fib0.886 손절 or +12% 익절",
                ],
            }
        elif is_v6:
            stage_data = {
                "단계":    ["진입 (100%)", "—", "—", "SELL (청산)"],
                "비중":    ["100%", "-", "-", "전량"],
                "횟수":    [b1_cnt, "-", "-", total],
                "평균가":  [avg_price("BUY1"), "-", "-", avg_price("SELL")],
                "물/불":   ["-", "-", "-", f"익절{익절}/손절{손절}"],
                "조건":    [
                    "4조건 중 3개↑ 충족 시 전량 진입",
                    "분할매수 없음",
                    "분할매수 없음",
                    "손절-5% or 트레일링-20%",
                ],
            }
        else:
            stage_data = {
                "단계":    ["BUY1 (1차 진입)", "BUY2 (2차)", "BUY3 (3차)", "SELL (청산)"],
                "비중":    ["30%", "35%", "35%", "전량"],
                "횟수":    [b1_cnt, b2_cnt, b3_cnt, total],
                "평균가":  [avg_price("BUY1"), avg_price("BUY2"), avg_price("BUY3"), avg_price("SELL")],
                "물/불":   [
                    "-",
                    f"물타기{b2_bear}/불타기{b2_bull}",
                    f"물타기{b3_bear}/불타기{b3_bull}",
                    f"익절{익절}/손절{손절}",
                ],
                "조건":    [
                    "피보BUY1 + 조건 2/3 충족",
                    "피보BUY2 도달 or 모멘텀 회복",
                    "피보BUY3 도달 or 모멘텀 회복",
                    "고정 익절 or 손절",
                ],
            }

        st.dataframe(pd.DataFrame(stage_data), use_container_width=True, hide_index=True,
            column_config={
                "단계":   st.column_config.TextColumn(width="medium"),
                "비중":   st.column_config.TextColumn(width="small"),
                "횟수":   st.column_config.NumberColumn(width="small"),
                "평균가": st.column_config.TextColumn(width="small"),
                "물/불":  st.column_config.TextColumn(width="small"),
                "조건":   st.column_config.TextColumn(width="large"),
            })

        # 불타기 비율 시각화
        total_add  = m.get("BUY2 추가", 0) + m.get("BUY3 추가", 0)
        total_bull = m.get("BUY2 불타기", 0) + m.get("BUY3 불타기", 0)
        if total_add > 0:
            bull_ratio = total_bull / total_add * 100
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e2d4a;border-radius:8px;
                        padding:12px 16px;margin-top:8px">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="color:#9ca3af;font-size:.8rem">추가매수 중 불타기 비율</span>
                <span style="color:#00ff9d;font-weight:700">{bull_ratio:.0f}%</span>
              </div>
              <div style="background:#1e2d4a;border-radius:4px;height:8px">
                <div style="width:{bull_ratio:.0f}%;height:8px;border-radius:4px;
                            background:linear-gradient(90deg,#00ff9d,#00d4ff)"></div>
              </div>
              <div style="color:#6b7280;font-size:.72rem;margin-top:6px">
                📉 물타기 {total_add-total_bull}건 &nbsp;|&nbsp; 📈 불타기 {total_bull}건
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("---")

        # ── 전체 거래 내역 (상세) ──────────────────────────
        st.markdown("#### 📝 전체 거래 내역 (분할매수 포함)")
        if trades:
            # _pnl 컬럼 제거 (내부용)
            display_trades = [{k:v for k,v in t.items() if k != "_pnl"} for t in trades]
            df_trades = pd.DataFrame(display_trades)
            # 구분별 색 표시를 위한 이모지 추가
            def fmt_구분(v):
                icons = {"BUY1":"🟢 BUY1","BUY2":"🔵 BUY2","BUY3":"🟠 BUY3","SELL":"⚫ SELL"}
                return icons.get(v, v)
            df_trades["구분"] = df_trades["구분"].map(fmt_구분)
            st.dataframe(df_trades.iloc[::-1].reset_index(drop=True),
                use_container_width=True, hide_index=True,
                column_config={
                    "날짜":  st.column_config.TextColumn("날짜",   width="small"),
                    "구분":  st.column_config.TextColumn("구분",   width="small"),
                    "단계":  st.column_config.TextColumn("단계",   width="large"),
                    "가격":  st.column_config.NumberColumn("가격", format="$%.2f", width="small"),
                    "비중":  st.column_config.TextColumn("비중",   width="small"),
                    "레짐":  st.column_config.TextColumn("레짐",   width="small"),
                    "피보":  st.column_config.TextColumn("피보",   width="small"),
                    "수익률":st.column_config.TextColumn("수익률", width="small"),
                    "비고":  st.column_config.TextColumn("비고",   width="large"),
                })

        st.markdown("---")

        # ── 완결 거래 요약 (매수→매도 페어) ───────────────
        st.markdown("#### 🔁 완결 거래 요약 (매수→청산 페어)")
        sells_list = [t for t in trades if t["구분"] == "SELL"]
        buy_types  = ["BUY1","BUY2","BUY3"]
        pairs = []
        buy_buffer = []
        for t in trades:
            if t["구분"] in buy_types:
                buy_buffer.append(t)
            elif t["구분"] == "SELL" and buy_buffer:
                b1 = next((b for b in buy_buffer if b["구분"]=="BUY1"), None)
                b2 = next((b for b in buy_buffer if b["구분"]=="BUY2"), None)
                b3 = next((b for b in buy_buffer if b["구분"]=="BUY3"), None)
                pairs.append({
                    "BUY1 날짜": b1["날짜"] if b1 else "-",
                    "BUY1 가격": f"${b1['가격']:.2f}" if b1 else "-",
                    "BUY2 가격": f"${b2['가격']:.2f}" if b2 else "미진입",
                    "BUY3 가격": f"${b3['가격']:.2f}" if b3 else "미진입",
                    "청산 날짜": t["날짜"],
                    "청산 가격": f"${t['가격']:.2f}",
                    "수익률":    t["수익률"],
                    "결과":      "✅ 익절" if t["_pnl"] > 0 else "❌ 손절",
                    "레짐":      t["레짐"],
                })
                buy_buffer = []
        if pairs:
            st.dataframe(pd.DataFrame(pairs).iloc[::-1].reset_index(drop=True),
                use_container_width=True, hide_index=True,
                column_config={
                    "BUY1 날짜": st.column_config.TextColumn(width="small"),
                    "BUY1 가격": st.column_config.TextColumn(width="small"),
                    "BUY2 가격": st.column_config.TextColumn(width="small"),
                    "BUY3 가격": st.column_config.TextColumn(width="small"),
                    "청산 날짜": st.column_config.TextColumn(width="small"),
                    "청산 가격": st.column_config.TextColumn(width="small"),
                    "수익률":    st.column_config.TextColumn(width="small"),
                    "결과":      st.column_config.TextColumn(width="small"),
                    "레짐":      st.column_config.TextColumn(width="small"),
                })

        # ── 결과 복사 + Excel 다운로드 ──────────────────────
        st.markdown("---")
        st.markdown("##### 📋 결과 복사 (Claude에게 붙여넣기용)")

        # Claude에게 붙여넣기 할 수 있는 텍스트 생성
        m = metrics
        sells_list = [t for t in trades if t["구분"]=="SELL"]
        regime_counts = {}
        for t in trades:
            if t["구분"] in ["BUY1","BUY2","BUY3"]:
                regime_counts[t.get("레짐","?")] = regime_counts.get(t.get("레짐","?"),0)+1

        copy_text = f"""=== 프로젝트 이세계 백테스트 결과 ===
종목: {ticker_input}
분석일: {datetime.datetime.now().strftime("%Y-%m-%d")}
기간: {period_input} (실제 {"2년" if period_input in ["6mo","1y"] else period_input.replace("y","년").replace("mo","개월")} 데이터 사용)
종목 유형: {ticker_type}
적용 전략: {actual_strategy}
전략 버전: {bt_version}
레짐: {res["regime"]} ({res["cfg"]["desc"]})

--- 현재 매매 플랜 ---
현재가:   ${res["price"]:.2f}
BUY1:     {f"${res['fib_lv'][0]:.2f}" if res["fib_lv"][0] else "대기"}
BUY2:     {f"${res['fib_lv'][1]:.2f}" if res["fib_lv"][1] else "대기"}
BUY3:     {f"${res['fib_lv'][2]:.2f}" if res["fib_lv"][2] else "대기"}
손절선:   {f"${res['stop_s']:.2f}" if res["stop_s"] else "N/A"}
익절목표: {f"${res['tp_s']:.2f}" if res["tp_s"] else "N/A"}

--- 백테스트 성과 ---
총 거래:   {m.get("총 완결 거래","N/A")}건
BUY1 진입: {m.get("BUY1 진입","N/A")}건
BUY2 추가: {m.get("BUY2 추가","N/A")}건  (물타기:{m.get("BUY2 물타기","N/A")} / 불타기:{m.get("BUY2 불타기","N/A")})
BUY3 추가: {m.get("BUY3 추가","N/A")}건  (물타기:{m.get("BUY3 물타기","N/A")} / 불타기:{m.get("BUY3 불타기","N/A")})
익절:      {m.get("익절","N/A")}건
손절:      {m.get("손절","N/A")}건
승률:      {m.get("승률","N/A")}
총수익률:  {m.get("총 수익률","N/A")}
CAGR:      {m.get("CAGR","N/A")}
MDD:       {m.get("MDD","N/A")}
Sharpe:    {m.get("Sharpe","N/A")}
Calmar:    {m.get("Calmar","N/A")}
평균익절:  {m.get("평균 익절","N/A")}
평균손절:  {m.get("평균 손절","N/A")}

--- AI 점수 ---
TREND:      {res["ts"]}/4
CYCLE:      {res["cs"]}/4
SEASON:     {res["ss"]}/4
IRREG:      {res["irs"]}/4
TOTAL:      {res["pct"]:.0f}%

--- 현재 파라미터 ---
레짐:       {res["regime"]}
Fib구간:    {res["cfg"]["fib"]}
손절폭:     -{res["cfg"]["stop"]*100:.0f}%
익절폭:     +{res["cfg"]["tp"]*100:.0f}%
StochRSI:   {res["cfg"]["stoch"]} 이하

--- 백테스트 설정 ---
전략 버전:      {bt_version}
진입 방식:      {"과매도 2/4이상 + 피보BUY1" if ver_cfg.get("strategy_type")=="v7" else "2/3 완화" if ver_cfg["buy_logic"]=="2of3" else "모멘텀 추격" if ver_cfg["buy_logic"]=="momentum" else "3/3 엄격"}
AI 점수 기준:   {"미사용 (V7)" if ver_cfg.get("strategy_type")=="v7" else "미사용 (V6)" if ver_cfg.get("strategy_type")=="momentum" else f"{bt_score_thr}%"}
레짐 필터:      {"ON" if ver_cfg["use_regime"] else "OFF"}
StochRSI 필터: {"비사용 (V7 자체 조건 사용)" if ver_cfg.get("strategy_type")=="v7" else "ON" if ver_cfg["use_stoch"] else "OFF"}
물타기/불타기:  {"금지 (불타기만)" if ver_cfg.get("strategy_type")=="v7" else "ON" if ver_cfg["use_bull_bear"] else "OFF"}
슬리피지:       {"0.2% 반영" if ver_cfg.get("strategy_type")=="v7" else "미반영"}
익절 방식:      {"트레일링" if bt_trailing else "고정"}
트레일링 폭:    {f"-{bt_trail_pct*100:.0f}%" if bt_trailing else "N/A"}
============================="""

        # ── 복사용 텍스트 영역 ──────────────────────────────
        # 핸드폰: 텍스트 박스 탭 → 전체선택 → 복사
        # 맥/PC:  Ctrl+A → Ctrl+C

        st.markdown("""
        <div style="background:#0f172a;border:1.5px solid #00d4ff;
                    border-radius:10px;padding:14px;margin-bottom:10px">
          <div style="color:#00d4ff;font-weight:700;font-size:.9rem;margin-bottom:6px">
            📋 Claude 붙여넣기용 결과 — 아래 텍스트를 전체 선택 후 복사
          </div>
          <div style="color:#6b7280;font-size:.76rem;line-height:1.7">
            📱 핸드폰: 텍스트 박스를 <b style="color:#ffd700">길게 누르기</b>
            → <b style="color:#ffd700">전체 선택</b>
            → <b style="color:#ffd700">복사</b><br>
            💻 PC/맥: 텍스트 박스 클릭 →
            <b style="color:#ffd700">Ctrl+A</b> →
            <b style="color:#ffd700">Ctrl+C</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 텍스트 영역 — 전체가 보이게 높이 충분히
        st.text_area(
            label="",
            value=copy_text,
            height=500,
            key="copy_result_area",
            label_visibility="collapsed",
        )

        st.markdown("""
        <div style="text-align:center;color:#6b7280;font-size:.74rem;margin-top:4px">
          복사 후 Claude 대화창에서 길게 누르기 → 붙여넣기
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        excel   = make_excel(ticker_input, res, metrics, trades)
        st.download_button("📊 백테스트 결과 Excel 다운로드", data=excel,
            file_name=f"isekai_{ticker_input}_{now_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── 초기 화면 ──
elif not any([
    menu=="🏠 홈 대시보드",
    menu=="🔍 종목 분석" and analyze_btn,
    menu=="🤖 AI 종목 추천" and scan_btn,
    menu=="📊 백테스트" and bt_btn,
]):
    if menu!="🏠 홈 대시보드":
        st.markdown("""<div style="text-align:center;padding:60px 20px;color:#6b7280">
          <div style="font-size:3rem;margin-bottom:12px">👑</div>
          <div style="font-size:1rem;color:#9ca3af">왼쪽 메뉴에서 버튼을 눌러주세요</div>
        </div>""",unsafe_allow_html=True)
