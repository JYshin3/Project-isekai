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
    "UPtrend":   {"fib":[0.382,0.500,0.618],"stoch":30,"stop":0.07,"tp":0.30,"desc":"📈 상승장"},
    "RANGE":     {"fib":[0.500,0.618,0.786],"stoch":20,"stop":0.05,"tp":0.20,"desc":"➡️ 박스장"},
    "DOWNtrend": {"fib":[0.618,0.786,0.886],"stoch":15,"stop":0.05,"tp":0.15,"desc":"📉 하락장"},
    "UNKNOWN":   {"fib":[0.500,0.618,0.786],"stoch":20,"stop":0.07,"tp":0.25,"desc":"❓ 불명"},
}

# 미국 대형주 추천 풀 (자동 스캔용)
DEFAULT_WATCHLIST = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","AVGO","ORCL",
    "CRM","NFLX","ADBE","QCOM","INTC","MU","NOW","SNPS","KLAC","AMAT",
    "JPM","GS","BAC","V","MA","BRK-B","UNH","LLY","JNJ","PFE",
    "XOM","CVX","COP","EOG","SLB","NEE","DUK","SO","D","AEP",
    "SPY","QQQ","IWM","DIA","VRT","PLTR","SMCI","ARM","MRVL","CRWD"
]

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
    # Regime
    atr_med=df["ATR"].median()
    def classify(row):
        p,m,r,a,d=row["Close"],row["MA200"],row["ROC"],row["ATR"],row["ADX"]
        if any(pd.isna(x) for x in [m,r,a,d]): return "UNKNOWN"
        above=p>m; trend=d>20; highv=a>atr_med
        if above and r>0 and trend: return "UPtrend"
        if (not above) and r<0 and trend and highv: return "DOWNtrend"
        return "RANGE"
    df["Regime"]=df.apply(classify,axis=1)
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

    return df

# ════════════════════════════════════════════════════════════
# 단일 종목 분석 (캐시)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def analyze(ticker, period="1y"):
    try:
        # MA200 계산 + 백테스트에 최소 252봉 필요
        # 1y = 252봉 → START=200 이후 거래 구간 52봉뿐 → 자동 2y 확장
        actual_period = "2y" if period in ["6mo","1y"] else period
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

def run_backtest(df):
    """
    우리 전략 그대로 백테스트:
    - 레짐별 피보나치 구간에서 분할매수 (BUY1/2/3)
    - 레짐별 비중 (30% / 34% / 33%)
    - 레짐별 손절/익절
    - 가중평균단가 기반 손절/익절 계산
    """
    trades = []
    capital = 1.0
    # 포지션 상태
    stage = 0          # 0=없음 1=BUY1진입 2=BUY2추가 3=BUY3추가
    ep = []            # 진입가 리스트
    ew = []            # 진입 비중 리스트
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
        score_ok  = float(pct) >= 50
        no_down   = regime != "DOWNtrend"

        # ── BUY1 진입 ─────────────────────────────────
        if stage == 0 and trend_ok and stoch_ok and score_ok and no_down:
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
        # 불타기 판단 모델 (BUY1 이후 매 봉마다 계산)
        elif stage == 1 and no_down:
            macd_ok    = int(row.get("MACD_rising",    0)) == 1
            higher_low = int(row.get("HigherLow",      0)) == 1
            stoch_esc  = int(row.get("StochRSI_escape",0)) == 1
            bull_score = int(row.get("BullAdd_score",  0))
            is_bull    = bull_score >= 2
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
        elif stage == 2 and no_down:
            macd_ok    = int(row.get("MACD_rising",    0)) == 1
            higher_low = int(row.get("HigherLow",      0)) == 1
            stoch_esc  = int(row.get("StochRSI_escape",0)) == 1
            bull_score = int(row.get("BullAdd_score",  0))
            is_bull    = bull_score >= 2
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

    dates  = pd.to_datetime([t["날짜"] for t in sells])
    n_years = max((dates[-1] - dates[0]).days / 365.25, 0.1)
    cagr    = (equity[-1] ** (1 / n_years) - 1) * 100
    sharpe  = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252)) if np.std(pnls) > 0 else 0
    calmar  = cagr / abs(mdd) if mdd != 0 else 0

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
        period_input=st.selectbox("기간",["6mo","1y","2y","3y"],index=1)
    if menu=="🤖 AI 종목 추천":
        custom_list=st.text_area("스캔 종목 목록",
            value="\n".join(DEFAULT_WATCHLIST[:20]),height=200)
        scan_period=st.selectbox("기간",["6mo","1y"],index=1)
        scan_btn=st.button("🚀 AI 스캔 시작",use_container_width=True,type="primary")
    if menu=="🔍 종목 분석":
        analyze_btn=st.button("🔍 분석하기",use_container_width=True,type="primary")
    if menu=="📊 백테스트":
        bt_btn=st.button("🧪 백테스트 실행",use_container_width=True,type="primary")
    st.markdown("---")
    st.markdown('<div class="warn">⚠️ 참고용 분석입니다.<br>투자 손익은 본인 책임입니다.</div>',
                unsafe_allow_html=True)

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
        df_home=pd.DataFrame(table_data)
        st.dataframe(df_home, use_container_width=True, hide_index=True,
            column_config={
                "종목":st.column_config.TextColumn("종목",width="small"),
                "현재가":st.column_config.TextColumn("현재가",width="small"),
                "1주":st.column_config.TextColumn("1주 수익률",width="small"),
                "1개월":st.column_config.TextColumn("1개월 수익률",width="small"),
                "점수":st.column_config.TextColumn("AI 점수",width="small"),
                "레짐":st.column_config.TextColumn("장세",width="small"),
                "신호":st.column_config.TextColumn("신호",width="medium"),
                "행동":st.column_config.TextColumn("권장 행동",width="small"),
            })
    st.markdown("---")
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
    with st.spinner(f"{ticker_input} 분석 중..."):
        res=analyze(ticker_input,period_input)
    if res is None:
        st.error("데이터를 가져올 수 없습니다. 티커를 확인하세요."); st.stop()

    st.markdown(f"### 🔍 {ticker_input} 분석 결과")

    # ── 핵심 정보 한 줄 ──
    sig_colors={"strong_buy":"#00ff9d","buy":"#4ade80","watch":"#ffd700","hold":"#9ca3af","sell":"#ff4757"}
    sc=sig_colors.get(res["sig_k"],"#9ca3af")

    c1,c2,c3,c4,c5=st.columns(5)
    mcard(c1,"현재가",f"${res['price']:.2f}","#00d4ff")
    mcard(c2,"AI 점수",f"{res['pct']:.0f}%",sc,f"{res['ts']+res['cs']+res['ss']+res['irs']:.0f}/16 모듈")
    mcard(c3,"장세",res["cfg"]["desc"],"#e8eaf6")
    mcard(c4,"신호",res["signal"],sc)
    mcard(c5,"권장 행동",res["action"],sc)
    st.markdown("---")

    # ── 매매 플랜 표 (계산 근거 포함) ──
    st.markdown("#### 📋 매매 플랜")

    # 피보나치 계산 근거 표시
    row_data = res["row"]
    sw_h = float(row_data["sw_high"]) if not pd.isna(row_data["sw_high"]) else None
    sw_l = float(row_data["sw_low"])  if not pd.isna(row_data["sw_low"])  else None
    rng_val = float(row_data["rng"]) if not pd.isna(row_data["rng"]) else None

    if sw_h and sw_l and rng_val:
        fib_basis_ok = sw_h > res["price"]
        basis_color  = "#00ff9d" if fib_basis_ok else "#ff4757"
        basis_msg    = "✅ 정상 (고점 → 되돌림 구간)" if fib_basis_ok else "⚠️ 현재가가 스윙 고점 위 — 피보나치 신뢰도 낮음"
        st.markdown(f"""
        <div style="background:#111827;border:1px solid {basis_color};border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;font-size:.82rem">
          <b style="color:#e8eaf6">📐 피보나치 계산 근거</b><br>
          <span style="color:#6b7280">스윙 고점:</span>
          <span style="color:#00d4ff;font-family:monospace"> ${sw_h:.2f}</span> &nbsp;|&nbsp;
          <span style="color:#6b7280">스윙 저점:</span>
          <span style="color:#7b5ea7;font-family:monospace"> ${sw_l:.2f}</span> &nbsp;|&nbsp;
          <span style="color:#6b7280">범위(Range):</span>
          <span style="color:#ffd700;font-family:monospace"> ${rng_val:.2f}</span> &nbsp;|&nbsp;
          <span style="color:#6b7280">현재가:</span>
          <span style="color:#e8eaf6;font-family:monospace"> ${res["price"]:.2f}</span><br>
          <span style="color:{basis_color}">{basis_msg}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("스윙 고점/저점 계산 불가 — 데이터 부족")

    # 모든 상황에서 매매 플랜 표 표시
    # (피보나치 계산 불가 시에도 상태와 이유를 명확히 표시)
    all_none = all(f is None for f in res["fib_lv"])
    if all_none:
        st.warning("⚠️ 피보나치 매수 대기 구간: 현재가가 최근 스윙 고점 위에 있습니다. 조정 후 진입 구간이 생성됩니다.")

    # 사용된 스윙 정보 (analyze에서 반환된 값 활용)
    sw_h_used   = res.get("sw_h_used",   sw_h)
    rng_used    = res.get("rng_used",    rng_val)

    def fmt_price(f):
        return f"${f:.2f}" if f else "대기 중"
    def fmt_dist(f, price):
        if not f: return "—"
        d = (f/price-1)*100
        return f"{d:+.1f}%"
    def fmt_calc(f_lvl, sh, rng):
        if not sh or not rng: return "-"
        return f"${sh:.2f} - ${rng:.2f}×{f_lvl}"

    plan_data={
        "구분":["1차 매수 (BUY1)","2차 매수 (BUY2)","3차 매수 (BUY3)","손절선","익절 목표"],
        "목표가":[
            fmt_price(res["fib_lv"][0]),
            fmt_price(res["fib_lv"][1]),
            fmt_price(res["fib_lv"][2]),
            fmt_price(res["stop_s"]),
            fmt_price(res["tp_s"]),
        ],
        "현재가 대비":[
            fmt_dist(res["fib_lv"][0], res["price"]),
            fmt_dist(res["fib_lv"][1], res["price"]),
            fmt_dist(res["fib_lv"][2], res["price"]),
            fmt_dist(res["stop_s"],    res["price"]),
            fmt_dist(res["tp_s"],      res["price"]),
        ],
        "계산식":[
            fmt_calc(res["cfg"]["fib"][0], sw_h_used, rng_used),
            fmt_calc(res["cfg"]["fib"][1], sw_h_used, rng_used),
            fmt_calc(res["cfg"]["fib"][2], sw_h_used, rng_used),
            f"평균단가 × {1-res['cfg']['stop']:.2f}",
            f"평균단가 × {1+res['cfg']['tp']:.2f}",
        ],
        "상태":[
            "⏳ 대기" if not res["fib_lv"][0] else "✅ 유효",
            "⏳ 대기" if not res["fib_lv"][1] else "✅ 유효",
            "⏳ 대기" if not res["fib_lv"][2] else "✅ 유효",
            "✅" if res["stop_s"] else "—",
            "✅" if res["tp_s"]   else "—",
        ],
        "설명":[
            f"Fib {res['cfg']['fib'][0]} — 1차 진입 (30%)",
            f"Fib {res['cfg']['fib'][1]} — 2차 추가 (35%)",
            f"Fib {res['cfg']['fib'][2]} — 3차 추가 (35%)",
            f"손절 -{res['cfg']['stop']*100:.0f}% / Fib 0.886 중 높은 값",
            f"익절 +{res['cfg']['tp']*100:.0f}% 전량 청산",
        ]
    }
    st.dataframe(pd.DataFrame(plan_data), use_container_width=True, hide_index=True,
        column_config={
            "구분":        st.column_config.TextColumn("구분",       width="medium"),
            "목표가":      st.column_config.TextColumn("목표가",     width="small"),
            "현재가 대비": st.column_config.TextColumn("현재가 대비",width="small"),
            "계산식":      st.column_config.TextColumn("계산식",     width="medium"),
            "상태":        st.column_config.TextColumn("상태",       width="small"),
            "설명":        st.column_config.TextColumn("설명",       width="large"),
        })

    st.markdown("---")

    # ── 📝 내 매매 기록 입력 ──
    st.markdown("#### 📝 내 매매 기록")
    st.caption("실제로 매수한 가격을 입력하면 현재 수익률과 매매 플랜 대비 평가를 보여줍니다.")

    with st.expander("✏️ 내 매수 기록 입력하기", expanded=False):
        mc1, mc2, mc3 = st.columns(3)
        my_price1 = mc1.number_input("내 매수가 1 ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        my_qty1   = mc1.number_input("수량 1 (주)", min_value=0, value=0, step=1)
        my_price2 = mc2.number_input("내 매수가 2 ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        my_qty2   = mc2.number_input("수량 2 (주)", min_value=0, value=0, step=1)
        my_price3 = mc3.number_input("내 매수가 3 ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
        my_qty3   = mc3.number_input("수량 3 (주)", min_value=0, value=0, step=1)

        my_entries = [(p,q) for p,q in [(my_price1,my_qty1),(my_price2,my_qty2),(my_price3,my_qty3)] if p>0 and q>0]

        if my_entries:
            total_cost = sum(p*q for p,q in my_entries)
            total_qty  = sum(q for _,q in my_entries)
            my_avg     = total_cost / total_qty
            my_pnl_pct = (res["price"] - my_avg) / my_avg * 100
            my_pnl_usd = (res["price"] - my_avg) * total_qty
            pnl_color  = "#00ff9d" if my_pnl_pct > 0 else "#ff4757"

            # 매매 플랜 대비 평가
            if res["fib_lv"][0] and my_avg <= res["fib_lv"][0]:
                plan_eval = "✅ 완벽 — BUY1 이하 진입"
            elif res["fib_lv"][1] and my_avg <= res["fib_lv"][1]:
                plan_eval = "🟡 양호 — BUY2 구간 진입"
            elif res["fib_lv"][2] and my_avg <= res["fib_lv"][2]:
                plan_eval = "🟠 보통 — BUY3 구간 진입"
            else:
                plan_eval = "⚠️ 계획보다 높은 가격에 진입"

            # 손절/익절까지 거리
            stop_dist = f"{(res['stop_s']/my_avg-1)*100:+.1f}%" if res["stop_s"] else "N/A"
            tp_dist   = f"{(res['tp_s']/my_avg-1)*100:+.1f}%"   if res["tp_s"]   else "N/A"

            st.markdown(f"""
            <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:16px;margin-top:12px">
              <div style="font-size:.82rem;color:#6b7280;margin-bottom:10px">📊 내 매매 현황</div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px">
                <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
                  <div style="color:#6b7280;font-size:.7rem">평균단가</div>
                  <div style="color:#00d4ff;font-weight:700;font-size:1.1rem">${my_avg:.2f}</div>
                </div>
                <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
                  <div style="color:#6b7280;font-size:.7rem">현재 수익률</div>
                  <div style="color:{pnl_color};font-weight:700;font-size:1.1rem">{my_pnl_pct:+.2f}%</div>
                </div>
                <div style="background:#111827;border-radius:8px;padding:10px;text-align:center">
                  <div style="color:#6b7280;font-size:.7rem">평가손익</div>
                  <div style="color:{pnl_color};font-weight:700;font-size:1.1rem">${my_pnl_usd:+.2f}</div>
                </div>
              </div>
              <div style="font-size:.82rem;line-height:2">
                <span style="color:#6b7280">총 보유 주수:</span> <b>{total_qty}주</b> &nbsp;|&nbsp;
                <span style="color:#6b7280">총 투자금:</span> <b>${total_cost:.2f}</b><br>
                <span style="color:#6b7280">매매플랜 평가:</span> <b>{plan_eval}</b><br>
                <span style="color:#6b7280">손절까지:</span>
                <b style="color:#ff4757">{stop_dist}</b> &nbsp;|&nbsp;
                <span style="color:#6b7280">익절 목표까지:</span>
                <b style="color:#00ff9d">{tp_dist}</b>
              </div>
            </div>""", unsafe_allow_html=True)

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
            "캔들":        "🟢" if c >= o else "🔴",
            "고가":        f"${h:.2f}",
            "저가":        f"${l:.2f}",
            "매매플랜 위치": plan_status(c),
            "BUY1까지":    dist_to_buy1(c),
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
    # 📰 뉴스 감성 분석
    # ════════════════════════════════════════════════════════════
    st.markdown("#### 📰 실시간 뉴스 감성 분석")

    def get_news_sentiment(ticker):
        try:
            tk_obj = yf.Ticker(ticker)
            news   = tk_obj.news or []
            if not news:
                return 0.0, []
            headlines = [n.get("title","") for n in news[:15]]

            # VADER → TextBlob → 키워드 순 fallback
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                sia    = SentimentIntensityAnalyzer()
                scores = [sia.polarity_scores(h)["compound"] for h in headlines]
            except ImportError:
                try:
                    from textblob import TextBlob
                    scores = [TextBlob(h).sentiment.polarity for h in headlines]
                except ImportError:
                    pos_kw = ["beat","surge","jump","strong","record","growth",
                              "bull","gain","profit","upgrade","buy","rally","up"]
                    neg_kw = ["miss","fall","drop","weak","loss","bear","down",
                              "cut","warn","risk","downgrade","sell","crash"]
                    scores = []
                    for h in headlines:
                        hl = h.lower()
                        s  = sum(1 for w in pos_kw if w in hl) \
                           - sum(1 for w in neg_kw if w in hl)
                        scores.append(max(-1.0, min(1.0, s / 3.0)))

            avg = float(np.mean(scores)) if scores else 0.0
            return avg, list(zip(headlines, scores))
        except Exception:
            return 0.0, []

    with st.spinner("뉴스 가져오는 중..."):
        sentiment, news_list = get_news_sentiment(ticker_input)

    if news_list:
        # 감성 요약 카드
        sent_color = "#00ff9d" if sentiment > 0.1 else \
                     "#ff4757" if sentiment < -0.1 else "#ffd700"
        sent_label = "긍정적 📈" if sentiment > 0.1 else \
                     "부정적 📉" if sentiment < -0.1 else "중립 ➡️"
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d4a;border-radius:10px;
                    padding:14px;text-align:center;margin-bottom:12px">
          <div style="color:#6b7280;font-size:.78rem">평균 뉴스 감성</div>
          <div style="font-size:1.6rem;font-weight:700;color:{sent_color};margin:4px 0">
            {sentiment:+.3f}
          </div>
          <div style="color:{sent_color};font-size:.85rem">{sent_label}</div>
        </div>""", unsafe_allow_html=True)

        # 뉴스 헤드라인 표
        news_rows = []
        for headline, score in news_list:
            sc = float(score)
            news_rows.append({
                "감성":     "📈 긍정" if sc > 0.1 else "📉 부정" if sc < -0.1 else "➡️ 중립",
                "점수":     f"{sc:+.3f}",
                "헤드라인": headline,
            })
        df_news = pd.DataFrame(news_rows)
        st.dataframe(df_news, use_container_width=True, hide_index=True,
            column_config={
                "감성":     st.column_config.TextColumn("감성",     width="small"),
                "점수":     st.column_config.TextColumn("점수",     width="small"),
                "헤드라인": st.column_config.TextColumn("헤드라인", width="large"),
            })
    else:
        st.info("뉴스 데이터를 가져올 수 없습니다.")
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
    # 🎯 XGBoost — BUY1/2/3 도달 확률 예측
    # ════════════════════════════════════════════════════════════
    st.markdown("#### 🎯 BUY 도달 확률 예측 (XGBoost AI)")
    st.caption("현재 시장 상태를 기반으로 향후 5일 안에 각 매수 구간에 도달할 확률을 예측합니다.")

    def build_reach_dataset(df, target_price, horizon=5):
        """
        특정 가격(target_price)에 향후 horizon일 내 도달 여부 라벨 생성
        look-ahead bias 방지: 마지막 horizon행은 학습 제외
        """
        df = df.copy()
        # 미래 horizon일 고가 최대값 → target 터치 여부
        df["future_max"] = df["High"].shift(-1).rolling(horizon).max()
        df["reach"]      = (df["future_max"] >= target_price).astype(int)
        # BUY 레벨까지 거리 (%)
        df["dist_pct"]   = (df["Close"] - target_price) / target_price * 100
        # MA 거리
        df["dist_ma20"]  = (df["Close"] - df["MA20"])  / df["MA20"]  * 100
        df["dist_ma60"]  = (df["Close"] - df["MA60"])  / df["MA60"]  * 100
        df["dist_ma200"] = (df["Close"] - df["MA200"]) / df["MA200"] * 100
        # 변동성 비율
        df["vol_ratio"]  = df["Vol5"] / (df["Vol20"] + 1e-9)
        features = [
            "dist_pct","dist_ma20","dist_ma60","dist_ma200",
            "ATR","ADX","ROC","StochRSI","vol_ratio",
            "TrendScore","CycleScore","SeasonalScore","IrregularScore",
        ]
        df_clean = df.dropna(subset=features + ["reach"])
        # 마지막 horizon행은 라벨 신뢰 불가 → 제외
        df_clean = df_clean.iloc[:-horizon] if len(df_clean) > horizon else df_clean
        if len(df_clean) < 30:
            return None, None, features
        return df_clean[features], df_clean["reach"], features

    def train_reach_model(df, target_price):
        try:
            import xgboost as xgb
        except ImportError:
            return None, "xgboost 미설치"
        X, y, features = build_reach_dataset(df, target_price)
        if X is None or y is None:
            return None, "데이터 부족"
        if y.sum() < 5:
            return None, "도달 샘플 부족 (항상 관망 구간)"
        # 80% 학습 / 20% 검증 (look-ahead 방지)
        split  = int(len(X) * 0.8)
        X_tr, y_tr = X.iloc[:split], y.iloc[:split]
        # 클래스 불균형 보정
        pos   = int(y_tr.sum())
        neg   = len(y_tr) - pos
        ratio = max(1, neg // pos) if pos > 0 else 1
        model = xgb.XGBClassifier(
            max_depth=4, learning_rate=0.07, n_estimators=180,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=ratio,          # ✅ 불균형 보정
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
        )
        model.fit(X_tr, y_tr)
        return model, features

    def predict_reach_prob(model, df, target_price, features):
        df2 = df.copy()
        df2["dist_pct"]   = (df2["Close"] - target_price) / target_price * 100
        df2["dist_ma20"]  = (df2["Close"] - df2["MA20"])  / df2["MA20"]  * 100
        df2["dist_ma60"]  = (df2["Close"] - df2["MA60"])  / df2["MA60"]  * 100
        df2["dist_ma200"] = (df2["Close"] - df2["MA200"]) / df2["MA200"] * 100
        df2["vol_ratio"]  = df2["Vol5"] / (df2["Vol20"] + 1e-9)
        row = df2[features].dropna().iloc[[-1]]
        if row.empty:
            return None
        return float(model.predict_proba(row)[0][1]) * 100

    # BUY1 / BUY2 / BUY3 순서로 예측
    buy_targets = [
        ("BUY1", res["fib_lv"][0], res["cfg"]["fib"][0], "#00ff9d"),
        ("BUY2", res["fib_lv"][1], res["cfg"]["fib"][1], "#ffd700"),
        ("BUY3", res["fib_lv"][2], res["cfg"]["fib"][2], "#ff8c00"),
    ]

    prob_rows = []
    with st.spinner("XGBoost 모델 학습 중 (약 3~5초)..."):
        for label, target_price, fib_lvl, color in buy_targets:
            if target_price is None:
                prob_rows.append({
                    "구분": label, "목표가": "N/A", "도달 확률 (5일)": "-",
                    "예측": "-", "현재가 대비": "-",
                })
                continue
            result_model = train_reach_model(res["df"], target_price)
            if result_model[0] is None:
                prob_rows.append({
                    "구분": label,
                    "목표가": f"${target_price:.2f}",
                    "도달 확률 (5일)": f"({result_model[1]})",
                    "예측": "예측 불가",
                    "현재가 대비": f"{(target_price/res['price']-1)*100:+.1f}%",
                })
                continue
            model, features = result_model
            prob = predict_reach_prob(model, res["df"], target_price, features)
            if prob is None:
                prediction = "계산 오류"
                prob_str   = "-"
            else:
                prob_str   = f"{prob:.1f}%"
                prediction = "🟢 높음" if prob >= 60 else \
                             "🟡 보통" if prob >= 35 else "🔴 낮음"
            prob_rows.append({
                "구분":           label,
                "목표가":         f"${target_price:.2f}",
                "현재가 대비":    f"{(target_price/res['price']-1)*100:+.1f}%",
                "도달 확률 (5일)": prob_str,
                "예측":           prediction,
                "Fib 레벨":       str(fib_lvl),
            })

    # 결과 표
    df_prob = pd.DataFrame(prob_rows)
    st.dataframe(df_prob, use_container_width=True, hide_index=True,
        column_config={
            "구분":           st.column_config.TextColumn("구분",           width="small"),
            "목표가":         st.column_config.TextColumn("목표가",         width="small"),
            "현재가 대비":    st.column_config.TextColumn("현재가 대비",    width="small"),
            "도달 확률 (5일)": st.column_config.TextColumn("도달 확률 (5일)", width="medium"),
            "예측":           st.column_config.TextColumn("예측",           width="small"),
            "Fib 레벨":       st.column_config.TextColumn("Fib 레벨",       width="small"),
        })

    # 시각적 확률 바
    for row in prob_rows:
        if row["도달 확률 (5일)"] in ["-","(데이터 부족)","(도달 샘플 부족 (항상 관망 구간))","예측 불가","계산 오류"]:
            continue
        try:
            pct_val = float(row["도달 확률 (5일)"].replace("%",""))
            bar_color = "#00ff9d" if pct_val>=60 else "#ffd700" if pct_val>=35 else "#ff4757"
            st.markdown(f"""
            <div style="margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;
                          font-size:.8rem;color:#9ca3af;margin-bottom:3px">
                <span>{row['구분']} — {row['목표가']}</span>
                <span style="color:{bar_color};font-weight:700">{row['도달 확률 (5일)']}</span>
              </div>
              <div style="background:#1e2d4a;border-radius:4px;height:8px">
                <div style="width:{min(pct_val,100):.1f}%;height:8px;border-radius:4px;
                            background:{bar_color};transition:width 1s"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        except Exception:
            continue

    st.markdown("""
    <div style="background:#111827;border:1px solid #1e2d4a;border-radius:8px;
                padding:12px 16px;margin-top:10px;font-size:.76rem;color:#6b7280;line-height:1.8">
      <b style="color:#e8eaf6">📌 모델 설명</b><br>
      • 과거 데이터로 XGBoost 학습 → 현재 시장 상태 입력 → 5일 내 도달 확률 출력<br>
      • 🟢 60% 이상: 적극 고려 &nbsp;|&nbsp; 🟡 35~60%: 관망 &nbsp;|&nbsp; 🔴 35% 미만: 진입 비추천<br>
      • 과거 성과가 미래를 보장하지 않습니다. 참고용으로만 활용하세요.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 🤖 AI 종목 추천
# ════════════════════════════════════════════════════════════
elif menu=="🤖 AI 종목 추천" and scan_btn:
    tickers=[t.strip().upper() for t in custom_list.split("\n") if t.strip()]
    if not tickers: st.error("종목을 입력하세요."); st.stop()

    st.markdown(f"### 🤖 AI 종목 추천 — {len(tickers)}개 스캔 중")
    prog=st.progress(0); results=[]

    for i,tk in enumerate(tickers):
        res=analyze(tk,scan_period)
        if res: results.append(res)
        prog.progress((i+1)/len(tickers))

    prog.empty()
    results=sorted(results,key=lambda x:x["pct"],reverse=True)

    # ── 추천 요약 ──
    strong_buys=[r for r in results if r["sig_k"]=="strong_buy"]
    buys=[r for r in results if r["sig_k"]=="buy"]
    sells=[r for r in results if r["sig_k"]=="sell"]

    c1,c2,c3,c4=st.columns(4)
    mcard(c1,"🟢 강력 매수",f"{len(strong_buys)}개","#00ff9d")
    mcard(c2,"🟢 매수",f"{len(buys)}개","#4ade80")
    mcard(c3,"🔴 회피",f"{len(sells)}개","#ff4757")
    mcard(c4,"분석 종목",f"{len(results)}개","#6b7280")
    st.markdown("---")

    # ── 전체 순위 표 ──
    st.markdown("#### 📊 종목 순위 (AI 점수 높은 순)")
    table_rows=[]
    for i,res in enumerate(results):
        sc={"strong_buy":"#00ff9d","buy":"#4ade80","watch":"#ffd700","hold":"#9ca3af","sell":"#ff4757"}
        table_rows.append({
            "순위":i+1,"종목":res["ticker"],
            "현재가":f"${res['price']:.2f}",
            "AI 점수":f"{res['pct']:.0f}%",
            "TREND":f"{res['ts']}/4","CYCLE":f"{res['cs']}/4",
            "SEASON":f"{res['ss']}/4","IRREG":f"{res['irs']}/4",
            "장세":res["cfg"]["desc"],
            "StochRSI":f"{res['stoch']:.1f}",
            "신호":res["signal"],"행동":res["action"],
            "1주":f"{res['ret_1w']:+.1f}%","1개월":f"{res['ret_1m']:+.1f}%",
        })
    df_rank=pd.DataFrame(table_rows)
    st.dataframe(df_rank,use_container_width=True,hide_index=True,
        column_config={
            "순위":st.column_config.NumberColumn(width="small"),
            "종목":st.column_config.TextColumn(width="small"),
            "현재가":st.column_config.TextColumn(width="small"),
            "AI 점수":st.column_config.TextColumn(width="small"),
            "신호":st.column_config.TextColumn(width="medium"),
            "행동":st.column_config.TextColumn(width="small"),
        })
    st.markdown("---")

    # ── TOP 5 강력 매수 종목 상세 ──
    top5=[r for r in results if r["sig_k"] in ["strong_buy","buy"]][:5]
    if top5:
        st.markdown("#### 🏆 TOP 5 매수 후보 — 상세")
        for res in top5:
            with st.expander(f"  {res['ticker']}  |  ${res['price']:.2f}  |  {res['signal']}  |  점수 {res['pct']:.0f}%"):
                col1,col2,col3=st.columns(3)
                with col1:
                    st.markdown("**매수 플랜**")
                    plan={
                        "구분":["1차 매수","2차 매수","3차 매수","손절선","익절 목표"],
                        "가격":[
                            f"${res['fib_lv'][0]:.2f}" if res['fib_lv'][0] else "N/A",
                            f"${res['fib_lv'][1]:.2f}" if res['fib_lv'][1] else "N/A",
                            f"${res['fib_lv'][2]:.2f}" if res['fib_lv'][2] else "N/A",
                            f"${res['stop_s']:.2f}" if res['stop_s'] else "N/A",
                            f"${res['tp_s']:.2f}" if res['tp_s'] else "N/A",
                        ]
                    }
                    st.dataframe(pd.DataFrame(plan),hide_index=True,use_container_width=True)
                with col2:
                    st.markdown("**점수 분석**")
                    scores={"모듈":["TREND","CYCLE","SEASON","IRREG"],
                            "점수":[f"{res['ts']}/4",f"{res['cs']}/4",f"{res['ss']}/4",f"{res['irs']}/4"]}
                    st.dataframe(pd.DataFrame(scores),hide_index=True,use_container_width=True)
                with col3:
                    st.markdown("**지표**")
                    inds={"지표":["StochRSI","ADX","ROC","1주","1개월"],
                          "값":[f"{res['stoch']:.1f}",f"{res['adx']:.1f}",
                                f"{res['roc']:+.1f}%",f"{res['ret_1w']:+.1f}%",f"{res['ret_1m']:+.1f}%"]}
                    st.dataframe(pd.DataFrame(inds),hide_index=True,use_container_width=True)
                st.plotly_chart(draw_chart(res),use_container_width=True)

    # Excel 다운로드
    st.markdown("---")
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        df_rank.to_excel(writer,sheet_name="종목순위",index=False)
        if top5:
            top5_data=[{"종목":r["ticker"],"점수":f"{r['pct']:.0f}%",
                "BUY1":f"${r['fib_lv'][0]:.2f}" if r['fib_lv'][0] else "N/A",
                "손절":f"${r['stop_s']:.2f}" if r['stop_s'] else "N/A",
                "익절":f"${r['tp_s']:.2f}" if r['tp_s'] else "N/A"} for r in top5]
            pd.DataFrame(top5_data).to_excel(writer,sheet_name="TOP5",index=False)
    buf.seek(0)
    now_str=datetime.datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button("📊 추천 결과 Excel 다운로드",data=buf.getvalue(),
        file_name=f"isekai_추천_{now_str}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ════════════════════════════════════════════════════════════
# 📊 백테스트
# ════════════════════════════════════════════════════════════
elif menu=="📊 백테스트" and bt_btn:
    with st.spinner(f"{ticker_input} 백테스트 중... (MA200 안정화 필요로 2y 권장)"):
        res = analyze(ticker_input, period_input)
        if res is None: st.error("데이터 오류"); st.stop()
        trades, metrics = run_backtest(res["df"])

    st.markdown(f"### 📊 {ticker_input} 백테스트 결과 ({period_input})")

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
        st.markdown("#### 📊 분할매수 단계별 통계")
        stage_data = {
            "단계":      ["BUY1 (1차 진입)", "BUY2 (2차)", "BUY3 (3차)", "SELL (청산)"],
            "비중":      ["30%", "35%", "35%", "전량"],
            "전체":      [m["BUY1 진입"], m["BUY2 추가"], m["BUY3 추가"], m["총 완결 거래"]],
            "물타기":    ["-", m["BUY2 물타기"], m["BUY3 물타기"], "-"],
            "불타기":    ["-", m["BUY2 불타기"], m["BUY3 불타기"], "-"],
            "의미":      [
                "피보나치 BUY1 도달 → 1차 진입 (30%)",
                f"물타기{m['BUY2 물타기']}건(Fib 하단) / 불타기{m['BUY2 불타기']}건(상승전환 후 추격)",
                f"물타기{m['BUY3 물타기']}건(Fib 하단) / 불타기{m['BUY3 불타기']}건(상승전환 후 추격)",
                f"익절 {m['익절']}건 / 손절 {m['손절']}건",
            ],
        }
        st.dataframe(pd.DataFrame(stage_data), use_container_width=True, hide_index=True,
            column_config={
                "단계":   st.column_config.TextColumn(width="medium"),
                "비중":   st.column_config.TextColumn(width="small"),
                "전체":   st.column_config.TextColumn(width="small"),
                "물타기": st.column_config.TextColumn(width="small"),
                "불타기": st.column_config.TextColumn(width="small"),
                "의미":   st.column_config.TextColumn(width="large"),
            })

        # 불타기 비율 시각화
        total_add = m["BUY2 추가"] + m["BUY3 추가"]
        total_bull = m["BUY2 불타기"] + m["BUY3 불타기"]
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

        # ── Excel 다운로드 ──────────────────────────────────
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
