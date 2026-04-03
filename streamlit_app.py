# 👑 프로젝트 이세계 V13 — BACKTEST LAB COMPLETE EDITION
# V1(Fib) → V5(Regime) → V8(Weight Opt) → V9(4-Factor) → V11(ML) → V12/V13(NLP)
# 실행: streamlit run streamlit_app.py

import io, base64, datetime, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="프로젝트 이세계 V13", page_icon="👑",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&family=JetBrains+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif;}
.main-title{text-align:center;font-size:2rem;font-weight:700;
  background:linear-gradient(135deg,#00d4ff,#7b5ea7,#00ff9d);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem;}
.sub-title{text-align:center;color:#6b7280;font-size:.83rem;margin-bottom:1rem;}
.mc{background:#111827;border:1px solid #1e2d4a;border-radius:10px;padding:13px 10px;text-align:center;}
.mc-lbl{color:#6b7280;font-size:.7rem;margin-bottom:3px;font-family:'JetBrains Mono',monospace;}
.mc-val{font-size:1.15rem;font-weight:700;}
.sig-buy {background:#0d2b1a;border:2px solid #00ff9d;border-radius:12px;padding:16px;text-align:center;}
.sig-sell{background:#2b0d0d;border:2px solid #ff4757;border-radius:12px;padding:16px;text-align:center;}
.sig-hold{background:#2b2500;border:2px solid #ffd700;border-radius:12px;padding:16px;text-align:center;}
.sig-ban {background:#1a0d0d;border:2px solid #7f1d1d;border-radius:12px;padding:16px;text-align:center;}
.sig-txt-buy {color:#00ff9d;font-size:1.6rem;font-weight:900;}
.sig-txt-sell{color:#ff4757;font-size:1.6rem;font-weight:900;}
.sig-txt-hold{color:#ffd700;font-size:1.6rem;font-weight:900;}
.sig-txt-ban {color:#ef4444;font-size:1.6rem;font-weight:900;}
.badge-up   {background:#0d2b1a;color:#00ff9d;padding:3px 9px;border-radius:20px;font-size:.76rem;font-weight:700;}
.badge-down {background:#2b0d0d;color:#ff4757;padding:3px 9px;border-radius:20px;font-size:.76rem;font-weight:700;}
.badge-range{background:#1a1a0d;color:#ffd700;padding:3px 9px;border-radius:20px;font-size:.76rem;font-weight:700;}
.strat-card{background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:10px;}
.warn{background:#1a1500;border:1px solid rgba(255,215,0,.25);border-radius:8px;
      padding:10px 14px;color:#ffd700;font-size:.76rem;line-height:1.6;}
.bar-wrap{background:#1e2d4a;border-radius:4px;height:5px;margin-top:5px;}
.bar-fill{height:5px;border-radius:4px;}
</style>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 상수
# ════════════════════════════════════════════════════════════
SWING = 30
TP_FINAL = 1.30
REGIME_PARAMS = {
    "UPtrend":   {"fib":[0.382,0.500,0.618],"stoch":30,"stop":0.07,"tp":0.30,"desc":"상승장"},
    "RANGE":     {"fib":[0.500,0.618,0.786],"stoch":20,"stop":0.05,"tp":0.20,"desc":"박스장"},
    "DOWNtrend": {"fib":[0.618,0.786,0.886],"stoch":15,"stop":0.05,"tp":0.15,"desc":"하락장"},
    "UNKNOWN":   {"fib":[0.500,0.618,0.786],"stoch":20,"stop":0.07,"tp":0.25,"desc":"불명"},
}
REGIME_BG = {
    "UPtrend":"rgba(0,220,120,0.08)","RANGE":"rgba(255,210,0,0.08)",
    "DOWNtrend":"rgba(255,60,60,0.08)","UNKNOWN":"rgba(120,120,120,0.04)",
}

# ════════════════════════════════════════════════════════════
# 데이터 다운로드
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close","High","Low","Open","Volume"]).reset_index()
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime":"Date"}, inplace=True)
    return df

# ════════════════════════════════════════════════════════════
# 공통 지표 계산
# ════════════════════════════════════════════════════════════
def add_indicators(df):
    # MA
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA60"]  = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["ROC"]   = df["Close"].pct_change(10)
    # ATR
    pc = df["Close"].shift(1)
    df["TR"] = np.maximum(df["High"]-df["Low"],
               np.maximum((df["High"]-pc).abs(),(df["Low"]-pc).abs()))
    df["ATR"]     = df["TR"].rolling(14).mean()
    df["ATR_sum"] = df["TR"].rolling(14).sum()
    # ADX (수정된 계산)
    pdm_raw = df["High"].diff()
    ndm_raw = -df["Low"].diff()
    df["PDM"] = np.where((pdm_raw>ndm_raw)&(pdm_raw>0), pdm_raw, 0.0)
    df["NDM"] = np.where((ndm_raw>pdm_raw)&(ndm_raw>0), ndm_raw, 0.0)
    atr_s = df["ATR_sum"].replace(0,np.nan)
    df["PDI"] = 100*df["PDM"].rolling(14).sum()/atr_s
    df["NDI"] = 100*df["NDM"].rolling(14).sum()/atr_s
    dx_d = (df["PDI"]+df["NDI"]).replace(0,np.nan)
    df["DX"]  = 100*(df["PDI"]-df["NDI"]).abs()/dx_d
    df["ADX"] = df["DX"].rolling(14).mean()
    # RSI / StochRSI
    delta = df["Close"].diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    df["RSI"] = 100-(100/(1+gain/loss.replace(0,np.nan)))
    rmin=df["RSI"].rolling(14).min(); rmax=df["RSI"].rolling(14).max()
    df["StochRSI"] = (df["RSI"]-rmin)/(rmax-rmin+1e-9)*100
    # 수익률 & 변동성
    df["Return"] = df["Close"].pct_change()
    df["Vol5"]   = df["Return"].rolling(5).std()
    df["Vol20"]  = df["Return"].rolling(20).std()
    # Swing
    df["sw_high"] = df["High"].rolling(SWING).max()
    df["sw_low"]  = df["Low"].rolling(SWING).min()
    df["rng"]     = df["sw_high"]-df["sw_low"]
    # Gradient
    df["MA20_grad"]  = df["MA20"].diff()
    df["ADX_grad"]   = df["ADX"].diff()
    df["ROC_grad"]   = df["ROC"].diff()
    df["Price_grad"] = df["Close"].diff(5)
    # 캔들 패턴
    body=abs(df["Close"]-df["Open"]); total=df["High"]-df["Low"]+1e-9
    lw=df[["Close","Open"]].min(axis=1)-df["Low"]
    uw=df["High"]-df[["Close","Open"]].max(axis=1)
    df["Hammer"]    = ((lw>body*2)&(uw<body*0.5)).astype(int)
    df["InvHammer"] = ((uw>body*2)&(df["Close"]>df["Open"])).astype(int)
    df["BullEngulf"]= ((df["Close"]>df["Open"])&(body/total>0.6)).astype(int)
    df["CandleSignal"] = df["Hammer"]|df["InvHammer"]|df["BullEngulf"]
    return df

# ════════════════════════════════════════════════════════════
# 레짐 분류 (V5)
# ════════════════════════════════════════════════════════════
def add_regime(df):
    atr_med = df["ATR"].median()
    def classify(row):
        p,m,r,a,d=row["Close"],row["MA200"],row["ROC"],row["ATR"],row["ADX"]
        if any(pd.isna(x) for x in [m,r,a,d]): return "UNKNOWN"
        above=p>m; trend=d>20; highv=a>atr_med
        if above and r>0 and trend: return "UPtrend"
        if (not above) and r<0 and trend and highv: return "DOWNtrend"
        return "RANGE"
    df["Regime"] = df.apply(classify,axis=1)
    return df

# ════════════════════════════════════════════════════════════
# 4-Factor Score (V9)
# ════════════════════════════════════════════════════════════
def add_v9_scores(df):
    # 각 지표 z-score 정규화 후 합산
    def zscore(s, win=60):
        m=s.rolling(win).mean(); sd=s.rolling(win).std().replace(0,np.nan)
        return (s-m)/sd

    # TrendScore (0~4)
    df["TrendScore"] = (
        (df["Close"]>df["MA60"]).astype(int)+
        (df["MA20_grad"]>0).astype(int)+
        (df["ROC_grad"]>0).astype(int)+
        (df["ADX_grad"]>0).astype(int)
    )
    # CycleScore (0~4)
    def ar1(x):
        if len(x)<2 or np.std(x)<1e-10: return np.nan
        return float(np.corrcoef(x[:-1],x[1:])[0,1])
    df["AR1"] = df["Return"].rolling(5).apply(ar1,raw=True)
    df["WavePulse"] = ((df["Close"].diff()>0).astype(int)-(df["Close"].diff()<0).astype(int)).rolling(5).sum()
    df["CycleScore"] = (
        (df["StochRSI"]<20).astype(int)+
        (df["AR1"]>0).astype(int)+
        (df["Vol5"]<df["Vol20"]).astype(int)+  # 변동성 축소
        (df["WavePulse"]>0).astype(int)
    )
    # SeasonalScore (0~4)
    df["Month"]=df["Date"].dt.month; df["DOW"]=df["Date"].dt.dayofweek; df["DOY"]=df["Date"].dt.dayofyear
    mb={1:1.10,2:1.02,3:1.05,4:1.04,5:0.97,6:0.96,7:1.03,8:0.95,9:0.92,10:1.00,11:1.08,12:1.12}
    db={0:0.97,1:1.02,2:1.04,3:1.03,4:1.05}
    df["MonthBias"]=df["Month"].map(mb); df["DOWbias"]=df["DOW"].map(db)
    df["SantaRally"]=((df["DOY"]>=350)|(df["DOY"]<=5)).astype(int)
    df["SeasonalScore"]=(
        (df["MonthBias"]>1.0).astype(int)+(df["DOWbias"]>1.0).astype(int)+
        df["SantaRally"].astype(int)+((df["Month"]<=4)|(df["Month"]>=11)).astype(int)
    )
    # IrregularScore (0~4)
    df["VolMA20"]=df["Volume"].rolling(20).mean()
    df["HVI"]=(df["Volume"]/(df["VolMA20"]+1e-9)>1.5).astype(int)
    cr=df["High"]-df["Low"]+1e-9; uw=df["High"]-df[["Close","Open"]].max(axis=1)
    lw2=df[["Close","Open"]].min(axis=1)-df["Low"]
    df["PPI"]=(lw2-uw)/cr
    df["PPI_norm"]=(df["PPI"]-df["PPI"].rolling(20).min())/(df["PPI"].rolling(20).max()-df["PPI"].rolling(20).min()+1e-9)
    df["Outlier"]=(df["Return"].abs()>3*df["Return"].rolling(20).std()).astype(int)
    sv=df["Return"].rolling(5).std(); lv=df["Return"].rolling(20).std().replace(0,np.nan)
    df["VIX_Alert"]=(sv/lv>2).astype(int)
    df["IrregularScore"]=(
        (df["HVI"]==0).astype(int)+(df["Outlier"]==0).astype(int)+
        (df["VIX_Alert"]==0).astype(int)+(df["PPI_norm"]>0.5).astype(int)
    )
    # 통합 점수 (최대 20)
    cond_stoch=(df["StochRSI"]<25).astype(int)
    cond_ma=(df["Close"]>df["MA60"]).astype(int)
    cond_candle=df["CandleSignal"]
    grad_bonus=((df["MA20_grad"]>0).astype(int)+(df["ROC_grad"]>0).astype(int)+
                (df["ADX_grad"]>0).astype(int)+(df["Price_grad"]>0).astype(int)>=3).astype(int)
    df["TotalScore"]=(df["TrendScore"]+df["CycleScore"]+df["SeasonalScore"]+
                      df["IrregularScore"]+cond_stoch+cond_ma+cond_candle+grad_bonus)
    df["ScorePct"]=df["TotalScore"]/20*100
    return df

# ════════════════════════════════════════════════════════════
# 성과 지표 계산
# ════════════════════════════════════════════════════════════
def calc_metrics(trades, label=""):
    if not trades: return {}
    sells=[t for t in trades if t["type"]=="SELL"]
    if not sells: return {}
    wins=[t for t in sells if t["pnl"]>0]; losses=[t for t in sells if t["pnl"]<=0]
    total=len(sells); wr=len(wins)/total*100 if total>0 else 0
    aw=float(np.mean([t["pnl"] for t in wins]))*100 if wins else 0
    al=float(np.mean([t["pnl"] for t in losses]))*100 if losses else 0
    rr=abs(aw/al) if al!=0 else 0
    # 에쿼티
    pnls=np.array([t["pnl"] for t in sells])
    equity=np.cumprod(1+pnls); peak=np.maximum.accumulate(equity)
    dd=(equity-peak)/peak; mdd=float(dd.min()*100)
    # CAGR (실제 날짜 기준)
    dates=pd.to_datetime([t["date"] for t in sells])
    n_years=(dates[-1]-dates[0]).days/365.25 if len(dates)>1 else 1
    cagr=(equity[-1]**(1/n_years)-1)*100 if n_years>0 else 0
    daily_ret=pd.Series(pnls)
    sharpe=float(daily_ret.mean()/daily_ret.std()*np.sqrt(252)) if daily_ret.std()>0 else 0
    calmar=cagr/abs(mdd) if mdd!=0 else 0
    total_ret=float((equity[-1]-1)*100)
    return dict(label=label,total=total,wins=len(wins),losses=len(losses),wr=wr,aw=aw,al=al,rr=rr,
                mdd=mdd,cagr=cagr,sharpe=sharpe,calmar=calmar,total_ret=total_ret,
                equity=equity,dates=dates,pnls=pnls)

# ════════════════════════════════════════════════════════════
# V1 — Fibonacci 전략
# ════════════════════════════════════════════════════════════
def backtest_v1(df):
    """순수 피보나치 눌림목 + MA60 필터"""
    cash=1.0; pos=0; entry=0; trades=[]; START=max(60,SWING)
    for i in range(START,len(df)):
        row=df.iloc[i]; p=float(row["Close"])
        rng=row["rng"]; sh=row["sw_high"]
        if pd.isna(rng) or rng==0 or pd.isna(sh): continue
        F1=sh-rng*0.382; ma60=row["MA60"]
        if pd.isna(ma60): continue
        if pos==0 and p<=F1 and p>float(ma60):
            pos=1; entry=p
            trades.append({"type":"BUY","date":str(row["Date"])[:10],"price":p,"pnl":0})
        if pos==1:
            pnl=(p-entry)/entry
            if pnl>=0.15 or pnl<=-0.07:
                cash*=(1+pnl); pos=0
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl})
    return trades

# ════════════════════════════════════════════════════════════
# V5 — Regime 전환 전략
# ════════════════════════════════════════════════════════════
def backtest_v5(df):
    """레짐별 피보나치 구간 + 손절 자동 변경"""
    cash=1.0; pos=0; entry=0; trades=[]; START=max(60,SWING)
    for i in range(START,len(df)):
        row=df.iloc[i]; p=float(row["Close"])
        regime=row["Regime"]; cfg=REGIME_PARAMS[regime]
        rng=row["rng"]; sh=row["sw_high"]
        if pd.isna(rng) or rng==0 or pd.isna(sh): continue
        F1=sh-rng*cfg["fib"][0]; ma60=row["MA60"]
        stoch=row["StochRSI"]
        if any(pd.isna(x) for x in [ma60,stoch]): continue
        if pos==0 and p<=F1 and p>float(ma60) and stoch<cfg["stoch"] and regime!="DOWNtrend":
            pos=1; entry=p
            trades.append({"type":"BUY","date":str(row["Date"])[:10],"price":p,"pnl":0,"regime":regime})
        if pos==1:
            pnl=(p-entry)/entry
            if pnl>=cfg["tp"] or pnl<=-cfg["stop"]:
                cash*=(1+pnl); pos=0
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl,"regime":regime})
    return trades

# ════════════════════════════════════════════════════════════
# V8 — 비중 최적화 (Grid Search)
# ════════════════════════════════════════════════════════════
def simulate_3stage(df_r, weights, regime):
    cfg=REGIME_PARAMS[regime]; fibs=cfg["fib"]; sthr=cfg["stoch"]; stop_r=cfg["stop"]
    w1,w2,w3=weights; stage=0; ep=[]; eq=[]; capital=1.0; curve=[]; trades=[]
    START=max(60,SWING)
    for i in range(START,len(df_r)):
        row=df_r.iloc[i]; p=float(row["Close"])
        ma60=row["MA60"]; stoch=row["StochRSI"]; rng=row["rng"]; sh=row["sw_high"]
        if any(pd.isna(x) for x in [rng,sh,stoch,ma60]) or rng==0:
            curve.append(capital); continue
        fib_lv=[sh-rng*f for f in fibs]; fib886=sh-rng*0.886; trend_ok=p>float(ma60)
        if stage==0 and trend_ok and p<=fib_lv[0] and stoch<sthr:
            stage=1; ep,eq=[p],[w1]
        elif stage==1 and trend_ok and p<=fib_lv[1]:
            stage=2; ep.append(p); eq.append(w2)
        elif stage==2 and trend_ok and p<=fib_lv[2]:
            stage=3; ep.append(p); eq.append(w3)
        elif stage>0:
            tq=sum(eq); avg=sum(pi*qi for pi,qi in zip(ep,eq))/tq
            stop_p=max(avg*(1-stop_r),fib886)
            if p<stop_p:
                pnl=(p-avg)/avg; capital*=(1+pnl*tq)
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl,"regime":regime})
                stage=0; ep=[]; eq=[]
            elif p>=avg*TP_FINAL:
                pnl=(p-avg)/avg; capital*=(1+pnl*tq)
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl,"regime":regime})
                stage=0; ep=[]; eq=[]
        curve.append(capital)
    if len(curve)<2: return 0.0,0.0,trades
    arr=np.array(curve); peak=np.maximum.accumulate(arr)
    mdd=float(((arr-peak)/peak).min()); ret=float(arr[-1]-1.0)
    return ret,mdd,trades

def optimize_weights(df,regime):
    df_r=df[df["Regime"]==regime].copy().reset_index(drop=True)
    if len(df_r)<150: return (0.33,0.34,0.33)
    best_score=-np.inf; best_w=(0.33,0.34,0.33)
    for w1 in np.arange(0.1,0.71,0.1):
        for w2 in np.arange(0.1,0.71,0.1):
            w3=round(1.0-w1-w2,10)
            if w3<=0.05: continue
            raw=np.array([w1,w2,w3]); ws=tuple(raw/raw.sum())
            ret,mdd,_=simulate_3stage(df_r,ws,regime)
            if mdd==0: continue
            score=ret/abs(mdd)
            if score>best_score: best_score=score; best_w=ws
    return best_w

def backtest_v8(df, progress_cb=None):
    optimal={}
    for i,rn in enumerate(["UPtrend","RANGE","DOWNtrend"]):
        optimal[rn]=optimize_weights(df,rn)
        if progress_cb: progress_cb((i+1)/3)
    optimal["UNKNOWN"]=(0.33,0.34,0.33)
    all_trades=[]
    for rn in ["UPtrend","RANGE","DOWNtrend"]:
        df_r=df[df["Regime"]==rn].copy().reset_index(drop=True)
        if len(df_r)<60: continue
        _,_,trades=simulate_3stage(df_r,optimal[rn],rn)
        all_trades.extend(trades)
    return all_trades, optimal

# ════════════════════════════════════════════════════════════
# V9 — 4-Factor Score 기반 전략
# ════════════════════════════════════════════════════════════
def backtest_v9(df):
    """TotalScore 기반 매수·익절·손절"""
    cash=1.0; pos=0; entry=0; trades=[]; regime_entry="UNKNOWN"; START=max(60,SWING)
    for i in range(START,len(df)):
        row=df.iloc[i]; p=float(row["Close"])
        score=row["ScorePct"]; regime=row["Regime"]; cfg=REGIME_PARAMS[regime]
        ma60=row["MA60"]
        if pd.isna(ma60) or pd.isna(score): continue
        # 매수: 점수 45% 이상 + MA60 위 + 하락장 제외
        if pos==0 and score>=45 and p>float(ma60) and regime!="DOWNtrend":
            pos=1; entry=p; regime_entry=regime
            trades.append({"type":"BUY","date":str(row["Date"])[:10],"price":p,"pnl":0,"regime":regime,"score":f"{score:.0f}%"})
        if pos==1:
            pnl=(p-entry)/entry
            cfg_exit=REGIME_PARAMS[regime_entry]
            if pnl>=cfg_exit["tp"] or pnl<=-cfg_exit["stop"]:
                cash*=(1+pnl); pos=0
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl,"regime":regime})
    return trades

# ════════════════════════════════════════════════════════════
# V11 — ML 기반 (RandomForest 경량)
# ════════════════════════════════════════════════════════════
def backtest_v11(df):
    """RandomForest로 다음 봉 방향 예측 후 매매"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        st.warning("scikit-learn 미설치 → V9 방식으로 대체합니다.")
        return backtest_v9(df)

    features=["TrendScore","CycleScore","SeasonalScore","IrregularScore",
               "StochRSI","ADX","ROC","Vol5","RSI"]
    df_ml=df.dropna(subset=features+["Return"]).copy()
    df_ml["Label"]=(df_ml["Return"].shift(-1)>0).astype(int)  # 다음 봉 상승 = 1
    df_ml=df_ml.dropna(subset=["Label"])

    split=int(len(df_ml)*0.7)
    X_tr=df_ml[features].iloc[:split]; y_tr=df_ml["Label"].iloc[:split]
    X_te=df_ml[features].iloc[split:]; y_te=df_ml["Label"].iloc[split:]

    scaler=StandardScaler(); X_tr_s=scaler.fit_transform(X_tr); X_te_s=scaler.transform(X_te)
    clf=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1)
    clf.fit(X_tr_s,y_tr)

    df_ml["MLSignal"]=0
    df_ml.iloc[split:,df_ml.columns.get_loc("MLSignal")]=clf.predict(X_te_s)

    # 백테스트 (테스트셋만)
    cash=1.0; pos=0; entry=0; trades=[]; regime_entry="UNKNOWN"
    test_df=df_ml.iloc[split:].reset_index(drop=True)
    for i in range(len(test_df)):
        row=test_df.iloc[i]; p=float(row["Close"]); regime=row["Regime"]
        cfg=REGIME_PARAMS[regime]; ma60=row["MA60"]
        if pd.isna(ma60): continue
        if pos==0 and row["MLSignal"]==1 and p>float(ma60) and regime!="DOWNtrend":
            pos=1; entry=p; regime_entry=regime
            trades.append({"type":"BUY","date":str(row["Date"])[:10],"price":p,"pnl":0,"regime":regime})
        if pos==1:
            pnl=(p-entry)/entry; cfg_exit=REGIME_PARAMS[regime_entry]
            if pnl>=cfg_exit["tp"] or pnl<=-cfg_exit["stop"]:
                cash*=(1+pnl); pos=0
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl,"regime":regime})

    # 피처 중요도 반환
    importances=pd.Series(clf.feature_importances_,index=features).sort_values(ascending=False)
    return trades, importances

# ════════════════════════════════════════════════════════════
# V12 — 뉴스 감성 (FinVADER / 경량 TextBlob fallback)
# ════════════════════════════════════════════════════════════
def get_news_sentiment(ticker):
    """yfinance 뉴스 헤드라인 감성 분석"""
    try:
        import yfinance as yf2
        tk_obj=yf2.Ticker(ticker)
        news=tk_obj.news
        if not news: return 0.0, []
        headlines=[n.get("title","") for n in news[:20]]
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia=SentimentIntensityAnalyzer()
            scores=[sia.polarity_scores(h)["compound"] for h in headlines]
        except ImportError:
            try:
                from textblob import TextBlob
                scores=[TextBlob(h).sentiment.polarity for h in headlines]
            except ImportError:
                # 완전 fallback: 단순 키워드 카운트
                pos_kw=["beat","surge","jump","strong","record","growth","bull","up","gain","profit"]
                neg_kw=["miss","fall","drop","weak","loss","bear","down","cut","warn","risk"]
                scores=[]
                for h in headlines:
                    hl=h.lower()
                    s=sum(1 for w in pos_kw if w in hl)-sum(1 for w in neg_kw if w in hl)
                    scores.append(max(-1,min(1,s/3)))
        avg_score=float(np.mean(scores)) if scores else 0.0
        return avg_score, list(zip(headlines,scores))
    except Exception:
        return 0.0, []

def backtest_v12(df, ticker):
    """V9 기반 + 뉴스 감성 가중치 조정"""
    sentiment, news_list = get_news_sentiment(ticker)
    # 감성이 부정(-0.2 이하)이면 매수 기준 올림, 긍정(0.2 이상)이면 낮춤
    score_threshold = 45 - sentiment*15  # 범위 30~60
    cash=1.0; pos=0; entry=0; trades=[]; regime_entry="UNKNOWN"; START=max(60,SWING)
    for i in range(START,len(df)):
        row=df.iloc[i]; p=float(row["Close"])
        score=row["ScorePct"]; regime=row["Regime"]; ma60=row["MA60"]
        if pd.isna(ma60) or pd.isna(score): continue
        if pos==0 and score>=score_threshold and p>float(ma60) and regime!="DOWNtrend":
            pos=1; entry=p; regime_entry=regime
            trades.append({"type":"BUY","date":str(row["Date"])[:10],"price":p,"pnl":0,
                           "regime":regime,"sentiment":f"{sentiment:+.2f}"})
        if pos==1:
            pnl=(p-entry)/entry; cfg_exit=REGIME_PARAMS[regime_entry]
            if pnl>=cfg_exit["tp"] or pnl<=-cfg_exit["stop"]:
                cash*=(1+pnl); pos=0
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl,"regime":regime})
    return trades, sentiment, news_list

# ════════════════════════════════════════════════════════════
# V13 — 실적 발표 위험 조정
# ════════════════════════════════════════════════════════════
def get_earnings_risk(ticker):
    """다음 실적 발표일까지 거리 및 과거 실적 후 변동성"""
    try:
        tk_obj=yf.Ticker(ticker)
        cal=tk_obj.calendar
        if cal is None or cal.empty: return None, 0.0
        if "Earnings Date" in cal.index:
            earn_date=pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
            days_to_earn=(earn_date-pd.Timestamp.now()).days
        else:
            days_to_earn=999
        return days_to_earn, earn_date if days_to_earn<999 else None
    except Exception:
        return 999, None

def backtest_v13(df, ticker):
    """V12 + 실적 발표 전후 포지션 사이즈 조정"""
    sentiment, news_list = get_news_sentiment(ticker)
    days_to_earn, earn_date = get_earnings_risk(ticker)
    # 실적 발표 5일 이내면 포지션 50% 축소 (earnings_risk)
    earnings_risk = days_to_earn is not None and isinstance(days_to_earn,int) and days_to_earn<=5
    score_threshold=45-sentiment*15
    cash=1.0; pos=0; entry=0; size=1.0; trades=[]; regime_entry="UNKNOWN"; START=max(60,SWING)
    for i in range(START,len(df)):
        row=df.iloc[i]; p=float(row["Close"])
        score=row["ScorePct"]; regime=row["Regime"]; ma60=row["MA60"]
        if pd.isna(ma60) or pd.isna(score): continue
        # 실적 발표 직전 마지막 5봉은 사이즈 축소
        is_near_earn=(i>=len(df)-5 and earnings_risk)
        pos_size=0.5 if is_near_earn else 1.0
        if pos==0 and score>=score_threshold and p>float(ma60) and regime!="DOWNtrend":
            pos=1; entry=p; size=pos_size; regime_entry=regime
            trades.append({"type":"BUY","date":str(row["Date"])[:10],"price":p,"pnl":0,
                           "regime":regime,"size":f"{pos_size:.0%}"})
        if pos==1:
            pnl=(p-entry)/entry*size; cfg_exit=REGIME_PARAMS[regime_entry]
            if (p-entry)/entry>=cfg_exit["tp"] or (p-entry)/entry<=-cfg_exit["stop"]:
                cash*=(1+pnl); pos=0
                trades.append({"type":"SELL","date":str(row["Date"])[:10],"price":p,"pnl":pnl,"regime":regime})
    return trades, sentiment, news_list, days_to_earn, earn_date

# ════════════════════════════════════════════════════════════
# 멀티 종목 스캐너
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def scan_ticker(ticker, period):
    df=load_data(ticker,period)
    if df is None or len(df)<100: return None
    df=add_indicators(df); df=add_regime(df); df=add_v9_scores(df)
    valid=df.dropna(subset=["TotalScore","StochRSI","MA60"])
    if valid.empty: return None
    row=valid.iloc[-1]
    regime=row["Regime"]; cfg=REGIME_PARAMS[regime]
    price=float(row["Close"]); pct=float(row["ScorePct"])
    if regime=="DOWNtrend":   sig="🔴 관망"; sig_k="sell"
    elif pct>=75:             sig="🟢 강력 매수"; sig_k="buy"
    elif pct>=60:             sig="🟢 매수"; sig_k="buy"
    elif pct>=45:             sig="🟡 매수 검토"; sig_k="hold"
    elif pct>=30:             sig="⚪ 관망"; sig_k="hold"
    else:                     sig="🔴 금지"; sig_k="ban"
    return {"ticker":ticker,"regime":regime,"price":price,"score":float(row["TotalScore"]),
            "pct":pct,"signal":sig,"sig_k":sig_k,
            "ts":int(row["TrendScore"]),"cs":int(row["CycleScore"]),
            "ss":int(row["SeasonalScore"]),"irs":int(row["IrregularScore"]),
            "stoch":float(row["StochRSI"]),"adx":float(row["ADX"]),"df":df}

# ════════════════════════════════════════════════════════════
# 차트 함수
# ════════════════════════════════════════════════════════════
def draw_price_chart(df, trades, ticker, fib_lv=None, stop_s=None, tp_s=None):
    df_c=df.tail(200).copy()
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.56,0.22,0.22],
        vertical_spacing=0.03,
        subplot_titles=[f"{ticker} — 가격·피보나치·BUY/SELL 타이밍","StochRSI","ADX"])
    # 캔들
    fig.add_trace(go.Candlestick(x=df_c["Date"],open=df_c["Open"],high=df_c["High"],
        low=df_c["Low"],close=df_c["Close"],name="가격",
        increasing_line_color="#00ff9d",decreasing_line_color="#ff4757",
        increasing_fillcolor="#00ff9d",decreasing_fillcolor="#ff4757"),row=1,col=1)
    # MA
    for cn,color,name in [("MA20","#00d4ff","MA20"),("MA60","#ffd700","MA60"),("MA200","#ff8c00","MA200")]:
        if cn in df_c.columns:
            fig.add_trace(go.Scatter(x=df_c["Date"],y=df_c[cn],line=dict(color=color,width=1.2),
                name=name,opacity=0.85),row=1,col=1)
    # 레짐 배경
    if "Regime" in df_c.columns:
        rs=df_c[["Date","Regime"]].copy(); rs["prev"]=rs["Regime"].shift(1)
        bounds=rs[rs["Regime"]!=rs["prev"]].index.tolist()+[len(df_c)-1]; prev_i=0
        for b in bounds[1:]:
            seg=df_c.iloc[prev_i]["Regime"]; x0=df_c.iloc[prev_i]["Date"]; x1=df_c.iloc[min(b,len(df_c)-1)]["Date"]
            fig.add_vrect(x0=x0,x1=x1,fillcolor=REGIME_BG.get(seg,"rgba(120,120,120,0.04)"),
                          opacity=1,layer="below",line_width=0); prev_i=b
    # 피보나치
    if fib_lv:
        for f_val,fc,fl in zip(fib_lv,["#00ff9d","#ffd700","#ff8c00"],["BUY1","BUY2","BUY3"]):
            if f_val:
                fig.add_hline(y=f_val,line_dash="dash",line_color=fc,line_width=1.2,
                    annotation_text=f"  {fl}: ${f_val:.2f}",annotation_font_color=fc,row=1,col=1)
    if stop_s:
        fig.add_hline(y=stop_s,line_dash="dot",line_color="#ff4757",line_width=1.5,
            annotation_text=f"  STOP: ${stop_s:.2f}",annotation_font_color="#ff4757",row=1,col=1)
    if tp_s:
        fig.add_hline(y=tp_s,line_dash="dot",line_color="#ffd700",line_width=1.5,
            annotation_text=f"  TP: ${tp_s:.2f}",annotation_font_color="#ffd700",row=1,col=1)
    # BUY/SELL 마커
    if trades:
        df_trd=pd.DataFrame(trades)
        buys=df_trd[df_trd["type"]=="BUY"]; sells=df_trd[df_trd["type"]=="SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["date"],y=buys["price"],mode="markers",
                marker=dict(color="#00ff9d",size=11,symbol="triangle-up",line=dict(color="#fff",width=1)),
                name="매수 ▲"),row=1,col=1)
        if not sells.empty:
            win_s=sells[sells["pnl"]>0]; loss_s=sells[sells["pnl"]<=0]
            if not win_s.empty:
                fig.add_trace(go.Scatter(x=win_s["date"],y=win_s["price"],mode="markers",
                    marker=dict(color="#00d4ff",size=11,symbol="triangle-down",line=dict(color="#fff",width=1)),
                    name="익절 ▼"),row=1,col=1)
            if not loss_s.empty:
                fig.add_trace(go.Scatter(x=loss_s["date"],y=loss_s["price"],mode="markers",
                    marker=dict(color="#ff4757",size=11,symbol="triangle-down",line=dict(color="#fff",width=1)),
                    name="손절 ▼"),row=1,col=1)
    # StochRSI
    if "StochRSI" in df_c.columns:
        fig.add_trace(go.Scatter(x=df_c["Date"],y=df_c["StochRSI"],
            line=dict(color="#7b5ea7",width=1.5),name="StochRSI",
            fill="tozeroy",fillcolor="rgba(123,94,167,0.1)"),row=2,col=1)
        for y,c in [(20,"#00ff9d"),(80,"#ff4757")]:
            fig.add_hline(y=y,line_dash="dot",line_color=c,line_width=1,row=2,col=1)
    # ADX
    if "ADX" in df_c.columns:
        fig.add_trace(go.Scatter(x=df_c["Date"],y=df_c["ADX"],
            line=dict(color="#00d4ff",width=1.5),name="ADX"),row=3,col=1)
        fig.add_hline(y=20,line_dash="dot",line_color="#ffd700",line_width=1,row=3,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="#060810",plot_bgcolor="#0b0f1a",
        font=dict(color="#e8eaf6",size=11),height=760,margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(orientation="h",y=1.02,x=0),xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor="#1e2d4a"); fig.update_yaxes(gridcolor="#1e2d4a")
    return fig

def draw_equity_chart(m, label=""):
    if not m or "equity" not in m: return None
    equity=m["equity"]; dates=m["dates"]; pnls=m["pnls"]
    cum_ret=(equity-1)*100
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=cum_ret,fill="tozeroy",
        line=dict(color="#00d4ff",width=2),fillcolor="rgba(0,212,255,0.07)",name="누적수익률"))
    wins_idx=[i for i,p in enumerate(pnls) if p>0]; loss_idx=[i for i,p in enumerate(pnls) if p<=0]
    if wins_idx:
        fig.add_trace(go.Scatter(x=dates[wins_idx],y=cum_ret[wins_idx],mode="markers",
            marker=dict(color="#00ff9d",size=9,symbol="triangle-up"),name="익절 ▲"))
    if loss_idx:
        fig.add_trace(go.Scatter(x=dates[loss_idx],y=cum_ret[loss_idx],mode="markers",
            marker=dict(color="#ff4757",size=9,symbol="triangle-down"),name="손절 ▼"))
    fig.add_hline(y=0,line_dash="dot",line_color="#6b7280",line_width=1)
    # Drawdown 영역
    peak=np.maximum.accumulate(equity); dd=(equity-peak)/peak*100
    fig.add_trace(go.Scatter(x=dates,y=dd,fill="tozeroy",line=dict(color="#ff4757",width=1),
        fillcolor="rgba(255,71,87,0.06)",name="Drawdown",yaxis="y2"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="#060810",plot_bgcolor="#0b0f1a",
        height=340,margin=dict(l=10,r=10,t=36,b=10),
        title=dict(text=f"📈 {label} 에쿼티 커브 + Drawdown",font=dict(size=13)),
        font=dict(color="#e8eaf6"),yaxis_ticksuffix="%",
        yaxis2=dict(overlaying="y",side="right",ticksuffix="%",showgrid=False,
                    title="DD",tickfont=dict(color="#ff4757")))
    return fig

def draw_score_radar(ts,cs,ss,irs):
    categories=["TREND","CYCLE","SEASON","IRREG","TREND"]
    values=[ts,cs,ss,irs,ts]
    fig=go.Figure(go.Scatterpolar(r=values,theta=categories,fill="toself",
        fillcolor="rgba(0,212,255,0.12)",line=dict(color="#00d4ff",width=2),name="점수"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,4],
        gridcolor="#1e2d4a",linecolor="#1e2d4a"),angularaxis=dict(gridcolor="#1e2d4a")),
        template="plotly_dark",paper_bgcolor="#111827",
        height=260,margin=dict(l=20,r=20,t=20,b=20),showlegend=False)
    return fig

# ════════════════════════════════════════════════════════════
# 리포트 (Excel + HTML)
# ════════════════════════════════════════════════════════════
def make_excel(ticker,strategy,m,trades):
    buf=io.BytesIO()
    now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        summary=pd.DataFrame({"항목":["종목","전략","분석일","총거래","승률(%)","총수익률(%)",
            "CAGR(%)","MDD(%)","Sharpe","Calmar","손익비(RR)","평균익절(%)","평균손절(%)"],
            "값":[ticker,strategy,now,m.get("total","N/A"),f"{m.get('wr',0):.1f}",
                  f"{m.get('total_ret',0):.1f}",f"{m.get('cagr',0):.1f}",f"{m.get('mdd',0):.1f}",
                  f"{m.get('sharpe',0):.2f}",f"{m.get('calmar',0):.2f}",f"{m.get('rr',0):.2f}",
                  f"{m.get('aw',0):.1f}",f"{m.get('al',0):.1f}"]})
        summary.to_excel(writer,sheet_name="성과요약",index=False)
        if trades:
            sells=[t for t in trades if t["type"]=="SELL"]
            if sells:
                df_t=pd.DataFrame(sells)
                if "pnl" in df_t.columns: df_t["pnl_pct"]=(df_t["pnl"]*100).round(2)
                df_t.to_excel(writer,sheet_name="거래내역",index=False)
    buf.seek(0); return buf.getvalue()

def make_html_report(ticker,strategy,m,trades,news_list=None,sentiment=None):
    sc=""; now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_ret=m.get("total_ret",0)
    sc="#00ff9d" if total_ret>0 else "#ff4757"
    sells=[t for t in trades if t["type"]=="SELL"]
    rows=""
    for t in sorted(sells,key=lambda x:x["date"],reverse=True)[:50]:
        c="#00ff9d" if t["pnl"]>0 else "#ff4757"
        rows+=f"<tr><td>{t['date']}</td><td style='color:{c}'>{'✅익절' if t['pnl']>0 else '❌손절'}</td><td>${t['price']:.2f}</td><td style='color:{c}'>{t['pnl']*100:+.1f}%</td><td>{t.get('regime','')}</td></tr>"
    news_section=""
    if news_list:
        news_rows="".join(f"<tr><td>{h}</td><td style='color:{'#00ff9d' if s>0 else '#ff4757'}'>{s:+.2f}</td></tr>" for h,s in news_list[:10])
        news_section=f"<div class='s'><h2>📰 뉴스 감성 분석 (평균: {sentiment:+.2f})</h2><table><thead><tr><th>헤드라인</th><th>감성</th></tr></thead><tbody>{news_rows}</tbody></table></div>"
    def cv(l,v,c="#e8eaf6"):
        return f"<div class='c'><div class='l'>{l}</div><div class='v' style='color:{c}'>{v}</div></div>"
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<title>프로젝트 이세계 V13 — {ticker}</title>
<style>body{{background:#060810;color:#e8eaf6;font-family:'Segoe UI',sans-serif;margin:0;padding:20px}}
h1{{text-align:center;background:linear-gradient(135deg,#00d4ff,#7b5ea7,#00ff9d);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:1.9rem}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:12px 0}}
.c{{background:#111827;border:1px solid #1e2d4a;border-radius:10px;padding:12px;text-align:center}}
.v{{font-size:1.3rem;font-weight:700;margin:4px 0}}.l{{color:#6b7280;font-size:.72rem}}
.s{{background:#0b0f1a;border:1px solid #1e2d4a;border-radius:10px;padding:14px;margin:12px 0}}
h2{{color:#00d4ff;font-size:.92rem;margin-bottom:8px}}
table{{width:100%;border-collapse:collapse}}th{{background:#111827;color:#6b7280;padding:8px;text-align:left;font-size:.72rem}}
td{{padding:8px;border-bottom:1px solid #1e2d4a;font-size:.78rem}}
.foot{{text-align:center;color:#374151;font-size:.68rem;margin-top:20px}}</style></head><body>
<h1>👑 프로젝트 이세계 V13</h1>
<p style="text-align:center;color:#6b7280">{ticker} · {strategy} · {now}</p>
<div class="s"><h2>🧪 백테스트 성과</h2><div class="g">
{cv("총수익률",f"{m.get('total_ret',0):.1f}%",sc)}
{cv("CAGR",f"{m.get('cagr',0):.1f}%","#00d4ff")}
{cv("MDD",f"{m.get('mdd',0):.1f}%","#ff4757")}
{cv("Sharpe",f"{m.get('sharpe',0):.2f}","#7b5ea7")}
{cv("Calmar",f"{m.get('calmar',0):.2f}","#00d4ff")}
{cv("승률",f"{m.get('wr',0):.1f}%","#00ff9d" if m.get('wr',0)>=55 else "#ff4757")}
{cv("손익비",f"{m.get('rr',0):.2f}","#00ff9d" if m.get('rr',0)>=1.5 else "#ffd700")}
{cv("총거래",f"{m.get('total',0)}건","#6b7280")}
</div></div>
{news_section}
<div class="s"><h2>📝 최근 거래 내역 (최대 50건)</h2>
<table><thead><tr><th>날짜</th><th>유형</th><th>가격</th><th>수익률</th><th>레짐</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class="foot">⚠ 본 분석은 참고용이며 투자 손익은 본인 책임입니다. · 프로젝트 이세계 V13</div>
</body></html>"""

# ════════════════════════════════════════════════════════════
# 헬퍼 UI
# ════════════════════════════════════════════════════════════
def mcard(col,label,val,color="#e8eaf6",sub=""):
    col.markdown(f"""<div class="mc"><div class="mc-lbl">{label}</div>
    <div class="mc-val" style="color:{color}">{val}</div>
    <div style="color:#6b7280;font-size:.68rem;margin-top:2px">{sub}</div></div>""",unsafe_allow_html=True)

def score_bar(col,label,val,color,maxv=4):
    col.markdown(f"""<div class="mc"><div class="mc-lbl">{label}</div>
    <div class="mc-val" style="color:{color}">{val}/{maxv}</div>
    <div class="bar-wrap"><div class="bar-fill" style="width:{val/maxv*100}%;background:{color}"></div></div>
    </div>""",unsafe_allow_html=True)

def metrics_row(m, cols=8):
    keys=[("총 거래",f"{m.get('total',0)}건","#00d4ff",""),
          ("승률",f"{m.get('wr',0):.1f}%","#00ff9d" if m.get('wr',0)>=55 else "#ff4757",""),
          ("총 수익률",f"{m.get('total_ret',0):.1f}%","#00ff9d" if m.get('total_ret',0)>0 else "#ff4757",""),
          ("CAGR",f"{m.get('cagr',0):.1f}%","#00d4ff",""),
          ("MDD",f"{m.get('mdd',0):.1f}%","#ff4757",""),
          ("Sharpe",f"{m.get('sharpe',0):.2f}","#7b5ea7","✅" if m.get('sharpe',0)>=1 else ""),
          ("Calmar",f"{m.get('calmar',0):.2f}","#00d4ff","✅" if m.get('calmar',0)>=1 else ""),
          ("손익비(RR)",f"{m.get('rr',0):.2f}","#00ff9d" if m.get('rr',0)>=1.5 else "#ffd700","✅" if m.get('rr',0)>=1.5 else "⚠")]
    c_list=st.columns(len(keys))
    for col,(lbl,val,color,sub) in zip(c_list,keys):
        mcard(col,lbl,val,color,sub)

# ════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════
STRATEGIES={
    "V1 — Fibonacci 눌림목":"V1",
    "V5 — 레짐(UP/RANGE/DOWN) 전환":"V5",
    "V8 — 비중 자동 최적화 (Grid Search)":"V8",
    "V9 — 4-Factor Score (AROS 모델)":"V9",
    "V11 — ML (RandomForest)":"V11",
    "V12 — 뉴스 감성 분석 (NLP)":"V12",
    "V13 — 실적 톤 분석 (Earnings)":"V13",
}

with st.sidebar:
    st.markdown("### ⚙️ 설정")
    mode=st.radio("모드",["📊 단일 종목 백테스트","🌐 멀티 종목 스캐너"],index=0)
    st.markdown("---")
    if "단일" in mode:
        ticker=st.text_input("티커",value="VRT").upper().strip()
        period=st.selectbox("기간",["1y","2y","3y","5y"],index=1)
        strategy_name=st.selectbox("전략 버전",list(STRATEGIES.keys()))
        strategy_key=STRATEGIES[strategy_name]
        run_btn=st.button("🚀 백테스트 실행",use_container_width=True,type="primary")
    else:
        ticker_text=st.text_area("종목 목록 (줄바꿈)",
            value="VRT\nNVDA\nTSLA\nMSFT\nAAPL\nQQQ\nSPY\nAMZN",height=170)
        period=st.selectbox("기간",["1y","2y","3y"],index=1)
        run_btn=st.button("🚀 전체 스캔",use_container_width=True,type="primary")
        ticker=""; strategy_key="V9"; strategy_name="V9"
    st.markdown("---")
    st.markdown("**전략 설명**")
    desc_map={
        "V1":"순수 피보나치 눌림목. 가장 단순하고 빠름.",
        "V5":"레짐별 자동 전환. 상/박스/하락장 대응.",
        "V8":"비중 그리드서치 최적화. 가장 정교한 FIB.",
        "V9":"4요소 통합 스코어. AROS 핵심 모델.",
        "V11":"ML 방향 예측. 테스트셋 기준 백테스트.",
        "V12":"뉴스 감성 가중. NLP 경량 연동.",
        "V13":"실적 위험 조정. 최종 완성형.",
    }
    if "단일" in mode:
        st.markdown(f'<div class="strat-card" style="background:#0f172a;border:1px solid #1e3a5f;border-radius:10px;padding:12px;font-size:.8rem;color:#9ca3af">{desc_map.get(strategy_key,"")}</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="warn">⚠️ 참고용 분석입니다.<br>투자 손익은 본인 책임입니다.</div>',unsafe_allow_html=True)

# ── 헤더 ──
st.markdown('<div class="main-title">👑 프로젝트 이세계 V13</div>',unsafe_allow_html=True)
st.markdown('<div class="sub-title">V1 Fibonacci → V5 Regime → V8 Weight Opt → V9 AROS → V11 ML → V12 NLP → V13 Earnings</div>',unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 단일 종목 백테스트
# ════════════════════════════════════════════════════════════
if "단일" in mode and run_btn:
    if not ticker: st.error("티커를 입력하세요."); st.stop()

    with st.spinner(f"📡 {ticker} 데이터 로드 중..."):
        df=load_data(ticker,period)
        if df is None or len(df)<100:
            st.error("데이터를 가져올 수 없습니다."); st.stop()
        df=add_indicators(df); df=add_regime(df); df=add_v9_scores(df)

    # 현재 시그널
    valid=df.dropna(subset=["TotalScore","StochRSI","MA60"])
    row=valid.iloc[-1] if not valid.empty else None
    price=float(row["Close"]) if row is not None else 0
    pct=float(row["ScorePct"]) if row is not None else 0
    regime=row["Regime"] if row is not None else "UNKNOWN"
    cfg=REGIME_PARAMS[regime]
    rng=row["rng"]; sh=row["sw_high"]
    fib_lv=[sh-rng*f for f in cfg["fib"]] if (not pd.isna(rng) and rng>0 and not pd.isna(sh)) else [None,None,None]
    avg_s=sum(fib_lv[i]*[0.33,0.34,0.33][i] for i in range(3)) if fib_lv[0] else None
    stop_s=avg_s*(1-cfg["stop"]) if avg_s else None; tp_s=avg_s*TP_FINAL if avg_s else None

    if regime=="DOWNtrend":   final="🔴 관망 (하락장)"; sig_k="sell"
    elif pct>=75:             final="🟢 강력 매수 (왕급)"; sig_k="buy"
    elif pct>=60:             final="🟢 강력 매수"; sig_k="buy"
    elif pct>=45:             final="🟡 매수 검토"; sig_k="hold"
    elif pct>=30:             final="⚪ 관망"; sig_k="hold"
    else:                     final="🔴 매수 금지"; sig_k="ban"

    st.markdown(f"#### 📅 {ticker}  ·  {str(row['Date'])[:10] if row is not None else ''}  ·  {strategy_name}")

    # 시그널 + 레짐 + 점수
    sm={"buy":("sig-buy","sig-txt-buy"),"sell":("sig-sell","sig-txt-sell"),
        "hold":("sig-hold","sig-txt-hold"),"ban":("sig-ban","sig-txt-ban")}
    dc,tc=sm.get(sig_k,("sig-hold","sig-txt-hold"))
    c1,c2,c3,c4=st.columns([2,1,1,1])
    with c1:
        st.markdown(f"""<div class="{dc}"><div class="{tc}">{final}</div>
          <div style="color:#6b7280;font-size:.78rem;margin-top:5px">점수 {pct:.1f}% ({row['TotalScore']:.0f}/20)</div>
          <div style="background:#1e2d4a;border-radius:4px;height:5px;margin:8px auto;max-width:240px">
          <div style="width:{pct:.1f}%;height:5px;border-radius:4px;background:linear-gradient(90deg,#00d4ff,#00ff9d)"></div></div></div>""",unsafe_allow_html=True)
    badge_map={"UPtrend":"badge-up","DOWNtrend":"badge-down","RANGE":"badge-range","UNKNOWN":"badge-range"}
    with c2:
        st.markdown(f"""<div class="mc"><div class="mc-lbl">레짐</div>
          <div style="margin:6px 0"><span class="{badge_map.get(regime,'badge-range')}">{regime}</span></div>
          <div style="color:#6b7280;font-size:.7rem">{cfg['desc']}</div></div>""",unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="mc"><div class="mc-lbl">현재가</div>
          <div class="mc-val" style="color:#00d4ff">${price:.2f}</div></div>""",unsafe_allow_html=True)
    with c4:
        rc=df["Regime"].value_counts(); td=len(df)
        st.markdown(f"""<div class="mc"><div class="mc-lbl">레짐 분포</div>
          <div style="font-size:.72rem;line-height:1.7">
          🟢UP {rc.get('UPtrend',0)/td*100:.0f}%<br>
          🟡RG {rc.get('RANGE',0)/td*100:.0f}%<br>
          🔴DN {rc.get('DOWNtrend',0)/td*100:.0f}%</div></div>""",unsafe_allow_html=True)
    st.markdown("---")

    # 스코어 레이더 + 점수 카드
    c_radar,c_scores=st.columns([1,2])
    with c_radar:
        st.markdown("##### 🎯 4-Factor 레이더")
        if row is not None:
            fig_r=draw_score_radar(int(row["TrendScore"]),int(row["CycleScore"]),
                                   int(row["SeasonalScore"]),int(row["IrregularScore"]))
            st.plotly_chart(fig_r,use_container_width=True)
    with c_scores:
        st.markdown("##### 🧩 모듈별 점수")
        sb1,sb2,sb3,sb4,sb5=st.columns(5)
        if row is not None:
            score_bar(sb1,"TREND",int(row["TrendScore"]),"#00d4ff")
            score_bar(sb2,"CYCLE",int(row["CycleScore"]),"#7b5ea7")
            score_bar(sb3,"SEASON",int(row["SeasonalScore"]),"#ffd700")
            score_bar(sb4,"IRREG",int(row["IrregularScore"]),"#ff8c00")
            score_bar(sb5,"TOTAL",int(row["TotalScore"]),"#00ff9d",20)
        st.markdown("##### 📋 매매 플랜")
        p1,p2,p3,p4=st.columns(4)
        mcard(p1,"BUY1",f"${fib_lv[0]:.2f}" if fib_lv[0] else "N/A","#00ff9d")
        mcard(p2,"손절선",f"${stop_s:.2f}" if stop_s else "N/A","#ff4757")
        mcard(p3,"익절목표",f"${tp_s:.2f}" if tp_s else "N/A","#ffd700")
        mcard(p4,"손절폭",f"-{cfg['stop']*100:.0f}%","#ff8c00")
    st.markdown("---")

    # 백테스트 실행
    st.markdown(f"##### 🧪 백테스트 — {strategy_name}")
    trades=[]; sentiment=None; news_list=[]; importances=None; days_to_earn=None; earn_date_val=None
    ml_note=""

    if strategy_key=="V1":
        with st.spinner("V1 Fibonacci 백테스트 중..."): trades=backtest_v1(df)
    elif strategy_key=="V5":
        with st.spinner("V5 레짐 전환 백테스트 중..."): trades=backtest_v5(df)
    elif strategy_key=="V8":
        prog_bar=st.progress(0)
        with st.spinner("V8 비중 최적화 중 (30~90초)..."):
            trades,_=backtest_v8(df, lambda p: prog_bar.progress(p))
        prog_bar.empty()
    elif strategy_key=="V9":
        with st.spinner("V9 AROS 백테스트 중..."): trades=backtest_v9(df)
    elif strategy_key=="V11":
        with st.spinner("V11 ML 학습 + 백테스트 중..."):
            result=backtest_v11(df)
            if isinstance(result,tuple): trades,importances=result
            else: trades=result; ml_note="scikit-learn 미설치 → V9 대체"
    elif strategy_key=="V12":
        with st.spinner("V12 뉴스 감성 분석 중..."):
            trades,sentiment,news_list=backtest_v12(df,ticker)
    elif strategy_key=="V13":
        with st.spinner("V13 실적 분석 + 백테스트 중..."):
            trades,sentiment,news_list,days_to_earn,earn_date_val=backtest_v13(df,ticker)

    m=calc_metrics(trades,strategy_name)

    if m:
        metrics_row(m)
        st.markdown("")
        eq_fig=draw_equity_chart(m,f"{ticker} — {strategy_name}")
        if eq_fig: st.plotly_chart(eq_fig,use_container_width=True)
    else:
        st.warning("거래 없음 — 기간을 늘리거나 전략을 변경하세요.")

    # ML 피처 중요도
    if importances is not None:
        st.markdown("##### 🤖 ML 피처 중요도")
        fig_imp=go.Figure(go.Bar(x=importances.values,y=importances.index,orientation="h",
            marker_color="#00d4ff"))
        fig_imp.update_layout(template="plotly_dark",paper_bgcolor="#060810",plot_bgcolor="#0b0f1a",
            height=260,margin=dict(l=10,r=10,t=20,b=10),font=dict(color="#e8eaf6",size=11))
        st.plotly_chart(fig_imp,use_container_width=True)

    # 뉴스 감성
    if news_list:
        st.markdown("##### 📰 뉴스 감성 분석")
        sent_color="#00ff9d" if sentiment>0.1 else "#ff4757" if sentiment<-0.1 else "#ffd700"
        st.markdown(f'<div class="mc" style="margin-bottom:10px"><div class="mc-lbl">평균 감성 점수</div><div class="mc-val" style="color:{sent_color}">{sentiment:+.3f}</div><div style="color:#6b7280;font-size:.7rem">{"긍정적" if sentiment>0.1 else "부정적" if sentiment<-0.1 else "중립"}</div></div>',unsafe_allow_html=True)
        df_news=pd.DataFrame(news_list,columns=["헤드라인","감성점수"])
        df_news["감성점수"]=df_news["감성점수"].map("{:+.3f}".format)
        st.dataframe(df_news,use_container_width=True,hide_index=True)

    # 실적 발표 정보
    if days_to_earn is not None and isinstance(days_to_earn,int) and days_to_earn<999:
        earn_str=str(earn_date_val)[:10] if earn_date_val else "N/A"
        color="#ff4757" if days_to_earn<=5 else "#ffd700" if days_to_earn<=14 else "#00ff9d"
        st.markdown(f'<div class="mc" style="margin-bottom:10px"><div class="mc-lbl">다음 실적 발표까지</div><div class="mc-val" style="color:{color}">{days_to_earn}일 ({earn_str})</div><div style="color:#6b7280;font-size:.7rem">{"⚠ 포지션 축소 구간" if days_to_earn<=5 else ""}</div></div>',unsafe_allow_html=True)
    st.markdown("---")

    # 차트
    st.markdown("##### 📊 인터랙티브 차트")
    st.plotly_chart(draw_price_chart(df,trades,ticker,fib_lv,stop_s,tp_s),use_container_width=True)

    # 거래 내역
    sells=[t for t in trades if t["type"]=="SELL"]
    if sells:
        st.markdown("##### 📝 거래 내역")
        df_t=pd.DataFrame(sells).sort_values("date",ascending=False).head(30)
        if "pnl" in df_t.columns: df_t["수익률"]=df_t["pnl"].map(lambda x:f"{x*100:+.1f}%")
        df_t=df_t.rename(columns={"date":"날짜","price":"가격","regime":"레짐"})
        cols=[c for c in ["날짜","유형","가격","수익률","레짐"] if c in df_t.columns]
        st.dataframe(df_t[cols] if cols else df_t,use_container_width=True,hide_index=True)

    # 다운로드
    if m:
        st.markdown("---"); st.markdown("##### 📄 리포트 다운로드")
        dl1,dl2=st.columns(2)
        now_str=datetime.datetime.now().strftime("%Y%m%d_%H%M")
        html_bytes=make_html_report(ticker,strategy_name,m,trades,news_list,sentiment)
        b64=base64.b64encode(html_bytes.encode()).decode()
        dl1.markdown(f"""<a href="data:text/html;base64,{b64}" download="isekai_{ticker}_{now_str}.html"
           style="display:block;text-align:center;padding:11px;background:rgba(0,212,255,.12);
                  border:1px solid #00d4ff;border-radius:8px;color:#00d4ff;font-weight:700;text-decoration:none;">
          📄 HTML 리포트 다운로드</a>""",unsafe_allow_html=True)
        excel_bytes=make_excel(ticker,strategy_name,m,trades)
        dl2.download_button("📊 Excel 리포트 다운로드",data=excel_bytes,
            file_name=f"isekai_{ticker}_{now_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ════════════════════════════════════════════════════════════
# 멀티 종목 스캐너
# ════════════════════════════════════════════════════════════
elif "멀티" in mode and run_btn:
    tickers=[t.strip().upper() for t in ticker_text.split("\n") if t.strip()]
    if not tickers: st.error("종목을 입력하세요."); st.stop()
    st.markdown(f"#### 🌐 {len(tickers)}개 종목 스캔 — V9 AROS Score 기준")
    prog=st.progress(0); results=[]
    for i,tk in enumerate(tickers):
        with st.spinner(f"  {tk} ({i+1}/{len(tickers)})..."):
            res=scan_ticker(tk,period)
        if res is None:
            results.append({"Ticker":tk,"레짐":"오류","시그널":"❌","점수":0,"점수(%)":"0%",
                             "TREND":"-","CYCLE":"-","SEASON":"-","IRREG":"-","StochRSI":"-","ADX":"-","_res":None})
        else:
            results.append({"Ticker":tk,"레짐":res["regime"],"시그널":res["signal"],
                "점수":res["score"],"점수(%)":f"{res['pct']:.1f}%",
                "TREND":res["ts"],"CYCLE":res["cs"],"SEASON":res["ss"],"IRREG":res["irs"],
                "StochRSI":f"{res['stoch']:.1f}","ADX":f"{res['adx']:.1f}","_res":res})
        prog.progress((i+1)/len(tickers))

    results=sorted(results,key=lambda x:x["점수"],reverse=True)
    st.markdown("---"); st.markdown("### 📊 종목 분석 — 신호 강한 순")
    df_table=pd.DataFrame([{k:v for k,v in r.items() if k!="_res"} for r in results])
    st.dataframe(df_table,use_container_width=True,hide_index=True)
    st.markdown("---")

    # 개별 차트
    st.markdown("### 📈 개별 차트")
    valid_tks=[r["Ticker"] for r in results if r["_res"]]
    if valid_tks:
        selected=st.selectbox("종목 선택",valid_tks)
        sel=next((r["_res"] for r in results if r["Ticker"]==selected and r["_res"]),None)
        if sel:
            df2=sel["df"]; r2=sel
            row2=df2.dropna(subset=["TotalScore"]).iloc[-1]
            fig2=draw_price_chart(df2,[],selected)
            st.plotly_chart(fig2,use_container_width=True)
            sm2,sm3,sm4,sm5=st.columns(4)
            score_bar(sm2,"TREND",r2["ts"],"#00d4ff"); score_bar(sm3,"CYCLE",r2["cs"],"#7b5ea7")
            score_bar(sm4,"SEASON",r2["ss"],"#ffd700"); score_bar(sm5,"IRREG",r2["irs"],"#ff8c00")

# ── 초기 화면 ──
if not run_btn:
    st.markdown("""<div style="text-align:center;padding:44px 20px;color:#6b7280">
      <div style="font-size:3.2rem;margin-bottom:12px">👑</div>
      <div style="font-size:1rem;color:#9ca3af">왼쪽 사이드바에서 종목·전략을 선택하고<br>
        <strong style="color:#00d4ff">백테스트 실행</strong>을 누르세요</div>
    </div>""",unsafe_allow_html=True)
    st.markdown("---")
    rows_info=[
        [("V1","📐","피보나치 눌림목","가장 단순·빠름"),
         ("V5","🌍","레짐 전환","상/박스/하락 자동 대응"),
         ("V8","⚖️","비중 최적화","Grid Search + Calmar"),
         ("V9","🧬","AROS 모델","4-Factor 통합 스코어")],
        [("V11","🤖","ML 예측","RandomForest 방향 예측"),
         ("V12","📰","뉴스 감성","NLP 경량 연동"),
         ("V13","📊","실적 분석","Earnings 위험 조정"),
         ("🌐","🌐","멀티 스캔","여러 종목 동시 분석")]
    ]
    for row_items in rows_info:
        cols=st.columns(4)
        for col,(key,icon,title,desc) in zip(cols,row_items):
            col.markdown(f"""<div class="mc" style="padding:14px">
              <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
              <div style="font-weight:700;margin-bottom:3px;color:#00d4ff;font-size:.82rem">{key}</div>
              <div style="color:#e8eaf6;font-size:.82rem;font-weight:600">{title}</div>
              <div style="color:#6b7280;font-size:.7rem;margin-top:3px">{desc}</div></div>""",unsafe_allow_html=True)
