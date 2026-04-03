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
    # 스윙 피보나치
    df["sw_high"]=df["High"].rolling(SWING).max()
    df["sw_low"]=df["Low"].rolling(SWING).min()
    df["rng"]=df["sw_high"]-df["sw_low"]
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
    return df

# ════════════════════════════════════════════════════════════
# 단일 종목 분석 (캐시)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def analyze(ticker, period="1y"):
    try:
        df=yf.download(ticker,period=period,interval="1d",auto_adjust=True,progress=False)
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
        rng=row["rng"]; sh=row["sw_high"]
        if not pd.isna(rng) and rng>0 and not pd.isna(sh):
            fib_lv=[sh-rng*f for f in cfg["fib"]]; fib886=sh-rng*0.886
        else:
            fib_lv=[None,None,None]; fib886=None
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
        }
    except Exception:
        return None

# ════════════════════════════════════════════════════════════
# 백테스트 엔진
# ════════════════════════════════════════════════════════════
def run_backtest(df):
    trades=[]; cash=1.0; pos=0; entry=0; regime_entry="UNKNOWN"
    START=max(60,SWING)
    for i in range(START,len(df)):
        row=df.iloc[i]; p=float(row["Close"])
        pct=row["ScorePct"]; regime=row["Regime"]; cfg=REGIME_PARAMS[regime]
        ma60=row["MA60"]
        if pd.isna(ma60) or pd.isna(pct): continue
        if pos==0 and pct>=45 and p>float(ma60) and regime!="DOWNtrend":
            pos=1; entry=p; regime_entry=regime
            trades.append({"날짜":str(row["Date"])[:10],"구분":"매수",
                           "가격":round(p,2),"수익률":"-","레짐":regime,"비고":f"점수 {pct:.0f}%"})
        if pos==1:
            pnl=(p-entry)/entry; cfg_e=REGIME_PARAMS[regime_entry]
            if pnl>=cfg_e["tp"] or pnl<=-cfg_e["stop"]:
                cash*=(1+pnl); pos=0
                result="✅ 익절" if pnl>0 else "❌ 손절"
                trades.append({"날짜":str(row["Date"])[:10],"구분":"매도",
                               "가격":round(p,2),"수익률":f"{pnl*100:+.1f}%","레짐":regime,"비고":result})
    sells=[t for t in trades if t["구분"]=="매도"]
    if not sells: return trades,{}
    pnls=np.array([float(t["수익률"].replace("%","").replace("+",""))/100 for t in sells])
    wins=[p for p in pnls if p>0]; losses=[p for p in pnls if p<=0]
    equity=np.cumprod(1+pnls); peak=np.maximum.accumulate(equity)
    mdd=float(((equity-peak)/peak).min()*100)
    wr=len(wins)/len(pnls)*100
    dates=pd.to_datetime([t["날짜"] for t in sells])
    n_years=max((dates[-1]-dates[0]).days/365.25,0.1)
    cagr=(equity[-1]**(1/n_years)-1)*100
    sharpe=float(np.mean(pnls)/np.std(pnls)*np.sqrt(252)) if np.std(pnls)>0 else 0
    calmar=cagr/abs(mdd) if mdd!=0 else 0
    metrics={
        "총 거래":len(sells),"승률":f"{wr:.1f}%",
        "총 수익률":f"{(equity[-1]-1)*100:.1f}%",
        "CAGR":f"{cagr:.1f}%","MDD":f"{mdd:.1f}%",
        "Sharpe":f"{sharpe:.2f}","Calmar":f"{calmar:.2f}",
        "평균 익절":f"{np.mean(wins)*100:.1f}%" if wins else "-",
        "평균 손절":f"{np.mean(losses)*100:.1f}%" if losses else "-",
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

    # ── 매매 플랜 표 ──
    st.markdown("#### 📋 매매 플랜")
    plan_data={
        "구분":["1차 매수 (BUY1)","2차 매수 (BUY2)","3차 매수 (BUY3)","손절선","익절 목표"],
        "가격":[
            f"${res['fib_lv'][0]:.2f}" if res['fib_lv'][0] else "N/A",
            f"${res['fib_lv'][1]:.2f}" if res['fib_lv'][1] else "N/A",
            f"${res['fib_lv'][2]:.2f}" if res['fib_lv'][2] else "N/A",
            f"${res['stop_s']:.2f}" if res['stop_s'] else "N/A",
            f"${res['tp_s']:.2f}" if res['tp_s'] else "N/A",
        ],
        "현재가 대비":[
            f"{(res['fib_lv'][i]/res['price']-1)*100:+.1f}%" if res['fib_lv'][i] else "-"
            for i in range(3)
        ]+[
            f"{(res['stop_s']/res['price']-1)*100:+.1f}%" if res['stop_s'] else "-",
            f"{(res['tp_s']/res['price']-1)*100:+.1f}%" if res['tp_s'] else "-",
        ],
        "설명":[
            f"Fib {res['cfg']['fib'][0]} 눌림목","Fib {res['cfg']['fib'][1]} 눌림목",
            f"Fib {res['cfg']['fib'][2]} 눌림목",
            f"-{res['cfg']['stop']*100:.0f}% 또는 Fib 0.886",
            f"+{res['cfg']['tp']*100:.0f}% 목표",
        ]
    }
    st.dataframe(pd.DataFrame(plan_data),use_container_width=True,hide_index=True,
        column_config={"구분":st.column_config.TextColumn(width="medium"),
                       "가격":st.column_config.TextColumn(width="small"),
                       "현재가 대비":st.column_config.TextColumn(width="small"),
                       "설명":st.column_config.TextColumn(width="large")})
    st.markdown("---")

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

    # ── 차트 ──
    st.markdown("#### 📈 차트")
    st.plotly_chart(draw_chart(res),use_container_width=True)

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
    with st.spinner(f"{ticker_input} 백테스트 중..."):
        res=analyze(ticker_input,period_input)
        if res is None: st.error("데이터 오류"); st.stop()
        trades,metrics=run_backtest(res["df"])

    st.markdown(f"### 📊 {ticker_input} 백테스트 결과 ({period_input})")

    if not metrics:
        st.warning("거래 없음 — 기간을 늘려보세요.")
    else:
        # ── 성과 지표 표 ──
        m=metrics
        col1,col2=st.columns([1,1])
        with col1:
            st.markdown("#### 🏆 성과 지표")
            metrics_df=pd.DataFrame({
                "지표":list(m.keys()),"결과":list(m.values())
            })
            st.dataframe(metrics_df,use_container_width=True,hide_index=True)
        with col2:
            st.markdown("#### 💡 해석")
            wr_val=float(m["승률"].replace("%",""))
            cagr_val=float(m["CAGR"].replace("%",""))
            mdd_val=float(m["MDD"].replace("%",""))
            sharpe_val=float(m["Sharpe"])
            interpretations=[
                f"승률 {m['승률']} → {'✅ 양호 (55% 이상)' if wr_val>=55 else '⚠️ 개선 필요'}",
                f"CAGR {m['CAGR']} → {'✅ 우수' if cagr_val>=15 else '⚠️ 보통' if cagr_val>=5 else '❌ 저조'}",
                f"MDD {m['MDD']} → {'✅ 안전' if abs(mdd_val)<=15 else '⚠️ 주의' if abs(mdd_val)<=30 else '❌ 위험'}",
                f"Sharpe {m['Sharpe']} → {'✅ 우수 (1.0 이상)' if sharpe_val>=1 else '⚠️ 보통' if sharpe_val>=0.5 else '❌ 저조'}",
            ]
            for interp in interpretations:
                st.markdown(f"- {interp}")

        st.markdown("---")

        # ── 거래 내역 표 ──
        st.markdown("#### 📝 거래 내역")
        if trades:
            df_trades=pd.DataFrame(trades)
            st.dataframe(df_trades,use_container_width=True,hide_index=True,
                column_config={
                    "날짜":st.column_config.TextColumn(width="small"),
                    "구분":st.column_config.TextColumn(width="small"),
                    "가격":st.column_config.NumberColumn(format="$%.2f",width="small"),
                    "수익률":st.column_config.TextColumn(width="small"),
                    "레짐":st.column_config.TextColumn(width="small"),
                    "비고":st.column_config.TextColumn(width="medium"),
                })
        st.markdown("---")

        # ── 차트 (매매 타이밍 포함) ──
        st.markdown("#### 📈 차트 + 매매 타이밍")
        st.plotly_chart(draw_chart(res,trades),use_container_width=True)

        # ── Excel 다운로드 ──
        now_str=datetime.datetime.now().strftime("%Y%m%d_%H%M")
        excel=make_excel(ticker_input,res,metrics,trades)
        st.download_button("📊 백테스트 결과 Excel 다운로드",data=excel,
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
