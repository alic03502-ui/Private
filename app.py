
import os, math, sqlite3, datetime as dt, re
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Banker AI v7", page_icon="⚽", layout="wide")
API_HOST="https://v3.football.api-sports.io"
DB=Path("football_banker_ai.db")

# ========================= DATABASE =========================
def db():
    con=sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS odds_snapshots(
        ts TEXT, fixture_id INTEGER, home TEXT, away TEXT, market TEXT,
        selection TEXT, odds REAL, bookmaker TEXT, status TEXT DEFAULT 'OPEN')""")
    con.execute("""CREATE TABLE IF NOT EXISTS bets(
        placed_at TEXT, fixture_id INTEGER, home TEXT, away TEXT, market TEXT,
        selection TEXT, odds REAL, probability REAL, stake REAL, result TEXT,
        pnl REAL, note TEXT)""")
    con.commit()
    return con

# ========================= API =========================
def api(key,path,params=None):
    r=requests.get(API_HOST+path,headers={"x-apisports-key":key},
                   params=params or {},timeout=25)
    if r.status_code==401: raise RuntimeError("API key rejected (401).")
    if r.status_code==429: raise RuntimeError("API quota/rate limit reached (429).")
    r.raise_for_status()
    d=r.json()
    if d.get("errors"): raise RuntimeError(str(d["errors"]))
    return d.get("response",[])

@st.cache_data(ttl=30)
def fixtures_today(key):
    return api(key,"/fixtures",{"date":dt.datetime.now().strftime("%Y-%m-%d"),
                                "timezone":"Africa/Lagos"})

@st.cache_data(ttl=15)
def fixtures_live(key):
    return api(key,"/fixtures",{"live":"all"})

@st.cache_data(ttl=300)
def season_fixtures(key,league,season):
    return api(key,"/fixtures",{"league":league,"season":season})

@st.cache_data(ttl=1800)
def team_stats(key,team,league,season):
    x=api(key,"/teams/statistics",{"team":team,"league":league,"season":season})
    return x[0] if x else {}

@st.cache_data(ttl=3600)
def predictions(key,fid):
    x=api(key,"/predictions",{"fixture":fid})
    return x[0] if x else {}

@st.cache_data(ttl=180)
def fixture_odds(key,fid,live=False):
    return api(key,"/odds/live" if live else "/odds",{"fixture":fid})

@st.cache_data(ttl=3600)
def h2h(key,a,b,last=10):
    return api(key,"/fixtures/headtohead",{"h2h":f"{a}-{b}","last":last})

# ========================= MODEL =========================
def clamp(x): return max(.0005,min(.9995,float(x)))
def pois(k,l): return math.exp(-l)*l**k/math.factorial(k)
def formscore(s):
    if not s:return .5
    a=[{"W":1,"D":.5,"L":0}.get(x,.5) for x in str(s)[-5:]]
    return sum(a)/len(a) if a else .5
def matrix(h,a,n=10):
    m=np.outer([pois(i,h) for i in range(n+1)],[pois(j,a) for j in range(n+1)])
    return m/m.sum()
def markets(h,a):
    m=matrix(h,a); p={}
    p["home_win"]=np.tril(m,-1).sum()
    p["draw"]=np.trace(m); p["away_win"]=np.triu(m,1).sum()
    p["1x"]=p["home_win"]+p["draw"]; p["x2"]=p["draw"]+p["away_win"]
    p["12"]=1-p["draw"]
    for n in [.5,1.5,2.5,3.5,4.5,5.5,6.5]:
        k=str(n).replace(".","_")
        o=sum(m[i,j] for i in range(11) for j in range(11) if i+j>n)
        p[f"over_{k}"]=o; p[f"under_{k}"]=1-o
    p["btts_yes"]=sum(m[i,j] for i in range(1,11) for j in range(1,11))
    p["btts_no"]=1-p["btts_yes"]
    p["home_to_score"]=1-sum(m[0,j] for j in range(11))
    p["away_to_score"]=1-sum(m[i,0] for i in range(11))
    return {k:clamp(v) for k,v in p.items()}

def avg_goal(stats,side):
    try:return float(stats["goals"][side]["average"]["total"])
    except:return None

def context(key,f):
    h=f["teams"]["home"]; a=f["teams"]["away"]
    lg=f["league"]["id"]; season=f["league"]["season"]
    hs=team_stats(key,h["id"],lg,season); aas=team_stats(key,a["id"],lg,season)
    hgf=avg_goal(hs,"for") or 1.25; hga=avg_goal(hs,"against") or 1.10
    agf=avg_goal(aas,"for") or 1.20; aga=avg_goal(aas,"against") or 1.15
    hf=hs.get("form",""); af=aas.get("form","")
    hx=max(.10,(.65*hgf+.35*aga)*(.92+.16*formscore(hf))*1.04)
    ax=max(.10,(.65*agf+.35*hga)*(.92+.16*formscore(af))*.98)
    return hx,ax,hf,af

def prediction_probs(pred):
    try:
        q=pred["predictions"]["percent"]
        return {k:clamp(float(str(q[k]).replace("%",""))/100)
                for k in ["home","draw","away"]}
    except:return {}

def flatten_odds(raw):
    rows=[]
    for item in raw:
        for bm in item.get("bookmakers",[]):
            for bet in bm.get("bets",[]):
                name=(bet.get("name") or "").lower()
                for v in bet.get("values",[]):
                    try:o=float(v.get("odd"))
                    except:continue
                    val=str(v.get("value",""))
                    mk=None
                    if name in ["match winner","1x2","fulltime result"]:
                        mk={"Home":"home_win","Draw":"draw","Away":"away_win",
                            "1":"home_win","X":"draw","2":"away_win"}.get(val)
                    elif "goals over/under" in name or "over/under" in name:
                        z=re.search(r"([0-9]+(?:\.[05])?)",val)
                        if z:
                            line=z.group(1).replace(".","_")
                            mk=("over_" if "over" in val.lower() else "under_")+line
                    elif "both teams" in name or "both teams to score" in name:
                        mk="btts_yes" if val.lower()=="yes" else "btts_no" if val.lower()=="no" else None
                    if mk: rows.append((mk,o,bm.get("name",""),val))
    return rows

def score_market(prob,odds):
    imp=1/odds
    edge=(prob-imp)*100
    ev=(prob*odds-1)*100
    return imp,edge,ev

# ========================= RISK ENGINE =========================
def kelly(p,odds):
    b=odds-1
    return max(0,(p*odds-1)/b) if b>0 else 0

def stake(bankroll,p,odds,frac=.25,cap=0.02):
    raw=bankroll*kelly(p,odds)*frac
    return min(raw,bankroll*cap)

def risk_label(ev,confidence):
    if confidence>=80 and ev>=8:return "A — Strong value"
    if confidence>=72 and ev>=4:return "B — Good value"
    if confidence>=65 and ev>=2:return "C — Marginal"
    return "PASS"

def scan(key,fixtures,live=False,minp=.68,minev=3,bankroll=100000):
    out=[]
    for f in fixtures:
        try:
            fid=f["fixture"]["id"]; hx,ax,hform,aform=context(key,f)
            p=markets(hx,ax)
            pr=prediction_probs(predictions(key,fid))
            if pr:
                p["home_win"]=clamp(.70*p["home_win"]+.30*pr["home"])
                p["draw"]=clamp(.70*p["draw"]+.30*pr["draw"])
                p["away_win"]=clamp(.70*p["away_win"]+.30*pr["away"])
            odds=flatten_odds(fixture_odds(key,fid,live))
            best={}
            for mk,o,bm,val in odds:
                if mk in p and (mk not in best or o>best[mk][0]):best[mk]=(o,bm,val)
            for mk,(o,bm,val) in best.items():
                imp,edge,ev=score_market(p[mk],o)
                conf=100*(.70*p[mk]+.18*clamp((ev+5)/25)+.12*clamp((edge+5)/20))
                if p[mk]>=minp and ev>=minev:
                    st=stake(bankroll,p[mk],o)
                    out.append({
                        "Fixture ID":fid,"Home":f["teams"]["home"]["name"],
                        "Away":f["teams"]["away"]["name"],
                        "Market":mk.replace("_"," ").title(),"Selection":val,
                        "Odds":round(o,2),"Probability %":round(p[mk]*100,2),
                        "Implied %":round(imp*100,2),"Edge %":round(edge,2),
                        "EV %":round(ev,2),"Confidence":round(conf,2),
                        "Risk":risk_label(ev,conf),"Suggested Stake":round(st,2),
                        "Bookmaker":bm,"Model xG":f"{hx:.2f}-{ax:.2f}",
                        "Form":f"{hform}/{aform}"
                    })
        except Exception:
            continue
    if not out:return pd.DataFrame()
    return pd.DataFrame(out).sort_values(["Confidence","EV %"],ascending=False).drop_duplicates(["Home","Away","Market"])

# ========================= BACKTEST =========================
def backtest_csv(df):
    required={"date","home","away","market","odds","result"}
    missing=required-set(df.columns)
    if missing: raise ValueError("Missing columns: "+", ".join(sorted(missing)))
    d=df.copy()
    d["odds"]=pd.to_numeric(d["odds"],errors="coerce")
    d=d.dropna(subset=["odds"])
    def win(r):
        res=str(r["result"]).lower()
        mk=str(r["market"]).lower()
        if mk=="home_win": return res=="home"
        if mk=="draw": return res=="draw"
        if mk=="away_win": return res=="away"
        if mk=="over_2_5": return res=="over_2.5"
        if mk=="under_2_5": return res=="under_2.5"
        if mk=="btts_yes": return res=="yes"
        if mk=="btts_no": return res=="no"
        return np.nan
    d["win"]=d.apply(win,axis=1)
    d=d.dropna(subset=["win"])
    d["pnl"]=np.where(d["win"],d["odds"]-1,-1)
    return d

# ========================= UI =========================
st.title("⚽ Football Banker AI — Professional v7")
st.caption("Live scanning • EV • risk management • bankroll • backtesting • saved odds snapshots")

with st.sidebar:
    st.header("API-Sports")
    key=st.secrets.get("API_FOOTBALL_KEY", os.getenv("API_FOOTBALL_KEY", ""))
    if key:
        st.success("API-Sports key loaded from Streamlit Secrets")
    page=st.radio("Module",["🏆 Live Scanner","📈 Backtest","💰 Bankroll","🗃️ Odds Capture"])
    bankroll=st.number_input("Bankroll",min_value=1000.0,value=100000.0,step=1000.0)
    minp=st.slider("Minimum probability",.50,.95,.68,.01)
    minev=st.slider("Minimum EV %",0.,30.,3.,.5)
if not key:
    st.error("API-Sports key not found. Add API_FOOTBALL_KEY to Streamlit Secrets and reboot the app.")
    st.stop()

if page=="🏆 Live Scanner":
    mode=st.radio("Fixture source",["Today's fixtures","Live matches"],horizontal=True)
    limit=st.slider("Fixtures to scan",5,25,10)
    if st.button("🔄 Refresh"):
        st.cache_data.clear(); st.rerun()
    try:
        fx=(fixtures_live(key) if mode=="Live matches" else fixtures_today(key))[:limit]
        st.metric("Fixtures",len(fx))
        df=scan(key,fx,mode=="Live matches",minp,minev,bankroll)
        if len(df):
            st.subheader("🏆 TOP 5–10 BANKERS")
            st.dataframe(df.head(10),use_container_width=True,hide_index=True)
            st.download_button("Download banker report",df.to_csv(index=False),"banker_report.csv","text/csv")
            st.subheader("All qualifying +EV bets")
            st.dataframe(df,use_container_width=True,hide_index=True)
        else: st.info("No selection passed your probability + EV thresholds. NO BET is a valid output.")
    except Exception as e: st.error(str(e))

elif page=="📈 Backtest":
    st.header("Historical Backtesting")
    st.write("Use recorded odds/results to test whether the strategy actually has an edge. API-Sports only retains pre-match odds for the last 7 days, so long-horizon ROI testing requires odds captured by this app or an imported historical dataset.")
    template=pd.DataFrame(columns=["date","home","away","market","odds","result"])
    st.download_button("Download backtest CSV template",template.to_csv(index=False),"backtest_template.csv","text/csv")
    up=st.file_uploader("Upload historical odds/results CSV",type=["csv"])
    if up:
        try:
            bt=backtest_csv(pd.read_csv(up))
            if len(bt):
                n=len(bt); wins=int(bt["win"].sum()); roi=bt["pnl"].sum()/n*100
                st.columns(4)[0].metric("Bets",n)
                st.columns(4)[1].metric("Hit rate",f"{wins/n*100:.1f}%")
                st.columns(4)[2].metric("ROI / unit",f"{roi:.2f}%")
                st.columns(4)[3].metric("Net units",f"{bt['pnl'].sum():.2f}")
                bt["equity"]=bt["pnl"].cumsum()
                st.line_chart(bt.set_index(pd.to_datetime(bt["date"]))["equity"])
                st.dataframe(bt,use_container_width=True,hide_index=True)
            else: st.warning("No valid rows.")
        except Exception as e: st.error(str(e))

elif page=="💰 Bankroll":
    st.header("Bankroll & Risk Management")
    con=db(); bets=pd.read_sql_query("select * from bets order by placed_at desc",con)
    if len(bets):
        pnl=bets["pnl"].fillna(0).sum()
        st.metric("Recorded P&L",f"{pnl:,.2f}")
        st.metric("Recorded bets",len(bets))
        bets["equity"]=bets["pnl"].fillna(0).cumsum()
        st.line_chart(bets["equity"])
        st.dataframe(bets,use_container_width=True,hide_index=True)
    else: st.info("No settled bets recorded yet.")
    st.subheader("Risk rules")
    st.write("Default suggestion: quarter-Kelly with a 2% bankroll cap per selection. This is a risk-control heuristic, not a guarantee of profit. Avoid staking more because a pick is labelled 'banker'.")

elif page=="🗃️ Odds Capture":
    st.header("Odds Snapshot Capture")
    st.write("API-Sports says live odds have no historical archive. This module stores snapshots locally so you can build your own odds-movement and ROI dataset.")
    limit=st.slider("Fixtures to capture",1,20,5)
    if st.button("Capture today's odds now"):
        try:
            fx=fixtures_today(key)[:limit]; con=db(); now=dt.datetime.now().isoformat(timespec="seconds")
            count=0
            for f in fx:
                fid=f["fixture"]["id"]
                for mk,o,bm,val in flatten_odds(fixture_odds(key,fid)):
                    con.execute("insert into odds_snapshots values(?,?,?,?,?,?,?,?,'OPEN')",
                                (now,fid,f["teams"]["home"]["name"],f["teams"]["away"]["name"],mk,val,o,bm));count+=1
            con.commit(); st.success(f"Captured {count} odds records.")
        except Exception as e: st.error(str(e))
    con=db(); snap=pd.read_sql_query("select * from odds_snapshots order by ts desc limit 500",con)
    st.dataframe(snap,use_container_width=True,hide_index=True)

st.divider()
st.caption("Model outputs are probabilistic estimates, not guarantees. Use historical validation, conservative exposure limits and a strict no-bet threshold.")
