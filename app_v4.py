
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="台股量價籌碼與技術分析系統 V4",
    page_icon="📈",
    layout="wide",
)

TWSE_UNIVERSE = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_UNIVERSE = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"

FIB_RATIOS = [0.191, 0.382, 0.5, 0.618, 0.809]

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
[data-testid="stMetric"] {
  background: rgba(30,38,60,.42);
  border: 1px solid rgba(120,140,190,.22);
  padding: 12px;
  border-radius: 14px;
}
.small-note {opacity:.72; font-size:.85rem}
</style>
""", unsafe_allow_html=True)

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    out = df[needed].copy()
    for c in needed:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0)
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_universe():
    rows = []
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(TWSE_UNIVERSE, timeout=15, headers=headers)
        r.raise_for_status()
        for x in r.json():
            code = str(x.get("公司代號", "")).strip()
            name = str(x.get("公司簡稱", x.get("公司名稱", ""))).strip()
            industry = str(x.get("產業別", "")).strip()
            if code:
                rows.append({"代碼": code, "名稱": name, "市場": "上市", "Yahoo": f"{code}.TW", "產業": industry})
    except Exception:
        pass

    try:
        r = requests.get(TPEX_UNIVERSE, timeout=15, headers=headers)
        r.raise_for_status()
        for x in r.json():
            code = str(
                x.get("SecuritiesCompanyCode",
                x.get("公司代號",
                x.get("公司代碼",
                x.get("Code", ""))))
            ).strip()
            name = str(
                x.get("CompanyAbbreviation",
                x.get("公司簡稱",
                x.get("公司名稱",
                x.get("Name", ""))))
            ).strip()
            industry = str(x.get("SecuritiesIndustryCode", x.get("產業別", ""))).strip()
            if code:
                rows.append({"代碼": code, "名稱": name, "市場": "上櫃", "Yahoo": f"{code}.TWO", "產業": industry})
    except Exception:
        pass

    uni = pd.DataFrame(rows)
    if not uni.empty:
        uni = uni.drop_duplicates(subset=["代碼", "市場"])
        uni["代碼"] = uni["代碼"].astype(str)
    return uni

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol: str, period="2y"):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=15,
        )
        return clean_ohlcv(df)
    except Exception:
        return pd.DataFrame()

def resolve_symbol(code: str, universe: pd.DataFrame):
    code = str(code).strip().upper()
    if code.endswith(".TW") or code.endswith(".TWO"):
        return code, ""
    if not universe.empty:
        m = universe[universe["代碼"] == code]
        if not m.empty:
            row = m.iloc[0]
            return row["Yahoo"], row["名稱"]
    return f"{code}.TW", ""

def calc_kd(df: pd.DataFrame):
    if df.empty or len(df) < 10:
        return np.nan, np.nan
    ll = df["Low"].rolling(9, min_periods=9).min()
    hh = df["High"].rolling(9, min_periods=9).max()
    den = (hh - ll).replace(0, np.nan)
    rsv = ((df["Close"] - ll) / den * 100).fillna(50)

    k_prev = d_prev = 50.0
    k_vals, d_vals = [], []
    for v in rsv:
        k_now = (2/3) * k_prev + (1/3) * float(v)
        d_now = (2/3) * d_prev + (1/3) * k_now
        k_vals.append(k_now)
        d_vals.append(d_now)
        k_prev, d_prev = k_now, d_now
    return float(k_vals[-1]), float(d_vals[-1])

def resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

def find_high_volume_k(df, volume_multiple=2.0, baseline_days=20, recent_days=30):
    if len(df) < baseline_days + 5:
        return None
    work = df.copy()
    work["BaseVol"] = work["Volume"].shift(1).rolling(baseline_days).mean()
    work["VolRatio"] = work["Volume"] / work["BaseVol"].replace(0, np.nan)
    recent = work.tail(recent_days).dropna(subset=["VolRatio"])
    if recent.empty:
        return None
    candidates = recent[recent["VolRatio"] >= volume_multiple]
    if candidates.empty:
        idx = recent["VolRatio"].idxmax()
        qualifies = False
    else:
        idx = candidates["VolRatio"].idxmax()
        qualifies = True
    row = recent.loc[idx]
    pos = work.index.get_loc(idx)
    after = work.iloc[pos+1:]
    return {"date": idx, "row": row, "after": after, "ratio": float(row["VolRatio"]), "qualifies": qualifies}

def moving_average_info(df):
    work = df.copy()
    for n in (5,10,20):
        work[f"MA{n}"] = work["Close"].rolling(n).mean()
    last = work.iloc[-1]
    ma5, ma10, ma20 = [float(last[f"MA{n}"]) for n in (5,10,20)]
    if ma5 > ma10 > ma20:
        state = "多頭排列"
    elif ma5 < ma10 < ma20:
        state = "空頭排列"
    else:
        state = "均線糾結"
    return work, ma5, ma10, ma20, state

def fibonacci_info(df):
    swing = df.tail(min(120, len(df))).copy()
    low_idx = swing["Low"].idxmin()
    high_idx = swing["High"].idxmax()
    low = float(swing.loc[low_idx, "Low"])
    high = float(swing.loc[high_idx, "High"])
    rng = max(high-low, 1e-9)
    upswing = low_idx < high_idx
    if upswing:
        levels = {r: high - rng*r for r in FIB_RATIOS}
    else:
        levels = {r: low + rng*r for r in FIB_RATIOS}
    close = float(df.iloc[-1]["Close"])
    nearest = min(levels, key=lambda r: abs(levels[r]-close))
    return {
        "low": low, "high": high, "upswing": upswing,
        "levels": levels, "nearest_ratio": nearest,
        "nearest_level": levels[nearest]
    }

def local_extrema(arr, mode="max", order=3):
    pts = []
    for i in range(order, len(arr)-order):
        w = arr[i-order:i+order+1]
        if mode == "max" and arr[i] == np.max(w):
            pts.append((i, float(arr[i])))
        elif mode == "min" and arr[i] == np.min(w):
            pts.append((i, float(arr[i])))
    return pts

def detect_patterns(df):
    recent = df.tail(80)
    highs = recent["High"].to_numpy()
    lows = recent["Low"].to_numpy()
    peaks = local_extrema(highs, "max")
    troughs = local_extrema(lows, "min")
    patterns = []

    if len(troughs) >= 2:
        a, b = troughs[-2], troughs[-1]
        sim = 1 - abs(a[1]-b[1]) / max(a[1], b[1])
        if b[0]-a[0] >= 8 and sim >= 0.97:
            conf = min(92, int(62 + max(0, sim-0.97)/0.03*25))
            patterns.append(("W底（雙底）", conf, "偏多"))

    if len(peaks) >= 2:
        a, b = peaks[-2], peaks[-1]
        sim = 1 - abs(a[1]-b[1]) / max(a[1], b[1])
        if b[0]-a[0] >= 8 and sim >= 0.97:
            conf = min(92, int(62 + max(0, sim-0.97)/0.03*25))
            patterns.append(("M頭（雙頂）", conf, "偏空"))

    if len(peaks) >= 3 and len(troughs) >= 3:
        px = np.array([x for x,_ in peaks[-4:]], dtype=float)
        py = np.array([y for _,y in peaks[-4:]], dtype=float)
        tx = np.array([x for x,_ in troughs[-4:]], dtype=float)
        ty = np.array([y for _,y in troughs[-4:]], dtype=float)
        ps = np.polyfit(px, py, 1)[0] / max(np.mean(py),1)
        ts = np.polyfit(tx, ty, 1)[0] / max(np.mean(ty),1)

        if abs(ps) < .0015 and ts > .001:
            patterns.append(("上升三角形", 74, "偏多"))
        elif ps < -.001 and abs(ts) < .0015:
            patterns.append(("下降三角形", 74, "偏空"))
        elif ps < -.001 and ts > .001:
            patterns.append(("三角收斂", 68, "等待"))
        elif abs(ps) < .0015 and abs(ts) < .0015:
            patterns.append(("箱型盤整", 64, "等待"))

    patterns = sorted(patterns, key=lambda x: x[1], reverse=True)[:3]
    return patterns or [("暫無高可信度型態", 0, "等待")]

def analyze_df(df, volume_multiple=2.0, baseline_days=20, recent_days=30):
    if df.empty or len(df) < 80:
        return None

    hv = find_high_volume_k(df, volume_multiple, baseline_days, recent_days)
    if hv is None:
        return None

    row = hv["row"]
    after = hv["after"]
    current = df.iloc[-1]
    hv_low, hv_high = float(row["Low"]), float(row["High"])
    hv_mid = (hv_low + hv_high)/2
    current_close = float(current["Close"])

    min_after = current_close if after.empty else float(after["Low"].min())
    after_avg_vol = float(current["Volume"]) if after.empty else float(after["Volume"].mean())

    not_broken = min_after >= hv_low*0.995
    close_above_low = current_close >= hv_low
    close_above_mid = current_close >= hv_mid
    breakout = current_close > hv_high
    volume_contract = after_avg_vol < float(row["Volume"]) if float(row["Volume"]) > 0 else False

    weekly = resample_ohlcv(df, "W-FRI")
    monthly = resample_ohlcv(df, "ME")
    wk, wd = calc_kd(weekly)
    mk, md = calc_kd(monthly)

    work, ma5, ma10, ma20, ma_state = moving_average_info(df)
    fib = fibonacci_info(df)
    patterns = detect_patterns(df)

    score_parts = {
        "高量K達門檻": 15 if hv["qualifies"] else 0,
        "高量K低點未破": 25 if not_broken else 0,
        "現價站回高量低點": 10 if close_above_low else 0,
        "現價站高量K中值": 8 if close_above_mid else 0,
        "後續量縮": 12 if volume_contract else 0,
        "週K值低於30": 10 if (not math.isnan(wk) and wk < 30) else 0,
        "月K值低於30": 10 if (not math.isnan(mk) and mk < 30) else 0,
        "突破高量K高點": 10 if breakout else 0,
        "MA5>MA10>MA20": 10 if ma_state == "多頭排列" else 0,
    }
    raw_score = sum(score_parts.values())
    score = min(100, int(round(raw_score / 110 * 100)))

    if not close_above_low:
        status, grade = "高量結構轉弱", "偏弱"
    elif score >= 78:
        status, grade = "多訊號共振", "偏多"
    elif score >= 62:
        status, grade = "高量不破・觀察突破", "偏多整理"
    elif score >= 45:
        status, grade = "高量後整理", "中性"
    else:
        status, grade = "訊號不足", "觀察"

    return {
        "score": score, "status": status, "grade": grade,
        "current_close": current_close,
        "hv_date": hv["date"], "hv_low": hv_low, "hv_high": hv_high,
        "hv_open": float(row["Open"]), "hv_close": float(row["Close"]),
        "hv_ratio": hv["ratio"], "qualifies": hv["qualifies"],
        "not_broken": not_broken, "volume_contract": volume_contract,
        "breakout": breakout,
        "wk": wk, "wd": wd, "mk": mk, "md": md,
        "score_parts": score_parts,
        "work": work, "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma_state": ma_state,
        "fib": fib, "patterns": patterns,
    }

def pattern_geometry(df):
    """Causal pivots: each pivot needs three subsequent bars before use.

    Boundaries are frozen at formation; confirmation uses subsequent close
    crossings with a 0.5% buffer. Similarity is geometry, not probability.
    """
    w = df.tail(90)
    if len(w) < 25:
        return []
    h, l, c = [w[k].to_numpy(dtype=float) for k in ('High', 'Low', 'Close')]
    if not np.isfinite(np.concatenate([h, l, c])).all():
        return []
    found = {}
    for end in range(24, len(w)):
        peaks = local_extrema(h[:end+1], 'max')
        lows = local_extrema(l[:end+1], 'min')
        candidates = []
        for name, pts, opposite, bullish in [
            ('W底', lows, h, True), ('M頭', peaks, l, False)
        ]:
            if len(pts) < 2:
                continue
            a, b = pts[-2:]
            if b[0]-a[0] < 8 or abs(a[1]-b[1])/max(a[1], b[1], 1e-9) > .03:
                continue
            middle = opposite[a[0]+1:b[0]]
            j = a[0]+1+int(np.argmax(middle) if bullish else np.argmin(middle))
            neck = float(opposite[j])
            base = min(a[1], b[1]) if bullish else max(a[1], b[1])
            height = abs(neck-base)
            if height < .03 * max(neck, 1e-9):
                continue
            candidates.append(dict(name=name, points=[a,(j,neck),b], start=a[0],
                upper=(0., neck if bullish else base), lower=(0., base if bullish else neck),
                height=height, direction=1 if bullish else -1,
                similarity=round(100*(1-abs(a[1]-b[1])/height),1)))
        if len(peaks) >= 3 and len(lows) >= 3:
            pp, tt = peaks[-3:], lows[-3:]
            up = np.polyfit(*np.array(pp).T, 1)
            down = np.polyfit(*np.array(tt).T, 1)
            start = min(pp[0][0], tt[0][0])
            span = end-start
            upper, lower = np.polyval(up,end), np.polyval(down,end)
            width = np.polyval(up,start)-np.polyval(down,start)
            if span >= 15 and width > .03*c[end] and upper > lower:
                error = max(max(abs(y-np.polyval(up,x)) for x,y in pp),
                            max(abs(y-np.polyval(down,x)) for x,y in tt)) / width
                pu, pl = up[0]*span/width, down[0]*span/width
                name = None
                if abs(pu) <= .15 and pl > .2:
                    name = '上升三角形'
                elif pu < -.2 and abs(pl) <= .15:
                    name = '下降三角形'
                elif pu < -.2 and pl > .2:
                    name = '三角收斂'
                elif abs(pu) <= .15 and abs(pl) <= .15:
                    name = '箱型盤整'
                # Require candles to respect the fitted boundaries during formation.
                xs = np.arange(start,end+1)
                contained = np.mean((h[start:end+1] <= np.polyval(up,xs)+.15*width) &
                                    (l[start:end+1] >= np.polyval(down,xs)-.15*width))
                if name and error <= .15 and contained >= .9:
                    candidates.append(dict(name=name, points=sorted(pp+tt), start=start,
                        upper=tuple(up), lower=tuple(down), height=float(width), direction=0,
                        similarity=round(100*(1-error),1)))
        for p in candidates:
            key = (p['name'], tuple(x for x,y in p['points']))
            if key in found:
                continue
            p.update(formed=end, event=None, target=None, status='未確認')
            # A bar must cross a frozen boundary AFTER the pivots became knowable.
            for k in range(end+1,len(w)):
                u, d = np.polyval(p['upper'],k), np.polyval(p['lower'],k)
                if u <= d:
                    break
                upward = c[k] > u*1.005 and c[k-1] <= np.polyval(p['upper'],k-1)*1.005
                downward = c[k] < d*.995 and c[k-1] >= np.polyval(p['lower'],k-1)*.995
                direction = 1 if upward else -1 if downward else 0
                if not direction:
                    continue
                if p['direction'] and direction != p['direction']:
                    p['status'] = '未確認（反向破壞）'
                    break
                p['event'] = k
                p['status'] = '已確認突破' if direction == 1 else '已確認跌破'
                target = (u if direction == 1 else d)+direction*p['height']
                p['target'] = float(target) if target > 0 else None
                p['event_direction'] = direction
                if (direction == 1 and c[-1] <= np.polyval(p['upper'],len(w)-1)) or (direction == -1 and c[-1] >= np.polyval(p['lower'],len(w)-1)):
                    p['status'] += '（現價已回到線內）'
                break
            found[key] = p
    # One recent formation per type; retain a recent confirmed pattern where present.
    result = []
    for name in ('W底','M頭','上升三角形','下降三角形','三角收斂','箱型盤整'):
        choices = [p for p in found.values() if p['name']==name and p['formed'] >= len(w)-45]
        if choices:
            result.append(max(choices,key=lambda p:(p['points'][-1][0],-p['formed'])))
    return result


def draw_pattern_geometry(fig, df, patterns):
    w = df.tail(90)
    colors = ['#e9a23b','#b677ee','#20bfa9','#e96a83','#4e9deb','#a3b83e']
    for p, color in zip(patterns, colors):
        group = p['name']
        pts = p['points']
        fig.add_trace(go.Scatter(x=[w.index[x] for x,y in pts], y=[y for x,y in pts],
            mode='markers+text', text=[f'{group}轉折{i+1}' for i in range(len(pts))],
            textposition='top center', marker=dict(color=color,size=8),
            name=f"{group}｜{p['status']}", legendgroup=group))
        finish = len(w)-1
        while finish > p['formed'] and np.polyval(p['upper'],finish) <= np.polyval(p['lower'],finish):
            finish -= 1
        for key,label in [('upper','壓力／上緣'),('lower','支撐／下緣')]:
            if p['name']=='W底' and key=='upper' or p['name']=='M頭' and key=='lower':
                label='頸線'
            xx=[p['start'],finish]
            fig.add_trace(go.Scatter(x=[w.index[x] for x in xx], y=np.polyval(p[key],xx),
                mode='lines',line=dict(color=color,dash='dash'),name=f'{group} {label}',legendgroup=group))
        if p['event'] is not None:
            k=p['event']
            fig.add_trace(go.Scatter(x=[w.index[k]],y=[w.iloc[k]['Close']],mode='markers',
                marker=dict(size=14,color=color,symbol='triangle-up' if p['event_direction']==1 else 'triangle-down'),
                name=f'{group} 確認收盤',legendgroup=group))
            if p['target'] is not None:
                fig.add_trace(go.Scatter(x=[w.index[k],w.index[-1]],y=[p['target']]*2,
                    mode='lines',line=dict(color=color,dash='dot'),
                    name=f"{group} 量測目標 {p['target']:.2f}（估算）",legendgroup=group))
    return fig


def candlestick_chart(df, result):
    show = result["work"].tail(90)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=show.index, open=show["Open"], high=show["High"],
        low=show["Low"], close=show["Close"], name="日K"
    ))
    for n in (5,10,20):
        fig.add_trace(go.Scatter(
            x=show.index, y=show[f"MA{n}"], mode="lines", name=f"MA{n}"
        ))
    for r, level in result["fib"]["levels"].items():
        fig.add_hline(y=level, line_dash="dot",
                      annotation_text=f"Fib {r:.3f} {level:.2f}")
    fig.add_hline(y=result["hv_low"], line_dash="dash",
                  annotation_text=f"高量K低點 {result['hv_low']:.2f}")
    fig.add_hline(y=result["hv_high"], line_dash="dash",
                  annotation_text=f"高量K高點 {result['hv_high']:.2f}")
    if result["hv_date"] in show.index:
        fig.add_shape(type="line", x0=result["hv_date"], x1=result["hv_date"], y0=0, y1=1, yref="paper", line=dict(dash="dash"))
    fig.update_layout(
        height=560, margin=dict(l=10,r=10,t=40,b=10),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend_orientation="h"
    )
    return draw_pattern_geometry(fig, df, pattern_geometry(df))

def scan_one(row, volume_multiple, recent_days):
    df = fetch_history(row["Yahoo"], "1y")
    result = analyze_df(df, volume_multiple, 20, recent_days)
    if result is None:
        return None
    p = result["patterns"][0]
    return {
        "代碼": row["代碼"], "名稱": row["名稱"], "市場": row["市場"],
        "分數": result["score"], "判讀": result["status"],
        "最新收盤": round(result["current_close"], 2),
        "高量倍數": round(result["hv_ratio"], 2),
        "高量K未跌破": "是" if result["not_broken"] else "否",
        "MA結構": result["ma_state"],
        "週K": round(result["wk"], 1),
        "月K": round(result["mk"], 1),
        "型態": p[0],
        "型態相似度": p[1],
        "最近Fib": f"{result['fib']['nearest_ratio']:.3f}",
        "突破高量K": "是" if result["breakout"] else "否",
    }

st.title("📈 台股量價籌碼與技術分析系統 V4")
st.caption("技術分析版 V4｜技術型態自動畫線 × 高量K × 週/月KD × MA5/10/20 × Fibonacci × 型態辨識 × 選股掃描")
st.info("資料來源：Yahoo Finance 歷史行情；上市/上櫃公司名單使用 TWSE、TPEx OpenAPI。技術訊號僅作研究與篩選，不代表買賣建議。")

with st.sidebar:
    st.header("判讀參數")
    volume_multiple = st.slider("高量門檻（相對前20日均量）", 1.5, 4.0, 2.0, 0.1)
    recent_days = st.slider("尋找最近幾個交易日的高量K", 10, 60, 30, 5)
    st.markdown("---")
    st.markdown("**Fibonacci 比例**")
    st.write("0.191 / 0.382 / 0.500 / 0.618 / 0.809")
    st.markdown("---")
    st.caption("V4 新增技術型態自動畫線；V3 保留：MA5/10/20、費波那契、型態辨識、掃描欄位擴充。")

universe = load_universe()
tab1, tab2, tab3 = st.tabs(["🔎 單檔分析", "🧭 選股掃描", "📘 判讀規則"])

with tab1:
    c1, c2 = st.columns([2,1])
    with c1:
        code = st.text_input("股票代碼", value="2330", placeholder="例如：2330、1301、6488")
    with c2:
        run = st.button("開始分析", type="primary", use_container_width=True)

    if run or code:
        symbol, known_name = resolve_symbol(code, universe)
        with st.spinner(f"正在讀取 {symbol} 歷史行情…"):
            df = fetch_history(symbol, "2y")
            if df.empty and symbol.endswith(".TW"):
                alt = f"{str(code).strip()}.TWO"
                df2 = fetch_history(alt, "2y")
                if not df2.empty:
                    symbol, df = alt, df2

        if df.empty:
            st.error("抓不到這檔股票的歷史行情。請確認股票代碼，或稍後再試。")
        else:
            name = known_name
            if not name and not universe.empty:
                raw = symbol.replace(".TW","").replace(".TWO","")
                m = universe[universe["代碼"] == raw]
                if not m.empty:
                    name = m.iloc[0]["名稱"]

            result = analyze_df(df, volume_multiple, 20, recent_days)
            if result is None:
                st.warning("歷史資料不足，暫時無法完成判讀。")
            else:
                st.subheader(f"{symbol.replace('.TW','').replace('.TWO','')} {name}".strip())

                a,b,c,d = st.columns(4)
                a.metric("最新收盤", f"{result['current_close']:.2f}")
                b.metric("綜合分數", f"{result['score']} / 100")
                c.metric("近期最大量", f"{result['hv_ratio']:.2f}×",
                         delta="達高量門檻" if result["qualifies"] else f"未達 {volume_multiple:.1f}× 門檻")
                d.metric("均線結構", result["ma_state"])

                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("MA5", f"{result['ma5']:.2f}")
                m2.metric("MA10", f"{result['ma10']:.2f}")
                m3.metric("MA20", f"{result['ma20']:.2f}")
                m4.metric("週KD", f"K {result['wk']:.1f}｜D {result['wd']:.1f}")
                m5.metric("月KD", f"K {result['mk']:.1f}｜D {result['md']:.1f}")

                if result["score"] >= 78:
                    st.success(f"🟢 系統判讀：{result['status']}｜{result['grade']}")
                elif result["score"] >= 45:
                    st.warning(f"🟡 系統判讀：{result['status']}｜{result['grade']}")
                else:
                    st.error(f"🔴 系統判讀：{result['status']}｜{result['grade']}")

                st.plotly_chart(candlestick_chart(df, result), use_container_width=True)

                f1, f2 = st.columns([1.15, 1])
                with f1:
                    st.markdown("### Fibonacci")
                    st.caption(
                        f"自動波段：{result['fib']['low']:.2f} → {result['fib']['high']:.2f}；"
                        f"目前最接近 Fib {result['fib']['nearest_ratio']:.3f}"
                    )
                    fib_df = pd.DataFrame([
                        {
                            "比例": f"{r:.3f}",
                            "價位": round(level,2),
                            "距現價%": round((level/result["current_close"]-1)*100,2)
                        }
                        for r, level in result["fib"]["levels"].items()
                    ])
                    st.dataframe(fib_df, hide_index=True, use_container_width=True)

                with f2:
                    st.markdown("### V4 技術型態自動畫線")
                    geometry = pattern_geometry(df)
                    if geometry:
                        st.dataframe(pd.DataFrame([{
                            "型態": p["name"], "確認狀態": p["status"],
                            "形成可知日": str(df.tail(90).index[p["formed"]].date()),
                            "確認日": str(df.tail(90).index[p["event"]].date()) if p["event"] is not None else "未確認",
                            "量測目標（估算）": p["target"],
                        } for p in geometry]), hide_index=True)
                    else:
                        st.info("未確認：近期沒有符合畫線條件的型態。")
                    st.caption("轉折點需右側3根K線確認；線段固定後，後續收盤跨越線外0.5%才標記突破／跌破。未加入成交量確認；當日尚未收盤時，訊號仍可能變動。目標為型態高度量測估算，不保證到達。")
                    st.markdown("### V3 型態相似度（保留）")
                    pat_df = pd.DataFrame([
                        {"型態": p, "相似度": f"{conf}%" if conf else "—", "方向": direction}
                        for p,conf,direction in result["patterns"]
                    ])
                    st.dataframe(pat_df, hide_index=True, use_container_width=True)
                    st.caption("型態是演算法相似度提示；未突破頸線／趨勢線前，不視為已確認訊號。")

                left, right = st.columns([1.2,1])
                with left:
                    st.markdown("### 高量K關鍵資料")
                    key_df = pd.DataFrame([
                        ["高量K日期", pd.Timestamp(result["hv_date"]).strftime("%Y-%m-%d")],
                        ["開盤", f"{result['hv_open']:.2f}"],
                        ["最高", f"{result['hv_high']:.2f}"],
                        ["最低", f"{result['hv_low']:.2f}"],
                        ["收盤", f"{result['hv_close']:.2f}"],
                        ["相對前20日均量", f"{result['hv_ratio']:.2f}×"],
                        ["是否達高量門檻", "是 ✅" if result["qualifies"] else "否"],
                        ["後續是否跌破", "否 ✅" if result["not_broken"] else "是 ⚠️"],
                        ["後續是否量縮", "是 ✅" if result["volume_contract"] else "否"],
                        ["是否突破高量K高點", "是 ✅" if result["breakout"] else "否"],
                    ], columns=["項目","結果"])
                    st.dataframe(key_df, hide_index=True, use_container_width=True)

                with right:
                    st.markdown("### 評分拆解")
                    score_df = pd.DataFrame(
                        [{"條件": k, "得分": v} for k,v in result["score_parts"].items()]
                    )
                    st.dataframe(score_df, hide_index=True, use_container_width=True)

                st.markdown("### 關鍵價位")
                st.write(
                    f"高量K防守價：**{result['hv_low']:.2f}** ｜ "
                    f"高量K突破價：**{result['hv_high']:.2f}** ｜ "
                    f"最近 Fibonacci：**{result['fib']['nearest_level']:.2f}**"
                )

with tab2:
    st.markdown("### 量價技術選股掃描")
    st.caption("先用自選清單快速掃描；全市場大量下載時，Yahoo Finance 可能暫時限流。")

    source = st.radio("掃描範圍", ["自選股票", "交易所名單"], horizontal=True)
    scan_rows = pd.DataFrame()

    if source == "自選股票":
        watch = st.text_area(
            "輸入股票代碼（逗號、空白或換行分隔）",
            value="2330, 2317, 2454, 2308, 2382, 3231, 1301, 2881, 2882, 2603"
        )
        codes = [x.strip() for x in watch.replace(","," ").replace("\n"," ").split() if x.strip()]
        rows = []
        for c in codes:
            sym, _ = resolve_symbol(c, universe)
            name = ""
            market = "上市" if sym.endswith(".TW") else "上櫃"
            if not universe.empty:
                m = universe[universe["代碼"] == c]
                if not m.empty:
                    name = m.iloc[0]["名稱"]
                    market = m.iloc[0]["市場"]
                    sym = m.iloc[0]["Yahoo"]
            rows.append({"代碼": c, "名稱": name, "市場": market, "Yahoo": sym})
        scan_rows = pd.DataFrame(rows)
    else:
        if universe.empty:
            st.error("目前無法取得交易所公司名單，請改用『自選股票』。")
        else:
            markets = st.multiselect("市場", ["上市","上櫃"], default=["上市","上櫃"])
            filtered = universe[universe["市場"].isin(markets)].copy()
            limit_opt = st.selectbox("本次掃描數量", [50,100,200,300,"全部"], index=1)
            if limit_opt != "全部":
                filtered = filtered.head(int(limit_opt))
            scan_rows = filtered

    min_score = st.slider("最低顯示分數", 0,100,60,5)
    only_not_broken = st.checkbox("只顯示高量K未跌破", value=True)

    if st.button("執行掃描", type="primary", key="scan"):
        if scan_rows.empty:
            st.warning("沒有可掃描的股票。")
        else:
            results = []
            progress = st.progress(0)
            status = st.empty()
            total = len(scan_rows)

            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {
                    ex.submit(scan_one, row, volume_multiple, recent_days): i
                    for i, (_,row) in enumerate(scan_rows.iterrows(), start=1)
                }
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    try:
                        r = fut.result()
                        if r:
                            results.append(r)
                    except Exception:
                        pass
                    progress.progress(done/total)
                    status.caption(f"已完成 {done} / {total}")

            out = pd.DataFrame(results)
            if out.empty:
                st.warning("本次沒有取得可判讀的結果；可能是資料來源暫時限流。")
            else:
                out = out[out["分數"] >= min_score]
                if only_not_broken:
                    out = out[out["高量K未跌破"] == "是"]
                out = out.sort_values(["分數","高量倍數"], ascending=[False,False]).reset_index(drop=True)
                out.insert(0, "排名", range(1,len(out)+1))

                st.success(f"掃描完成，共找到 {len(out)} 檔符合目前條件。")
                st.dataframe(out, hide_index=True, use_container_width=True, height=620)

                csv = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "下載掃描結果 CSV",
                    data=csv,
                    file_name=f"台股技術掃描_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

with tab3:
    st.markdown("""
### v3 判讀規則

**高量K**  
當日成交量與前20個交易日平均量比較；預設 2 倍以上才算達門檻。若沒有達門檻，系統仍會顯示近期最大量，但清楚標示「未達門檻」。

**MA5 / MA10 / MA20**  
- MA5 > MA10 > MA20：多頭排列
- MA5 < MA10 < MA20：空頭排列
- 其他：均線糾結

**Fibonacci**  
自動抓最近 120 個交易日的波段高低點，使用：
0.191 / 0.382 / 0.500 / 0.618 / 0.809

**型態辨識**  
目前先辨識：
- W底（雙底）
- M頭（雙頂）
- 上升三角形
- 下降三角形
- 三角收斂
- 箱型盤整

型態只提供「相似度」，不直接等同買賣訊號。

**使用方式**  
先用掃描器找 60 分以上股票，再進單檔分析檢查：
高量K是否守住、均線排列、週/月KD、Fibonacci位置與型態是否同方向。
""")
