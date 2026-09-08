
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
    page_title="台股量價籌碼分析系統",
    page_icon="📈",
    layout="wide",
)

TWSE_UNIVERSE = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_UNIVERSE = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"

# ---------- UI ----------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1500px;}
[data-testid="stMetric"] {
  background: rgba(30,38,60,.42);
  border: 1px solid rgba(120,140,190,.22);
  padding: 12px;
  border-radius: 14px;
}
.small-note {opacity:.72; font-size:.85rem}
.score-good {color:#24c78e;font-weight:800}
.score-mid {color:#f0b84b;font-weight:800}
.score-bad {color:#f0646b;font-weight:800}
</style>
""", unsafe_allow_html=True)

# ---------- Data helpers ----------
def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # Single ticker sometimes still returns a MultiIndex.
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
    out = out[out["Volume"].fillna(0) >= 0]
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def load_universe():
    rows = []
    headers = {"User-Agent": "Mozilla/5.0"}

    # TWSE
    try:
        r = requests.get(TWSE_UNIVERSE, timeout=15, headers=headers)
        r.raise_for_status()
        data = r.json()
        for x in data:
            code = str(x.get("公司代號", "")).strip()
            name = str(x.get("公司簡稱", x.get("公司名稱", ""))).strip()
            industry = str(x.get("產業別", "")).strip()
            if code:
                rows.append({"代碼": code, "名稱": name, "市場": "上市", "Yahoo": f"{code}.TW", "產業": industry})
    except Exception:
        pass

    # TPEx
    try:
        r = requests.get(TPEX_UNIVERSE, timeout=15, headers=headers)
        r.raise_for_status()
        data = r.json()
        for x in data:
            # The TPEx endpoint has used both Chinese and English-like field labels in different releases.
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

    uni = pd.DataFrame(rows).drop_duplicates(subset=["代碼", "市場"])
    if not uni.empty:
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

    # Fallback: listed first, then OTC will be checked by caller.
    return f"{code}.TW", ""

def calc_kd(df: pd.DataFrame):
    if df.empty or len(df) < 10:
        return np.nan, np.nan
    ll = df["Low"].rolling(9, min_periods=9).min()
    hh = df["High"].rolling(9, min_periods=9).max()
    den = (hh - ll).replace(0, np.nan)
    rsv = ((df["Close"] - ll) / den * 100).fillna(50)

    k_vals, d_vals = [], []
    k_prev = d_prev = 50.0
    for v in rsv:
        k_now = (2/3) * k_prev + (1/3) * float(v)
        d_now = (2/3) * d_prev + (1/3) * k_now
        k_vals.append(k_now)
        d_vals.append(d_now)
        k_prev, d_prev = k_now, d_now
    return float(k_vals[-1]), float(d_vals[-1])

def resample_ohlcv(df: pd.DataFrame, rule: str):
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
    # Compare today's volume with the average of the PRIOR N sessions, excluding today.
    work["BaseVol"] = work["Volume"].shift(1).rolling(baseline_days).mean()
    work["VolRatio"] = work["Volume"] / work["BaseVol"].replace(0, np.nan)

    recent = work.tail(recent_days).dropna(subset=["VolRatio"])
    candidates = recent[recent["VolRatio"] >= volume_multiple]
    if candidates.empty:
        # Still report the largest recent ratio as a "near candidate".
        idx = recent["VolRatio"].idxmax()
        row = recent.loc[idx]
        qualifies = False
    else:
        idx = candidates["VolRatio"].idxmax()
        row = candidates.loc[idx]
        qualifies = True

    pos = work.index.get_loc(idx)
    after = work.iloc[pos + 1:]
    return {
        "date": idx,
        "row": row,
        "after": after,
        "ratio": float(row["VolRatio"]),
        "qualifies": qualifies,
    }

def analyze_df(df, volume_multiple=2.0, baseline_days=20, recent_days=30):
    if df.empty or len(df) < 80:
        return None

    hv = find_high_volume_k(df, volume_multiple, baseline_days, recent_days)
    if hv is None:
        return None

    row = hv["row"]
    after = hv["after"]
    current = df.iloc[-1]
    hv_low = float(row["Low"])
    hv_high = float(row["High"])
    hv_mid = (hv_low + hv_high) / 2
    current_close = float(current["Close"])

    if after.empty:
        min_after = current_close
        after_avg_vol = float(current["Volume"])
    else:
        min_after = float(after["Low"].min())
        after_avg_vol = float(after["Volume"].mean())

    not_broken = min_after >= hv_low * 0.995  # 0.5% tolerance for intraday noise
    close_above_low = current_close >= hv_low
    close_above_mid = current_close >= hv_mid
    breakout = current_close > hv_high
    volume_contract = after_avg_vol < float(row["Volume"]) if float(row["Volume"]) > 0 else False

    weekly = resample_ohlcv(df, "W-FRI")
    monthly = resample_ohlcv(df, "ME")
    wk, wd = calc_kd(weekly)
    mk, md = calc_kd(monthly)

    score_parts = {}
    score_parts["高量K達門檻"] = 15 if hv["qualifies"] else 0
    score_parts["高量K低點未破"] = 25 if not_broken else 0
    score_parts["現價站回高量K低點"] = 10 if close_above_low else 0
    score_parts["現價站高量K中值"] = 8 if close_above_mid else 0
    score_parts["後續量縮"] = 12 if volume_contract else 0
    score_parts["週K值低於30"] = 10 if (not math.isnan(wk) and wk < 30) else 0
    score_parts["月K值低於30"] = 10 if (not math.isnan(mk) and mk < 30) else 0
    score_parts["突破高量K高點"] = 10 if breakout else 0
    score = int(sum(score_parts.values()))

    if not close_above_low:
        status = "高量結構轉弱"
        grade = "偏弱"
    elif score >= 75:
        status = "高量不破・強勢"
        grade = "偏多"
    elif score >= 60:
        status = "高量不破・觀察突破"
        grade = "偏多整理"
    elif score >= 45:
        status = "高量後整理"
        grade = "中性"
    else:
        status = "訊號不足"
        grade = "觀察"

    return {
        "score": score,
        "status": status,
        "grade": grade,
        "current_close": current_close,
        "hv_date": hv["date"],
        "hv_low": hv_low,
        "hv_high": hv_high,
        "hv_open": float(row["Open"]),
        "hv_close": float(row["Close"]),
        "hv_ratio": hv["ratio"],
        "qualifies": hv["qualifies"],
        "not_broken": not_broken,
        "volume_contract": volume_contract,
        "breakout": breakout,
        "wk": wk,
        "wd": wd,
        "mk": mk,
        "md": md,
        "score_parts": score_parts,
    }

def candlestick_chart(df, result):
    show = df.tail(90)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=show.index,
        open=show["Open"], high=show["High"],
        low=show["Low"], close=show["Close"],
        name="日K"
    ))

    hv_date = result["hv_date"]
    fig.add_hline(y=result["hv_low"], line_dash="dot",
                  annotation_text=f"高量K低點 {result['hv_low']:.2f}")
    fig.add_hline(y=result["hv_high"], line_dash="dot",
                  annotation_text=f"高量K高點 {result['hv_high']:.2f}")
    if hv_date in show.index:
        fig.add_vline(x=hv_date, line_dash="dash",
                      annotation_text=f"高量K {pd.Timestamp(hv_date).strftime('%Y-%m-%d')}")

    fig.update_layout(
        height=510,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        hovermode="x unified",
    )
    return fig

def score_color(score):
    if score >= 75:
        return "🟢"
    if score >= 60:
        return "🟩"
    if score >= 45:
        return "🟡"
    return "🔴"

# ---------- Header ----------
st.title("📈 台股量價籌碼分析系統")
st.caption("真實行情版 v2｜高量K × 週/月KD × 量價結構 × 自動評分 × 選股掃描")
st.info("資料來源：Yahoo Finance 歷史行情；上市/上櫃公司名單使用 TWSE、TPEx OpenAPI。技術訊號僅作研究與篩選，不代表買賣建議。")

with st.sidebar:
    st.header("判讀參數")
    volume_multiple = st.slider("高量門檻（相對前20日均量）", 1.5, 4.0, 2.0, 0.1)
    recent_days = st.slider("尋找最近幾個交易日的高量K", 10, 60, 30, 5)
    st.markdown("---")
    st.markdown("""
    **目前評分邏輯**
    - 高量達門檻：15
    - 高量低點未破：25
    - 現價站回高量低點：10
    - 現價站高量K中值：8
    - 後續量縮：12
    - 週K < 30：10
    - 月K < 30：10
    - 突破高量K高點：10
    """)

universe = load_universe()
tab1, tab2, tab3 = st.tabs(["🔎 單檔分析", "🧭 選股掃描", "📘 判讀規則"])

# ---------- Single stock ----------
with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        code = st.text_input("股票代碼", value="2330", placeholder="例如：2330、1301、6488")
    with c2:
        run = st.button("開始分析", type="primary", use_container_width=True)

    if run or code:
        symbol, known_name = resolve_symbol(code, universe)
        with st.spinner(f"正在讀取 {symbol} 歷史行情…"):
            df = fetch_history(symbol, "2y")
            if df.empty and symbol.endswith(".TW"):
                symbol2 = f"{str(code).strip()}.TWO"
                df2 = fetch_history(symbol2, "2y")
                if not df2.empty:
                    symbol = symbol2
                    df = df2

        if df.empty:
            st.error("抓不到這檔股票的歷史行情。請確認股票代碼，或稍後再試。")
        else:
            name = known_name
            if not name and not universe.empty:
                raw_code = symbol.replace(".TW", "").replace(".TWO", "")
                m = universe[universe["代碼"] == raw_code]
                if not m.empty:
                    name = m.iloc[0]["名稱"]

            result = analyze_df(df, volume_multiple, 20, recent_days)
            if result is None:
                st.warning("歷史資料不足，暫時無法完成判讀。")
            else:
                st.subheader(f"{symbol.replace('.TW','').replace('.TWO','')} {name}".strip())
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("最新收盤", f"{result['current_close']:.2f}")
                m2.metric("總分", f"{result['score']} / 100")
                m3.metric("高量倍數", f"{result['hv_ratio']:.2f}×")
                m4.metric("週KD", f"K {result['wk']:.1f} / D {result['wd']:.1f}")
                m5.metric("月KD", f"K {result['mk']:.1f} / D {result['md']:.1f}")

                if result["score"] >= 75:
                    st.success(f"🟢 系統判讀：{result['status']}｜{result['grade']}")
                elif result["score"] >= 45:
                    st.warning(f"🟡 系統判讀：{result['status']}｜{result['grade']}")
                else:
                    st.error(f"🔴 系統判讀：{result['status']}｜{result['grade']}")

                st.plotly_chart(candlestick_chart(df, result), use_container_width=True)

                left, right = st.columns([1.2, 1])
                with left:
                    st.markdown("#### 高量K關鍵資料")
                    key_df = pd.DataFrame([
                        ["高量K日期", pd.Timestamp(result["hv_date"]).strftime("%Y-%m-%d")],
                        ["開盤", f"{result['hv_open']:.2f}"],
                        ["最高", f"{result['hv_high']:.2f}"],
                        ["最低", f"{result['hv_low']:.2f}"],
                        ["收盤", f"{result['hv_close']:.2f}"],
                        ["相對前20日均量", f"{result['hv_ratio']:.2f}×"],
                        ["後續是否跌破", "否 ✅" if result["not_broken"] else "是 ⚠️"],
                        ["後續是否量縮", "是 ✅" if result["volume_contract"] else "否"],
                        ["是否突破高量K高點", "是 ✅" if result["breakout"] else "否"],
                    ], columns=["項目", "結果"])
                    st.dataframe(key_df, hide_index=True, use_container_width=True)

                with right:
                    st.markdown("#### 評分拆解")
                    score_df = pd.DataFrame(
                        [{"條件": k, "得分": v} for k, v in result["score_parts"].items()]
                    )
                    st.dataframe(score_df, hide_index=True, use_container_width=True)

                st.markdown("#### 風險位置")
                st.write(
                    f"目前系統把 **{result['hv_low']:.2f}** 視為這根高量K的重要防守價。"
                    f"若後續收盤明顯跌破，『高量不破』的判讀會失效；"
                    f"若重新突破 **{result['hv_high']:.2f}**，則代表價格重新站上高量K壓力區。"
                )

# ---------- Scanner ----------
def scan_one(row, volume_multiple, recent_days):
    symbol = row["Yahoo"]
    df = fetch_history(symbol, "1y")
    result = analyze_df(df, volume_multiple, 20, recent_days)
    if result is None:
        return None
    return {
        "代碼": row["代碼"],
        "名稱": row["名稱"],
        "市場": row["市場"],
        "分數": result["score"],
        "判讀": result["status"],
        "最新收盤": round(result["current_close"], 2),
        "高量倍數": round(result["hv_ratio"], 2),
        "高量K日期": pd.Timestamp(result["hv_date"]).strftime("%Y-%m-%d"),
        "高量K低點": round(result["hv_low"], 2),
        "未跌破": "是" if result["not_broken"] else "否",
        "後續量縮": "是" if result["volume_contract"] else "否",
        "週K": round(result["wk"], 1),
        "月K": round(result["mk"], 1),
        "突破高量K": "是" if result["breakout"] else "否",
    }

with tab2:
    st.markdown("### 量價選股掃描")
    st.caption("先用自選清單快速掃描最實用；也可載入交易所公司名單。全市場大量下載時，Yahoo Finance 可能暫時限流。")

    source = st.radio("掃描範圍", ["自選股票", "交易所名單"], horizontal=True)

    scan_rows = pd.DataFrame()
    if source == "自選股票":
        watch = st.text_area(
            "輸入股票代碼（逗號、空白或換行分隔）",
            value="2330, 2317, 2454, 2308, 2382, 3231, 1301, 2881, 2882, 2603"
        )
        codes = [x.strip() for x in watch.replace(",", " ").replace("\n", " ").split() if x.strip()]
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
            markets = st.multiselect("市場", ["上市", "上櫃"], default=["上市", "上櫃"])
            filtered = universe[universe["市場"].isin(markets)].copy()
            limit_opt = st.selectbox("本次掃描數量", [50, 100, 200, 300, "全部"], index=1)
            if limit_opt != "全部":
                filtered = filtered.head(int(limit_opt))
            scan_rows = filtered

    min_score = st.slider("最低顯示分數", 0, 100, 60, 5)
    only_not_broken = st.checkbox("只顯示高量K未跌破", value=True)

    if st.button("執行掃描", type="primary", key="scan"):
        if scan_rows.empty:
            st.warning("沒有可掃描的股票。")
        else:
            results = []
            progress = st.progress(0)
            status = st.empty()
            total = len(scan_rows)

            # Moderate concurrency: useful, but less likely to trigger data-source throttling.
            with ThreadPoolExecutor(max_workers=6) as ex:
                futures = {
                    ex.submit(scan_one, row, volume_multiple, recent_days): i
                    for i, (_, row) in enumerate(scan_rows.iterrows(), start=1)
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
                    progress.progress(done / total)
                    status.caption(f"已完成 {done} / {total}")

            out = pd.DataFrame(results)
            if out.empty:
                st.warning("本次沒有取得可判讀的結果；可能是資料來源暫時限流。")
            else:
                out = out[out["分數"] >= min_score]
                if only_not_broken:
                    out = out[out["未跌破"] == "是"]
                out = out.sort_values(["分數", "高量倍數"], ascending=[False, False]).reset_index(drop=True)
                out.insert(0, "排名", range(1, len(out) + 1))

                st.success(f"掃描完成，共找到 {len(out)} 檔符合目前條件。")
                st.dataframe(out, hide_index=True, use_container_width=True, height=600)

                csv = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "下載掃描結果 CSV",
                    data=csv,
                    file_name=f"台股量價掃描_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )

with tab3:
    st.markdown("""
### 這套程式怎麼判斷？

**1. 先找近期的高量K**  
系統把當日成交量與「前 20 個交易日平均成交量」比較。預設達到 2 倍以上，就視為異常高量候選。

**2. 再看高量K之後有沒有破低**  
影片裡最重要的概念不是「看到爆量就買」，而是高量出現後，價格是否守得住那根K棒。  
目前程式把高量K最低價視為重要防守區，並保留 0.5% 的盤中雜訊容忍。

**3. 看後續是否量縮**  
如果高量之後整理時成交量下降，通常比「一路爆量下跌」健康，因此納入加分。

**4. 加入週KD與月KD**  
日線量價偏短線，所以再加入週、月尺度，避免只看一兩天的訊號。

**5. 用評分，而不是直接宣告洗盤或出貨**  
「洗盤／出貨」是市場解讀，不可能只靠一根K棒百分之百確認。這版因此使用 0–100 分，把不同證據疊加，再分成偏多、整理、中性、偏弱。

### 建議的實際用法
先用掃描器找出 60–75 分以上的股票，再進單檔分析看高量K位置、KD與後續價格。  
真正下決策前，仍應搭配基本面、產業趨勢、法人籌碼與大盤環境。
""")
