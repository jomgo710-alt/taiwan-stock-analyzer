# 台股量價籌碼分析系統 v2

這是一個可連網抓取真實行情的 Streamlit 版本。

## 功能
- 輸入台股代碼，自動辨識上市 `.TW` / 上櫃 `.TWO`
- 抓取 Yahoo Finance 日線歷史行情
- 自動尋找近期異常高量 K
- 判斷高量 K 後是否跌破、是否量縮、是否突破
- 計算週 KD、月 KD
- 0–100 分量價評分
- 自選股票批次掃描
- 讀取 TWSE / TPEx 公司名單做市場掃描
- 掃描結果匯出 CSV

## Mac 安裝與啟動
1. 先安裝 Python 3.10 以上。
2. 打開「終端機 Terminal」。
3. `cd` 到這個資料夾。
4. 建立虛擬環境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

5. 安裝套件：

```bash
pip install -r requirements.txt
```

6. 啟動：

```bash
streamlit run app.py
```

瀏覽器會自動打開系統。

## 一鍵啟動
也可以雙擊 `啟動.command`。第一次仍需允許 macOS 執行；若沒有安裝 Python，請先完成 Python 安裝。

## 資料說明
歷史價格由 yfinance 讀取 Yahoo Finance；股票公司名單取自 TWSE、TPEx OpenAPI。
外部資料來源可能發生暫時限流、延遲或缺漏，因此此工具適合作為研究與篩選，不應視為券商即時交易報價。
