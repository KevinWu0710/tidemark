# 本地 Cache：怎麼存、怎麼讀、怎麼又小又快

## 1. 一句話

下載一次 → 寫成 **Parquet 檔** 在 `data/cache/` → 回測只 `read_parquet`，**不再打永豐**。

## 2. 目錄結構

```text
data/cache/
  TMF/
    tick/2026-10-20.parquet   # 可選：原始逐筆（最大）
    1s/2026-10-20.parquet     # 秒 K（中）
    1m/2026-10-20.parquet     # 分 K（小，訊號研究首選）
    daily/...                 # 日線（最小）
  2330/
    1m/...
    daily/...
```

- **一檔一天**：改一天、缺一天都好補；回測按日期區間載入再串起來。
- **依 symbol 分開**：TMF 與股票互不覆蓋。
- **空檔 = 失效**：抓到空資料會刪／不當作有效 cache。

## 3. 回測怎麼讀（不下載）

```text
scripts/run_backtest.py / scan_signals
        ↓
src/data/bars.py  load_1m_range / load_1s_range
        ↓
LocalStore.load(symbol, date, kind)  → 只讀本地 parquet
```

流程：`fetch_history` 先填滿 cache → 之後所有回測／掃描只碰硬碟。

## 4. 為什麼用 Parquet（小＋快）

| 做法 | 說明 |
|------|------|
| 欄式儲存 | 回測常只讀 `ts/open/high/low/close/session`，不必解全表 |
| **zstd 壓縮** | 體積明顯小於 CSV／未壓縮 parquet |
| 精簡欄位 | tick 只留必要欄；丟掉多餘 bid/ask 複本 |
| 預設研究用 **1m** | 一天約千根；tick 一天可數十萬列 |

建議：

- **訊號調參／日常回測** → `--kind 1m`（最小、最快）
- **要秒級成交／滑價細驗** → `--kind 1s`（可再從 tick 重採樣）
- **原始稽核** → 才 `--keep-tick` 留 tick（磁碟最大）

## 5. CLI

```bash
# TMF 分 K 長歷史（推薦補資料）
python scripts/fetch_history.py --symbol TMF --kind 1m --preset full

# TMF 秒 K（較大；ticks 先 AllDay，空則 RangeTime 半天切片合併）
python scripts/fetch_history.py --symbol TMF --kind 1s --start 2025-01-01 --end 2026-10-21
# 註：SIM 部分日期 AllDay／分段皆回空陣列（非流量超限、非假日）；kbars 仍可能有 1m

# 股票 1m
python scripts/fetch_history.py --symbol 2330 --kind 1m --start 2025-01-01 --end 2026-10-21

# 已有 cache 會 skip；重抓加 --force
```

結果摘要：`outputs/fetch_history_report.json`（不要盯 log）。

TMF 1m `full`（TMFR1）：上市 2024-07-29 → 昨天；缺日多為休市。讀取會合併 `data/cache/TMF/` 與舊路徑 `data/cache/tmf/TMF/`。

## 6. TMFR1

微型台指 **近月連續** 代碼。歷史回補用它；當月碼（如 TMFK6）查舊日常空。
