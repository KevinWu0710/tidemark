# DATA — 行情落地規則

## 1. 目標

- 所有回測／研究 **必須能離線重跑**
- 來源可為 FinMind 或永豐 Shioaji；落地格式統一 parquet
- 空資料 `[]` / empty DataFrame = **失效**，刪檔或重抓，禁止當有效 cache

## 2. 目錄

```text
data/
  raw/              # 原始下載（可保留 API 原樣）
  cache/
    {SYMBOL}/       # TMF / 2330 / ...
      tick/YYYY-MM-DD.parquet
      1s/YYYY-MM-DD.parquet
      1m/YYYY-MM-DD.parquet
      daily/...
  cache/tmf/TMF/... # legacy 路徑仍可讀（LocalStore fallback）
```

實作入口：`src/data/local_store.py` → `LocalStore`  
通用抓取：`scripts/fetch_history.py`（`fetch_tmf.py` = `--symbol TMF`）

## 3. Schema（1s 最低欄位）

| column | type | note |
|--------|------|------|
| ts | datetime64[ns, Asia/Taipei] or ISO str convertible | 必填 |
| open | float | |
| high | float | |
| low | float | |
| close | float | 回測預設價欄 |
| volume | float/int | 可 0 |
| session | str | `day` \| `night` |

預設不存 tick（`--keep-tick` 才寫）；`1s` 由 tick 重採樣；研究補資料優先 `--kind 1m`（kbars）。詳見 `CACHE.md`。

## 4. 時段切割

| session | 時間（Asia/Taipei） |
|---------|---------------------|
| day | 08:45:00 ≤ t ≤ 13:45:00 |
| night | 15:00:00 ≤ t 或 t ≤ 05:00:00 |
| closed | 其餘（開盤前靜默等）— 1s cache **剔除** |

### FinMind tick 交易日歸屬（重要）

對 `start_date = D` 的 tick 檔：

| 時段 | 歸屬 |
|------|------|
| 00:00–05:00 | D 的夜盤（after_market） |
| 08:45–13:45 | D 的日盤（position） |
| 15:00–24:00 | **下一交易日** 的夜盤 |

實作：`src/data/sessions.py` + `src/data/tmf_pipeline.py`

## 4.1 近月

- 當日 tick 依 `contract_date` 加總 volume，取最大者為 front month
- 含 `/` 的價差單列剔除
- 換月週請人工抽查 front 是否合理

## 5. 近月合約

- TMF 有月份碼；研究／回測用 **近月連續** 或當日主力（OI／量最大）
- 換月規則必須寫在 fetcher 註解與本檔；禁止 silently 混月

## 6. 秘密

- `FINMIND_TOKEN`、`SHIOAJI_*` 只在 `.env`
- 日誌禁止打印完整 token

## 7. Fetch CLI

見 [`CLI.md`](CLI.md) — `python scripts/fetch_history.py ...`

### TMFR1（重要）

- **TMFR1** = 微型台指**近月連續契約**（Rolled front month）
- 用**當月契約**抓舊日期常變空；歷史回補**必須優先 TMFR1**
- `ShioajiFutures.CONTINUOUS_CODE = "TMFR1"`；`ticks()` 預設 prefer continuous
- 探測指令：`python scripts/fetch_history.py --symbol TMF --probe-earliest` → `outputs/tmfr1_probe.json`
- 實測 SIM：`TMFR1` **上市日 2024-07-29 即有 tick**；連打過快 API 會回空，fetch CLI 有 pause／retry

### 推薦路徑

```bash
# 通用：TMF 全歷史（TMFR1）；結果 → outputs/fetch_history_report.json
python scripts/fetch_history.py --symbol TMF --preset full

# 現股 1m
python scripts/fetch_history.py --symbol 2330 --start 2026-01-01 --end 2026-03-31 --kind 1m

# 薄包裝（等同 --symbol TMF）
python scripts/fetch_tmf.py --preset recent6m
python scripts/fetch_tmf.py --preset full
python scripts/fetch_tmf.py --date 2026-10-20
```

### 回測區間建議（見對話共識）

| 用途 | 區間 |
|------|------|
| IS 調參 | 2024-08 → 2026-03 |
| OOS 驗證 | 2026-04 → 最近 |
| 秒級資料上限 | TMF 上市 2024-07-29 起（更早無真 TMF） |

### 資料變長之後要不要重掃訊號？

**要。** Cache 補齊後應重跑 IS/OOS（`scan_signals.py` / overnight），用同一套無偷看規則驗證；鎖定參數可先當 baseline，以新 OOS 為準決定是否改 lookback。訊號模組仍只產買賣點；口數／停損仍在 strategy。

### 美股／財報過濾？

**現階段不必。** 現役 `tmf_v1` 是台指 breakout，不依賴個股財報或美股日曆。若之後當 **可選 filter 層**（例如美股大跌日降曝險），掛在 strategy／session director 前，**不要**寫進 `src/signals/` 價訊核心。

實作：

- `src/data/shioaji_futures.py` — tick 主來源
- `src/data/finmind_futures.py` — daily；tick 需 Sponsor
- `src/data/tmf_pipeline.py` — front month + resample + cache
