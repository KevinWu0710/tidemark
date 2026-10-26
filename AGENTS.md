# AGENTS.md — tidemark 入口（每個 session 先讀）

> 給所有 AI agent。詳細文件在 `docs/ai/`。

## 必讀順序

1. 本檔
2. `docs/ai/PROJECT.md`
3. `docs/ai/ARCHITECTURE.md`
4. `docs/ai/DATA.md`
5. `docs/ai/BACKTEST.md`
6. `docs/ai/CLI.md`
7. `docs/ai/ROADMAP.md`（看目前做到哪一 phase）

索引：`docs/ai/README.md`

## 品牌

| key | value |
|-----|-------|
| name | `tidemark`（**全小寫**） |
| display_zh | 潮線 |
| notify_prefix | `[tidemark]` |
| channels | LINE / UI / Telegram 同一套 |

## 鎖定參數（第一階段）

| key | value |
|-----|-------|
| instrument | TMF |
| sessions | day + night |
| capital | 10000 |
| max_lots | 1 |
| strategy | `tmf_complementary`（紙上主路徑；`paper_strategy: complementary`）|
| strategy_baseline | `tmf_v1`（RSI 腿／對照） |

## 優先序（使用者鎖定）

1. **先賺錢**：資料 → 訊號 → IS/OOS 回測 → paper  
2. **後包裝**：像素辦公室、LINE、Telegram、戰情 UI → edge 驗證前禁止開做

## Token／費用紀律（重要）

長任務標準流程（回測／掃描／抓資料）：

1. **啟動**指令（可背景跑）
2. **不要**讀 log、不要輪詢 terminal 等進度
3. 等系統「任務結束通知」，或使用者說「看結果」
4. **只讀結果檔**做分析（例如 `outputs/signal_scan.json`、`outputs/*_report.json`），不要整份 log 當主輸入

禁止：執行中反覆 `AwaitShell`／反覆打開 `.log` 空轉。  
省 Cursor token = 省使用者費用。

## 硬規則

1. **獨立專案**：禁止把 `gemini_cli_stock` 整包邏輯搬進來當核心。
2. **一次一件事**：遵守 `docs/ai/ROADMAP.md` phase；未完成不跳級。
3. **回測無偷看**：時刻 `t` 只能用 `≤ t`；見 `BACKTEST.md`。
4. **資料落地**：parquet 在 `data/cache/`；空 cache = 失效重抓。
5. **口數**：任何下單路徑 `max_lots=1`。
6. **秘密**：只讀 `.env`；禁止 commit token。
7. **測試**：改完至少 `python scripts/smoke_check.py`。
8. **指令**：只准用 `docs/ai/CLI.md` 列出的指令／旗標。

## 模組地圖

```text
src/core/       types + settings
src/data/       LocalStore + fetch providers
src/signals/    pure SignalDecision
src/strategy/   max_lots / session director
src/backtest/   SecondBarBacktester
src/trading/    paper + sinopac stub
scripts/        smoke / fetch / backtest CLI
config/         settings.yaml
```
