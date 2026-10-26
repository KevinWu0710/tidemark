# ROADMAP — 一次一件事

> AI：完成當前 checkbox 前，**禁止**開始下一階段功能。
>
> **優先序（使用者鎖定）**：先主功能、以驗證能賺錢為主。  
> 像素辦公室 / LINE / Telegram / UI → **全部延後**，等 OOS 有 edge 再說。

## Phase 0 — 專案骨架 ✅

- [x] repo `tidemark`、品牌小寫
- [x] settings 鎖定 TMF / 日夜 / 10000 / 1 口
- [x] smoke_check
- [x] docs/ai + AGENTS + `.cursor/rules`

## Phase 1 — 資料落地 ✅

- [x] FinMind client + tick→1s pipeline（`src/data/tmf_pipeline.py`）
- [x] Shioaji tick 主路徑（`src/data/shioaji_futures.py`）
- [x] `scripts/fetch_tmf.py` / `fetch_history.py`（`--preset recent6m|full` / `--date` / `--force`）
- [x] 單日實測 2026-10-20：tick→1s，sessions=day+night
- [x] `--preset recent6m` 下載完成
- [x] `--preset full` 補齊上市～昨天（TMFR1 1m；`outputs/fetch_history_report.json`）
- [x] 空資料當失效（unlink stale）
- [x] AI docs：`docs/ai/*` + `.cursor/rules`

## Phase 2 — 秒級回測可跑真資料

- [x] `scripts/run_backtest.py` 讀 cache（`tmf_v1` 仍為 HOLD 佔位）
- [x] 驗證無 look-ahead（`tests/test_no_lookahead.py`；smoke 會跑）
- [x] 輸出 summary 到 `outputs/`（gitignore）

## Phase 3 — `tmf_v1` 真實訊號

- [x] 第一輪價訊掃描（27 組）→ 候選 breakout day lookback=40
- [x] `tmf_v1` 接上該組合；細析腳本 `analyze_strategy.py`
- [x] 快速 O(n) breakout + `overnight_research.py` 擴掃 lookback
- [x] `MORNING_REPORT`：lookback=60 strict_pass（IS/OOS/FULL 皆正）
- [x] `tmf_v1` 曾鎖 lookback=60 day breakout（短窗）；全歷史後改鎖
- [x] 控回撤說明 `DRAWDOWN_CONTROL.md` + `research_drawdown_control.py`
- [x] 全歷史＋當沖 → 再鎖 **高日勝率 RSI28 reclaim20 TP40 stop300**（IS/OOS 日勝率>80%）
- [x] paper SIM 適配：`SinopacFuturesTrader`（SIM＋max_lots=1）+ `scripts/run_paper_sim_probe.py`；覆盤 CLI 已可重播鎖策略
- [x] 秒級覆蓋補抓（1s 222→272）+ 重跑 `verify_1s_lock` — SIM tick 洞（mid_IS 幾乎無）；嚴格 path 未過
- [x] **A** 接受限制：1s `gate=REFERENCE_ONLY`（覆蓋率&lt;50% 不挡主流程；主證據=1m+paper）
- [x] MACD 主訊號掃描 300 組 — **0 過關**（`outputs/macd_primary_report.json`）；維持 RSI28 鎖
- [x] TORB probe＋RSI 參數高原＋五門閘門／DSR — **0 過關**（`outputs/torb_plateau_report.json`）；維持 RSI28 鎖；鎖本身亦未過 t≥2／DSR
- [x] **日盤事前過濾**：夜盤≤-120→開盤60分不進 + 11:15截止（`day_prefilters`）；紙上對照過關（IS/OOS/FULL 皆升）→ `prefilter_enabled: true`；勝率日夜分開
- [x] **日頻專用雙軌並行（Multi-Strategy Ensemble）**：
  - 軌道 1 `tmf_daily_secure`：穩健高勝率型（HL 覆核：WR≈59.9%/OOS 63.2%，TP200/SL50，+50pt 保本，1.2x 量能，淨利 **+13,061**）。
  - 軌道 2 `tmf_daily_trend`：暴利型（HL 覆核：WR≈44.3%，TP200/SL50，無保本，淨利 **+30,802**）。
  - 底層 `breakeven_points` + 單測 + smoke 通過；YAML 已掛兩槽。**注意**：secure ⊆ trend（非互補）；尚無一口帳本編排器；紙上主路徑仍 `tmf_v1`。覆核：`verify_ensemble.py`。
  - [x] **事前分流器 v1**（secure/trend）：OOS WR 64.9%／+4973，但是「少做的 secure」，非真切換。見 `ensemble_selector_report.json`。
  - [x] **事前分流器 v2 真互補**（ORB vs RSI28）：`research_complementary_selector.py` → `complementary_selector_report.json`（含重疊 Jaccard、與 v1 比較）。
  - [x] **互補分流紙上覆盤**：鎖 `gap≥120→orb else rsi`，一口；已升成 **紙上主路徑**（`trading.paper_strategy: complementary`；`run_paper_standard.py`）。基線對照：`paper_strategy: tmf_v1`。
  - [x] **互補硬化**：ORB `breakeven=50`（無虧損冷卻）；`research_complementary_harden.py`；紙上 OOS DD↓、仍淨利>0；t/DSR 未過嚴格閘。SIM 探測連線 OK（`run_paper_sim_probe.py`）。
  - [x] **日流程自動化 scaffold**：`SimDayLoop` + `scripts/run_sim_day.py`（互補分流→一口；預設純 Paper dry-run；`--connect`／`--place-sim` 待可下單探測後再開）。
  - [x] **盤中即時 tick 觸發**：`MinuteBarBuilder` + `LiveTickLoop` + `scripts/run_live_tick.py`（tick→1m 關棒→互補；預設 1m 重播；`--subscribe` TickFOPv1；`--place-sim` 待探測後）。
  - [x] **模組邊界重構**：`ComplementarySignal`（signals）+ `ComplementaryRiskStrategy`（strategy）+ 薄編排 `live_tick`／`sim_day`；下單仍只在 trading。

## Phase 4 — Paper / 永豐 SIM

- [x] PaperBroker 完整帳本（權益／成本／回合；`save_session` 覆盤落盤）
- [x] 覆盤 CLI：`scripts/run_paper_replay.py` → `outputs/paper_sessions/` + `paper_session_latest.json`
- [x] SinopacFuturesTrader 實作（SIM；LIVE 需 allow_live）
- [x] 仍 max_lots=1（PaperBroker + Sinopac 皆 enforce）
- [x] **D** 常態紙上覆盤 — **pass**（`outputs/paper_standard_report.json`：IS +1884／OOS +4352／FULL +6236）
- [x] 日流程自動化 dry-run（`run_sim_day.py`）；SIM 真下單閘門尚未開
- [x] 即時 tick→1m→互補（`run_live_tick.py`）；盤中 `--subscribe` 已接；真下單閘門尚未開

## Phase 4b — Ops／簡單 UI（Docker 全自動）

- [x] `run_ops_daemon.py`：盤中 Paper 模擬 + 每晚覆盤；記錄 `outputs/ops/`
- [x] 簡單戰情 UI：訊號／策略／下單三分欄 + journal（`run_ui.py` :8080）
- [x] `docker compose up -d` 預設啟動 `ops` + `ui`（restart unless-stopped）

## Phase 5 — 通知

- [ ] LINE / Telegram / webhook，前綴 `[tidemark]`

## Phase 6 — 下一商品（存股／正 2…）

- [ ] 使用者明確下令後才開新 strategy slot
