# docs/ai — AI 必讀索引

> 給 Cursor / Claude / Gemini 等 agent。人類也可讀，但寫法以 **機器可掃** 為優先。

## 開新 session 必讀順序

1. [`../../AGENTS.md`](../../AGENTS.md) — 入口硬規則
2. [`PROJECT.md`](PROJECT.md) — 現況、鎖定參數、品牌
3. [`ARCHITECTURE.md`](ARCHITECTURE.md) — 模組邊界
4. [`DATA.md`](DATA.md) — 行情落地與抓取
5. [`BACKTEST.md`](BACKTEST.md) — 秒級無偷看回測
6. [`CLI.md`](CLI.md) — 真實可跑指令（禁止捏造旗標）
6b. [`DOCKER.md`](DOCKER.md) — Docker Desktop 別台部署
7. [`ROADMAP.md`](ROADMAP.md) — 一次一件事的順序
8. [`DRAWDOWN_CONTROL.md`](DRAWDOWN_CONTROL.md) — 控回撤版（點數／一口／停損／maxDD）
9. [`SIGNALS_VS_STRATEGY.md`](SIGNALS_VS_STRATEGY.md) — 訊號／策略硬邊界
10. [`CACHE.md`](CACHE.md) — 本地 parquet：小、快、離線讀
11. [`STRATEGY_ENSEMBLE.md`](STRATEGY_ENSEMBLE.md) — 雙軌並行策略與設定說明（高勝率 vs 大獲利）

## 快速真相表

| Key | Value |
|-----|-------|
| brand | `tidemark` (lowercase only) |
| display_zh | 潮線 |
| notify_prefix | `[tidemark]` |
| instrument_now | `TMF` |
| sessions | `day` + `night` |
| capital | `10000` |
| max_lots | `1` |
| paper_strategy | `complementary`（`gap≥120→orb else rsi`；基線可切 `tmf_v1`） |
| sibling_repo | `gemini_cli_stock` — **do not merge codebases** |
