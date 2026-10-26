# CLI — 真實指令（禁止捏造）

工作目錄：repo root `tidemark/`

## Agent 執行核准（少彈窗）

Cursor：**Settings → Agents → Approvals & Execution**

| 模式 | 效果 |
|------|------|
| **Run Everything** | 完全不詢問（本機最省事；接受風險） |
| Auto-review | 白名單／沙箱自動跑；其餘由分類器決定 |

專案／使用者白名單：`.cursor/permissions.json`、`%USERPROFILE%\.cursor\permissions.json`（已含 `python` 等）。

## 環境

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Docker Desktop（別台／全自動）

詳見 [`DOCKER.md`](DOCKER.md)。

```bash
docker compose up -d --build          # ops + UI（開機常駐）
# 瀏覽 http://localhost:8080  → 訊號｜策略｜下單
docker compose logs -f ops
docker compose --profile tools run --rm smoke
python scripts/run_ops_daemon.py --once-evening --date YYYY-MM-DD   # 測覆盤
python scripts/run_ui.py                                              # 本機 UI
```

## 指令表

| 目的 | 指令 |
|------|------|
| 煙霧測試 | `python scripts/smoke_check.py` |
| 通用歷史抓取（TMF／現股） | `python scripts/fetch_history.py --symbol TMF\|2330 ...` |
| （抓取結果檔） | `outputs/fetch_history_report.json` ← agent 讀這個，不要盯 log |
| 階段1：近 6 個月秒級 | `python scripts/fetch_tmf.py --preset recent6m` |
| 階段2：上市～昨天全歷史 | `python scripts/fetch_tmf.py --preset full` |
| 自訂區間 | `python scripts/fetch_tmf.py --start YYYY-MM-DD --end YYYY-MM-DD` |
| 單日抓取 | `python scripts/fetch_tmf.py --date YYYY-MM-DD` |
| 回測（有資料後） | `python scripts/run_backtest.py --start YYYY-MM-DD --end YYYY-MM-DD` |
| 秒級複核鎖策略 | `python scripts/verify_1s_lock.py` |
| （複核結果） | `outputs/verify_1s_report.json` ← 覆蓋率＋gate（REFERENCE_ONLY／STRICT）；1s 洞時不挡主流程 |
| 標準紙上覆盤 IS/OOS/FULL（主路徑=complementary） | `python scripts/run_paper_standard.py` → `outputs/paper_standard_report.json` |
| （改跑純 RSI 基線） | settings 設 `trading.paper_strategy: tmf_v1` 後再跑上列 |
| 互補分流覆盤（同鎖，專用報告） | `python scripts/run_paper_complementary.py` → `outputs/paper_complementary_report.json` |
| 補缺 1s mid_IS（偏斜主因） | `python scripts/fetch_history.py --symbol TMF --kind 1s --start 2025-02-20 --end 2026-03-31 --report outputs/fetch_1s_mid_is_report.json` |
| 補缺 1s（寬區間） | `python scripts/fetch_history.py --symbol TMF --kind 1s --start 2025-02-20 --end 2026-10-22 --report outputs/fetch_1s_fill_report.json` |
| （1s 覆蓋盤點） | `outputs/1s_coverage_report.json` |
| 永豐 SIM 探測 | `python scripts/run_paper_sim_probe.py` → `outputs/paper_sim_probe.json` |
| （SIM 真下單） | `python scripts/run_paper_sim_probe.py --live-order --price <限價>`（仍 simulation；禁 LIVE） |
| 日流程自動化（互補→一口；預設純 Paper dry-run） | `python scripts/run_sim_day.py [--date YYYY-MM-DD]` → `outputs/sim_day_*.json` |
| （盤中輪詢） | `python scripts/run_sim_day.py --watch --interval 60` |
| （連 SIM dry mirror） | `python scripts/run_sim_day.py --connect` |
| （SIM 真下單，待可探測後） | `python scripts/run_sim_day.py --place-sim`（仍 simulation；禁 LIVE） |
| 盤中即時 tick 觸發（離線 1m 重播） | `python scripts/run_live_tick.py [--date YYYY-MM-DD]` → `outputs/live_tick_*.json` |
| （訂閱永豐 SIM TickFOPv1） | `python scripts/run_live_tick.py --subscribe [--max-seconds 120]` |
| （tick 路徑 SIM 真下單） | `python scripts/run_live_tick.py --subscribe --place-sim`（仍 simulation；禁 LIVE） |
| 無虧損日過濾研究 | `python scripts/research_no_loss_days.py` → `outputs/no_loss_days_report.json` |
| 指標確認（MACD/KD/BB…） | `python scripts/research_indicator_boost.py` → `outputs/indicator_boost_report.json` |
| MACD 主訊號掃描（對照 RSI 鎖） | `python scripts/research_macd_primary.py` → `outputs/macd_primary_report.json` |
| TORB probe＋RSI 高原＋五門／DSR | `python scripts/research_torb_plateau.py` → `outputs/torb_plateau_report.json` |
| 日頻／覆蓋率（日勝率>70%＋階梯覆蓋＋新訊號族） | `python scripts/research_daily_frequency.py [--min-daily-win-rate 0.7] [--coverage-tiers 0.15,0.3,0.5,0.8]` → `outputs/daily_frequency_report.json` |
| 日頻專用軌道（密訊號／多空日夜；與 tmf_v1 分開） | `python scripts/research_daily_trade.py [--min-daily-win-rate 0.55] [--coverage-tiers 0.5,0.7,0.9]` → `outputs/daily_trade_report.json` |
| 上帝視角日買賣點→規則→無偷看回測 | `python scripts/research_god_view_daily.py` → `outputs/god_view_daily_report.json` |
| 日頻：god 特徵簇（局部高低／RSI 帶／EMA，無 reclaim）→路徑 TP/stop→回測 | `python scripts/research_daily_from_god.py` → `outputs/daily_from_god_report.json` |
| 日頻稀訊號：bounce／8棒＋冷卻／每盤上限（不強求多筆） | `python scripts/research_daily_from_god_sparse.py` → `outputs/daily_from_god_sparse_report.json` |
| 日頻訊號重想：開盤偏差／壓縮突破／趨勢回踩＋ATR出場 | `python scripts/research_daily_signal_rethink.py` → `outputs/daily_signal_rethink_report.json` |
| 多／空分訊號再合成（一口） | `python scripts/research_long_short_split.py [--min-daily-win-rate 0.7] [--top-k 8]` → `outputs/long_short_split_report.json` |
| 只做多事前過濾／監控 | `python scripts/research_long_prefilters.py [--min-daily-win-rate 0.8]` → `outputs/long_prefilters_report.json` |
| 虧損日局式＋多訊號腳本 | `python scripts/research_loss_regimes.py` → `outputs/loss_regimes_report.json` |
| （局式配方） | `outputs/regime_signal_recipes.json` + `src/signals/regime_scripts.py` |
| IS/OOS 訊號掃描（**快／並行，主路徑**） | `python scripts/scan_signals_fast.py --workers 4 [--resume]` |
| （日盤當沖／不隔夜） | `python scripts/scan_signals_fast.py --daytrade --workers 4 --force` |
| （日盤當沖加寬網格） | `python scripts/scan_signals_fast.py --daytrade --grid wide --workers 8 --force` |
| （高日勝率 RSI 小停利網格） | `python scripts/scan_signals_fast.py --daytrade --grid wr_hunt --min-daily-win-rate 0.8 --force` |
| （多訊號獵正：當沖→可隔夜） | `python scripts/hunt_positive_signals.py` |
| （慢引擎，抽樣複核） | `python scripts/scan_signals.py` |
| （掃描結果檔） | `outputs/signal_scan_report.json` ← agent 讀這個，不要盯 log（`RUNNING`→`DONE`） |
| （各組中間結果） | `outputs/scan_shards/{run_id}/*.json` ← 可斷點續跑；`--merge-only` 只合併 |
| （過關規則） | **IS>0 且 OOS>0**；可加 `--min-daily-win-rate 0.8`（**日盤／夜盤分開**的交易日勝率皆須 >80%，有成交的 session 才計；禁止日夜混算） |
| （高日勝率獵尋） | `python scripts/hunt_positive_signals.py`（內建 min-daily-win-rate 0.8） |
| 無偷看測試 | `python tests/test_no_lookahead.py`（亦含於 smoke） |
| 紙上覆盤重播 | `python scripts/run_paper_replay.py --start YYYY-MM-DD --end YYYY-MM-DD [--tag name] [--prefilter\|--no-prefilter]` |
| （快路徑，預設） | 同上；`--no-fast` 改慢引擎 |
| （覆盤結果） | `outputs/paper_sessions/*.json` + `outputs/paper_session_latest.json` |
| （常態三區間） | 見上方 `run_paper_standard.py` |
| 日盤事前過濾紙上對照 | `python scripts/run_paper_prefilter_compare.py` → `outputs/paper_prefilter_compare.json` |
| 雙軌 ensemble 驗證（secure/trend） | `python scripts/verify_ensemble.py` → `outputs/ensemble_verify_report.json` |
| 事前分流器（開盤特徵→secure/trend，IS 鎖／OOS 驗） | `python scripts/research_ensemble_selector.py` → `outputs/ensemble_selector_report.json` |
| 真互補分流（ORB vs RSI28，IS／OOS＋舊分流比較） | `python scripts/research_complementary_selector.py` → `outputs/complementary_selector_report.json` |
| 互補硬化（保本／冷卻掃描＋t／DSR；`--apply` 寫鎖） | `python scripts/research_complementary_harden.py [--apply]` → `outputs/complementary_harden_report.json` |
| ORB 進階掃描（研究用） | `python scripts/research_orb_advanced.py` → `outputs/orb_advanced_report.json` |

| 單一策略細析 | `python scripts/analyze_strategy.py --tag breakout40_day` |
| （細析結果檔） | `outputs/breakout40_day_report.json` |
| 夜間擴掃（快） | `python scripts/overnight_research.py` |
| （明早報告） | `outputs/MORNING_REPORT.md` + `outputs/best_strategy_lock.json` |
| 控回撤研究 | `python scripts/research_drawdown_control.py` |
| 指標確認加勝率（RSI+MACD/KD/…） | `python scripts/research_indicator_boost.py` |
| （指標加勝率結果） | `outputs/indicator_boost_report.json` |
| 無虧損交易日過濾搜尋 | `python scripts/research_no_loss_days.py` |
| （無虧損日結果） | `outputs/no_loss_days_report.json` |
| （控回撤說明） | `docs/ai/DRAWDOWN_CONTROL.md` |
| （控回撤結果） | `outputs/DRAWDOWN_CONTROL_RESULT.md` + `drawdown_control_report.json` |

### 建議順序

1. `--preset recent6m`（先把 pipeline／策略組合跑通）
2. `--preset full`（補 2024-07-29～昨天；已有 cache 會 skip）
3. 研究切分：IS≈上市～2026-03；OOS≈2026-04～最近

### fetch_history.py flags（主 CLI；`fetch_tmf.py` 為 `--symbol TMF` 薄包裝）

| flag | meaning |
|------|---------|
| `--symbol` | `TMF`（預設）或現股代號如 `2330` |
| `--preset` | `recent6m` \| `full`（TMF=上市→昨天；現股≈2y） |
| `--date` | 單日 |
| `--start` / `--end` | 日期區間（含） |
| `--source` | `shioaji`（**預設**）\| `finmind`（日線／Sponsor tick） |
| `--kind` | `1m`（**預設**，kbars 小／快）\| `1s`（tick 重採樣）\| `tick`\| `daily` |
| `--keep-tick` | 額外存原始 tick（大）；預設不存 |
| `--force` | 忽略既有 cache 重抓 |
| `--include-weekend` | 連六日也打 API（通常空） |
| `--probe-earliest` | 二分搜尋 TMFR1 最早有資料日（僅 TMF）；寫入 report 後結束 |
| `--report` | 結果 JSON（預設 `outputs/fetch_history_report.json`） |

```bash
# 探測 TMFR1 最早日 → outputs/tmfr1_probe.json + fetch_history_report.json
python scripts/fetch_history.py --symbol TMF --probe-earliest

# 長資料補齊（1m；已有 cache 會 skip）
python scripts/fetch_history.py --symbol TMF --kind 1m --preset full
```

Cache 說明：`docs/ai/CACHE.md`。結果檔 `status`：`RUNNING`｜`DONE`（勿盯 log）。

**TMFR1**：微型台指「近月連續契約」。歷史回補走 `TMFR1`；當月碼（如 `TMFK6`）舊日常空。

現股支援：`--kind 1m`（kbars）／`daily`（FinMind）；`tick`／`1s` 可用但較重。

### 資料來源現況（2026-10）

| source | dataset | 本機 token 狀態 |
|--------|---------|-----------------|
| shioaji | ticks / kbars | **SIM key** 可用 → 秒／分 K 主路徑；連打過快會回空（CLI 已 throttle＋retry） |
| shioaji LIVE | — | 本機 LIVE secret 若無效則忽略，仍走 SIM |
| finmind | TaiwanFuturesDaily | register 可用 |
| finmind | TaiwanFuturesTick / KBar | 需 Sponsor；register 會 400 |

## 不存在的指令

- 沒有 `main.py --simulate-day`
- 沒有與 `gemini_cli_stock` 共用的 `cli.py ui`
- 未列在本檔的 flag = 不要假設存在
