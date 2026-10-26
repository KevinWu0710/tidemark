# BACKTEST — 秒級、無偷看

## 0. 日／夜勝率（禁止混算）

- **筆數勝率／交易日勝率**必須依 `session=day|night` **分開**統計（`src/backtest/metrics.py` → `by_session`）。
- 過關門檻用 `passes_session_daily_win_rate`：有成交的盤各自 > 門檻；**禁止**用日夜合計的混算值。
- `summarize()` 頂層 `daily_win_rate` / `win_rate` = **日盤**；夜盤見 `night_daily_win_rate` / `by_session.night`。混算僅 `daily_win_rate_mixed`（除錯用）。

## 1. 硬規則

```text
for i in 0..n-1:
    history = bars[0 : i+1]   # inclusive current only
    decision = strategy.on_bar(history)
    # NEVER use bars[i+1:] for indicators or fills
```

- 禁止：整段 `df['signal'] = df.close.shift(-1)` 這類未來函數
- 禁止：先算完全部再回填成交
- 允許：預先載入當日 parquet，但評估時仍只能 slice 到 `i`

實作：`src/backtest/engine.py` → `SecondBarBacktester`  
驗證：`tests/test_no_lookahead.py`（engine 只看到 `[:i+1]`；breakout 通道不含當根／未來）

## 5. 訊號掃描：並行＋斷點（避免重跑整輪）

| 做法 | 說明 |
|------|------|
| 快引擎 | breakout 用 `fast_breakout` O(n)，不要用慢 `SecondBarBacktester` 掃全網格 |
| 一組一檔 | 每個參數組合寫 `outputs/scan_shards/{run_id}/{id}.json` |
| 並行 | `scan_signals_fast.py --workers N`（ProcessPool） |
| 續跑 | `--resume` 跳過已有 shard；掛掉後不用從頭 |
| 比較 | 全到齊後 merge → `signal_scan_report.json` 排名 |
| 日盤當沖 | `--daytrade`：只掃 day + 日盤最後一根 `flatten_eod`（不隔夜） |

## 2. 成交假設（conservative）

| 動作 | 成交價 |
|------|--------|
| BUY / 開多 | mid + slip_ticks |
| SELL / 開空 | mid - slip_ticks |
| CLOSE 多 | mid - slip_ticks |
| CLOSE 空 | mid + slip_ticks |

成本：`fee_per_side` + `notional * tax_rate`（見 settings）

## 3. 口數與資金

- `max_lots=1`：已有倉再 BUY/SELL → HOLD（blocked）
- PnL：點數 × `point_value`（TMF=10）
- equity curve 每秒 mark-to-market

## 4. 驗證通過才進下一步

| Gate | 條件 |
|------|------|
| smoke | `scripts/smoke_check.py` |
| research | 有本地 cache + 可重現回測 |
| edge | OOS（樣本外）淨利 > 0 且可接受回撤 |
| paper | PaperBroker 帳本穩定；常態覆盤 `run_paper_standard.py` → `paper_standard_report.json` |
| 1s | 覆蓋率≥50% 才 STRICT；否則 `REFERENCE_ONLY`（SIM tick 洞不挡主流程） |
| live | 使用者明確要求 + 保證金足夠 |

未過 gate 禁止實作 live 下單。
