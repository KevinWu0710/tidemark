# ARCHITECTURE

## 1. 分層（SRP）

```text
scripts/  (CLI)
    ↓
trading/*_loop     ← 薄編排（live_tick / sim_day）
    ↓
strategy/          ← 導演：口數／資金紀律／tick 風控
    ↓
signals/           ← 純邏輯，無 I/O（含 ComplementarySignal）
    ↓
backtest/ | trading/paper|sinopac
    ↓
data/LocalStore
```

| Layer | Path | Responsibility | Forbidden |
|-------|------|----------------|-----------|
| core | `src/core/` | types, settings | broker calls |
| data | `src/data/` | fetch + local parquet | strategy rules |
| signals | `src/signals/` | decide BUY/SELL/HOLD/CLOSE（價訊） | 口數／本金／停損／券商 |
| strategy | `src/strategy/` | max_lots、stop、單日虧損 | 發明進場指標 |

詳細邊界：[`SIGNALS_VS_STRATEGY.md`](SIGNALS_VS_STRATEGY.md)
| backtest | `src/backtest/` | second-by-second replay | look-ahead |
| trading | `src/trading/` | paper 帳本／broker | invent signals |

Paper：`PaperBroker` 記成交、成本、回合淨利；每次覆盤用 `save_session` 寫 `outputs/paper_sessions/`（要留賺賠紀錄）。

## 2. 關鍵型別

見 `src/core/types.py`：

- `Side`: FLAT | LONG | SHORT
- `Session`: day | night
- `TradeAction`: HOLD | BUY | SELL | CLOSE
- `SignalDecision`: action, reason, confidence, meta, ts

## 3. 新增策略 SOP（現階段不要做，僅預留）

1. 新 `src/signals/{name}.py` 繼承 `BaseSignal`
2. 新或擴充 `src/strategy/`
3. `config/settings.yaml` → `strategies` 新增一筆且 **預設 enabled: false**
4. 先回測 → paper → 才考慮 live

## 4. 與舊 repo 邊界

| 舊 `gemini_cli_stock` | 本 `tidemark` |
|----------------------|---------------|
| 00631L 波段 / 00919 存股 / 戰情室 | TMF 先 |
| SharedServices / SwingStrategy | 自己的 DI 風格，不硬搬 |
| FinMind Sponsor 可參考概念 | 自己實作 client，token 讀本專案 `.env` |
