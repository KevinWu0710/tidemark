# 訊號 vs 策略（硬邊界）

> 使用者共識：兩者**必須分模組**，禁止混在同一類別裡。

## 訊號 `src/signals/`

**只回答：現在有沒有買點／賣點／該平倉？**

| 允許 | 禁止 |
|------|------|
| 看行情、指標、session 標籤 | 口數、本金、保證金 |
| 輸出 `BUY` / `SELL` / `CLOSE` / `HOLD` | 停損點數金額、單日虧損上限 |
| `reason` / `confidence` / `meta` | 呼叫券商、讀寫持倉檔 |
| 純函式、可單測 | `stop_points`、`max_lots`、`capital` |

現役訊號：
- `TmfV1Signal` → 日盤 breakout `lookback=60`（價訊）
- **`ComplementarySignal`** → 探針分流 orb|rsi|flat + 當日意圖（引擎同源 `complementary_fast`）

## 策略 `src/strategy/`

**只回答：訊號來了，能不能做、做多大、風控怎麼砍？**

| 允許 | 禁止 |
|------|------|
| `max_lots=1`、禁止加碼 | 自己發明進場指標 |
| `stop_points`、`max_daily_loss`、**日盤事前過濾**（夜盤代理／開盤禁制／11:15 截止） | 重算 RSI／突破（那是訊號） |
| 記錄 `entry_price`、當日已實現損益 | 直接 HTTP 下單（交給 trading） |
| 擋下違規的 BUY／SELL | |

現役策略：
- `TmfSessionStrategy`
- **`ComplementaryRiskStrategy`** → 一口、ORB/RSI 停損保本 TP、10 秒 tick 風控、當日已平倉禁再開

即時編排（薄）：`trading/live_tick_loop` = tick→1m → Signal → Risk → Paper/Sinopac。

## 設定檔怎麼拆

`config/settings.yaml` 的 `strategies[].params` 可同時有兩類鍵，但**讀取責任分離**：

- 訊號只讀：`inner` / `lookback` / `sessions`（價訊相關）
- 策略只讀：`stop_points` / `max_daily_loss` / `max_lots` / `capital`

## 資料流

```text
bars → Signal.evaluate() → SignalDecision
                         ↓
              Strategy.on_bar() 套用風控
                         ↓
              Backtest / Paper / Broker
```
