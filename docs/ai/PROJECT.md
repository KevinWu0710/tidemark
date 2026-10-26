# PROJECT — tidemark 現況真相

## 1. 這是什麼

`tidemark` 是獨立台股／台指量化專案（策略、訊號、回測、下單、通知）。

- **不是** `gemini_cli_stock` 的子目錄或 fork 整合版
- 未來可加：存股、正 2、現股——但必須等當前階段做完
- 產品名全小寫：`tidemark`；中文：潮線
- LINE / UI / Telegram 一律前綴：`[tidemark]`

## 2. 第一階段鎖定（不可擅自改）

| 項目 | 值 | 說明 |
|------|-----|------|
| instrument | `TMF` | 微型台指；1 點 = 10 TWD |
| sessions | day + night | 日 08:45–13:45；夜 15:00–05:00 |
| capital | `10000` | paper 權益／紀律；非保證夠交易所保證金 |
| max_lots | `1` | 任何路徑不得超過 1 口 |
| strategy_slot | `tmf_complementary`（紙上主路徑） | 探針分流：`gap≥120→orb else rsi`；一口；見 `best_complementary_lock.json` |
| strategy_baseline | `tmf_v1`（啟用，對照） | RSI28 腿；設 `paper_strategy: tmf_v1` 可單跑基線 |
| orb_leg | `tmf_daily_trend` | 開盤缺口 ORB；互補選 orb 時使用 |

### 保證金注意

交易所 TMF 原始保證金通常 ≫ 10000。  
`capital: 10000` = 紙上模擬與風控上限。真盤前必須重核券商保證金與權益。

## 3. 台指商品速查（擴充時用，現在只做 TMF）

| 代碼 | 名稱 | 點值 | 階段 |
|------|------|------|------|
| TMF | 微型台指 | 10 | **現役** |
| MTX | 小型台指 | 50 | 之後 |
| TX | 臺股期貨 | 200 | 之後 |

## 4. 設定單一真相

- 執行參數：`config/settings.yaml`
- 秘密：`.env`（範本 `.env.example`）
- 禁止把 token／憑證寫進 md 或程式碼

## 4.1 資料來源（秒級）

| 優先 | 來源 | 用途 |
|------|------|------|
| 1 | 永豐 Shioaji SIM ticks | TMF tick → 本地 1s（主路徑） |
| 2 | FinMind Daily | 日線／session 研究（register 可用） |
| — | FinMind Tick/KBar | 需 Sponsor；目前 token 為 register 會 400 |

## 5. Do / Don't

### Do

- 改碼前讀 `AGENTS.md` + 本檔
- 行情落地 `data/cache/`
- 回測 walk-forward 逐秒
- 改完跑 `python scripts/smoke_check.py`

### Don't

- 不要從 `gemini_cli_stock` 複製 God Class / holding_0050 邏輯進來當核心
- 不要一次開多商品或多策略
- 不要向量化整表用未來 bar 算訊號
- 不要在未驗證 edge 前接 live 下單
- 不要把品牌寫成 `TideMark` / `TIDEMARK`（對外顯示允許 UI title case，**repo／設定／前綴維持小寫 tidemark**）
