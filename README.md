# tidemark

台股／台指量化工作台（品牌全小寫：`tidemark`，中文：**潮線**）。

- LINE / UI / Telegram 推播前綴統一：`[tidemark]`
- 與既有 `gemini_cli_stock` **完全分離**；本 repo 有自己的策略、訊號與回測思維
- 之後可擴充：存股、正 2、現股…但一次只做完一件事

## 第一階段鎖定（先做完再做新的）

| 項目 | 設定 |
|------|------|
| 商品 | **TMF**（微型台指） |
| 時段 | **日盤 + 夜盤** |
| 資金 | **10,000**（paper） |
| 口數 | **最多 1 口** |

> **保證金提醒**：交易所公告的 TMF 原始保證金通常遠高於 1 萬。  
> 本階段 `capital: 10000` 是 **紙上權益／紀律上限**，用來練訊號與回測；真盤下單前必須再核對券商保證金與權益數。

## 設計原則

1. **秒級 walk-forward**：回測每一秒推進，時刻 `t` 只能看到 `≤ t` 的資料（不能偷看答案）
2. **資料落地**：Tick／1s 存本地 `data/cache/`，不依賴當下 API 才能重跑
3. **訊號與下單分離**：Signal 純邏輯；執行層（paper / 永豐）可替換
4. **一次一口、一個策略**：`tmf_v1` 驗證賺錢且穩定後，才開下一個策略槽

## 目錄

```
config/           設定（品牌、資金、口數、時段）
src/
  core/           型別、設定載入
  data/           本地儲存與資料提供者
  signals/        訊號（無 I/O）
  strategy/       導演：串訊號 + 風控口數
  backtest/       秒級回測引擎
  trading/        paper / 永豐
scripts/          抓資料、跑回測 CLI
data/cache/       本地行情（gitignore）
logs/             日誌與帳本（gitignore）
```

## AI / 開新 session

每個 AI session **先讀**：

1. `AGENTS.md`
2. `docs/ai/README.md` → `PROJECT.md` → 其餘

Cursor 已掛 `.cursor/rules/`（alwaysApply）。

## 快速開始

```bash
cd tidemark
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

python scripts/smoke_check.py
python scripts/fetch_tmf.py --date 2026-03-20
```

## 品牌用法

| 通道 | 用法 |
|------|------|
| LINE | 訊息前綴 `[tidemark]`，顯示名可用「潮線」 |
| UI | 產品名 `tidemark` |
| Telegram | bot 顯示名 tidemark；username 建議 `tidemark_tw_bot` |

## 現況

- [x] 專案骨架與設定鎖定（TMF / 日夜 / 1 萬 / 1 口）
- [ ] 本地行情抓取與 cache
- [ ] 秒級回測引擎
- [ ] `tmf_v1` 真實訊號
- [ ] paper 模擬下單（永豐 SIM）
- [ ] LINE / Telegram 通知
- [ ] 擴充存股 / 正 2（TMF 做完再說）
